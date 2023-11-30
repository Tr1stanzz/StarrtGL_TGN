import math
import logging
import time
import sys
import argparse
import torch
import random
import numpy as np
import pdb
import pickle
import os
from pathlib import Path
from os.path import abspath, join, dirname
from evaluation.evaluation import eval_edge_prediction, eval_cluster_quality
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_batch_neighbor_finder, select_free_gpu, parse_config
from sklearn.metrics import average_precision_score, roc_auc_score
from DistGraphLoader import DataSet,  partition_load
from Sample.neighbor_sampler import NeighborSampler
from Sample.base import NegativeSampling
from DistGraphLoader import DistGraphData
from DistGraphLoader import DistributedDataLoader
from DistGraphLoader import DistCustomPool 
import distparser as distparser
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from shared_mailbox import SharedMailBox, SharedRPCMemoryManager

torch.autograd.set_detect_anomaly = True
torch.manual_seed(0)
np.random.seed(0)

def compute_shift_time_statics(edge_ts, edge_index):
    edge_index = edge_index.numpy()
    timestamps = edge_ts.numpy()
    sources, destinations = edge_index
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

args = distparser.args

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
FANOUT = args.fanout


Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
LABEL_PATH = f'./data/{args.data}_node_label.npy' 

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

device_num = 0
device_string = 'cuda:{}'.format(device_num)
device = torch.device(device_string)

rank = distparser._get_worker_rank()

sample_param, memory_param, gnn_param, train_param = parse_config('config/TGN.yml')
DistCustomPool.init_distribution('127.0.0.1',9675,'127.0.0.1',10023,backend = "nccl")
pdata = partition_load("./data/"+DATA, algo="metis")    
graph = DistGraphData(pdata = pdata,edge_index= pdata.edge_index, full_edge = False)
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_shift_time_statics(graph.data.edge_ts, graph.edge_index)
sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=NUM_LAYER, fanout=[FANOUT], graph_data=graph, workers=10,is_root_ts = 0,policy = 'recent' if args.recent else 'uniform',graph_name = DATA+'_train')
train_data = torch.masked_select(graph.edge_index,graph.data.train_mask).reshape(2,-1)
train_ts = torch.masked_select(graph.edge_ts,graph.data.train_mask)
val_data = torch.masked_select(graph.edge_index,graph.data.val_mask).reshape(2,-1)
val_ts = torch.masked_select(graph.edge_ts,graph.data.val_mask)
test_data = torch.masked_select(graph.edge_index,graph.data.test_mask).reshape(2,-1)
test_ts = torch.masked_select(graph.edge_ts,graph.data.test_mask) 
train_data = DataSet(edges = train_data,ts =train_ts,labels = torch.ones(train_data.shape[-1]),eids = torch.nonzero(graph.data.train_mask).view(-1))
test_data = DataSet(edges = test_data,ts =test_ts,labels = torch.ones(test_data.shape[-1]),eids = torch.nonzero(graph.data.test_mask).view(-1))
val_data = DataSet(edges = val_data,ts = val_ts,labels = torch.ones(val_data.shape[-1]),eids = torch.nonzero(graph.data.val_mask).view(-1))
neg_sampler = NegativeSampling('triplet')
trainloader = DistributedDataLoader('train',graph,train_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = BATCH_SIZE,shuffle=False,cache_memory_size = 0,drop_last=True,cs = 1)
testloader = DistributedDataLoader('test',graph,test_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = BATCH_SIZE,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)
valloader = DistributedDataLoader('val',graph,val_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = BATCH_SIZE,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)

n_node_features = 0 if graph.data.x is None else graph.data.x.shape[1]
n_edge_features = 0 if graph.data.edge_attr is None else graph.data.edge_attr.shape[1]
combine_first = False
# if 'combine_neighs' in train_param and train_param['combine_neighs']:
#     combine_first = True
for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}_{i}.pth'
  Path("results/").mkdir(parents=True, exist_ok=True)
  if args.use_memory:
    #注册Memory
    mailbox = SharedMailBox(device = device) 
    #mailbox = SharedRPCMemoryManager('cuda')暂时不启用
    #构建mailbox的映射，partptr是分区划分的数组，默认存在graph里面，local_num_node是本地数量
    mailbox.build_map(local_num_nodes=graph.partptr[rank+1]-graph.partptr[rank],partptr=graph.partptr)
    mailbox.create_empty_memory(memory_param,n_edge_features)
    dist.barrier()
  else:
    mailbox = None
  model = TGN(device=device,
            n_node_features=n_node_features, n_edge_features=n_edge_features,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            use_inner_product=args.use_inner_product,
            dyrep=args.dyrep).to(device)
  model = DDP(model,find_unused_parameters=True)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  model = model.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  sample_time = time.time()

  for epoch in range(NUM_EPOCH):
    train_aps = list()
    cnt = 0
    start_epoch = time.time()
    model.train()
    m_loss = []
    if mailbox is not None:
        mailbox.reset()
    last_time = time.time()
    # with torch.autograd.detect_anomaly():
    for batchData in trainloader:
        loss = 0
        sample_time += time.time() - last_time
        t_tot_s = time.time()
        t_prep_s = time.time()
        size = batchData.roots.edges.shape[1]
        edge_features = graph.data.edge_attr[batchData.roots.eids].to(device)
        # pdb.set_trace()
        pos_prob, neg_prob = model(batchData, edge_features, mailbox=mailbox, n_neighbors=FANOUT)
        loss = criterion(pos_prob, torch.ones_like(pos_prob)) 
        loss = loss + criterion(neg_prob, torch.zeros_like(neg_prob))
        # print("Before backwarad")
        loss.backward(retain_graph=True)
        # print(cnt)
        cnt+=1
        optimizer.step()
        m_loss.append(loss.item())
        y_pred = torch.cat([pos_prob, neg_prob], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_prob.size(0)), torch.zeros(neg_prob.size(0))], dim=0)
        train_aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
    train_ap = float(torch.tensor(train_aps).mean()) 
    print('\ttrain mean loss:{:.4f}  train ap:{:4f}'.format(np.mean(m_loss),train_ap))