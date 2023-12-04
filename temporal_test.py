import argparse
import os
import numpy as np
from DistGraphLoader import DataSet, partition_load
from Sample.base import NegativeSampling
from Sample.neighbor_sampler import NeighborSampler
path1=os.path.abspath('.')  
import torch
from part.Utils import GraphData
from DistGraphLoader import DistGraphData
from DistGraphLoader import DistributedDataLoader
from torch_geometric.data import Data
import distparser
from DistCustomPool import CustomPool
import DistCustomPool
from torch.distributed import rpc
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributed.rpc import TensorPipeRpcBackendOptions
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import BatchData
from part.Utils import GraphData
from torch_geometric.sampler.neighbor_sampler import NeighborSampler as PYGSampler
from shared_mailbox import SharedMailBox, SharedRPCMemoryManager
from utils.utils import EarlyStopMonitor, get_neighbor_finder, select_free_gpu, parse_config
# from tgl.sampler import *
# from tgl.utils import *
"""
test command 
python test.py --world_size 2 --rank 0 
--world_size', default=4, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='rank of the worker')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
parser.add_argument('--num_sampler', type=int, default=10, metavar='S',
                    help='number of samplers')
parser.add_argument('--queue_size', type=int, default=10, metavar='S',
                    help='sampler queue size')
"""
sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'l2':5e-7
             }
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

def check_samples(batchData):
    edge_index = batchData.edge_index

def main():   
    DistCustomPool.init_distribution('127.0.0.1',9675,'127.0.0.1',10023,backend = "gloo")
    sample_param, memory_param, gnn_param, train_param = parse_config('config/TGN.yml')
    pdata = partition_load("./data/wiki", algo="metis")    
    args = distparser.args
    rank = distparser._get_worker_rank()
    graph = DistGraphData(pdata = pdata,edge_index= pdata.edge_index, full_edge = False)
    gnn_dim_node = 0 if graph.data.x is None else graph.data.x.shape[1]
    gnn_dim_edge = 0 if graph.data.edge_attr is None else graph.data.edge_attr.shape[1]
    print(graph.data.edge_ts.shape)
    print(graph.edge_index.shape)
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_shift_time_statics(graph.data.edge_ts, graph.edge_index)
    print("Mean time shift src: {}, std: {}".format(mean_time_shift_src, std_time_shift_src))
    print("Mean time shift dst: {}, std: {}".format(mean_time_shift_dst, std_time_shift_dst))
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10], graph_data=graph, workers=10,is_root_ts = 0,graph_name = "wiki_train")
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
    trainloader = DistributedDataLoader('train',graph,train_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = 1)
    testloader = DistributedDataLoader('test',graph,test_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)
    valloader = DistributedDataLoader('val',graph,val_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)
    mailbox = SharedMailBox(device = torch.device('cuda')) 
    #mailbox = SharedRPCMemoryManager('cuda')暂时不启用
    #构建mailbox的映射，partptr是分区划分的数组，默认存在graph里面，local_num_node是本地数量
    mailbox.build_map(local_num_nodes=graph.partptr[rank+1]-graph.partptr[rank],partptr=graph.partptr)
    mailbox.create_empty_memory(memory_param,gnn_dim_edge)
    dist.barrier()

    # count_node = 0
    # count_edge = 0
    # count_x_byte = 0
    # start_time = time.time()
    cnt = 0
    for batchData in trainloader:
        if cnt <= 5:
            # print(batchData.roots.eids)
            print(batchData.roots.edges.shape)
            # print(batchData.roots.ts)
            # print(batchData.nids.shape)
            # print(batchData.nids)

            # edge_index = batchData.edge_index[0]
            # edge_ts = batchData.edge_ts[0]
            # eids = batchData.eids[0]
            # neighbor_data = torch.cat([edge_index,edge_ts[0,:].unsqueeze(0),eids.unsqueeze(0)],dim = 0)
            # sample_nodes, node_inverse_indices, node_count = torch.unique_consecutive(edge_index[1,:], return_counts=True, return_inverse=True)
            # sources_batch = batchData.meta_data['src_id']
            # destinations_batch = batchData.meta_data['dst_pos_id']
            # negatives_batch = batchData.meta_data['dst_neg_id']
            # timestamps = batchData.roots.ts
            # is_sorted = torch.equal(timestamps, torch.sort(timestamps)[0])
            # print(is_sorted)
            # sample_ts, ts_inverse_indices, ts_count = torch.unique_consecutive(edge_ts[1,:], return_counts=True, return_inverse=True)
            # _, node_begins = np.unique(node_inverse_indices.cpu().numpy(), return_index=True)
            # _, ts_begins = np.unique(ts_inverse_indices.cpu().numpy(), return_index=True)
            # ts_begins = torch.tensor(ts_begins)
            # idx = 123
            # print("src:", sources_batch[idx], "dst:", destinations_batch[idx], "neg:", negatives_batch[idx], " ts:", timestamps[idx])
            # ts_idx = torch.nonzero(sample_ts==timestamps[idx])[:,0]
            # print("ts_idx:", ts_idx)
            # neighbor_lst = []
            # for s, c in zip(ts_begins[ts_idx], ts_count[ts_idx]):
            #     neighbors = torch.narrow(neighbor_data,1,s,c)
            #     if(torch.unique(torch.cat([neighbors[1,:], sources_batch[idx].view(-1)],dim=0),dim=0).shape[0] == 1):
            #         neighbor_lst.append(neighbors)
            # print("neighbor_lst:", neighbor_lst)
            # neighbor_tensor  = neighbor_lst[0]
            # source_neighbors, source_edge_idxs, source_edge_times = neighbor_tensor[0,:].numpy(), neighbor_tensor[3,:].numpy(), neighbor_tensor[2,:].numpy()
            # src = sources_batch
            # dst = destinations_batch
            # nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)

            # print(sample_ts, ts_begins, ts_count)
            # print(batchData.roots.edges[:,idx], batchData.roots.ts[idx])
            # print(batchData.nids)
            # print(sources_batch)
            # print(destinations_batch)
            # print(sources_batch)
            # print(torch.unique(from_ts).shape)
            # print(torch.unique(batchData.roots.ts).shape)
            # # print(batchData.meta_data)
            # print(batchData.edge_index[0][:,:20])
            # print(batchData.edge_index[0].shape)
            # print(batchData.eids[0].shape)
            # print(torch.min(batchData.eids[0]))
            # print(torch.max(batchData.eids[0]))
            # print(torch.unique(batchData.eids[0]).shape)
            # print(batchData.nids.shape)
            # print(batchData.nids.shape)
            # memory,memory_ts,mail,mail_ts = mailbox.get_memory_by_scatter(batchData.nids.to('cuda'))
            # print(memory.shape)
            # print(memory_ts.sum(0))
            # print(mail.shape)
            # print(mail_ts.sum(0))
            # mail_ts[[10,11]] = 1
            # unique_node_ids = torch.nonzero(mail_ts > memory_ts).squeeze(1)
            # print(unique_node_ids)
            print(batchData.x.shape)
            # print(batchData.edge_ts[0].shape)
            # print(batchData.edge_ts[0])
            # print(batchData.eids[0])
            # print(batchData.eids[0].shape)
            # print(batchData.edge_attr.shape)
            # print(batchData.roots.ts)
            # print(batchData.roots.edges)
            # print(sources_batch.shape, destinations_batch.shape, negatives_batch.shape)
            # pass
        elif cnt >5:
            break
        cnt = cnt+1
    #    count_node += batchData.nids.size(0)
    #    count_x_byte += batchData.x.numel()*batchData.x.element_size()
    #    for edge_list in batchData.edge_index:
    #        count_edge += edge_list.size(1)
    #    #print(batchData.x)
    #   #count_edge += batchData.edge_index.size(1)
    #    dt = time.time() - start_time
    #    print('{} count node {},count edge {}, node TPS {},edge TPS {}, x size {}, x TPS {} byte'
    #          .format(cnt,count_node,count_edge,count_node/dt,count_edge/dt,count_x_byte,count_x_byte/dt),batchData.x.size(0),batchData.x.element_size(),batchData.x.numel())
    DistCustomPool.close_distribution()
if __name__ == "__main__":
    main()
