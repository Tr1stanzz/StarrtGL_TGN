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
from model import GraphSAGE
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
    pdata = partition_load("./wiki", algo="metis")    
    graph = DistGraphData(pdata = pdata,edge_index= pdata.edge_index, full_edge = False)
    print(graph.data.edge_ts.shape)
    print(graph.edge_index.shape)
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_shift_time_statics(graph.data.edge_ts, graph.edge_index)
    print("Mean time shift src: {}, std: {}".format(mean_time_shift_src, std_time_shift_src))
    print("Mean time shift dst: {}, std: {}".format(mean_time_shift_dst, std_time_shift_dst))
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10,10], graph_data=graph, workers=10,is_root_ts = 0,graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,graph.data.train_mask).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,graph.data.train_mask)
    val_data = torch.masked_select(graph.edge_index,graph.data.val_mask).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,graph.data.val_mask)
    test_data = torch.masked_select(graph.edge_index,graph.data.test_mask).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,graph.data.test_mask) 
    train_data = DataSet(edges = train_data,ts =train_ts,labels = torch.ones(train_data.shape[-1]))
    test_data = DataSet(edges = test_data,ts =test_ts,labels = torch.ones(test_data.shape[-1]))
    val_data = DataSet(edges = val_data,ts = val_ts,labels = torch.ones(val_data.shape[-1]))
    neg_sampler = NegativeSampling('triplet')
    trainloader = DistributedDataLoader('train',graph,train_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = 1)
    testloader = DistributedDataLoader('test',graph,test_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)
    valloader = DistributedDataLoader('val',graph,val_data,sampler = sampler,neg_sampler=neg_sampler,batch_size = 600,shuffle=False,cache_memory_size = 0,drop_last=True,cs = None)


    # count_node = 0
    # count_edge = 0
    # count_x_byte = 0
    # start_time = time.time()
    cnt = 0
    for batchData in trainloader:
        if cnt < 1:
            sources_batch = batchData.meta_data['src_id']
            destinations_batch = batchData.meta_data['dst_pos_id']
            negatives_batch = batchData.meta_data['dst_neg_id']
            from_ts = batchData.edge_ts[0][1,:]
            print(batchData.roots.labels.shape)
            # print(torch.unique(from_ts).shape)
            # print(torch.unique(batchData.roots.ts).shape)
            # # print(batchData.meta_data)
            # print(batchData.edge_index[0].shape)
            # print(batchData.eids[0].shape)
            # print(torch.min(batchData.eids[0]))
            # print(torch.max(batchData.eids[0]))
            # print(torch.unique(batchData.eids[0]).shape)
            # print(batchData.nids.shape)
            # print(batchData.x.shape)
            # print(batchData.edge_ts[0].shape)
            # print(batchData.edge_attr.shape)
            # print(batchData.roots.ts.shape)
            # print(batchData.roots.edges.shape)
            # print(sources_batch.shape, destinations_batch.shape, negatives_batch.shape)
            # pass
        else:
            break
        cnt = cnt+1
        print(cnt)
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
