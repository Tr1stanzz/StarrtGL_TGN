from enum import Enum
import torch
import sys
from os.path import abspath, join, dirname
import numpy as np
sys.path.insert(0, join(abspath(dirname(__file__))))
from Cache.utils import create_memory
from message_worker import _get_batch_data, _sample_node_neighbors_server, _sample_node_neighbors_single, _temporal_sample_node_neighbors_single, get_cache_node, pre_sample
from part.Utils import GraphData
from typing import Optional
import torch.distributed as dist
import DistCustomPool 
from shared_graph_memory import GraphInfoInShm
import distparser as parser
from torch_geometric.data import Data
import os.path as osp
import graph_store
import math
def partition_load(root: str, algo: str = "metis") -> Data:
    rank = parser._get_worker_rank()
    world_size = parser._get_world_size()
    fn = osp.join(root, f"{algo}_{world_size}", f"{rank:03d}")
    return torch.load(fn)
class DataSet:
    def __init__(self,nodes = None,edges = None,labels = None, ts = None, **kwargs):
        self.nodes = nodes
        self.edges = edges
        self.ts = ts
        self.labels = labels
        for k, v in kwargs.items():
            setattr(self, k, v)
    def get_next(self,l = 0,r = None):
        nodes = None
        edges = None
        if hasattr(self,'nodes') and self.nodes is not None:
            if r is not None:
                nodes = self.nodes[l:r]
            else:
                nodes = self.nodes[l:]
        elif self.edges is not None :
            if r is not None:
                edges = self.edges[:,l:r]
            else:
                edges = self.edges[:,l:]
        if r is not None:
            labels = self.labels[l:r]
        else:
            labels = self.labels[l:]
        d = DataSet(nodes,edges,labels)
        for k,v in self.__dict__.items():
            if k =='edges' or k == 'nodes' or k=='labels' or v is None:
                continue
            if r is not None:
                setattr(d,k,v[l:r])
            else:
                setattr(d,k,v[l:])
        return d
    def shuffle(self):
        d = DataSet()
        len_n = len(self.nodes) if hasattr(self,'nodes') else self.edges.shape[-1]
        indx = torch.randperm(len_n)
        for k,v in self.__dict__.items():
            if v is None:
                continue
            if k == 'edges':
                setattr(d,k,v[:,indx])
            elif v is None:
                continue   
            else:
                setattr(d,k,v[indx])
        return d
    #def set_local_node_dataset(self,partptr):
    #    l,r = partptr
    #    len_n = len(self.nodes)
    #    for k,v in self.__dict__:
    #        if v is not None and len(v) == len_n:
    #            self.__setattr__(k,v.masked_select(v >= l and v < r))
    #def set_local_edge_dataset(self,partptr):
    #    l,r = partptr
    #    len_n = len(self.edges)
    #    for k,v in self.__dict__:
    #        if k == 'edges':
    #            
    #        elif v is not None and len(v) == len_n:
    #            self.__setattr__(k,v.masked_select(v >= l and v < r))
        
class DistGraphData(GraphData):
    
    def __init__(self,pdata = None,edge_index = None,path = None,full_edge = True):
        if path is not None: 
            self.rank = parser._get_worker_rank()

            path = path + '/rank_' + str(self.rank)
            print('load graph ',path)
            super(DistGraphData,self).__init__(path)
            # if(full_edge == False):
            #     self.edge_index = self.data.edge_index
        else:
            #dst和edge在一个分区，src不一定
            #本地节点id
            pdata.edge_index = pdata.edge_index.long()
            self.ids = pdata.ids
            if edge_index is not None:
                edge_index = pdata.edge_index
            #特征信息
            self.data = Data()
            self.data.x = pdata.x
            self.data.y = pdata.y
            self.data.y = self.data.y.reshape(-1)
            self.data.train_mask = pdata.train_mask
            self.data.test_mask = pdata.test_mask
            self.data.val_mask = pdata.val_mask
            print(pdata.__dict__)
            if hasattr(pdata,'edge_ts'):
                self.data.edge_ts = pdata.edge_ts
            else:
                self.edge_ts = None
            if hasattr(pdata,'edge_attr'):
                self.data.edge_attr = pdata.edge_attr
            else:
                self.data.edge_attr = None
            
            self.rank = parser._get_worker_rank()
            #通信后获得索引
            world_size = parser._get_world_size()
            sample_group = DistCustomPool.get_sample_group()
            self.partition_id = self.rank
            self.partptr = torch.zeros(world_size + 1).int()
            self.edgeptr = torch.zeros(world_size + 1).int()
            self.num_edge_part = torch.zeros(world_size+1).int()
            self.num_nodes = 0
            self.num_edges = 0
            self.partitions = world_size
            self.num_parts = self.partitions
            self.num_feasures =  pdata.x.size()[1]
            global_edge = []
            if world_size != 1 and full_edge is not False:
                for rank in range(world_size):
                    #dst在同一个分区，src不一定，映射dst和src的
                    rev_msg = [len(self.ids),len(edge_index[0,:])]
                    # print(rev_msg)
                    dist.broadcast_object_list(rev_msg,rank,group=sample_group)
                    self.num_edge_part[rank + 1] = self.num_edge_part[rank] + rev_msg[1]
                    self.num_nodes = self.num_nodes + rev_msg[0]
                    self.partptr[rank + 1] = self.num_nodes
                    self.edgeptr[rank + 1] = self.num_edges_part[rank+1]
                dict.barrier()
                dic = []
                for rank in range(world_size):
                    if rank != self.rank:
                        rev_idx = torch.zeros(self.partptr[rank+1] - self.partptr[rank]).type_as(self.ids)
                    else:
                        rev_idx = self.ids.clone()
                    dist.broadcast(rev_idx,rank,group=sample_group)
                    dic.append(rev_idx)
                dic = torch.cat(dic,0)
                idx = torch.arange(len(dic)).long()
                dict_ptr = torch.zeros_like(idx)
                dict_ptr[dic] = idx
                self.data.edge_index = dict_ptr[edge_index]
                dist.barrier()
                for rank in range(world_size):
                    #edge_index映射成新的id
                    if rank != self.rank:
                        rev_idx = torch.zeros(2,self.num_edge_part[rank + 1] - self.num_edge_part[rank]).type_as(self.data.edge_index)
                    else:
                        rev_idx = self.data.edge_index.clone()
                    # print('edge',rev_idx)
                    dist.broadcast(rev_idx,rank,group=sample_group)
                    global_edge.append(rev_idx)
                self.edge_index = torch.cat(global_edge,1)
                if hasattr(pdata,'edge_ts'):
                    global_ts = []
                    for rank in range(world_size):
                    #edge_index映射成新的id
                        if rank != self.rank:
                            rev_idx = torch.zeros(self.num_edge_part[rank + 1] - self.num_edge_part[rank]).type_as(pdata.edge_ts)
                        else:
                            rev_idx = pdata.edge_ts.clone()
                    # print('edge',rev_idx)
                    dist.broadcast(rev_idx,rank,group=sample_group)
                    global_ts.append(rev_idx)
                    self.edge_ts = torch.cat(global_ts,0)
                    self.eid = torch.arange(0,len(self.edge_ts))
                dist.barrier()
            elif world_size ==1 :
                self.num_nodes = len(self.ids)
                self.num_edges = len(edge_index[0,:])
                self.partptr[1] = self.num_nodes
                self.edgeptr[1] = self.num_edges
                self.edge_index = edge_index
                self.data.edge_index = edge_index
                if hasattr(pdata,'edge_ts'):
                    self.edge_ts = self.data.edge_ts
                
                self.eid = torch.arange(0,self.edge_index.shape[-1])
                
            else:
                for rank in range(world_size):
                    #dst在同一个分区，src不一定，映射dst和src的
                    rev_msg = [len(self.ids),len(edge_index[0,:])]
                    # print(rev_msg)
                    dist.broadcast_object_list(rev_msg,rank,group=sample_group)
                    self.num_edge_part[rank + 1] = self.num_edge_part[rank] + rev_msg[1]
                    self.num_nodes = self.num_nodes + rev_msg[0]
                    self.partptr[rank + 1] = self.num_nodes
                    self.num_edges = self.num_edges + rev_msg[1]
                    self.edgeptr[rank + 1] = self.num_edges
                #edge_index to映射成新的id
                dic = []
                for rank in range(world_size):
                    if rank != self.rank:
                        rev_idx = torch.zeros(self.partptr[rank+1] - self.partptr[rank]).type_as(self.ids)
                        #print(torch.distributed.get_backend() )
                        if torch.distributed.get_backend() == 'nccl':
                            rev_idx = rev_idx.to('cuda')
                    else:
                        rev_idx = self.ids.clone()
                        #print(torch.distributed.get_backend() )
                        if torch.distributed.get_backend() == 'nccl':
                            rev_idx = rev_idx.to('cuda')
                    dist.broadcast(rev_idx,rank,group=sample_group)
                    dic.append(rev_idx)
                dic = torch.cat(dic,0)
                idx = torch.arange(len(dic),device = dic.device)
                dict_ptr = torch.zeros_like(idx,device=idx.device)
                dict_ptr[dic] = idx
                self.edge_index = dict_ptr[edge_index].to('cpu')
                self.data.edge_index = self.edge_index
                self.eid = torch.arange(self.edgeptr[self.rank],self.edgeptr[self.rank+1])
                if hasattr(pdata,'edge_ts'):
                    self.edge_ts = self.data.edge_ts
                dist.barrier()
            self.num_edges = edge_index[0].numel()

    
    def edgeindex2ptr(self,ids,edge_index,ptr_index,rank):
        #构建映射
        dic = dict(zip(ids.tolist(),torch.arange(len(ids)).tolist()))
        for i in range(len(edge_index)):
            id = int(edge_index[i].item())
            #print(id)
            if id in dic:
                # print(id)
                if(rank!=self.rank):
                    print('part ',id)
                ptr_index[i] = dic[id] + self.partptr[rank]
        # print(ptr_index)                

class DistributedDataLoader:
    ''' 
     Args:
            data_path: the path of loaded graph ,each part 0 of graph is saved on $path$/rank_0
            num_replicas: the num of worker
            

    '''
    def __init__(
            self,
            graph_name,
            graph,
            dataset = None,
            sampler = None,
            neg_sampler = None,
            collate_fn = None,
            batch_size: Optional[int]=None,
            shuffle:bool = True,
            seed:int = 0,
            drop_last = False,
            device  = 'cpu',
            cache_policy = 'presample',
            cache_memory_size = 0,#cache存储字节大小
            cache_remote_weight = 0,
            cs = None,
            **kwargs
    ):
        assert sampler is not None
        graph_store.set_local_sampler(sampler.graph_name,sampler)
        self.pool = DistCustomPool.get_sampler_pool()
        self.cs = cs
        self.graph_name = graph_name
        self.batch_size = batch_size
        self.num_workers = parser._get_num_sampler()
        self.queue_size = parser._get_queue_size()
        #if(queue_size is None):
        #    queue_size = self.num_workers * 4 if self.num_workers > 0 else 4
        self.num_pending = 0
        # self.num_node_features = graph.num_node_features
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.drop_last = drop_last
        self.recv_idxs = 0
        self.queue = []
        self.shuffle = shuffle
        self.is_closed = False
        self.sampler = sampler
        self.epoch = 0
        self.graph = graph
        if parser._get_world_size() > 1:
            self.graph_inshm = GraphInfoInShm(graph)
        if self.pool is None:
            self.sampler_info = sampler 
        else :
            self.sampler_info = (sampler.num_nodes,sampler.num_layers,sampler.fanout,sampler.workers)#sampler#._get_sample_info()
        #self.graph_inshm._copy_sampler_to_shame(*sampler._get_neighbors_and_deg())
        self.neg_sampler = neg_sampler
        self.shuffle=shuffle
        self.seed=seed
        self.kwargs=kwargs
        self.rank = parser._get_worker_rank()
        self.dataset = dataset
        if hasattr(self.dataset,'nodes') and self.dataset.nodes is not None:
            self._get_expected_idx(drop_last,self.dataset.nodes.size(0))
            
        elif hasattr(self.dataset,'edges') and self.dataset.edges is not None:
            self._get_expected_idx(drop_last,self.dataset.edges.size(1))
        
        self.device =  device
        if self.dataset.ts is None:
            self.cache_memory_size = cache_memory_size
        else:
            self.cache_memory_size = 0
            
        iscached  = self.cache_memory_size != 0 and (self.device=='gpu' or parser._get_world_size()>1)
        
        self.iscached = iscached
        if(self.pool is not None and parser._get_world_size() > 1 ):
            self.pool.set_collate_fn(self.collate_fn,self.sampler_info,self.graph_name,self.graph_inshm)
            dist.barrier()
        elif parser._get_world_size() > 1 :
            self.local_pool = DistCustomPool.LocalSampler()
            self.local_pool.set_collate_fn(self.graph_name,self.graph_inshm,iscached)

        if self.cache_memory_size != 0 and (self.device=='gpu' or parser._get_world_size()>1):
            if cache_policy == 'presample':
                self._pre_sampler(cache_remote_weight)
            #elif cache_policy == 'degree':
                #self._degree_cache(self.data)
            
    def _pre_sampler(self,weight):
        for _ in range(10):
            self.__iter__()
            while self.recv_idxs < self.expected_idx:
            #print(self.recv_idxs)
                batch_data = self._next_data()
        #print(self.graph.partptr[self.rank],self.graph.partptr[self.rank+1])
                pre_sample(self.sampler,batch_data,self.graph_name,weight,self.graph.partptr[self.rank],self.graph.partptr[self.rank+1],self.device)
                self.recv_idxs += 1
        cache_node_list = get_cache_node(self.graph_name,self.cache_memory_size)
        create_memory(self.graph_name,self.graph,self.cache_memory_size,cache_node_list,self.device)
                

    def _get_expected_idx(self,drop_last,data_length):
        if parser._get_world_size() > 1:
            world_size = parser._get_world_size()
            sample_group = DistCustomPool.get_sample_group()
            expected_data = data_length
            for rank in range(world_size):
                len = torch.tensor(data_length)
                if(torch.distributed.get_backend() == 'nccl'):
                    len = len.to('cuda')
                dist.broadcast(len,rank,group=sample_group)
                #if(drop_last is True):
                expected_data = min(expected_data,int(len.item()))
                #else:
                #    expected_data = max(expected_data,int(len.item()))
#
            self.expected_idx = expected_data // self.batch_size
#
            if(not self.drop_last and expected_data % self.batch_size != 0 and self.cs is None):
                self.expected_idx += 1
            print('expected_index ',self.expected_idx)
#
        else:
            self.expected_idx = data_length// self.batch_size
            if(not self.drop_last and data_length % self.batch_size != 0):
                self.expected_idx += 1


    def __next__(self):
        #print(self.sampler,self.num_samples,self.kwargs)
        #self.sampleGraph(self.sampler,self.num_samplers,self.kwargs)
        #return self.sampleGraph(self.sampler,self.num_samples,**self.kwargs)
        #print(self.recv_idxs,self.expected_idx)
        if(self.pool is None and parser._get_world_size() == 1):
            if self.recv_idxs < self.expected_idx:
                batch_data = self._sample_next_batch()
                self.recv_idxs += 1
                assert batch_data is not None
                return batch_data
            else :
                raise StopIteration
        else :
            
            num_reqs = min(self.queue_size - self.num_pending,self.expected_idx - self.submitted)
            for _ in range(num_reqs):
                result = self.local_pool.get_result()
                if result is not None:
                    self.recv_idxs += 1
                    self.num_pending -= 1
                    return result
                else:
                    next_data = self._next_data()
                    if next_data is None:
                        continue
                    assert next_data is not None
                    if next_data.ts is None:
                        self.local_pool.sample_next(self.graph_name,self.sampler_info,self.neg_sampler,next_data)
                    else:
                        self.local_pool.temporal_sample_next(self.graph_name,self.sampler_info,self.neg_sampler,next_data)
                    self.submitted = self.submitted + 1
                    self.num_pending = self.num_pending + 1
                    
            while(self.recv_idxs < self.expected_idx):
                result = self.local_pool.get_result()
                if result is not None:
                    self.recv_idxs += 1
                    self.num_pending -= 1
                    return result
                
            assert self.num_pending == 0
            raise StopIteration
                    
    def _sample_next_batch(self):
        next_data = self._next_data()
        if next_data is None:
            return None
        if next_data.ts is not None:
            batch_data = _temporal_sample_node_neighbors_single(self.graph,self.graph_name,self.sampler,self.neg_sampler,next_data)
        else:
            batch_data = _sample_node_neighbors_single(self.graph,self.graph_name,self.sampler,self.neg_sampler,next_data)

        return batch_data


    def _next_data(self):
        if self.current_pos >= len(self.input_dataset.labels):
            return None
        next_idx = None
        if self.current_pos + self.batch_size > len(self.input_dataset.labels):
            if self.drop_last:
                return None
            else:
                next_data = self.input_dataset.get_next(self.current_pos)
                self.current_pos = 0
        else:
            next_data = self.input_dataset.get_next(self.current_pos,self.current_pos + self.batch_size)
            self.current_pos += self.batch_size
            
        return next_data

        
    def __iter__(self):
        if self.cs is None:
            if self.shuffle:
                self.input_dataset = self.dataset.shuffle()
            else:
                self.input_dataset = self.dataset
            self.recv_idxs = 0
            self.current_pos = 0
            self.num_pending = 0
            self.submitted = 0
        else:
            self.input_dataset = self.dataset
            self.recv_idxs = 0
            if self.drop_last and self.cs is not None:
                self.current_pos = int(math.floor(np.random.uniform(0,self.batch_size/self.cs))*self.cs)
                if self.drop_last:
                    self.expected_idx -= 1
            self.num_pending = 0
            self.submitted = 0
        return self



    def __del__(self):
        if self.pool is not None:
            self.pool.delete_collate_fn(self.graph_name)
    

    def set_epoch(self,epoch):
        self.epoch = epoch 
        
        
        
