import sys
from os.path import abspath, join, dirname
import time

sys.path.insert(0, join(abspath(dirname(__file__))))
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from multiprocessing import Barrier
from multiprocessing.connection import Client
import queue
import sys
import argparse
import traceback
from torch.distributed.rpc import RRef, rpc_async, remote
import torch.distributed.rpc as rpc
import torch
from Cache.cache_memory import get_cpu_cache_data
from part.Utils import GraphData
from Sample.neighbor_sampler import NeighborSampler
import time
from typing import Optional
import torch.distributed as dist
import torch.multiprocessing as map
from BatchData import BatchData
import asyncio 
import os
from Sample.neighbor_sampler import NeighborSampler
from graph_store import _get_graph
from shared_graph_memory import GraphInfoInShm
import distparser as parser
from logger import logger
from countTPS import tps
from Cache.presample_cores import update_count,get_max_rank
from part.torch_utils import sparse_get_index

WORKER_RANK = parser._get_worker_rank()
NUM_SAMPLER = parser._get_num_sampler()
WORLD_SIZE = parser._get_world_size()
QUEUE_SIZE =parser._get_queue_size()
MAX_QUEUE_SIZE = parser._get_max_queue_size()
RPC_NAME = parser._get_RPC_NAME()



#@rpc.functions.async_execution
def _get_local_attr(data_name,nodes,eid):   
    graph = _get_graph(data_name)
    if(nodes.size(0)!=0):
        local_id = graph.get_localId_by_partitionId(parser._get_worker_rank(),nodes)
        node_attr = graph.select_attr(local_id)
    else:
        node_attr = None
    if(eid is not None and eid.size(0)!=0 and graph.has_edge_attr() is True):
        edge_attr = graph.select_edge_attr_by_global(parser._get_worker_rank(),eid)
    else:
        edge_attr = None
    return node_attr,edge_attr

def _request_remote_attr(rank,data_name,nodes,eid):
    t1  = time.time()
    fut = rpc_async(
        parser._get_RPC_NAME().format(rank),
        _get_local_attr,
        args=(data_name,nodes,eid,)
    )
    #logger.debug('request {}'.format(time.time()-t1))
    return fut
#ThreadPoolExecutor pool 
def _request_all_remote_attr(data_name,nodes_list):
    worker_size = parser._get_world_size()
    worker_rank = parser._get_worker_rank()
    futs = []
    for rank in range(worker_size):
        if(rank == worker_rank):
            futs.append(None)
            continue
        else:
            if(nodes_list[rank].size(0) == 0):
                futs.append(None)
            else:
                futs.append(_request_remote_attr(rank,data_name,nodes_list[rank]))
    return futs

def _check_future_finish(futs):
    check = True
    for _,fut in futs:
        if fut is not None and fut.done() is False:
            check = False
    return check
cnt_remote = 0
total_t0 = 0
total_t1 = 0
total_t2 = 0
total_t3 = 0
total_t4 = 0
cnt = 0
def _split_node_part_and_submit(data_name,node_id_list,eid = None,is_cached = False):

    graph = _get_graph(data_name)
    worker_size = parser._get_world_size()
    local_rank = parser._get_worker_rank()
    futs = []
    #local_mask = (graph.partptr[local_rank]<=node_id_list) & (node_id_list<graph.partptr[local_rank+1])
    if isinstance(node_id_list,list):
        local_node = node_id_list[0]#torch.masked_select(node_id_list,local_mask)
        node_id_list = node_id_list[1]#torch.masked_select(node_id_list,~local_mask)
    else:
        local_mask = (graph.partptr[local_rank]<=node_id_list) & (node_id_list<graph.partptr[local_rank+1])
        local_node = node_id_list[local_mask]
        node_id_list = node_id_list[~local_mask]
    if eid is not None:
        local_mask = (graph.edgeptr[local_rank]<=eid) & (eid<graph.edgeptr[local_rank+1])
        local_eid = eid[local_mask]
        eid_list = eid[~local_mask]
    else:
        local_eid =None 
    len0 = len(node_id_list)
    #缓存里没有local节点
    if is_cached:
        cached_list,node_id_list,value = get_cpu_cache_data(data_name,node_id_list)
    else:
        cached_list = None
        value = None
    t4 = time.time()
    global cnt_remote
    for rank in range(worker_size):
        if(rank != local_rank):
            part_mask = (graph.partptr[rank]<=node_id_list) & (node_id_list<graph.partptr[rank+1])
            part_node = torch.masked_select(node_id_list,part_mask)
            if eid is not None:
                part_mask = (graph.edgeptr[rank]<=eid_list) & (eid_list<graph.edgeptr[rank+1])
                part_edge = torch.masked_select(eid_list,part_mask)
            else:
                part_edge = None
            sendlen = 0
            totlen = len(part_node)
            limit_len = 500
            if(part_node.size(0) != 0 | (part_edge is not None and part_edge.size(0) != 0)):
                futs.append(((part_node,part_edge),_request_remote_attr(rank,data_name,part_node,part_edge)))
                            #if(part_node.size(0) != 0):
            #    futs.append((part_node,_request_remote_attr(rank,data_name,part_node)))
           #tps.count_mess = tps.count_mess + 1
           #tps.t_mask =tps.t_mask + t2 - t1
           #tps.t_rpc = tps.t_rpc + t3 -t2
           #cnt_remote = cnt_remote + len(part_node)
    #tps.print_t2()


    #logger.debug('t0 {} t1 {} t2 {} t3 {} t4 {}'.format(total_t0/cnt,total_t1/cnt,total_t2/cnt,total_t3/cnt,total_t4/cnt))
    return (local_node,local_eid),futs,cached_list,value

def _get_batch_feature(data_name,local_nodes,futs,cached=False,value=None):
    nids = [local_nodes[0]]
    eids = [local_nodes[1]]
    #print(local_nodes[0],local_nodes[1])
    x,edge_attr = _get_local_attr(data_name,local_nodes[0],local_nodes[1])
    x = [x]
    if edge_attr is not None:
        edge_attr = [edge_attr]
    if cached is True and value is not None:
        nids.append(cached)
        x.append(value)
    for (part_node,part_feature) in futs:
        nids.append(part_node[0])
        x.append(part_feature.value()[0])
        if edge_attr is not None and part_feature.value()[1] is not None:
            eids.append(part_node[1])
            edge_attr.append(part_feature.value()[1])
    nids = torch.cat(nids,0)
    x = torch.cat(x,0)
    if edge_attr is not None:
        eids = torch.cat(eids,0)
        edge_attr = torch.cat(edge_attr,0)
    else:
        eids  = None
    return nids,eids,x,edge_attr
    
def _get_batch_data(kwargs):
    data_name = kwargs.get("data_name")
    input_data= kwargs.get("input_data")
    local_nodes,futs,cached,value = kwargs.get("union_args")
    sampled_out = kwargs.get("sampled_out")
    meta_data = sampled_out.metadata
    edge_index = sampled_out.edge_index_list
    edge_id = sampled_out.eid_list
    nids,eids,x,edge_attr = _get_batch_feature(data_name,local_nodes,futs,cached,value)
    if meta_data is None:
        input_data.nodes = sparse_get_index(input_data.nodes.contiguous(),nids.contiguous())
    else:
        if 'edge_label_index' in meta_data:
            meta_data['edge_label_index'] = sparse_get_index(meta_data['edge_label_index'], nids.contiguous).view(2,-1)
        else : 
            for k,v in meta_data:
                meta_data[k] = sparse_get_index(v.contiguous(), nids.contiguous)
    for i in range(len(edge_index)):
        edge_index[i]  = sparse_get_index(edge_index[i].view(-1),nids.contiguous()).view(2,-1)
    if eids is not None:
        for i in range(len(edge_id)):
            edge_id[i] = sparse_get_index(edge_id[i].contiguous(),eids.contiguous())
    else:
        edge_id = None
    return BatchData(nids = nids,x=x,edge_index = edge_index,root=input_data, eid =edge_id, edge_attr = edge_attr,meta_data = meta_data)

def _get_temporal_batch_data(kwargs):
    data_name = kwargs.get("data_name")
    input_data= kwargs.get("input_data")
    local_nodes,futs,cached,value = kwargs.get("union_args")
    sampled_out = kwargs.get("sampled_out")
    nids,eids,x,edge_attr = _get_batch_feature(data_name,local_nodes,futs,cached,value)
    meta_data = sampled_out.metadata
    edge_ts = sampled_out.eid_ts_list
    edge_id = sampled_out.eid_list if eids is not None else None
    meta_data = sampled_out.metadata
    edge_index = sampled_out.edge_index_list
    if meta_data is None:
        input_data.nodes = sparse_get_index(input_data.nodes.contiguous(),nids.contiguous())
    else:
        if 'edge_label_index' in meta_data:
            meta_data['edge_label_index'] = sparse_get_index(meta_data['edge_label_index'].view(-1), nids.contiguous()).view(2,-1)
        else : 
            for k,v in meta_data.items():
                meta_data[k] = sparse_get_index(v.long().contiguous(), nids.contiguous())
    for i in range(len(edge_index)):
        edge_index[i]  = sparse_get_index(edge_index[i].view(-1),nids.contiguous()).view(2,-1)
    if edge_id is not None:
        for i in range(len(edge_id)):
            edge_id[i] = sparse_get_index(edge_id[i].contiguous(),eids.contiguous())
    return BatchData(nids = nids, x=x,edge_index = edge_index,roots = input_data,eids = edge_id
                     ,edge_attr= edge_attr,edge_ts = edge_ts,meta_data= meta_data)

    
    
def _sample_node_neighbors_server(data_name,input_data,sampled_out,iscached = False):
    '''
    sample the struct of the subgraph
    '''
    #print('sample node neighbors server')
    t1 = time.time()
    neighbor_nodes = sampled_out.node
    eids = torch.cat(sampled_out.eid_list,-1).unique()
    local_info,futs,cached,value = _split_node_part_and_submit(data_name,node_id_list = neighbor_nodes,eid = eids,is_cached = iscached)
    t2 = time.time()
    #logger.debug('sample server {}'.format(t2-t1))
    return local_info,futs,cached,value

def _temporal_sample_node_neighbors_server(data_name,input_data,sampled_out,iscached = False):
    '''
    sample the struct of the subgraph
    '''
    #print('sample node neighbors server')
    metadata = sampled_out.metadata
    edge_index = sampled_out.edge_index_list
    eid = sampled_out.eid_list
    if metadata is not None:
        if 'edge_label_index' in metadata:
            nids = torch.cat((metadata['edge_label_index'],torch.cat(edge_index,-1).view(-1)),0).unique()
        
        else:
            lis = []
            for k,v in metadata.items():
                lis.append(v.long().contiguous())
            lis.append(torch.cat(edge_index,-1).view(-1))
            nids = torch.cat(lis,0).unique()    
    else:
        nids = torch.cat((input_data.nodes,torch.cat(edge_index,-1).view(-1)),0).unique()
    eids = torch.cat(eid,0).unique()
    local_info,futs,cached,value = _split_node_part_and_submit(data_name,node_id_list = nids,eid = eids,is_cached = iscached)
    #logger.debug('sample server {}'.format(t2-t1))
    return local_info,futs,cached,value

def _sample_node_neighbors_single(graph,graph_name,sampler,neg_sampler,input_data):
    #本地特征值
    if input_data.nodes is not None:
        out = sampler.sample_from_nodes(input_data.nodes.reshape(-1),parser.SAMPLE_TYPE)
        #metadata = None
        neighbor_nodes = out.node
        sampled_edge_index = out.edge_index_list
        eids = out.eid_list
        metadata = None
    elif input_data.edges is not None:
        out = sampler.sample_from_edges(input_data.edges,parser.SAMPLE_TYPE) 
        neighbor_nodes, sampled_edge_index, metadata =out.node,out.edge_index_list,out.metadata
        eids = out.eid_list
    nids = neighbor_nodes
    edge_index = sampled_edge_index#= sampler.sample_from_nodes(input_data.reshape(-1), parser.SAMPLE_TYPE)
    for i in range(len(edge_index)):
        edge_index[i][:] = sparse_get_index(edge_index[i].view(-1).contiguous(),nids.contiguous()).view(2,-1)
    nids = nids
    x= graph.select_attr(nids)
    e_id = None
    edge_attr = None
    if graph.has_edge_attr() is True:
        e_id = out.eid_list
        eids,inverse = torch.cat(eids,-1).unique(return_inverse = True)
        edge_attr = graph.select_edge_attr(eids)
        j = 0
        for i in range(len(e_id)):
            l = len(e_id[i])
            e_id[i] = inverse[j:j+l]
            j += l
    roots = input_data
    if(roots.nodes is not None):
        roots.nodes = sparse_get_index(roots.nodes.contiguous(),nids.contiguous())
        
    elif (roots.edges is not None):
        if 'edge_label_index' in metadata:
            metadata['edge_label_index'] = sparse_get_index(metadata['edge_label_index'], nids.contiguous()).view(2,-1)
        else : 
            for k,v in metadata.items():
                metadata[k] = sparse_get_index(v, nids.contiguous)
    return BatchData(nids,edge_index,roots=roots,x=x,eids= eids,edge_attr = edge_attr,meta_data = metadata)

def _temporal_sample_node_neighbors_single(graph,graph_name,sampler,neg_sampler,input_data):
    #本地采样
    if input_data.nodes is not None:
        out =  sampler.sample_from_nodes(nodes=input_data.nodes.reshape(-1),ts=input_data.ts.reshape(-1),with_outer_sample=parser.SAMPLE_TYPE)
        edge_index, e_id, e_ts = out.edge_index_list,out.eid_list,out.eid_ts_list
        metadata = None
    elif input_data.edges is not None:
        out = sampler.sample_from_edges(input_data.edges,parser.SAMPLE_TYPE,input_data.ts,input_data.labels,neg_sampler) 
        edge_index,e_id,e_ts, metadata =  out.edge_index_list,out.eid_list,out.eid_ts_list,out.metadata
    #edge_index,e_id,e_ts = sampler.sample_from_nodes_with_before(node,ts,parser.SAMPLE_TYPE)
    if metadata is not None:
        if 'edge_label_index' in metadata:
            nids,inverse = torch.cat((metadata['edge_label_index'],torch.cat(edge_index,-1).view(-1)),0).unique(return_inverse = True)
        
        else:
            lis = []
            for k,v in metadata.items():
                lis.append(v.long().contiguous())
            lis.append(torch.cat(edge_index,-1).view(-1))
            nids,inverse = torch.cat(lis,0).unique(return_inverse = True)    
    else:
        nids,inverse = torch.cat((input_data.nodes,torch.cat(edge_index,-1).view(-1)),0).unique(return_inverse = True)
    x= graph.select_attr(nids)
    roots = input_data
    i = 0
    if(roots.nodes is not None):
        i+=len(roots.nodes)
        roots.nodes = inverse[:i]#sparse_get_index(roots.nodes.contiguous(),nids.contiguous())
    elif (roots.edges is not None):
        if 'edge_label_index' in metadata:
            i += len(metadata['edge_label_index'])
            metadata['edge_label_index'] = inverse[:i]#sparse_get_index(metadata['edge_label_index'].view(-1), nids.contiguous).view(2,-1)
        else : 
            for k,v in metadata.items():
                l = len(v)
                metadata[k] = inverse[i:i+l]#sparse_get_index(v.long().contiguous(), nids.contiguous())
                i+=l
    _edge_index = inverse[i:].view(2,-1)
    j = 0
    for i in range(len(edge_index)):
        l = edge_index[i].shape[1]
        edge_index[i] = _edge_index[:,j:j+l]
        j+=l
    eid,inverse = torch.cat(e_id,0).unique(return_inverse = True)
    edge_attr = graph.select_edge_attr(eid)
    j = 0
    for i in range(len(e_id)):
        l = len(e_id[i])
        e_id[i] = inverse[j:j+l]
        j += l
        #sparse_get_index(e_id[i].contiguous(),eid.contiguous())
    
    #roots = sparse_get_index(.contiguous(),nids.contiguous())
    return BatchData(nids = nids ,edge_index = edge_index,roots = roots, x=x,eids = e_id
                     ,edge_attr= edge_attr,edge_ts = e_ts,meta_data = metadata)
   # return BatchData(roots = input_data[0], root_ts = input_data[1],edge_index = edge_index,edge_ts= e_ts,x=x,y=y,ts=input_data[1])

#存储比例，加载到共享内存,广播缓存
def pre_sample(sampler,input_data,data_name,weight,l,r,device='cpu'):
    #以本地节点为中心进行一次采样
    nodes,_ = sampler.sample_from_nodes(input_data.nodes.reshape(-1), parser.SAMPLE_TYPE)
    nodes = torch.cat((nodes[0],nodes[1]),0)
    #print('sample nodes',nodes.masked_select(torch.logical_or(nodes<l,nodes >= r) ))
    update_count(data_name,nodes.reshape(-1),device,weight,l,r)
    
def get_cache_node(data_name,space_size):
    return get_max_rank(data_name,space_size)





