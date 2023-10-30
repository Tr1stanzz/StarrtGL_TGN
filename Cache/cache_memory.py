
import parser
import time
import distparser as parser
from graph_store import _get_graph
from share_memory_util import _copy_to_share_memory, _get_existing_share_memory, _get_from_share_memory
import torch
from torch.distributed.rpc import RRef, rpc_async, remote
from os.path import abspath, join, dirname
import sys
sys.path.insert(0, join(abspath(dirname(__file__))))
from cpu_cache_manager import cache_data2mem,get_from_cache;
cache_index = {}
cache_mem = {}
sparse_index = {}

#def get_from_cpu(nodeId):
def _get_local_attr(data_name,nodes):   
    graph = _get_graph(data_name)
    local_id = graph.get_localId_by_partitionId(parser._get_worker_rank(),nodes)
    return graph.select_attr(local_id)

def _request_remote_attr(rank,data_name,nodes):
    t1  = time.time()
    fut = rpc_async(
        parser._get_RPC_NAME().format(rank),
        _get_local_attr,
        args=(data_name,nodes,)
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


def _split_node_part_and_submit(data_name,node_id_list):
    t0 = time.time()
    graph = _get_graph(data_name)
    worker_size = parser._get_world_size()
    local_rank = parser._get_worker_rank()
    futs = []
    for rank in range(worker_size):
        if(rank != local_rank):
            part_mask = (graph.partptr[rank]<=node_id_list) & (node_id_list<graph.partptr[rank+1])
            part_node = torch.masked_select(node_id_list,part_mask)
            sendlen = 0
            totlen = len(part_node)
            limit_len = 500
            if(part_node.size(0) != 0):
                while(totlen - sendlen >= limit_len):
                    futs.append((part_node[sendlen:sendlen+limit_len],_request_remote_attr(rank,data_name,part_node[sendlen:sendlen+limit_len])))
                    sendlen += limit_len
                if(sendlen < totlen):
                    futs.append((part_node[sendlen:],_request_remote_attr(rank,data_name,part_node[sendlen:])))
    local_mask = (graph.partptr[local_rank]<=node_id_list) & (node_id_list<graph.partptr[local_rank+1])
    local_node = torch.masked_select(node_id_list,local_mask)
    #logger.debug('size {},split {} {} {}'.format(node_id_list.size(0),t2-t1,t3-t2,t4-t3))
    return local_node,futs

def create_empty_cache(data_name,mem_size,x_len):
    mem = torch.zeros(mem_size,x_len)
    mem.contiguous()
    indx = torch.zeros(mem_size).long()
    indx.contiguous()
    cache_index[data_name] = indx
    cache_mem[data_name] = mem

def load_data2cache(data_name,index):
    local,futs = _split_node_part_and_submit(data_name,index)
    num = 0
    mem = cache_mem[data_name]
    indx = cache_index[data_name]
    for id,fut in futs:
      while(fut.done() == False):
         continue
      f = fut.value()
      mem[num:num+len(f),:] = f[:,:]
      indx[num:num+len(f)] = id[:]
      num = num + len(f)
    print('len',len(indx),mem,indx)
    #while(True):
    #    continue
    cache_data2mem(data_name,indx,mem)
    #ind = torch.unsqueeze(indx,0)
    #values = torch.arange(1,len(indx)+1)
    #print(len(indx))
    #sparse_index[data_name] =  dict(zip(indx.tolist(),torch.arange(len(indx)).tolist()))#torch.sparse_coo_tensor(indices = ind,values=values)
    #print(sparse_index[data_name].size(-1))

def load_cpu_cache_memory(data_name,indx_data,mem_data):
   name,data_shape,data_dtype =indx_data
   shm = _get_existing_share_memory(name)
   idx = _get_from_share_memory(shm,data_shape,data_dtype)
   name,data_shape,data_dtype = mem_data
   shm = _get_existing_share_memory(name)
   mem = _get_from_share_memory(shm,data_shape,data_dtype)
   cache_index[data_name] = idx
   cache_mem[data_name] = mem
all_time = 0
cnt = 0
def get_cpu_cache_data(data_name,index):
    #return None,index,None
    cache_data = get_from_cache(data_name,index)
    global all_time
    all_time = all_time + len(cache_data.cache_index)
    global cnt
    cnt = cnt+len(index)
    print(len(cache_data.cache_index),len(cache_data.uncache_index),len(index),all_time/cnt)

    return cache_data.cache_index,cache_data.uncache_index,cache_data.cache_data
    #t0 = time.time()
    #ind = []
    #sparse = sparse_index[data_name]
    #is_cached = []
    #for i in index:
    #    id = int(i.item())
        #print(id)
    #    if id in sparse:
    #        ind.append(sparse[id])
    #        is_cached.append(True)
    #    else:
    #        is_cached.append(False)
        #if i < sparse.size(-1):
        #    idx = sparse[i]
        #    if(idx != 0): 
        #        ind.append(sparse[i])
        #        is_cached.append(True)
        #    else:
        #        is_cached.append(False)
        #else:
        #    is_cached.append(False)
    #t1 = time.time()
    #is_cached =torch.tensor(is_cached)
    #cache_index = torch.masked_select(index,is_cached)
    #uncache_index = torch.masked_select(index,~is_cached)
    #if ind:
    #    ind = torch.tensor(ind)
    #    cache_value = torch.index_select(cache_mem[data_name],0,ind)
    #else:
    #    cache_value = None
    #global all_time
    #all_time = all_time +t1-t0#len(ind)
    #global cnt
    #cnt = cnt+1#len(index)
    #print(all_time/cnt)
    #return cache_index,uncache_index,cache_value