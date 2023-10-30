import sys
from os.path import abspath, join, dirname
from collections import deque
from enum import Enum
from multiprocessing.connection import Client
import queue
from threading import Thread
import traceback
from Cache.cache_memory import load_cpu_cache_memory, load_data2cache
import torch
from torch.distributed.rpc import RRef, rpc_async, remote
from Sample.base import SampleOutput
from Sample.neighbor_sampler import NeighborSampler
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import graph_store
import os
import time
from message_worker import _check_future_finish, _get_batch_data,_get_temporal_batch_data, _sample_node_neighbors_server , _temporal_sample_node_neighbors_server
from rpc_server import  RpcCommand, close_rpc,  start_rpc_listener,start_rpc_caller
import distparser as parser
import sys
from logger import logger
from countTPS import tps

WORKER_RANK = parser._get_worker_rank()
NUM_SAMPLER = parser._get_num_sampler()
WORLD_SIZE = parser._get_world_size()
QUEUE_SIZE =parser._get_queue_size()
MAX_QUEUE_SIZE = parser._get_max_queue_size()
RPC_NAME = parser._get_RPC_NAME()

class MpCommand(Enum):
    """Enum class for multiprocessing command"""

    INIT_RPC = 0  # Not used in the task queue
    SET_COLLATE_FN = 1
    CALL_BARRIER = 2
    DELETE_COLLATE_FN = 3
    CALL_COLLATE_FN = 4
    CALL_FN_ALL_WORKERS = 5
    FINALIZE_POOL = 6
    CACHE_MEM = 7
    LOAD_CACHE_DATA = 8

start = time.time()
total_time = 0
sample_group = None
custom_pool = None
def get_sampler_pool():
    global custom_pool
    return custom_pool

def get_sample_group():
    global sample_group
    return sample_group
 
def wait_thread(waitfut,data_queue):
    global total_time
    global start
    while(True):
        if len(waitfut) > 0:
            result = waitfut[0]
            if result == -1:
                break
            future_to_check = result.get("union_args")[1]
            #print('check',time.time()-start)
            if(_check_future_finish(future_to_check) is True):
                start = time.time()
                batch_data = _get_batch_data(result)
                total_time = total_time+time.time()-result.get("append_time")
                tps.wait_fut = tps.wait_fut + start - result.get("append_time")
                tps.t_union = tps.t_union + time.time()-start
                tps.count_union = tps.count_union + 1
                #logger.debug('time wait for answer {}, time for get batch {},total_time {}'.format(start - result.get("append_time"),time.time()-start,total_time))
                #tps.print_()
                waitfut.popleft()
                if not isinstance(data_queue,deque):
                    data_queue.put(
                        (
                            result.get("data_name"),
                            batch_data,
                        )
                    
                    )
                else:
                    data_queue.append(
                        batch_data
                    )
custom_pool = None
rpc_server_prco = None
#rpc_server_queue = None
#rpc_server_barrier = None
keep_polling = True


total_t0 = 0
total_t1 = 0
total_t2 = 0
total_t3 = 0
total_t4 = 0
cnt = 0
def _get_from_wait_thread(waitfut,data_queue,is_temporal = False):
    
    while(len(waitfut) > 0):
        result = waitfut[0]
        future_to_check = result.get("union_args")[1]
            #print('check',time.time()-start)
        if(_check_future_finish(future_to_check) is not True):
            break
        waitfut.popleft()
            #start = time.time()
        if is_temporal:
            batch_data = _get_temporal_batch_data(result)
        else:
            batch_data = _get_batch_data(result)
            #total_time = total_time+time.time()-result.get("append_time")
            #tps.wait_fut = tps.wait_fut + start - result.get("append_time")
            #tps.t_union = tps.t_union + time.time()-start
            #tps.count_union = tps.count_union + 1
            #logger.debug('time wait for answer {}, time for get batch {},total_time {}'.format(start - result.get("append_time"),time.time()-start,total_time))
            #tps.print_()
        if not isinstance(data_queue,deque):
                
            data_queue.put(
                (
                    result.get("data_name"),
                    batch_data,
                )

            )
        else:
            data_queue.append(
                        batch_data
                    )
class LocalSampler:
    def __init__(self):
        self.waitfut = deque()
        self.data_queue = deque()
        #self.thread = Thread(target=wait_thread,args=(self.waitfut,self.data_queue))
        #self.thread.start()
    def set_collate_fn(self,dataloader_name,graph_inshm,iscached):
        self.dataloader_name = dataloader_name
        self.iscached = iscached
        graph_store._set_graph(dataloader_name,graph_inshm)
        global rpc_server_queue
        rpc_server_queue.put((RpcCommand.SET_GRAPH,(dataloader_name,graph_inshm)))
        rpc_server_queue.put((RpcCommand.CALL_BARRIER,tuple()))
        global rpc_server_barrier
        rpc_server_barrier.wait()
        self.results=[]
        dist.barrier()
    def sample_next(self,dataloader_name,sampler,neg_sampler,input_data):
        t1 = time.time()
        if input_data.nodes is not None:
            out = sampler.sample_from_nodes(nodes=input_data.nodes.reshape(-1),with_outer_sample=parser.SAMPLE_TYPE)
            #neighbor_nodes, sampled_edge_index,sampled_eid_list = out.node,out.edge_index_list,out.eid_list
            #sampler.sample_from_node(input_data.nodes.reshape(-1),input_data.ts.reshape(-1),parser.SAMPLE_TYPE)
            metadata = None
        elif input_data.edges is not None:
            out = sampler.sample_from_edges(edges = input_data.edges,with_outer_sample=parser.SAMPLE_TYPE,edge_label = input_data.labels,neg_sampling = neg_sampler) 
            #neighbor_nodes, sampled_edge_index,metadata
        t2 = time.time()
        union_args=_sample_node_neighbors_server(dataloader_name,input_data,out,self.iscached)
        t3 = time.time()    
        #logger.debug('sample {},get union_args {}'.format(t2-t1,t3-t2))
        start = time.time()
        result = {
           "data_name":dataloader_name,
           "input_data":input_data,
           "union_args":union_args,
           "sampled_out":out,
           "append_time":time.time()}
        
       # future_to_check = result.get("union_args")[1]
            #print('check',time.time()-start)
       # while(_check_future_finish(future_to_check) is not True):
       #     continue
            #start = time.time()
        #batch_data = _get_batch_data(result)
        #self.data_queue.append(
        #    batch_data
        #)
       #
        self.waitfut.append(result)
        
    def temporal_sample_next(self,graph_name,sampler,neg_sampler,input_data):
        assert hasattr(input_data,'ts') and input_data.ts is not None
        t1 = time.time()
        #调用采样函数
        if input_data.nodes is not None:
            #edge_index, edge_id, edge_ts 
            out =  sampler.sample_from_nodes(nodes=input_data.nodes.reshape(-1),ts=input_data.ts.reshape(-1),with_outer_sample=parser.SAMPLE_TYPE)
            metadata = None
        elif input_data.edges is not None:
            out = sampler.sample_from_edges(edges = input_data.edges,with_outer_sample=parser.SAMPLE_TYPE,ets=input_data.ts,edge_label=input_data.labels,neg_sampling = neg_sampler) 
            #edge_index,edge_id,edge_ts, metadata = 
            #edge_index, edge_id, edge_ts, meta_data =  sampler.sample_from_nodes_with_before(input_data[0].reshape(-1),input_data[1].reshape(-1),parser.SAMPLE_TYPE)
        t2 = time.time()
        union_args=_temporal_sample_node_neighbors_server(graph_name,input_data,out,self.iscached)
        t3 = time.time()    
        start = time.time()
        result = {
           "data_name":graph_name,
           "input_data": input_data,
           "union_args":union_args,
           "sampled_out": out,
           "append_time":time.time(),
        }
        
        self.waitfut.append(result)
    
    def get_result(self):
        _get_from_wait_thread(self.waitfut,self.data_queue,is_temporal = True)   
        if(len(self.data_queue)!=0):
            result = self.data_queue[0]
            self.data_queue.popleft()
            return result
        else:
            return None
        
    def __del__(self):
        dist.barrier()
        global rpc_server_queue
        rpc_server_queue.put(
            (RpcCommand.UNLOAD_GRAPH, (self.dataloader_name,))
        )
        global rpc_server_barrier
        if(keep_polling is True):
            rpc_server_barrier.wait()
        graph_inshm = graph_store._get_graph(self.dataloader_name)
        graph_store._del_graph(self.dataloader_name)
        graph_inshm._close_graph_in_shame()
        graph_inshm._unlink_graph_in_shame()
        self.waitfut.append(-1)
        
def init_process(sampler_id,rpc_config,comm_config):
    """start the proxy of rpc on the process """
    rpc_master_addr,rpc_port,num_worker_threads = rpc_config
    start_rpc_caller(rpc_master_addr,
                    rpc_port,
                    {
                        "num_worker_threads": num_worker_threads,
                        "rpc_name": RPC_NAME,
                        "rpc_world_size": WORLD_SIZE * (NUM_SAMPLER + 1),
                        "worker_rank":  WORKER_RANK,
                        "rpc_worker_rank": (NUM_SAMPLER + 1) * WORKER_RANK + sampler_id + 1}
                    )
    print('start work')
    global start
    iscached = False
    try:
        
        
        
        data_queue,task_queue,barrier = comm_config
        collate_fn_dict = {}
        sampler_dict = {}
        keep_poll = True
        waitfut = deque()
        thread = Thread(target=wait_thread,args=(waitfut,data_queue))
        thread.start()
        while keep_poll or graph_store._get_size() > 0:
            if(len(waitfut)>QUEUE_SIZE):
                continue
            try:
                command,args = task_queue.get(timeout=5)
            except queue.Empty:
                continue
            print(command)
            if command == MpCommand.SET_COLLATE_FN:
                dataloader_name,graph_inshm,func,sampler_info = args
                graph_store._set_graph(dataloader_name,graph_inshm)
                collate_fn_dict[dataloader_name] = func
                #neighbors,deg = graph_inshm._get_sampler_from_shame()
                #sampler = sampler_info #NeighborSampler(*sampler_info,neighbors= neighbors,deg = deg)
                graph = graph_store._get_graph(dataloader_name)
                #row, col = graph.edge_index
                #tnb = get_neighbors(row.contiguous(), col.contiguous(), graph.num_nodes)
                sampler = NeighborSampler(*sampler_info,graph_data= graph)
                sampler_dict[dataloader_name] = sampler

            elif command == MpCommand.CALL_BARRIER:
                barrier.wait()

            elif command == MpCommand.DELETE_COLLATE_FN:
                (dataloader_name,) = args
                del collate_fn_dict[dataloader_name]
                del sampler_dict[dataloader_name]
                graph_inshm = graph_store._get_graph(dataloader_name)
                graph_store._del_graph(dataloader_name)
                graph_inshm._close_graph_in_shame()
                if(barrier.wait()==0):
                    graph_inshm._unlink_graph_in_shame()

            elif command == MpCommand.CALL_COLLATE_FN:
                dataloader_name, collate_args, start_time = args
                t1 = time.time()
                #out = sampler_dict[dataloader_name].sample_from_nodes((dataloader_name,collate_args,time.time()))
                #neighbor_nodes = out.node
                #sampled_edge_index = torch.cat((out.row.reshape(1,-1),out.col.reshape(1,-1)),0)
                neighbor_nodes, sampled_edge_index= sampler_dict[dataloader_name].sample_from_nodes(collate_args)
                t2 = time.time()
                union_args=_sample_node_neighbors_server(dataloader_name,neighbor_nodes,iscached)
                t3 = time.time()
                start = time.time()
                #logger.debug('wait for input {},sample {},get union_args {}'.format(t1-start_time,t2-t1,t3-t2))          
                tps.wait_input = tps.wait_input + t1-start_time
                tps.t_sample = tps.t_sample + t2-t1
                tps.t_get = tps.t_get + t3-t2
                tps.count_sample = tps.count_sample + 1
                tps.count_union = tps.count_union + 1
                #tps.print_()
                result = {
                    "data_name":dataloader_name,
                    "input_size":collate_args.size(0),
                    "nids":neighbor_nodes,
                    "union_args":union_args,
                    "edge_index":sampled_edge_index,
                    "append_time":time.time()}
                future_to_check = result.get("union_args")[1]
            #print('check',time.time()-start)
                while(_check_future_finish(future_to_check) is not True):
                    continue
            #start = time.time()
                batch_data = _get_batch_data(result)
                data_queue.put(
                    (
                    result.get("data_name"),
                    batch_data,
                    )
                 )
                #data_queue.put(
                #    (
                #        dataloader_name,
                #        collate_fn_dict[dataloader_name](collate_args),
                #    )
                #)
            elif command == MpCommand.CALL_FN_ALL_WORKERS:
                func, func_args = args
                func(func_args)
            elif command == MpCommand.FINALIZE_POOL:
                keep_poll = False 
                graph_store._clear_all(barrier)
                close_rpc()
            elif command == MpCommand.CACHE_MEM:
                data_name,index_data,mem_data = args
                load_cpu_cache_memory(data_name,index_data,mem_data)
                iscached = True
                barrier.wait()
            elif command == MpCommand.LOAD_CACHE_DATA:
                data_name,index = args
                load_data2cache(data_name,index)
            else:
                raise Exception("Unknown command")
            
    except Exception as e:
        traceback.print_exc()
        raise e
    

class CustomPool:
    def __init__(self,rpc_config):
        self.max_queue_size = MAX_QUEUE_SIZE
        self.num_samplers = NUM_SAMPLER
        ctx = mp.get_context("spawn")
        self.result_queue = ctx.Queue(self.max_queue_size)
        self.results = {}
        self.task_queues = []
        self.process_list = []
        self.current_proc_id = 0
        self.cache_result_dict = {}
        self.barrier = ctx.Barrier(self.num_samplers)
        for rank in range(self.num_samplers):
            task_queue = ctx.Queue(self.max_queue_size)
            self.task_queues.append(task_queue)
            proc = ctx.Process(
                target=init_process,
                args=(
                    rank,rpc_config,
                    (self.result_queue,task_queue,self.barrier),
                )
                    
            )
            proc.daemon=True
            proc.start()
            self.process_list.append(proc)
        self.call_barrier()

    def load_share_cache(self,data_name,share_index_data,share_cache_data):
        for i in range(self.num_samplers):
            self.task_queues[i].put(
                (MpCommand.CACHE_MEM,(data_name,share_index_data,share_cache_data))
            )
    def initilize_cache(self,data_name,index):
        self.task_queues[0].put(
                (MpCommand.LOAD_CACHE_DATA,(data_name,index))
            ) 
        self.call_barrier()
    def set_collate_fn(self,func,sampler,dataloader_name,graph_inshm):
        for i in range(self.num_samplers):
            self.task_queues[i].put(
                (MpCommand.SET_COLLATE_FN,(dataloader_name,graph_inshm,func,sampler))
            )
        global rpc_server_queue
        rpc_server_queue.put((RpcCommand.SET_GRAPH,(dataloader_name,graph_inshm)))
        rpc_server_queue.put((RpcCommand.CALL_BARRIER,tuple()))
        global rpc_server_barrier
        rpc_server_barrier.wait()
        self.call_barrier()
        self.results[dataloader_name]=[]

    def submit_task(self, dataloader_name, args):
        """Submit task to workers"""
        # Round robin
        self.task_queues[self.current_proc_id].put(
            (MpCommand.CALL_COLLATE_FN, (dataloader_name, args,time.time()))
        )
        self.current_proc_id = (self.current_proc_id + 1) % self.num_samplers
    
    def submit_temporal_task(self, dataloader_name, args):
        """Submit task to workers"""
        # Round robin
        self.task_queues[self.current_proc_id].put(
            (MpCommand.CALL_COLLATE_FN, (dataloader_name, args,time.time()))
        )
        self.current_proc_id = (self.current_proc_id + 1) % self.num_samplers
        
        
    def submit_task_to_all_workers(self, func, args):
        """Submit task to all workers"""
        for i in range(self.num_samplers):
            self.task_queues[i].put(
                (MpCommand.CALL_FN_ALL_WORKERS, (func, args))
            )
    def get_result(self, dataloader_name, timeout=1800):
        """Get result from result queue"""
        if dataloader_name not in self.results:
            raise Exception(
                f"Got result from an unknown dataloader {dataloader_name}."
            )
        while len(self.results[dataloader_name]) == 0:
            data_name, data = self.result_queue.get(timeout=timeout)
            self.results[data_name].append(data)
        return self.results[dataloader_name].pop(0)

    def delete_collate_fn(self, dataloader_name):
        """Delete collate function"""
        global rpc_server_queue
        rpc_server_queue.put(
            (RpcCommand.UNLOAD_GRAPH, (dataloader_name,))
        )
        global rpc_server_barrier
        if(keep_polling is True):
            rpc_server_barrier.wait()
        for i in range(self.num_samplers):
            self.task_queues[i].put(
                (MpCommand.DELETE_COLLATE_FN, (dataloader_name,))
            )
        if dataloader_name in self.results:
            del self.results[dataloader_name] 
    
    def call_barrier(self):
        """Call barrier at all workers"""
        for i in range(self.num_samplers):
            self.task_queues[i].put((MpCommand.CALL_BARRIER, tuple()))

    def close(self):
        """Close worker pool"""
        for i in range(self.num_samplers):
            self.task_queues[i].put(
                (MpCommand.FINALIZE_POOL, tuple()), block=False
            )
        time.sleep(0.5)  # Fix for early python version

    def join(self):
        """Join the close process of worker pool"""
        for i in range(self.num_samplers):
            self.process_list[i].join()
    


def init_distribution(master_addr=None,master_port=None,rpc_master_addr=None,rpc_port=None, num_worker_threads = 2,backend = "gloo"):
    print('init distribution')
    global sample_group 
    global custom_pool
    world_size = parser._get_world_size()
    worker_rank = parser._get_worker_rank()
    num_sampler = parser._get_num_sampler()
    init_method="tcp://{}:{}".format(master_addr,master_port)
    sample_group = dist.init_process_group(backend=backend, world_size=world_size, rank=worker_rank,init_method=init_method,group_name='sample-default-group')
        
    if world_size > 1:
        if master_addr is None or master_port is None:
            raise Exception(
                    f"The master address is unknown."
                )
        if rpc_port is None:
            raise Exception(
                    f"The rpc listener address is unknown."
                )
        
        ctx = mp.get_context("spawn")
        global rpc_server_queue
        rpc_server_queue = ctx.Queue(MAX_QUEUE_SIZE)
        global rpc_server_barrier
        rpc_server_barrier = ctx.Barrier(2)
        rpc_server_proc= ctx.Process(target = start_rpc_listener,
                                     args=(rpc_master_addr,
                                           rpc_port,
                                           {"num_worker_threads": num_worker_threads,
                                            "rpc_name": RPC_NAME,
                                            "rpc_world_size": world_size * (num_sampler + 1),
                                            "worker_rank":  worker_rank,
                                            "rpc_worker_rank": (num_sampler + 1) * worker_rank
                                            },
                                            (rpc_server_queue,rpc_server_barrier)),
                                     )
        rpc_server_proc.start()
        if parser._get_num_sampler()>1:
            custom_pool = CustomPool((rpc_master_addr,rpc_port,num_worker_threads))
        else:
            start_rpc_caller(rpc_master_addr,
                rpc_port,
                {
                    "num_worker_threads": num_worker_threads,
                    "rpc_name": RPC_NAME,
                    "rpc_world_size": WORLD_SIZE * (NUM_SAMPLER + 1),
                    "worker_rank":  WORKER_RANK,
                    "rpc_worker_rank": (NUM_SAMPLER + 1) * WORKER_RANK  + 1}
                )
        dist.barrier()
    else:
        custom_pool = None
    
def close_distribution():
    global custom_pool
    global rpc_server_queue
    global rpc_server_barrier
    global keep_polling 
    if parser._get_world_size()>1 and custom_pool is not None:
        dist.barrier()
        print('del')
        rpc_server_queue.put((RpcCommand.STOP_RPC,tuple()))
        rpc_server_barrier.wait()
        custom_pool.close()
        keep_polling = False
    elif parser._get_world_size()>1 and custom_pool is None:
        dist.barrier()
        rpc_server_queue.put((RpcCommand.STOP_RPC,tuple()))
        rpc_server_barrier.wait()
        keep_polling = False
        close_rpc()
    #rpc.init_rpc(OBSERVER_NAME.format(rank),rank=rank,world_size=world_size)
    #if(rank==0):
    #    message_worker.init_sampler_worker()
    #dist.barrier()
    #print(OBSERVER_NAME.format(rank))
    #worker = sampler_worker(world_size,OBSERVER_NAME)
    #if(rank==0):
    #rrefs = 
    #master_worker = sampler_worker(world_size,OBSERVER_NAME)
