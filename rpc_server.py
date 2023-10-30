from enum import Enum
from multiprocessing.connection import Listener
import pickle
import queue
from threading import Thread
from multiprocessing import Pool, Lock, Value
import torch
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.distributed import rpc
import graph_store
import distparser

class RpcCommand(Enum):
    SET_GRAPH = 0
    CALL_BARRIER = 1
    UNLOAD_GRAPH =2
    STOP_RPC = 3
"""
class RPCHandler:
    def __init__(self):
        self._functions= { }
    def register_function(self,func):
        self._functions[func.__name__] = func
    def handle_connection(self,connection):
        try:
            while True:
                func_name,args,kwargs  = pickle.loads(connection.recv())
                try:
                    r=self._functions[func_name] (*args,**kwargs)
                    connection.send(pickle.dumps(r))
                except Exception as e:
                    connection.send(pickle.dumps(e))
        except EOFError:
            pass
def start_rpc_server(handler,address,queue):
        sock = Listener(address)
        queue.put(RpcCommand.FINISH_LISTEN)
        while True:
            client = sock.accept()
            now_thread=Thread(target = handler.handle_connection, args=(client,))
            now_thread.daemon = True
            now_thread.start()

"""
NUM_WORKER_THREADS = 128
def start_rpc_listener(rpc_master,rpc_port,rpc_config,mp_config):
    print(rpc_config)
    num_worker_threads = rpc_config.get("num_worker_threads", NUM_WORKER_THREADS)
    rpc_name = rpc_config.get("rpc_name", "rpcserver{}")
    world_size = rpc_config.get("rpc_world_size", 2)
    worker_rank = rpc_config.get("worker_rank", 0)
    rpc_rank = rpc_config.get("rpc_worker_rank", 0)
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://{}:{}".format(rpc_master,rpc_port)
    rpc_backend_options.num_worker_threads = num_worker_threads
    task_queue,barrier = mp_config
    print(rpc_master,rpc_port)

    rpc.init_rpc(
            rpc_name.format(worker_rank),
            rank=rpc_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    keep_pooling = True
    while(keep_pooling):
        try:
            command,args = task_queue.get(timeout=5)
        except queue.Empty:
            continue
        if command == RpcCommand.SET_GRAPH:
            dataloader_name,graph_inshm= args

            graph_store._set_graph(dataloader_name,graph_inshm)
        elif command == RpcCommand.UNLOAD_GRAPH:
            dataloader_name= args
            graph_inshm = graph_store._get_graph(dataloader_name)
            graph_store._del_graph(dataloader_name)
            graph_inshm._close_graph_in_shame()
            print('unload dataloader')
            barrier.wait()
        elif command == RpcCommand.CALL_BARRIER:
            barrier.wait()
        elif command == RpcCommand.STOP_RPC:
            keep_pooling = False
            graph_store._clear_all()
            barrier.wait()
            close_rpc()


def start_rpc_caller(rpc_master,rpc_port,rpc_config):
    print(rpc_config)
    num_worker_threads = rpc_config.get("num_worker_threads",NUM_WORKER_THREADS)
    rpc_name = rpc_config.get("rpc_name","rpcserver{}")
    world_size = rpc_config.get("rpc_world_size", 2)
    worker_rank = rpc_config.get("worker_rank", 0)
    rpc_rank = rpc_config.get("rpc_worker_rank", 0)
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://{}:{}".format(rpc_master,rpc_port)
    rpc_backend_options.num_worker_threads = num_worker_threads
    num_sampler = distparser._get_num_sampler()
    for i in range(distparser._get_world_size()):
        if i == worker_rank:
            continue
        rpc_backend_options.set_device_map(rpc_name.format(i) ,{worker_rank:i})
        print(rpc_name.format(i) + "-{}".format( (num_sampler+1) *i + 1),{worker_rank:i})
        rpc_backend_options.set_device_map(rpc_name.format(i) + "-{}".format( (num_sampler+1) *i + 1),{worker_rank:i})
    rpc.init_rpc(
            rpc_name.format(worker_rank) + "-{}".format(rpc_rank),
            rank=rpc_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )


def close_rpc():
    rpc.shutdown(True)
"""
class RPCProxy:
    def __init__(self, connection):
        self._connection = connection
        
    def __getattr__(self, name):
        def do_rpc(*args, **kwargs):
            self._connection.send(pickle.dumps((name, args, kwargs)))
            result = pickle.loads(self._connection.recv())
            if isinstance(result, Exception):
                raise result
            return result
        return do_rpc   
"""