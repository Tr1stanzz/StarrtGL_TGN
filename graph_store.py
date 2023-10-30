import sys
from os.path import abspath, join, dirname
import time

sys.path.insert(0, join(abspath(dirname(__file__))))
graph_set={}

def _clear_all(barrier = None):
    global graph_set
    for key in graph_set:
        graph = graph_set[key]
        graph._close_graph_in_shame()
        print('clear ',key)
        if(barrier is not None and barrier.wait()==0):
            graph._unlink_graph_in_shame()
    graph_set = {}

def _set_graph(graph_name,graph_info):
    graph_info._get_graph_from_shm()
    graph_set[graph_name]=graph_info

def _get_graph(graph_name):
    return graph_set[graph_name]

def _del_graph(graph_name):
    graph_set.pop(graph_name)

def _get_size():
    return len(graph_set)


# local_sampler=None
local_sampler = {}
def set_local_sampler(graph_name,sampler):
    local_sampler[graph_name] = sampler

def get_local_sampler(sampler_name):
    
    assert sampler_name in local_sampler, 'Local_sampler doesn\'t has sampler_name'
    return local_sampler[sampler_name]