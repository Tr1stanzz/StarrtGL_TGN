from Cache.cache_memory import create_empty_cache, load_data2cache
from DistCustomPool import get_sampler_pool
import distparser as parser
from message_worker import _split_node_part_and_submit
from share_memory_util import _copy_to_share_memory, _get_existing_share_memory, _get_from_share_memory
import torch

def create_memory(data_name,graph,memory_size,index,device='cpu'):
   if device == 'cpu':
      if parser._get_num_sampler()== 1: 
        create_empty_cache(data_name,len(index),graph.data.x.size(1))
        load_data2cache(data_name,index)
               
      else:
         cache_mem = torch.zeros(len(index),graph.data.x.size(1))
         indx = torch.zeros(len(index))
         mem_data = _copy_to_share_memory(cache_mem)
         idx_data = _copy_to_share_memory(index)
         pool = get_sampler_pool()
         pool.load_share_cache(data_name,idx_data,mem_data)
         pool.initilize_cache(data_name,index)

