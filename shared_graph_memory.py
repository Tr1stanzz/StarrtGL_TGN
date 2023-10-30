import torch
from part.Utils import GraphData
from share_memory_util import _close_existing_shm, _copy_to_share_memory, _copy_to_shareable_list, _get_existing_share_memory, _get_from_share_memory, _get_from_shareable_list, _unlink_existing_shm
from Sample.neighbor_sampler import NeighborSampler
from torch_geometric.data import Data
class GraphInfoInShm(GraphData):
        def __init__(self,graph):
  
            self.partition_id = graph.partition_id
            self.partitions = graph.partitions
            self.num_nodes = graph.num_nodes
            self.num_edges = graph.num_edges
            #self.edge_index_info = _copy_to_share_memory(graph.edge_index)
            self.partptr = graph.partptr
            self.edgeptr = graph.edgeptr
            
            self.data_x_info=_copy_to_share_memory(graph.data.x) if graph.data.x is not None else None
            self.edge_index_info=_copy_to_share_memory(graph.edge_index)
            self.edge_attr_info=_copy_to_share_memory(graph.data.edge_attr) if graph.data.edge_attr is not None else None
            self.edge_ts_info=_copy_to_share_memory(graph.edge_ts) if graph.edge_ts is not None else None
            
            
            
            
        def _get_graph_from_shm(self):
            #self.edge_index_shm = _get_existing_share_memory(self.edge_index_info[0])
            #self.edge_index = _get_existing_share_memory(self.edge_index_shm,self.edge_index_info[1:])
            self.data = Data()
            if self.data_x_info is not None:
                self.data_x_shm = _get_existing_share_memory(self.data_x_info[0])
                self.data.x = _get_from_share_memory(self.data_x_shm,*self.data_x_info[1:])
            if self.edge_index_info is not None:
                self.edge_index_shm = _get_existing_share_memory(self.edge_index_info[0])
                self.edge_index = _get_from_share_memory(self.edge_index_shm,*self.edge_index_info[1:])
            if self.edge_attr_info is not None:
                self.edge_attr_shm = _get_existing_share_memory(self.edge_attr_info[0])
                self.data.edge_attr = _get_from_share_memory(self.edge_attr_shm,*self.edge_attr_info[1:])
            if self.edge_ts_info is not None:
                self.edge_ts_shm = _get_existing_share_memory(self.edge_ts_info[0])
                self.edge_ts = _get_from_share_memory(self.edge_ts_shm,*self.edge_ts_info[1:])

        def _get_sampler_from_shame(self):
            self.deg_shm = _get_from_shareable_list(*self.deg_info)#_get_existing_share_memory(self.deg_info)
            self.neighbors_shm = _get_from_shareable_list(*self.neighbors_info)
            self.deg = self.deg_shm
            self.neighbors = self.neighbors[0]

        def _copy_sampler_to_shame(self,neighbors,deg):
            self.deg_info = _copy_to_shareable_list(deg)#_copy_to_share_memory(deg)
            self.neighbors_info = _copy_to_shareable_list(neighbors)#_copy_to_share_memory(neighbors)
        
        def _close_graph_in_shame(self):
            #_close_existing_shm(self.edge_index_shm)
            #_close_existing_shm(self.deg_shm.shm)
            #_close_existing_shm(self.neighbors_shm.shm)
            _close_existing_shm(self.data_x_shm)
            #_close_existing_shm(self.data_y_shm)
            _close_existing_shm(self.edge_index_shm)
            _close_existing_shm(self.edge_attr_shm)
            _close_existing_shm(self.edge_ts_shm)

        def _unlink_graph_in_shame(self):
            #_unlink_existing_shm(self.edge_index_shm)
            _unlink_existing_shm(self.data_x_shm)
            #_unlink_existing_shm(self.data_y_shm)
            _unlink_existing_shm(self.edge_index_shm)
            _unlink_existing_shm(self.edge_attr_shm)
            _unlink_existing_shm(self.edge_ts_shm)
            


    #返回全局的节点id 所对应的分区

    
    
            
        