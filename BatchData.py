from distparser import WORLD_SIZE

import torch
class BatchData:
    '''
         Args:
            batch_size: the number of sampled node
            nids: the real id of sampled nodes and their neighbours 
            edge_index: the edge between nids, the nodes' id is the index of BatchData
            x: nodes' feature
            y: 
            eids: the edge id in subgraph of edge_index
            train_mask
            val_mask
            test_mask
    '''
    def __init__(self,nids=None,edge_index=None,roots=None,root_ts=None,x=None,y=None,eids=None,edge_attr=None,node_ts = None,edge_ts=None,meta_data = None):
        self.roots=roots
        if(nids is not None):
            self.nids=nids
        if node_ts is not None:
            self.node_ts =node_ts
            
        self.edge_index=edge_index
        self.x=x
        self.y=y
        if(eids is not None):
            self.eids=eids
        if(root_ts is not None):
            self.root_ts = root_ts
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if edge_ts is not None:
            self.edge_ts = edge_ts
        self.meta_data =meta_data

    def _check_with_graph(self,graph1,graph2):
        for i,id in enumerate(self.nids):
            if(id >= graph1.partptr[0] and id < graph1.partptr[1]):
                real_x = graph1.select_attr(graph1.get_localId_by_partitionId(0,torch.tensor(id)))
            else:
                real_x = graph2.select_attr(graph2.get_localId_by_partitionId(1,torch.tensor(id)))

    def __repr__(self):
        return "BatchData(batch_size = {},roots = {} , \
            nides = {} , edge_index = {} , x= {}, \
                y ={})".format(self.roots,self.nids,self.edge_index,self.x,self.eids,self.edge_attr,self.edge_ts,self.meta_data)

    def to(self,device = 'cpu'):
        if device == 'cpu':
            return
        else:
            self.y.to(device)
            self.edge_index.to(device)
            self.roots.to(device)
            self.nids.to(device)
            self.eids.to(device)
