from mimetypes import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from part.Utils import GraphData
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from BatchData import BatchData

class MyModel(torch.nn.Module):
    def __init__(self,graph:GraphData):
        super(MyModel, self).__init__()
        self.conv1 = GCNConv(graph.num_feasures, 128)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(128, 7)
        self.num_layers = 2
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean'):
        super(SAGEConv, self).__init__(aggr=aggr) # 使用mean聚合方式
        # 线性层
        self.w1 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        self.w2 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
    
    def message(self, x_j):
        # x_j [E, in_channels]
        
        # 将邻居特征进行特征映射
        wh = self.w2(x_j) # [E, out_channels]
        
        return wh
    
    def update(self, aggr_out, x):
        # aggr_out [num_nodes, out_channels]
        
        # 对自身节点进行特征映射
        wh = self.w1(x)
        
        return aggr_out + wh
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        

# 3.定义GraphSAGE网络
class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE, self).__init__()
        # self.gcns = []
        # self.gcns.append(SAGEConv(in_channels=num_node_features,out_channels=16))
        # self.gcns.append(SAGEConv(in_channels=16, out_channels=num_classes))
        
        self.conv1 = SAGEConv(in_channels=num_node_features,
                              out_channels=256)
        self.conv2 = SAGEConv(in_channels=256,
                              out_channels=num_classes)
        
    def forward(self, data:BatchData,type):
        #print(data.data.train_mask)
        ##nids = torch.nonzero(data.data.train_mask).reshape(-1) 
        #x=data.data.x
        #edge_indexs = data.edge_index
        nids,x, edge_indexs = data.nids,data.x, data.edge_index
        nids_dict = dict(zip(nids.tolist(), torch.arange(nids.numel()).tolist()))
        #edge_index_local = [] 
        #edge_index_local.append(torch.tensor([
        #    [nids_dict[int(s.item())] for s in edge_indexs[0][0]], 
        #    [nids_dict[int(s.item())] for s in edge_indexs[0][1]]
        #    ]) )
        #edge_index_local.append(torch.tensor([
        #    [nids_dict[int(s.item())] for s in edge_indexs[1][0]], 
        #    [nids_dict[int(s.item())] for s in edge_indexs[1][1]]
        #    ]) )
        x = self.conv1(x, edge_indexs[1])
        x = F.relu(x)
        if(type == 0):
            x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_indexs[0])
        print(f'x sizes {x.size()}')
        #root_nids = nids
        return F.log_softmax(x[data.roots.nodes], dim=1)
    
    
    def inference(self, data):
        nids,x, edge_indexs = data.nids,data.x, data.edge_index
        # x_global = [x[nids[i]] for i in range(x.shape[0])]
        nids_dict = dict(zip(nids.tolist(), torch.arange(nids.numel()).tolist()))
        edge_index_local = torch.tensor([
            [nids_dict[int(s.item())] for s in edge_indexs[0][0]], 
            [nids_dict[int(s.item())] for s in edge_indexs[0][1]]])
        x = self.conv1(x, edge_index_local)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index_local)
        print(f'x sizes {x.size()}')
        root_nids = [nids_dict[root.item()] for root in data.roots]
        return F.log_softmax(x[root_nids], dim=1)

        
        


