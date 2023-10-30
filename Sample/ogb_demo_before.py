import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import datasets
import time
from Utils import GraphData

def load_ogb_dataset(name, data_path):
    dataset = PygNodePropPredDataset(name=name, root=data_path)
    split_idx = dataset.get_idx_split()
    g = dataset[0]
    n_node = g.num_nodes
    node_data={}
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g, node_data

g, node_data = load_ogb_dataset('ogbn-products', "/home/hzq/code/gnn/dataset/")
print(g)
# for worker in [1,2,3,4,5,6,7,8,9,10,20,30]:
# import random
# timestamp = [random.randint(1, 5) for i in range(0, g.num_edges)]
# timestamp = torch.FloatTensor(timestamp)

print('begin load')
pre = time.time()
timestamp = torch.load('/home/hzq/code/gnn/my_sampler/TemporalSample/timestamp.my')
tnb = torch.load("tnb_before.my")
end = time.time()
print("load time:", end-pre)
row, col = g.edge_index
edge_weight=None
g_data = GraphData(id=1, edge_index=g.edge_index, timestamp=timestamp, data=g, partptr=torch.tensor([0, g.num_nodes//4, g.num_nodes//4*2, g.num_nodes//4*3, g.num_nodes]))

from neighbor_sampler import NeighborSampler, SampleType, get_neighbors

# print('begin tnb')
# pre = time.time()
# tnb = get_neighbors(row.contiguous(), col.contiguous(), g.num_nodes, 0, g_data.eid, edge_weight, timestamp)
# end = time.time()
# print("init tnb time:", end-pre)
# torch.save(tnb, "tnb_before.my")


pre = time.time()
sampler = NeighborSampler(g.num_nodes, 
                          tnb=tnb,
                          num_layers=2, 
                          fanout=[100,100], 
                          graph_data=g_data, 
                          workers=10, 
                          policy="uniform", 
                          is_root_ts=0, 
                          graph_name='a')
end = time.time()
print("init time:", end-pre)

# from torch_geometric.sampler import NeighborSampler, NumNeighbors, NodeSamplerInput, SamplerOutput
# pre = time.time()
# num_nei = NumNeighbors([100, 100])
# node_idx = NodeSamplerInput(input_id=None, node=torch.tensor(range(g.num_nodes//4, g.num_nodes//4+600000)))# (input_id=None, node=torch.masked_select(torch.arange(g.num_nodes),node_data['train_mask']))
# sampler = NeighborSampler(g, num_nei)
# end = time.time()
# print("init time:", end-pre)

ts = torch.tensor([i%5+1 for i in range(0, 600000)])
pre = time.time()
out = sampler.sample_from_nodes(torch.tensor(range(g.num_nodes//4, g.num_nodes//4+600000)), 
                                ts=ts, 
                                with_outer_sample=SampleType.Inner)# sampler.sample_from_nodes(torch.masked_select(torch.arange(g.num_nodes),node_data['train_mask']))

# out = sampler.sample_from_nodes(node_idx)
# node = out.node
# edge = [out.row, out.col]

end = time.time()
print('node:', out.node)
print('edge_index_list:', out.edge_index_list)
print('eid_list:', out.eid_list)
print('eid_ts_list:', out.eid_ts_list)
print("sample time", end-pre)