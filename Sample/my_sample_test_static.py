import torch
import time
from Utils import GraphData
seed = 10  # 你可以选择任何整数作为种子
torch.manual_seed(seed)

num_nodes1 = 10
fanout1 = [2,2]      # index 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
edge_index1 = torch.tensor([[1, 5, 7, 9, 2, 4, 6, 7, 8, 0, 1, 6, 2, 0, 1, 3, 5, 8, 9, 7, 4, 8, 2, 3, 5, 8],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 6, 6, 7, 7, 8, 9]])
edge_weight1 = torch.tensor([2, 1, 2, 1, 8, 6, 3, 1, 1, 1, 1, 5, 1, 1, 2, 1, 1, 1, 1, 5, 1, 2, 2, 2, 1, 1]).double()

g_data = GraphData(id=0, edge_index=edge_index1, data=None, partptr=torch.tensor([0, num_nodes1]))

edge_weight1 = None
# g_data.eid=None

from neighbor_sampler import NeighborSampler, SampleType
pre = time.time()
sampler = NeighborSampler(num_nodes1, 
                          num_layers=2, 
                          fanout=fanout1, 
                          edge_weight=edge_weight1,
                          graph_data=g_data, 
                          workers=2, 
                          graph_name='a',
                          policy="weighted")
end = time.time()
print("init time:", end-pre)

print("tnb.neighbors:", sampler.tnb.neighbors)
print("tnb.eid:", sampler.tnb.eid)
print("tnb.deg:", sampler.tnb.deg)
print("tnb.weight:", sampler.tnb.edge_weight)

# row,col = edge_index1
# update_edge_row =     row
# update_edge_col =     col
# update_edge_w =  torch.FloatTensor([i for i in range(edge_weight1.size(0))])
# print('tnb.edge_weight:', sampler.tnb.edge_weight)
# print('begin update')
# pre = time.time()
# sampler.tnb.update_edge_weight(sampler.tnb, update_edge_row.contiguous(), update_edge_col.contiguous(), update_edge_w.contiguous())
# end = time.time()
# print("update time:", end-pre)
# print('update_edge_row:', update_edge_row)
# print('update_edge_col:', update_edge_col)
# print('tnb.edge_weight:', sampler.tnb.edge_weight)


pre = time.time()

out = sampler.sample_from_nodes(torch.tensor([1,2]), 
                                with_outer_sample=SampleType.Whole)# sampler.sample_from_nodes(torch.masked_select(torch.arange(g.num_nodes),node_data['train_mask']))

end = time.time()
print('node:', out.node)
print('edge_index_list:', out.edge_index_list)
print('eid_list:', out.eid_list)
print('eid_ts_list:', out.eid_ts_list)
print("sample time:", end-pre)