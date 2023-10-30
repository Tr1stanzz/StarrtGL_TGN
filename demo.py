import torch
from Sample.neighbor_sampler import NeighborSampler, get_neighbors
# num_nodes = 4
edge_index = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5], [1, 0, 2, 4, 1, 3, 0, 2, 5, 3, 5, 0, 2]])
num_nodes = 6
num_neighbors = 2
# Run the neighbor sampling
tnb = get_neighbors(edge_index[0], edge_index[1], num_nodes)
sampler=NeighborSampler(tnb=tnb, num_nodes=num_nodes, num_layers=2, workers=2, fanout=[2, 1])

# neighbor_nodes, sampled_edge_index = sampler._sample_one_layer_from_nodes(nodes=torch.tensor([1,3]), fanout=num_neighbors)
neighbor_nodes, sampled_edge_index = sampler.sample_from_nodes(torch.tensor([1,2,3]))
    
# Print the result
print('neighbor_nodes_id: \n',neighbor_nodes, '\nedge_index: \n',sampled_edge_index)

# import torch_scatter
# nodes=torch.Tensor([1,2])
# row, col = edge_index
# deg = torch_scatter.scatter_add(torch.ones_like(row), row, dim=0, dim_size=num_nodes)
# neighbors1=torch.concat([row[row==nodes[i]] for i in range(0, nodes.shape[0])])
# print(neighbors1)
# neighbors2=torch.concat([col[row==nodes[i]] for i in range(0, nodes.shape[0])])
# print(neighbors2)
# neighbors=torch.stack([neighbors1, neighbors2], dim=0)
# print('neighbors: \n', neighbors[0]==1)

from Sample.base import NegativeSampling
edge_index = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5], [1, 0, 2, 4, 1, 3, 0, 2, 5, 3, 5, 0, 2]])
num_nodes = 6
# sampler
sampler=NeighborSampler(edge_index=edge_index, num_nodes=num_nodes, num_layers=2, workers=2, fanout=[2, 1])

# negative
weight = torch.tensor([0.3,0.1,0.1,0.1,0.3,0.1])
negative = NegativeSampling('binary', 2, weight)
# negative = NegativeSampling('triplet', 2, weight)
label=torch.tensor([1,2])
seed_edges = torch.tensor([[0,1],
                           [1,4]])
# result = sampler.sample_from_edges(edges=seed_edges)
result = sampler.sample_from_edges(edges=seed_edges, edge_label=label, neg_sampling=negative)
# Print the result
print('neighbor_nodes_id: \n',result[0], '\nedge_index: \n',result[1], '\nmetadata: \n',result[2])