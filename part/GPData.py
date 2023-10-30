import copy
import os.path as osp
import sys
from typing import Optional

import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat
import numpy as np

# def metis(edge_index: LongTensor, num_nodes: int, num_parts: int) -> Tuple[LongTensor, LongTensor]:
#     if num_parts <= 1:
#         return _nopart(edge_index, num_nodes)
#     adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
#     rowptr, col, _= adj_t.csr()
#     node_parts = torch.ops.torch_sparse.partition(rowptr, col, None, num_parts, num_parts < 8)
#     edge_parts = node_parts[edge_index[1]]
#     return node_parts, edge_parts
class GPDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_parts: int, recursive: bool = False,
                 save_dir: Optional[str] = None, log: bool = True):

        assert data.edge_index is not None

        self.num_parts = num_parts

        recursive_str = '_recursive' if recursive else ''
        filename = f'partition_{num_parts}{recursive_str}.pt'
        path = osp.join(save_dir or '', filename)
        if save_dir is not None and osp.exists(path):
            adj, partptr, perm = torch.load(path)
        else:
            if log:  # pragma: no cover
                print('Computing METIS partitioning...', file=sys.stderr)

            N, E = data.num_nodes, data.num_edges
            adj = SparseTensor(
                row=data.edge_index[0], col=data.edge_index[1],
                value=torch.arange(E, device=data.edge_index.device),
                sparse_sizes=(N, N))
            adj, partptr, perm = adj.partition(num_parts, recursive)
            # self.global_adj = adj
            
            if save_dir is not None:
                torch.save((adj, partptr, perm), path)

            if log:  # pragma: no cover
                print('Done!', file=sys.stderr)

        
        # 对于所有的点属性重排
        self.data = self.__permute_data__(data, perm, adj)
        self.global_adj = adj
        self.partptr = partptr
        perm_ = torch.zeros_like(perm)
        for i in range(len(perm)):
            perm_[perm[i]] = i
        self.perm = torch.stack([perm,perm_])
        

    def __permute_data__(self, data, node_idx, adj):
        out = copy.copy(data)
        for key, value in data.items():
            if data.is_node_attr(key):
                out[key] = value[node_idx]

        row, col, _ = adj.coo()
        out.edge_index = torch.stack([row, col], dim=0)
        out.adj = adj
        

        return out

    def __len__(self):
        return self.partptr.numel() - 1

    def __getitem__(self, idx):
        # 第idx个分区的起始id和分区的id数量
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start
        
        N, E = self.data.num_nodes, self.data.num_edges
        data = copy.copy(self.data)
        del data.num_nodes
        adj, data.adj = data.adj, None
        # 将邻接矩阵进行切分
        adj = adj.narrow(0, start, length).narrow(1, start, length)
        # 
        edge_idx = adj.storage.value()

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item.narrow(0, start, length)
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        row, col, _ = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f'  num_parts={self.num_parts}\n'
                f')')
        
