import argparse
import os
path1=os.path.abspath('.')  
import torch
from Sample.neighbor_sampler import NeighborSampler
from part.Utils import GraphData
from DistGraphLoader import DistGraphData
from DistGraphLoader import DistributedDataLoader
from torch_geometric.data import Data
import distparser
from DistCustomPool import CustomPool
import DistCustomPool
from torch.distributed import rpc
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributed.rpc import TensorPipeRpcBackendOptions
import time
from model import GraphSAGE
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import BatchData
from part.Utils import GraphData
from torch_geometric.sampler.neighbor_sampler import NeighborSampler as PYGSampler
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader


graph =   DistGraphData('/home/sxx/pycode/work/ogbn-products/metis_1')

loader = NeighborLoader(
    graph.data,
    num_neighbors=[2,3],
    batch_size = 100,
    input_nodes = graph.data.train_mask,
)
count_node = 0
count_edge = 0
count_x_byte = 0
start_time = time.time()
cnt = 0
for batchData in loader:
    cnt = cnt +1 
    count_node += batchData.x.size(0)
    count_edge += batchData.edge_index.size(1)
    count_x_byte += batchData.x.numel()*batchData.x.element_size()
    dt = time.time() - start_time
    print('{} count node {},count edge {}, node TPS {},edge TPS {}, x size {}, x TPS {} byte'
              .format(cnt,count_node,count_edge,count_node/dt,count_edge/dt,count_x_byte,count_x_byte/dt),batchData.x.size(0),batchData.x.element_size(),batchData.x.numel())
