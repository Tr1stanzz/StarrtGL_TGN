import argparse
import os

from DistGraphLoader import DataSet, partition_load
path1=os.path.abspath('.')  
import torch

from Sample.neighbor_sampler import NeighborSampler
from Sample.neighbor_sampler import get_neighbors
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
"""
test command 
python test.py --world_size 2 --rank 0 
--world_size', default=4, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='rank of the worker')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
parser.add_argument('--num_sampler', type=int, default=10, metavar='S',
                    help='number of samplers')
parser.add_argument('--queue_size', type=int, default=10, metavar='S',
                    help='sampler queue size')
"""
sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'l2':5e-7
             }

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data:DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(device='cpu')
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        # 需要通过DDP包装model 告诉model该复制到哪些gpu中
        self.model = DDP(model)
        self.total_correct = 0
        self.count = 0
    def _run_test(self,batchData:BatchData):
        self.count  = self.count +1 
        print(f'run epoch in batch data f {self.count}')
        l = len(batchData.edge_index)
        # print(f'batchData:len:{l},edge_index:{batchData.edge_index}')
        out = self.model(batchData,1)
        # print(f'out size: {out.size()}')
        # print(f'batchDate.y: {batchData.y}')
        # # batchData.y = F.one_hot(batchData.y, num_classes=7)
        # print(f'roots {batchData.roots}')
        # print(f'y size:{batchData.y.size()}')
        # print(f'mask :{batchData.train_mask}')
        ##loss = F.nll_loss(out, batchData.y)
        y = batchData.roots.labels#torch.masked_select(graph.data.y,graph.data.train_mask)
        loss = F.nll_loss(out, y)
        self.total_correct += int(out.argmax(dim=-1).eq(y).sum())
        ##self.total_correct += int(out.argmax(dim=-1).eq(batchData.y).sum())
        self.total_count += y.size(0)
    def _run_batch(self, batchData:BatchData):
        # graph = GraphData('/home/sxx/zlj/rpc_ps/part/metis_1/rank_0')
        self.count  = self.count +1 
        print(f'run epoch in batch data f {self.count}')
        l = len(batchData.edge_index)
        # print(f'batchData:len:{l},edge_index:{batchData.edge_index}')
        self.optimizer.zero_grad()
        out = self.model(batchData,0)
        # print(f'out size: {out.size()}')
        # print(f'batchDate.y: {batchData.y}')
        # # batchData.y = F.one_hot(batchData.y, num_classes=7)
        # print(f'roots {batchData.roots}')
        # print(f'y size:{batchData.y.size()}')
        # print(f'mask :{batchData.train_mask}')
        ##loss = F.nll_loss(out, batchData.y)
        y = batchData.roots.labels#torch.masked_select(graph.data.y,graph.data.train_mask)
        loss = F.nll_loss(out, y)
        loss.backward()
        self.optimizer.step()
        print(out.argmax(dim=-1))
        self.total_correct += int(out.argmax(dim=-1).eq(y).sum())
        ##self.total_correct += int(out.argmax(dim=-1).eq(batchData.y).sum())
        self.total_count += y.size(0)
        print('finish')

    def _run_epoch(self, epoch):
        self.total_correct = 0
        self.total_count = 0
        for batchData in self.train_data:
            if batchData.edge_index[0].numel()!=0:
                self._run_batch(batchData)
        approx_acc = self.total_correct / self.total_count

        print(f"=======[GPU{self.gpu_id}] Epoch {epoch}  | approx_acc: {approx_acc}=======")
        self.total_correct = 0
        self.total_count = 0
        for batchData in self.test_data:
            if batchData.edge_index[1].numel()!=0:
                self._run_test(batchData)
        approx_acc = self.total_correct / self.total_count

        print(f"=======[GPU{self.gpu_id}] Epoch {epoch}  | test_approx_acc: {approx_acc}=======")
    def _save_checkpoint(self, epoch):
        # 由于model现在有了一层ddp的封装，访问模型的参数需要model.module
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"=======Epoch {epoch} | Training checkpoint saved at {PATH}========")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # 对于checkpoint只保存一份即可
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
    
    @torch.no_grad()
    def test(layer_loader, model, data, split_idx, device, no_conv=False):
        # data.y is labels of shape (N, ) 
        model.eval()
        
        out = model.inference(data.x, layer_loader, device)
    #     out = model.inference_all(data)
        y_pred = out.exp()  # (N,num_classes)   
        
        losses = dict()
        for key in ['train', 'valid', 'test']:
            node_id = split_idx[key]
            node_id = node_id.to(device)
            losses[key] = F.nll_loss(out[node_id], data.roots.labels).item()
                
        return losses, y_pred

def main():   
    
    rank = distparser._get_worker_rank()
    DistCustomPool.init_distribution('127.0.0.1',9675,'127.0.0.1',10023,backend = "gloo")
    #graph =   DistGraphData('./part/metis_1')
    # graph =   DistGraphData(path='/home/sxx/pycode/work/ogbn-products/metis_2', full_edge = False)
    pdata = partition_load("./cora", algo="metis")
    graph = DistGraphData(pdata = pdata,edge_index= pdata.edge_index, full_edge = False)
    sampler = NeighborSampler(graph_name = 'sampler',graph_data= graph ,num_nodes = graph.num_nodes, num_layers=2, fanout=[10,10], workers=10)
    traindata = torch.arange(graph.partptr[rank],graph.partptr[rank+1])[graph.data.train_mask]
    trainlabel = graph.data.y[graph.data.train_mask]
    testdata =  torch.arange(graph.partptr[rank],graph.partptr[rank+1] )[graph.data.test_mask]
    testlabel = graph.data.y[graph.data.test_mask]
    train_input_data = DataSet(nodes = traindata,labels = trainlabel)
    test_input_data =DataSet(nodes = testdata,labels = testlabel)
    loader = DistributedDataLoader('train',graph,train_input_data,sampler = sampler,batch_size = 100,shuffle=True,cache_memory_size = 0)
    testloader = DistributedDataLoader('test',graph,test_input_data,sampler = sampler,batch_size = 100,shuffle=True)

    # count_node = 0
    # count_edge = 0
    # count_x_byte = 0
    # start_time = time.time()
    # cnt = 0
    # for batchData in loader:
    #    cnt = cnt+1
    #    count_node += batchData.nids.size(0)
    #    count_x_byte += batchData.x.numel()*batchData.x.element_size()
    #    for edge_list in batchData.edge_index:
    #        count_edge += edge_list.size(1)
    #    #print(batchData.x)
    #   #count_edge += batchData.edge_index.size(1)
    #    dt = time.time() - start_time
    #    print('{} count node {},count edge {}, node TPS {},edge TPS {}, x size {}, x TPS {} byte'
    #          .format(cnt,count_node,count_edge,count_node/dt,count_edge/dt,count_x_byte,count_x_byte/dt),batchData.x.size(0),batchData.x.element_size(),batchData.x.numel())
    epochs = 100
    model = 'sage_neighsampler'
    if model == 'sage_neighsampler':
        para_dict = sage_neighsampler_parameters
        num_classes = 7
        num_node_features = graph.num_feasures 
        model = GraphSAGE(num_node_features,num_classes)
        num_node_features = graph.num_feasures 
    print(f'Model {model} initialized')
    epochs = epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'])

    model.train()
    trainer = Trainer(model, loader,testloader, optimizer, graph.rank, 5)
    trainer.train(epochs)
#
    DistCustomPool.close_distribution()
if __name__ == "__main__":
    main()
