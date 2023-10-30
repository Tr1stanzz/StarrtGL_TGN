import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import MyModel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import BatchData
from part.Utils import GraphData

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # 指定主gpu和相应的端口号用于协调不同gpu之间的通信
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    # 初始化所有的gpu 每个gpu都拥有一个进程，彼此可以互相发现
    # rank是每个process的唯一标识符，world_size是总共的gpu个数
    # nccl 是nvidia的通信库的后端，用于分布式通信
    #init_process_group(backend="gloo", rank=rank, world_size=world_size)
    init_method="tcp://{}:{}".format("localhost",12355)
    init_process_group(backend="gloo", rank=rank, world_size=world_size,init_method=init_method)
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to('cpu')
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # 需要通过DDP包装model 告诉model该复制到哪些gpu中
        self.model = DDP(model)
        self.total_correct = 0
        self.count = 0
    def _run_batch(self, batchData:BatchData):
        graph = GraphData('/home/sxx/zlj/rpc_ps/part/metis_1/rank_0')
        self.count  = self.count +1 
        print(f'run epoch in batch data f {self.count}')
        l = len(batchData.edge_index)
        # print(f'batchData:len:{l},edge_index:{batchData.edge_index}')
        self.optimizer.zero_grad()
        out = self.model(batchData)
        print(f'out size: {out.size()}')
        print(f'batchDate.y: {batchData.y}')
        # batchData.y = F.one_hot(batchData.y, num_classes=7)
        print(f'roots {batchData.roots}')
        print(f'y size:{batchData.y.size()}')
        # print(f'mask :{batchData.train_mask}')
        ##loss = F.nll_loss(out, batchData.y)
        y = batchData.y#torch.masked_select(graph.data.y,graph.data.train_mask)
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
            self._run_batch(batchData)
        approx_acc = self.total_correct / self.total_count
        print(f"=======[GPU{self.gpu_id}] Epoch {epoch}  | approx_acc: {approx_acc}=======")
  
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


# def  load_train_objs():
#     train_set = 
#     model = 
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        # shuffle在sampler已经实现了
        shuffle=False,
        # 分布式采样能够不重合地在不同的gpu上对样本进行采样
        # sampler=DistributedSampler(dataset)
        
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    # 销毁ddp 所有进程
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', dest='total_epochs',type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', dest='save_every',type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--rank', '-r', dest='rank',type=int, help='Rank of this process')
    parser.add_argument('--world-size', dest='world_size',type=int, default=1, help='World size')
    args = parser.parse_args()
    
    
    # world_size = torch.cuda.device_count()
    world_size = 1
    # 一个分布式api，同时启动多个进程
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=2)
    main(args.rank,args.world_size,args.save_every,args.total_epochs,args.batch_size)
