import os.path as osp
import torch
class GraphData():
    def __init__(self, path):
        assert path is not None and osp.exists(path),'path 不存在'
        id,edge_index,data,partptr =torch.load(path)
        # 当前分区序号
        self.partition_id = id
        # 总分区数
        self.partitions = partptr.numel() - 1
        # 全图结构数据
        self.num_nodes = partptr[self.partitions]
        self.num_edges = edge_index[0].numel()
        self.edge_index = edge_index
        # 该分区下的数据（包含特征向量和子图结构）pyg Data数据结构
        self.data = data
        # 分区映射关系
        self.partptr = partptr

    def __init__(self, id, edge_index, data, partptr, timestamp=None):
        # 当前分区序号
        self.partition_id = id
        # 总分区数
        self.partitions = partptr.numel() - 1
        # 全图结构数据
        self.num_nodes = partptr[self.partitions]
        if edge_index is not None:
            self.num_edges = edge_index[0].numel()
        self.edge_index = edge_index
        self.edge_ts = timestamp
        # 该分区下的数据（包含特征向量和子图结构）pyg Data数据结构
        self.data = data
        # 分区映射关系
        self.partptr = partptr
        # edge id
        self.eid = torch.tensor([i for i in range(0, self.num_edges)])

    def select_attr(self,index):
        return torch.index_select(self.data.x,0,index)

    #返回全局的节点id 所对应的分区
    def get_part_num(self):
        return self.data.x.size()[0]

    def select_attr(self,index):
        return torch.index_select(self.data.x,0,index)
    def select_y(self,index):
        return torch.index_select(self.data.y,0,index)
    #返回全局的节点id 所对应的分区
    def get_localId_by_partitionId(self,id,index):
        #print(index)
        if(id == -1 or id == 0):
            return index
        else:
            return torch.add(index,-self.partptr[id])
    def get_globalId_by_partitionId(self,id,index):
        if(id == -1 or id == 0):
            return index
        else:
            return torch.add(index,self.partptr[id])

    def get_node_num(self):    
        return self.num_nodes
    
    
    def localId_to_globalId(self,id,partitionId:int = -1):
        '''
        将分区partitionId内的点id映射为全局的id
        '''
        if partitionId == -1:
            partitionId = self.partition_id
        assert id >=self.partptr[self.partition_id] and id < self.partptr[self.partition_id+1]
        ids_before = 0
        if self.partition_id>0:
            ids_before = self.partptr[self.partition_id-1]
        return id+ids_before
    
    def get_partitionId_by_globalId(self,id):
        '''
        通过全局id得到对应的分区序号
        '''
        partitionId = -1
        assert id>=0 and id<self.num_nodes,'id 超过范围'
        for i in range(self.partitions):
            if id>=self.partptr[i] and id<self.partptr[i+1]:
                partitionId = i
                break
        assert partitionId>=0, 'id 不存在对应的分区'
        return partitionId
    
    def get_nodes_by_partitionId(self,id):
        '''
        根据partitioId 返回该分区的节点数量
        
        '''
        assert id>=0 and id<self.partitions,'partitionId 非法'
        return (int)(self.partptr[id+1]-self.partptr[id])
        
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  partition_id={self.partition_id}\n'
                f'  data={self.data},\n'
                f'  global_info('
                f'num_nodes={self.num_nodes},'
                f' num_edges={self.num_edges},'
                f' num_parts={self.partitions},'
                f' edge_index=[2,{self.edge_index[0].numel()}])\n'
                f')')
