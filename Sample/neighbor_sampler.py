import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__))))
import math
import torch
import torch.multiprocessing as mp
from typing import Optional, Tuple

import graph_store
from distparser import SampleType, NUM_SAMPLER
from base import BaseSampler, NegativeSampling, SampleOutput
from sample_cores import get_neighbors, neighbor_sample_from_nodes, heads_unique
from torch.distributed.rpc import rpc_async

def outer_sample(graph_name, nodes, ts, fanout_index, with_outer_sample = SampleType.Outer):# 默认此时继续向外采样
    local_sampler = graph_store.get_local_sampler(graph_name)
    assert local_sampler is not None, 'Local_sampler is None!!!'
    out = local_sampler.sample_from_nodes(nodes, with_outer_sample, ts, fanout_index)
    return out

class NeighborSampler(BaseSampler):
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        fanout: list,
        graph_data,
        workers = 1,
        tnb = None,
        is_distinct = 0,
        policy = "uniform",
        is_root_ts = 0,
        edge_weight: Optional[torch.Tensor] = None,
        graph_name = None
    ) -> None:
        r"""__init__
        Args:
            num_nodes: the num of all nodes in the graph
            num_layers: the num of layers to be sampled
            fanout: the list of max neighbors' number chosen for each layer
            workers: the number of threads, default value is 1
            tnb: neighbor infomation table
            is_distinct: 1-need distinct, 0-don't need distinct
            policy: "uniform" or "recent" or "weighted"
            is_root_ts: 1-base on root's ts, 0-base on parent node's ts
            edge_weight: the initial weights of edges
            graph_name: the name of graph
        should provide edge_index or (neighbors, deg)
        """
        super().__init__()
        self.num_layers = num_layers
        # 线程数不超过torch默认的omp线程数
        self.workers = workers # min(workers, torch.get_num_threads())
        self.fanout = fanout
        self.num_nodes = num_nodes
        self.graph_data=graph_data
        self.policy = policy
        self.is_root_ts = is_root_ts
        self.is_distinct = is_distinct
        assert graph_name is not None
        self.graph_name = graph_name
        if(tnb is None):
            if(graph_data.edge_ts is not None):
                timestamp,ind = graph_data.edge_ts.sort()
                timestamp = timestamp.float().contiguous()
                eid = graph_data.eid[ind].contiguous()
                row, col = graph_data.edge_index[:,ind]
            else:
                eid = graph_data.eid
                timestamp = None
                row, col = graph_data.edge_index
            if(edge_weight is not None):
                edge_weight = edge_weight.float().contiguous()
            self.tnb = get_neighbors(row.contiguous(), col.contiguous(), num_nodes, is_distinct, eid, edge_weight, timestamp)
        else:
            assert tnb is not None
            self.tnb = tnb
    
    def _get_sample_info(self):
        return self.num_nodes,self.num_layers,self.fanout,self.workers
    
    def _get_sample_options(self):
        return {"is_distinct" : self.is_distinct,
                "policy" : self.policy,
                "is_root_ts" : self.is_root_ts,
                "with_eid" : self.tnb.with_eid, 
                "weighted" : self.tnb.weighted, 
                "with_timestamp" : self.tnb.with_timestamp}
    
    def insert_edges_with_timestamp(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor, 
            timestamp : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新节点数和tnb
        self.num_nodes = self.tnb.update_neighbors_with_time(
            row.contiguous(), 
            col.contiguous(), 
            timestamp.contiguous(), 
            eid.contiguous(), 
            self.is_distinct, 
            edge_weight.contiguous())
    
    def update_edges_weight(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新tnb的权重信息
        if self.tnb.with_eid:
            self.tnb.update_edge_weight(
                eid.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
        else:
            self.tnb.update_edge_weight(
                row.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
    
    def sample_from_nodes(
        self,
        nodes: torch.Tensor,
        with_outer_sample: SampleType,
        ts: Optional[torch.Tensor] = None,
        fanout_index=0
    ) -> SampleOutput:
        r"""Performs mutilayer sampling from the nodes specified in: nodes
        The specific number of layers is determined by parameter: num_layers
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, list].

        Args:
            nodes: the list of seed nodes index,
            ts: the timestamp of nodes, optional,
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            fanout_index: optional. Specify the index to fanout
        Returns:
            sampled_nodes: the node sampled
            sampled_edge_index_list: the edge sampled
        """
        out = SampleOutput()
        src_nodes = nodes
        sampled_edge_index_list = []
        sampled_eid_list = []
        sampled_eid_ts_list = []
        with_timestap = ts is not None
        assert self.workers > 0, 'Workers should be positive integer!!!'
        if with_outer_sample == SampleType.Whole:
            # sampled_nodes = [torch.LongTensor([]), torch.LongTensor([])]
            sampled_nodes = torch.LongTensor([])
            for i in range(fanout_index, self.num_layers):
                sampled_nodes_i, sampled_node_ts_i,sampled_edge_index_i,sampled_edge_id,sampled_edge_ts = self._sample_one_layer_from_nodes(nodes, self.fanout[i], ts)
                nodes = sampled_nodes_i
                if with_timestap: # ts操作
                    ts = sampled_node_ts_i
                    sampled_eid_ts_list.append(sampled_edge_ts)
                else: # sampled_nodes合并
                    sampled_nodes = torch.cat([sampled_nodes, sampled_nodes_i])
                sampled_edge_index_list.append(sampled_edge_index_i)
                sampled_eid_list.append(sampled_edge_id)            
            if not with_timestap: # sampled_nodes合并
                sampled_nodes = heads_unique(sampled_nodes, src_nodes, self.workers)
        elif with_outer_sample == SampleType.Inner:
            sampled_nodes = torch.LongTensor([])
            for i in range(fanout_index, self.num_layers):
                sampled_nodes_i, sampled_node_ts_i,sampled_edge_index_i,sampled_edge_id,sampled_edge_ts = self._sample_one_layer_from_nodes(nodes, self.fanout[i], ts)
                mask = (sampled_nodes_i >= self.graph_data.partptr[self.graph_data.partition_id]) & \
                       (sampled_nodes_i < self.graph_data.partptr[self.graph_data.partition_id+1])
                nodes = sampled_nodes_i[mask]# 只往外部分区采样一跳，因此第二层开始外部分区节点不进行邻居采样
                if with_timestap: # ts操作
                    ts = sampled_node_ts_i[0]
                    sampled_eid_ts_list.append(sampled_edge_ts)
                else: # sampled_nodes合并
                    sampled_nodes = torch.cat([sampled_nodes, sampled_nodes_i])
                sampled_edge_index_list.append(sampled_edge_index_i)
                sampled_eid_list.append(sampled_edge_id)
            if not with_timestap: # sampled_nodes合并
                sampled_nodes = heads_unique(sampled_nodes, src_nodes, self.workers)
        elif with_outer_sample == SampleType.Outer:
            sampled_nodes_i, sampled_node_ts_i,sampled_edge_index_i,sampled_edge_id,sampled_edge_ts = self._sample_one_layer_from_nodes(nodes, self.fanout[fanout_index], ts)
            #将结果合并入sampled_nodes, sampled_edge_index_list, sampled_eid_list, sampled_eid_ts_list
            if with_timestap: # ts操作
                sampled_eid_ts_list.append(sampled_edge_ts) 
            else: # sampled_nodes合并
                sampled_nodes = sampled_nodes_i
            sampled_edge_index_list.append(sampled_edge_index_i)
            sampled_eid_list.append(sampled_edge_id)
                
            if fanout_index+1<self.num_layers: 
                # 如果接下去还有层的话，采样接下去的层，否则返回结果
                sampled_edge_index_list_merge_part = [torch.LongTensor([[],[]]) for i in range(0, self.num_layers-fanout_index-1)]
                sampled_eid_list_merge_part = [torch.LongTensor([]) for i in range(0, self.num_layers-fanout_index-1)]
                if with_timestap: # ts操作
                    sampled_eid_ts_list_merge_part = [torch.FloatTensor([]) for i in range(0, self.num_layers-fanout_index-1)]
                futs = []
                for i in range(0, self.graph_data.partitions):
                    if i==self.graph_data.partition_id: # 本地采样
                        mask = (sampled_nodes_i >= self.graph_data.partptr[self.graph_data.partition_id]) & \
                               (sampled_nodes_i < self.graph_data.partptr[self.graph_data.partition_id+1])
                        to_sample_node = sampled_nodes_i[mask]
                        if with_timestap: # ts操作
                            to_sample_time = sampled_node_ts_i[mask]
                        else:
                            to_sample_time = None
                        out_i = self.sample_from_nodes(to_sample_node, with_outer_sample, to_sample_time, fanout_index+1)
                        sampled_nodes_part_i = out_i.node
                        sampled_edge_index_part_i = out_i.edge_index_list
                        sampled_eid_part_i = out_i.eid_list
                        sampled_eid_ts_part_i = out_i.eid_ts_list
                        #print(torch.tensor(self.tnb.deg)[sampled_nodes_i[0]].max())
                        if not with_timestap: # sampled_nodes合并
                            sampled_nodes = torch.cat([sampled_nodes, sampled_nodes_part_i])
                        for i in range(0, self.num_layers-fanout_index-1):
                            if(len(sampled_eid_part_i[i])==0):
                                continue
                            sampled_edge_index_list_merge_part[i] = torch.cat([sampled_edge_index_list_merge_part[i], sampled_edge_index_part_i[i]], dim=-1)
                            sampled_eid_list_merge_part[i] = torch.cat((sampled_eid_list_merge_part[i],sampled_eid_part_i[i]),0)
                            if with_timestap: # ts操作
                                sampled_eid_ts_list_merge_part[i] = torch.cat((sampled_eid_ts_list_merge_part[i],sampled_eid_ts_part_i[i]),-1)
                    else: # 远程采样
                        mask = (self.graph_data.partptr[i] <= sampled_nodes_i) &(sampled_nodes_i < self.graph_data.partptr[i+1])
                        to_sample_node = sampled_nodes_i[mask]
                        if with_timestap: # ts操作
                            to_sample_time = sampled_node_ts_i[mask]
                        else:
                            to_sample_time = None
                        # 非本地节点使用rpc通讯调取外分区采样：rpc sample_from_nodes(sampled_nodes_layer[i], fanout_index+1)
                        if(len(to_sample_node)==0):
                            futs.append(None)
                        else:
                            fut = rpc_async((NUM_SAMPLER + 1) * i  + 1, outer_sample, 
                                            args=(self.graph_name,to_sample_node,to_sample_time,fanout_index+1,with_outer_sample))
                            futs.append(fut)
                for i in range(self.graph_data.partitions-1):
                    if(futs[i] == None):
                        continue
                    futs[i].wait()
                    out_i = futs[i].value()
                    sampled_nodes_part_i = out_i.node
                    sampled_edge_index_part_i = out_i.edge_index_list
                    sampled_eid_part_i = out_i.eid_list
                    sampled_eid_ts_part_i = out_i.eid_ts_list
                    if not with_timestap: # sampled_nodes合并
                        sampled_nodes = torch.cat([sampled_nodes, sampled_nodes_part_i])
                    for i in range(0, self.num_layers-fanout_index-1):
                        sampled_edge_index_list_merge_part[i] = torch.cat([sampled_edge_index_list_merge_part[i], sampled_edge_index_part_i[i]], dim=-1)
                        sampled_eid_list_merge_part[i] = torch.cat((sampled_eid_list_merge_part[i],sampled_eid_part_i[i]),0)
                        if with_timestap: # ts操作
                            sampled_eid_ts_list_merge_part[i] = torch.cat((sampled_eid_ts_list_merge_part[i],sampled_eid_ts_part_i[i]),-1)
                sampled_edge_index_list.extend(sampled_edge_index_list_merge_part)
                sampled_eid_list.extend(sampled_eid_list_merge_part)
                if with_timestap: # ts操作
                    sampled_eid_ts_list.extend(sampled_eid_ts_list_merge_part)
                if fanout_index==0 and not with_timestap:
                    #最外层返回前合并结果并去重
                    sampled_nodes = heads_unique(sampled_nodes, src_nodes, self.workers)
        else:
            raise Exception("with_outer_sample has error value!!! ")
        out.edge_index_list = sampled_edge_index_list
        out.eid_list = sampled_eid_list
        if with_timestap: # ts操作
            out.eid_ts_list = sampled_eid_ts_list
        else:
            out.node = sampled_nodes
        return out
    
    def sample_from_edges(
        self,
        edges: torch.Tensor,
        with_outer_sample: SampleType,
        ets: Optional[torch.Tensor] = None,
        edge_label: Optional[torch.Tensor] = None,
        neg_sampling: Optional[NegativeSampling] = None
    ) -> SampleOutput:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        Args:
            edges: the list of seed edges index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            ets: the timestamp of edges, optional,
            edge_label: the label for the seed edges.
            neg_sampling: The negative sampling configuration
        Returns:
            sampled_edge_index_list: the edges sampled
            sampled_eid_list: the edges' id sampled
            sampled_eid_ts_list:the edges' timestamp sampled
            metadata: other infomation
        """
        src, dst = edges
        num_pos = src.numel()
        num_neg = 0
        with_timestap = ets is not None
        seed_ts = None
        if edge_label is None:
            edge_label = torch.ones(num_pos)

        if neg_sampling is not None:
            num_neg = math.ceil(num_pos * neg_sampling.amount)
            if neg_sampling.is_binary():
                src_neg = neg_sampling.sample(num_neg, self.num_nodes)
                # print('src_neg',src_neg.dtype)
                src = torch.cat([src, src_neg], dim=0)
                dst_neg = neg_sampling.sample(num_neg, self.num_nodes)
                dst = torch.cat([dst, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets, ets], dim=0)
            if neg_sampling.is_triplet():
                dst_neg = neg_sampling.sample(num_neg, self.num_nodes)
                dst = torch.cat([dst, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets], dim=0)
        else:
            if with_timestap: # ts操作
                seed_ts = torch.cat([ets, ets], dim=0)

        seed = torch.cat([src, dst], dim=0)
        if with_timestap: # ts操作
            pair, inverse_seed= torch.unique(torch.stack([seed, seed_ts],0), return_inverse=True, dim=1)
            seed, seed_ts = pair
            seed = seed.long()
        else:
            seed, inverse_seed = seed.unique(return_inverse=True)
        
        if with_outer_sample==SampleType.Outer:
            # 划分seed分区
            seeds_part=[]
            seeds_ts_part=[]
            for i in range(len(self.graph_data.partptr)-1):
                mask = (seed< self.graph_data.partptr[i+1]) & (self.graph_data.partptr[i] <= seed)
                seeds_part.append(seed[mask])
                if with_timestap: # ts操作
                    seeds_ts_part.append(seed_ts[mask])
                else:
                    seeds_ts_part.append(None)
            # 对各个分区seed执行点采样
            futs = []
            for i in range(0, self.graph_data.partitions):
                if i==self.graph_data.partition_id:
                    out_i = self.sample_from_nodes(seeds_part[i], with_outer_sample, seeds_ts_part[i])
                    sampled_nodes = out_i.node
                    sampled_edge_index_list = out_i.edge_index_list
                    sampled_eid_list = out_i.eid_list
                    sampled_eid_ts_list = out_i.eid_ts_list
                else:
                    fut = rpc_async((NUM_SAMPLER + 1) * i  + 1, outer_sample, 
                                    args=(self.graph_name,seeds_part[i], seeds_ts_part[i], 0, with_outer_sample))
                    futs.append(fut)
            # 获取远程采样结果并合并
            for i in range(0,self.graph_data.partitions-1):
                fut = futs[i]
                fut.wait()
                out_i = fut.value()
                sampled_nodes_part_i = out_i.node
                sampled_edge_index_list_part_i = out_i.edge_index_list
                sampled_eid_list_part_i = out_i.eid_list
                sampled_eid_ts_list_part_i = out_i.eid_ts_list
                #将各个part结果合并
                if not with_timestap: # sampled_nodes合并
                    sampled_nodes = torch.cat([sampled_nodes, sampled_nodes_part_i])
                for i in range(0, self.num_layers-1):
                    sampled_edge_index_list[i] = torch.cat([sampled_edge_index_list[i], sampled_edge_index_list_part_i[i]], dim=-1)
                    sampled_eid_list[i] = torch.cat([sampled_eid_list[i], sampled_eid_list_part_i[i]], dim=-1)
                    if with_timestap: # ts操作
                        sampled_eid_ts_list[i] = torch.cat([sampled_eid_ts_list[i], sampled_eid_ts_list_part_i[i]], dim=-1)
            out = SampleOutput()
            out.edge_index_list = sampled_edge_index_list
            out.eid_list = sampled_eid_list
            if with_timestap: # ts操作
                out.eid_ts_list = sampled_eid_ts_list
            else:
                out.node = sampled_nodes
        else:
            out = self.sample_from_nodes(seed, with_outer_sample, seed_ts)

        if neg_sampling is None or neg_sampling.is_binary():
            edge_label_index = inverse_seed.view(2, -1)
            # edge_label_index是原本edge每一个node_id在采样节点重的序号
            # edge_label是seed links的标签
            out.metadata = {'edge_label_index':seed[edge_label_index], 'edge_label':edge_label}
        elif neg_sampling.is_triplet():
            src_index = inverse_seed[:num_pos]
            dst_pos_index = inverse_seed[num_pos:2 * num_pos]
            dst_neg_index = inverse_seed[2 * num_pos:]
            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)
            # src_index是seed里src点的索引
            # dst_pos_index是seed里dst_pos点的索引
            # dst_neg_index是seed里dst_neg点的索引
            out.metadata = {'src_id':seed[src_index], 'dst_pos_id':seed[dst_pos_index], 'dst_neg_id':seed[dst_neg_index]}
        # sampled_nodes最前方是原始序列的采样起点也就是去重后的seed
        return out
    
    def _sample_one_layer_from_nodes(
        self,
        nodes: torch.Tensor,
        fanout: int,
        root_ts: Optional[torch.Tensor]=None
     ):
        if(root_ts is None):
            row, col, eid, sampled_nodes = neighbor_sample_from_nodes(
                nodes.contiguous(), self.tnb, self.graph_data.partition_id, self.graph_data.partptr.tolist(), fanout, self.workers, 
                self.policy, None, None, None, None)
            return sampled_nodes, None, torch.stack([row,col], dim=0), eid, None
        else:
            row, col, eid, e_ts, src_ts, sampled_nodes, sampled_ts = neighbor_sample_from_nodes(
                nodes.contiguous(), self.tnb, self.graph_data.partition_id, self.graph_data.partptr.tolist(), fanout, self.workers, 
                self.policy, root_ts.float().contiguous(), self.is_root_ts, None, None)
            return sampled_nodes, sampled_ts, torch.stack([row,col], dim=0), eid, torch.stack([e_ts,src_ts])
        
    def _sample_one_layer_from_nodes_slice(
        self,
        nodes: torch.Tensor,
        fanout: int,
        end: float,
        start: Optional[float]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Performs sampling from the nodes specified in: nodes,
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, torch.Tensor].

        Args:
            nodes: the list of seed nodes index
            fanout: the number of max neighbors chosen
        Returns:
            sampled_nodes: the nodes sampled
            sampled_edge_index: the edges sampled
        """
        row, col, sampled_nodes = neighbor_sample_from_nodes(
                nodes.contiguous(), self.tnb, self.graph_data.partition_id, self.graph_data.partptr.tolist(), fanout, self.workers, 
                self.policy, None, None, start, end)
        return sampled_nodes, torch.stack([row,col], dim=0)


if __name__=="__main__":
    # edge_index1 = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
    #                             [1, 0, 2, 4, 1, 3, 0, 3, 5, 0, 2]])# , 2, 5
    edge_index1 = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
                                [1, 0, 2, 0, 4, 1, 3, 0, 3, 3, 5, 0, 2]])# , 2, 5
    edge_weight1 = None
    timeStamp=torch.FloatTensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    num_nodes1 = 6
    num_neighbors = 2
    # Run the neighbor sampling
    from Utils import GraphData
    g_data = GraphData(id=0, edge_index=edge_index1, timestamp=timeStamp, data=None, partptr=torch.tensor([0, num_nodes1]))
    sampler = NeighborSampler(num_nodes=num_nodes1, 
                              num_layers=3, 
                              fanout=[2, 1, 1], 
                              edge_weight=edge_weight1, 
                              graph_data=g_data, 
                              graph_name='a',
                              workers=4,
                              is_root_ts=0,
                              is_distinct = 0)

    out = sampler.sample_from_nodes(torch.tensor([1,2]),
                                    with_outer_sample=SampleType.Whole, 
                                    ts=torch.tensor([1, 2]))
    # out = sampler.sample_from_edges(torch.tensor([[1,2],[4,0]]), 
    #                                 with_outer_sample=SampleType.Whole, 
    #                                 ets = torch.tensor([1, 2]))
    
    # Print the result
    print('node:', out.node)
    print('edge_index_list:', out.edge_index_list)
    print('eid_list:', out.eid_list)
    print('eid_ts_list:', out.eid_ts_list)
    print('metadata: ', out.metadata)
