import torch
import torch.multiprocessing as mp
from typing import Optional, Tuple

from base import BaseSampler, NegativeSampling, SampleOutput
from neighbor_sampler import NeighborSampler, SampleType

class RandomWalkSampler(BaseSampler):
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
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
            workers: the number of threads, default value is 1
            tnb: neighbor infomation table
            is_distinct: 1-need distinct, 0-don't need distinct
            policy: "uniform" or "recent" or "weighted"
            is_root_ts: 1-base on root's ts, 0-base on parent node's ts
            edge_weight: the initial weights of edges
            graph_name: the name of graph
        """
        super().__init__()
        # if(edge_index is not None):
        #     self.sampler = NeighborSampler(num_nodes, num_layers, [1 for _ in range(num_layers)], 
        #                                    workers, edge_index=edge_index)
        # elif(tnb is not None):
        #     self.sampler = NeighborSampler(num_nodes, num_layers, [1 for _ in range(num_layers)], 
        #                                    workers, tnb=tnb)
        # else:
        #     raise Exception("Not enough parameters")
        
        self.sampler = NeighborSampler(num_nodes, 
                          tnb=tnb,
                          num_layers=2, 
                          fanout=[1 for i in range(num_layers)], 
                          graph_data=graph_data,
                          edge_weight = edge_weight, 
                          workers=workers, 
                          policy=policy,
                          graph_name=graph_name,
                          is_distinct = is_distinct,
                          is_root_ts = is_root_ts
                          )
        self.num_layers = num_layers
        # 线程数不超过torch默认的omp线程数
        self.workers = min(workers, torch.get_num_threads())

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
        ts: Optional[torch.Tensor] = None
    ) -> SampleOutput:
        r"""Performs mutilayer sampling from the nodes specified in: nodes
        The specific number of layers is determined by parameter: num_layers
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, list].

        Args:
            nodes: the list of seed nodes index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
        Returns:
            sampled_nodes: the node sampled
            sampled_edge_index: the edge sampled
        """
        return self.sampler.sample_from_nodes(nodes, with_outer_sample, ts)
    
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
            edge_label: the label for the seed edges.
            neg_sampling: The negative sampling configuration
        Returns:
            sampled_nodes: the nodes sampled
            sampled_edge_index_list: the edges sampled
        """
        return self.sampler.sample_from_edges(edges, with_outer_sample, ets, edge_label, neg_sampling)


if __name__=="__main__":
    edge_index1 = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
                                [1, 0, 2, 0, 4, 1, 3, 0, 3, 3, 5, 0, 2]])# , 2, 5
    timeStamp=torch.FloatTensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    edge_weight1 = None
    num_nodes1 = 6
    num_neighbors = 2
    # Run the neighbor sampling
    from Utils import GraphData
    g_data = GraphData(id=0, edge_index=edge_index1, timestamp=timeStamp, data=None, partptr=torch.tensor([0, num_nodes1]))
    # Run the random walk sampling
    sampler=RandomWalkSampler(num_nodes=num_nodes1, 
                              num_layers=3,
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

