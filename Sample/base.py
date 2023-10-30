
import torch
from torch import Tensor
from enum import Enum
import math
from abc import ABC
from typing import Any, List, Optional, Tuple, Union

from distparser import SampleType


class NegativeSamplingMode(Enum):
    # 'binary': Randomly sample negative edges in the graph.
    binary = 'binary'
    # 'triplet': Randomly sample negative destination nodes for each positive
    # source node.
    triplet = 'triplet'
    

class NegativeSampling:
    r"""The negative sampling configuration of a
    :class:`~torch_geometric.sampler.BaseSampler` when calling
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.
    Args:
        mode (str): The negative sampling mode
            (:obj:`"binary"` or :obj:`"triplet"`).
            If set to :obj:`"binary"`, will randomly sample negative links
            from the graph.
            If set to :obj:`"triplet"`, will randomly sample negative
            destination nodes for each positive source node.
        amount (int or float, optional): The ratio of sampled negative edges to
            the number of positive edges. (default: :obj:`1`)
        weight (torch.Tensor, optional): A node-level vector determining the
            sampling of nodes. Does not necessariyl need to sum up to one.
            If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
    """
    mode: NegativeSamplingMode
    amount: Union[int, float] = 1
    weight: Optional[Tensor] = None

    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        amount: Union[int, float] = 1,
        weight: Optional[Tensor] = None,
    ):
        self.mode = NegativeSamplingMode(mode)
        self.amount = amount
        self.weight = weight

        if self.amount <= 0:
            raise ValueError(f"The attribute 'amount' needs to be positive "
                             f"for '{self.__class__.__name__}' "
                             f"(got {self.amount})")

        if self.is_triplet():
            if self.amount != math.ceil(self.amount):
                raise ValueError(f"The attribute 'amount' needs to be an "
                                 f"integer for '{self.__class__.__name__}' "
                                 f"with 'triplet' negative sampling "
                                 f"(got {self.amount}).")
            self.amount = math.ceil(self.amount)

    def is_binary(self) -> bool:
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def sample(self, num_samples: int,
               num_nodes: Optional[int] = None) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        if self.weight is None:
            if num_nodes is None:
                raise ValueError(
                    f"Cannot sample negatives in '{self.__class__.__name__}' "
                    f"without passing the 'num_nodes' argument")
            return torch.randint(num_nodes, (num_samples, ))

        if num_nodes is not None and self.weight.numel() != num_nodes:
            raise ValueError(
                f"The 'weight' attribute in '{self.__class__.__name__}' "
                f"needs to match the number of nodes {num_nodes} "
                f"(got {self.weight.numel()})")
        return torch.multinomial(self.weight, num_samples, replacement=True)

class SampleOutput:
    node: Optional[torch.Tensor] = None
    edge_index_list: Optional[List[torch.Tensor]] = None
    eid_list: Optional[List[torch.Tensor]] = None
    eid_ts_list: Optional[List[torch.Tensor]] = None
    metadata: Optional[Any] = None

class BaseSampler(ABC):
    r"""An abstract base class that initializes a graph sampler and provides
    :meth:`_sample_one_layer_from_nodes`
    :meth:`_sample_one_layer_from_nodes_parallel`
    :meth:`sample_from_nodes` routines.
    """

    def sample_from_nodes(
        self,
        nodes: torch.Tensor,
        with_outer_sample: SampleType,
        **kwargs
    ) -> Tuple[torch.Tensor, list]:
        r"""Performs mutilayer sampling from the nodes specified in: nodes
        The specific number of layers is determined by parameter: num_layers
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, list].

        Args:
            nodes: the list of seed nodes index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            **kwargs: other kwargs
        Returns:
            sampled_nodes: the nodes sampled
            sampled_edge_index_list: the edges sampled
        """
        raise NotImplementedError
    
    def sample_from_edges(
        self,
        edges: torch.Tensor,
        with_outer_sample: SampleType,
        edge_label: Optional[torch.Tensor] = None,
        neg_sampling: Optional[NegativeSampling] = None
    ) -> Tuple[torch.Tensor, list]:
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
            metadata: other infomation
        """
        raise NotImplementedError

    # def _sample_one_layer_from_nodes(
    #     self,
    #     nodes:torch.Tensor,
    #     **kwargs
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     r"""Performs sampling from the nodes specified in: nodes,
    #     returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, torch.Tensor].

    #     Args:
    #         nodes: the list of seed nodes index
    #         **kwargs: other kwargs
    #     Returns:
    #         sampled_nodes: the nodes sampled
    #         sampled_edge_index: the edges sampled
    #     """
    #     raise NotImplementedError
    
    # def _sample_one_layer_from_nodes_parallel(
    #     self, 
    #     nodes: torch.Tensor,
    #     **kwargs
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     r"""Performs sampling paralleled from the nodes specified in: nodes,
    #     returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, torch.Tensor].

    #     Args:
    #         nodes: the list of seed nodes index
    #         **kwargs: other kwargs
    #     Returns:
    #         sampled_nodes: the nodes sampled
    #         sampled_edge_index: the edges sampled
    #     """
    #     raise NotImplementedError
