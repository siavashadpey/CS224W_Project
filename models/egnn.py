from typing import Union, Callable, Tuple, Optional

import torch 
from torch import Tensor

from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj

from .egnn import EGNNConv

class EGNN(torch.nn.Module):
    r"""The Equivariant Graph Neural Network Model from the `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/abs/2102.09844>`_ paper, using the :class:`EGNNConv` operator for
    message passing

    Args:
        in_channels     (int): Size of each input node feature.
        hidden_channels (int): Size of each hidden node feature.
        num_layers      (int): Number of EGNNConv layers.
        edge_dim        (int): Size of each edge feature. (default: :obj:`0`)
        update_pos      (bool): If set to :obj:`False`, node positions will not be updated. (default: :obj:`True`)
        act             (str or Callable): Activation function to use. (default: :obj:`"SiLU"`)
    """

    def __init__(
            self,
            in_channels    : int,
            hidden_channels: int,
            num_layers     : int,
            edge_dim       : int = 0,
            update_pos     : bool = True,
            act            : Union['str', Callable] = "SiLU",
            skip_connection: bool = False):
        super().__init__()

        self.embedding_node = torch.nn.Linear(in_channels, hidden_channels)

        act = activation_resolver(act) 
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels + 1 + edge_dim, hidden_channels),
                act,
                torch.nn.Linear(hidden_channels, hidden_channels),
                act)
            nn_node = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels, hidden_channels),
                act,
                torch.nn.Linear(hidden_channels, hidden_channels))
            if update_pos:
                nn_pos = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    act,
                    torch.nn.Linear(hidden_channels, 1))
            else:
                nn_pos = None
            self.convs.append(EGNNConv(nn_edge, nn_node, nn_pos, skip_connection=skip_connection))

    def forward(self,
                edge_index: Adj,
                x: Tensor,
                pos: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        x = self.embedding_node(x)
        for conv in self.convs:
            x, pos = conv(x, pos, edge_index, edge_attr)

        return (x, pos)
        