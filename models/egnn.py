from typing import Union, Callable, Tuple, Optional

import torch 
from torch import Tensor

from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj

from . import EGNNConv

class EGNN(torch.nn.Module):
    r"""The Equivariant Graph Neural Network Model from the `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/abs/2102.09844>`_ paper, using the :class:`EGNNConv` operator for
    message passing

    Args:
        in_channels     (int): Size of each input node feature.
        hidden_channels (int): Size of each hidden node feature.
        num_layers      (int): Number of EGNNConv layers.
        pos_dim         (int): Dimension of node positions.
        edge_dim        (int): Size of each edge feature. (default: :obj:`0`)
        update_pos      (bool): If set to :obj:`False`, node positions will not be updated. (default: :obj:`True`)
        act             (str or Callable): Activation function to use. (default: :obj:`"SiLU"`)
    """

    def __init__(
            self,
            in_channels    : int,
            hidden_channels: int,
            num_layers     : int,
            pos_dim        : int,
            edge_dim       : int = 0,
            update_pos     : bool = True,
            act            : Union['str', Callable] = "SiLU",
            skip_connection: bool = False,
            pos_scale      : float = 0.0):
        super().__init__()

        assert skip_connection is False or in_channels == hidden_channels, "Skip connection requires in_channels == hidden_channels"
        act = activation_resolver(act) 

        # Store position scale as a learnable parameter
        self.pos_scale_logit = torch.nn.Parameter(torch.logit(torch.tensor(pos_scale))) if pos_scale > 0 else None

        self.convs = torch.nn.ModuleList()

        nn_edge = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels + 1 + edge_dim, hidden_channels),
            act,
            torch.nn.Linear(hidden_channels, hidden_channels),
            act)
        nn_node = torch.nn.Sequential(
            torch.nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
            torch.nn.Linear(hidden_channels, hidden_channels))
        if update_pos:
            layer_pos = torch.nn.Linear(hidden_channels, 1, bias=True)
            torch.nn.init.xavier_uniform_(layer_pos.weight, gain=0.001)
            torch.nn.init.zeros_(layer_pos.bias)
            nn_pos = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                act,
                layer_pos)
        else:
            nn_pos = None
        self.convs.append(EGNNConv(nn_edge, nn_node, pos_dim, nn_pos, skip_connection))
        for _ in range(num_layers-1):
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
                layer_pos = torch.nn.Linear(hidden_channels, 1, bias=True)
                torch.nn.init.xavier_uniform_(layer_pos.weight, gain=0.001)
                torch.nn.init.zeros_(layer_pos.bias)
                nn_pos = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    act,
                    layer_pos)
            else:
                nn_pos = None
            self.convs.append(EGNNConv(nn_edge, nn_node, pos_dim, nn_pos, skip_connection))
    
    def forward(self,
                x: Tensor,
                pos: Tensor,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        pos_clamp = 500.0  # for final position 
        pos_update_clamp = 5.0  # for aggregated position update

        for conv in self.convs:
            x, pos_new = conv(x, pos, edge_index, edge_attr)

        # scale position updates
        if self.pos_scale_logit is not None:
            pos_update = pos_new - pos
            pos_update = pos_update * torch.sigmoid(self.pos_scale_logit)

            # softer clamping - allow larger updates
            pos_update = torch.clamp(pos_update, min=-pos_update_clamp, max=pos_update_clamp)
            pos = pos + pos_update

            # clamp final positions
            pos = torch.clamp(pos, min=-pos_clamp, max=pos_clamp)
            
        return (x, pos)
        