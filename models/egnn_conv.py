from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    Size,
)

import torch_scatter

class EGNNConv(MessagePassing):
    """
    Equivariant Graph Neural Network Layer.
    Args:
        nn_edge  (Callable): Neural network for edge message computation.
        nn_node  (Callable): Neural network for node feature updates.
        pos_dim  (int): Dimension of node positions. 
        nn_pos   (Callable): Neural network for position updates. (optional)
        skip_connection (bool): If set to :obj:`True`, adds skip connections to node features. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **x**: node features :math:`(|\mathcal{V}|, F_{in})`
        - **pos**: node positions :math:`(|\mathcal{V}|, pos_{dim})`
        - **edge_index**: graph connectivity :math:`(2, |\mathcal{E}|)`
        - **edge_attr**: edge features :math:`(|\mathcal{E}|, F_{edge})` *(optional)*
        - **output**: tuple containing
            - updated node features :math:`(|\mathcal{V}|, F_{out})`
            - updated node positions :math:`(|\mathcal{V}|, pos_{dim})`
        """
    def __init__(self, 
                 nn_edge        : Callable,
                 nn_node        : Callable,
                 pos_dim        : int,
                 nn_pos         : Optional[Callable] = None,
                 skip_connection: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.nn_edge = nn_edge
        self.nn_node = nn_node
        self.nn_pos = nn_pos
        self.skip_connection = skip_connection
        self.pos_dim = pos_dim
        self.eps = 1e-8  # Epsilon for numerical stability
        self.clamp = True
        self.clamp_magnitude = 10.0
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        for nn in [self.nn_edge, self.nn_node]:
            reset(nn)
        if self.nn_pos is not None:
            reset(self.nn_pos)

    def forward(self,
                x          : Tensor,
                pos        : Tensor,
                edge_index : Adj, 
                edge_attr  : Optional[Tensor] = None,
                size       : Size = None) -> Tuple[Tensor, Tensor]:

        assert x.size(0) == pos.size(0), "x and pos must have same number of nodes"
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1), "edge_attr must match number of edges"
        assert self.skip_connection is False or x.size(-1) == self.nn_node[-1].out_features, "For skip connection, input and output node feature dimensions must match"

        #TODO: compute weighted message
            # Check for NaN in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf in x input to EGNNConv")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("NaN/Inf in pos input to EGNNConv")
            pos = torch.nan_to_num(pos, nan=0.0, posinf=10.0, neginf=-10.0)

        # Perform message passing
        message = self.propagate(edge_index, x=(x, x), pos=(pos, pos), edge_attr=edge_attr, size=size)
        
        if torch.isnan(message).any() or torch.isinf(message).any():
            print("NaN/Inf in message from propagate")
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
        if self.nn_pos is not None:
            (message_node, message_pos) = torch.split(message, [message.size(-1) - self.pos_dim, self.pos_dim], dim=-1)
                
            # Clamp position updates to prevent explosion
            if self.clamp:
                message_pos = torch.clamp(message_pos, min=-1.0, max=1.0)
            
            out_pos = pos + message_pos
        else:
            message_node = message
            out_pos = pos
        
        node_input = torch.cat([x, message_node], dim=-1)
        
        # Clamp before passing through neural network
        if self.clamp:
            node_input = torch.clamp(node_input, min=-10.0, max=10.0)
        
        out_node = self.nn_node(node_input)
        
        # Check output for NaN
        if torch.isnan(out_node).any() or torch.isinf(out_node).any():
            print("NaN/Inf in nn_node output")
            out_node = torch.nan_to_num(out_node, nan=0.0, posinf=1.0, neginf=-1.0)
        

        if self.skip_connection:
            out_node = out_node + x

         # Final clamp
        if self.clamp:
            out_node = torch.clamp(out_node, min=-10.0, max=10.0)
            out_pos = torch.clamp(out_pos, min=-100.0, max=100.0)

        return (out_node, out_pos)

    def message(self, 
                x_i      : Tensor, 
                x_j      : Tensor, 
                pos_i    : Tensor, 
                pos_j    : Tensor, 
                edge_attr: Optional[Tensor] = None) -> Tensor:

        pos_diff = pos_i - pos_j
        square_norm = torch.sum(pos_diff ** 2, dim=-1, keepdim=True)
        
        if self.clamp:
            # Clamp square_norm to prevent extreme values
            square_norm = torch.clamp(square_norm, min=self.eps, max=1000.0)
        
            # Normalize square_norm for better numerical stability
            # Use log scale for very large distances
            square_norm = torch.log(square_norm + 1.0)

        if edge_attr is not None:
            if self.clamp:
                edge_attr = torch.clamp(edge_attr, min=-10.0, max=10.0)
            out = torch.cat([x_j, x_i, square_norm, edge_attr], dim=-1)
        else:
            out = torch.cat([x_j, x_i, square_norm], dim=-1)

        if self.clamp:
            out = torch.clamp(out, min=-10.0, max=10.0)

        out = self.nn_edge(out)

        # Check for NaN after nn_edge
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf in nn_edge output")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.nn_pos is not None:
            out_pos_j = self.nn_pos(out)

            # Check for NaN in position update
            if torch.isnan(out_pos_j).any() or torch.isinf(out_pos_j).any():
                print("NaN/Inf in nn_pos output")
                out_pos_j = torch.nan_to_num(out_pos_j, nan=0.0, posinf=0.1, neginf=-0.1)
            
            # Normalize position difference with better stability
            dist = torch.sqrt(square_norm + self.eps)
            pos_diff_normalized = pos_diff / (dist + self.eps)

            if self.clamp:
                pos_diff_normalized = torch.clamp(pos_diff_normalized, min=-1.0, max=1.0)
                out_pos_j = torch.clamp(out_pos_j, min=-1.0, max=1.0)
 
            message_pos_j = pos_diff_normalized * out_pos_j
            out = torch.cat([out, message_pos_j], dim=-1)
 
        return out
    
    def aggregate(self, 
                  inputs  : Tensor, 
                  index   : Tensor, 
                  dim_size: Optional[int] = None) -> Tensor:
        
        out_pos = None
        if self.nn_pos is not None:
            (message_node, message_pos) = torch.split(inputs, [inputs.size(-1) - self.pos_dim, self.pos_dim], dim=-1)
            out_pos = torch_scatter.scatter_mean(message_pos, index, dim=0, dim_size=dim_size)
        else:
            message_node = inputs
        
        out = torch_scatter.scatter_add(message_node, index, dim=0, dim_size=dim_size)

        if out_pos is not None:
            out = torch.cat([out, out_pos], dim=-1)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_edge={self.nn_edge}, nn_node={self.nn_node}, nn_pos={self.nn_pos if self.nn_pos is not None else "None"})'
    
