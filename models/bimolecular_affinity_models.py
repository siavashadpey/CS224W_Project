from typing import Union, Callable, Tuple

import torch
from torch import nn, Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import subgraph

from models.egnn import EGNN

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int, 
                 num_layers: int, 
                 edge_dim: int, 
                 act: Union['str', Callable] = "SiLU",
                 skip_connection: bool = False):
        """
         Encoder using EGNN layers.
         Args:
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden node features.
            num_layers (int): Number of EGNN layers.
            edge_dim (int): Dimension of edge features.
            act (Union['str', Callable]): Activation function to use. (default: "SiLU")
            skip_connection (bool): Whether to use skip connections. (default: False)
        """
        super(Encoder, self).__init__()
        update_pos = False  # Encoder does not update positions
        if skip_connection:
            # GNN0 is a single layer EGNN without skip connection to change feature dimension
            # GNN1 is a multi-layer EGNN with skip connections
            GNN0 = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=1,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=False)
            GNN1 = EGNN(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers-1,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=True)
            self.model = nn.Sequential(GNN0, GNN1)
        else:
            self.model = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=False)
            
    def forward(self,
                x  : torch.Tensor,
                pos: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, pos, edge_index, edge_attr)

class Decoder(nn.Module):
    """
    Decoder using EGNN layers to reconstruct node positions.
    Args:
         in_channels (int): Number of input node features.
         hidden_channels (int): Number of hidden node features.
         num_layers (int): Number of EGNN layers.
         edge_dim (int): Dimension of edge features.
         act (Union['str', Callable]): Activation function to use. (default: "SiLU")
         skip_connection (bool): Whether to use skip connections. (default: False)
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 edge_dim: int,
                 act: Union['str', Callable] = "SiLU",
                 skip_connection: bool = False):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        if skip_connection and in_channels != hidden_channels:
            EGNN0 = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=1,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=False)
            EGNN1 = EGNN(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers-1,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=True)
            self.model = nn.Sequential(EGNN0, EGNN1)
        else:
            self.model = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                pos_dim=3,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=skip_connection)

    def forward(self,
                x  : torch.Tensor,
                pos: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x, pos = self.model(x, pos, edge_index, edge_attr)
        return pos

class MaskedGeometricAutoencoder(nn.Module):
    """
    Masked geometric autoencoder.
    
    Args:
        encoder (nn.Module): Encoder model.
        decoder (nn.Module): Decoder model.
        masking_ratio (float): Ratio of masking to apply on the input data.
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 masking_ratio: float):
        super(MaskedGeometricAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking_ratio = masking_ratio
        self.masked_node_token = nn.Parameter(torch.randn(1, decoder.in_channels))

    def forward(self,
                x  : Tensor,
                pos: Tensor,
                edge_index: Adj,
                edge_attr: Tensor) -> Tuple[Tensor, Tensor]:

        # Randomly mask nodes and their edges.
        num_nodes = x.size(0)
        num_masked = int(self.masking_ratio * num_nodes)
        node_indices_perm = torch.randperm(num_nodes, device=x.device)
        mask_indices, vis_indices = node_indices_perm[:num_masked], node_indices_perm[num_masked:]

        # Subgraph for visible nodes. Edges connected to masked nodes are removed.
        edge_index_v, edge_attr_v = subgraph(vis_indices, edge_index, edge_attr)
        x_v = x[vis_indices]
        pos_v = pos[vis_indices]

        # Encode visible nodes
        x_v, pos_v = self.encoder(x_v, pos_v, edge_index_v, edge_attr_v)

        # Unlike the visible nodes, which use the encoded features and positions,
        # the masked nodes share the same learnable token
        # and their positions are initialized with random noise 
        z_m = self.masked_node_token.repeat(num_masked, 1).to(x.device)
        pos_m = torch.randn((num_masked, pos.size(1)), device=pos.device)

        # Combine visible and masked nodes
        z = torch.empty((num_nodes, x_v.size(1)), device=x.device)
        z[vis_indices,:] = x_v
        z[mask_indices,:] = z_m
        pos[vis_indices,:] = pos_v
        pos[mask_indices,:] = pos_m

        # Decode to reconstruct masked node positions
        pos = self.decoder(z, pos, edge_index, edge_attr)

        return pos[mask_indices,:], mask_indices
