from typing import Union, Callable, Tuple, List

import torch
from torch import nn, Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import subgraph, to_dense_batch
from torch_geometric.data import Data 

from models.egnn import EGNN

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int, 
                 num_layers: int, 
                 edge_dim: int, 
                 pos_dim: int,
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
        self.skip_connection = skip_connection
        self.hidden_channels = hidden_channels

        if skip_connection:
            # GNN0 is a single layer EGNN without skip connection to change feature dimension
            # GNN1 is a multi-layer EGNN with skip connections
            self.GNN0 = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=1,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=False)
            self.GNN1 = EGNN(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers-1,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=True)
        else:
            # single EGNN module
            self.EGNN_all = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=update_pos,
                act=act,
                skip_connection=False)
            
    def forward(self,
                x  : torch.Tensor,
                pos: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # need to explicitly pass params into EGNNs 
        if self.skip_connection:
            x, pos = self.GNN0(x, pos, edge_index, edge_attr)
            x, pos = self.GNN1(x, pos, edge_index, edge_attr)
        else: 
            x, pos = self.EGNN_all(x, pos, edge_index, edge_attr)
        
        return x, pos

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
                 pos_dim: int,
                 act: Union['str', Callable] = "SiLU",
                 skip_connection: bool = False):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.skip_connection = skip_connection

        if skip_connection and in_channels != hidden_channels:
            self.EGNN0 = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=1,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=False)
            self.EGNN1 = EGNN(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers-1,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=True)
        else:
            self.EGNN_all = EGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                pos_dim=pos_dim,
                edge_dim=edge_dim,
                update_pos=True,
                act=act,
                skip_connection=skip_connection)

    def forward(self,
                x  : torch.Tensor,
                pos: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor) -> torch.Tensor:
        if self.skip_connection and self.in_channels != self.hidden_channels: 
            x, pos = self.EGNN0(x, pos, edge_index, edge_attr)
            x, pos = self.EGNN1(x, pos, edge_index, edge_attr)
        else:
            x, pos = self.EGNN_all(x, pos, edge_index, edge_attr)

        # return updated position tensor
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
        # Ensure masked token dimension matches encoder output dimension (hidden channels)
        self.masked_node_token = nn.Parameter(torch.randn(1, self.encoder.hidden_channels) * 0.01)

    def forward(self,
                x  : Tensor,
                pos: Tensor,
                edge_index: Adj,
                edge_attr: Tensor, 
                batch_indices: Tensor) -> Tuple[Tensor, Tensor]:

    
        # Randomly mask nodes and their edges.
        num_nodes = x.size(0)
        num_masked = int(self.masking_ratio * num_nodes)
        num_visible = num_nodes - num_masked

        # ensure masking is valid
        if num_visible < 1: 
            raise ValueError("Masking ratio is too high, resulting in zero or negative visible nodes")
        
        node_indices_perm = torch.randperm(num_nodes, device=x.device)
        mask_indices, vis_indices = node_indices_perm[:num_masked], node_indices_perm[num_masked:]

        # Subgraph for visible nodes. Edges connected to masked nodes are removed.
        edge_index_v, edge_attr_v = subgraph(
            vis_indices, 
            edge_index, 
            edge_attr,
            num_nodes=num_nodes, # the full number of nodes in the batch
            relabel_nodes=True  # relabels edges from 0 to num_visible-1
            )
        # sliced features and batch index for the visible nodes 
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
        pos_combined = torch.empty((num_nodes, pos.size(1)), device=pos.device)
        
        z[vis_indices,:] = x_v
        z[mask_indices,:] = z_m
        pos_combined[vis_indices,:] = pos_v
        pos_combined[mask_indices,:] = pos_m

        # Decode to reconstruct masked node positions
        pos_reconstructed = self.decoder(z, pos_combined, edge_index, edge_attr)

        return pos_reconstructed[mask_indices,:], mask_indices
