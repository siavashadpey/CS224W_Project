import os 
from typing import Union, Callable, Tuple, Optional, List

from models import EGNN

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch_geometric.typing import Adj
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from google.cloud import storage

seed_everything(1021)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

'''
Implementation of Experimnent 5.2 in `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/abs/2102.09844>`_ paper
'''
class AE(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 nb_layers: int,
                 pos_dim: int,
                 edge_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.encoder_layers = torch.nn.ModuleList()
        self.encoder_layers.append(EGNN(in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=1,
                            pos_dim=pos_dim,
                            edge_dim=edge_dim,
                            update_pos=True,
                            skip_connection=False))
        if nb_layers > 1:
            self.encoder_layers.append(EGNN(in_channels=hidden_channels,
                                hidden_channels=hidden_channels,
                                num_layers=nb_layers-1,
                                pos_dim=pos_dim,
                                edge_dim=edge_dim,
                                update_pos=True,
                                skip_connection=True))

        self.a = nn.Parameter(torch.tensor([-1.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))


    def forward(self,
                edge_index: Adj,
                x: Tensor,
                edge_attr: Tensor) -> Tensor:
        nb_nodes = x.size(0)
        pos = torch.randn((nb_nodes, self.pos_dim)).to(device) * 0.1

        for layer in self.encoder_layers:
            x, pos = layer(x, pos, edge_index, edge_attr)

        dist_sqr = torch.cdist(pos, pos, p=2)**2  # Squared distances
        dist_sqr = self.a* dist_sqr + self.b
        diagonal_mask = torch.eye(nb_nodes, device=device, dtype=torch.bool)
        dist_sqr = dist_sqr.masked_fill(diagonal_mask,  -1e10) # remove self-loops
        dist_sqr = dist_sqr
        return dist_sqr 

def GenerateDataset(num_graphs_per_node: int,
                    nodes_per_graph: List[int],
                    prob_edge: float) -> List[Data]:
    dataset = []
    for n_nodes in nodes_per_graph:
        for _ in range(num_graphs_per_node):
            edge_index = erdos_renyi_graph(n_nodes, prob_edge)
            edge_index, _ = remove_self_loops(edge_index)
            x = torch.ones((n_nodes, 1), dtype=torch.float)
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)
            A = to_dense_adj(edge_index, max_num_nodes=n_nodes)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, A=A)
            dataset.append(data)

    return dataset


def get_dense_edge_index_and_attr(edge_index: Tensor, nb_nodes: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    full_edge_index = []
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if i != j:
                full_edge_index.append((i, j))
    edge_attr = torch.zeros((nb_nodes**2-nb_nodes, 1), dtype=torch.float32).to(device)
    full_edge_index = torch.tensor(full_edge_index, dtype=torch.long).t().to(device) 
    matches = (full_edge_index.unsqueeze(2) == edge_index.unsqueeze(1)).all(dim=0)
    edge_mask = matches.any(dim=1)
    edge_attr[edge_mask, 0] = 1.0
    return full_edge_index, edge_attr


def train(epoch: int,
          model: Callable,
          loss_fn: Callable,
          optimizer: Callable,
          train_loader: DataLoader):
    total_loss = 0
    total_adj_error = 0
    total_wrong_edges = 0
    total_edges = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    model.train()
    for batch in train_loader:
        g = batch.to(device)
        optimizer.zero_grad()
        nb_nodes = g[0].num_nodes 
        # Construct a fully connected graph as per paper with the actual edges indicated by edge_attr
        edge_index, edge_attr = get_dense_edge_index_and_attr(g.edge_index, nb_nodes, device)
        logits = model(edge_index, g.x, edge_attr)
        logits_all = torch.empty((nb_nodes**2*batch.num_graphs,1), device=device)
        for i in range(batch.num_graphs):
            # Extract the diagonal block for graph i
            logits_i = logits[i*nb_nodes:(i+1)*nb_nodes, i*nb_nodes:(i+1)*nb_nodes].reshape(-1,1)
            logits_all[i*nb_nodes**2:(i+1)*nb_nodes**2,:]  = logits_i

        loss = loss_fn(logits_all, g.A)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            adj_error, wrong_edges = adjacency_error(logits_all, g.A.view(-1,1), nb_nodes, batch.num_graphs)
            tp, fp, fn = tp_fp_fn(logits_all, g.A.view(-1,1))
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_adj_error += adj_error
            total_wrong_edges += wrong_edges
            total_edges += (nb_nodes**2 - nb_nodes)*batch.num_graphs

    return {
        "total_loss": total_loss / len(train_loader),
        "adj_error": total_adj_error / len(train_loader),
        "f1": f1(total_tp, total_fp, total_fn),
        "wrong_edges": total_wrong_edges,
        "edges": total_edges
    }

def test(model: Callable,
         loss_fn: Callable,
         test_loader: DataLoader):
    total_loss = 0
    total_adj_error = 0
    total_wrong_edges = 0
    total_edges = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            g = batch.to(device)
            nb_nodes = g[0].num_nodes
            edge_index, edge_attr = get_dense_edge_index_and_attr(g.edge_index, nb_nodes, device)
            logits = model(edge_index, g.x, edge_attr)

            logits_all = torch.empty((nb_nodes**2*batch.num_graphs,1), device=device)
            for i in range(batch.num_graphs):
                # Extract the diagonal block for graph i
                logits_i = logits[i*nb_nodes:(i+1)*nb_nodes, i*nb_nodes:(i+1)*nb_nodes].reshape(-1,1)
                logits_all[i*nb_nodes**2:(i+1)*nb_nodes**2,:]  = logits_i

            loss = loss_fn(logits_all, g.A)
            total_loss += loss.item()
            adj_error, wrong_edges = adjacency_error(logits_all, g.A.view(-1,1), nb_nodes, batch.num_graphs)
            tp, fp, fn = tp_fp_fn(logits_all, g.A.view(-1,1))
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_adj_error += adj_error
            total_wrong_edges += wrong_edges
            total_edges += (nb_nodes**2 - nb_nodes)*batch.num_graphs

    return {
        "total_loss": total_loss / len(test_loader),
        "adj_error": total_adj_error / len(test_loader),
        "f1": f1(total_tp, total_fp, total_fn),
        "wrong_edges": total_wrong_edges,
        "edges": total_edges
    }

def adjacency_error(logits, adj_gt, n_nodes, n_graphs):
    adj_pred = (F.sigmoid(logits) > 0.5).type(torch.float32)
    adj_errors = torch.abs(adj_pred - adj_gt)
    wrong_edges = torch.sum(adj_errors)/n_graphs
    adj_error = wrong_edges/ (n_nodes ** 2 - n_nodes)
    return adj_error.item(), wrong_edges.item()

def tp_fp_fn(logits, adj_gt):
        adj_pred_binary = (F.sigmoid(logits) > 0.5).type(torch.float32)
        tp = torch.sum(adj_pred_binary * adj_gt).item()
        fp = torch.sum(adj_pred_binary * (1 - adj_gt)).item()
        fn = torch.sum((1 - adj_pred_binary) * adj_gt).item()
        return tp, fp, fn

def f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1_score

CHECKPOINT_DIR = os.environ.get("AIP_MODEL_DIR", "./checkpoints")
GCS_BUCKET = os.environ.get("GCS_BUCKET", None)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, train_results, test_results, file_path):
    """Save checkpoint locally or to GCS depending on path."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_results['total_loss'],
        'test_loss': test_results['total_loss'],
        'train_adj_error': train_results['adj_error'],
        'test_adj_error': test_results['adj_error'],
        'train_f1': train_results['f1'],
        'test_f1': test_results['f1']
    }

    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved locally: {file_path}")

    if GCS_BUCKET:
        try:
            gcs_path = f"checkpoints/{os.path.basename(file_path)}"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(file_path)
            print(f"Checkpoint uploaded to GCS: gs://{GCS_BUCKET}/{gcs_path}")
        except Exception as e:
            print(f"Failed to upload checkpoint to GCS: {e}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, checkpoint


RESUME_FROM = None  # Set to checkpoint path to resume, e.g., "checkpoints/checkpoint_epoch_50.pt"

if __name__ == "__main__":

    # Graph parameters
    NODES_PER_EDGE = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    PROB_EDGE = 0.25
    BATCH_SIZE = 128
    NUM_GRAPHS_PER_NODE_TRAIN = 5
    NUM_GRAPHS_PER_NODE_TEST = 5
    POS_DIM = 8
    EDGE_DIM = 1

    # Model parameters
    IN_CHANNELS = 1
    HIDDEN_CHANNELS = 64
    NB_LAYERS = 4

    # Optimization/loss parameters
    LEARNING_RATE = 1E-4
    NB_EPOCHS = 5

    # Prepare dataloaders
    train_dataset = GenerateDataset(num_graphs_per_node=NUM_GRAPHS_PER_NODE_TRAIN,
                                    nodes_per_graph=NODES_PER_EDGE,
                                    prob_edge=PROB_EDGE)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    test_dataset = GenerateDataset(num_graphs_per_node=NUM_GRAPHS_PER_NODE_TEST,
                                   nodes_per_graph=NODES_PER_EDGE,
                                   prob_edge=PROB_EDGE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Prepare model
    model = AE(in_channels=IN_CHANNELS,
               hidden_channels=HIDDEN_CHANNELS,
               nb_layers=NB_LAYERS,
               pos_dim=POS_DIM,
               edge_dim=EDGE_DIM).to(device) 

    # Prepare optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weight = torch.tensor([1.0 - PROB_EDGE], device=device)
    loss_fn = lambda logits, A: F.binary_cross_entropy_with_logits(logits.view(-1,1), A.view(-1,1), weight=weight, reduction='mean')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        start_epoch, checkpoint_data = load_checkpoint(model, optimizer, RESUME_FROM)
        print(f"Resumed from epoch {start_epoch}")
        print(f"Previous test loss: {checkpoint_data['test_loss']:.4f}")
    else:
        start_epoch = 0

    # Training loop
    best_test_loss = float('inf')
    for epoch in range(NB_EPOCHS):
        train_results = train(epoch=epoch, 
                     model=model, 
                     loss_fn=loss_fn, 
                     optimizer=optimizer, 
                     train_loader=train_loader)
        
        if epoch % 1 == 0:

            test_results = test(model=model,
                             loss_fn=loss_fn,
                             test_loader=test_loader)
            print(f"Epoch {epoch}/{NB_EPOCHS}, Train Loss: {train_results['total_loss']:.4f},  Test Loss: {test_results['total_loss']:.4f}" +
                f"   Train Adj Error: {train_results['adj_error']:.4f},  Test Adj Error: {test_results['adj_error']:.4f}" +
                f"   Train F1: {train_results['f1']:.4f},  Test F1: {test_results['f1']:.4f}" +
                f"   Train Wrong Edges (%): {train_results['wrong_edges'] / train_results['edges'] * 100:.2f},  Test Wrong Edges (%): {test_results['wrong_edges'] / test_results['edges'] * 100:.2f}")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, train_results, test_results, checkpoint_path)

        # Save best model
        if test_results['total_loss'] < best_test_loss:
            best_test_loss = test_results['total_loss']
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model.pt")
            save_checkpoint(model, optimizer, epoch, train_results, test_results, checkpoint_path)
            print(f"New best model saved at epoch {epoch} with test loss {best_test_loss:.4f}")
    
    print("Training completed.")
    print(f"Best test loss: {best_test_loss:.4f}")
    final_path = os.path.join(CHECKPOINT_DIR, f"final_model.pt")
    save_checkpoint(model, optimizer, NB_EPOCHS, train_results, test_results, final_path)