import argparse
import os
import logging
from typing import Callable
import sys
from pathlib import Path

# Add project root to sys.path so modules like 'models' and 'utils' can be found
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch import nn, torch
from torch_geometric import seed_everything
from torch_geometric.data import Data as PyGData


seed_everything(1313)

from models.bimolecular_affinity_models import MaskedGeometricAutoencoder, Encoder, Decoder
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.losses import l2_loss
from utils.gcs_dataset_loader import create_gcs_dataloaders

# Set up logger for this module
logger = logging.getLogger(__name__)

"""
TODO:
 - Prediction head model (models/bimolecular_affinity_models.py)
 - Prediction head training script (scripts/train_prediction_head.py)
"""

def train_one_epoch(data_loader, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: Callable,
                    device: torch.device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        batch_indices = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
        loss = loss_fn(predicted_pos, batch.pos[mask_indices])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval(data_loader,
         model: nn.Module, 
         eval_fn: Callable,
         device: torch.device):
    """
    Evaluate the model on a dataset.
    """
    total_error = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            batch_indices = batch.batch if hasattr(batch, 'batch') else  torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
            total_error += eval_fn(predicted_pos, batch.pos[mask_indices])

    avg_error = total_error / len(data_loader)
    return avg_error 

def main():
    parser = argparse.ArgumentParser(description="Main script for the project.")
    
    # GCS arguments
    parser.add_argument('--gcs_bucket', type=str, required=True, help='GCS bucket name')
    parser.add_argument('--train_prefix', type=str, required=True, help='GCS prefix for training data')
    parser.add_argument('--val_prefix', type=str, required=True, help='GCS prefix for validation data')
    parser.add_argument('--test_prefix', type=str, required=True, help='GCS prefix for test data')
    parser.add_argument('--cache_dir', type=str, default='/tmp/pyg_cache', help='Local cache directory')

    # Model save/load
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model.')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    
    # Model architecture
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers in the model.')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers in the model.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for the model.')
    parser.add_argument('--masking_ratio', type=float, default=0.45, help='Ratio of masking for input data.')
    
    # Checkpointing
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Periodically save checkpoints each number of epochs.')

    args = parser.parse_args()

    # --- Initialize Logging ---
    # setup logging for Vertex AI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # --------------------------

    os.makedirs(args.model_save_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load training, testing, and validation data from GCS
    logger.info("Loading datasets from GCS...")
    train_loader, val_loader, test_loader, train_dataset = create_gcs_dataloaders(
        bucket_name=args.gcs_bucket,
        train_prefix=args.train_prefix,
        val_prefix=args.val_prefix,
        test_prefix=args.test_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_cache_dir=args.cache_dir,
        force_download=False
    )

    # Model
    encoder = Encoder(
             in_channels=train_dataset.node_features, 
             hidden_channels=args.hidden_dim, 
             num_layers=args.num_encoder_layers,
             pos_dim=3,
             edge_dim=train_dataset.edge_features_dim,
             skip_connection=True)
    decoder = Decoder(
             in_channels=args.hidden_dim,
             hidden_channels=args.hidden_dim,
             num_layers=args.num_decoder_layers,
             pos_dim=3,
             edge_dim=train_dataset.edge_features_dim,
             skip_connection=True)
    model = MaskedGeometricAutoencoder(
             encoder=encoder, 
             decoder=decoder,
             masking_ratio=args.masking_ratio)
    
    for m in [encoder, decoder, model]:
        m.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_initial = 0

    # load model and optim checkpoints if specified
    if args.load_model_path:
        epoch_initial, checkpoint = load_checkpoint(model, optim, args.load_model_path)
        logger.info(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        logger.info(f'Last epoch training loss: {checkpoint["train_loss"]}, test loss: {checkpoint["test_loss"]}')

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    # Training loop
    for epoch in range(epoch_initial, args.num_epochs):
        train_loss = train_one_epoch(train_loader,
                                     model,
                                     optim,
                                     l2_loss,
                                     device)

        if epoch % args.checkpoint_interval == 0:
            val_loss = eval(val_loader,
                            model,
                            l2_loss,
                            device)
            test_loss = eval(test_loader,
                            model,
                            l2_loss,
                            device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                print(f"New best model found at epoch {best_epoch} with val loss {best_val_loss:.4f}")

            save_checkpoint(model,
                            optim,
                            epoch,
                            train_loss,
                            val_loss,
                            test_loss,
                            f"{args.model_save_path}/checkpoint_epoch_{epoch}.pt")
            logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
        logger.info(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()