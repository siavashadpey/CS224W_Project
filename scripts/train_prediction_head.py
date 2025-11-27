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
from torch_geometric.loader import DataLoader

seed_everything(1313)

from models.bimolecular_affinity_models import RegressionHead, Encoder
from utils.checkpoint_utils import save_checkpoint
from utils.gcs_dataset_loader import GCSPyGDataset

# Set up logger for this module
logger = logging.getLogger(__name__)

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
        predictions = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
        
        targets = batch.y.view(-1, 1).float()
        loss = loss_fn(predictions, targets)
        
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
            predictions = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
            targets = batch.y.view(-1, 1).float()
            total_error += eval_fn(predictions, targets)

    avg_error = total_error / len(data_loader)
    return avg_error

def main():
    parser = argparse.ArgumentParser(description="Train regression head for binding affinity prediction.")
    
    # dataset 
    parser.add_argument('--train_data_path', type=str, required=True, help='Local path to training data.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Local path to test data.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    
    # Model args
    parser.add_argument('--pretrained_checkpoint', type=str, default=None, 
                        help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='Freeze encoder weights during training')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels in encoder')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--skip_connection', action='store_true', help='Use skip connections in encoder')
    
    # Training args
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    
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
    
    # load data
    trainval_dataset = GCSPyGDataset(root="", file_paths=[args.train_data_path])
    test_dataset = GCSPyGDataset(root="", file_paths=[args.test_data_path])

    # split trainval dataset into train and val sets (80-20 split)
    val_size = int(0.2 * len(trainval_dataset))
    train_size = len(trainval_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    node_features = trainval_dataset.node_features
    edge_features_dim = trainval_dataset.edge_features_dim
    
    # build model
    encoder = Encoder(
        in_channels=node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_encoder_layers,
        edge_dim=edge_features_dim,
        pos_dim=3,
        act="SiLU",
        skip_connection=args.skip_connection
    )
    
    # load pretrained weights if we have them
    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint)
        
        # grab just the encoder part from the autoencoder checkpoint
        encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
                new_key = key[8:]  # remove 'encoder.' prefix
                encoder_state_dict[new_key] = value
        
        encoder.load_state_dict(encoder_state_dict)
        logger.info(f"Loaded model from {args.pretrained_checkpoint}")
    
    model = RegressionHead(
        encoder=encoder,
        hidden_channels=args.hidden_channels,
        freeze_encoder=args.freeze_encoder
    )
    
    model = model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_initial = 0
    
    os.makedirs(args.model_save_path, exist_ok=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    
    # train loop
    for epoch in range(epoch_initial, args.num_epochs):
        train_loss = train_one_epoch(train_loader, model, optim, nn.MSELoss(), device)
        
        if epoch % args.checkpoint_interval == 0:
            val_loss = eval(val_loader, model, nn.MSELoss(), device)
            test_loss = eval(test_loader, model, nn.MSELoss(), device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                print(f"New best model found at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            
            save_checkpoint(
                model,
                optim,
                epoch,
                train_loss,
                val_loss,
                test_loss,
                f"{args.model_save_path}/checkpoint_epoch_{epoch}.pt"
            )
            logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    if best_model_state is not None:
        torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
        logger.info(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")

if __name__ == '__main__':
    main()