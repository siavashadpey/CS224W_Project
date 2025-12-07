import argparse
import os
import logging
from typing import Callable, List
import sys
from pathlib import Path

# Add project root to sys.path so modules like 'models' and 'utils' can be found
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch import nn, torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

seed_everything(1313)

from models.bimolecular_affinity_models import Encoder, LinearRegressionHead, MLPRegressionHead, EGNNRegressionHead
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.gcs_dataset_loader import GCSPyGDataset
from utils.eval import pearson_correlation_coefficient

# Set up logger for this module
logger = logging.getLogger(__name__)

"""
TODO:
    - more sophisticated heads with nonlinearity, e.g.:
       * MLPs-> Pool -> Linear (done)
       * 1 EGNN layer -> Pool -> Linear (done)
       * 1 EGNN layer  (supernode) -> Linear 
"""

def train_one_epoch(data_loader, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
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
        lr_scheduler.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval(data_loader,
         model: nn.Module, 
         eval_fns: List[Callable],
         device: torch.device):
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    total_evals = [0 for _ in eval_fns]
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            batch_indices = batch.batch if hasattr(batch, 'batch') else  torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            predictions = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
            targets = batch.y.view(-1, 1).float()
            for i, eval_fn in enumerate(eval_fns):
                total_evals[i] += eval_fn(predictions, targets)

    avg_evals = [total / len(data_loader) for total in total_evals]
    return avg_evals

def main():
    parser = argparse.ArgumentParser(description="Train regression head for binding affinity prediction.")
    
    # dataset 
    parser.add_argument('--train_data_path', type=str, required=True, help='Local path to training data.')
    parser.add_argument('--val_data_path', type=str, required=False, help='Local path to validation data.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Local path to test data.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    
    # Encoder-related args
    parser.add_argument('--pretrained_checkpoint', type=str, default=None, help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights during training')
    parser.add_argument('--encoder_hidden_channels', type=int, default=64, help='Hidden channels in encoder')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model.')
    
    # Regression head args 
    parser.add_argument('--pooling_method', type=str, default='global_mean_pool', help='Global pooling method: global_mean_pool, global_add_pool, global_max_pool.')
    parser.add_argument('--head_method', type=str, default='mlp', help='Type of regression head: linear, mlp, egnn.')
    parser.add_argument('--head_hidden_channels', type=int, default=64, help='Hidden channels in regression head (if applicable)')
    parser.add_argument('--head_num_layers', type=int, default=2, help='Number of layers in regression head (if applicable)')

    # Training args
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=.001, help='Learning rate')
    parser.add_argument('--learning_rate_gamma', type=float, default=0.999, help='Learning rate decay factor.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save checkpoints')
    
    args = parser.parse_args()

    assert(args.pretrained_checkpoint is not None or not args.freeze_encoder), \
    "Pretrained checkpoint path must be provided if encoder is frozen."

    assert(args.load_model_path is None or args.pretrained_checkpoint is None), \
    "Cannot load both a pretrained checkpoint and a model checkpoint."
    
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
    logger.info("Loading datasets...")

    # load data
    train_dataset = GCSPyGDataset(root="", file_paths=[args.train_data_path])
    val_dataset = GCSPyGDataset(root="", file_paths=[args.val_data_path])
    test_dataset = GCSPyGDataset(root="", file_paths=[args.test_data_path])

    logger.info(f"train dataset size: {len(train_dataset)}")
    logger.info(f"val dataset size: {len(val_dataset)}")
    logger.info(f"test dataset size: {len(test_dataset)}")

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
    
    node_features = train_dataset.node_features
    edge_features_dim = train_dataset.edge_features_dim
    
    # build model
    encoder = Encoder(
        in_channels=node_features,
        hidden_channels=args.encoder_hidden_channels,
        num_layers=args.num_encoder_layers,
        edge_dim=edge_features_dim,
        pos_dim=3,
        skip_connection=True)
    
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
    elif args.load_model_path:
        epoch_initial, checkpoint = load_checkpoint(model, optim, lr_scheduler, args.load_model_path)
        logger.info(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        logger.info(f'Last epoch training loss: {checkpoint["train_loss"]}, {checkpoint["val_loss"]}, test loss: {checkpoint["test_loss"]}')
    
    if args.freeze_encoder:
        logger.info("Freezing encoder parameters.")
    else:
        logger.info("Training encoder parameters.")
        
    if args.head_method == 'mlp':
        model = MLPRegressionHead(
            encoder=encoder,
            hidden_channels=args.head_hidden_channels,
            num_layers=args.head_num_layers,
            act="ReLU",
            global_pool=args.pooling_method)
    elif args.head_method == 'linear':
        model = LinearRegressionHead(encoder=encoder)
    elif args.head_method == 'egnn':
        model = EGNNRegressionHead(
            encoder=encoder,
            hidden_channels=args.head_hidden_channels,
            num_layers=args.head_num_layers,
            edge_dim=edge_features_dim,
            act="ReLU",
            skip_connection=False,
            global_pool=args.pooling_method)
    else:
        raise ValueError(f"Unknown head method: {args.head_method}")


    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in model: {num_params}")
    
    model = model.to(device)

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.learning_rate_gamma)
    epoch_initial = 0
    
    os.makedirs(args.model_save_path, exist_ok=True)
    
    best_val_eval = float('inf')
    best_model_state = None
    best_epoch = -1
    
    # train loop
    for epoch in range(epoch_initial, args.num_epochs):
        train_loss = train_one_epoch(train_loader, model, optim, lr_scheduler, nn.MSELoss(), device)
        
        if epoch % args.checkpoint_interval == 0:
            val_MLE, val_R  = eval(val_loader, model, [nn.MSELoss(), pearson_correlation_coefficient], device)
            test_MLE, test_R = eval(test_loader, model, [nn.MSELoss(), pearson_correlation_coefficient], device)
            
            if val_MLE < best_val_eval:
                best_val_eval = val_MLE
                best_model_state = model.state_dict()
                best_epoch = epoch
                logger.info(f"New best model found at epoch {best_epoch} with val MSE {best_val_eval:.4f} and Pearson Corr Coeff {val_R:.4f}")
            
            save_checkpoint(
                model,
                optim,
                lr_scheduler,
                epoch,
                train_loss,
                val_MLE,
                test_MLE,
                f"{args.model_save_path}/checkpoint_epoch_{epoch}.pt"
            )
            logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val MSE: {val_MLE:.4f},  Val R: {val_R:.4f}, Test MSE: {test_MLE:.4f}, Test R: {test_R:.4f}")
    
    if best_model_state is not None:
        torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
        logger.info(f"Best model saved from epoch {best_epoch} with val MSE {best_val_eval:.4f}")
if __name__ == '__main__':
    main()