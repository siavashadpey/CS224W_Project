import argparse
import os
import logging
from typing import Callable
import sys
import math

from torch import nn, torch
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from models.bimolecular_affinity_models import MaskedGeometricAutoencoder, Encoder, Decoder
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.losses import l2_loss
from utils.gcs_dataset_loader import GCSPyGDataset, create_gcs_dataloaders

seed_everything(1313)

logger = logging.getLogger(__name__)


def report_metric(metric_value, current_epoch):
    """Reports a metric to the Vertex AI hyperparameter tuning service.
    
    Args:
        metric_values (List[float]): List of metric values to report.
        metric_names (List[str]): List of metric names corresponding to the values.
        current_epoch (int): Current epoch number for global_step.
    """
    import hypertune

    is_valid_metric = (
        metric_value is not None and
        isinstance(metric_value, (int, float)) and
        not math.isnan(metric_value) and not math.isinf(metric_value))
    
    if is_valid_metric:
        hpt = hypertune.HyperTune()

        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=metric_value,
            global_step=current_epoch
        )
    else:
        logger.warning(f"Invalid metric value: {metric_value} at epoch {current_epoch}")
    
def is_vertex_ai_trial():
    """ Check if the code is running on a Vertex AI hyperparameter tuning trial."""
    return "CLOUD_ML_TRIAL_ID" in os.environ

def is_vertex_ai():
    """ Check if the code is running on Vertex AI."""
    return "AIP_MODEL_DIR" in os.environ

def train_one_epoch(data_loader, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    loss_fn: Callable,
                    device: torch.device):
    """
    Train the model for one epoch.

    Args:
        data_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loss_fn (Callable): Loss function to use.
        device (torch.device): Device to run the training on.
    """
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        loss = loss_fn(predicted_pos, batch.pos[mask_indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # better stability
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval(data_loader,
         model: nn.Module, 
         eval_fn: Callable,
         device: torch.device):
    """
    Evaluate the model on a dataset.

    Args:
        data_loader: DataLoader for the evaluation dataset.
        model (nn.Module): The model to evaluate.
        eval_fn (Callable): Evaluation function to apply.
        device (torch.device): Device to run the evaluation on.
    """
    pos_all = []
    predicted_pos_all = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
            pos_all.append(batch.pos[mask_indices])
            predicted_pos_all.append(predicted_pos)

    pos_all = torch.cat(pos_all, dim=0)
    predicted_pos_all = torch.cat(predicted_pos_all, dim=0)
    error = eval_fn(predicted_pos_all, pos_all)
    return error

def main():
    parser = argparse.ArgumentParser(description="Main script for the project.")
    
    # dataset 
    parser.add_argument('--train_data_path', type=str, required=True, help='Local path to training data.')
    parser.add_argument('--val_data_path', type=str, required=True, help='Local path to validation data.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Local path to test data.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')

    # Model save/load
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model.')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--learning_rate_gamma', type=float, default=0.999, help='Learning rate decay factor.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    
    # Model architecture
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers in the model.')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers in the model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size for the model.')
    parser.add_argument('--masking_ratio', type=float, default=0.45, help='Ratio of masking for input data.')
    
    # Checkpointing
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Periodically save checkpoints each number of epochs.')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(args.model_save_path, exist_ok=True)

    if is_vertex_ai():
        logger.info("Running on Vertex AI.")
    else:
        logger.info("Running locally.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load training, testing, and validation data
    logger.info("Loading datasets...")

    if is_vertex_ai():
        GCS_BUCKET = os.environ.get("GCS_BUCKET")
        print(GCS_BUCKET, args.train_data_path)
        train_loader, val_loader, test_loader, train_dataset = create_gcs_dataloaders(
            bucket_name=GCS_BUCKET,
            train_prefix=args.train_data_path,
            val_prefix=args.val_data_path,
            test_prefix=args.test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            local_cache_dir="/tmp/data_cache"
        )
    else:
        train_dataset = GCSPyGDataset(root="", file_paths=[args.train_data_path])
        val_dataset = GCSPyGDataset(root="", file_paths=[args.val_data_path])   
        test_dataset = GCSPyGDataset(root="", file_paths=[args.test_data_path])

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

    logger.info(f"Train loader size: {len(train_loader)}, {len(train_loader.dataset)} samples")
    logger.info(f"Val loader size: {len(val_loader)}, {len(val_loader.dataset)} samples")
    logger.info(f"Test loader size: {len(test_loader)}, {len(test_loader.dataset)} samples")
    logger.info(f"Example data in train dataset: {train_loader.dataset[0]}")

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
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in model: {num_params}")

    for m in [encoder, decoder, model]:
        m.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = ExponentialLR(optim, gamma=args.learning_rate_gamma)
    epoch_initial = 0

    # load model and optim checkpoints if specified
    if args.load_model_path:
        epoch_initial, checkpoint = load_checkpoint(model, optim, lr_scheduler, args.load_model_path)
        logger.info(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        logger.info(f'Last epoch training loss: {checkpoint["train_loss"]}, {checkpoint["val_loss"]}, test loss: {checkpoint["test_loss"]}')

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    if is_vertex_ai_trial():
        base_name_template = f"checkpoint_trial_{os.environ.get('CLOUD_ML_TRIAL_ID', 'default')}_epoch_{{}}.pt"
    else:
        base_name_template = "checkpoint_epoch_{}.pt"

    try:
        # Training loop
        for epoch in range(epoch_initial, args.num_epochs):
            train_loss = train_one_epoch(train_loader,
                                         model,
                                         optim,
                                         lr_scheduler,
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
                    logger.info(f"New best model found at epoch {best_epoch} with val loss {best_val_loss:.4f}")

                checkpoint_basename = base_name_template.format(epoch)
                save_checkpoint(model,
                                optim,
                                lr_scheduler,
                                epoch,
                                train_loss,
                                val_loss,
                                test_loss,
                                f"{args.model_save_path}/{checkpoint_basename}")
                logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

                if is_vertex_ai_trial():
                    report_metric(best_val_loss, epoch)

        if best_model_state is not None:
            torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
            logger.info(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        if is_vertex_ai_trial():
            sys.exit(0) # to allow hp tuning to continue
        else:
            sys.exit(1)
if __name__ == "__main__":
    main()