import argparse
import os
import logging
from typing import Callable, List
import sys
import sys 
import math

from torch import nn, torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from models.bimolecular_affinity_models import Encoder, LinearRegressionHead, MLPRegressionHead, EGNNRegressionHead, EGNNMLPRegressionHead
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, download_blob, save_csv_file
from utils.gcs_dataset_loader import GCSPyGDataset, create_gcs_dataloaders
from utils.eval import pearson_correlation_coefficient, scaled_pK_rmse

seed_everything(1313)

logger = logging.getLogger(__name__)

def report_metric(metric_values, metric_names, current_epoch):
    """Reports a metric to the Vertex AI hyperparameter tuning service.
    
    Args:
        metric_values (List[float]): List of metric values to report.
        metric_names (List[str]): List of metric names corresponding to the values.
        current_epoch (int): Current epoch number for global_step.
    """
    import hypertune
    for metric_value, metric_name in zip(metric_values, metric_names):
        is_valid_metric = (
            metric_value is not None and
            isinstance(metric_value, (int, float)) and
            not math.isnan(metric_value) and not math.isinf(metric_value))

        if is_valid_metric:
            hpt = hypertune.HyperTune()

            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=metric_name, 
                metric_value=metric_value,
                global_step=current_epoch
            )
        else:
            logger.warning(f"Invalid metric value for {metric_name}: {metric_value}")

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
        data_loader: DataLoader for the training dataset.
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

    Args:
        data_loader: DataLoader for the evaluation dataset.
        model (nn.Module): The model to evaluate.
        eval_fns (List[Callable]): List of evaluation functions to apply.
        device (torch.device): Device to run the evaluation on.
    """
    targets_all = []
    predictions_all = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            batch_indices = batch.batch if hasattr(batch, 'batch') else  torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            predictions = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
            targets = batch.y.view(-1, 1).float()
            predictions_all.append(predictions)
            targets_all.append(targets)

    predictions_all = torch.cat(predictions_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    evals = [eval_fn(predictions_all, targets_all) for eval_fn in eval_fns]
    return evals        
    

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
    parser.add_argument('--head_method', type=str, default='mlp', help='Type of regression head: linear, mlp, egnn, egnn_mlp.')
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
        train_loader, val_loader, test_loader, train_dataset= create_gcs_dataloaders(
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
        pretrained_checkpoint_path = download_blob(args.pretrained_checkpoint)
        checkpoint = torch.load(pretrained_checkpoint_path)
        
        # grab just the encoder part from the autoencoder checkpoint
        encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
                new_key = key[8:]  # remove 'encoder.' prefix
                encoder_state_dict[new_key] = value
        
        encoder.load_state_dict(encoder_state_dict)
        logger.info(f"Loaded model from {pretrained_checkpoint_path}")

    if args.freeze_encoder:
        logger.info("Freezing encoder parameters.")
    else:
        logger.info("Training encoder parameters.")
        
    # Build the regression head
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
    elif args.head_method == 'egnn_mlp':
        model = EGNNMLPRegressionHead(
            encoder=encoder,
            gnn_hidden_channels=args.head_hidden_channels,
            gnn_num_layers=args.head_num_layers,
            edge_dim=edge_features_dim,
            mlp_hidden_channels=args.head_hidden_channels,
            mlp_num_layers=args.head_num_layers,
            gnn_act="SiLU",
            global_pool=args.pooling_method,
            mlp_act="ReLU")
    else:
        raise ValueError(f"Unknown head method: {args.head_method}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in model: {num_params}")
    
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.learning_rate_gamma)
    epoch_initial = 0

    # Load the model, if continuing a training run
    if args.load_model_path:
        epoch_initial, checkpoint = load_checkpoint(model, optim, lr_scheduler, args.load_model_path)
        logger.info(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        logger.info(f'Last epoch training loss: {checkpoint["train_loss"]}, val loss: {checkpoint["val_loss"]}, test loss: {checkpoint["test_loss"]}')

        val_MSE, val_R  = eval(val_loader, model, [scaled_pK_rmse, pearson_correlation_coefficient], device)
        test_MSE, test_R = eval(test_loader, model, [scaled_pK_rmse, pearson_correlation_coefficient], device) 
        logger.info(f'Post-load validation MSE: {val_MSE}, R: {val_R}')
        logger.info(f'Post-load test MSE: {test_MSE}, R: {test_R}')

    # Freeze encoder if specified
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    
    best_val_eval = float('inf')
    best_model_state = None
    best_epoch = -1

    if is_vertex_ai_trial():
        base_name_template = f"checkpoint_trial_{os.environ.get('CLOUD_ML_TRIAL_ID', 'default')}_epoch_{{}}.pt"
    else:
        base_name_template = "checkpoint_epoch_{}.pt"

    epoch_log = []
    try:
        # train loop
        for epoch in range(epoch_initial, args.num_epochs):
            train_loss = train_one_epoch(train_loader, model, optim, lr_scheduler, nn.MSELoss(), device)

            if epoch % args.checkpoint_interval == 0:
                val_MSE, val_R  = eval(val_loader, model, [scaled_pK_rmse, pearson_correlation_coefficient], device)
                test_MSE, test_R = eval(test_loader, model, [scaled_pK_rmse, pearson_correlation_coefficient], device)

                epoch_log.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_MSE": val_MSE.cpu().item(),
                    "val_R": val_R.cpu().item(),
                    "test_MSE": test_MSE.cpu().item(),
                    "test_R": test_R.cpu().item()
                })

                if val_MSE < best_val_eval:
                    best_val_eval = val_MSE
                    best_model_state = model.state_dict()
                    best_epoch = epoch
                    logger.info(f"New best model found at epoch {best_epoch} with val MSE {best_val_eval:.4f} and Pearson Corr Coeff {val_R:.4f}")

                checkpoint_basename = base_name_template.format(epoch)
                save_checkpoint(
                    model,
                    optim,
                    lr_scheduler,
                    epoch,
                    train_loss,
                    val_MSE,
                    test_MSE,
                    f"{args.model_save_path}/{checkpoint_basename}"
                )
                logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val MSE: {val_MSE:.4f},  Val R: {val_R:.4f}, Test MSE: {test_MSE:.4f}, Test R: {test_R:.4f}")

                if is_vertex_ai_trial():
                    report_metric([val_MSE, val_R], ['mse', 'pearson_coeff'], epoch)
    
        if best_model_state is not None:
            torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
            logger.info(f"Best model saved from epoch {best_epoch} with val MSE {best_val_eval:.4f}")
    
        csv_path = os.path.join(args.model_save_path, "epoch_metrics.csv")
        save_csv_file(epoch_log, csv_path)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        if is_vertex_ai_trial():
            sys.exit(0) # to allow hp tuning to continue
        else:
            sys.exit(1)
if __name__ == '__main__':
    main()