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

torch.autograd.set_detect_anomaly(True)

seed_everything(1313)

from models.bimolecular_affinity_models import MaskedGeometricAutoencoder, Encoder, Decoder
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.losses import l2_loss
from utils.gcs_dataset_loader import create_gcs_dataloaders

# Set up logger for this module
logger = logging.getLogger(__name__)

"""
TODO:
    - learning rate scheduler (weight decay)
"""

def train_one_epoch(data_loader, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: Callable,
                    device: torch.device,
                    epoch: int):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    valid_batch_count = 0

    # Tracking stats
    exploding_count = 0
    moderate_count = 0
    normal_count = 0
    skipped_count = 0
    nan_count = 0

    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        batch_indices = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long, device=device)

        try:
            predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
        
            # Check masking
            if len(mask_indices) == 0:
                skipped_count += 1
                logger.warning(f"Batch {batch_idx}: No nodes masked - skipping!")
                continue

            loss = loss_fn(predicted_pos, batch.pos[mask_indices])
            loss_value = loss.item()
            
            # loss thresholding
            if loss_value > 1000.0:  # Loss too high
                logger.warning(f"Batch {batch_idx}: Loss {loss_value:.4f} > 1000, Num nodes: {batch.x.shape[0]}, Num masked: {len(mask_indices)}, Predicted range: [{predicted_pos.min().item():.4f}, {predicted_pos.max().item():.4f}], Ground truth range: [{batch.pos[mask_indices].min().item():.4f}, {batch.pos[mask_indices].max().item():.4f}]")
                #skipped_count += 1
                #continue

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Batch {batch_idx}: Invalid loss {loss.item()} - skipping!")
                nan_count += 1
                skipped_count += 1
                continue

            loss.backward()

            # Check gradients BEFORE clipping
            total_grad_norm = 0.0
            max_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_norm)
            total_grad_norm = total_grad_norm ** 0.5
            
            # Skip batches with exploding gradients
            if total_grad_norm > 10000.0:  # Gradient explosion threshold
                logger.warning(f"Batch {batch_idx}: Max grad norm={max_grad_norm}, Exploding gradient norm={total_grad_norm:.2f}")
                exploding_count += 1
                # skipped_count += 1
                # optimizer.zero_grad()  # Clear the bad gradients
                # continue
            elif total_grad_norm > 1000.0:
                moderate_count += 1
                if batch_idx % 100 == 0:
                    logger.warning(f"Batch {batch_idx}: Moderate gradient norm={total_grad_norm:.2f}")
            else:
                normal_count += 1

            # *** Add gradient clipping to avoid exploding gradients causing NaN loss
            # clip_grad_norm_ does in-place update and returns original to grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check for NaN gradients after clipping
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                logger.warning(f"Batch {batch_idx}: NaN gradient detected after clipping- skipping!")
                nan_count += 1
                skipped_count += 1
                optimizer.zero_grad()
                continue

            optimizer.step()
            total_loss += loss_value
            valid_batch_count += 1

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: loss={loss_value:.4f}, grad_norm={total_grad_norm:.4f}")

        except Exception as e:
            logger.error(f"Batch {batch_idx}: Error {e} - skipping!")
            skipped_count += 1
            continue

    if valid_batch_count == 0:
        logger.error("No valid batches in epoch!")
        return float('inf'), 100.0
    
    avg_loss = total_loss / valid_batch_count
    total_processed = exploding_count + moderate_count + normal_count
    exploding_pct = 100.0 * exploding_count / max(1, total_processed)

    logger.info(f"Epoch {epoch} summary:")
    logger.info(f"   Valid_batches={valid_batch_count}")
    logger.info(f"   Skipped batches: {skipped_count}")
    logger.info(f"   Normal gradients: {normal_count}")
    logger.info(f"   Moderate gradients: {moderate_count}")
    logger.info(f"   EXPLODING gradients: {exploding_count} ({exploding_pct:.2f}%)")
    logger.info(f"   Avg loss: {avg_loss:.4f}")

    logger.info(f"[GRADIENT_STATS] epoch={epoch}, exploding_grad_pct={exploding_pct:.2f}%")

    return avg_loss, exploding_pct

def eval(data_loader,
         model: nn.Module, 
         eval_fn: Callable,
         device: torch.device):
    """
    Evaluate the model on a dataset.
    """
    if len(data_loader) == 0:
        logger.warning("Data loader is empty!")
        return float('inf')

    total_error = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                batch = batch.to(device)
                
                batch_indices = batch.batch if hasattr(batch, 'batch') else  torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
                predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch_indices)
                
                if len(mask_indices) == 0:
                    logger.warning(f"Eval batch {batch_idx}: No masked nodes, skipping")
                    continue

                loss = eval_fn(predicted_pos, batch.pos[mask_indices])
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Eval batch {batch_idx}: Invalid loss {loss.item()}")
                    continue

                total_error += loss.item()
                total_samples += 1

            except Exception as e:
                logger.error(f"Eval batch {batch_idx}: Error {e}, skipping")
                continue
        
        if total_samples == 0:
            logger.error("No valid evaluation samples!")
            return float('inf')

    avg_error = total_error / total_samples

    return avg_error 

def report_hyperparameter_tuning_metric(val_loss, epoch, exploding_grad_pct=None):
    """Report metric for Vertex AI Hyperparameter Tuning"""
    try:
        import hypertune

        # catch inf/nan so that hyperparam tuning isn't killed
        if val_loss is None or not isinstance(val_loss, (int, float)):
            logger.warning(f"Invalid val_loss type: {type(val_loss)}, skipping metric report")
            return

        if not (0 < val_loss < 1e6): # Sanity check
            logger.warning(f"Invalid val_loss value: {val_loss}, skipping metric report")
            return
        
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=val_loss,
            global_step=epoch
        )

        if exploding_grad_pct is not None and 0 <= exploding_grad_pct <= 100:
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='exploding_grad_pct',
                metric_value=float(exploding_grad_pct),
                global_step=epoch
            )
            logger.info(f"Reported val_loss={val_loss:.4f}, exploding_grad_pct={exploding_grad_pct:.2f}% at epoch {epoch}")
        else:
            logger.info(f"Reported val_loss={val_loss:.4f} at epoch {epoch}")
        
    
    except ImportError:
        logger.debug("hypertune not available, skipping metric reporting")
    except Exception as e:
        logger.warning(f"Failed to report metric: {e}")

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
    parser.add_argument('--pos_scale', type=float, default=0.0, help='Scale the position update')

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
    logger.info(args)

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
             skip_connection=True, 
             pos_scale=args.pos_scale)
    decoder = Decoder(
             in_channels=args.hidden_dim,
             hidden_channels=args.hidden_dim,
             num_layers=args.num_decoder_layers,
             pos_dim=3,
             edge_dim=train_dataset.edge_features_dim,
             skip_connection=True, 
             pos_scale=args.pos_scale)
    model = MaskedGeometricAutoencoder(
             encoder=encoder, 
             decoder=decoder,
             masking_ratio=args.masking_ratio)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")

    for m in [encoder, decoder, model]:
        m.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_initial = 0

    # load model and optim checkpoints if specified
    if args.load_model_path:
        epoch_initial, checkpoint = load_checkpoint(model, optim, args.load_model_path)
        logger.info(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        logger.info(f'Last epoch training loss: {checkpoint["train_loss"]}, test loss: {checkpoint["test_loss"]}')

    logger.info("Checking weight initialization...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"NaN/Inf in parameter: {name}")
            logger.warning(f"   Shape: {param.shape}, values: {param.flatten()[:10]}")

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    # Training loop
    for epoch in range(epoch_initial, args.num_epochs):
        train_loss, exploding_grad_pct = train_one_epoch(train_loader,
                                     model,
                                     optim,
                                     l2_loss,
                                     device,
                                     epoch)

         # Check if training diverged
        if train_loss == float('inf') or train_loss != train_loss:
            logger.error(f"Training diverged at epoch {epoch}")
            sys.exit(1)
        
        val_loss = eval(val_loader, model, l2_loss, device)

        if val_loss == float('inf') or val_loss != val_loss:
            logger.error(f"Validation failed at epoch {epoch}")
            sys.exit(1)

        # Report to hyperparameter tuning service EVERY epoch
        report_hyperparameter_tuning_metric(val_loss, epoch, exploding_grad_pct)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            logger.info(f"New best model found at epoch {best_epoch} with val loss {best_val_loss:.4f}")

        # Log progress every epoch
        logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ExplodingGrad={exploding_grad_pct:.2f}%")

        # Save checkpoint periodically
        if (epoch+1) % args.checkpoint_interval == 0:
            test_loss = eval(test_loader, model, l2_loss, device)

            save_checkpoint(model,
                            optim,
                            epoch,
                            train_loss,
                            val_loss,
                            test_loss,
                            f"{args.model_save_path}/checkpoint_epoch_{epoch}.pt")
            logger.info(f"Checkpoint saved. Test Loss: {test_loss:.4f}")
        
    if best_model_state is not None:
        torch.save(best_model_state, f"{args.model_save_path}/best_model.pt")
        logger.info(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    model.load_state_dict(best_model_state)
    final_test_loss = eval(test_loader, model, l2_loss, device)
    logger.info(f"FInal test loss (best model): {final_test_loss:.4f}")

if __name__ == "__main__":
    main()
