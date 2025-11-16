import argparse
import os
from typing import Callable, Union, Tuple, Optional

from torch import nn, torch

from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

seed_everything(1313)

from models.bimolecular_affinity_models import MaskedGeometricAutoencoder, Encoder, Decoder
from io import save_checkpoint, load_checkpoint
from utils.losses import chamfer_l2_distance

"""
TODO:
 - Loss function. Should work with batch of graphs (disconnected graph)
 - Data loader (train, test, eval) (data/pdbbind_clean_data.py)
 - Prediction head model (models/bimolecular_affinity_models.py)
 - Prediction head training script (scripts/train_prediction_head.py)
"""

def train_one_epoch(DataLoader: DataLoader, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: Callable):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in DataLoader:
        batch = batch.to(model.device)
        optimizer.zero_grad()
        predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        loss = loss_fn(predicted_pos, batch.pos[mask_indices])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(DataLoader)
    return avg_loss

def eval(DataLoader: DataLoader,
         model: nn.Module, 
         eval_fn: Callable):
    """
    Evaluate the model on the a dataset.
    """
    total_error = 0
    model.eval()
    for batch in DataLoader:
        batch = batch.to(model.device)
        predicted_pos, mask_indices = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        total_error += eval_fn(predicted_pos, batch.pos[mask_indices])

    avg_error = total_error / len(DataLoader)
    return avg_error 

def main():
    parser = argparse.ArgumentParser(description="Main script for the project.")
    parser.add_argument('--data_path', type=str, help='Path to the dataset.')
    parser.add_argument('--model_save_path', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers in the model.')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers in the model.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for the model.')
    parser.add_argument('--masking_ratio', type=float, default=0.45, help='Ratio of masking for input data.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Periodically save checkpoints each number of epochs. Also used for loss computation frequency on val and test sets.')

    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # TODO: Load training, testing, and validation data

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
        print(f"Loaded model from {args.load_model_path} at epoch {epoch_initial}")
        print(f'Last epoch training loss: {checkpoint["train_loss"]}, test loss: {checkpoint["test_loss"]}')

    # Training loop
    for epoch in range(epoch_initial, args.num_epochs):
        train_loss = train_one_epoch(train_dataset,
                                   model,
                                   optim,
                                   chamfer_l2_distance)

        if epoch % args.checkpoint_interval == 0:
            val_loss = eval(val_dataset,
                            model,
                            chamfer_l2_distance)
            test_loss = eval(test_dataset,
                            model,
                            chamfer_l2_distance)

            save_checkpoint(model,
                            optim,
                            epoch,
                            train_loss,
                            val_loss,
                            test_loss,
                            f"{args.model_save_path}/checkpoint_epoch_{epoch}.pt")
            print(f"Epoch {epoch}/{args.num_epochs}: Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}")

if __name__ == "__main__":
    main()