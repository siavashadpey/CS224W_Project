import argparse
from typing import Callable, Union, Tuple, Optional

from torch import nn, torch, Tensor

from torch_geometric.typing import Adj

from models.bimolecular_affinity_models import MaskedGeometricAutoencoder

"""
TODO:
 - Training loop
 - Evaluation function
 - Loss function
 - Refactoring: utils/losses.py, 
"""

"""
TODO:
 - Data loader (train, test, eval) (data/pdbbind_clean_data.py)
 - Save checkpoints (including epoch, optimizer state) (utils/io.py)
 - Load checkpoint (including epoch, optimizer state) (utils/io.py)
 - Prediction head model (models/bimolecular_affinity_models.py)
 - Prediction head training script (scripts/train_prediction_head.py)
"""

def train_one_epoch():
    pass

def eval():
    pass

def main():
    parser = argparse.ArgumentParser(description="Main script for the project.")
    parser.add_argument('--data_path', type=str, help='Path to the dataset.')
    parser.add_argument('--model_save_path', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers in the model.')
    parser.add_argument('--masking_ratio', type=float, default=0.45, help='Ratio of masking for input data.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Periodically save checkpoints each number of epochs.')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate the model on validation set each number of epochs.')

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load training, testing, and validation data

    # Initialize (or load) model and optimizer

    # Training loop
    for epoch in range(args.num_epochs):
        train_one_epoch()

if __name__ == "__main__":
    main()