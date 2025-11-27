from typing import Tuple
import os 

import torch
from torch import nn

def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int, 
                    train_loss: float,
                    val_loss: float,
                    test_loss: float,
                    file_path: str):
    """Save checkpoint locally and, possibly, to GCS if GCS_BUCKET is set 
        in environment variables.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }

    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved locally: {file_path}")

    GCS_BUCKET = os.environ.get("GCS_BUCKET")
    if GCS_BUCKET:
        from google.cloud import storage

        try:
            gcs_path = f"checkpoints/{os.path.basename(file_path)}"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(file_path)
            print(f"Checkpoint uploaded to GCS: gs://{GCS_BUCKET}/{gcs_path}")
        except Exception as e:
            print(f"Failed to upload checkpoint to GCS: {e}")

def load_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    filepath: str) -> Tuple[int, dict]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, checkpoint
