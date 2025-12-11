from typing import Tuple
import os 
import csv


import torch
from torch import nn

def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int, 
                    train_loss: float,
                    val_loss: float,
                    test_loss: float,
                    file_path: str):
    """Save checkpoint locally and, possibly, to GCS if GCS_BUCKET is set 
        in environment variables.

    Args:
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler to save.
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        test_loss (float): Test loss.
        file_path (str): Local file path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
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
            gcs_path = f"checkpoints/{os.environ.get('CLOUD_ML_JOB_ID', 'default')}/{os.path.basename(file_path)}"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(file_path)
            print(f"Checkpoint uploaded to GCS: gs://{GCS_BUCKET}/{gcs_path}")
        except Exception as e:
            print(f"Failed to upload checkpoint to GCS: {e}")

def load_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    filepath: str) -> Tuple[int, dict]:
    """Load model checkpoint
    
    Args:
        model (nn.Module): Model to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler to load state into
        filepath (str): Path to the checkpoint file. Can be a GCS URI (gs://bucket/path) or local path.
    """
    filepath = download_blob(filepath)

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, checkpoint


def download_blob(gcs_uri, destination_folder="/tmp/checkpoints"):
    """Downloads a file from GCS to a local path inside the container.
    
    Args:
        gcs_uri (str): GCS URI of the file to download.
        destination_folder (str): Local folder to save the downloaded file.
    """
    if not gcs_uri.startswith("gs://"):
        return gcs_uri # It's already a local path
    
    from google.cloud import storage

    path_parts = gcs_uri.replace("gs://", "").split("/")
    bucket_name = path_parts.pop(0)
    blob_name = "/".join(path_parts)
    
    os.makedirs(destination_folder, exist_ok=True)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    file_name = os.path.basename(blob_name)
    local_file_path = os.path.join(destination_folder, file_name)
    
    print(f"Downloading {gcs_uri} to {local_file_path}...")
    blob.download_to_filename(local_file_path)
    print("Download complete.")
    
    return local_file_path

def save_csv_file(data: list, file_path: str):
    """Save a list of dictionaries to a CSV file.

    Args:
        data (list): List of dictionaries to save.
        file_path (str): Path to the CSV file.
    """

    if not data:
        print("No data to save.")
        return

    keys = data[0].keys()
    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

    GCS_BUCKET = os.environ.get("GCS_BUCKET")
    if GCS_BUCKET:
        from google.cloud import storage

        try:
            gcs_path = f"checkpoints/{os.environ.get('CLOUD_ML_JOB_ID', 'default')}/{os.path.basename(file_path)}"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(file_path)
            print(f"CSV file uploaded to GCS: gs://{GCS_BUCKET}/{gcs_path}")
        except Exception as e:
            print(f"Failed to upload CSV file to GCS: {e}")
    
    print(f"CSV file saved: {file_path}")