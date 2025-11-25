import os
import glob
import torch
from typing import List, Tuple
from pathlib import Path
import logging

# Use the official Google Cloud Storage client
try:
    # only import the client if we actually need to perform a download (skip for test)
    if os.environ.get('SKIP_GCS_DOWNLOAD') != 'True':
        from google.cloud import storage
    else:
        # if skipping download, set storage to None
        storage = None
except ImportError:
    logging.warning("Warning: 'google-cloud-storage' not found. Please install it using 'pip install google-cloud-storage'.")
    storage = None # Placeholder if import fails

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

# Set up logger for this module
logger = logging.getLogger(__name__)

def download_gcs_files(bucket_name: str, prefix: str, local_cache_dir: str, force_download: bool) -> List[str]:
    """
    Downloads all files under a specific GCS prefix to a local directory.
    
    Args:
        bucket_name: The GCS bucket name (e.g., 'my-bucket').
        prefix: The GCS path prefix to search within (e.g., 'data/').
        local_cache_dir: The local directory to save files to (e.g., '/tmp/pyg_cache/train').
        force_download: If True, redownload files even if they exist locally.
        
    Returns:
        A list of local file paths for the downloaded files.
    """
    # --- Bypass GCS Download for local testing ---
    if os.environ.get('SKIP_GCS_DOWNLOAD') == 'True':
        logger.warning(f"SKIP_GCS_DOWNLOAD is True. Simulating files found in {local_cache_dir}")

        # Read the list of files present in the cache directory
        local_file_paths = [
            os.path.join(local_cache_dir, f)
            for f in os.listdir(local_cache_dir)
            if f.endswith('.pt')
        ]

        if not local_file_paths:
            logger.error(f"No mock files found in simulated cache directory: {local_cache_dir}")
            raise RuntimeError("Local test failed: mock data not found in cache location")

        logger.info(f"Local files found and simulated: {len(local_file_paths)}")
        return local_file_paths
    # -------

    if storage is None:
        raise RuntimeError("The 'google-cloud-storage' library is required but failed to import.")

    logger.info(f"Starting download from gs://{bucket_name}/{prefix} to {local_cache_dir}")
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure local directory exists
    Path(local_cache_dir).mkdir(parents=True, exist_ok=True)

    local_file_paths = []
    
    # List all blobs (files) under the prefix
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Ignore directory markers if they exist (blobs with size 0 and name ending in '/')
        if blob.name.endswith('/'):
            continue
            
        file_name = os.path.basename(blob.name)
        local_path = os.path.join(local_cache_dir, file_name)

        if os.path.exists(local_path) and not force_download:
            logger.info(f"Skipping download, file already exists locally: {file_name}")
        else:
            logger.info(f"Downloading {blob.name} to {local_path}...")
            try:
                blob.download_to_filename(local_path)
            except Exception as e:
                logger.error(f"Error downloading {blob.name}: {e}")
                continue # Skip this file if download fails

        local_file_paths.append(local_path)
        
    logger.info(f"Download complete. Total files found: {len(local_file_paths)}")
    return local_file_paths


class GCSPyGDataset(Dataset):
    """
    A PyTorch Geometric Dataset wrapper that loads pre-saved Data objects 
    from local files, typically downloaded from GCS. This design is ideal for 
    large datasets (like 20GB) that cannot fit into memory, as it loads samples 
    on demand from the high-speed local disk cache.
    
    Assumes each file is a single torch_geometric.data.Data object saved 
    via torch.save(data, 'file.pt').
    """
    def __init__(self, root: str, file_paths: List[str], transform=None, pre_transform=None):
        self.file_paths = file_paths
        self._node_features = None
        self._edge_features_dim = None
        
        # call the parent constructor with the root path
        super().__init__(root, transform, pre_transform)
        
        # Inspect the first item to determine feature dimensions needed for the model initialization
        if self.file_paths:
            logger.info("Inspecting first data sample to determine feature dimensions...")
            first_data = self.get(0)
            # Assuming 'x' is node features and 'edge_attr' is edge features
            self._node_features = first_data.x.size(1) if first_data.x is not None else 0
            self._edge_features_dim = first_data.edge_attr.size(1) if first_data.edge_attr is not None else 0
        
        logger.info(f"Dataset initialized with {len(self)} samples.")
        logger.info(f"Determined Node Features Dim: {self.node_features}")
        logger.info(f"Determined Edge Features Dim: {self.edge_features_dim}")

    @property
    def raw_file_names(self) -> List[str]:
        # This dataset loads directly from provided paths, so no raw files to process
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # This dataset is itself the processed data
        return [os.path.basename(p) for p in self.file_paths]

    def len(self) -> int:
        return len(self.file_paths)

    def get(self, idx: int) -> Data:
        """Loads a single Data object from a file path."""
        file_path = self.file_paths[idx]
        try:
            # Load the pre-saved Data object
            # explicitly set weights_only=False to load PyG Data object
            data = torch.load(file_path, weights_only=False)
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise RuntimeError(f"Failed to load graph data at index {idx} from {file_path}")

    @property
    def node_features(self) -> int:
        """Returns the dimension of the node features."""
        return self._node_features
    
    @property
    def edge_features_dim(self) -> int:
        """Returns the dimension of the edge features."""
        return self._edge_features_dim


def create_gcs_dataloaders(
    bucket_name: str, 
    train_prefix: str, 
    val_prefix: str, 
    test_prefix: str, 
    batch_size: int, 
    num_workers: int, 
    local_cache_dir: str, 
    force_download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, GCSPyGDataset]:
    """
    Downloads data from GCS, creates PyTorch Geometric Datasets, and 
    initializes DataLoaders.

    Args:
        bucket_name, train_prefix, val_prefix, test_prefix: GCS parameters.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of subprocesses to use for data loading.
        local_cache_dir: Base directory for caching data locally.
        force_download: If True, redownload existing files.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader, train_dataset).
    """

    # --- 1. Download Data ---
    
    # 1.1 Training Data
    train_cache_dir = os.path.join(local_cache_dir, 'train')
    train_paths = download_gcs_files(bucket_name, train_prefix, train_cache_dir, force_download)
    
    # 1.2 Validation Data
    val_cache_dir = os.path.join(local_cache_dir, 'val')
    val_paths = download_gcs_files(bucket_name, val_prefix, val_cache_dir, force_download)
    
    # 1.3 Test Data
    test_cache_dir = os.path.join(local_cache_dir, 'test')
    test_paths = download_gcs_files(bucket_name, test_prefix, test_cache_dir, force_download)


    # --- 2. Create Datasets ---
    
    # The root for the dataset is just the base cache directory
    dataset_root = local_cache_dir 
    
    # We must create the training dataset first, as the main script uses 
    # its properties (node_features, edge_features_dim) for model initialization.
    train_dataset = GCSPyGDataset(root=dataset_root, file_paths=train_paths)
    val_dataset = GCSPyGDataset(root=dataset_root, file_paths=val_paths)
    test_dataset = GCSPyGDataset(root=dataset_root, file_paths=test_paths)
    
    
    # --- 3. Create DataLoaders ---
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    logger.info("All DataLoaders created successfully.")
    
    return train_loader, val_loader, test_loader, train_dataset

if __name__ == '__main__':
    # Simple test case for local execution (requires a GCS client/creds)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Testing gcs_dataset_loader locally ---")
    logger.info("To run this test, you need to set up Google Cloud authentication and a test bucket.")
    try:
        # Create a dummy .pt file for testing the Dataset class locally
        dummy_data = Data(x=torch.randn(10, 5), pos=torch.randn(10, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2, 4))
        test_dir = './gcs_test_cache/train'
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        torch.save(dummy_data, os.path.join(test_dir, 'dummy_1.pt'))
        
        # Test the Dataset class
        test_paths = [os.path.join(test_dir, 'dummy_1.pt')]
        test_dataset = GCSPyGDataset(root='./gcs_test_cache', file_paths=test_paths)
        logger.info(f"Test Dataset Node Features: {test_dataset.node_features}") 
        logger.info(f"Test Dataset Edge Features: {test_dataset.edge_features_dim}") 
        
        # Clean up dummy files
        os.remove(os.path.join(test_dir, 'dummy_1.pt'))
        os.rmdir(test_dir)
        os.rmdir('./gcs_test_cache')

    except Exception as e:
        logger.error(f"Local test failed due to: {e}")
        logger.warning("This is expected if torch_geometric or google-cloud-storage are not installed.")