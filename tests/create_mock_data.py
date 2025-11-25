import os
import torch
from pathlib import Path
from torch_geometric.data import Data
import logging
import shutil

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_mock_data(base_path: str, num_samples: int = 10):
    """
    Generates mock PyTorch Geometric Data objects and saves them 
    to a local directory structure.
    
    Args:
        base_path: The local directory to use as the cache root (e.g., /tmp/pyg_cache).
        num_samples: The number of graph samples to generate for each split.
    """
    if os.path.exists(base_path):
        logger.warning(f"Removing existing directory: {base_path}")
        shutil.rmtree(base_path)

    # Define the mock GCS prefixes relative to the base path
    mock_prefixes = {
        'train': os.path.join(base_path, 'mock_data', 'train'),
        'val': os.path.join(base_path, 'mock_data', 'val'),
        'test': os.path.join(base_path, 'mock_data', 'test'),
    }
    
    # Define consistent feature dimensions for the mock data
    NODE_FEATURES_DIM = 5
    EDGE_FEATURES_DIM = 4

    logger.info(f"Generating {num_samples} mock graph files for each of the three splits...")

    for split, path in mock_prefixes.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        for i in range(num_samples):
            # Create a mock Data object
            num_nodes = torch.randint(low=10, high=50, size=(1,)).item()
            
            # Node features (x), position (pos), edge index, and edge attributes (edge_attr)
            mock_data = Data(
                x=torch.randn(num_nodes, NODE_FEATURES_DIM),
                pos=torch.randn(num_nodes, 3),
                edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)), # Dense mock edges
                edge_attr=torch.randn(num_nodes * 2, EDGE_FEATURES_DIM)
            )
            
            file_path = os.path.join(path, f'{split}_graph_{i:03d}.pt')
            torch.save(mock_data, file_path)
        
        logger.info(f"Generated {num_samples} files in: {path}")

    logger.info("Mock data generation complete.")


if __name__ == '__main__':
    # Customize the number of mock samples here
    # The output will be created in ./mock_cache
    generate_mock_data(base_path='./mock_gcs_root', num_samples=5)
    
    # After running this script, files ready to be 'downloaded'
    # by gcs_dataset_loader.py (though we will skip the real download step
    # for local testing, or point the GCS code to a real path).