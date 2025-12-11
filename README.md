# CS224W Project: EGNN-MAE for Binding Affinity Prediction

Self-supervised pretraining of Equivariant Graph Neural Networks using Masked Autoencoders for protein-ligand binding affinity prediction.

## Overview

This project implements a novel self-supervised learning approach for molecular property prediction:
- **Masked Autoencoder (MAE)** pretraining framework for learning geometric molecular representations
- **E(n) Equivariant Graph Neural Networks (EGNN)** ensuring rotation, translation, and reflection invariance for node features and equivariance for position updates.
- **Regression Heads** applied to binding affinity prediction
- **Position normalization** and stability techniques for training on 3D molecular structures

## Repository Structure
```
CS224W_Project/
├── models/                             # Core model architectures
│   ├── egnn.py                         # EGNN implementation 
│   ├── egnn_conv.py                    # EGNN convolutional layer
│   └── bimolecular_affinity_models.py  # MAE, Encoder, Decoder, Regression heads
├── scripts/                            # Training and evaluation scripts
│   ├── deploy_to_vertex_ai.sh          # MAE pretraining script
│   ├── train_masked_autoencoder.py     # MAE pretraining script (another one)
│   ├── train_prediction_head.py        # Model training script for the regression task
│   ├── run_gcp_vertex_mae.py           # MAE training job submission script
│   ├── run_gcp_vertex_head.py          # Model training job submission script
│   ├── run_gcp_vertex_hp_tuning_head.py# MAE pretraining hyperparameter tunning job submission script
├── utils/                              # Utility modules
│   ├── gcs_dataset_loader.py           # Data loading (supports GCS)
│   └── checkpoint_utils.py             # Checkpoint management
│   └── losses.py                       # Loss functions
│   └── eval.py                         # Eval functions
│   └── checkpoint_utils.py             # Checkpoint management
├── utils/                              # Tests
├── utils/                              # Notebooks for dataset analysis
├── Dockerfile                          # Dockerfile for building Docker image
```

## Key Model Components
### 1. Masked Geometric Autoencoder
```python
class MaskedGeometricAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, masking_ratio):
        # Learnable tokens for masked node
        self.masked_node_token = nn.Parameter(...)
```

### 2. EGNN Layer
```python
class EGNN(torch.nn.Module):
```
**Key Implementation Details**
- **E(3) equivariance**: Rotation, translation, and reflection invariant features and equivariant positions

### 3. Regression Heads
- **MLP model**:  MLP layers -> Global pooling -> Linear layer.
- **EGNN model**: EGNN layers -> Global pooling -> Linear layer.
- **EGNN-MLP model**: EGNN layers -> Global pooling -> MLP layers -> Linear layer.
- **Global pooling**: Mean, Add, and Max supported.

### 4. Training Stability Techniques

Refer to [here](https://github.com/siavashadpey/CS224W_Project/edit/feature/vertex-gcs-pipeline/README.md).

## Installation
### Local Setup
```bash
# Clone repository
git clone https://github.com/siavashadpey/CS224W_Project.git
cd CS224W_Project
```
### Docker Setup
```bash
# Build Docker image
docker build -t cs224w-training:latest .

# Or pull from GCR (if available)
docker pull gcr.io//cs224w-training:latest
```

## Running Pre-Training
### Local Training (CPU/GPU)
```bash
# Set environment variables
export LEARNING_RATE=0.0005
export HIDDEN_DIM=256
export NUM_ENCODER_LAYERS=4
export NUM_DECODER_LAYERS=4
export MASKING_RATIO=0.3
export NUM_EPOCHS=50
export BATCH_SIZE=32

# Run training
python scripts/train_masked_autoencoder.py \
    --learning_rate $LEARNING_RATE \
    --hidden_dim $HIDDEN_DIM \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_decoder_layers $NUM_DECODER_LAYERS \
    --masking_ratio $MASKING_RATIO \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE
```

### Docker Training
```bash
# Using the convenience script
LEARNING_RATE=0.0005 \
HIDDEN_DIM=256 \
NUM_EPOCHS=50 \
./run_training_docker_vm.sh

# Or manually
docker run --gpus all \
    -e LEARNING_RATE=0.0005 \
    -e HIDDEN_DIM=256 \
    -e NUM_EPOCHS=50 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    cs224w-training:latest
```

### Vertex AI Training
```bash
# Set up GCP project
export PROJECT_ID=
export REGION=us-central1
export BUCKET_NAME=

# Submit training job
./deploy_to_vertex_ai.sh

# Monitor job
gcloud ai custom-jobs list --region=$REGION --limit=5

# Stream logs
gcloud ai custom-jobs stream-logs  --region=$REGION
```

### Hyperparameter Tuning
```bash
# Submit hyperparameter tuning job to Vertex AI
python scripts/submit_hyperparam_tuning.py \
    --project_id $PROJECT_ID \
    --region $REGION \
    --bucket_name $BUCKET_NAME \
    --display_name "egnn-mae-tuning" \
    --max_trial_count 20 \
    --parallel_trials 4 \
    --num_epochs 100
```
**Tuning Parameters:**
- `learning_rate`: [0.00001, 0.001] (log scale)
- `hidden_dim`: [64, 128, 256]
- `num_encoder_layers`: [3, 4, 5]
- `num_decoder_layers`: [3, 4, 5]
- `masking_ratio`: [0.3, 0.6]
- `batch_size`: [16, 32, 64, 128, 256]

## Training Configuration
```python
NUM_EPOCHS = 100                    
```

### Hyperparameter Tuning Recommendations
```python
# Model Architecture
HIDDEN_DIM = 256                    # Encoder/decoder hidden dimensions
NUM_ENCODER_LAYERS = 4              # EGNN layers in encoder
NUM_DECODER_LAYERS = 4              # EGNN layers in decoder
MASKING_RATIO = 0.3                 # 30% of atoms masked

# Training
LEARNING_RATE = 0.0001              # Higher LR works well with normalization
BATCH_SIZE = 16

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler (Optional)
lr_scheduler = ExponentialLR(optimizer, gamma=args.learning_rate_gamma)
```

## Data

The project uses **PDBbind CleanSplit** dataset (Graber et al., 2024), preprocessed as PyTorch Geometric datasets:

- **Training set**: 13,168 protein-ligand complexes
- **Validation set**: 3294 complexes
- **Test set**: CASF-2016 benchmark (282 complexes)

**Data Location** (GCS):
```
gs://cs224w-2025-mae-gnn-central/data_w_pos/
├── plgems_train.pt
├── plgems_validation.pt
└── plgems_full_casf2016.pt
```

**Graph Structure**:
- Nodes: Ligand atoms + protein amino acids
- Node features: Atom type, charge, hybridization (ligand); one-hot residue type (protein)
- Edges: Covalent bonds + non-covalent interactions (<5Å)
- Coordinates: 3D positions for all atoms

## Checkpoints

Checkpoints are saved to:
- **Local**: `./checkpoints/`
- **GCS**: `gs://<bucket>/checkpoints/<trial_id>/`
```bash
# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Citation
```bibtex
@misc{cs224w2024maeegnn,
  title={Self-Supervised Masked Autoencoder Pretraining of Equivariant Graph Neural Networks for Binding Affinity Prediction},
  author={[Siavosh Shadpey, Vasanti Wall-Persad]},
  year={2025},
  note={CS224W Course Project, Stanford University}
}
```
