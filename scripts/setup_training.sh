#!/bin/bash

# Training setup script for InstructPix2Pix fashion dataset
# This script ensures all prerequisites are met before training

set -e  # Exit on any error

echo "=== InstructPix2Pix Training Setup ==="

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR="$SCRIPT_DIR/.."

# Change to project directory
cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p logs
mkdir -p stable_diffusion/models/ldm/stable-diffusion-v1

# Check if pretrained Stable Diffusion checkpoint exists
CHECKPOINT_PATH="stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Pretrained Stable Diffusion checkpoint not found. Downloading..."
    bash scripts/download_pretrained_sd.sh
    echo "Checkpoint downloaded successfully!"
else
    echo "Pretrained checkpoint already exists: $CHECKPOINT_PATH"
fi

# Verify fashion dataset exists
if [ ! -d "data/fashion-edit-dataset" ]; then
    echo "ERROR: Fashion edit dataset not found at data/fashion-edit-dataset"
    echo "Please ensure the dataset is properly created and placed in the correct location."
    exit 1
else
    echo "Fashion dataset found: data/fashion-edit-dataset"
    # Count dataset samples
    SAMPLE_COUNT=$(find data/fashion-edit-dataset -type d -name "0*" | wc -l)
    echo "Dataset contains $SAMPLE_COUNT samples"
fi

# Check/create conda environment
if ! conda env list | grep -q "ip2p"; then
    echo "Creating conda environment from environment.yaml..."
    conda env create -f environment.yaml
    echo "Environment 'ip2p' created successfully!"
else
    echo "Conda environment 'ip2p' already exists"
fi

# Verify current environment
if [[ "$CONDA_DEFAULT_ENV" != "ip2p" ]]; then
    echo "WARNING: Not in 'ip2p' conda environment. Current environment: $CONDA_DEFAULT_ENV"
    echo "Please activate the ip2p environment: conda activate ip2p"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -8
else
    echo "WARNING: nvidia-smi not found. Cannot verify GPU availability."
fi

echo ""
echo "=== Setup Complete ==="
echo "Ready to start training with:"
echo "  - Config: configs/train_fashion.yaml" 
echo "  - SLURM script: train_fashion_edit.slurm"
echo ""
echo "To submit the job: sbatch train_fashion_edit.slurm"