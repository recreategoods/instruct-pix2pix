#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ip2p

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/sc/home/thomas.chille/instruct-pix2pix"
export WANDB_API_KEY=937ef632699ed7d9674fb195535f0d90153f8921

# Change to project directory
cd /sc/home/thomas.chille/instruct-pix2pix

# Start small test training
echo "Starting single GPU test..."
if python main.py \
    --name test-single-gpu \
    --base configs/train_fashion.yaml \
    --train \
    --gpus "0" \
    --logdir logs/test-training; then
    echo "SUCCESS: Training completed successfully!"
    exit 0
else
    echo "FAILURE: Training failed with exit code $?"
    exit 1
fi