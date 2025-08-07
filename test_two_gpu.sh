#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ip2p-h100

# Set environment variables for debugging
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:/sc/home/thomas.chille/instruct-pix2pix"
export WANDB_API_KEY=937ef632699ed7d9674fb195535f0d90153f8921

# NCCL debugging flags
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TREE_THRESHOLD=0

# Change to project directory
cd /sc/home/thomas.chille/instruct-pix2pix

# Start two GPU test training
echo "Starting two GPU distributed test..."
echo "Node: $(hostname)"
echo "GPUs: 0,1"
echo "Running on GPUs 0,1"

if python main.py \
    --name test-two-gpu \
    --base configs/train_fashion.yaml \
    --train \
    --gpus "0,1" \
    --logdir logs/test-training; then
    echo "SUCCESS: Two GPU training completed successfully!"
    exit 0
else
    echo "FAILURE: Two GPU training failed with exit code $?"
    exit 1
fi