#!/bin/bash
#SBATCH --job-name=ip2p-fashion-1gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --output=logs/fashion_1gpu_test_%j.out
#SBATCH --error=logs/fashion_1gpu_test_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ip2p-h100

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/sc/home/thomas.chille/instruct-pix2pix"
export WANDB_API_KEY=937ef632699ed7d9674fb195535f0d90153f8921

# Change to project directory
cd /sc/home/thomas.chille/instruct-pix2pix

# Start single GPU training
echo "Starting single GPU training (TEST RUN)..."
echo "Node: $(hostname)"
echo "GPU: 0"

python main.py \
    --name fashion-1gpu-test \
    --base configs/train_fashion.yaml \
    --train \
    --gpus "0" \
    --logdir logs/fashion-test