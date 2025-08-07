#!/bin/bash
#SBATCH --job-name=ip2p-fashion-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --output=logs/fashion_8gpu_%j.out
#SBATCH --error=logs/fashion_8gpu_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ip2p-h100

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:/sc/home/thomas.chille/instruct-pix2pix"
export WANDB_API_KEY=937ef632699ed7d9674fb195535f0d90153f8921

# NCCL debugging flags
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TREE_THRESHOLD=0

# Change to project directory
cd /sc/home/thomas.chille/instruct-pix2pix

# Start eight GPU distributed training
echo "Starting eight GPU distributed training..."
echo "Node: $(hostname)"
echo "GPUs: 0,1,2,3,4,5,6,7"
echo "Running distributed training with torch.distributed.run"

srun --cpu-bind=none python -m torch.distributed.run --nproc_per_node=8 main.py \
    --name fashion-8gpu-training-$(date +%Y%m%d_%H%M%S) \
    --base configs/train_fashion.yaml \
    --train \
    --gpus "0,1,2,3,4,5,6,7" \
    --logdir logs/fashion-training
