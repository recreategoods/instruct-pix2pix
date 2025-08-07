#!/bin/bash
#SBATCH --job-name=ip2p-test-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --output=logs/test_2gpu_%j.out
#SBATCH --error=logs/test_2gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thomas@recreategoods.com

# Run the two GPU test script with srun (disable CPU binding)
srun --cpu-bind=none ./test_two_gpu.sh