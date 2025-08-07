#!/bin/bash
#SBATCH --job-name=ip2p-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thomas@recreategoods.com

# Run the test script
./test_single_gpu.sh