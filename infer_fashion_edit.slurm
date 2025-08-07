#!/bin/bash
#SBATCH --job-name=infer_fashion_edit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --output=logs/infer_fashion_edit_%j.out
#SBATCH --error=logs/infer_fashion_edit_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPU count: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) ($CUDA_VISIBLE_DEVICES)"
echo "Job time limit: ${SBATCH_TIMELIMIT:-$(grep "#SBATCH --time=" $0 | cut -d'=' -f2)}"
echo "Start time: $(date)"

# Load modules if needed (adjust based on your cluster)
# module load cuda/12.1
# module load python/3.8

# Navigate to project directory
cd /sc/home/thomas.chille/instruct-pix2pix

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ip2p-h100

# Create logs directory if it doesn't exist
mkdir -p logs

# Create empty log files immediately so they can be monitored
touch logs/infer_fashion_edit_${SLURM_JOB_ID}.out
touch logs/infer_fashion_edit_${SLURM_JOB_ID}.err

# Quick GPU check
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | tr '\n' ', ' | sed 's/, $//')"


# Fixed port
PORT=8000

echo "Starting FastAPI server on port $PORT..."
echo "Server will be accessible at: http://$(hostname):$PORT"
echo "API documentation at: http://$(hostname):$PORT/docs"

# Navigate to infer directory and start the server
cd infer

# Start ngrok tunnel in background
echo "Starting ngrok tunnel..."
ngrok http --url=legal-guppy-balanced.ngrok-free.app 8000 &
NGROK_PID=$!

# Wait a moment for ngrok to start
sleep 5

echo "Ngrok tunnel started with PID: $NGROK_PID"
echo "Local server http://$(hostname):$PORT tunneled via https://legal-guppy-balanced.ngrok-free.app"

# Start the FastAPI server
python api_server.py --host 0.0.0.0 --port $PORT

# Clean up ngrok when server stops
echo "Stopping ngrok tunnel..."
kill $NGROK_PID 2>/dev/null || true

echo "Server stopped at: $(date)"