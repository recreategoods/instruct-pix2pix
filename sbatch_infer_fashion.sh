#!/bin/bash

# Parse command line arguments
SBATCH_ARGS=""
CUSTOM_TIME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --time)
            SBATCH_ARGS="$SBATCH_ARGS --time $2"
            CUSTOM_TIME="$2"
            echo "Override time limit: $2"
            shift 2
            ;;
        --time=*)
            SBATCH_ARGS="$SBATCH_ARGS $1"
            CUSTOM_TIME="${1#--time=}"
            echo "Override time limit: ${1#--time=}"
            shift
            ;;
        *)
            SBATCH_ARGS="$SBATCH_ARGS $1"
            shift
            ;;
    esac
done

# Submit the SLURM job and capture the job ID
echo "Submitting SLURM job..."
if [ -n "$CUSTOM_TIME" ]; then
    JOB_OUTPUT=$(SBATCH_TIMELIMIT="$CUSTOM_TIME" sbatch $SBATCH_ARGS infer_fashion_edit.slurm)
else
    JOB_OUTPUT=$(sbatch $SBATCH_ARGS infer_fashion_edit.slurm)
fi

# Parse job ID from sbatch output (format: "Submitted batch job 123456")
JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+$')

if [ -z "$JOB_ID" ]; then
    echo "Error: Could not parse job ID from: $JOB_OUTPUT"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"

# Create log files immediately
LOG_DIR="logs"
OUT_LOG="${LOG_DIR}/infer_fashion_edit_${JOB_ID}.out"
ERR_LOG="${LOG_DIR}/infer_fashion_edit_${JOB_ID}.err"

mkdir -p "$LOG_DIR"
touch "$OUT_LOG"
touch "$ERR_LOG"

echo "Created log files:"
echo "  Output: $OUT_LOG"
echo "  Error:  $ERR_LOG"
echo ""
echo "Starting to tail output log (Ctrl+C to stop)..."
echo "----------------------------------------"

# Tail the output log
tail -f "$OUT_LOG"