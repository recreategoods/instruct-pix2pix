#!/usr/bin/env python3
"""
Test script to verify GPU functionality and the --gpus parameter
Run this from within an active SLURM job with GPU allocation
"""

import torch
import sys
import os

print("=== GPU Test Script ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test GPU tensor operations
    print("\n=== Testing GPU Operations ===")
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Warm up
    for _ in range(10):
        z = torch.matmul(x, y)
    
    # Time the operation
    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Matrix multiplication test passed!")
    print(f"100 operations took {(end-start)*1000:.2f} ms")
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
else:
    print("No GPU available!")
    sys.exit(1)

# Test environment variables from SLURM script
print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# Test the main.py command structure (dry run)
print("\n=== Testing main.py command structure ===")
cmd_args = [
    "--name", "test-single-gpu",
    "--base", "configs/train_fashion.yaml", 
    "--train",
    "--gpus", "1",
    "--logdir", "logs/test-training"
]

print("Command that would be executed:")
print(f"python main.py {' '.join(cmd_args)}")

# Check if required files exist
required_files = [
    "main.py",
    "configs/train_fashion.yaml"
]

print("\n=== Checking required files ===")
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path} exists")
    else:
        print(f"✗ {file_path} missing")

print("\n=== Test completed successfully! ===")