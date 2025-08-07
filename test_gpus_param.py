#!/usr/bin/env python3
"""
Test script to specifically test the --gpus parameter parsing and functionality
"""

import argparse
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

# Add the same path as main.py
sys.path.append("./stable_diffusion")

def test_gpus_param():
    print("=== Testing --gpus Parameter ===")
    
    # Test different gpus parameter formats
    test_cases = [
        "1",           # Single GPU as string
        1,             # Single GPU as int
        "0,1",         # Multiple GPUs as string  
        [0, 1],        # Multiple GPUs as list
    ]
    
    for i, gpus_val in enumerate(test_cases):
        print(f"\nTest case {i+1}: gpus = {gpus_val} (type: {type(gpus_val)})")
        
        try:
            # Create a trainer with the gpus parameter
            trainer = Trainer(
                gpus=gpus_val,
                max_epochs=1,
                logger=False,
                enable_checkpointing=False,
                accelerator='auto'
            )
            
            print(f"  ✓ Trainer created successfully")
            print(f"  ✓ Device IDs: {trainer.device_ids}")
            print(f"  ✓ Number of devices: {trainer.num_devices}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_argument_parsing():
    print("\n=== Testing Argument Parsing ===")
    
    # Create parser similar to main.py
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    
    # Test different command line formats
    test_commands = [
        ["--gpus", "1"],
        ["--gpus", "0"],
        ["--gpus", "0,1"],
    ]
    
    for cmd in test_commands:
        print(f"\nTesting command: {' '.join(cmd)}")
        try:
            args = parser.parse_args(cmd)
            print(f"  ✓ Parsed gpus value: {args.gpus} (type: {type(args.gpus)})")
            
            # Test if we can handle the parsed value
            if isinstance(args.gpus, str) and ',' in args.gpus:
                ngpu = len(args.gpus.strip(",").split(','))
            elif isinstance(args.gpus, int):
                ngpu = 1 if args.gpus > 0 else 0
            else:
                ngpu = 1
                
            print(f"  ✓ Calculated number of GPUs: {ngpu}")
            
        except Exception as e:
            print(f"  ✗ Error parsing: {e}")

def test_cuda_environment():
    print("\n=== CUDA Environment ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if __name__ == "__main__":
    test_cuda_environment()
    test_argument_parsing() 
    test_gpus_param()
    print("\n=== All tests completed ===")