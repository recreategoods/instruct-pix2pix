#!/usr/bin/env python3

import torch
import os

print("=== CUDA Debug Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available!")

print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

print("\n=== PyTorch Lightning Test ===")
try:
    import pytorch_lightning as pl
    print(f"PyTorch Lightning version: {pl.__version__}")
    
    # Test creating a simple trainer
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)
    print(f"Trainer accelerator: {trainer.accelerator}")
    print(f"Trainer device_ids: {getattr(trainer, 'device_ids', 'Not available')}")
except Exception as e:
    print(f"PyTorch Lightning error: {e}")