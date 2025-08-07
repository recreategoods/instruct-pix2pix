#!/usr/bin/env python3
"""
Transform HuggingFace dataset to InstructPix2Pix format.

This script converts a HuggingFace dataset (parquet files) to the directory structure
expected by InstructPix2Pix for training. It handles resume functionality, duplicate
class names, and filters out removed records.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple
from datasets import load_dataset, Features, Value, Image, DatasetDict
from PIL import Image as PILImage
import pickle

def load_progress(progress_file: str) -> Dict:
    """Load progress from pickle file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    return {
        'processed_indices': set(),
        'class_name_counts': defaultdict(int),
        'prompt_counter': 0,
        'seeds_mapping': {}
    }

def save_progress(progress_file: str, progress: Dict):
    """Save progress to pickle file."""
    with open(progress_file, 'wb') as f:
        pickle.dump(progress, f)

def make_unique_seed(class_name: str, class_name_counts: Dict[str, int]) -> str:
    """Generate unique seed from class_name, adding counter if needed."""
    # class_name should be a string number like '0019891'
    if class_name_counts[class_name] == 0:
        unique_seed = class_name
    else:
        unique_seed = f"{class_name}-{class_name_counts[class_name]}"
    
    class_name_counts[class_name] += 1
    return unique_seed

def sanitize_filename(filename: str) -> str:
    """Sanitize filename - for numeric class_names this should be minimal."""
    # For string numbers like '0019891', just ensure it's clean
    if filename.isdigit():
        return filename
    
    # Fallback: replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove any remaining problematic characters
    filename = ''.join(c for c in filename if c.isprintable())
    
    # Limit length to avoid filesystem issues
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def transform_dataset(
    output_dir: str,
    progress_file: str = "transform_progress.pkl",
    resume: bool = True,
    batch_size: int = 1000
):
    """Transform HuggingFace dataset to InstructPix2Pix format."""
    
    # Load or initialize progress
    if resume:
        progress = load_progress(progress_file)
        print(f"Resuming from {len(progress['processed_indices'])} processed examples")
    else:
        progress = {
            'processed_indices': set(),
            'class_name_counts': defaultdict(int),
            'prompt_counter': 0,
            'seeds_mapping': {}
        }
        print("Starting fresh transformation")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset features
    features = Features({
        "input_image": Image(),
        "output_image": Image(),
        "original_caption": Value("string"),
        "edit_instruction": Value("string"),
        "resulting_caption": Value("string"),
        "class_name": Value("string"),
        "status": Value("string"),
    })
    
    # Load dataset from parquet files
    print("Loading dataset from parquet files...")
    dataset = DatasetDict({
        "train": load_dataset("parquet", 
            data_files={"train": "/sc/home/thomas.chille/.cache/hf/hf/data/train/*.parquet"}, 
            features=features)["train"],
        "val": load_dataset("parquet", 
            data_files={"val": "/sc/home/thomas.chille/.cache/hf/hf/data/val/*.parquet"}, 
            features=features)["val"],
        "test": load_dataset("parquet", 
            data_files={"test": "/sc/home/thomas.chille/.cache/hf/hf/data/test/*.parquet"}, 
            features=features)["test"],
    })
    
    print(f"Dataset loaded: {sum(len(split) for split in dataset.values())} total examples")
    
    # Process each split
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} examples)...")
        
        # Process in batches to avoid memory issues
        total_examples = len(split_data)
        num_batches = (total_examples + batch_size - 1) // batch_size
        
        # Find the first batch that needs processing
        start_batch_idx = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_examples)
            
            # Check if entire batch is already processed
            batch_processed_count = 0
            for idx in range(start_idx, end_idx):
                example_id = f"{split_name}_{idx}"
                if example_id in progress['processed_indices']:
                    batch_processed_count += 1
            
            # If batch is not fully processed, start from here
            if batch_processed_count < (end_idx - start_idx):
                start_batch_idx = batch_idx
                break
            else:
                print(f"Skipping batch {batch_idx + 1}/{num_batches} (examples {start_idx}-{end_idx-1}) - already processed")
        
        # Process from the first unprocessed batch onwards
        for batch_idx in range(start_batch_idx, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_examples)
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (examples {start_idx}-{end_idx-1})...")
            
            # Get batch slice
            batch_data = split_data.select(range(start_idx, end_idx))
            
            batch_processed = 0
            batch_errors = 0
            
            try:
                for local_idx, example in enumerate(batch_data):
                    idx = start_idx + local_idx
                    # Create unique identifier for this example
                    example_id = f"{split_name}_{idx}"
                    
                    # Skip if already processed
                    if example_id in progress['processed_indices']:
                        continue
                    
                    try:
                        # Skip removed records
                        if example.get('status') == 'removed':
                            print(f"Skipping removed record: {example_id}")
                            progress['processed_indices'].add(example_id)
                            continue
                        
                        # Get class name (should be string number like '0019891')
                        class_name = example['class_name']
                        if not class_name:
                            class_name = f"{idx:07d}"  # Use zero-padded index as fallback
                        
                        # Sanitize class name for filesystem (minimal for numeric strings)
                        class_name = sanitize_filename(class_name)
                        
                        # Generate unique seed
                        unique_seed = make_unique_seed(class_name, progress['class_name_counts'])
                        
                        # Create prompt directory
                        prompt_dir = output_path / f"{progress['prompt_counter']:07d}"
                        prompt_dir.mkdir(exist_ok=True)
                        
                        # Save images
                        input_image = example['input_image']
                        output_image = example['output_image']
                        
                        if input_image and output_image:
                            # Convert to PIL if needed and save
                            if hasattr(input_image, 'save'):
                                input_image.save(prompt_dir / f"{unique_seed}_0.jpg", "JPEG")
                            else:
                                PILImage.fromarray(input_image).save(prompt_dir / f"{unique_seed}_0.jpg", "JPEG")
                            
                            if hasattr(output_image, 'save'):
                                output_image.save(prompt_dir / f"{unique_seed}_1.jpg", "JPEG")
                            else:
                                PILImage.fromarray(output_image).save(prompt_dir / f"{unique_seed}_1.jpg", "JPEG")
                        
                        # Create prompt.json
                        prompt_data = {
                            "input": example['original_caption'] or "",
                            "edit": example['edit_instruction'] or "",
                            "output": example['resulting_caption'] or ""
                        }
                        
                        with open(prompt_dir / "prompt.json", 'w') as f:
                            json.dump(prompt_data, f, indent=2)
                        
                        # Store mapping for seeds.json
                        progress['seeds_mapping'][f"{progress['prompt_counter']:07d}"] = [unique_seed]
                        
                        # Update counters
                        progress['prompt_counter'] += 1
                        progress['processed_indices'].add(example_id)
                        batch_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing example {example_id}: {e}")
                        batch_errors += 1
                        continue
                
                # Log batch completion with detailed stats
                print(f"Batch {batch_idx + 1}/{num_batches} completed:")
                print(f"  - Processed: {batch_processed} examples")
                print(f"  - Errors: {batch_errors} examples")
                print(f"  - Total processed so far: {len(progress['processed_indices'])}")
                print(f"  - Overall progress: {len(progress['processed_indices'])}/{total_examples} ({100*len(progress['processed_indices'])/total_examples:.1f}%)")
                
                # Save progress after each batch
                save_progress(progress_file, progress)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                # Save progress even on batch error
                save_progress(progress_file, progress)
                continue
    
    # Generate seeds.json
    print("\nGenerating seeds.json...")
    seeds_data = []
    for prompt_dir, seeds in progress['seeds_mapping'].items():
        seeds_data.append([prompt_dir, seeds])
    
    # Sort by prompt directory number
    seeds_data.sort(key=lambda x: int(x[0]))
    
    with open(output_path / "seeds.json", 'w') as f:
        json.dump(seeds_data, f, indent=2)
    
    # Final progress save
    save_progress(progress_file, progress)
    
    print(f"\nTransformation complete!")
    print(f"Total examples processed: {len(progress['processed_indices'])}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt directories created: {progress['prompt_counter']}")
    print(f"Seeds.json created with {len(seeds_data)} entries")

def main():
    parser = argparse.ArgumentParser(description="Transform HuggingFace dataset to InstructPix2Pix format")
    parser.add_argument("--output_dir", "-o", required=True, 
                       help="Output directory for InstructPix2Pix dataset")
    parser.add_argument("--progress_file", "-p", default="transform_progress.pkl",
                       help="Progress file for resume functionality")
    parser.add_argument("--no_resume", action="store_true",
                       help="Start fresh instead of resuming")
    parser.add_argument("--batch_size", "-b", type=int, default=1000,
                       help="Batch size for processing (default: 1000)")
    
    args = parser.parse_args()
    
    transform_dataset(
        output_dir=args.output_dir,
        progress_file=args.progress_file,
        resume=not args.no_resume,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()