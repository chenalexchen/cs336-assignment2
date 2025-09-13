#!/usr/bin/env python3
"""
Convert tokenized text files to NumPy arrays for efficient training.
"""

import numpy as np
import os
from pathlib import Path

def convert_tokens_to_npy(input_file: str, output_file: str):
    """Convert token ID file to NumPy array."""
    print(f"Converting {input_file} -> {output_file}")
    
    # Check file size to decide processing approach
    file_size = os.path.getsize(input_file)
    print(f"  Input file size: {file_size / 1024 / 1024 / 1024:.2f}GB")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if file_size > 1024 * 1024 * 1024:  # > 1GB, use memory-efficient approach
        print("  Using memory-efficient streaming conversion...")
        
        # Count tokens first
        token_count = 0
        with open(input_file, 'r') as f:
            for chunk in iter(lambda: f.read(8192), ''):
                token_count += chunk.count(' ') + (1 if chunk and not chunk[-1].isspace() else 0)
        
        print(f"  Estimated tokens: {token_count:,}")
        
        # Create memory-mapped array
        tokens_array = np.memmap(output_file.replace('.npy', '_temp.dat'), 
                                dtype=np.uint32, mode='w+', shape=(token_count,))
        
        # Stream tokens directly to memory-mapped array
        idx = 0
        with open(input_file, 'r') as f:
            buffer = ''
            for chunk in iter(lambda: f.read(8192), ''):
                buffer += chunk
                while ' ' in buffer:
                    token_str, buffer = buffer.split(' ', 1)
                    if token_str.strip():
                        tokens_array[idx] = int(token_str)
                        idx += 1
                        if idx % 1000000 == 0:
                            print(f"    Processed {idx:,} tokens...")
            
            # Handle last token
            if buffer.strip():
                tokens_array[idx] = int(buffer.strip())
                idx += 1
        
        # Save as proper .npy file
        actual_tokens = tokens_array[:idx]  # Trim to actual size
        np.save(output_file, actual_tokens)
        
        # Clean up temp file
        os.remove(output_file.replace('.npy', '_temp.dat'))
        
        print(f"  Saved {idx:,} tokens to {output_file} ({idx * 4:,} bytes)")
        return idx
        
    else:
        # Small file, use original approach
        with open(input_file, 'r') as f:
            tokens = [int(x) for x in f.read().split()]
        
        print(f"  Loaded {len(tokens):,} tokens")
        
        # Convert to NumPy array with appropriate dtype
        tokens_array = np.array(tokens, dtype=np.uint32)
        
        # Save as .npy file
        np.save(output_file, tokens_array)
        
        print(f"  Saved to {output_file} ({tokens_array.nbytes:,} bytes)")
        return len(tokens)

def main():
    """Convert both training and validation datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert tokenized datasets to NumPy format")
    parser.add_argument("--dataset", type=str, default="tiny_stories_10000", 
                       help="Dataset directory name (default: tiny_stories_10000)")
    parser.add_argument("--validation-only", action="store_true", 
                       help="Only convert validation dataset")
    parser.add_argument("--train-only", action="store_true", 
                       help="Only convert training dataset")
    
    args = parser.parse_args()
    
    # Define paths
    base_path = f"training_data/{args.dataset}"
    
    train_ids = f"{base_path}/train/tokens.ids"
    train_npy = f"{base_path}/train/tokens.npy"
    
    val_ids = f"{base_path}/validation/tokens.ids"
    val_npy = f"{base_path}/validation/tokens.npy"
    
    print(f"🔄 Converting tokenized datasets to NumPy format")
    print(f"Dataset: {args.dataset}")
    print("=" * 50)
    
    total_tokens = 0
    
    # Convert validation dataset
    if not args.train_only:
        if os.path.exists(val_ids):
            print("Converting validation dataset...")
            val_tokens = convert_tokens_to_npy(val_ids, val_npy)
            total_tokens += val_tokens
        else:
            print(f"⚠️  Validation file not found: {val_ids}")
    
    # Convert training dataset
    if not args.validation_only:
        if os.path.exists(train_ids):
            print("Converting training dataset...")
            train_tokens = convert_tokens_to_npy(train_ids, train_npy)
            total_tokens += train_tokens
        else:
            print(f"⚠️  Training file not found: {train_ids}")
            print("   Waiting for tokenization to complete...")
    
    print("=" * 50)
    print(f"✅ Conversion complete! Total tokens: {total_tokens:,}")
    
    if os.path.exists(train_npy) and os.path.exists(val_npy):
        print(f"📁 Ready for training:")
        print(f"   Training data: {train_npy}")
        print(f"   Validation data: {val_npy}")

if __name__ == "__main__":
    main()