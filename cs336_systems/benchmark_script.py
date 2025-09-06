"""
Benchmarking script for forward and backward passes of Transformer models.

This script supports:
- Initializing models with given hyperparameters
- Generating random batch data
- Running warm-up steps before timing
- Timing forward-only or forward+backward passes
- Using torch.cuda.synchronize() for accurate GPU timing
"""

import argparse
import timeit
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

from cs336_basics.transformers import TransformerLM


def create_model(
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 2048,
    vocab_size: int = 10000,
    context_length: int = 512,
    device: str = "cuda"
) -> TransformerLM:
    """Initialize a Transformer model with given hyperparameters."""
    model = TransformerLM(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        theta=10000.0,  # RoPE theta parameter
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=n_layers
    )
    model = model.to(device)
    return model


def create_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random input and target tensors."""
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


def benchmark_forward_only(
    model: TransformerLM,
    x: torch.Tensor,
    n_steps: int,
    warmup_steps: int = 5
) -> float:
    """Benchmark forward pass only."""
    device = next(model.parameters()).device
    
    # Warm-up
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Timing
    print(f"Timing {n_steps} forward-only steps...")
    start_time = timeit.default_timer()
    
    for _ in range(n_steps):
        with torch.no_grad():
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    return total_time


def benchmark_forward_backward(
    model: TransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    warmup_steps: int = 5
) -> float:
    """Benchmark forward and backward passes."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    
    # Warm-up
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Timing
    print(f"Timing {n_steps} forward+backward steps...")
    start_time = timeit.default_timer()
    
    for _ in range(n_steps):
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    return total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer model forward/backward passes")
    
    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    
    # Batch parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Benchmark parameters
    parser.add_argument("--n_steps", type=int, default=10, help="Number of steps to time")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run for statistics")
    parser.add_argument("--forward_only", action="store_true", help="Benchmark forward pass only")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print("=== Transformer Model Benchmarking ===")
    print(f"Model config: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"Data config: batch_size={args.batch_size}, context_length={args.context_length}, vocab_size={args.vocab_size}")
    print(f"Benchmark config: n_steps={args.n_steps}, warmup_steps={args.warmup_steps}, n_trials={args.n_trials}, device={args.device}")
    print(f"Mode: {'Forward only' if args.forward_only else 'Forward + Backward'}")
    print()
    
    # Initialize model
    print("Initializing model...")
    model = create_model(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=args.device
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # Create random batch
    print("Creating random batch...")
    x, y = create_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=args.device
    )
    
    # Run multiple trials
    trial_times = []
    for trial in range(args.n_trials):
        if args.n_trials > 1:
            print(f"\n--- Trial {trial + 1}/{args.n_trials} ---")
        
        if args.forward_only:
            total_time = benchmark_forward_only(model, x, args.n_steps, args.warmup_steps)
        else:
            total_time = benchmark_forward_backward(model, x, y, args.n_steps, args.warmup_steps)
        
        avg_time = total_time / args.n_steps
        trial_times.append(avg_time)
        
        if args.n_trials > 1:
            print(f"Trial {trial + 1} average time per step: {avg_time:.4f} seconds")
    
    # Calculate statistics
    mean_time = np.mean(trial_times)
    std_time = np.std(trial_times, ddof=1) if len(trial_times) > 1 else 0.0
    
    # Report results
    print(f"\n=== Results ===")
    if args.n_trials > 1:
        print(f"Number of trials: {args.n_trials}")
        print(f"Mean time per step: {mean_time:.4f} ± {std_time:.4f} seconds")
        print(f"Standard deviation: {std_time:.4f} seconds")
        print(f"Coefficient of variation: {(std_time/mean_time)*100:.2f}%")
        print(f"Mean steps per second: {1/mean_time:.2f}")
        print(f"Individual trial times: {[f'{t:.4f}' for t in trial_times]}")
    else:
        print(f"Average time per step: {mean_time:.4f} seconds")
        print(f"Steps per second: {1/mean_time:.2f}")


if __name__ == "__main__":
    main()