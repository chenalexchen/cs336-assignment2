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
import torch.cuda.nvtx as nvtx
import numpy as np
from typing import Optional, Tuple, List

from cs336_basics.transformers import TransformerLM, AdamWOptimizer


def create_model(
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 2048,
    vocab_size: int = 10000,
    context_length: int = 512,
    device: str = "cuda",
    use_nvtx: bool = False
) -> TransformerLM:
    """Initialize a Transformer model with given hyperparameters."""
    model = TransformerLM(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        theta=10000.0,  # RoPE theta parameter
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=n_layers,
        use_nvtx=use_nvtx
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
    warmup_steps: int = 5,
    use_nvtx: bool = False,
    use_mixed_precision: bool = False,
    memory_profile: bool = False
) -> float:
    """Benchmark forward pass only."""
    device = next(model.parameters()).device
    
    # Warm-up
    print(f"Running {warmup_steps} warm-up steps...")
    if use_nvtx:
        with nvtx.range("warmup"):
            for step in range(warmup_steps):
                with nvtx.range(f"warmup_step_{step}"):
                    with torch.no_grad():
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                _ = model(x)
                        else:
                            _ = model(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
    else:
        for step in range(warmup_steps):
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        _ = model(x)
                else:
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
    
    # Timing
    print(f"Timing {n_steps} forward-only steps...")
    
    # Start memory profiling if requested (after warm-up)
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    start_time = timeit.default_timer()
    
    if use_nvtx:
        with nvtx.range("forward_timing"):
            for step in range(n_steps):
                with nvtx.range(f"forward_step_{step}"):
                    with torch.no_grad():
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                _ = model(x)
                        else:
                            _ = model(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
    else:
        for step in range(n_steps):
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        _ = model(x)
                else:
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    # Save memory snapshot and stop profiling if requested
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    return total_time


def benchmark_forward_backward(
    model: TransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    warmup_steps: int = 5,
    use_nvtx: bool = False,
    use_mixed_precision: bool = False,
    memory_profile: bool = False
) -> float:
    """Benchmark forward and backward passes."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Warm-up
    print(f"Running {warmup_steps} warm-up steps...")
    if use_nvtx:
        with nvtx.range("warmup"):
            for step in range(warmup_steps):
                with nvtx.range(f"warmup_step_{step}"):
                    model.zero_grad()
                    with nvtx.range("forward"):
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(x)
                                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        else:
                            logits = model(x)
                            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    with nvtx.range("backward"):
                        if use_mixed_precision:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
    else:
        for step in range(warmup_steps):
            model.zero_grad()
            if use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Timing
    print(f"Timing {n_steps} forward+backward steps...")
    
    # Start memory profiling if requested (after warm-up)
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    start_time = timeit.default_timer()
    
    if use_nvtx:
        with nvtx.range("forward_backward_timing"):
            for step in range(n_steps):
                with nvtx.range(f"train_step_{step}"):
                    model.zero_grad()
                    with nvtx.range("forward"):
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(x)
                                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        else:
                            logits = model(x)
                            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    with nvtx.range("backward"):
                        if use_mixed_precision:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
    else:
        for step in range(n_steps):
            model.zero_grad()
            if use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    # Save memory snapshot and stop profiling if requested
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    return total_time


def benchmark_forward_backward_optimizer(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
    warmup_steps: int = 5,
    use_nvtx: bool = False,
    use_mixed_precision: bool = False,
    memory_profile: bool = False
) -> float:
    """Benchmark forward, backward, and optimizer step."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Warm-up
    print(f"Running {warmup_steps} warm-up steps...")
    if use_nvtx:
        with nvtx.range("warmup"):
            for step in range(warmup_steps):
                with nvtx.range(f"warmup_step_{step}"):
                    optimizer.zero_grad()
                    with nvtx.range("forward"):
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(x)
                                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        else:
                            logits = model(x)
                            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    with nvtx.range("backward"):
                        if use_mixed_precision:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    with nvtx.range("optimizer"):
                        if use_mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
    else:
        for step in range(warmup_steps):
            optimizer.zero_grad()
            if use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Timing
    print(f"Timing {n_steps} forward+backward+optimizer steps...")
    
    # Start memory profiling if requested (after warm-up)
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    start_time = timeit.default_timer()
    
    if use_nvtx:
        with nvtx.range("forward_backward_optimizer_timing"):
            for step in range(n_steps):
                with nvtx.range(f"train_step_{step}"):
                    optimizer.zero_grad()
                    with nvtx.range("forward"):
                        if use_mixed_precision:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(x)
                                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        else:
                            logits = model(x)
                            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    with nvtx.range("backward"):
                        if use_mixed_precision:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    with nvtx.range("optimizer"):
                        if use_mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
    else:
        for step in range(n_steps):
            optimizer.zero_grad()
            if use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    # Save memory snapshot and stop profiling if requested
    if memory_profile and device.type == "cuda":
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
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
    parser.add_argument("--with_optimizer", action="store_true", help="Include optimizer step in backward benchmark")
    parser.add_argument("--use_nvtx", action="store_true", help="Enable NVTX annotations for profiling")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Enable mixed precision training with autocast")
    parser.add_argument("--memory_profile", action="store_true", help="Enable memory profiling with snapshot")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Determine benchmark mode
    if args.forward_only:
        mode = "Forward only"
    elif args.with_optimizer:
        mode = "Forward + Backward + Optimizer"
    else:
        mode = "Forward + Backward"
    
    print("=== Transformer Model Benchmarking ===")
    print(f"Model config: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"Data config: batch_size={args.batch_size}, context_length={args.context_length}, vocab_size={args.vocab_size}")
    print(f"Benchmark config: n_steps={args.n_steps}, warmup_steps={args.warmup_steps}, n_trials={args.n_trials}, device={args.device}")
    print(f"Mode: {mode}")
    print(f"NVTX annotations: {'Enabled' if args.use_nvtx else 'Disabled'}")
    print(f"Mixed precision: {'Enabled' if args.use_mixed_precision else 'Disabled'}")
    print(f"Memory profiling: {'Enabled' if args.memory_profile else 'Disabled'}")
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
        device=args.device,
        use_nvtx=args.use_nvtx
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
    
    # Create optimizer if needed
    optimizer = None
    if args.with_optimizer and not args.forward_only:
        print("Creating optimizer...")
        optimizer = AdamWOptimizer(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Run multiple trials
    trial_times = []
    for trial in range(args.n_trials):
        if args.n_trials > 1:
            print(f"\n--- Trial {trial + 1}/{args.n_trials} ---")
        
        if args.forward_only:
            total_time = benchmark_forward_only(model, x, args.n_steps, args.warmup_steps, args.use_nvtx, args.use_mixed_precision, args.memory_profile)
        elif args.with_optimizer:
            total_time = benchmark_forward_backward_optimizer(model, optimizer, x, y, args.n_steps, args.warmup_steps, args.use_nvtx, args.use_mixed_precision, args.memory_profile)
        else:
            total_time = benchmark_forward_backward(model, x, y, args.n_steps, args.warmup_steps, args.use_nvtx, args.use_mixed_precision, args.memory_profile)
        
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