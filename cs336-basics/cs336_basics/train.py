#!/usr/bin/env python3
"""
Training script for transformer language model.

This script implements a complete training pipeline including:
- Configurable model and optimizer hyperparameters via command line
- Memory-efficient dataset loading with np.memmap
- Checkpoint saving and loading
- Periodic logging of training and validation metrics
- Learning rate scheduling
- Gradient clipping and accumulation
"""

import argparse
import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
import numpy as np

from .transformers import (
    TransformerLM,
    AdamWOptimizer,
    cross_entropy,
    gradient_clipping,
    cosine_learning_rate_schedule,
    get_batch,
    load_dataset,
    save_checkpoint,
    load_checkpoint,
)


def setup_logging(use_wandb: bool = False, project_name: str = "transformer-lm", **wandb_kwargs):
    """Setup logging to console and optionally to Weights & Biases."""

    logger = {"use_wandb": use_wandb, "step": 0}

    if use_wandb:
        try:
            import wandb

            wandb.init(project=project_name, **wandb_kwargs)
            logger["wandb"] = wandb
            print("✓ Weights & Biases logging enabled")
        except ImportError:
            print("⚠ wandb not available, falling back to console logging only")
            logger["use_wandb"] = False

    return logger


def log_metrics(logger: Dict[str, Any], metrics: Dict[str, float], step: int):
    """Log metrics to console and optionally to wandb."""

    # Console logging
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Step {step}: {metrics_str}")

    # Wandb logging
    if logger["use_wandb"]:
        logger["wandb"].log(metrics, step=step)


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a transformer language model")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed forward dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    parser.add_argument("--disable_norm", action="store_true", help="Disable RMSNorm layers (ablation study)")
    parser.add_argument("--disable_rope", action="store_true", help="Disable RoPE position encoding (ablation study)")
    parser.add_argument("--post_norm", action="store_true", help="Use post-norm instead of pre-norm (ablation study)")
    parser.add_argument("--use_silu", action="store_true", help="Use standard SiLU feed-forward instead of SwiGLU (ablation study)")
    parser.add_argument("--use_nvtx", action="store_true", help="Use NVTX annotated scaled_dot_product_attention for profiling")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Learning rate warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--cosine_steps", type=int, default=90000, help="Steps for cosine annealing")

    # Data and I/O
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (.npy)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")

    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Steps between evaluation")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Steps between checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")

    # System
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"], help="Model dtype")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")

    return parser.parse_args()


def load_datasets(train_path: str, val_path: str):
    """Load training and validation datasets with memory mapping."""
    print(f"Loading datasets...")

    train_data = load_dataset(train_path, mmap_mode="r")
    val_data = load_dataset(val_path, mmap_mode="r")

    print(f"✓ Training data: {len(train_data):,} tokens")
    print(f"✓ Validation data: {len(val_data):,} tokens")

    return train_data, val_data


def create_model_and_optimizer(args) -> tuple[TransformerLM, AdamWOptimizer]:
    """Create model and optimizer based on arguments."""

    # Determine dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    normalization_status = "disabled" if args.disable_norm else "enabled"
    rope_status = "disabled" if args.disable_rope else "enabled"
    norm_type = "post-norm" if args.post_norm else "pre-norm"
    activation_type = "SiLU" if args.use_silu else "SwiGLU"
    nvtx_status = "enabled" if args.use_nvtx else "disabled"
    
    # Only show norm type if normalization is enabled
    norm_info = f"RMSNorm {normalization_status}" + (f" ({norm_type})" if not args.disable_norm else "")
    print(f"Creating model with {args.num_layers} layers, {args.d_model} dim, {args.num_heads} heads ({norm_info}, RoPE {rope_status}, {activation_type} activation, NVTX {nvtx_status})...")

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        use_norm=not args.disable_norm,
        use_rope=not args.disable_rope,
        pre_norm=not args.post_norm,
        use_swiglu=not args.use_silu,
        use_nvtx=args.use_nvtx,
    ).to(device=args.device, dtype=dtype)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model created: {total_params:,} total params, {trainable_params:,} trainable")

    # Create optimizer
    optimizer = AdamWOptimizer(
        model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )

    # Optional: compile model for optimization
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    return model, optimizer


def evaluate_model(model: TransformerLM, val_data, args, max_eval_batches: int = 100) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for _ in range(max_eval_batches):
            try:
                # Sample validation batch
                x, y = get_batch(val_data, args.batch_size, args.context_length, args.device)

                # Forward pass
                logits = model(x)  # [batch_size, seq_len, vocab_size]

                # Reshape for cross entropy: [batch_size * seq_len, vocab_size] and [batch_size * seq_len]
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)

                # Compute loss
                loss = cross_entropy(logits_flat, targets_flat)

                total_loss += loss.item()
                total_tokens += y.numel()
                num_batches += 1

            except Exception as e:
                print(f"Warning: Evaluation batch failed: {e}")
                break

    model.train()

    if num_batches == 0:
        return {"val_loss": float("inf"), "val_perplexity": float("inf")}

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return {"val_loss": avg_loss, "val_perplexity": perplexity, "val_tokens": total_tokens}


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 Starting transformer language model training")
    print("=" * 60)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Save training configuration
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Configuration saved to {config_path}")

    # Setup logging
    logger = setup_logging(
        use_wandb=args.use_wandb, project_name=args.wandb_project, name=args.wandb_run_name, config=vars(args)
    )

    # Load datasets
    train_data, val_data = load_datasets(args.train_data, args.val_data)

    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(args)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"✓ Resumed from step {start_step}")

    print(f"\n🎯 Training configuration:")
    print(f"   • Steps: {start_step} → {args.max_steps}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Learning rate: {args.learning_rate} → {args.min_learning_rate}")
    print(f"   • Device: {args.device}")
    print(f"   • Dtype: {args.dtype}")
    print()

    # Training loop
    model.train()
    step = start_step
    start_time = time.time()

    while step < args.max_steps:
        # Get learning rate for current step
        lr = cosine_learning_rate_schedule(
            step, args.learning_rate, args.min_learning_rate, args.warmup_steps, args.cosine_steps
        )

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample training batch
        try:
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        except Exception as e:
            print(f"Error sampling batch at step {step}: {e}")
            step += 1
            continue

        # Forward pass
        logits = model(x)  # [batch_size, seq_len, vocab_size]

        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)

        # Compute loss
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        step += 1

        # Logging
        if step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            tokens_per_sec = (step - start_step) * args.batch_size * args.context_length / elapsed_time

            metrics = {
                "train_loss": loss.item(),
                "learning_rate": lr,
                "tokens_per_second": tokens_per_sec,
                "elapsed_time": elapsed_time,
            }

            log_metrics(logger, metrics, step)

        # Evaluation
        if step % args.eval_interval == 0:
            print(f"\n📊 Evaluating at step {step}...")
            eval_metrics = evaluate_model(model, val_data, args)
            log_metrics(logger, eval_metrics, step)
            print()

        # Checkpointing
        if step % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")

    # Final evaluation
    print(f"\n🎉 Training completed! Running final evaluation...")
    final_metrics = evaluate_model(model, val_data, args)
    log_metrics(logger, final_metrics, step)

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, step, final_checkpoint_path)
    print(f"💾 Final checkpoint saved to {final_checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n✅ Training finished in {total_time:.1f}s ({total_time / 3600:.2f}h)")

    if logger["use_wandb"]:
        logger["wandb"].finish()


if __name__ == "__main__":
    main()
