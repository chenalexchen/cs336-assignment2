#!/usr/bin/env python3
"""
Test how batch size and d_model affect maximum sequence length
"""

import torch
import traceback
from .flash_attention import FlashAttentionTriton

def test_config(seq_len, d_model, batch_size, dtype=torch.bfloat16):
    """Test a specific configuration"""
    try:
        # Generate inputs
        torch.manual_seed(42)
        Q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        grad_O = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda')

        # Forward pass
        output = FlashAttentionTriton.apply(Q, K, V, True)
        torch.cuda.synchronize()

        # Backward pass
        output.backward(grad_O, retain_graph=True)
        torch.cuda.synchronize()

        # Clean up
        del Q, K, V, grad_O, output
        torch.cuda.empty_cache()

        return True, "SUCCESS"

    except Exception as e:
        error_str = str(e)
        # Clean up on failure
        torch.cuda.empty_cache()

        if "out of memory" in error_str.lower():
            return False, "OOM"
        elif "shared memory" in error_str.lower():
            return False, "Shared Memory"
        else:
            return False, f"Other: {error_str[:50]}"

def find_max_seq_for_config(d_model, batch_size):
    """Find maximum sequence length for given d_model and batch_size"""
    print(f"\n🔍 Testing d_model={d_model}, batch_size={batch_size}")

    # Binary search for maximum sequence length
    low, high = 32, 8192
    max_working = 0

    while low <= high:
        mid = (low + high) // 2
        success, error = test_config(mid, d_model, batch_size)

        print(f"  seq_len={mid:4d}: {'✅' if success else '❌'} {error}")

        if success:
            max_working = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_working

def main():
    print("🧪 Testing Memory Scaling for FlashAttention Triton")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test different configurations
    configs = [
        # (d_model, batch_size)
        (64, 1),
        (64, 4),
        (64, 16),
        (128, 1),
        (128, 4),
        (128, 16),
        (256, 1),
        (256, 4),
        (256, 16),
        (512, 1),
        (512, 4),
    ]

    results = []

    for d_model, batch_size in configs:
        max_seq = find_max_seq_for_config(d_model, batch_size)
        results.append((d_model, batch_size, max_seq))

    print("\n" + "=" * 80)
    print("📊 SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'d_model':<8} {'batch_size':<11} {'max_seq_len':<12} {'total_tokens':<12}")
    print("-" * 50)

    for d_model, batch_size, max_seq in results:
        total_tokens = batch_size * max_seq
        print(f"{d_model:<8} {batch_size:<11} {max_seq:<12} {total_tokens:<12}")

    # Look for the sweet spot for 128K tokens
    print(f"\n🎯 CONFIGURATIONS THAT SUPPORT ~128K TOTAL TOKENS:")
    target = 128 * 1024
    for d_model, batch_size, max_seq in results:
        total_tokens = batch_size * max_seq
        if total_tokens >= target:
            print(f"  d_model={d_model}, batch_size={batch_size}, max_seq={max_seq} -> {total_tokens:,} tokens")

if __name__ == "__main__":
    main()