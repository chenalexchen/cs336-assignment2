#!/usr/bin/env python3
"""
Test very large sequence lengths with small d_model
"""

import torch
from .flash_attention import FlashAttentionTriton

def test_large_sequences():
    """Test sequence lengths beyond 8K with optimal parameters"""

    print("🚀 Testing Large Sequence Lengths (>8K)")
    print("=" * 60)

    # Use parameters that worked well
    d_model = 64
    batch_size = 1  # Start with batch_size=1 to maximize sequence length

    # Test larger sequence lengths
    test_lengths = [1048576, 2097152, 4194304]  # 1M, 2M, 4M

    for seq_len in test_lengths:
        print(f"\n📏 Testing seq_len={seq_len:,} (d_model={d_model}, batch_size={batch_size})")

        try:
            torch.manual_seed(42)
            Q = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16, device='cuda', requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16, device='cuda', requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16, device='cuda', requires_grad=True)
            grad_O = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16, device='cuda')

            # Check memory before
            mem_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory before: {mem_before:.2f} GB")

            # Forward pass
            output = FlashAttentionTriton.apply(Q, K, V, True)
            torch.cuda.synchronize()

            mem_after_fwd = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory after forward: {mem_after_fwd:.2f} GB")

            # Backward pass
            output.backward(grad_O, retain_graph=True)
            torch.cuda.synchronize()

            mem_after_bwd = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory after backward: {mem_after_bwd:.2f} GB")

            print(f"  ✅ seq_len={seq_len:,} SUCCESS!")

            # Clean up
            del Q, K, V, grad_O, output
            torch.cuda.empty_cache()

        except Exception as e:
            error_str = str(e)
            if "out of memory" in error_str.lower():
                print(f"  ❌ seq_len={seq_len:,} FAILED: Out of memory")
            elif "shared memory" in error_str.lower():
                print(f"  ❌ seq_len={seq_len:,} FAILED: Shared memory limit")
            else:
                print(f"  ❌ seq_len={seq_len:,} FAILED: {error_str[:100]}")

            torch.cuda.empty_cache()
            break

if __name__ == "__main__":
    test_large_sequences()