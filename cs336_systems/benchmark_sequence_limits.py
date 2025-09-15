#!/usr/bin/env python3
"""
Test maximum sequence length support for FlashAttention Triton implementation
"""

import torch
import traceback
from .flash_attention import FlashAttentionTriton

def test_sequence_length(seq_len, d_model=256, batch_size=16, dtype=torch.bfloat16):
    """Test a specific sequence length"""
    print(f"Testing seq_len={seq_len}, d_model={d_model}, batch_size={batch_size}, dtype={dtype}")

    try:
        # Generate inputs
        torch.manual_seed(42)
        Q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
        grad_O = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda')

        print(f"  Memory before forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Forward pass
        output = FlashAttentionTriton.apply(Q, K, V, True)
        torch.cuda.synchronize()

        print(f"  Memory after forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Forward pass: SUCCESS")

        # Backward pass
        output.backward(grad_O, retain_graph=True)
        torch.cuda.synchronize()

        print(f"  Memory after backward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Backward pass: SUCCESS")
        print(f"  ✅ seq_len={seq_len} PASSED")

        # Clean up
        del Q, K, V, grad_O, output
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        error_str = str(e)
        if "out of memory" in error_str.lower() or "oom" in error_str.lower():
            print(f"  ❌ seq_len={seq_len} FAILED: Out of memory")
        elif "shared memory" in error_str.lower():
            print(f"  ❌ seq_len={seq_len} FAILED: Shared memory limit")
        else:
            print(f"  ❌ seq_len={seq_len} FAILED: {error_str}")
            # Print full traceback for debugging
            traceback.print_exc()

        # Clean up on failure
        torch.cuda.empty_cache()
        return False

def find_maximum_sequence_length():
    """Binary search to find maximum supported sequence length"""

    print("🔍 Finding maximum sequence length support for FlashAttention Triton")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Test smaller lengths first due to larger batch size and d_model
    test_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]  # 128 to 8K

    max_working = 64  # Start very conservatively with larger batch size and d_model

    for seq_len in test_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        print("-" * 40)

        success = test_sequence_length(seq_len)

        if success:
            max_working = seq_len
            print(f"✅ {seq_len} works!")
        else:
            print(f"❌ {seq_len} failed - stopping here")
            break

    print("\n" + "=" * 80)
    print(f"🏁 RESULTS:")
    print(f"Maximum working sequence length: {max_working:,}")
    print(f"This corresponds to ~{max_working/1024:.1f}K tokens")

    # Test a few more specific lengths if we have room
    if max_working >= 8192:
        print(f"\n🔬 Testing intermediate lengths...")

        # Test some values between max_working and the failed length
        if max_working == 8192:
            candidates = [10240, 12288, 14336]  # 10K, 12K, 14K
        elif max_working == 16384:
            candidates = [20480, 24576, 28672]  # 20K, 24K, 28K
        elif max_working == 32768:
            candidates = [40960, 49152, 57344]  # 40K, 48K, 56K
        elif max_working == 65536:
            candidates = [81920, 98304, 114688]  # 80K, 96K, 112K
        else:
            candidates = []

        for seq_len in candidates:
            print(f"\n📏 Testing sequence length: {seq_len}")
            success = test_sequence_length(seq_len)
            if success:
                max_working = seq_len
                print(f"✅ {seq_len} works!")
            else:
                print(f"❌ {seq_len} failed")
                break

    print("\n" + "=" * 80)
    print(f"🎯 FINAL RESULT:")
    print(f"Maximum supported sequence length: {max_working:,}")
    print(f"This is {max_working/1024:.1f}K tokens")
    print("=" * 80)

    return max_working

if __name__ == "__main__":
    find_maximum_sequence_length()