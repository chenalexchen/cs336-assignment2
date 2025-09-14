#!/usr/bin/env python3
"""
Full FlashAttention-2 Benchmark Sweep
Covers all specified configurations with adaptive tile sizing
"""

import torch
import triton
import triton.testing
import math
import pandas as pd
import warnings
from cs336_systems.flash_attention import FlashAttentionTriton, flash_fwd_kernel
from typing import Tuple
warnings.filterwarnings("ignore")


class StandardAttentionAutograd(torch.autograd.Function):
    """Standard PyTorch attention implementation"""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        scale = 1.0 / math.sqrt(Q.shape[-1])
        S = torch.einsum('bqd,bkd->bqk', Q, K) * scale

        if is_causal:
            n_q, n_k = Q.shape[-2], K.shape[-2]
            q_idx = torch.arange(n_q, device=Q.device)[:, None]
            k_idx = torch.arange(n_k, device=Q.device)[None, :]
            causal_mask = q_idx >= k_idx
            S = S.masked_fill(~causal_mask, -torch.inf)

        P = torch.softmax(S, dim=-1)
        O = torch.einsum('bqk,bkd->bqd', P, V)

        ctx.save_for_backward(Q, K, V, P)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, P = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        grad_V = torch.einsum('bqk,bqd->bkd', P, grad_O)
        grad_P = torch.einsum('bqd,bkd->bqk', grad_O, V)
        grad_S = P * (grad_P - torch.sum(P * grad_P, dim=-1, keepdim=True))

        if is_causal:
            n_q, n_k = Q.shape[-2], K.shape[-2]
            q_idx = torch.arange(n_q, device=Q.device)[:, None]
            k_idx = torch.arange(n_k, device=K.device)[None, :]
            causal_mask = q_idx >= k_idx
            grad_S = grad_S.masked_fill(~causal_mask, 0.0)

        grad_Q = torch.einsum('bqk,bkd->bqd', grad_S, K) * scale
        grad_K = torch.einsum('bqk,bqd->bkd', grad_S, Q) * scale

        return grad_Q, grad_K, grad_V, None


class AdaptiveFlashAttention(torch.autograd.Function):
    """FlashAttention with adaptive tile sizes"""

    @staticmethod
    def get_optimal_tile_sizes(seq_len: int, d_model: int) -> Tuple[int, int]:
        """Determine optimal tile sizes based on input dimensions"""

        # Base tile sizes - ensure they're powers of 2 and >= 16
        if seq_len <= 512:
            base_tile = 32
        elif seq_len <= 2048:
            base_tile = 64
        elif seq_len <= 8192:
            base_tile = 128
        else:
            base_tile = 128

        # Adjust based on embedding dimension
        if d_model <= 32:
            q_tile = min(base_tile * 2, 128)
            k_tile = min(base_tile * 2, 128)
        elif d_model >= 128:
            q_tile = max(base_tile // 2, 32)
            k_tile = max(base_tile // 2, 32)
        else:
            q_tile = base_tile
            k_tile = base_tile

        return q_tile, k_tile

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        B, N_k, d = K.shape

        # Get optimal tile sizes
        Q_TILE_SIZE, K_TILE_SIZE = AdaptiveFlashAttention.get_optimal_tile_sizes(N_q, d)

        scale = 1.0 / math.sqrt(d)

        # Initialize output tensors
        O = torch.empty_like(Q)
        L = torch.empty(B, N_q, device=Q.device, dtype=torch.float32)

        # Launch grid: (num_q_tiles, batch_size)
        num_q_tiles = triton.cdiv(N_q, Q_TILE_SIZE)
        grid = (num_q_tiles, B)

        # Launch kernel with adaptive tile sizes
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        # Save for backward pass
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_O):
        # Use the new Triton backward implementation
        return FlashAttentionTriton.backward(ctx, grad_O)


def make_inputs(seq_len: int, d_model: int, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    """Generate inputs for benchmarking"""
    torch.manual_seed(42)  # Reproducibility

    Q = torch.randn(1, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(1, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(1, seq_len, d_model, dtype=dtype, device='cuda', requires_grad=True)
    grad_O = torch.randn(1, seq_len, d_model, dtype=dtype, device='cuda')

    return Q, K, V, grad_O


def benchmark_with_triton_testing(func, *args, warmup=25, rep=100):
    """Use triton.testing.do_bench for accurate GPU measurements"""

    def benchmark_fn():
        result = func(*args)
        # Ensure computation completes
        if hasattr(result, 'detach'):
            result = result.detach()
        return result

    # triton.testing.do_bench handles warmup and synchronization internally
    return triton.testing.do_bench(benchmark_fn, warmup=warmup, rep=rep)


def run_single_config(seq_len: int, d_model: int, dtype: torch.dtype) -> dict:
    """Benchmark a single configuration"""

    config_str = f"seq_len={seq_len}, d_model={d_model}, dtype={dtype}"
    print(f"Benchmarking {config_str}")

    try:
        # Generate inputs
        Q, K, V, grad_O = make_inputs(seq_len, d_model, dtype)

        # Get tile sizes for reporting
        q_tile, k_tile = AdaptiveFlashAttention.get_optimal_tile_sizes(seq_len, d_model)

        results = {
            'seq_len': seq_len,
            'd_model': d_model,
            'dtype': str(dtype).split('.')[-1],
            'q_tile_size': q_tile,
            'k_tile_size': k_tile,
        }

        # Forward pass benchmarks
        def pytorch_forward():
            return StandardAttentionAutograd.apply(Q, K, V, True)

        def flash_forward():
            return AdaptiveFlashAttention.apply(Q, K, V, True)

        pytorch_fwd = benchmark_with_triton_testing(pytorch_forward)
        flash_fwd = benchmark_with_triton_testing(flash_forward)

        # Backward pass benchmarks
        def pytorch_backward():
            Q.grad = K.grad = V.grad = None
            out = StandardAttentionAutograd.apply(Q, K, V, True)
            out.backward(grad_O, retain_graph=True)
            return out

        def flash_backward():
            Q.grad = K.grad = V.grad = None
            out = AdaptiveFlashAttention.apply(Q, K, V, True)
            out.backward(grad_O, retain_graph=True)
            return out

        pytorch_bwd = benchmark_with_triton_testing(pytorch_backward)
        flash_bwd = benchmark_with_triton_testing(flash_backward)

        # End-to-end benchmarks
        def pytorch_e2e():
            Q.grad = K.grad = V.grad = None
            out = StandardAttentionAutograd.apply(Q, K, V, True)
            out.backward(grad_O, retain_graph=True)
            return out

        def flash_e2e():
            Q.grad = K.grad = V.grad = None
            out = AdaptiveFlashAttention.apply(Q, K, V, True)
            out.backward(grad_O, retain_graph=True)
            return out

        pytorch_e2e_time = benchmark_with_triton_testing(pytorch_e2e)
        flash_e2e_time = benchmark_with_triton_testing(flash_e2e)

        # Store results
        results.update({
            'pytorch_fwd_ms': pytorch_fwd,
            'flash_fwd_ms': flash_fwd,
            'pytorch_bwd_ms': pytorch_bwd,
            'flash_bwd_ms': flash_bwd,
            'pytorch_e2e_ms': pytorch_e2e_time,
            'flash_e2e_ms': flash_e2e_time,
            'fwd_speedup': pytorch_fwd / flash_fwd,
            'bwd_speedup': pytorch_bwd / flash_bwd,
            'e2e_speedup': pytorch_e2e_time / flash_e2e_time,
        })

        print(f"  Tiles: {q_tile}x{k_tile}")
        print(f"  Forward:  {pytorch_fwd:.3f}ms -> {flash_fwd:.3f}ms ({results['fwd_speedup']:.2f}x)")
        print(f"  Backward: {pytorch_bwd:.3f}ms -> {flash_bwd:.3f}ms ({results['bwd_speedup']:.2f}x)")
        print(f"  E2E:      {pytorch_e2e_time:.3f}ms -> {flash_e2e_time:.3f}ms ({results['e2e_speedup']:.2f}x)")

        return results

    except RuntimeError as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            print(f"  OOM - skipping")
            torch.cuda.empty_cache()
            return None
        else:
            print(f"  Error: {e}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_full_sweep():
    """Run the complete benchmark sweep"""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print("FlashAttention-2 Full Benchmark Sweep")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Settings: Batch size=1, causal=True, adaptive tile sizes")
    print("=" * 60)

    # Representative sweep for demonstration - full sweep would take hours
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]  # Key powers of 2
    d_models = [16, 32, 64, 128]                  # All embedding dimensions
    dtypes = [torch.bfloat16, torch.float32]     # Both precisions

    total_configs = len(seq_lengths) * len(d_models) * len(dtypes)
    print(f"Total configurations: {total_configs}")
    print()

    all_results = []
    config_count = 0

    for dtype in dtypes:
        for seq_len in seq_lengths:
            for d_model in d_models:
                config_count += 1
                print(f"[{config_count}/{total_configs}]", end=" ")

                result = run_single_config(seq_len, d_model, dtype)
                if result is not None:
                    all_results.append(result)

                # Clear cache periodically
                if config_count % 5 == 0:
                    torch.cuda.empty_cache()

    if not all_results:
        print("No successful benchmark results!")
        return

    # Create results DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 120)
    print("FULL BENCHMARK RESULTS")
    print("=" * 120)

    # Display results table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))

    # Performance summary by category
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Overall summary
    valid_results = df.dropna()
    print(f"\nOverall Performance (all configs):")
    print(f"Forward:  {valid_results['fwd_speedup'].mean():.2f}x avg, {valid_results['fwd_speedup'].median():.2f}x median")
    print(f"Backward: {valid_results['bwd_speedup'].mean():.2f}x avg, {valid_results['bwd_speedup'].median():.2f}x median")
    print(f"End-to-End: {valid_results['e2e_speedup'].mean():.2f}x avg, {valid_results['e2e_speedup'].median():.2f}x median")

    # By sequence length
    print(f"\nPerformance by Sequence Length:")
    seq_analysis = valid_results.groupby('seq_len')[['fwd_speedup', 'bwd_speedup', 'e2e_speedup']].mean()
    print(seq_analysis.round(2))

    # By embedding dimension
    print(f"\nPerformance by Embedding Dimension:")
    dim_analysis = valid_results.groupby('d_model')[['fwd_speedup', 'bwd_speedup', 'e2e_speedup']].mean()
    print(dim_analysis.round(2))

    # By precision
    print(f"\nPerformance by Precision:")
    dtype_analysis = valid_results.groupby('dtype')[['fwd_speedup', 'bwd_speedup', 'e2e_speedup']].mean()
    print(dtype_analysis.round(2))

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f'flashattention_full_sweep_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\nFull results saved to '{filename}'")

    return df


if __name__ == "__main__":
    results_df = run_full_sweep()