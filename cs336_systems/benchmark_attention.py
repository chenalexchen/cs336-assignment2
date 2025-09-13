"""
Benchmark attention implementation at different scales.

This script benchmarks attention implementations with:
- Fixed batch size of 8
- No multihead attention (single head)
- Cartesian product of d_model [16, 32, 64, 128] and seq_len [256, 1024, 4096, 8192, 16384]
- Times 100 forward and backward passes with warm-up
- Measures memory usage before backward pass
- Uses torch.cuda.synchronize() for accurate timing
"""

import itertools
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from einops import einsum

# Import the attention function from cs336_basics (assignment1)
from cs336_basics.transformers import scaled_dot_product_attention

# Create compiled version of attention with fallback for older GPUs
try:
    # Check CUDA capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:  # CUDA capability < 7.0
            print(f"Warning: CUDA capability {capability[0]}.{capability[1]} < 7.0, using 'aot_eager' backend")
            compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention, backend="aot_eager")
        else:
            compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)
    else:
        compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)
except Exception as e:
    print(f"Warning: torch.compile failed, falling back to eager mode: {e}")
    # Fall back to using the original function
    compiled_scaled_dot_product_attention = scaled_dot_product_attention


def create_attention_inputs(
    batch_size: int,
    seq_len: int, 
    d_model: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V tensors for attention."""
    torch.manual_seed(42)  # For reproducible results
    
    q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.float32)
    
    return q, k, v


def create_causal_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Create causal mask for attention."""
    # Create causal mask: queries can only attend to keys at the same position or earlier
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32))
    return mask.unsqueeze(0)  # Add batch dimension


def benchmark_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    n_trials: int = 100,
    warmup_trials: int = 5,
    use_compiled: bool = False
) -> float:
    """Benchmark forward pass of attention."""
    device = q.device
    attention_fn = compiled_scaled_dot_product_attention if use_compiled else scaled_dot_product_attention
    
    # Warmup
    for _ in range(warmup_trials):
        with torch.no_grad():
            _ = attention_fn(q, k, v, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for _ in range(n_trials):
        with torch.no_grad():
            output = attention_fn(q, k, v, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) / n_trials


def benchmark_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    n_trials: int = 100,
    warmup_trials: int = 5,
    use_compiled: bool = False
) -> Tuple[float, float]:
    """Benchmark backward pass of attention and measure memory before backward."""
    device = q.device
    attention_fn = compiled_scaled_dot_product_attention if use_compiled else scaled_dot_product_attention
    
    # Warmup
    for _ in range(warmup_trials):
        q_warm = q.clone().detach().requires_grad_(True)
        k_warm = k.clone().detach().requires_grad_(True) 
        v_warm = v.clone().detach().requires_grad_(True)
        
        output = attention_fn(q_warm, k_warm, v_warm, mask)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Get memory usage before backward pass (during one forward pass)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    q_test = q.clone().detach().requires_grad_(True)
    k_test = k.clone().detach().requires_grad_(True)
    v_test = v.clone().detach().requires_grad_(True)
    
    output = attention_fn(q_test, k_test, v_test, mask)
    grad_output = torch.randn_like(output)
    
    # Measure memory before backward pass
    memory_before_backward = 0
    if device.type == "cuda":
        memory_before_backward = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    # Clear gradients and prepare for timing
    if hasattr(q_test, 'grad') and q_test.grad is not None:
        q_test.grad.zero_()
    if hasattr(k_test, 'grad') and k_test.grad is not None:
        k_test.grad.zero_()
    if hasattr(v_test, 'grad') and v_test.grad is not None:
        v_test.grad.zero_()
    
    # Timing backward passes
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for _ in range(n_trials):
        # Clone tensors for each trial to avoid accumulating gradients
        q_trial = q.clone().detach().requires_grad_(True)
        k_trial = k.clone().detach().requires_grad_(True)
        v_trial = v.clone().detach().requires_grad_(True)
        
        output = attention_fn(q_trial, k_trial, v_trial, mask)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    backward_time = (end_time - start_time) / n_trials
    
    return backward_time, memory_before_backward


def run_comparison_benchmark(device: str = "cuda") -> List[Dict[str, Any]]:
    """Run comparison benchmark between compiled and uncompiled attention."""
    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    
    results = []
    
    print("=== Compiled vs Uncompiled Attention Comparison ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print()
    
    print("| d_model | seq_len | Uncompiled Forward (ms) | Compiled Forward (ms) | Forward Speedup | Uncompiled Backward (ms) | Compiled Backward (ms) | Backward Speedup | Status |")
    print("|---------|---------|-------------------------|----------------------|-----------------|--------------------------|------------------------|------------------|--------|")
    
    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        config = {
            'batch_size': batch_size,
            'd_model': d_model,
            'seq_len': seq_len,
            'uncompiled_forward_time_ms': None,
            'compiled_forward_time_ms': None,
            'forward_speedup': None,
            'uncompiled_backward_time_ms': None,
            'compiled_backward_time_ms': None,
            'backward_speedup': None,
            'status': 'success',
            'error': None
        }
        
        try:
            # Create inputs
            q, k, v = create_attention_inputs(batch_size, seq_len, d_model, device)
            mask = create_causal_mask(seq_len, device)
            
            # Benchmark uncompiled version
            uncompiled_forward_time = benchmark_attention_forward(q, k, v, mask, use_compiled=False)
            uncompiled_backward_time, _ = benchmark_attention_backward(q, k, v, mask, use_compiled=False)
            
            # Benchmark compiled version
            compiled_forward_time = benchmark_attention_forward(q, k, v, mask, use_compiled=True)
            compiled_backward_time, _ = benchmark_attention_backward(q, k, v, mask, use_compiled=True)
            
            # Calculate speedups
            forward_speedup = uncompiled_forward_time / compiled_forward_time
            backward_speedup = uncompiled_backward_time / compiled_backward_time
            
            config.update({
                'uncompiled_forward_time_ms': uncompiled_forward_time * 1000,
                'compiled_forward_time_ms': compiled_forward_time * 1000,
                'forward_speedup': forward_speedup,
                'uncompiled_backward_time_ms': uncompiled_backward_time * 1000,
                'compiled_backward_time_ms': compiled_backward_time * 1000,
                'backward_speedup': backward_speedup
            })
            
            status = "✓"
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                config['status'] = 'OOM'
                config['error'] = str(e)
                status = "OOM"
            else:
                config['status'] = 'error'
                config['error'] = str(e)
                status = "ERROR"
                
            # Clear cache after OOM
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            config['status'] = 'error'
            config['error'] = str(e)
            status = "ERROR"
        
        # Format output
        if config['status'] == 'success':
            uncompiled_forward_str = f"{config['uncompiled_forward_time_ms']:.2f}"
            compiled_forward_str = f"{config['compiled_forward_time_ms']:.2f}"
            forward_speedup_str = f"{config['forward_speedup']:.2f}x"
            uncompiled_backward_str = f"{config['uncompiled_backward_time_ms']:.2f}"
            compiled_backward_str = f"{config['compiled_backward_time_ms']:.2f}"
            backward_speedup_str = f"{config['backward_speedup']:.2f}x"
        else:
            uncompiled_forward_str = compiled_forward_str = forward_speedup_str = "-"
            uncompiled_backward_str = compiled_backward_str = backward_speedup_str = "-"
        
        print(f"| {d_model:7} | {seq_len:7} | {uncompiled_forward_str:23} | {compiled_forward_str:20} | {forward_speedup_str:15} | {uncompiled_backward_str:24} | {compiled_backward_str:22} | {backward_speedup_str:16} | {status:6} |")
        
        results.append(config)
    
    return results


def run_benchmark_suite(device: str = "cuda") -> List[Dict[str, Any]]:
    """Run complete benchmark suite."""
    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    
    results = []
    
    print("=== Attention Benchmarking Suite ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"d_model values: {d_model_values}")
    print(f"Sequence length values: {seq_len_values}")
    print()
    
    print("| d_model | seq_len | Forward Time (ms) | Backward Time (ms) | Memory Before Backward (MB) | Status |")
    print("|---------|---------|-------------------|--------------------|-----------------------------|--------|")
    
    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        config = {
            'batch_size': batch_size,
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_ms': None,
            'backward_time_ms': None,
            'memory_mb': None,
            'status': 'success',
            'error': None
        }
        
        try:
            # Create inputs
            q, k, v = create_attention_inputs(batch_size, seq_len, d_model, device)
            mask = create_causal_mask(seq_len, device)
            
            # Benchmark forward pass
            forward_time = benchmark_attention_forward(q, k, v, mask, use_compiled=False)
            config['forward_time_ms'] = forward_time * 1000
            
            # Benchmark backward pass
            backward_time, memory_mb = benchmark_attention_backward(q, k, v, mask, use_compiled=False)
            config['backward_time_ms'] = backward_time * 1000
            config['memory_mb'] = memory_mb
            
            status = "✓"
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                config['status'] = 'OOM'
                config['error'] = str(e)
                status = "OOM"
            else:
                config['status'] = 'error'
                config['error'] = str(e)
                status = "ERROR"
                
            # Clear cache after OOM
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            config['status'] = 'error'
            config['error'] = str(e)
            status = "ERROR"
        
        # Format output
        forward_str = f"{config['forward_time_ms']:.2f}" if config['forward_time_ms'] is not None else "-"
        backward_str = f"{config['backward_time_ms']:.2f}" if config['backward_time_ms'] is not None else "-" 
        memory_str = f"{config['memory_mb']:.1f}" if config['memory_mb'] is not None else "-"
        
        print(f"| {d_model:7} | {seq_len:7} | {forward_str:17} | {backward_str:18} | {memory_str:27} | {status:6} |")
        
        results.append(config)
    
    return results


def analyze_memory_scaling(results: List[Dict[str, Any]]):
    """Analyze how memory usage scales with sequence length."""
    print("\n=== Memory Scaling Analysis ===")
    
    # Group by d_model and analyze sequence length scaling
    d_model_groups = {}
    for result in results:
        if result['status'] == 'success' and result['memory_mb'] is not None:
            d_model = result['d_model']
            if d_model not in d_model_groups:
                d_model_groups[d_model] = []
            d_model_groups[d_model].append((result['seq_len'], result['memory_mb']))
    
    for d_model in sorted(d_model_groups.keys()):
        data = sorted(d_model_groups[d_model])
        print(f"\nd_model = {d_model}:")
        print("  seq_len -> memory (MB)")
        for seq_len, memory in data:
            print(f"  {seq_len:6} -> {memory:6.1f}")
        
        # Calculate scaling ratio
        if len(data) >= 2:
            ratios = []
            for i in range(1, len(data)):
                seq_ratio = data[i][0] / data[i-1][0]
                mem_ratio = data[i][1] / data[i-1][1]
                ratios.append(mem_ratio / seq_ratio)
            avg_scaling = sum(ratios) / len(ratios)
            print(f"  Average memory scaling vs sequence length: {avg_scaling:.2f}x")


def analyze_compilation_speedups(results: List[Dict[str, Any]]):
    """Analyze compilation speedups across different configurations."""
    print("\n=== torch.compile Speedup Analysis ===")
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful compilation results to analyze.")
        return
    
    print("\nSpeedup by sequence length:")
    seq_len_groups = {}
    for result in successful_results:
        seq_len = result['seq_len']
        if seq_len not in seq_len_groups:
            seq_len_groups[seq_len] = []
        seq_len_groups[seq_len].append((result['forward_speedup'], result['backward_speedup']))
    
    for seq_len in sorted(seq_len_groups.keys()):
        forward_speedups = [x[0] for x in seq_len_groups[seq_len]]
        backward_speedups = [x[1] for x in seq_len_groups[seq_len]]
        avg_forward = sum(forward_speedups) / len(forward_speedups)
        avg_backward = sum(backward_speedups) / len(backward_speedups)
        print(f"  seq_len {seq_len:5}: Forward {avg_forward:.2f}x, Backward {avg_backward:.2f}x")
    
    print("\nSpeedup by d_model:")
    d_model_groups = {}
    for result in successful_results:
        d_model = result['d_model']
        if d_model not in d_model_groups:
            d_model_groups[d_model] = []
        d_model_groups[d_model].append((result['forward_speedup'], result['backward_speedup']))
    
    for d_model in sorted(d_model_groups.keys()):
        forward_speedups = [x[0] for x in d_model_groups[d_model]]
        backward_speedups = [x[1] for x in d_model_groups[d_model]]
        avg_forward = sum(forward_speedups) / len(forward_speedups)
        avg_backward = sum(backward_speedups) / len(backward_speedups)
        print(f"  d_model {d_model:3}: Forward {avg_forward:.2f}x, Backward {avg_backward:.2f}x")


def memory_accounting_example(batch_size: int = 8, seq_len: int = 4096, d_model: int = 64):
    """
    Perform memory accounting for attention computation.
    
    Memory usage in attention:
    1. Input tensors Q, K, V: 3 * batch_size * seq_len * d_model * 4 bytes (fp32)
    2. Attention scores (Q @ K^T): batch_size * seq_len * seq_len * 4 bytes
    3. Attention weights (after softmax): batch_size * seq_len * seq_len * 4 bytes  
    4. Output (P @ V): batch_size * seq_len * d_model * 4 bytes
    5. Gradients for backward: Similar to forward tensors
    
    The quadratic term (seq_len^2) dominates for large sequences.
    """
    print(f"\n=== Memory Accounting Example ===")
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    
    # Forward pass memory
    qkv_memory = 3 * batch_size * seq_len * d_model * 4  # Q, K, V tensors
    scores_memory = batch_size * seq_len * seq_len * 4   # Attention scores
    weights_memory = batch_size * seq_len * seq_len * 4  # Attention weights (softmax output)
    output_memory = batch_size * seq_len * d_model * 4   # Output tensor
    
    forward_memory = qkv_memory + scores_memory + weights_memory + output_memory
    
    # Backward pass adds gradients
    backward_memory = forward_memory * 2  # Rough estimate (activations + gradients)
    
    print(f"\nForward pass memory breakdown:")
    print(f"  Q, K, V tensors: {qkv_memory / (1024**2):.1f} MB")
    print(f"  Attention scores: {scores_memory / (1024**2):.1f} MB")
    print(f"  Attention weights: {weights_memory / (1024**2):.1f} MB") 
    print(f"  Output tensor: {output_memory / (1024**2):.1f} MB")
    print(f"  Total forward: {forward_memory / (1024**2):.1f} MB")
    print(f"\nEstimated total (forward + backward): {backward_memory / (1024**2):.1f} MB")
    print(f"\nQuadratic component (attention matrix): {(scores_memory + weights_memory) / (1024**2):.1f} MB")
    print(f"Linear component (Q,K,V,output): {(qkv_memory + output_memory) / (1024**2):.1f} MB")


def main():
    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"PyTorch version: {torch.__version__}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print()
    
    # Run comparison benchmark (compiled vs uncompiled)
    print("Running compiled vs uncompiled comparison...")
    comparison_results = run_comparison_benchmark(device)
    
    print("\n" + "="*80 + "\n")
    
    # Run original benchmark suite for memory analysis
    results = run_benchmark_suite(device)
    
    # Analyze memory scaling
    analyze_memory_scaling(results)
    
    # Memory accounting example
    # Find a configuration that causes OOM or use largest successful config
    oom_configs = [r for r in results if r['status'] == 'OOM']
    if oom_configs:
        config = oom_configs[0]  # First OOM config
        print(f"\n=== Memory Analysis for OOM Configuration ===")
        memory_accounting_example(config['batch_size'], config['seq_len'], config['d_model'])
    else:
        print(f"\n=== Memory Analysis for Largest Configuration ===") 
        memory_accounting_example(8, 16384, 128)
    
    # Compilation speedup analysis
    analyze_compilation_speedups(comparison_results)
    
    # Summary and recommendations
    print(f"\n=== Summary and Recommendations ===")
    
    successful_configs = [r for r in results if r['status'] == 'success']
    oom_configs = [r for r in results if r['status'] == 'OOM']
    
    if successful_configs:
        max_seq_len = max(r['seq_len'] for r in successful_configs)
        max_d_model = max(r['d_model'] for r in successful_configs if r['seq_len'] == max_seq_len)
        print(f"Largest successful configuration: d_model={max_d_model}, seq_len={max_seq_len}")
    
    if oom_configs:
        min_oom_seq_len = min(r['seq_len'] for r in oom_configs)
        min_oom_d_model = min(r['d_model'] for r in oom_configs if r['seq_len'] == min_oom_seq_len)
        print(f"Smallest OOM configuration: d_model={min_oom_d_model}, seq_len={min_oom_seq_len}")
    
    # Compilation recommendations
    successful_compilations = [r for r in comparison_results if r['status'] == 'success']
    if successful_compilations:
        avg_forward_speedup = sum(r['forward_speedup'] for r in successful_compilations) / len(successful_compilations)
        avg_backward_speedup = sum(r['backward_speedup'] for r in successful_compilations) / len(successful_compilations)
        print(f"Average torch.compile speedup: Forward {avg_forward_speedup:.2f}x, Backward {avg_backward_speedup:.2f}x")
    
    print("\nOptimization strategies:")
    print("1. Use torch.compile for automatic kernel optimization")
    print("2. Use FlashAttention - computes attention without materializing the full attention matrix")
    print("3. Use gradient checkpointing - trade compute for memory by recomputing activations")
    print("4. Use sequence parallelism - split sequence dimension across multiple GPUs")
    print("5. Use sparse attention patterns - only compute attention for relevant positions")


if __name__ == "__main__":
    main()