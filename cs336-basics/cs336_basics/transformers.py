import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import math
from einops import einsum
from jaxtyping import Float
from collections.abc import Callable, Iterable
from typing import Optional, IO, BinaryIO
import os


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Parameters:
            d_in: int final dimension of the input
            d_out: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        stddev = math.sqrt(2 / (d_out + d_in))
        nn.init.trunc_normal_(self.weight, mean=0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the paramete
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.emb, mean=0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        """
        Parameters
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        normalized = x / rms
        output = self.g * normalized

        return output.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        stddev = math.sqrt(2 / (d_ff + d_model))
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1, mean=0.0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w3, mean=0.0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w2, mean=0.0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_ff = silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))
        w3_x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_ff_w3_x = silu_ff * w3_x
        return einsum(self.w2, silu_ff_w3_x, "d_model d_ff, ... d_ff -> ... d_model")


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        stddev = math.sqrt(2 / (d_ff + d_model))
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1, mean=0.0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w2, mean=0.0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard feed-forward: Linear -> SiLU -> Linear
        hidden = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        activated = silu(hidden)
        return einsum(self.w2, activated, "d_model d_ff, ... d_ff -> ... d_model")


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048, theta_base: float = 10000.0, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta_base = theta_base

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        # Precompute frequency matrix
        self._build_rotation_cache(max_seq_len, device)

    def _build_rotation_cache(self, max_seq_len: int, device: torch.device = None):
        """Precompute cos and sin matrices for efficiency"""
        # Compute frequencies for each dimension pair
        freqs = 1.0 / (self.theta_base ** (torch.arange(0, self.d_model, 2, device=device).float() / self.d_model))

        # Create position indices
        positions = torch.arange(max_seq_len, device=device).float()

        # Compute angle matrix: (max_seq_len, d_model // 2)
        angles = torch.outer(positions, freqs)

        # Precompute cos and sin
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        # Register as non-persistent buffers
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            positions: Position indices of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape as x
        """
        # Check if we need to expand cache
        max_pos = positions.max().item()
        if max_pos >= self.max_seq_len:
            self._build_rotation_cache(max_pos + 1, device=positions.device)
            self.max_seq_len = max_pos + 1

        # Get cos and sin for the given positions
        cos = self.cos_cache[positions]  # (..., seq_len, d_model // 2)
        sin = self.sin_cache[positions]  # (..., seq_len, d_model // 2)

        # Split x into even and odd dimensions
        x_even = x[..., 0::2]  # (..., seq_len, d_model // 2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_model // 2)

        # Apply rotation
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        # Interleave back to original shape
        result = torch.zeros_like(x)
        result[..., 0::2] = x_even_rot
        result[..., 1::2] = x_odd_rot

        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    normalized_x = x - x_max
    exp_x = torch.exp(normalized_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    # Compute attention scores: Q @ K^T
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")

    # Scale by sqrt(d_k)
    d_k = Q.shape[-1]
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)

    # Apply attention weights to values
    output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return output


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    with nvtx.range("computing attention scores"):
        # Compute attention scores: Q @ K^T
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        
        # Scale by sqrt(d_k)
        d_k = Q.shape[-1]
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
    
    with nvtx.range("computing softmax"):
        # Apply softmax to get attention weights
        attn_weights = softmax(scores, dim=-1)
    
    with nvtx.range("final matmul"):
        # Apply attention weights to values
        output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        theta_base: float = 10000.0,
        use_nvtx: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Multi-head self-attention with optional RoPE

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE
            use_rope: Whether to use RoPE for position encoding
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.use_nvtx = use_nvtx

        # Use concatenated weight matrices for efficiency (as expected by tests)
        # Shape: [num_heads * head_dim, d_model] = [d_model, d_model]
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        # Initialize weights
        self._init_weights()

        # RoPE for position encoding (applied to head_dim, not d_model)
        if use_rope:
            self.rope = RoPE(d_model=self.head_dim, max_seq_len=max_seq_len, theta_base=theta_base, device=device)
        else:
            self.rope = None

        # Causal mask for self-attention
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _init_weights(self):
        """Initialize projection weights"""
        std = math.sqrt(2.0 / (self.d_model + self.d_model))
        for param in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.trunc_normal_(param, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

        # Output projection gets smaller std for residual connections
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std / math.sqrt(2), a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            token_positions: Input token positions of shape [batch_size, seq_len]
            mask: Optional attention mask of shape [seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections for Q, K, V using concatenated weights
        Q = einsum(x, self.q_proj, "batch seq d_in, d_model d_in -> batch seq d_model")
        K = einsum(x, self.k_proj, "batch seq d_in, d_model d_in -> batch seq d_model")
        V = einsum(x, self.v_proj, "batch seq d_in, d_model d_in -> batch seq d_model")

        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply RoPE to Q and K if enabled
        if self.use_rope and self.rope is not None:
            if token_positions is None:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                positions = positions.unsqueeze(1).expand(-1, self.num_heads, -1)
            else:
                positions = token_positions

            Q = self.rope(Q, positions)
            K = self.rope(K, positions)
            # V is not rotated!

        # Apply causal mask if no custom mask provided
        if mask is None:
            mask = self.causal_mask[:seq_len, :seq_len]

        # Compute scaled dot-product attention
        if self.use_nvtx:
            output = annotated_scaled_dot_product_attention(Q, K, V, mask)
        else:
            output = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back to [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = einsum(output, self.o_proj, "batch seq d_in, d_model d_in -> batch seq d_model")

        return output


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, use_norm: bool = True, use_rope: bool = True, pre_norm: bool = True, use_swiglu: bool = True, use_nvtx: bool = False):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, use_rope=use_rope, theta_base=theta, use_nvtx=use_nvtx)
        
        # Conditionally create SwiGLU or standard SiLU feed-forward network
        if use_swiglu:
            self.ffn = SwiGLUFeedForward(d_model, d_ff)
        else:
            self.ffn = SiLUFeedForward(d_model, d_ff)
            
        self.use_norm = use_norm
        self.pre_norm = pre_norm
        
        # Conditionally create RMSNorm layers
        if use_norm:
            self.rms_attn = RMSNorm(d_model)
            self.rms_ffn = RMSNorm(d_model)
        else:
            self.rms_attn = nn.Identity()
            self.rms_ffn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm: LayerNorm -> Sublayer -> Residual
            attn = self.attention(self.rms_attn(x)) + x
            ffn = self.ffn(self.rms_ffn(attn)) + attn
        else:
            # Post-norm: Sublayer -> Residual -> LayerNorm
            attn = self.rms_attn(self.attention(x) + x)
            ffn = self.rms_ffn(self.ffn(attn) + attn)
        return ffn


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        use_norm: bool = True,
        use_rope: bool = True,
        pre_norm: bool = True,
        use_swiglu: bool = True,
        use_nvtx: bool = False,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [Transformer(d_model, num_heads, d_ff, max_seq_len=context_length, theta=theta, use_norm=use_norm, use_rope=use_rope, pre_norm=pre_norm, use_swiglu=use_swiglu, use_nvtx=use_nvtx) for i in range(num_layers)]
        )
        self.token_emb = Embedding(vocab_size, d_model)
        self.use_norm = use_norm
        
        # Conditionally create final RMSNorm layer
        if use_norm:
            self.ln_final = RMSNorm(d_model)
        else:
            self.ln_final = nn.Identity()
            
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        for transformer_block in self.transformers:
            x = transformer_block(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with numerical stability.

    Args:
        logits: Unnormalized logits of shape [batch_size, vocab_size]
        targets: Target class indices of shape [batch_size]

    Returns:
        Average cross-entropy loss (scalar tensor)
    """
    batch_size = logits.shape[0]

    # Numerical stability: subtract max logit from each example
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits

    # Compute log softmax: log(softmax(x)) = x - log(sum(exp(x)))
    exp_logits = torch.exp(stable_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=-1)  # [batch_size]
    log_sum_exp = torch.log(sum_exp_logits)  # [batch_size]

    # Get the logit for the correct class for each example
    target_logits = stable_logits[torch.arange(batch_size, device=targets.device), targets]  # [batch_size]

    # Cross-entropy: -log(softmax(x_correct)) = -(x_correct - log(sum(exp(x))))
    losses = log_sum_exp - target_logits  # [batch_size]

    # Return average loss
    return torch.mean(losses)


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
    ):
        defaults = {"lr": lr, "beta_1": betas[0], "beta_2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> torch.Tensor:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # AdamW weight decay (applied to parameters, not gradients)
                p.data.mul_(1 - lr * weight_decay)

                # Exponential moving average of gradient values
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                # Bias correction
                bias_correction1 = 1 - beta_1 ** state["step"]
                bias_correction2 = 1 - beta_2 ** state["step"]

                # Update parameters
                step_size = lr / bias_correction1
                denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.data.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


def cosine_learning_rate_schedule(step: int, a_max: float, a_min: float, warmup_steps: int, cosine_steps: int) -> float:
    if step < warmup_steps:
        return step * a_max / warmup_steps

    if step < cosine_steps:
        return a_min + 0.5 * (1 + math.cos((step - warmup_steps) * math.pi / (cosine_steps - warmup_steps))) * (
            a_max - a_min
        )

    return a_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by global norm across all parameters.

    Args:
        parameters: Iterable of parameters whose gradients to clip
        max_l2_norm: Maximum allowed L2 norm of the gradient vector
    """
    # Collect all gradient tensors
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    if not grads:
        return

    # Compute global norm across all parameters
    total_norm = torch.norm(torch.cat(grads))

    # If global norm exceeds max_norm, scale all gradients by the same factor
    if total_norm > max_l2_norm:
        scaling_factor = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scaling_factor)


def get_batch(dataset, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random batches from a dataset for language modeling.

    This function supports both in-memory numpy arrays and memory-mapped arrays
    (created with np.memmap or loaded with mmap_mode='r').

    Args:
        dataset: 1D numpy array or memmap of integer token IDs in the dataset
        batch_size: Desired batch size to sample
        context_length: Desired context length of each sampled example
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length):
        - input sequences (x)
        - corresponding language modeling labels (y = x shifted by 1)
    """
    # Calculate the number of valid starting positions
    # We need context_length + 1 tokens total to create both x and y sequences
    max_start_idx = len(dataset) - context_length

    if max_start_idx <= 0:
        raise ValueError(f"Dataset length ({len(dataset)}) must be greater than context_length ({context_length})")

    # Randomly sample starting indices for each batch item
    start_indices = torch.randint(0, max_start_idx, (batch_size,))

    # Pre-allocate tensors for efficiency
    x = torch.zeros(batch_size, context_length, dtype=torch.long)
    y = torch.zeros(batch_size, context_length, dtype=torch.long)

    for i, start_idx in enumerate(start_indices):
        # Extract the sequence of length context_length + 1
        # This works efficiently with both regular arrays and memory-mapped arrays
        sequence = dataset[start_idx : start_idx + context_length + 1]

        # Validate the sequence doesn't contain unexpected values
        # This is especially important with memory-mapped data
        if len(sequence) != context_length + 1:
            raise ValueError(f"Sequence at index {start_idx} has length {len(sequence)}, expected {context_length + 1}")

        # Input sequence: first context_length tokens
        x[i] = torch.from_numpy(sequence[:-1].copy()).long()
        # Target sequence: last context_length tokens (shifted by 1)
        y[i] = torch.from_numpy(sequence[1:].copy()).long()

    # Move tensors to the specified device
    return x.to(device), y.to(device)


def load_dataset(file_path: str, dtype=None, mmap_mode: str = "r"):
    """
    Load a dataset from disk with optional memory mapping for large files.

    Args:
        file_path: Path to the dataset file (.npy format)
        dtype: Data type of the array (e.g., np.uint16, np.int32)
        mmap_mode: Memory map mode ('r' for read-only, None for regular loading)

    Returns:
        Numpy array or memmap object containing the dataset
    """
    import numpy as np

    if mmap_mode is not None:
        # Load as memory-mapped array for large datasets
        dataset = np.load(file_path, mmap_mode=mmap_mode)

        # Verify data integrity for memory-mapped arrays
        if len(dataset) == 0:
            raise ValueError(f"Dataset {file_path} is empty")

        print(f"Loaded memory-mapped dataset: {file_path}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Dtype: {dataset.dtype}")
        print(f"  First few values: {dataset[:10].tolist()}")
        print(f"  Value range: [{dataset.min()}, {dataset.max()}]")

        return dataset
    else:
        # Load entire dataset into memory
        dataset = np.load(file_path)
        print(f"Loaded in-memory dataset: {file_path} (shape: {dataset.shape}, dtype: {dataset.dtype})")
        return dataset


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = dict()
    state["model"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    state["iteration"] = iteration
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
