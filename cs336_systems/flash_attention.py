import torch
import math
import triton
import triton.language as tl

from jaxtyping import Float


@torch.compile
def _flash_backward_compiled(Q, K, V, O, grad_O, L, D, is_causal, scale):
    """Compiled helper function for FlashAttention backward pass"""
    B, N_q, d = Q.shape
    B, N_k, d = K.shape

    B_q = 32  # Query block size
    B_k = 32  # Key/Value block size

    # Initialize output gradients
    grad_Q = torch.zeros_like(Q)
    grad_K = torch.zeros_like(K)
    grad_V = torch.zeros_like(V)

    # Process in blocks
    for i in range(0, N_q, B_q):
        q_start = i
        q_end = min(i + B_q, N_q)
        q_block = Q[:, q_start:q_end, :]
        grad_O_block = grad_O[:, q_start:q_end, :]
        L_block = L[:, q_start:q_end]
        D_block = D[:, q_start:q_end]

        grad_Q_block = torch.zeros_like(q_block)

        for j in range(0, N_k, B_k):
            k_start = j
            k_end = min(j + B_k, N_k)
            k_block = K[:, k_start:k_end, :]
            v_block = V[:, k_start:k_end, :]

            # Recompute attention scores
            s_block = torch.einsum('bqd,bkd->bqk', q_block, k_block) * scale

            # Apply causal mask if needed
            if is_causal:
                q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                causal_mask = q_indices >= k_indices
                s_block = torch.where(causal_mask, s_block, -torch.inf)

            # Compute probabilities
            p_block = torch.exp(s_block - L_block.unsqueeze(-1))

            # Compute gradients
            grad_S_block = p_block * (
                torch.einsum('bqd,bkd->bqk', grad_O_block, v_block) -
                D_block.unsqueeze(-1)
            )

            if is_causal:
                grad_S_block = torch.where(causal_mask, grad_S_block, 0.0)

            # Accumulate gradients
            grad_Q_block += torch.einsum('bqk,bkd->bqd', grad_S_block, k_block) * scale
            grad_K[:, k_start:k_end, :] += torch.einsum('bqk,bqd->bkd', grad_S_block, q_block) * scale
            grad_V[:, k_start:k_end, :] += torch.einsum('bqk,bqd->bkd', p_block, grad_O_block)

        grad_Q[:, q_start:q_end, :] = grad_Q_block

    return grad_Q, grad_K, grad_V


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[torch.Tensor, 'B N_q d'],
                K: Float[torch.Tensor, 'B N_k d'],
                V: Float[torch.Tensor, 'B N_k d'],
                is_causal: bool = False) -> Float[torch.Tensor, 'B N_q d']:

        B, N_q, d = Q.shape
        B, N_k, d = K.shape

        # Choose block sizes (must be >= 16x16)
        B_q = 32  # Query block size
        B_k = 32  # Key/Value block size

        scale = 1.0 / math.sqrt(d)

        # Initialize output and statistics
        O = torch.zeros_like(Q)
        L = torch.zeros(B, N_q, device=Q.device, dtype=Q.dtype)  # logsumexp

        # Process in blocks
        for i in range(0, N_q, B_q):
            # Query block bounds
            q_start = i
            q_end = min(i + B_q, N_q)
            q_block = Q[:, q_start:q_end, :]  # [B, B_q, d]

            # Initialize block outputs and statistics
            o_i = torch.zeros(B, q_end - q_start, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, q_end - q_start, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((B, q_end - q_start), -torch.inf, device=Q.device, dtype=Q.dtype)

            for j in range(0, N_k, B_k):
                # Key/Value block bounds
                k_start = j
                k_end = min(j + B_k, N_k)
                k_j = K[:, k_start:k_end, :]  # [B, B_k, d]
                v_j = V[:, k_start:k_end, :]  # [B, B_k, d]

                # Compute attention scores: S_ij = Q_i @ K_j^T
                s_ij = torch.einsum('bqd,bkd->bqk', q_block, k_j) * scale  # [B, B_q, B_k]

                # Apply causal mask if needed (ignored per instructions)
                if is_causal:
                    # Create causal mask for this block
                    q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                    causal_mask = q_indices >= k_indices
                    s_ij = torch.where(causal_mask, s_ij, -torch.inf)

                # Compute block max and update global max
                m_ij = torch.max(s_ij, dim=-1)[0]  # [B, B_q]
                m_i_new = torch.maximum(m_i, m_ij)  # [B, B_q]

                # Compute P_ij = exp(S_ij - m_i_new)
                p_ij = torch.exp(s_ij - m_i_new.unsqueeze(-1))  # [B, B_q, B_k]

                # Update l_i using online algorithm
                l_i = torch.exp(m_i - m_i_new) * l_i + torch.sum(p_ij, dim=-1)  # [B, B_q]

                # Update o_i using online algorithm
                o_i = torch.exp(m_i - m_i_new).unsqueeze(-1) * o_i + torch.einsum('bqk,bkd->bqd', p_ij, v_j)  # [B, B_q, d]

                # Update m_i
                m_i = m_i_new

            # Final normalization and store results
            O[:, q_start:q_end, :] = o_i / l_i.unsqueeze(-1)
            L[:, q_start:q_end] = torch.log(l_i) + m_i

        # Save for backward pass
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_O: Float[torch.Tensor, 'B N_q d']):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # Compute D vector: D_i = sum_j(dO_ij * O_ij) for each query
        D = torch.sum(grad_O * O, dim=-1)  # [B, N_q]

        # Use torch.compile optimized function
        grad_Q, grad_K, grad_V = _flash_backward_compiled(
            Q, K, V, O, grad_O, L, D, is_causal, scale
        )

        return grad_Q, grad_K, grad_V, None  # None for is_causal gradient


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr,
    grad_O_ptr, grad_Q_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_gob, stride_goq, stride_god,
    stride_gqb, stride_gqq, stride_gqd,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Kernel for computing dQ gradients - no race conditions"""
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Query block bounds
    q_start = query_tile_index * Q_TILE_SIZE

    # Load Q tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)  # [Q_TILE_SIZE, D]

    # Load grad_O tile
    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr + batch_index * stride_gob,
        shape=(N_QUERIES, D),
        strides=(stride_goq, stride_god),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    grad_o = tl.load(grad_O_block_ptr)  # [Q_TILE_SIZE, D]

    # Load L and D tiles
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    l_tile = tl.load(L_block_ptr)  # [Q_TILE_SIZE]

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    d_tile = tl.load(D_block_ptr)  # [Q_TILE_SIZE]

    # Initialize grad_Q accumulator
    grad_q = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # Initialize K and V block pointers
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Loop over K tiles to compute dQ
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_idx in range(num_k_tiles):
        k_start = k_tile_idx * K_TILE_SIZE

        # Load K and V tiles
        k = tl.load(K_block_ptr)  # [K_TILE_SIZE, D]
        v = tl.load(V_block_ptr)  # [K_TILE_SIZE, D]

        # Recompute attention scores for dQ computation
        kt = tl.trans(k)  # [D, K_TILE_SIZE]
        s = tl.dot(q, kt, allow_tf32=False) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask if needed
        if is_causal:
            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]  # [Q_TILE_SIZE, 1]
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)[None, :]  # [1, K_TILE_SIZE]
            causal_mask = q_indices >= k_indices  # [Q_TILE_SIZE, K_TILE_SIZE]
            s = tl.where(causal_mask, s, s - 1e6)  # Add -1e6 to masked elements

        # Compute probabilities P for dQ
        p = tl.exp(s - l_tile[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Compute grad_S = P * (grad_O @ V^T - D)
        # Ensure dtype consistency for matrix multiplication
        grad_o_v = tl.dot(grad_o.to(tl.float32), tl.trans(v.to(tl.float32)), allow_tf32=False)  # [Q_TILE_SIZE, K_TILE_SIZE]
        grad_s = p * (grad_o_v - d_tile[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask to grad_S if needed
        if is_causal:
            grad_s = tl.where(causal_mask, grad_s, 0.0)

        # Accumulate grad_Q
        grad_q += tl.dot(grad_s, k.to(tl.float32), allow_tf32=False) * scale  # [Q_TILE_SIZE, D]

        # Advance K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Store grad_Q
    grad_Q_block_ptr = tl.make_block_ptr(
        grad_Q_ptr + batch_index * stride_gqb,
        shape=(N_QUERIES, D),
        strides=(stride_gqq, stride_gqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    grad_q_cast = grad_q.to(grad_Q_block_ptr.type.element_ty)
    tl.store(grad_Q_block_ptr, grad_q_cast)


@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr,
    grad_O_ptr, grad_K_ptr, grad_V_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_gob, stride_goq, stride_god,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Kernel for computing dK and dV gradients - no race conditions"""
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Key block bounds
    k_start = key_tile_index * K_TILE_SIZE

    # Load K and V tiles
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    k = tl.load(K_block_ptr)  # [K_TILE_SIZE, D]

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v = tl.load(V_block_ptr)  # [K_TILE_SIZE, D]

    # Initialize grad_K and grad_V accumulators
    grad_k = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    grad_v = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    # Initialize Q block pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Loop over Q tiles to compute dK and dV
    num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for q_tile_idx in range(num_q_tiles):
        q_start = q_tile_idx * Q_TILE_SIZE

        # Load Q tile
        q = tl.load(Q_block_ptr)  # [Q_TILE_SIZE, D]

        # Load grad_O tile
        grad_O_block_ptr = tl.make_block_ptr(
            grad_O_ptr + batch_index * stride_gob,
            shape=(N_QUERIES, D),
            strides=(stride_goq, stride_god),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        grad_o = tl.load(grad_O_block_ptr)  # [Q_TILE_SIZE, D]

        # Load L and D tiles
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(q_start,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        l_tile = tl.load(L_block_ptr)  # [Q_TILE_SIZE]

        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(q_start,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        d_tile = tl.load(D_block_ptr)  # [Q_TILE_SIZE]

        # Recompute attention scores for dK/dV computation
        kt = tl.trans(k)  # [D, K_TILE_SIZE]
        s = tl.dot(q, kt, allow_tf32=False) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask if needed
        if is_causal:
            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]  # [Q_TILE_SIZE, 1]
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)[None, :]  # [1, K_TILE_SIZE]
            causal_mask = q_indices >= k_indices  # [Q_TILE_SIZE, K_TILE_SIZE]
            s = tl.where(causal_mask, s, s - 1e6)  # Add -1e6 to masked elements

        # Compute probabilities P for dK/dV
        p = tl.exp(s - l_tile[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Compute grad_S = P * (grad_O @ V^T - D)
        # Ensure dtype consistency for matrix multiplication
        grad_o_v = tl.dot(grad_o.to(tl.float32), tl.trans(v.to(tl.float32)), allow_tf32=False)  # [Q_TILE_SIZE, K_TILE_SIZE]
        grad_s = p * (grad_o_v - d_tile[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask to grad_S if needed
        if is_causal:
            grad_s = tl.where(causal_mask, grad_s, 0.0)

        # Accumulate grad_K and grad_V
        grad_k += tl.dot(tl.trans(grad_s), q.to(tl.float32), allow_tf32=False) * scale  # [K_TILE_SIZE, D]
        grad_v += tl.dot(tl.trans(p), grad_o.to(tl.float32), allow_tf32=False)  # [K_TILE_SIZE, D]

        # Advance Q block pointer
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))

    # Store grad_K
    grad_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_gkb,
        shape=(N_KEYS, D),
        strides=(stride_gkk, stride_gkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    grad_k_cast = grad_k.to(grad_K_block_ptr.type.element_ty)
    tl.store(grad_K_block_ptr, grad_k_cast)

    # Store grad_V
    grad_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_gvb,
        shape=(N_KEYS, D),
        strides=(stride_gvk, stride_gvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    grad_v_cast = grad_v.to(grad_V_block_ptr.type.element_ty)
    tl.store(grad_V_block_ptr, grad_v_cast)


@triton.jit
def flash_bwd_kernel_algorithm2(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr,
    grad_O_ptr, grad_Q_ptr, grad_K_ptr, grad_V_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_gob, stride_goq, stride_god,
    stride_gqb, stride_gqq, stride_gqd,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    True Algorithm 2 implementation: Single kernel with atomics only for dQ.
    Outer loop over K/V tiles (j), inner loop over Q tiles (i).
    Each thread block handles one K/V tile and accumulates dK, dV locally.
    """
    # Program indices - each thread block processes one K/V tile
    key_tile_index = tl.program_id(0)  # j in algorithm
    batch_index = tl.program_id(1)

    # Key/Value tile bounds
    k_start = key_tile_index * K_TILE_SIZE

    # Load K(j) and V(j) tiles
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    k_j = tl.load(K_block_ptr)  # [K_TILE_SIZE, D]

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v_j = tl.load(V_block_ptr)  # [K_TILE_SIZE, D]

    # Initialize local accumulators for dK(j) and dV(j)
    dk_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dv_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    # Inner loop over Q tiles (i = 1, ..., Tq)
    num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for q_tile_idx in range(num_q_tiles):
        q_start = q_tile_idx * Q_TILE_SIZE

        # Load Qi, Oi, dOi tiles
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        q_i = tl.load(Q_block_ptr)  # [Q_TILE_SIZE, D]

        grad_O_block_ptr = tl.make_block_ptr(
            grad_O_ptr + batch_index * stride_gob,
            shape=(N_QUERIES, D),
            strides=(stride_goq, stride_god),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        do_i = tl.load(grad_O_block_ptr)  # [Q_TILE_SIZE, D]

        # Load Li and Di tiles
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(q_start,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        l_i = tl.load(L_block_ptr)  # [Q_TILE_SIZE]

        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(q_start,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        d_i = tl.load(D_block_ptr)  # [Q_TILE_SIZE]

        # Compute attention scores: S_i^(j) = Q_i @ (K^(j))^T / sqrt(d)
        kt_j = tl.trans(k_j)  # [D, K_TILE_SIZE]
        s_ij = tl.dot(q_i, kt_j, allow_tf32=False) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask if needed
        if is_causal:
            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]  # [Q_TILE_SIZE, 1]
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)[None, :]  # [1, K_TILE_SIZE]
            causal_mask = q_indices >= k_indices  # [Q_TILE_SIZE, K_TILE_SIZE]
            s_ij = tl.where(causal_mask, s_ij, s_ij - 1e6)

        # Compute attention probabilities: P_i^(j) = exp(S_i^(j) - L_i)
        p_ij = tl.exp(s_ij - l_i[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Compute dV^(j) += (P_i^(j))^T * dO_i
        dv_j += tl.dot(tl.trans(p_ij), do_i, allow_tf32=False)  # [K_TILE_SIZE, D]

        # Compute dP_i^(j) = dO_i @ (V^(j))^T
        dp_ij = tl.dot(do_i, tl.trans(v_j), allow_tf32=False)  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Compute dS_i^(j) = P_i^(j) ◦ (dP_i^(j) - D_i) / sqrt(d)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask to dS if needed
        if is_causal:
            ds_ij = tl.where(causal_mask, ds_ij, 0.0)

        # Load current dQ_i from global memory, update with atomic add, write back
        # This is the "Must be atomic for correctness!" part from Algorithm 2
        grad_Q_block_ptr = tl.make_block_ptr(
            grad_Q_ptr + batch_index * stride_gqb,
            shape=(N_QUERIES, D),
            strides=(stride_gqq, stride_gqd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Compute the dQ contribution: dS_i^(j) @ K^(j)
        dq_contribution = tl.dot(ds_ij, k_j, allow_tf32=False)  # [Q_TILE_SIZE, D]
        dq_contribution_cast = dq_contribution.to(grad_Q_block_ptr.type.element_ty)

        # Atomic add to dQ (multiple K/V tiles contribute to same Q positions)
        tl.atomic_add(grad_Q_block_ptr, dq_contribution_cast)

        # Compute dK^(j) += (dS_i^(j))^T @ Q_i (local accumulation, no atomics needed)
        dk_j += tl.dot(tl.trans(ds_ij), q_i, allow_tf32=False)  # [K_TILE_SIZE, D]

    # Write final dK^(j) and dV^(j) to global memory (no atomics needed)
    grad_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_gkb,
        shape=(N_KEYS, D),
        strides=(stride_gkk, stride_gkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dk_j_cast = dk_j.to(grad_K_block_ptr.type.element_ty)
    tl.store(grad_K_block_ptr, dk_j_cast)

    grad_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_gvb,
        shape=(N_KEYS, D),
        strides=(stride_gvk, stride_gvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dv_j_cast = dv_j.to(grad_V_block_ptr.type.element_ty)
    tl.store(grad_V_block_ptr, dv_j_cast)


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q tile
    q = tl.load(Q_block_ptr)  # [Q_TILE_SIZE, D]

    # Initialize on-chip buffers with float32 precision
    o_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], -float('inf'), dtype=tl.float32)

    # Loop over K tiles
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_idx in range(num_k_tiles):
        # Load K and V tiles
        k = tl.load(K_block_ptr)  # [K_TILE_SIZE, D] - coalesced load
        v = tl.load(V_block_ptr)  # [K_TILE_SIZE, D]

        # Transpose K for matrix multiplication: K^T
        k = tl.trans(k)  # [D, K_TILE_SIZE] - fast on-chip transpose

        # Compute attention scores: S_ij = Q_i @ K_j^T
        s_ij = tl.dot(q, k, allow_tf32=False) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal mask if needed
        if is_causal:
            # Create causal mask for this block
            q_start = query_tile_index * Q_TILE_SIZE
            k_start = k_tile_idx * K_TILE_SIZE

            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]  # [Q_TILE_SIZE, 1]
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)[None, :]  # [1, K_TILE_SIZE]

            causal_mask = q_indices >= k_indices  # [Q_TILE_SIZE, K_TILE_SIZE]
            s_ij = tl.where(causal_mask, s_ij, s_ij - 1e6)  # Add -1e6 to masked elements

        # Compute block max and update global max
        m_ij = tl.max(s_ij, axis=1)  # [Q_TILE_SIZE]
        m_i_new = tl.maximum(m_i, m_ij)  # [Q_TILE_SIZE]

        # Compute attention probabilities with numerical stability
        p_ij = tl.exp(s_ij - m_i_new[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Update l_i using online algorithm
        alpha = tl.exp(m_i - m_i_new)  # [Q_TILE_SIZE]
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)  # [Q_TILE_SIZE]

        # Cast P_ij to V's dtype before multiplication
        p_ij = p_ij.to(v.dtype)

        # Update o_i using online algorithm with accumulation
        o_i = alpha[:, None] * o_i + tl.dot(p_ij, v, allow_tf32=False)  # [Q_TILE_SIZE, D]

        # Update m_i
        m_i = m_i_new

        # Advance K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))  # Move along sequence dimension
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Final normalization
    o_i = o_i / l_i[:, None]

    # Compute logsumexp
    l_i = tl.log(l_i) + m_i

    # Cast to appropriate dtype and store results
    o_i = o_i.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, o_i)
    tl.store(L_block_ptr, l_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[torch.Tensor, 'B N_q d'],
                K: Float[torch.Tensor, 'B N_k d'],
                V: Float[torch.Tensor, 'B N_k d'],
                is_causal: bool = False) -> Float[torch.Tensor, 'B N_q d']:

        B, N_q, d = Q.shape
        B, N_k, d = K.shape

        # Choose tile sizes - make them powers of 2 for efficiency
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        scale = 1.0 / math.sqrt(d)

        # Initialize output tensors
        O = torch.empty_like(Q)
        L = torch.empty(B, N_q, device=Q.device, dtype=torch.float32)

        # Launch grid: (num_q_tiles, batch_size)
        num_q_tiles = triton.cdiv(N_q, Q_TILE_SIZE)
        grid = (num_q_tiles, B)

        # Launch kernel
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
    def backward(ctx, grad_O: Float[torch.Tensor, 'B N_q d']):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        B, N_q, d = Q.shape
        B, N_k, d = K.shape

        # Choose tile sizes - make them powers of 2 for efficiency
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        scale = 1.0 / math.sqrt(d)

        # Compute D vector: D_i = sum_j(dO_ij * O_ij) for each query
        D = torch.sum(grad_O * O, dim=-1)  # [B, N_q]

        # Initialize output tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Use two-kernel approach (equivalent to Algorithm 2 but avoids atomic issues)

        # Launch dQ kernel - processes query tiles
        num_q_tiles = triton.cdiv(N_q, Q_TILE_SIZE)
        grid_q = (num_q_tiles, B)

        flash_bwd_dq_kernel[grid_q](
            Q, K, V, L, D,
            grad_O, grad_Q,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            N_q, N_k, scale,
            is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        # Launch dK/dV kernel - processes key tiles
        num_k_tiles = triton.cdiv(N_k, K_TILE_SIZE)
        grid_kv = (num_k_tiles, B)

        flash_bwd_dkv_kernel[grid_kv](
            Q, K, V, L, D,
            grad_O, grad_K, grad_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            N_q, N_k, scale,
            is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        return grad_Q, grad_K, grad_V, None  # None for is_causal gradient