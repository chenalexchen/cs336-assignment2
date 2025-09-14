import torch
import math
import triton
import triton.language as tl

from jaxtyping import Float


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
        raise NotImplementedError('backward is not yet implemented')


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
        raise NotImplementedError('backward is not yet implemented')