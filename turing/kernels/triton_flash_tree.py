"""
SRAM-Resident Flash-Tree-Attention Triton Kernel for Speculative DAG Verification.
"""

from typing import Optional
import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:
    @triton.jit
    def _fused_flash_tree_attention_kernel(
        Q_ptr, K_prefix_ptr, V_prefix_ptr, K_tree_ptr, V_tree_ptr, Tree_Mask_ptr, Out_ptr,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kpb, stride_kph, stride_kpn, stride_kpd,
        stride_vpb, stride_vph, stride_vpn, stride_vpd,
        stride_ktb, stride_kth, stride_ktn, stride_ktd,
        stride_vtb, stride_vth, stride_vtn, stride_vtd,
        stride_tmm, stride_tmn,
        stride_ob, stride_oh, stride_om, stride_od,
        N_QUERIES, PREFIX_LEN,
        NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_b_h = tl.program_id(0)
        pid_m = tl.program_id(1)

        batch_id = pid_b_h // NUM_HEADS
        head_id = pid_b_h % NUM_HEADS

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        mask_m = offs_m < N_QUERIES
        q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # Online Softmax state
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Phase 1: Streaming Prefix KV Blocks
        if PREFIX_LEN > 0:
            for start_n in range(0, PREFIX_LEN, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < PREFIX_LEN

                kp_ptrs = K_prefix_ptr + batch_id * stride_kpb + head_id * stride_kph + offs_n[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
                vp_ptrs = V_prefix_ptr + batch_id * stride_vpb + head_id * stride_vph + offs_n[:, None] * stride_vpn + offs_d[None, :] * stride_vpd

                kp = tl.load(kp_ptrs, mask=mask_n[None, :], other=0.0)
                vp = tl.load(vp_ptrs, mask=mask_n[:, None], other=0.0)

                qk = tl.dot(q, kp) * sm_scale
                qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

                m_curr = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp(m_i - m_curr)
                p = tl.exp(qk - m_curr[:, None])

                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vp)
                m_i = m_curr

        # Phase 2: Streaming Speculative Tree Nodes (Filtered by 2D DAG Mask)
        for start_t in range(0, N_QUERIES, BLOCK_N):
            offs_t = start_t + tl.arange(0, BLOCK_N)
            mask_t = offs_t < N_QUERIES

            kt_ptrs = K_tree_ptr + batch_id * stride_ktb + head_id * stride_kth + offs_t[None, :] * stride_ktn + offs_d[:, None] * stride_ktd
            vt_ptrs = V_tree_ptr + batch_id * stride_vtb + head_id * stride_vth + offs_t[:, None] * stride_vtn + offs_d[None, :] * stride_vtd

            kt = tl.load(kt_ptrs, mask=mask_t[None, :], other=0.0)
            vt = tl.load(vt_ptrs, mask=mask_t[:, None], other=0.0)

            # Load DAG tree mask
            tm_ptrs = Tree_Mask_ptr + offs_m[:, None] * stride_tmm + offs_t[None, :] * stride_tmn
            tree_mask = tl.load(tm_ptrs, mask=mask_m[:, None] & mask_t[None, :], other=float("-inf"))

            qk_tree = (tl.dot(q, kt) * sm_scale) + tree_mask
            qk_tree = tl.where(mask_m[:, None] & mask_t[None, :], qk_tree, float("-inf"))

            m_curr = tl.maximum(m_i, tl.max(qk_tree, axis=1))
            alpha = tl.exp(m_i - m_curr)
            p = tl.exp(qk_tree - m_curr[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
            m_i = m_curr

        # Normalization
        out = acc / l_i[:, None]
        out_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_m[:, None])

def launch_triton_flash_tree_attention(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_tree: torch.Tensor,
    v_tree: torch.Tensor,
    tree_mask: torch.Tensor
) -> torch.Tensor:
    """
    Python launcher for Flash-Tree-Attention kernel.
    q: [Batch, Heads, NumQueries, HeadDim]
    k_prefix, v_prefix: [Batch, Heads, PrefixLen, HeadDim]
    k_tree, v_tree: [Batch, Heads, NumQueries, HeadDim]
    tree_mask: [NumQueries, NumQueries]
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in current environment")

    batch, heads, n_queries, head_dim = q.shape
    prefix_len = k_prefix.shape[2] if k_prefix is not None and k_prefix.numel() > 0 else 0

    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M = 16
    BLOCK_N = 32

    grid = (batch * heads, triton.cdiv(n_queries, BLOCK_M))

    _fused_flash_tree_attention_kernel[grid](
        q, k_prefix, v_prefix, k_tree, v_tree, tree_mask, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_prefix.stride(0) if prefix_len > 0 else 0,
        k_prefix.stride(1) if prefix_len > 0 else 0,
        k_prefix.stride(2) if prefix_len > 0 else 0,
        k_prefix.stride(3) if prefix_len > 0 else 0,
        v_prefix.stride(0) if prefix_len > 0 else 0,
        v_prefix.stride(1) if prefix_len > 0 else 0,
        v_prefix.stride(2) if prefix_len > 0 else 0,
        v_prefix.stride(3) if prefix_len > 0 else 0,
        k_tree.stride(0), k_tree.stride(1), k_tree.stride(2), k_tree.stride(3),
        v_tree.stride(0), v_tree.stride(1), v_tree.stride(2), v_tree.stride(3),
        tree_mask.stride(0), tree_mask.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_QUERIES=n_queries,
        PREFIX_LEN=prefix_len,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return out
