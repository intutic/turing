"""
Triton Flash-Decode Kernel for Latent Subspace Attention (SPECTRA Mode-B).
Performs direct attention in the rank-R latent subspace against INT8 cached singular coordinates.
Pre-absorbs the up-projection into Query: Q' = Q @ W_UPk^T in R^(GRP x R),
evaluates Softmax(Q' @ (Ck * sk)^T) @ (Cv * sv) in-SRAM,
and post-projects output: Out = Agg @ W_UPv in R^(GRP x d).
"""

from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _splitk_latent_decode_kernel(
        Qp_ptr,         # [B * NKV, GRP, R] (float16 / float32)
        Ck_ptr,         # [B, N, R] (int8)
        Sk_ptr,         # [B, N] (float16 / float32)
        Cv_ptr,         # [B, N, R] (int8)
        Sv_ptr,         # [B, N] (float16 / float32)
        M_out_ptr,      # [B * NKV, N_SPLIT, GRP] (float32)
        L_out_ptr,      # [B * NKV, N_SPLIT, GRP] (float32)
        A_out_ptr,      # [B * NKV, N_SPLIT, GRP, R] (float32)
        n_tokens,
        scale,
        stride_qp_bh,
        stride_ck_b, stride_ck_n,
        stride_sk_b,
        stride_m_bh, stride_m_s,
        stride_a_bh, stride_a_s,
        R: tl.constexpr,
        GRP: tl.constexpr,
        NKV: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_SPLIT: tl.constexpr
    ):
        bh = tl.program_id(0)
        sp = tl.program_id(1)
        b = bh // NKV

        offs_r = tl.arange(0, R)
        offs_g = tl.arange(0, GRP)

        # Load Q' [GRP, R]
        qp = tl.load(Qp_ptr + bh * stride_qp_bh + offs_g[:, None] * R + offs_r[None, :]).to(tl.float32)

        chunk = tl.cdiv(n_tokens, N_SPLIT)
        start = sp * chunk
        end = tl.minimum(start + chunk, n_tokens)

        ck_base = Ck_ptr + b * stride_ck_b
        cv_base = Cv_ptr + b * stride_ck_b
        sk_base = Sk_ptr + b * stride_sk_b
        sv_base = Sv_ptr + b * stride_sk_b

        m_i = tl.zeros([GRP], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([GRP], dtype=tl.float32)
        acc = tl.zeros([GRP, R], dtype=tl.float32)

        for s0 in range(start, end, BLOCK_N):
            offs_n = s0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end

            # Load C_K [BLOCK_N, R] (int8) and scale_K [BLOCK_N]
            ck_vals = tl.load(ck_base + offs_n[:, None] * stride_ck_n + offs_r[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
            sk_vals = tl.load(sk_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            kf = ck_vals * sk_vals[:, None]  # Dequantize in SRAM: [BLOCK_N, R]

            # In-SRAM dot product Q' @ K'^T: [GRP, BLOCK_N]
            scores = tl.dot(qp, tl.trans(kf)) * scale
            scores = tl.where(mask_n[None, :], scores, -float("inf"))

            # Online Softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # Load C_V [BLOCK_N, R] (int8) and scale_V [BLOCK_N]
            cv_vals = tl.load(cv_base + offs_n[:, None] * stride_ck_n + offs_r[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
            sv_vals = tl.load(sv_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            vf = cv_vals * sv_vals[:, None]

            # Accumulate: Softmax(P) @ V': [GRP, R]
            acc = acc * alpha[:, None] + tl.dot(p, vf)
            m_i = m_new

        tl.store(M_out_ptr + bh * stride_m_bh + sp * stride_m_s + offs_g, m_i)
        tl.store(L_out_ptr + bh * stride_m_bh + sp * stride_m_s + offs_g, l_i)
        tl.store(A_out_ptr + bh * stride_a_bh + sp * stride_a_s + offs_g[:, None] * R + offs_r[None, :], acc)


    @triton.jit
    def _combine_splitk_kernel(
        M_in_ptr,
        L_in_ptr,
        A_in_ptr,
        Out_ptr,
        stride_m_bh, stride_m_s,
        stride_a_bh, stride_a_s,
        stride_o_bh,
        R: tl.constexpr,
        GRP: tl.constexpr,
        N_SPLIT: tl.constexpr
    ):
        bh = tl.program_id(0)
        offs_g = tl.arange(0, GRP)
        offs_r = tl.arange(0, R)

        m = tl.zeros([GRP], dtype=tl.float32) - float("inf")
        for sp in range(N_SPLIT):
            ms = tl.load(M_in_ptr + bh * stride_m_bh + sp * stride_m_s + offs_g)
            m = tl.maximum(m, ms)

        l_total = tl.zeros([GRP], dtype=tl.float32)
        acc_total = tl.zeros([GRP, R], dtype=tl.float32)

        for sp in range(N_SPLIT):
            ms = tl.load(M_in_ptr + bh * stride_m_bh + sp * stride_m_s + offs_g)
            ls = tl.load(L_in_ptr + bh * stride_m_bh + sp * stride_m_s + offs_g)
            a_val = tl.load(A_in_ptr + bh * stride_a_bh + sp * stride_a_s + offs_g[:, None] * R + offs_r[None, :])

            weight = tl.exp(ms - m)
            l_total += weight * ls
            acc_total += weight[:, None] * a_val

        out = acc_total / tl.maximum(l_total[:, None], 1e-6)
        tl.store(Out_ptr + bh * stride_o_bh + offs_g[:, None] * R + offs_r[None, :], out)


def triton_latent_flash_decode(
    qp: torch.Tensor,     # [Batch, NKV, GRP, R] (Pre-projected Query in Subspace)
    ck: torch.Tensor,     # [Batch, SeqLen, R] (INT8 singular coordinates)
    sk: torch.Tensor,     # [Batch, SeqLen] (Float scale)
    cv: torch.Tensor,     # [Batch, SeqLen, R] (INT8 singular coordinates)
    sv: torch.Tensor,     # [Batch, SeqLen] (Float scale)
    head_dim: int,
    n_split: int = 4,
    block_n: int = 64
) -> torch.Tensor:
    """
    Executes in-SRAM Latent Flash-Decode (Mode-B) on CUDA.
    Returns: [Batch, NKV, GRP, R]
    """
    if not qp.is_cuda or not HAS_TRITON:
        # Fallback reference computation
        kf = ck.to(torch.float32) * sk.unsqueeze(-1)  # [Batch, SeqLen, R]
        vf = cv.to(torch.float32) * sv.unsqueeze(-1)
        scale = 1.0 / (head_dim ** 0.5)

        # qp: [B, NKV, GRP, R], kf: [B, N, R] -> [B, NKV, GRP, N]
        scores = torch.einsum('bkgr,bnr->bkgn', qp.to(torch.float32), kf) * scale
        p = torch.softmax(scores, dim=-1)
        # p: [B, NKV, GRP, N], vf: [B, N, R] -> [B, NKV, GRP, R]
        out = torch.einsum('bkgn,bnr->bkgr', p, vf)
        return out.to(qp.dtype)

    B, NKV, GRP, R = qp.shape
    n_tokens = ck.shape[1]
    BH = B * NKV
    qp_flat = qp.reshape(BH, GRP, R).contiguous()
    dev = qp.device

    M = torch.empty(BH, n_split, GRP, device=dev, dtype=torch.float32)
    L = torch.empty(BH, n_split, GRP, device=dev, dtype=torch.float32)
    A = torch.empty(BH, n_split, GRP, R, device=dev, dtype=torch.float32)
    out = torch.empty(BH, GRP, R, device=dev, dtype=qp.dtype)

    scale = 1.0 / (head_dim ** 0.5)

    _splitk_latent_decode_kernel[(BH, n_split)](
        qp_flat, ck, sk, cv, sv, M, L, A,
        n_tokens, scale,
        qp_flat.stride(0),
        ck.stride(0), ck.stride(1),
        sk.stride(0),
        M.stride(0), M.stride(1),
        A.stride(0), A.stride(1),
        R=R, GRP=GRP, NKV=NKV,
        BLOCK_N=block_n, N_SPLIT=n_split
    )

    _combine_splitk_kernel[(BH,)](
        M, L, A, out,
        M.stride(0), M.stride(1),
        A.stride(0), A.stride(1),
        out.stride(0),
        R=R, GRP=GRP, N_SPLIT=n_split
    )

    return out.reshape(B, NKV, GRP, R)
