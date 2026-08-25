"""
2D-Tiled Subspace Recirculation Triton Kernel for Deep-to-Shallow Recurrence.
"""

from typing import Optional
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
    def _fused_recirculation_mix_kernel(
        Shallow_ptr, Deep_ptr, U_ptr, Out_ptr,
        M, D,
        alpha,
        stride_sm, stride_sd,
        stride_dm, stride_dd,
        stride_ud, stride_ur,
        stride_om, stride_od,
        RANK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, RANK)

        # 1. Project Deep belief state into Rank-64 Subspace
        s_sub = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

        for d_step in range(0, D, BLOCK_D):
            offs_d_in = d_step + tl.arange(0, BLOCK_D)
            mask_d = (offs_m[:, None] < M) & (offs_d_in[None, :] < D)
            d_val = tl.load(Deep_ptr + offs_m[:, None] * stride_dm + offs_d_in[None, :] * stride_dd, mask=mask_d, other=0.0)

            u_val = tl.load(U_ptr + offs_d_in[:, None] * stride_ud + offs_r[None, :] * stride_ur, mask=(offs_d_in[:, None] < D), other=0.0)
            s_sub += tl.dot(d_val.to(tl.float16), u_val.to(tl.float16))

        # 2. Project Subspace state back to dimension slice
        u_slice = tl.load(U_ptr + offs_d[None, :] * stride_ud + offs_r[:, None] * stride_ur, mask=(offs_d[None, :] < D), other=0.0)
        recon = tl.dot(s_sub.to(tl.float16), u_slice.to(tl.float16))

        # 3. Load shallow activations and fuse mixing
        mask_out = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        shallow_val = tl.load(Shallow_ptr + offs_m[:, None] * stride_sm + offs_d[None, :] * stride_sd, mask=mask_out, other=0.0)

        out_val = shallow_val + (alpha * recon)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, out_val.to(tl.float16), mask=mask_out)

def launch_triton_subspace_recirculation(
    h_shallow: torch.Tensor,
    h_deep: torch.Tensor,
    u_proj: torch.Tensor,
    alpha: float = 0.15
) -> torch.Tensor:
    """
    Python launcher for 2D-tiled fused subspace recirculation.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in current environment")

    orig_shape = h_shallow.shape
    if h_shallow.dim() == 3:
        b, s, d = orig_shape
        h_shallow_2d = h_shallow.view(-1, d).contiguous()
        h_deep_2d = h_deep.view(-1, d).contiguous()
    else:
        h_shallow_2d = h_shallow.contiguous()
        h_deep_2d = h_deep.contiguous()

    m, d = h_shallow_2d.shape
    rank = u_proj.shape[1]

    out_2d = torch.empty_like(h_shallow_2d)

    BLOCK_M = 16
    BLOCK_D = 128

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(d, BLOCK_D))

    _fused_recirculation_mix_kernel[grid](
        h_shallow_2d, h_deep_2d, u_proj, out_2d,
        m, d,
        alpha,
        h_shallow_2d.stride(0), h_shallow_2d.stride(1),
        h_deep_2d.stride(0), h_deep_2d.stride(1),
        u_proj.stride(0), u_proj.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D
    )

    return out_2d.view(orig_shape)
