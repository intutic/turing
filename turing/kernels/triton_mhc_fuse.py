"""
Triton GPU Kernel: Fused Manifold-Constrained Hyper-Connections (mHC).
Combines stream pre-reduction, layer residual updates, and doubly-stochastic stream mixing in GPU SRAM.
"""

from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _mhc_stream_mix_kernel(
        streams_ptr,        # [N_tokens, NumStreams, HiddenDim]
        res_map_ptr,        # [NumStreams, NumStreams]
        layer_up_ptr,       # [N_tokens, HiddenDim]
        post_weights_ptr,   # [NumStreams]
        out_streams_ptr,    # [N_tokens, NumStreams, HiddenDim]
        n_tokens: tl.constexpr,
        hidden_dim: tl.constexpr,
        num_streams: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
    ):
        t_idx = tl.program_id(0) # token index
        d_idx = tl.program_id(1) * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_idx < hidden_dim

        if t_idx >= n_tokens:
            return

        # Load layer update [BLOCK_SIZE_D]
        up_val = tl.load(layer_up_ptr + t_idx * hidden_dim + d_idx, mask=d_mask, other=0.0)

        # For 4 streams: load streams into registers, add beta * update, and multiply with H_res
        s0 = tl.load(streams_ptr + (t_idx * num_streams + 0) * hidden_dim + d_idx, mask=d_mask, other=0.0)
        s1 = tl.load(streams_ptr + (t_idx * num_streams + 1) * hidden_dim + d_idx, mask=d_mask, other=0.0)
        s2 = tl.load(streams_ptr + (t_idx * num_streams + 2) * hidden_dim + d_idx, mask=d_mask, other=0.0)
        s3 = tl.load(streams_ptr + (t_idx * num_streams + 3) * hidden_dim + d_idx, mask=d_mask, other=0.0)

        # Load post gating weights
        b0 = tl.load(post_weights_ptr + 0)
        b1 = tl.load(post_weights_ptr + 1)
        b2 = tl.load(post_weights_ptr + 2)
        b3 = tl.load(post_weights_ptr + 3)

        # Sigmoid activation on weights
        b0 = 1.0 / (1.0 + tl.exp(-b0))
        b1 = 1.0 / (1.0 + tl.exp(-b1))
        b2 = 1.0 / (1.0 + tl.exp(-b2))
        b3 = 1.0 / (1.0 + tl.exp(-b3))

        u0 = s0 + b0 * up_val
        u1 = s1 + b1 * up_val
        u2 = s2 + b2 * up_val
        u3 = s3 + b3 * up_val

        # Load H_res [4, 4]
        h00 = tl.load(res_map_ptr + 0 * 4 + 0)
        h01 = tl.load(res_map_ptr + 0 * 4 + 1)
        h02 = tl.load(res_map_ptr + 0 * 4 + 2)
        h03 = tl.load(res_map_ptr + 0 * 4 + 3)

        h10 = tl.load(res_map_ptr + 1 * 4 + 0)
        h11 = tl.load(res_map_ptr + 1 * 4 + 1)
        h12 = tl.load(res_map_ptr + 1 * 4 + 2)
        h13 = tl.load(res_map_ptr + 1 * 4 + 3)

        h20 = tl.load(res_map_ptr + 2 * 4 + 0)
        h21 = tl.load(res_map_ptr + 2 * 4 + 1)
        h22 = tl.load(res_map_ptr + 2 * 4 + 2)
        h23 = tl.load(res_map_ptr + 2 * 4 + 3)

        h30 = tl.load(res_map_ptr + 3 * 4 + 0)
        h31 = tl.load(res_map_ptr + 3 * 4 + 1)
        h32 = tl.load(res_map_ptr + 3 * 4 + 2)
        h33 = tl.load(res_map_ptr + 3 * 4 + 3)

        out0 = u0 * h00 + u1 * h10 + u2 * h20 + u3 * h30
        out1 = u0 * h01 + u1 * h11 + u2 * h21 + u3 * h31
        out2 = u0 * h02 + u1 * h12 + u2 * h22 + u3 * h32
        out3 = u0 * h03 + u1 * h13 + u2 * h23 + u3 * h33

        tl.store(out_streams_ptr + (t_idx * num_streams + 0) * hidden_dim + d_idx, out0, mask=d_mask)
        tl.store(out_streams_ptr + (t_idx * num_streams + 1) * hidden_dim + d_idx, out1, mask=d_mask)
        tl.store(out_streams_ptr + (t_idx * num_streams + 2) * hidden_dim + d_idx, out2, mask=d_mask)
        tl.store(out_streams_ptr + (t_idx * num_streams + 3) * hidden_dim + d_idx, out3, mask=d_mask)


def mhc_stream_mix_cuda(
    streams: torch.Tensor,
    layer_update: torch.Tensor,
    res_map: torch.Tensor,
    post_weights: torch.Tensor
) -> torch.Tensor:
    """
    Fused multi-stream residual update and Birkhoff mixing.
    """
    orig_shape = streams.shape
    num_streams = orig_shape[-2]
    hidden_dim = orig_shape[-1]
    n_tokens = streams.numel() // (num_streams * hidden_dim)

    if not streams.is_cuda or not HAS_TRITON or num_streams != 4:
        # Pure PyTorch fallback
        beta = torch.sigmoid(post_weights).view(1, 1, num_streams, 1)
        updated = streams + beta * layer_update.unsqueeze(-2)
        return torch.matmul(updated.transpose(-2, -1), res_map.t()).transpose(-2, -1)

    streams_flat = streams.view(n_tokens, num_streams, hidden_dim).contiguous()
    layer_up_flat = layer_update.view(n_tokens, hidden_dim).contiguous()
    out = torch.empty_like(streams_flat)

    BLOCK_SIZE_D = 128
    grid = (n_tokens, triton.cdiv(hidden_dim, BLOCK_SIZE_D))

    _mhc_stream_mix_kernel[grid](
        streams_flat,
        res_map.contiguous(),
        layer_up_flat,
        post_weights.contiguous(),
        out,
        n_tokens=n_tokens,
        hidden_dim=hidden_dim,
        num_streams=num_streams,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out.view(orig_shape)
