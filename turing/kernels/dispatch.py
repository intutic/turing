"""
Dynamic Hardware Dispatcher for CUDA Triton, Apple Silicon MPS, and Vectorized CPU Ops.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

HAS_CUDA = torch.cuda.is_available()

try:
    from .triton_swiglu import launch_triton_swiglu
    from .triton_flash_tree import launch_triton_flash_tree_attention
    from .triton_w4a16 import launch_triton_w4a16_gemm
    from .triton_recirculation import launch_triton_subspace_recirculation
    from .triton_swiglu import HAS_TRITON
except ImportError:
    HAS_TRITON = False

def dispatch_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_size: int = 256
) -> torch.Tensor:
    """
    Dispatches SwiGLU computation with active channel pruning.
    Uses Triton on CUDA if available, otherwise vectorized PyTorch.
    """
    if HAS_CUDA and HAS_TRITON and x.is_cuda and x.dtype == torch.float16:
        try:
            return launch_triton_swiglu(x, w_gate, w_up, w_down, active_tiles, tile_size)
        except Exception:
            pass # Fall back to PyTorch

    # Vectorized PyTorch Fallback
    active_count = len(active_tiles)
    if active_count == 0:
        return torch.zeros_like(x)

    active_idx_list = active_tiles.tolist()
    # Fast path: Contiguous active tiles (0 .. active_count - 1)
    if active_idx_list == list(range(active_count)):
        active_dim = active_count * tile_size
        w_g_act = w_gate[:, :active_dim]
        w_u_act = w_up[:, :active_dim]
        w_d_act = w_down[:active_dim, :]
    else:
        gate_slices = []
        up_slices = []
        down_slices = []
        for t in active_idx_list:
            c_start = t * tile_size
            c_end = c_start + tile_size
            gate_slices.append(w_gate[:, c_start:c_end])
            up_slices.append(w_up[:, c_start:c_end])
            down_slices.append(w_down[c_start:c_end, :])

        w_g_act = torch.cat(gate_slices, dim=-1)
        w_u_act = torch.cat(up_slices, dim=-1)
        w_d_act = torch.cat(down_slices, dim=0)

    gate = F.silu(torch.matmul(x, w_g_act))
    up = torch.matmul(x, w_u_act)
    h = gate * up
    out = torch.matmul(h, w_d_act)
    return out

def dispatch_flash_tree_attention(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_tree: torch.Tensor,
    v_tree: torch.Tensor,
    tree_mask: torch.Tensor
) -> torch.Tensor:
    """
    Dispatches Flash-Tree-Attention DAG speculative verification.
    """
    if HAS_CUDA and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        try:
            return launch_triton_flash_tree_attention(q, k_prefix, v_prefix, k_tree, v_tree, tree_mask)
        except Exception:
            pass

    # PyTorch Fallback
    # Concatenate prefix + tree keys/values
    batch, heads, n_q, head_dim = q.shape
    prefix_len = k_prefix.shape[2] if k_prefix is not None and k_prefix.numel() > 0 else 0

    if prefix_len > 0:
        k_all = torch.cat([k_prefix, k_tree], dim=2)
        v_all = torch.cat([v_prefix, v_tree], dim=2)
    else:
        k_all = k_tree
        v_all = v_tree

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k_all.transpose(-1, -2)) * scale # [Batch, Heads, N_Q, PrefixLen + N_Q]

    # Apply causal tree mask to the tree part
    if prefix_len > 0:
        scores[:, :, :, prefix_len:] += tree_mask.unsqueeze(0).unsqueeze(0)
    else:
        scores += tree_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v_all)
    return out

def dispatch_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    """
    Dispatches W4A16 packed INT4 matrix-vector multiplication.
    """
    if HAS_CUDA and HAS_TRITON and x.is_cuda and x.dtype == torch.float16:
        try:
            return launch_triton_w4a16_gemm(x, w_packed, scales, group_size)
        except Exception:
            pass

    # PyTorch Fallback
    n, k_half = w_packed.shape
    k = k_half * 2

    # Unpack nibbles
    w_low = (w_packed & 0x0F).to(torch.float32) - 8.0
    w_high = ((w_packed >> 4) & 0x0F).to(torch.float32) - 8.0

    w_unpacked = torch.empty((n, k), dtype=torch.float32, device=w_packed.device)
    w_unpacked[:, 0::2] = w_low
    w_unpacked[:, 1::2] = w_high

    # Expand scales
    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :k].to(torch.float32)
    w_dequant = (w_unpacked * scales_expanded).to(x.dtype)

    return torch.matmul(x, w_dequant.t())

def dispatch_subspace_recirculation(
    h_shallow: torch.Tensor,
    h_deep: torch.Tensor,
    u_proj: torch.Tensor,
    alpha: float = 0.15
) -> torch.Tensor:
    """
    Dispatches Subspace Recirculation mixing.
    """
    if HAS_CUDA and HAS_TRITON and h_shallow.is_cuda and h_shallow.dtype == torch.float16:
        try:
            return launch_triton_subspace_recirculation(h_shallow, h_deep, u_proj, alpha)
        except Exception:
            pass

    # PyTorch Fallback
    dtype = h_shallow.dtype
    s_sub = torch.matmul(h_deep, u_proj.to(dtype=dtype, device=h_shallow.device))
    recon = torch.matmul(s_sub, u_proj.t().to(dtype=dtype, device=h_shallow.device))
    return h_shallow + (alpha * recon)
