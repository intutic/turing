"""
Grouped GEMM for top-k routed MoE layers on CUDA, with no host synchronisation, so a decode step that contains MoE
layers can be captured in a CUDA graph.

The token-expert pairs (token t's j-th choice is pair t * top_k + j) are sorted by expert on the device and each
expert's pairs padded to a multiple of BLOCK_M rows (`moe_align`). Two Triton kernels then run one program per
(block of rows, tile of columns), each block belonging to one expert, so every expert's weights are read once per
block of its tokens:

  gate_up: h[row] = act(x[token] @ gate[e].T) * (x[token] @ up[e].T)   (SiLU, as in SparseMoE)
  down:    y[pair] = routing_weight[pair] * (h[row] @ down[e].T)

and the output is the sum of each token's top_k rows of y. Shapes depend only on the number of pairs and experts.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU and MPS builds: SparseMoE keeps its per-expert loop
    triton = tl = None
    HAS_TRITON = False


def moe_align(experts: torch.Tensor, num_experts: int, block_m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairs grouped by expert, on the device of `experts` ([T, top_k] expert ids).

    Returns sorted_pairs [P] (pair ids, each expert's group padded to a multiple of block_m with the sentinel
    T * top_k) and block_expert [P // block_m] (the expert of each block; num_experts past the last group).
    P = n + min(num_experts, n) * (block_m - 1) for n pairs, an upper bound that depends only on the shapes."""
    flat = experts.reshape(-1)
    n = flat.numel()
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    order = torch.argsort(flat, stable=True)  # pair ids, grouped by expert
    sorted_e = flat[order]
    rank = torch.arange(n, device=flat.device) - (torch.cumsum(counts, 0) - counts)[sorted_e]
    total = n + min(num_experts, n) * (block_m - 1)
    total = -(-total // block_m) * block_m
    sorted_pairs = torch.full((total,), n, dtype=torch.int64, device=flat.device)
    sorted_pairs.scatter_(0, pad_start[sorted_e] + rank, order)
    block_start = torch.arange(0, total, block_m, device=flat.device)
    block_expert = torch.searchsorted(pad_end, block_start, right=True)
    return sorted_pairs.to(torch.int32), block_expert.to(torch.int32)


def fused_moe_supported(x: torch.Tensor, gate_up: torch.Tensor, act) -> bool:
    return (HAS_TRITON and x.is_cuda and gate_up.is_cuda and act is torch.nn.functional.silu
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32) and gate_up.dtype == x.dtype)


if HAS_TRITON:
    @triton.jit
    def _moe_gate_up_kernel(X, W, PAIRS, BLOCK_E, H, n_pairs, num_experts, K, F,
                            s_xt, s_we, s_wn, s_hr,
                            TOP_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr, IEEE: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        tok = pair // TOP_K
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < F
        offs_k = tl.arange(0, BLOCK_K)
        w = W + e.to(tl.int64) * s_we
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            k_ok = k < K
            x = tl.load(X + tok[:, None].to(tl.int64) * s_xt + k[None, :], mask=valid[:, None] & k_ok[None, :],
                        other=0.0)
            wg = tl.load(w + offs_n[None, :] * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            wu = tl.load(w + (offs_n[None, :] + F) * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :],
                         other=0.0)
            if IEEE:
                acc_g += tl.dot(x, wg, input_precision="ieee")
                acc_u += tl.dot(x, wu, input_precision="ieee")
            else:
                acc_g += tl.dot(x, wg)
                acc_u += tl.dot(x, wu)
        # SparseMoE rounds both projections to the model dtype before the activation; so does this
        g = acc_g.to(H.dtype.element_ty).to(tl.float32)
        u = acc_u.to(H.dtype.element_ty).to(tl.float32)
        h = (g * tl.sigmoid(g)).to(H.dtype.element_ty).to(tl.float32) * u
        tl.store(H + rows[:, None].to(tl.int64) * s_hr + offs_n[None, :], h.to(H.dtype.element_ty),
                 mask=valid[:, None] & n_ok[None, :])

    @triton.jit
    def _moe_down_kernel(Hs, W, PAIRS, BLOCK_E, RW, Y, n_pairs, num_experts, F, N,
                         s_hr, s_we, s_wn, s_yp,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        w = W + e.to(tl.int64) * s_we
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, F, BLOCK_K):
            k = k0 + offs_k
            k_ok = k < F
            h = tl.load(Hs + rows[:, None].to(tl.int64) * s_hr + k[None, :], mask=valid[:, None] & k_ok[None, :],
                        other=0.0)
            wd = tl.load(w + offs_n[None, :] * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            if IEEE:
                acc += tl.dot(h, wd, input_precision="ieee")
            else:
                acc += tl.dot(h, wd)
        # as SparseMoE: the expert's output in the model dtype, then scaled by its fp32 routing weight
        rw = tl.load(RW + pair, mask=valid, other=0.0)
        y = acc.to(Hs.dtype.element_ty).to(tl.float32) * rw[:, None]
        tl.store(Y + pair[:, None].to(tl.int64) * s_yp + offs_n[None, :], y, mask=valid[:, None] & n_ok[None, :])


def fused_moe(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, weights: torch.Tensor,
              experts: torch.Tensor) -> torch.Tensor:
    """sum_j weights[t, j] * down[e] @ (silu(gate[e] @ x[t]) * up[e] @ x[t]) with e = experts[t, j], for x [T, H],
    gate_up [E, 2F, H], down [E, H, F], weights [T, top_k] (fp32) and experts [T, top_k]. Returns [T, H] in x's
    dtype; the top_k terms are summed in fp32."""
    T, H = x.shape
    E, two_f, _ = gate_up.shape
    F = two_f // 2
    top_k = experts.shape[1]
    n = T * top_k
    block_m = 16 if n <= 1024 else 64
    sorted_pairs, block_expert = moe_align(experts, E, block_m)
    x, gate_up, down = x.contiguous(), gate_up.contiguous(), down.contiguous()
    ieee = x.dtype == torch.float32
    h = torch.empty(sorted_pairs.numel(), F, device=x.device, dtype=x.dtype)
    nb = block_expert.numel()
    _moe_gate_up_kernel[(nb, triton.cdiv(F, 64))](
        x, gate_up, sorted_pairs, block_expert, h, n, E, H, F,
        x.stride(0), gate_up.stride(0), gate_up.stride(1), h.stride(0),
        TOP_K=top_k, BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64, IEEE=ieee, num_warps=4)
    y = torch.empty(n, H, device=x.device, dtype=torch.float32)
    _moe_down_kernel[(nb, triton.cdiv(H, 64))](
        h, down, sorted_pairs, block_expert, weights.reshape(-1).float().contiguous(), y, n, E, F, H,
        h.stride(0), down.stride(0), down.stride(1), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64, IEEE=ieee, num_warps=4)
    return y.view(T, top_k, H).sum(1).to(x.dtype)
