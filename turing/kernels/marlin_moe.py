"""
Marlin W4A16 GEMM for packed 4-bit MoE experts: vLLM v0.31.0's MoE Marlin kernel (Apache-2.0; Marlin by Elias Frantar,
IST-DASLab), vendored in csrc/marlin_moe/ (see NOTICE). Optional, like turing.kernels.marlin: setup.py builds it
(turing.turing_marlin_moe) only with TURING_BUILD_MARLIN=1 where nvcc is available; HAS_MARLIN_MOE is False otherwise
and SparseMoE keeps its Triton kernels (turing.kernels.triton_moe).

Why both: the Triton int4 kernels unpack codes with ordinary arithmetic between the load and the matrix multiply, so
Triton cannot prefetch the weights; cold, they read at 150-180 GB/s of an L4's 262 and prefill-sized inputs are bound
by the unpacking. Marlin fetches weight tiles ahead into shared memory and unpacks with bit operations.

Each expert's two matrices (gate_up [2F, H] and down [H, F], Turing's packed format) are converted once the way
turing.kernels.marlin converts a layer (Marlin's AWQ-INT4 case) and stacked. A forward follows vLLM's
fused_marlin_moe: token-expert pairs grouped by expert in blocks of 8 to 64 rows (triton_moe.moe_align, the layout of
vLLM's moe_align_block_size), one GEMM for gate and up, the activation, one GEMM for down that also applies the
routing weights, and the sum over each token's experts. No host sync: shapes depend only on the token count.
Parity: tests/test_marlin_moe.py.
"""

from typing import NamedTuple, Tuple

import torch

from .marlin import GROUPS, HAS_MARLIN, _workspace, marlin_pack
from .triton_moe import moe_align, moe_sum

try:
    from turing import turing_marlin_moe as _C
    HAS_MARLIN_MOE = HAS_MARLIN  # packing uses the dense extension's repack
except ImportError:
    _C = None
    HAS_MARLIN_MOE = False

Packed = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # codes [E, rows, cols / 2], scales, zeros (turing.models.quant)


class MarlinExperts(NamedTuple):
    gate_up: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # int32 [E, H / 16, 4F] tiles, scales [E, H / g, 2F], zeros
    down: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # int32 [E, F / 16, 2H], scales [E, F / g, H], zeros
    workspace: torch.Tensor  # int32 locks, zero between calls
    hidden: int
    ffn: int


def marlin_moe_supported(hidden: int, ffn: int, group: int, dtype: torch.dtype, device) -> bool:
    """The MoE Marlin kernels run experts of this shape in this build: vLLM's shape rule for both matrices (output
    widths 2F and H in tiles of 64, input widths H and F in tiles of 128 and more than one group), sm_80+."""
    device = torch.device(device)
    return (HAS_MARLIN_MOE and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) and group in GROUPS
            and all(n % 64 == 0 and k % 128 == 0 and k % group == 0 and k > group
                    for n, k in ((2 * ffn, hidden), (hidden, ffn)))
            and torch.cuda.get_device_capability(device) >= (8, 0))


def _pack(packed: Packed, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One matrix of every expert, expert by expert (a layer's worth at once would hold its unpacked codes)."""
    q, s, z = packed
    out = None
    for e in range(q.shape[0]):
        w = marlin_pack(q[e], s[e], z[e], dtype)
        if out is None:
            out = tuple(torch.empty((q.shape[0], *t.shape), dtype=t.dtype, device=t.device)
                        for t in (w.qweight, w.scales, w.zeros))
        for dst, src in zip(out, (w.qweight, w.scales, w.zeros)):
            dst[e].copy_(src)
    return out


def marlin_moe_pack(gate_up: Packed, down: Packed, dtype: torch.dtype) -> MarlinExperts:
    """A SparseMoE's packed experts (on a CUDA device) -> Marlin's copy for activations of `dtype`; check
    marlin_moe_supported first. The inputs are left as they are."""
    with torch.inference_mode(False), torch.no_grad():  # normal tensors: used in and out of inference mode
        # locks of the experts' own: in a captured step their kernels run beside the dense layers'
        # (SparseMoE._two_branches)
        locks = _workspace(gate_up[0].device.index or 0, "moe")
        return MarlinExperts(_pack(gate_up, dtype), _pack(down, dtype), locks, gate_up[0].shape[2] * 2,
                             down[0].shape[2] * 2)


def marlin_moe_store(w: MarlinExperts, slot: int, gate_up: Packed, down: Packed, dtype: torch.dtype) -> None:
    """One expert's packed matrices (on the device, without the expert dimension) into slot `slot` of w: how an
    offloaded layer replaces a resident expert (SparseMoE._load)."""
    packed = [marlin_pack(q, s, z, dtype) for q, s, z in (gate_up, down)]  # both before the slot changes
    for dst, m in zip((w.gate_up, w.down), packed):
        for d, src in zip(dst, (m.qweight, m.scales, m.zeros)):
            d[slot].copy_(src)


def _block_rows(pairs: int, num_experts: int) -> int:
    """vLLM's choice (fused_marlin_moe): the smallest block of 8 to 64 rows that an average expert does not fill."""
    for block in (8, 16, 32, 48, 64):
        if pairs / num_experts / block < 0.9:
            break
    return block


def marlin_moe(x: torch.Tensor, w: MarlinExperts, weights: torch.Tensor, experts: torch.Tensor, act) -> torch.Tensor:
    """sum_j weights[t, j] * down[e] @ (act(gate[e] @ x[t]) * up[e] @ x[t]) with e = experts[t, j], for x [T, H] in
    the dtype w was packed for, weights [T, top_k] (fp32) and experts [T, top_k]. Returns [T, H] in x's dtype; the
    top_k terms are summed in fp32, as the Triton kernels do."""
    T, H = x.shape
    top_k, E, F = experts.shape[1], w.gate_up[0].shape[0], w.ffn
    n = T * top_k
    block = _block_rows(n, E)
    pairs, block_expert, used = moe_align(experts, E, block)  # the kernel reads the first `used` rows only
    pairs, block_expert = pairs.to(torch.int32), block_expert.to(torch.int32)  # no-ops on CUDA but for one token
    weights = weights.reshape(-1).float().contiguous()
    x = x.contiguous()
    gate_up = _C.moe_wna16_marlin_gemm(x, *w.gate_up, w.workspace, pairs, block_expert, used, weights, block, top_k,
                                       False, T, 2 * F, H)  # [pairs, 2F]: gate then up
    if act is torch.nn.functional.silu:
        from .triton_fused_ops import silu_mul
        h = silu_mul(gate_up[:, :F], gate_up[:, F:]).contiguous()
    else:
        h = (act(gate_up[:, :F]) * gate_up[:, F:]).contiguous()
    # down: every pair is its own row now, scaled by its routing weight in the kernel
    y = _C.moe_wna16_marlin_gemm(h, *w.down, w.workspace, pairs, block_expert, used, weights, block, 1, True, n, H, F)
    return moe_sum(y, T, x.dtype)
