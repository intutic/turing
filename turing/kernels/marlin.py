"""
Marlin W4A16 GEMM for Turing's packed 4-bit layers: vLLM v0.31.0's Marlin kernel (Apache-2.0; Marlin by Elias Frantar,
IST-DASLab), vendored in csrc/marlin/ (see NOTICE). Optional: setup.py builds it (turing.turing_marlin) only with
TURING_BUILD_MARLIN=1 where nvcc is available; HAS_MARLIN is False otherwise and QuantLinear keeps its other paths.

Turing's format (turing.models.quant: qweight [N, K/2] uint8, low nibble = even column; scales [N, K/group] fp16; zeros
[N, K/group] uint8; w = (q - zero) * scale) is Marlin's AWQ-INT4 case (uint4 with integer zero points). marlin_pack
converts a layer once the way vLLM prepares an AWQ checkpoint: the codes into AWQ's int32 layout, then
awq_marlin_repack on the GPU; scales, zero points and bias as vLLM's marlin_permute_scales, marlin_zero_points and
marlin_permute_bias (marlin_utils.py). Parity: tests/test_marlin.py.
"""

import functools
from typing import NamedTuple, Optional, Tuple

import torch

try:
    from turing import turing_marlin as _C
    HAS_MARLIN = True
except ImportError:
    _C = None
    HAS_MARLIN = False

GROUPS = (64, 128)  # group_blocks 4 and 8: the kernels this build instantiates (csrc/marlin/kernel_configs.h)
MAX_BLOCKS_PER_SM = 4  # vLLM's MARLIN_MAX_BLOCKS_PER_SM: lock ints per SM in the workspace
SYMMETRIC_ZERO = 8  # the zero point of every group of a symmetric 4-bit grid (GPTQ's sym=true)
# vLLM marlin_utils.get_scale_perms()
SCALE_PERM = [i + 8 * j for i in range(8) for j in range(8)]
SCALE_PERM_SINGLE = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
# Nibble j of an AWQ int32 (and of Marlin's packed zero points) holds column INTERLEAVE[j] of its 8 columns
INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]


class MarlinWeights(NamedTuple):
    qweight: torch.Tensor  # int32 [K / 16, N * 2], Marlin tiles
    scales: torch.Tensor  # [K / group, N] in the activation dtype, permuted
    zeros: torch.Tensor  # int32 [K / group, N / 8], permuted and packed
    bias: Optional[torch.Tensor]  # [N] in the activation dtype, permuted
    workspace: torch.Tensor  # int32 locks, zero between calls
    n: int
    k: int


def marlin_supported(N: int, K: int, group: int, dtype: torch.dtype, device) -> bool:
    """Marlin runs y = x W^T for W [N, K] in this build: vLLM's shape rule (N % 64, K % 128; marlin_utils
    verify_marlin_supports_shape), more than one group per row (channelwise is not instantiated), sm_80+."""
    device = torch.device(device)
    return (HAS_MARLIN and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) and group in GROUPS
            and N % 64 == 0 and K % 128 == 0 and K % group == 0 and K > group
            and torch.cuda.get_device_capability(device) >= (8, 0))


def _pack8(x: torch.Tensor) -> torch.Tensor:
    """[..., 8 n] values 0..15 -> int32 [..., n] with column INTERLEAVE[j] of each 8 in bits 4j..4j+3."""
    x = x.to(torch.uint8).unflatten(-1, (-1, 8))[..., INTERLEAVE]
    return (x[..., 0::2] | (x[..., 1::2] << 4)).contiguous().view(torch.int32).squeeze(-1)  # little-endian bytes


def marlin_layout(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, dtype: torch.dtype,
                  bias: Optional[torch.Tensor] = None, symmetric: Optional[bool] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The device-independent part of marlin_pack: (AWQ int32 weight [K, N / 8], Marlin scales [K / group, N],
    Marlin zero points int32 [K / group, N / 8], Marlin bias [N] or None). symmetric (default: whether every zero
    point is 8, as quant.symmetric gives them): no zero points are passed on, and Marlin runs its kernel without
    them (b_type kU4B8, what a symmetric GPTQ checkpoint takes in vLLM), which reads and subtracts none."""
    from ..models.quant import unpack
    N, G = scales.shape
    awq = _pack8(unpack(qweight).t())  # AWQ: K rows, 8 output columns per int32
    s = scales.t().to(dtype).reshape(-1, len(SCALE_PERM))[:, SCALE_PERM].reshape(G, N).contiguous()
    every_eight = bool((zeros == SYMMETRIC_ZERO).all())
    if symmetric and not every_eight:
        raise ValueError("a matrix with zero points other than 8 cannot be packed as symmetric")
    if every_eight if symmetric is None else symmetric:
        b = None if bias is None else \
            bias.to(dtype).reshape(-1, len(SCALE_PERM_SINGLE))[:, SCALE_PERM_SINGLE].reshape(N).contiguous()
        return awq, s, torch.empty(0, dtype=torch.int32, device=qweight.device), b
    z = _pack8(zeros.t().reshape(-1, len(SCALE_PERM))[:, SCALE_PERM].reshape(G, N))  # marlin_zero_points
    b = None if bias is None else \
        bias.to(dtype).reshape(-1, len(SCALE_PERM_SINGLE))[:, SCALE_PERM_SINGLE].reshape(N).contiguous()
    return awq, s, z, b


@functools.lru_cache(maxsize=None)
def _workspace(index: int, user: str = "") -> torch.Tensor:
    """The kernel's lock array on device `index`, one per `user`: kernels that share one must run one after the
    other (two at once on the same locks never finish)."""
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    with torch.inference_mode(False):
        return torch.zeros(sms * MAX_BLOCKS_PER_SM, dtype=torch.int32, device=torch.device("cuda", index))


def marlin_pack(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, dtype: torch.dtype,
                bias: Optional[torch.Tensor] = None, symmetric: Optional[bool] = None) -> MarlinWeights:
    """A QuantLinear's buffers (on a CUDA device) -> Marlin's copy for activations of `dtype`; check
    marlin_supported first. The inputs are left as they are. symmetric: as marlin_layout."""
    N, K = qweight.shape[0], qweight.shape[1] * 2
    with torch.inference_mode(False), torch.no_grad():  # normal tensors: used in and out of inference mode
        awq, s, z, b = marlin_layout(qweight, scales, zeros, dtype, bias, symmetric)
        q = _C.awq_marlin_repack(awq, K, N)
    return MarlinWeights(q, s, z, b, _workspace(qweight.device.index or 0), N, K)


def marlin_gemm(x: torch.Tensor, w: MarlinWeights) -> torch.Tensor:
    """x [M, K] in the dtype w was packed for -> x W^T (+ bias) [M, N] in that dtype. fp32 global reduce on and
    atomic-add reduce off: vLLM's defaults."""
    x = x.contiguous()
    return _C.marlin_gemm(x, w.qweight, w.bias, w.scales, w.zeros, w.workspace, x.shape[0], w.n, w.k, False, True)
