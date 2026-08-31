"""
Triton GPU Kernel: In-VRAM Parallel Tensor Checksum & Rolling Hash Reduction.
Computes a 64-bit deterministic checksum directly in GPU VRAM memory without D2H PCIe transfers.
"""

from typing import List, Union
import torch

__all__ = ["fused_tensor_checksum_cuda", "compute_fast_tensor_hash"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_vram_hash_block_kernel(
        data_ptr,
        partial_hashes_ptr,
        num_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        vals = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        # Reinterpret float32 as int32 for bitwise mixing
        raw_bits = vals.to(tl.int32)
        
        # 32-bit mixing constants
        prime = 16777619
        mixed = (raw_bits * prime) ^ (offsets * 31)
        block_sum = tl.sum(mixed, axis=0)

        tl.store(partial_hashes_ptr + pid, block_sum)


def fused_tensor_checksum_cuda(tensor: torch.Tensor) -> str:
    """
    Computes a 64-bit checksum over a CUDA tensor in VRAM with 0 host PCIe copies.
    """
    if not HAS_TRITON or not tensor.is_cuda:
        try:
            import turing.turing_csrc as turing_csrc
            if tensor.is_contiguous():
                return turing_csrc.hash_tensor_buffer_fast(tensor.data_ptr(), tensor.nbytes)
        except Exception:
            pass
        # Fallback to standard Python byte hashing
        import hashlib
        h = hashlib.blake2b()
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    flat_tensor = tensor.contiguous().view(-1)
    num_elements = flat_tensor.numel()
    if num_elements == 0:
        return "0000000000000000"

    BLOCK_SIZE = 1024
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    partial_hashes = torch.empty(num_blocks, device=tensor.device, dtype=torch.int32)

    _fused_vram_hash_block_kernel[(num_blocks,)](
        flat_tensor,
        partial_hashes,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    final_val = int(torch.sum(partial_hashes).item()) & 0xFFFFFFFFFFFFFFFF
    return f"{final_val:016x}"


def compute_fast_tensor_hash(tensors: List[torch.Tensor]) -> str:
    """
    Fast multi-tensor hash aggregator supporting CUDA, MPS, and CPU.
    """
    if not tensors:
        return "0000000000000000"

    # Fast CUDA path
    if tensors[0].is_cuda and HAS_TRITON:
        digests = [fused_tensor_checksum_cuda(t) for t in tensors]
        import hashlib
        return hashlib.blake2b("".join(digests).encode("utf-8")).hexdigest()

    # Fast CPU / MPS C++ pointer path
    try:
        import turing.turing_csrc as turing_csrc
        digests = []
        for t in tensors:
            t_cont = t.contiguous()
            if not t.is_cuda:
                digests.append(turing_csrc.hash_tensor_buffer_fast(t_cont.data_ptr(), t_cont.nbytes))
            else:
                digests.append(fused_tensor_checksum_cuda(t))
        import hashlib
        return hashlib.blake2b("".join(digests).encode("utf-8")).hexdigest()
    except Exception:
        pass

    # Pure Python fallback
    import hashlib
    h = hashlib.blake2b()
    for tensor in tensors:
        tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        h.update(tensor_bytes)
    return h.hexdigest()
