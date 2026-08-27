"""
SVD-Compressed Network KV Wire Format for Cross-Pod Transfer.
Compresses KV cache blocks using Rank-64 Subspace Projection and Symmetric INT8 Quantization,
yielding ~75% reduction in network payload size for P/D disaggregated and distributed serving.
"""

import struct
from typing import Tuple, List, Optional
import torch

from ..core.hybrid_mesh import TensorSerializer


class SVDNetworkKVWireCodec:
    """
    Encodes and decodes KV cache blocks into SVD-compressed INT8 wire format
    for high-speed cross-pod network transfer over TCP / RDMA side-channels.

    Binary Wire Layout:
    [Magic: uint32 (0x5356444B = 'SVDK')]
    [Version: uint8 (1)]
    [NumTokens: uint32]
    [NumHeads: uint32]
    [Rank: uint32]
    [TokenIDs: uint32 * NumTokens]
    [Serialized k_sub_int8 (TensorSerializer)]
    [Serialized k_scale (TensorSerializer)]
    [Serialized v_sub_int8 (TensorSerializer)]
    [Serialized v_scale (TensorSerializer)]
    """

    MAGIC = 0x5356444B  # 'SVDK'
    VERSION = 1

    @classmethod
    def encode(
        cls,
        k_tensor: torch.Tensor,  # [SeqLen, Heads, HeadDim]
        v_tensor: torch.Tensor,  # [SeqLen, Heads, HeadDim]
        u_proj: torch.Tensor,    # [HeadDim, Rank]
        token_ids: Optional[List[int]] = None,
    ) -> bytes:
        """
        Projects KV tensors into SVD subspace, quantizes to INT8, and serializes to bytes.
        """
        seq_len, num_heads, _ = k_tensor.shape
        rank = u_proj.shape[1]
        toks = token_ids if token_ids is not None else [0] * seq_len

        # 1. Project into SVD subspace
        k_sub = torch.matmul(k_tensor, u_proj)  # [SeqLen, Heads, Rank]
        v_sub = torch.matmul(v_tensor, u_proj)

        # 2. Symmetric INT8 Quantization
        k_scale = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        v_scale = torch.amax(torch.abs(v_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0

        k_int8 = torch.clamp(torch.round(k_sub / k_scale), -128, 127).to(torch.int8)
        v_int8 = torch.clamp(torch.round(v_sub / v_scale), -128, 127).to(torch.int8)

        # 3. Serialize tensors
        k_int8_bytes = TensorSerializer.serialize(k_int8.to(torch.float32), compress_int8=True)
        k_scale_bytes = TensorSerializer.serialize(k_scale, compress_int8=False)
        v_int8_bytes = TensorSerializer.serialize(v_int8.to(torch.float32), compress_int8=True)
        v_scale_bytes = TensorSerializer.serialize(v_scale, compress_int8=False)

        # 4. Pack header + tokens + tensor payloads
        header = struct.pack(
            f"<IBIII{len(toks)}I",
            cls.MAGIC,
            cls.VERSION,
            seq_len,
            num_heads,
            rank,
            *toks,
        )

        return (
            header
            + struct.pack("<I", len(k_int8_bytes)) + k_int8_bytes
            + struct.pack("<I", len(k_scale_bytes)) + k_scale_bytes
            + struct.pack("<I", len(v_int8_bytes)) + v_int8_bytes
            + struct.pack("<I", len(v_scale_bytes)) + v_scale_bytes
        )

    @classmethod
    def decode(
        cls,
        payload: bytes,
        u_proj: torch.Tensor,    # [HeadDim, Rank]
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Deserializes SVD-compressed wire payload and reconstructs full-rank FP16 KV states.
        Returns: (k_reconstructed, v_reconstructed, token_ids)
        """
        magic, version, seq_len, num_heads, rank = struct.unpack_from("<IBIII", payload, 0)
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid SVDK magic: 0x{magic:08X}, expected 0x{cls.MAGIC:08X}")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported SVDK version: {version}")

        offset = struct.calcsize("<IBIII")
        toks = list(struct.unpack_from(f"<{seq_len}I", payload, offset))
        offset += struct.calcsize(f"<{seq_len}I")

        # Unpack serialized tensor payloads
        def _read_chunk(off: int) -> Tuple[bytes, int]:
            chunk_len = struct.unpack_from("<I", payload, off)[0]
            off += 4
            chunk_data = payload[off : off + chunk_len]
            return chunk_data, off + chunk_len

        k_int8_bytes, offset = _read_chunk(offset)
        k_scale_bytes, offset = _read_chunk(offset)
        v_int8_bytes, offset = _read_chunk(offset)
        v_scale_bytes, offset = _read_chunk(offset)

        k_int8_t = TensorSerializer.deserialize(k_int8_bytes, device=device)
        k_scale_t = TensorSerializer.deserialize(k_scale_bytes, device=device)
        v_int8_t = TensorSerializer.deserialize(v_int8_bytes, device=device)
        v_scale_t = TensorSerializer.deserialize(v_scale_bytes, device=device)

        # Dequantize & reconstruct via u_proj.T
        u_proj_dev = u_proj.to(device)
        k_sub_fp = k_int8_t * k_scale_t
        v_sub_fp = v_int8_t * v_scale_t

        k_recon = torch.matmul(k_sub_fp, u_proj_dev.t())
        v_recon = torch.matmul(v_sub_fp, u_proj_dev.t())

        return k_recon, v_recon, toks


def serialize_kv_block_svd(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    u_proj: torch.Tensor,
    token_ids: Optional[List[int]] = None,
) -> bytes:
    """Convenience functional wrapper for SVDNetworkKVWireCodec.encode."""
    return SVDNetworkKVWireCodec.encode(k_tensor, v_tensor, u_proj, token_ids)


def deserialize_kv_block_svd(
    payload: bytes,
    u_proj: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Convenience functional wrapper for SVDNetworkKVWireCodec.decode."""
    return SVDNetworkKVWireCodec.decode(payload, u_proj, device)
