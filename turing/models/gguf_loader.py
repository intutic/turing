"""
Native GGUF Binary File Loader & Dequantizer for Turing Engine.
Enables direct memory-mapped loading and execution of local .gguf model files
(e.g., Q4_0, Q4_K_M, Q8_0, FP16) without external conversion.
"""

import os
import math
import mmap
import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from .causal_lm import SubspaceCausalLM

__all__ = [
    "GGUFValueType",
    "GGMLType",
    "GGUFHeader",
    "GGUFTensorInfo",
    "GGUFReader",
    "GGUFDequantizer",
    "GGUFModelLoader",
    "create_test_gguf_file",
]


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


GGUF_MAGIC = 0x46554747  # 'GGUF' in Little Endian (0x47, 0x47, 0x55, 0x46)
DEFAULT_ALIGNMENT = 32


@dataclass
class GGUFHeader:
    magic: int
    version: int
    tensor_count: int
    kv_count: int


@dataclass
class GGUFTensorInfo:
    name: str
    n_dims: int
    shape: List[int]
    ggml_type: GGMLType
    offset: int


# (values per block, bytes per block) of every GGML type Turing dequantizes (ggml-common.h)
GGML_BLOCK_LAYOUT = {
    GGMLType.F32: (1, 4), GGMLType.F16: (1, 2), GGMLType.BF16: (1, 2),
    GGMLType.Q4_0: (32, 18), GGMLType.Q4_1: (32, 20), GGMLType.Q5_0: (32, 22), GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34), GGMLType.Q2_K: (256, 84), GGMLType.Q3_K: (256, 110), GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176), GGMLType.Q6_K: (256, 210), GGMLType.Q8_K: (256, 292),
}


def ggml_nbytes(ggml_type: GGMLType, num_elements: int) -> int:
    """Bytes that num_elements values of ggml_type take; raises for types Turing cannot dequantize."""
    if ggml_type not in GGML_BLOCK_LAYOUT:
        raise ValueError(f"GGML type {GGMLType(ggml_type).name} is not supported "
                         f"(supported: {', '.join(t.name for t in GGML_BLOCK_LAYOUT)})")
    per_block, block_bytes = GGML_BLOCK_LAYOUT[ggml_type]
    if num_elements % per_block:
        raise ValueError(f"{GGMLType(ggml_type).name}: {num_elements} values are not whole {per_block}-value blocks")
    return num_elements // per_block * block_bytes


def _f16(b: np.ndarray) -> np.ndarray:
    return b.view(np.float16).astype(np.float32)


def _bits(b: np.ndarray, shifts, mask: int) -> np.ndarray:
    """(nb, groups, n) bytes -> (nb, groups, len(shifts), n): each byte shifted right by each shift, masked."""
    return (b[:, :, None, :] >> np.array(shifts, dtype=np.uint8)[None, None, :, None]) & np.uint8(mask)


def _k_scale_min(s: np.ndarray):
    """The eight 6-bit scales and mins packed in the 12 scale bytes of a Q4_K / Q5_K block."""
    a, m, md = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    sc = np.concatenate([a & 63, (md & 0x0F) | ((a >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 63, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.astype(np.float32), mn.astype(np.float32)


class GGUFDequantizer:
    """
    Dequantizes GGML tensor bytes into PyTorch tensors, exactly as ggml does (bit for bit, checked against gguf-py).
    """

    @staticmethod
    def dequantize_numpy(data_bytes: bytes, ggml_type: GGMLType, num_elements: int) -> np.ndarray:
        """Pure-NumPy dequantization to a flat float32 array (the path when the native extension is absent)."""
        nbytes = ggml_nbytes(ggml_type, num_elements)
        if len(data_bytes) < nbytes:
            raise ValueError(f"GGUF tensor data is truncated: {len(data_bytes)} bytes, {nbytes} needed")
        t, n = GGMLType(ggml_type), num_elements
        if t == GGMLType.F32:
            return np.frombuffer(data_bytes, dtype=np.float32, count=n).copy()
        if t == GGMLType.F16:
            return np.frombuffer(data_bytes, dtype=np.float16, count=n).astype(np.float32)
        if t == GGMLType.BF16:
            return (np.frombuffer(data_bytes, dtype=np.uint16, count=n).astype(np.uint32) << 16).view(np.float32)

        per_block, block_bytes = GGML_BLOCK_LAYOUT[t]
        nb = n // per_block
        b = np.frombuffer(data_bytes, dtype=np.uint8, count=nb * block_bytes).reshape(nb, block_bytes)
        f32 = np.float32
        if t in (GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1):
            # values 0-15 of a block are the low nibbles, 16-31 the high ones; Q5 adds bit j of qh as bit 4 of value j
            has_min, has_high = t in (GGMLType.Q4_1, GGMLType.Q5_1), t in (GGMLType.Q5_0, GGMLType.Q5_1)
            d = _f16(b[:, 0:2])
            pos = 4 if has_min else 2
            q = _bits(b[:, None, pos + 4 * has_high:], (0, 4), 0x0F).reshape(nb, 32)
            if has_high:
                qh = b[:, pos:pos + 4].view(np.uint32)
                q = q | (((qh >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8) << 4)
            if has_min:
                return (d * q.astype(f32) + _f16(b[:, 2:4])).reshape(n)
            return (d * (q.astype(np.int8) - np.int8(16 if has_high else 8)).astype(f32)).reshape(n)
        if t == GGMLType.Q8_0:
            return (b[:, 2:].view(np.int8).astype(f32) * _f16(b[:, 0:2])).reshape(n)
        if t == GGMLType.Q8_K:
            return (b[:, 4:260].view(np.int8).astype(f32) * b[:, 0:4].view(f32)).reshape(n)
        if t == GGMLType.Q2_K:
            scales, d, dmin = b[:, 0:16], _f16(b[:, 80:82]), _f16(b[:, 82:84])
            dl = (d * (scales & 0x0F).astype(f32))[:, :, None]
            ml = (dmin * (scales >> 4).astype(f32))[:, :, None]
            q = _bits(b[:, 16:80].reshape(nb, 2, 32), (0, 2, 4, 6), 3).reshape(nb, 16, 16).astype(f32)
            return (dl * q - ml).reshape(n)
        if t == GGMLType.Q3_K:
            s, d = b[:, 96:108], _f16(b[:, 108:110])
            lo = _bits(s[:, None, 0:8], (0, 4), 0x0F).reshape(nb, 16)
            hi = _bits(s[:, None, 8:12], (0, 2, 4, 6), 3).reshape(nb, 16)
            dl = (d * ((lo | (hi << 4)).astype(np.int8) - np.int8(32)).astype(f32))[:, :, None]
            ql = _bits(b[:, 32:96].reshape(nb, 2, 32), (0, 2, 4, 6), 3).reshape(nb, 16, 16)
            qh = _bits(b[:, None, 0:32], tuple(range(8)), 1).reshape(nb, 16, 16)
            q = (ql.astype(np.int8) - ((qh ^ 1) << 2).astype(np.int8)).astype(f32)  # a clear hmask bit subtracts 4
            return (dl * q).reshape(n)
        if t in (GGMLType.Q4_K, GGMLType.Q5_K):
            d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
            sc, mn = _k_scale_min(b[:, 4:16])
            qs = b[:, 16:144] if t == GGMLType.Q4_K else b[:, 48:176]
            q = _bits(qs.reshape(nb, 4, 32), (0, 4), 0x0F).reshape(nb, 8, 32)
            if t == GGMLType.Q5_K:
                q = q | (_bits(b[:, None, 16:48], tuple(range(8)), 1).reshape(nb, 8, 32) << 4)
            return ((d * sc)[:, :, None] * q.astype(f32) - (dmin * mn)[:, :, None]).reshape(n)
        if t == GGMLType.Q6_K:
            sc, d = b[:, 192:208].view(np.int8).astype(f32), _f16(b[:, 208:210])
            ql = _bits(b[:, 0:128].reshape(nb, 2, 64), (0, 4), 0x0F).reshape(nb, 8, 32)
            qh = _bits(b[:, 128:192].reshape(nb, 2, 32), (0, 2, 4, 6), 3).reshape(nb, 8, 32)
            q = ((ql | (qh << 4)).astype(np.int8) - np.int8(32)).reshape(nb, 16, 16).astype(f32)
            return ((d * sc)[:, :, None] * q).reshape(n)
        raise AssertionError(f"no dequantizer for {t.name}")  # every GGML_BLOCK_LAYOUT type is handled above

    @staticmethod
    def dequantize(data_bytes: bytes, ggml_type: GGMLType, shape: List[int], target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        num_elements = int(np.prod(shape))
        ggml_nbytes(ggml_type, num_elements)  # unsupported types raise here, on either path
        try:
            from turing import turing_csrc  # native C++ AVX2 / NEON dequantizer
        except ImportError:
            turing_csrc = None
        if turing_csrc is not None:
            arr = turing_csrc.dequantize_gguf_simd(bytes(data_bytes), int(ggml_type), [num_elements])
        else:
            arr = GGUFDequantizer.dequantize_numpy(data_bytes, ggml_type, num_elements)
        return torch.from_numpy(arr).reshape(shape).to(target_dtype)


class GGUFReader:
    """
    Direct zero-copy reader and parser for GGUF binary files.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.file_obj = open(filepath, "rb")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        if hasattr(self.mmap_obj, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
            try:
                self.mmap_obj.madvise(mmap.MADV_WILLNEED)
            except Exception:
                pass
        
        self.header: Optional[GGUFHeader] = None
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: Dict[str, GGUFTensorInfo] = {}
        self.alignment: int = DEFAULT_ALIGNMENT
        self.tensor_data_offset: int = 0
        
        self._parse()

    def _read_str(self, pos: int) -> Tuple[str, int]:
        str_len = struct.unpack_from("<Q", self.mmap_obj, pos)[0]
        pos += 8
        str_bytes = self.mmap_obj[pos : pos + str_len]
        pos += str_len
        return str_bytes.decode("utf-8", errors="replace"), pos

    def _read_value(self, vtype: GGUFValueType, pos: int) -> Tuple[Any, int]:
        if vtype == GGUFValueType.UINT8:
            val = struct.unpack_from("<B", self.mmap_obj, pos)[0]
            return val, pos + 1
        elif vtype == GGUFValueType.INT8:
            val = struct.unpack_from("<b", self.mmap_obj, pos)[0]
            return val, pos + 1
        elif vtype == GGUFValueType.UINT16:
            val = struct.unpack_from("<H", self.mmap_obj, pos)[0]
            return val, pos + 2
        elif vtype == GGUFValueType.INT16:
            val = struct.unpack_from("<h", self.mmap_obj, pos)[0]
            return val, pos + 2
        elif vtype == GGUFValueType.UINT32:
            val = struct.unpack_from("<I", self.mmap_obj, pos)[0]
            return val, pos + 4
        elif vtype == GGUFValueType.INT32:
            val = struct.unpack_from("<i", self.mmap_obj, pos)[0]
            return val, pos + 4
        elif vtype == GGUFValueType.FLOAT32:
            val = struct.unpack_from("<f", self.mmap_obj, pos)[0]
            return val, pos + 4
        elif vtype == GGUFValueType.BOOL:
            val = bool(struct.unpack_from("<?", self.mmap_obj, pos)[0])
            return val, pos + 1
        elif vtype == GGUFValueType.STRING:
            return self._read_str(pos)
        elif vtype == GGUFValueType.UINT64:
            val = struct.unpack_from("<Q", self.mmap_obj, pos)[0]
            return val, pos + 8
        elif vtype == GGUFValueType.INT64:
            val = struct.unpack_from("<q", self.mmap_obj, pos)[0]
            return val, pos + 8
        elif vtype == GGUFValueType.FLOAT64:
            val = struct.unpack_from("<d", self.mmap_obj, pos)[0]
            return val, pos + 8
        elif vtype == GGUFValueType.ARRAY:
            elem_type_raw = struct.unpack_from("<I", self.mmap_obj, pos)[0]
            elem_type = GGUFValueType(elem_type_raw)
            pos += 4
            array_len = struct.unpack_from("<Q", self.mmap_obj, pos)[0]
            pos += 8
            
            elements = []
            for _ in range(array_len):
                elem, pos = self._read_value(elem_type, pos)
                elements.append(elem)
            return elements, pos
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _parse(self):
        if self.file_size < 24:
            raise ValueError(f"Invalid GGUF magic bytes: file size ({self.file_size} bytes) is smaller than 24-byte header")

        pos = 0
        magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", self.mmap_obj, pos)
        pos += 24
        
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic bytes: {hex(magic)}, expected {hex(GGUF_MAGIC)}")
        
        self.header = GGUFHeader(magic=magic, version=version, tensor_count=tensor_count, kv_count=kv_count)
        
        # 1. Parse Metadata Key-Value pairs
        for _ in range(kv_count):
            key, pos = self._read_str(pos)
            vtype_raw = struct.unpack_from("<I", self.mmap_obj, pos)[0]
            vtype = GGUFValueType(vtype_raw)
            pos += 4
            val, pos = self._read_value(vtype, pos)
            self.metadata[key] = val
            
        if "general.alignment" in self.metadata:
            self.alignment = int(self.metadata["general.alignment"])

        # 2. Parse Tensor Info entries
        for _ in range(tensor_count):
            name, pos = self._read_str(pos)
            n_dims = struct.unpack_from("<I", self.mmap_obj, pos)[0]
            pos += 4
            
            dims = []
            for _ in range(n_dims):
                dim = struct.unpack_from("<Q", self.mmap_obj, pos)[0]
                dims.append(dim)
                pos += 8
                
            # PyTorch shape is reverse of GGML column-major dimensions
            torch_shape = list(reversed(dims)) if dims else [1]
            
            type_raw = struct.unpack_from("<I", self.mmap_obj, pos)[0]
            ggml_type = GGMLType(type_raw)
            pos += 4
            
            offset = struct.unpack_from("<Q", self.mmap_obj, pos)[0]
            pos += 8
            
            self.tensor_infos[name] = GGUFTensorInfo(
                name=name,
                n_dims=n_dims,
                shape=torch_shape,
                ggml_type=ggml_type,
                offset=offset
            )

        # 3. Calculate aligned start of Tensor Data section
        remainder = pos % self.alignment
        if remainder != 0:
            self.tensor_data_offset = pos + (self.alignment - remainder)
        else:
            self.tensor_data_offset = pos

    def get_tensor_tensor(self, name: str, target_dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
        if name not in self.tensor_infos:
            raise KeyError(f"Tensor '{name}' not found in GGUF file {self.filepath}")
            
        info = self.tensor_infos[name]
        data_start = self.tensor_data_offset + info.offset
        byte_size = ggml_nbytes(info.ggml_type, int(np.prod(info.shape)))
        if data_start + byte_size > self.file_size:
            raise ValueError(f"GGUF file {self.filepath} is truncated: tensor '{name}' needs bytes "
                             f"{data_start}..{data_start + byte_size} of a {self.file_size}-byte file")
        data_bytes = self.mmap_obj[data_start : data_start + byte_size]
        tensor = GGUFDequantizer.dequantize(data_bytes, info.ggml_type, info.shape, target_dtype=target_dtype)
        return tensor.to(device)

    def close(self):
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            self.mmap_obj.close()
        if hasattr(self, "file_obj") and self.file_obj is not None:
            self.file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class GGUFModelLoader:
    """
    Loads a complete local .gguf file into a Turing SubspaceCausalLM model with tokenizer.
    """

    # GGUF architectures SubspaceCausalLM computes exactly (logit parity with HF is tested), and whether llama.cpp's
    # convert_hf_to_gguf permuted their q/k rows for interleaved RoPE (Turing's RoPE rotates halves, as HF does).
    # Mistral checkpoints convert to "llama". Others (gemma, phi3, deepseek2, ...) need model changes first.
    ARCHS = {"llama": True, "qwen2": False, "qwen3": False}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = GGUFReader(filepath)

    def _rope_scaling(self, arch: str, head_dim: int, rope_theta: float) -> Optional[Tuple[float, float, float, int]]:
        """Llama-3 RoPE scaling as ModelConfig.rope_scaling, from the rope_freqs.weight tensor llama.cpp's converter
        writes for it; None for plain RoPE. RoPE scaling types the model cannot compute yet raise."""
        meta = self.reader.metadata
        kind = meta.get(f"{arch}.rope.scaling.type", "none")
        if kind != "none":
            raise ValueError(f"RoPE scaling type '{kind}' ({arch}.rope.scaling.type in {self.filepath}) is not supported yet")
        if "rope_freqs.weight" not in self.reader.tensor_infos:
            return None
        factors = self.reader.get_tensor_tensor("rope_freqs.weight", torch.float32).numpy()
        # The converter writes only the per-frequency factors; low/high_freq_factor 1/4 and an 8192-token original
        # context are its defaults (and every Llama 3.x's). Anything else fails the check below rather than load wrong.
        orig = int(meta.get(f"{arch}.rope.scaling.original_context_length", 8192))
        scaling = (float(factors.max()), 1.0, 4.0, orig)
        dim = int(meta.get(f"{arch}.rope.dimension_count", head_dim))
        freqs = 1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        wavelen = 2 * math.pi / freqs
        smooth = (orig / wavelen - 1.0) / (4.0 - 1.0)
        expected = np.where(wavelen < orig / 4.0, 1.0,
                            np.where(wavelen > orig, scaling[0], 1 / ((1 - smooth) / scaling[0] + smooth)))
        if factors.shape != expected.shape or not np.allclose(factors, expected, rtol=1e-4, atol=1e-6):
            raise ValueError(f"rope_freqs.weight in {self.filepath} is not llama3 RoPE scaling with factor "
                             f"{scaling[0]}, low/high_freq_factor 1/4 and original context {orig}: not supported")
        return None if scaling[0] == 1.0 else scaling

    def extract_config(self, sparsity_ratio: float = 0.0) -> ModelConfig:
        meta = self.reader.metadata
        arch = meta.get("general.architecture", "llama")
        if arch not in self.ARCHS:
            raise ValueError(f"GGUF architecture '{arch}' is not supported (supported: {', '.join(self.ARCHS)})")
        name = meta.get("general.name", os.path.basename(self.filepath).replace(".gguf", ""))
        
        hidden_dim = int(meta.get(f"{arch}.embedding_length", 2048))
        num_layers = int(meta.get(f"{arch}.block_count", 16))
        num_heads = int(meta.get(f"{arch}.attention.head_count", 16))
        num_kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", num_heads))
        head_dim = int(meta.get(f"{arch}.attention.key_length", hidden_dim // num_heads))
        ffn_dim = int(meta.get(f"{arch}.feed_forward_length", hidden_dim * 4))
        
        vocab_size = 32000
        if "tokenizer.ggml.tokens" in meta:
            vocab_size = len(meta["tokenizer.ggml.tokens"])
        elif f"{arch}.vocab_size" in meta:
            vocab_size = int(meta[f"{arch}.vocab_size"])
            
        max_pos = int(meta.get(f"{arch}.context_length", 4096))
        rope_theta = float(meta.get(f"{arch}.rope.freq_base", 10000.0))
        
        tile_size = 64 if ffn_dim <= 4096 else 256
        total_tiles = ffn_dim // tile_size
        active_tiles = max(1, int(total_tiles * (1.0 - sparsity_ratio)))
        
        return ModelConfig(
            name=name,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            tile_size=tile_size,
            active_tiles=active_tiles,
            rank_sub=64 if hidden_dim >= 768 else 32,
            max_position_embeddings=max_pos,
            rope_theta=rope_theta,
            rms_norm_eps=float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
            rope_scaling=self._rope_scaling(arch, head_dim, rope_theta),
            attention_bias="blk.0.attn_q.bias" in self.reader.tensor_infos,     # Qwen2
            qk_norm="blk.0.attn_q_norm.weight" in self.reader.tensor_infos,     # Qwen3
        )

    def extract_tokenizer(self) -> Any:
        from .gguf_tokenizer import GGUFTokenizer
        return GGUFTokenizer(self.reader.metadata)

    @staticmethod
    def tensor_map(config: ModelConfig) -> Dict[str, str]:
        """SubspaceCausalLM parameter name -> llama.cpp GGUF tensor name, for every weight a checkpoint holds."""
        m = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
        per_layer = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
                     "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
                     "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
                     "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
                     "mlp.down_proj.weight": "ffn_down.weight"}
        if config.attention_bias:
            per_layer.update({f"self_attn.{p}_proj.bias": f"attn_{p}.bias" for p in "qkv"})
        if config.qk_norm:
            per_layer.update({f"self_attn.{p}_norm.weight": f"attn_{p}_norm.weight" for p in "qk"})
        for l in range(config.num_layers):
            m.update({f"layers.{l}.{k}": f"blk.{l}.{v}" for k, v in per_layer.items()})
        return m

    def load(
        self,
        sparsity_ratio: float = 0.0,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[SubspaceCausalLM, Any]:
        """Every GGUF tensor goes into the model, dequantized one at a time straight into the model's dtype (fp16 on
        CUDA, fp32 elsewhere, as the HF loader does). A file missing a weight, or holding one the model has no place
        for, raises instead of leaving weights at random init."""
        config = self.extract_config(sparsity_ratio=sparsity_ratio)
        if dtype is None:
            dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        names = self.reader.tensor_infos
        mapping = self.tensor_map(config)
        if "output.weight" not in names:
            mapping["lm_head.weight"] = "token_embd.weight"  # tied embeddings
        missing = sorted(g for g in mapping.values() if g not in names)
        unexpected = sorted(set(names) - set(mapping.values()) - {"rope_freqs.weight"})
        if missing or unexpected:
            raise ValueError(f"GGUF file {self.filepath} does not match a {config.num_layers}-layer "
                             f"{self.reader.metadata.get('general.architecture', 'llama')} model: "
                             f"missing tensors {missing}, unexpected tensors {unexpected}")

        # Built on the device in the target dtype: never a whole-model fp32 copy (a 70B model is 280 GB in fp32).
        # shortcut: the default dtype is process-wide while the model is built; build on "meta" if loads go concurrent.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = SubspaceCausalLM(config)
        finally:
            torch.set_default_dtype(default_dtype)

        permuted = self.ARCHS[self.reader.metadata.get("general.architecture", "llama")]
        params = dict(model.named_parameters())
        with torch.no_grad():
            for pname, gname in mapping.items():
                t = self.reader.get_tensor_tensor(gname, torch.float32)
                if permuted and gname.endswith(("attn_q.weight", "attn_q.bias")):
                    t = unpermute_qk(t, config.num_heads)
                elif permuted and gname.endswith(("attn_k.weight", "attn_k.bias")):
                    t = unpermute_qk(t, config.num_kv_heads)
                p = params[pname]
                if t.shape != p.shape:
                    raise ValueError(f"GGUF tensor '{gname}' has shape {tuple(t.shape)}; {pname} is {tuple(p.shape)}")
                p.copy_(t)
                del t
        tokenizer = self.extract_tokenizer()
        return model, tokenizer


def unpermute_qk(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """Undo convert_hf_to_gguf's q/k row permutation for llama (made for llama.cpp's interleaved RoPE):
    the file holds w.reshape(n_head, 2, head_dim // 2, ...).swapaxes(1, 2)."""
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).transpose(1, 2).reshape(w.shape)


def create_test_gguf_file(
    filepath: str,
    architecture: str = "llama",
    hidden_dim: int = 64,
    ffn_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    vocab_size: int = 128,
    max_position_embeddings: int = 256,
    quant_type: GGMLType = GGMLType.F16
) -> str:
    """
    Synthesizes a minimal valid binary GGUF file for fast local testing and CI verification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    metadata_entries = [
        ("general.architecture", GGUFValueType.STRING, architecture),
        ("general.name", GGUFValueType.STRING, f"test-tiny-{architecture}"),
        (f"{architecture}.embedding_length", GGUFValueType.UINT32, hidden_dim),
        (f"{architecture}.feed_forward_length", GGUFValueType.UINT32, ffn_dim),
        (f"{architecture}.block_count", GGUFValueType.UINT32, num_layers),
        (f"{architecture}.attention.head_count", GGUFValueType.UINT32, num_heads),
        (f"{architecture}.attention.head_count_kv", GGUFValueType.UINT32, num_kv_heads),
        (f"{architecture}.context_length", GGUFValueType.UINT32, max_position_embeddings),
        (f"{architecture}.rope.freq_base", GGUFValueType.FLOAT32, 10000.0),
        ("general.alignment", GGUFValueType.UINT32, DEFAULT_ALIGNMENT),
        ("tokenizer.ggml.tokens", GGUFValueType.ARRAY, (GGUFValueType.STRING, [f"<tok_{i}>" for i in range(vocab_size)])),
    ]

    tensors = []
    # 1. Embeddings & Norms
    tensors.append(("token_embd.weight", [vocab_size, hidden_dim], np.random.randn(vocab_size, hidden_dim).astype(np.float32)))
    tensors.append(("output_norm.weight", [hidden_dim], np.ones(hidden_dim, dtype=np.float32)))
    tensors.append(("output.weight", [vocab_size, hidden_dim], np.random.randn(vocab_size, hidden_dim).astype(np.float32)))
    
    # 2. Layers
    for l in range(num_layers):
        p = f"blk.{l}."
        tensors.append((f"{p}attn_norm.weight", [hidden_dim], np.ones(hidden_dim, dtype=np.float32)))
        tensors.append((f"{p}ffn_norm.weight", [hidden_dim], np.ones(hidden_dim, dtype=np.float32)))
        tensors.append((f"{p}attn_q.weight", [hidden_dim, hidden_dim], np.random.randn(hidden_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}attn_k.weight", [hidden_dim, hidden_dim], np.random.randn(hidden_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}attn_v.weight", [hidden_dim, hidden_dim], np.random.randn(hidden_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}attn_output.weight", [hidden_dim, hidden_dim], np.random.randn(hidden_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}ffn_gate.weight", [ffn_dim, hidden_dim], np.random.randn(ffn_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}ffn_up.weight", [ffn_dim, hidden_dim], np.random.randn(ffn_dim, hidden_dim).astype(np.float32)))
        tensors.append((f"{p}ffn_down.weight", [hidden_dim, ffn_dim], np.random.randn(hidden_dim, ffn_dim).astype(np.float32)))

    with open(filepath, "wb") as f:
        # Header: magic(4), version(4), tensor_count(8), kv_count(8)
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(metadata_entries)))
        
        # Write metadata KV
        for k, vtype, val in metadata_entries:
            k_bytes = k.encode("utf-8")
            f.write(struct.pack("<Q", len(k_bytes)))
            f.write(k_bytes)
            f.write(struct.pack("<I", int(vtype)))
            
            if vtype == GGUFValueType.STRING:
                s_bytes = val.encode("utf-8")
                f.write(struct.pack("<Q", len(s_bytes)))
                f.write(s_bytes)
            elif vtype == GGUFValueType.UINT32:
                f.write(struct.pack("<I", int(val)))
            elif vtype == GGUFValueType.FLOAT32:
                f.write(struct.pack("<f", float(val)))
            elif vtype == GGUFValueType.ARRAY:
                elem_type, elem_list = val
                f.write(struct.pack("<IQ", int(elem_type), len(elem_list)))
                for elem in elem_list:
                    if elem_type == GGUFValueType.STRING:
                        e_bytes = elem.encode("utf-8")
                        f.write(struct.pack("<Q", len(e_bytes)))
                        f.write(e_bytes)
                    elif elem_type == GGUFValueType.FLOAT32:
                        f.write(struct.pack("<f", float(elem)))
                    elif elem_type == GGUFValueType.INT32:
                        f.write(struct.pack("<i", int(elem)))

        # Prepare tensor data buffer
        tensor_data_bytes = bytearray()
        tensor_info_records = []
        
        for name, shape, tensor_arr in tensors:
            offset = len(tensor_data_bytes)
            # Encode tensor data
            if quant_type == GGMLType.F16:
                encoded = tensor_arr.astype(np.float16).tobytes()
            elif quant_type == GGMLType.Q8_0:
                # 32-element blocks
                flat = tensor_arr.flatten().astype(np.float32)
                blocks = len(flat) // 32
                b_list = []
                for bi in range(blocks):
                    chunk = flat[bi * 32 : (bi + 1) * 32]
                    amax = np.max(np.abs(chunk)) or 1e-4
                    delta = amax / 127.0
                    qs = np.clip(np.round(chunk / delta), -128, 127).astype(np.int8)
                    b_list.append(np.float16(delta).tobytes() + qs.tobytes())
                encoded = b"".join(b_list)
            else:
                encoded = tensor_arr.astype(np.float32).tobytes()
                
            tensor_data_bytes.extend(encoded)
            # Align next tensor
            rem = len(tensor_data_bytes) % DEFAULT_ALIGNMENT
            if rem != 0:
                tensor_data_bytes.extend(b"\x00" * (DEFAULT_ALIGNMENT - rem))
                
            tensor_info_records.append((name, shape, quant_type, offset))

        # Write tensor info records
        for name, shape, t_type, offset in tensor_info_records:
            n_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(n_bytes)))
            f.write(n_bytes)
            # GGML dims are reverse of shape
            ggml_dims = list(reversed(shape))
            f.write(struct.pack("<I", len(ggml_dims)))
            for d in ggml_dims:
                f.write(struct.pack("<Q", int(d)))
            f.write(struct.pack("<IQ", int(t_type), offset))

        # Pad to alignment before data section
        curr_pos = f.tell()
        rem = curr_pos % DEFAULT_ALIGNMENT
        if rem != 0:
            f.write(b"\x00" * (DEFAULT_ALIGNMENT - rem))
            
        # Write tensor binary data
        f.write(tensor_data_bytes)

    return filepath
