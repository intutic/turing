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
from typing import Callable, Dict, List, Tuple, Any, Optional, Union
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
    # read (so the metadata of such files loads, as turing serve --backend libllama needs), not dequantized
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42
    F8_E4M3 = 43


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

    def __init__(self, filepath: str, willneed: bool = True):
        """willneed: ask the OS to read the whole file ahead (the weights are loaded next); False when only the
        metadata is read (the libllama backend), which must not page in a model larger than RAM."""
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.file_obj = open(filepath, "rb")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        if willneed and hasattr(self.mmap_obj, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
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
    Loads a complete local .gguf file into a Turing SubspaceCausalLM model with tokenizer. Tensor names and metadata
    follow llama.cpp (convert_hf_to_gguf.py writes them, src/models/<arch>.cpp reads them); tests/test_gguf_parity.py
    checks every architecture against HuggingFace and scripts/eval_gguf_parity.py checks real files against llama.cpp.
    """

    # GGUF architecture -> the HF model_type SubspaceCausalLM reproduces for it. Mistral and Mixtral convert to "llama".
    ARCHS = {"llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "qwen2moe": "qwen2_moe", "qwen3moe": "qwen3_moe",
             "olmoe": "olmoe", "gemma": "gemma", "gemma2": "gemma2", "gemma3": "gemma3_text", "phi2": "phi",
             "phi3": "phi3", "gpt2": "gpt2", "gptneox": "gpt_neox"}
    # convert_hf_to_gguf permutes q/k rows for llama.cpp's interleaved RoPE (LLAMA_ROPE_TYPE_NORM) only for these;
    # the other archs above use NEOX RoPE, which rotates halves as HF and Turing do
    PERMUTED = {"llama"}
    # whether llama.cpp renormalises the top-k expert weights (build_moe_ffn's norm_w, fixed per arch)
    NORM_TOPK = {"llama": True, "qwen2moe": False, "qwen3moe": True, "olmoe": False}
    LAYER_NORM = {"phi2", "gpt2", "gptneox"}
    GEMMA = {"gemma", "gemma2", "gemma3"}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = GGUFReader(filepath)

    @property
    def arch(self) -> str:
        arch = self.reader.metadata.get("general.architecture", "llama")
        if arch not in self.ARCHS:
            raise ValueError(f"GGUF architecture '{arch}' is not supported (supported: {', '.join(self.ARCHS)})")
        return arch

    def _meta(self, key: str, default=None, uniform: bool = False):
        """{arch}.{key} from the metadata; uniform=True accepts a per-layer array only when every layer agrees."""
        v = self.reader.metadata.get(f"{self.arch}.{key}", default)
        if uniform and isinstance(v, list):
            if len(set(v)) != 1:
                raise ValueError(f"per-layer {self.arch}.{key} = {v} in {self.filepath} is not supported")
            v = v[0]
        return v

    def _rope_scaling(self, arch: str, head_dim: int, rope_theta: float, max_pos: int = 2048) -> Optional[dict]:
        """RoPE scaling as ModelConfig.rope_scaling (an HF rope-parameters dict), None for plain RoPE:
        - longrope from rope_factors_long/short.weight (Phi-3), with rope.scaling.attn_factor;
        - llama3 from rope_freqs.weight (convert_hf_to_gguf writes only the per-frequency factors);
        - linear and yarn from rope.scaling.type / factor / original_context_length. llama.cpp scales YaRN's cos/sin by
          rope.scaling.attn_factor times HF's default mscale (or the mscale ratio with yarn_log_multiplier);
          yarn_attn_factor and yarn_beta_fast/slow are HF's attention_factor and beta_fast/slow as the converter copied
          them (llama.cpp ignores those three for these archs, so it agrees only at HF's defaults)."""
        meta, names = self.reader.metadata, self.reader.tensor_infos

        def g(key, default=None):
            return meta.get(f"{arch}.rope.{key}", default)
        kind = g("scaling.type")
        factor = float(g("scaling.factor", g("scale_linear", 0.0)) or 0.0)
        orig = int(g("scaling.original_context_length", max_pos))
        attn = g("scaling.attn_factor")
        if "rope_factors_long.weight" in names:
            if kind not in (None, "longrope"):
                raise ValueError(f"{self.filepath} has LongRoPE factors and RoPE scaling type '{kind}'")
            long, short = (self.reader.get_tensor_tensor(f"rope_factors_{s}.weight", torch.float32).tolist()
                           for s in ("long", "short"))
            p = {"rope_type": "longrope", "long_factor": long, "short_factor": short,
                 "original_max_position_embeddings": orig}
            if attn is not None:
                p["attention_factor"] = float(attn)
            return p
        if "rope_freqs.weight" in names:
            if kind not in (None, "none", "linear") or factor not in (0.0, 1.0):
                raise ValueError(f"{self.filepath} has rope_freqs.weight and RoPE scaling '{kind}' x{factor}")
            return self._llama3_scaling(arch, head_dim, rope_theta)
        kind = kind or "linear"  # llama.cpp's default type; with no factor it does not scale
        if kind == "none" or factor in (0.0, 1.0):
            return None
        if kind == "linear":
            return {"rope_type": "linear", "factor": factor}
        if kind == "yarn":
            if float(g("scaling.yarn_ext_factor", 1.0)) != 1.0:
                raise ValueError(f"YaRN ext_factor != 1 ({arch}.rope.scaling.yarn_ext_factor in {self.filepath}) "
                                 "is not supported")

            def mscale(m):
                return 0.1 * m * math.log(factor) + 1.0 if factor > 1 else 1.0
            log_mul = float(g("scaling.yarn_log_multiplier", 0.0))
            att = (mscale(1.0) / mscale(log_mul) if log_mul else mscale(1.0)) * float(1.0 if attn is None else attn)
            p = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": orig,
                 "attention_factor": float(g("scaling.yarn_attn_factor", att))}
            for key, hf in (("yarn_beta_fast", "beta_fast"), ("yarn_beta_slow", "beta_slow"), ("yarn_truncate", "truncate")):
                if g(f"scaling.{key}") is not None:
                    p[hf] = g(f"scaling.{key}")
            return p
        raise ValueError(f"RoPE scaling type '{kind}' ({arch}.rope.scaling.type in {self.filepath}) is not supported")

    def _llama3_scaling(self, arch: str, head_dim: int, rope_theta: float) -> Optional[dict]:
        meta = self.reader.metadata
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
        if scaling[0] == 1.0:
            return None
        return {"rope_type": "llama3", "factor": scaling[0], "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": orig}

    def extract_config(self, sparsity_ratio: float = 0.0) -> ModelConfig:
        meta, names, arch, g = self.reader.metadata, self.reader.tensor_infos, self.arch, self._meta
        name = meta.get("general.name", os.path.basename(self.filepath).replace(".gguf", ""))
        hidden_dim = int(g("embedding_length"))
        num_layers = int(g("block_count"))
        num_heads = int(g("attention.head_count", uniform=True))
        num_kv_heads = int(g("attention.head_count_kv", num_heads, uniform=True))
        head_dim = int(g("attention.key_length", hidden_dim // num_heads))
        if int(g("attention.value_length", head_dim)) != head_dim:
            raise ValueError(f"{self.filepath}: key and value head sizes differ, which is not supported")
        ffn_dim = int(g("feed_forward_length", hidden_dim * 4, uniform=True))
        vocab_size = names["token_embd.weight"].shape[0] if "token_embd.weight" in names else \
            len(meta.get("tokenizer.ggml.tokens", [])) or 32000
        max_pos = int(g("context_length", 4096))
        rope_theta = float(g("rope.freq_base", 10000.0))
        rope_dim = int(g("rope.dimension_count", head_dim))
        layer_norm = arch in self.LAYER_NORM

        tile_size = 64 if ffn_dim <= 4096 else 256
        total_tiles = ffn_dim // tile_size
        kw = dict(
            arch=self.ARCHS[arch], hidden_dim=hidden_dim, ffn_dim=ffn_dim, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim, num_layers=num_layers, vocab_size=vocab_size,
            max_position_embeddings=max_pos, rope_theta=rope_theta,
            rms_norm_eps=float(g("attention.layer_norm_epsilon" if layer_norm else "attention.layer_norm_rms_epsilon",
                                 1e-5 if layer_norm else 1e-6)),
            rope_scaling=None if arch == "gpt2" else self._rope_scaling(arch, rope_dim, rope_theta, max_pos),
            rotary_dim=rope_dim if rope_dim != head_dim else None,
            attention_bias="blk.0.attn_q.bias" in names or "blk.0.attn_qkv.bias" in names,
        )
        q_norm = names.get("blk.0.attn_q_norm.weight")
        if q_norm is not None:  # per head (Qwen3, Gemma 3) or over the whole projection (OLMoE)
            kw["qk_norm" if int(np.prod(q_norm.shape)) == head_dim else "qk_norm_full"] = True
        if g("attention.clamp_kqv"):
            kw["clip_qkv"] = float(g("attention.clamp_kqv"))
        if layer_norm:
            kw.update(norm_type="layer", mlp_gated=False, mlp_bias=True, attention_bias=True,
                      o_proj_bias="blk.0.attn_output.bias" in names, hidden_act="gelu_new")
        if arch == "phi2":
            kw.update(lm_head_bias="output.bias" in names, parallel_residual=True, shared_layer_norm=True)
        elif arch == "gpt2":
            kw.update(learned_positions=True, max_position_embeddings=names["position_embd.weight"].shape[0])
        elif arch == "gptneox":
            kw.update(hidden_act="gelu", parallel_residual=bool(g("use_parallel_residual", True)))
        elif arch == "phi3" and int(g("attention.sliding_window", 0) or 0) > 0:
            # HF's Phi-3 attends within the window; llama.cpp currently disables Phi-3's SWA (identical up to it)
            kw["sliding_window"] = int(g("attention.sliding_window"))
        if arch in self.GEMMA:
            # the converter stores Gemma's RMSNorm weights as 1 + w, so the norms scale by the stored weight (offset 0)
            kw.update(embed_scale=hidden_dim ** 0.5, hidden_act="gelu_pytorch_tanh")
        if arch in ("gemma2", "gemma3"):
            # llama.cpp: 1/sqrt(n_embd / n_head) for the 27B models (46 / 62 layers), else 1/sqrt(head_dim)
            big = num_layers == {"gemma2": 46, "gemma3": 62}[arch]
            kw.update(sandwich_norm=True, attn_scale=float(hidden_dim // num_heads if big else head_dim) ** -0.5)
            if arch == "gemma2":
                kw.update(attn_softcap=float(g("attn_logit_softcapping", 50.0)),
                          final_softcap=float(g("final_logit_softcapping", 30.0)))
                window, pattern, local_theta = int(g("attention.sliding_window", 4096)), 2, g("rope.freq_base_swa")
            else:
                kw["final_softcap"] = float(g("final_logit_softcapping", 0.0)) or None
                window, pattern = int(g("attention.sliding_window", 0) or 0), 6
                local_theta = g("rope.freq_base_swa", 10000.0)
            if window > 0:
                pattern = g("attention.sliding_window_pattern", pattern)
                swa = list(map(bool, pattern)) if isinstance(pattern, list) else \
                    [pattern == 0 or i % pattern < pattern - 1 for i in range(num_layers)]  # llama.cpp set_swa_pattern
                kw.update(sliding_window=window, layer_types=["sliding_attention" if s else "full_attention" for s in swa],
                          rope_local_theta=None if local_theta is None else float(local_theta))

        num_experts = int(g("expert_count", 0) or 0)
        if num_experts:
            if arch not in self.NORM_TOPK:
                raise ValueError(f"GGUF architecture '{arch}' with experts is not supported")
            if int(g("expert_gating_func", 1)) != 1:
                raise ValueError(f"{self.filepath}: expert gating other than softmax is not supported")
            moe = [i for i in range(num_layers) if f"blk.{i}.ffn_gate_inp.weight" in names]
            down = names.get(f"blk.{moe[0]}.ffn_down_exps.weight") if moe else None
            if down is None:
                raise ValueError(f"{self.filepath} declares {num_experts} experts but has no expert tensors")
            shexp = names.get(f"blk.{moe[0]}.ffn_down_shexp.weight")
            kw.update(num_experts=num_experts, num_experts_per_tok=int(g("expert_used_count")),
                      moe_ffn_dim=down.shape[-1], norm_topk_prob=self.NORM_TOPK[arch],
                      shared_expert_ffn_dim=shexp.shape[-1] if shexp is not None else 0,
                      moe_layers=None if len(moe) == num_layers else moe)
            if arch == "llama":
                kw["arch"] = "mixtral"
        return ModelConfig(name=name, tile_size=tile_size,
                           active_tiles=max(1, int(total_tiles * (1.0 - sparsity_ratio))),
                           rank_sub=64 if hidden_dim >= 768 else 32, **kw)

    def extract_tokenizer(self) -> Any:
        from .gguf_tokenizer import GGUFTokenizer
        return GGUFTokenizer(self.reader.metadata)

    def tensor_map(self, config: ModelConfig) -> Dict[str, Tuple[Tuple[str, ...], Callable]]:
        """SubspaceCausalLM parameter name -> (the llama.cpp GGUF tensors it is made from, a function of those tensors
        giving the parameter). Fused tensors (attn_qkv, Phi-3's gate+up ffn_up, ffn_gate_up_exps) are split here, and
        q/k rows un-permuted for the archs in PERMUTED."""
        names, cfg = self.reader.tensor_infos, config
        permuted, plan = self.arch in self.PERMUTED, {}

        def take(param: str, *gnames: str, fn: Callable = lambda t: t):
            plan[param] = (gnames, fn)

        def norm(param: str, gname: str):
            take(param + ".weight", gname + ".weight")
            if cfg.norm_type == "layer":
                take(param + ".bias", gname + ".bias")

        take("embed_tokens.weight", "token_embd.weight")
        take("lm_head.weight", "output.weight" if "output.weight" in names else "token_embd.weight")
        if cfg.lm_head_bias:
            take("lm_head.bias", "output.bias")
        if cfg.learned_positions:
            take("embed_positions.weight", "position_embd.weight")
        norm("norm", "output_norm")
        H, Q, K = cfg.num_heads, cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
        for i in range(cfg.num_layers):
            b, t = f"blk.{i}.", f"layers.{i}."
            norm(t + "input_layernorm", b + "attn_norm")
            if cfg.sandwich_norm:
                take(t + "attn_out_norm.weight", b + "post_attention_norm.weight")
                take(t + "mlp_out_norm.weight", b + "post_ffw_norm.weight")
            if not cfg.shared_layer_norm:
                norm(t + "post_attention_layernorm", b + "ffn_norm")
            for kind in (("weight", "bias") if cfg.attention_bias else ("weight",)):
                fused = f"{b}attn_qkv.{kind}" in names
                for p, lo, hi, heads in (("q", 0, Q, H), ("k", Q, Q + K, cfg.num_kv_heads), ("v", Q + K, Q + 2 * K, 0)):
                    def fn(x, lo=lo, hi=hi, heads=heads, fused=fused):
                        x = x[lo:hi] if fused else x
                        return unpermute_qk(x, heads) if permuted and heads else x
                    take(f"{t}self_attn.{p}_proj.{kind}", f"{b}attn_qkv.{kind}" if fused else f"{b}attn_{p}.{kind}", fn=fn)
            take(t + "self_attn.o_proj.weight", b + "attn_output.weight")
            if cfg.o_proj_bias:
                take(t + "self_attn.o_proj.bias", b + "attn_output.bias")
            if cfg.qk_norm or cfg.qk_norm_full:
                take(t + "self_attn.q_norm.weight", b + "attn_q_norm.weight")
                take(t + "self_attn.k_norm.weight", b + "attn_k_norm.weight")
            if cfg.layer_is_moe(i):
                take(t + "mlp.router.weight", b + "ffn_gate_inp.weight")
                if b + "ffn_gate_up_exps.weight" in names:
                    take(t + "mlp.gate_up", b + "ffn_gate_up_exps.weight")
                else:
                    take(t + "mlp.gate_up", b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight",
                         fn=lambda gate, up: torch.cat([gate, up], dim=1))
                take(t + "mlp.down", b + "ffn_down_exps.weight")
                if cfg.shared_expert_ffn_dim:
                    for p in ("gate", "up", "down"):
                        take(f"{t}mlp.shared_expert.{p}_proj.weight", f"{b}ffn_{p}_shexp.weight")
                    take(t + "mlp.shared_expert_gate.weight", b + "ffn_gate_inp_shexp.weight",
                         fn=lambda x: x.reshape(1, -1))
                continue
            if cfg.mlp_gated and b + "ffn_gate.weight" not in names:  # Phi-3: gate and up rows fused in ffn_up
                F = cfg.ffn_dim
                take(t + "mlp.gate_proj.weight", b + "ffn_up.weight", fn=lambda x, F=F: x[:F])
                take(t + "mlp.up_proj.weight", b + "ffn_up.weight", fn=lambda x, F=F: x[F:])
                take(t + "mlp.down_proj.weight", b + "ffn_down.weight")
            else:
                for p in (("gate", "up", "down") if cfg.mlp_gated else ("up", "down")):
                    take(f"{t}mlp.{p}_proj.weight", f"{b}ffn_{p}.weight")
            if cfg.mlp_bias:
                take(t + "mlp.up_proj.bias", b + "ffn_up.bias")
                take(t + "mlp.down_proj.bias", b + "ffn_down.bias")
        return plan

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
        plan = self.tensor_map(config)
        used = {g for gnames, _ in plan.values() for g in gnames}
        missing = sorted(g for g in used if g not in names)
        unexpected = sorted(set(names) - used - {"rope_freqs.weight", "rope_factors_long.weight",
                                                 "rope_factors_short.weight"})
        if missing or unexpected:
            raise ValueError(f"GGUF file {self.filepath} does not match a {config.num_layers}-layer "
                             f"{self.arch} model: missing tensors {missing}, unexpected tensors {unexpected}")

        # Built on the device in the target dtype: never a whole-model fp32 copy (a 70B model is 280 GB in fp32).
        # shortcut: the default dtype is process-wide while the model is built; build on "meta" if loads go concurrent.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = SubspaceCausalLM(config)
        finally:
            torch.set_default_dtype(default_dtype)

        params = dict(model.named_parameters())
        last = (None, None)  # a fused tensor feeds consecutive parameters: dequantize it once
        with torch.no_grad():
            for pname, (gnames, fn) in plan.items():
                parts = []
                for gname in gnames:
                    if last[0] != gname:
                        last = (gname, self.reader.get_tensor_tensor(gname, torch.float32))
                    parts.append(last[1])
                t = fn(*parts)
                p = params[pname]
                if t.shape != p.shape:
                    raise ValueError(f"GGUF tensor {'+'.join(gnames)} gives shape {tuple(t.shape)}; "
                                     f"{pname} is {tuple(p.shape)}")
                p.copy_(t)
                del t, parts
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
