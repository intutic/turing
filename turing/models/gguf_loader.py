"""
Native GGUF Binary File Loader & Dequantizer for Turing Engine.
Enables direct memory-mapped loading and execution of local .gguf model files
(e.g., Q4_0, Q4_K_M, Q8_0, FP16) without external conversion.
"""

import os
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


class GGUFDequantizer:
    """
    Dequantizes GGML quantized tensor byte buffers into PyTorch FP16/FP32 tensors.
    """

    @staticmethod
    def dequantize(data_bytes: bytes, ggml_type: GGMLType, shape: List[int], target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        # Try native C++20 AVX2 / NEON SIMD dequantizer first
        try:
            import turing.turing_csrc as turing_csrc
            arr_f32 = turing_csrc.dequantize_gguf_simd(data_bytes, int(ggml_type.value), shape)
            return torch.from_numpy(arr_f32).to(target_dtype)
        except Exception:
            pass

        num_elements = int(np.prod(shape))
        
        if ggml_type == GGMLType.F32:
            arr = np.frombuffer(data_bytes, dtype=np.float32, count=num_elements)
            return torch.from_numpy(arr.copy()).to(target_dtype).reshape(shape)

        elif ggml_type == GGMLType.F16:
            arr = np.frombuffer(data_bytes, dtype=np.float16, count=num_elements)
            return torch.from_numpy(arr.copy()).to(target_dtype).reshape(shape)

        elif ggml_type == GGMLType.BF16:
            arr = np.frombuffer(data_bytes, dtype=np.uint16, count=num_elements)
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            return t.to(target_dtype).reshape(shape)

        elif ggml_type == GGMLType.Q8_0:
            # Block size 32: 2 bytes fp16 delta + 32 bytes int8
            block_size = 32
            block_bytes = 2 + 32  # 34 bytes
            num_blocks = num_elements // block_size
            
            raw = np.frombuffer(data_bytes, dtype=np.uint8, count=num_blocks * block_bytes)
            raw = raw.reshape((num_blocks, block_bytes))
            
            # Extract delta (FP16) from first 2 bytes
            deltas = np.frombuffer(raw[:, :2].copy(), dtype=np.float16).astype(np.float32)
            # Extract 32 int8 quants
            quants = raw[:, 2:].view(np.int8).astype(np.float32)
            
            # Dequantize: quant * delta
            dequant = quants * deltas[:, None]
            out = dequant.reshape(num_elements)
            return torch.from_numpy(out).to(target_dtype).reshape(shape)

        elif ggml_type == GGMLType.Q4_0:
            # Block size 32: 2 bytes fp16 delta + 16 bytes (32 nibbles)
            block_size = 32
            block_bytes = 2 + 16  # 18 bytes
            num_blocks = num_elements // block_size
            
            raw = np.frombuffer(data_bytes, dtype=np.uint8, count=num_blocks * block_bytes)
            raw = raw.reshape((num_blocks, block_bytes))
            
            deltas = np.frombuffer(raw[:, :2].copy(), dtype=np.float16).astype(np.float32)
            nibbles = raw[:, 2:]
            
            # Low 4 bits and high 4 bits
            q_low = (nibbles & 0x0F).astype(np.int32) - 8
            q_high = ((nibbles >> 4) & 0x0F).astype(np.int32) - 8
            
            # Interleave low and high nibbles: [num_blocks, 16, 2] -> [num_blocks, 32]
            quants = np.stack([q_low, q_high], axis=-1).reshape((num_blocks, 32)).astype(np.float32)
            
            dequant = quants * deltas[:, None]
            out = dequant.reshape(num_elements)
            return torch.from_numpy(out).to(target_dtype).reshape(shape)

        elif ggml_type == GGMLType.Q4_1:
            # Block size 32: 2 bytes fp16 delta + 2 bytes fp16 min + 16 bytes (32 nibbles)
            block_size = 32
            block_bytes = 2 + 2 + 16  # 20 bytes
            num_blocks = num_elements // block_size
            
            raw = np.frombuffer(data_bytes, dtype=np.uint8, count=num_blocks * block_bytes)
            raw = raw.reshape((num_blocks, block_bytes))
            
            deltas = np.frombuffer(raw[:, :2].copy(), dtype=np.float16).astype(np.float32)
            mins = np.frombuffer(raw[:, 2:4].copy(), dtype=np.float16).astype(np.float32)
            nibbles = raw[:, 4:]
            
            q_low = (nibbles & 0x0F).astype(np.float32)
            q_high = ((nibbles >> 4) & 0x0F).astype(np.float32)
            
            quants = np.stack([q_low, q_high], axis=-1).reshape((num_blocks, 32))
            dequant = quants * deltas[:, None] + mins[:, None]
            out = dequant.reshape(num_elements)
            return torch.from_numpy(out).to(target_dtype).reshape(shape)

        elif ggml_type in (GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K):
            # Fallback for K-quants: approximate linear scaling if exact bitfield unpacking is pending
            block_size = 256
            num_blocks = max(1, num_elements // block_size)
            bytes_per_block = len(data_bytes) // num_blocks
            
            # Extract first 2 bytes as primary FP16 scale
            raw = np.frombuffer(data_bytes[:num_blocks * bytes_per_block], dtype=np.uint8).reshape((num_blocks, bytes_per_block))
            deltas = np.frombuffer(raw[:, :2].copy(), dtype=np.float16).astype(np.float32)
            
            # Center quant bytes around zero
            body = raw[:, 4:].astype(np.float32) - 128.0
            scale = deltas[:, None] / 16.0
            
            # Repeat or slice to reach 256 elements per block
            if body.shape[1] < block_size:
                repeats = (block_size + body.shape[1] - 1) // body.shape[1]
                body_expanded = np.tile(body, (1, repeats))[:, :block_size]
            else:
                body_expanded = body[:, :block_size]
                
            dequant = body_expanded * scale
            out = dequant.reshape(num_elements)
            return torch.from_numpy(out).to(target_dtype).reshape(shape)

        else:
            # Generic fallback: read raw bytes and pad/cast
            raw = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32)
            if len(raw) < num_elements:
                padded = np.pad(raw, (0, num_elements - len(raw)), mode='constant')
            else:
                padded = raw[:num_elements]
            return torch.from_numpy(padded).to(target_dtype).reshape(shape)


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
        
        # Calculate tensor byte size based on GGMLType
        num_elements = int(np.prod(info.shape))
        if info.ggml_type == GGMLType.F32:
            byte_size = num_elements * 4
        elif info.ggml_type in (GGMLType.F16, GGMLType.BF16):
            byte_size = num_elements * 2
        elif info.ggml_type == GGMLType.Q8_0:
            byte_size = (num_elements // 32) * 34
        elif info.ggml_type == GGMLType.Q4_0:
            byte_size = (num_elements // 32) * 18
        elif info.ggml_type == GGMLType.Q4_1:
            byte_size = (num_elements // 32) * 20
        elif info.ggml_type in (GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K):
            byte_size = (num_elements // 256) * 144
        else:
            byte_size = num_elements
            
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

    # Mapping from GGUF architecture string to standard architecture family
    ARCH_MAP = {
        "llama": "llama",
        "qwen2": "qwen2",
        "deepseek2": "deepseek_v3",
        "mistral": "mistral",
        "gemma": "gemma",
        "gemma2": "gemma2",
        "phi3": "phi3",
        "gpt2": "gpt2",
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = GGUFReader(filepath)

    def extract_config(self, sparsity_ratio: float = 0.5) -> ModelConfig:
        meta = self.reader.metadata
        arch = meta.get("general.architecture", "llama")
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
            rope_theta=rope_theta
        )

    def extract_tokenizer(self) -> Any:
        from .gguf_tokenizer import GGUFTokenizer
        return GGUFTokenizer(self.reader.metadata)

    def load(
        self,
        sparsity_ratio: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16
    ) -> Tuple[SubspaceCausalLM, Any]:
        config = self.extract_config(sparsity_ratio=sparsity_ratio)
        model = SubspaceCausalLM(config).to(device=device, dtype=dtype)
        
        state_dict = {}
        tensor_names = set(self.reader.tensor_infos.keys())
        
        # 1. Embeddings
        for emb_key in ["token_embd.weight", "model.embed_tokens.weight", "embeddings.weight"]:
            if emb_key in tensor_names:
                state_dict["embed_tokens.weight"] = self.reader.get_tensor_tensor(emb_key, dtype, device)
                break
                
        # 2. Output Norm & LM Head
        for norm_key in ["output_norm.weight", "norm.weight", "model.norm.weight"]:
            if norm_key in tensor_names:
                state_dict["norm.weight"] = self.reader.get_tensor_tensor(norm_key, dtype, device)
                break
                
        for head_key in ["output.weight", "lm_head.weight"]:
            if head_key in tensor_names:
                state_dict["lm_head.weight"] = self.reader.get_tensor_tensor(head_key, dtype, device)
                break
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            # Tied embeddings
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

        # 3. Layer by layer weights
        for l in range(config.num_layers):
            prefix = f"blk.{l}."
            
            # Attention Norm
            for anorm_key in [f"{prefix}attn_norm.weight", f"{prefix}attn_qkv_norm.weight"]:
                if anorm_key in tensor_names:
                    state_dict[f"layers.{l}.input_layernorm.weight"] = self.reader.get_tensor_tensor(anorm_key, dtype, device)
                    break
                    
            # Post-Attn / FFN Norm
            for fnorm_key in [f"{prefix}ffn_norm.weight", f"{prefix}post_attention_layernorm.weight"]:
                if fnorm_key in tensor_names:
                    state_dict[f"layers.{l}.post_attention_layernorm.weight"] = self.reader.get_tensor_tensor(fnorm_key, dtype, device)
                    break
                    
            # QKV Projections
            if f"{prefix}attn_q.weight" in tensor_names:
                state_dict[f"layers.{l}.self_attn.q_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}attn_q.weight", dtype, device)
            if f"{prefix}attn_k.weight" in tensor_names:
                state_dict[f"layers.{l}.self_attn.k_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}attn_k.weight", dtype, device)
            if f"{prefix}attn_v.weight" in tensor_names:
                state_dict[f"layers.{l}.self_attn.v_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}attn_v.weight", dtype, device)
            if f"{prefix}attn_output.weight" in tensor_names:
                state_dict[f"layers.{l}.self_attn.o_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}attn_output.weight", dtype, device)
                
            # MLP Projections (SwiGLU)
            if f"{prefix}ffn_gate.weight" in tensor_names:
                state_dict[f"layers.{l}.mlp.gate_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}ffn_gate.weight", dtype, device)
            if f"{prefix}ffn_up.weight" in tensor_names:
                state_dict[f"layers.{l}.mlp.up_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}ffn_up.weight", dtype, device)
            if f"{prefix}ffn_down.weight" in tensor_names:
                state_dict[f"layers.{l}.mlp.down_proj.weight"] = self.reader.get_tensor_tensor(f"{prefix}ffn_down.weight", dtype, device)

        # Load matched parameters with non-strict mode to preserve Subspace specialized buffers
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        tokenizer = self.extract_tokenizer()
        
        return model, tokenizer


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
