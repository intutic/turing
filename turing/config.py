"""
Configuration dataclasses for Turing Engine model geometries and runtime parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch

@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    ffn_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int = 32000
    tile_size: int = 256
    active_tiles: int = 48
    rank_sub: int = 64
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    router_layer_idx: Optional[int] = None # Defaults to num_layers // 3

    @property
    def total_tiles(self) -> int:
        return self.ffn_dim // self.tile_size

    @property
    def active_subspace_dim(self) -> int:
        return self.active_tiles * self.tile_size

    @property
    def sparsity_ratio(self) -> float:
        return 1.0 - (self.active_subspace_dim / self.ffn_dim)

    def __post_init__(self):
        if self.router_layer_idx is None:
            self.router_layer_idx = max(1, self.num_layers // 3)

    @classmethod
    def from_pretrained(cls, model_name_or_id: str, sparsity_ratio: float = 0.5, token: Optional[str] = None) -> "ModelConfig":
        """
        Dynamically extracts ModelConfig architecture geometry from any Hugging Face Hub repository
        or local directory containing a config.json file. Zero hardcoding required.
        """
        try:
            from transformers import AutoConfig
            import os, huggingface_hub
            hf_token = token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
            hf_cfg = AutoConfig.from_pretrained(model_name_or_id, token=hf_token)
            hidden_dim = getattr(hf_cfg, "hidden_size", getattr(hf_cfg, "n_embd", 768))
            num_layers = getattr(hf_cfg, "num_hidden_layers", getattr(hf_cfg, "n_layer", 12))
            num_heads = getattr(hf_cfg, "num_attention_heads", getattr(hf_cfg, "n_head", 12))
            num_kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
            head_dim = getattr(hf_cfg, "head_dim", hidden_dim // num_heads)
            vocab_size = getattr(hf_cfg, "vocab_size", 32000)
            ffn_dim = getattr(hf_cfg, "intermediate_size", hidden_dim * 4)
            tile_size = 64 if ffn_dim <= 4096 else 256
            total_tiles = ffn_dim // tile_size
            active_tiles = max(1, int(total_tiles * (1.0 - sparsity_ratio)))
            return cls(
                name=model_name_or_id,
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
                max_position_embeddings=getattr(hf_cfg, "max_position_embeddings", 2048),
                rope_theta=getattr(hf_cfg, "rope_theta", 10000.0)
            )
        except Exception as e:
            raise ValueError(f"Could not dynamically derive ModelConfig from '{model_name_or_id}': {e}")

@dataclass
class TuringConfig:
    device: str = "auto"
    precision: str = "fp16" # "fp16", "bf16", "fp32"
    quantization: str = "w4a16" # "w4a16", "w8a8", "none"
    group_size: int = 128
    enable_recirculation: bool = True
    recirculation_alpha: float = 0.15
    enable_speculation: bool = False
    speculation_branching: int = 4
    speculation_depth: int = 3
    enable_hierarchical_paging: bool = True
    huge_page_size: int = 512
    medium_page_size: int = 64
    small_page_size: int = 16
    ot_epsilon: float = 0.05
    ot_eviction_interval: int = 16
    max_batch_size: int = 64
    max_concurrent_streams: int = 64
    tp_world_size: int = 1
    tp_rank: int = 0

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            # 1. NVIDIA CUDA or AMD ROCm
            if torch.cuda.is_available():
                return torch.device("cuda")
            # 2. Intel XPU (SYCL / OneAPI)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
            # 3. Apple Silicon Metal (MPS)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            # 4. Vulkan Compute
            elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
                return torch.device("vulkan")
            # 5. CPU Fallback (AVX2 SIMD)
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    @property
    def is_rocm(self) -> bool:
        return torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None

    @property
    def is_intel_xpu(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @property
    def is_metal(self) -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def resolve_dtype(self) -> torch.dtype:
        if self.precision in ("fp16", "float16"):
            return torch.float16
        elif self.precision in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32
