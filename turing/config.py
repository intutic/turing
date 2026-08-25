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
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        if self.precision in ("fp16", "float16"):
            return torch.float16
        elif self.precision in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32
