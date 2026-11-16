"""
Configuration dataclasses for Turing Engine model geometries and runtime parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch


def hf_rope_theta(hf_cfg, layer_type: Optional[str] = None) -> float:
    """RoPE base of a HuggingFace config; transformers 5 moved it from `rope_theta` into `rope_parameters` (per layer
    type for some models)."""
    p = getattr(hf_cfg, "rope_parameters", None) or {}
    if layer_type is not None and isinstance(p.get(layer_type), dict):
        return float(p[layer_type].get("rope_theta", 10000.0))
    return float(getattr(hf_cfg, "rope_theta", None) or p.get("rope_theta", 10000.0))


def _rope_dict(p: Optional[dict]) -> Optional[dict]:
    """A rope-parameters dict without the base and the partial rotary factor (kept elsewhere); None for plain RoPE."""
    from .core.rope import ROPE_TYPES
    p = {k: v for k, v in (p or {}).items() if k not in ("rope_theta", "partial_rotary_factor")}
    kind = p.get("rope_type", p.get("type", "default"))
    if kind not in ROPE_TYPES:
        raise ValueError(f"RoPE scaling type '{kind}' is not supported (known: {', '.join(ROPE_TYPES)})")
    return None if kind == "default" else p


def hf_rope_scaling(hf_cfg, layer_type: Optional[str] = None) -> Optional[dict]:
    """RoPE scaling of a HuggingFace config (of one layer type, for configs that set them per type, like Gemma 3) as
    a rope-parameters dict ({"rope_type": ..., "factor": ...}); None for plain RoPE. Unknown types raise."""
    p = getattr(hf_cfg, "rope_parameters", None) or getattr(hf_cfg, "rope_scaling", None) or {}
    if layer_type is not None and isinstance(p.get(layer_type), dict):
        p = p[layer_type]
    return _rope_dict(p)


def _legacy_llama3(t) -> dict:
    """rope_scaling used to be a (factor, low_freq_factor, high_freq_factor, original_max) llama3 tuple."""
    return {"rope_type": "llama3", "factor": float(t[0]), "low_freq_factor": float(t[1]),
            "high_freq_factor": float(t[2]), "original_max_position_embeddings": int(t[3])}


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
    rms_norm_eps: float = 1e-6
    enable_recirculation: bool = False
    kv_cache_dtype: str = "native"     # "native", "int8" or "int4" (see turing/core/kv_cache.py)
    kv_page_budget: float = 1.0        # Quest: share of 16-token pages a decode step reads; 1.0 reads all
    kv_select_min_tokens: int = 8192   # Quest only from this length: on an L4 it beats reading every page only at
                                       # batch >= 64 and >= 8K tokens (results_quest_threshold_cuda_smollm2_360m.json)
    kv_pool_gib: Optional[float] = None  # KV pool size; None = kv_pool_fraction of free CUDA memory (4 GiB off CUDA)
    kv_pool_fraction: float = 0.6      # share of free CUDA memory the KV pool takes when kv_pool_gib is unset
    cuda_graphs: bool = True           # replay decode steps as CUDA graphs (CUDA only; see DecodeGraphs)
    rope_scaling: Optional[dict] = None  # HF rope-parameters dict ({"rope_type": "llama3", ...}); see hf_rope_scaling
    attention_bias: bool = False       # q/k/v projection biases (Qwen2, GPT-2, OPT, GPT-NeoX, Phi-2)
    qk_norm: bool = False              # per-head norm on q and k before RoPE (Qwen3, Gemma 3)
    # Architecture features of the supported HuggingFace families; the defaults are Llama's.
    arch: str = "llama"                # HF model_type the weights came from
    norm_type: str = "rms"             # "rms" (RMSNorm) or "layer" (LayerNorm with bias)
    norm_offset: float = 0.0           # Gemma's RMSNorm scales by (1 + weight): offset 1.0
    hidden_act: str = "silu"
    mlp_gated: bool = True             # down(act(gate(x)) * up(x)); False: down(act(up(x))) (GPT-2, OPT, NeoX, Phi-2)
    mlp_bias: bool = False
    o_proj_bias: bool = False
    lm_head_bias: bool = False         # Phi-2
    rotary_dim: Optional[int] = None   # rotated dimensions of each head; None: all (partial rotary: NeoX, Phi)
    learned_positions: bool = False    # absolute position embeddings added to the input (GPT-2, OPT); no RoPE
    position_offset: int = 0           # OPT's position ids start at 2
    parallel_residual: bool = False    # x + attn(norm1(x)) + mlp(norm2(x)) (GPT-NeoX; Phi-2 with one shared norm)
    shared_layer_norm: bool = False    # Phi-2: attention and MLP read the same normed input
    pre_norm: bool = True              # False: post-LN layers (OPT-350m)
    final_norm: bool = True
    embed_scale: Optional[float] = None      # Gemma multiplies embeddings by sqrt(hidden)
    embed_proj_dim: Optional[int] = None     # OPT-350m: embeddings narrower than hidden, projected in and out
    sandwich_norm: bool = False        # Gemma 2/3: norms on the attention and MLP outputs as well
    attn_scale: Optional[float] = None       # None: 1 / sqrt(head_dim); Gemma: query_pre_attn_scalar ** -0.5
    attn_softcap: Optional[float] = None     # Gemma 2: tanh soft-capping of attention scores
    final_softcap: Optional[float] = None    # Gemma 2: tanh soft-capping of the logits
    sliding_window: Optional[int] = None     # tokens a sliding-window layer sees (key j visible to query i if j > i - w)
    layer_types: Optional[List[str]] = None  # per layer "full_attention" / "sliding_attention"; None: all one kind
    rope_local_theta: Optional[float] = None  # Gemma 3: RoPE base of the sliding-window layers
    rope_local_scaling: Optional[dict] = None
    qk_norm_full: bool = False         # OLMoE: q/k norms over the whole projection rather than per head
    clip_qkv: Optional[float] = None   # OLMoE option: clamp q, k, v
    num_experts: int = 0               # MoE (Mixtral, Qwen2/3-MoE, OLMoE): experts per MoE layer; 0 = dense model
    num_experts_per_tok: int = 0
    moe_ffn_dim: Optional[int] = None  # each expert's intermediate size (default ffn_dim)
    norm_topk_prob: bool = True        # renormalise the top-k router probabilities
    shared_expert_ffn_dim: int = 0     # Qwen2-MoE: a sigmoid-gated shared expert
    moe_layers: Optional[List[int]] = None  # indices of MoE layers; None = all layers when num_experts > 0

    @property
    def total_tiles(self) -> int:
        return max(1, self.ffn_dim // max(1, self.tile_size))

    @property
    def active_subspace_dim(self) -> int:
        return min(self.ffn_dim, self.active_tiles * self.tile_size)

    @property
    def sparsity_ratio(self) -> float:
        return 1.0 - (self.active_subspace_dim / max(1, self.ffn_dim))

    def layer_window(self, i: int) -> Optional[int]:
        """Sliding window of layer i, or None for full attention."""
        if not self.sliding_window:
            return None
        if self.layer_types is None:
            return self.sliding_window
        return self.sliding_window if self.layer_types[i] == "sliding_attention" else None

    def layer_is_moe(self, i: int) -> bool:
        return self.num_experts > 0 and (self.moe_layers is None or i in self.moe_layers)

    def __post_init__(self):
        if isinstance(self.rope_scaling, (tuple, list)):
            self.rope_scaling = _legacy_llama3(self.rope_scaling)
        if self.router_layer_idx is None:
            self.router_layer_idx = max(1, self.num_layers // 3)
        if self.active_tiles > self.total_tiles:
            self.active_tiles = self.total_tiles
        if self.rank_sub > self.hidden_dim:
            self.rank_sub = max(1, self.hidden_dim)

    @classmethod
    def from_gguf_metadata(cls, metadata: Dict[str, Any], sparsity_ratio: float = 0.0) -> "ModelConfig":
        """
        Dynamically derives ModelConfig from parsed GGUF metadata dictionary.
        """
        arch = metadata.get("general.architecture", "llama")
        name = metadata.get("general.name", f"gguf-{arch}")
        
        hidden_dim = int(metadata.get(f"{arch}.embedding_length", 2048))
        num_layers = int(metadata.get(f"{arch}.block_count", 16))
        num_heads = int(metadata.get(f"{arch}.attention.head_count", 16))
        num_kv_heads = int(metadata.get(f"{arch}.attention.head_count_kv", num_heads))
        head_dim = int(metadata.get(f"{arch}.attention.key_length", hidden_dim // num_heads))
        ffn_dim = int(metadata.get(f"{arch}.feed_forward_length", hidden_dim * 4))
        
        vocab_size = 32000
        if "tokenizer.ggml.tokens" in metadata:
            vocab_size = len(metadata["tokenizer.ggml.tokens"])
        elif f"{arch}.vocab_size" in metadata:
            vocab_size = int(metadata[f"{arch}.vocab_size"])
            
        max_pos = int(metadata.get(f"{arch}.context_length", 4096))
        rope_theta = float(metadata.get(f"{arch}.rope.freq_base", 10000.0))
        
        tile_size = 64 if ffn_dim <= 4096 else 256
        total_tiles = ffn_dim // tile_size
        active_tiles = max(1, int(total_tiles * (1.0 - sparsity_ratio)))
        
        return cls(
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

    @classmethod
    def from_pretrained(cls, model_name_or_id: str, sparsity_ratio: float = 0.0, token: Optional[str] = None) -> "ModelConfig":
        """
        Dynamically extracts ModelConfig architecture geometry from any Hugging Face Hub repository,
        local directory containing a config.json file, or local .gguf binary file. Zero hardcoding required.
        """
        try:
            import os
            if model_name_or_id.endswith(".gguf") and os.path.isfile(model_name_or_id):
                from .models.gguf_loader import GGUFReader
                reader = GGUFReader(model_name_or_id)
                return cls.from_gguf_metadata(reader.metadata, sparsity_ratio=sparsity_ratio)

            from transformers import AutoConfig
            import huggingface_hub
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
                rope_theta=hf_rope_theta(hf_cfg)
            )
        except Exception as e:
            raise ValueError(f"Could not dynamically derive ModelConfig from '{model_name_or_id}': {e}")

@dataclass
class TuringConfig:
    device: str = "auto"
    precision: str = "fp16" # "fp16", "bf16", "fp32"
    quantization: str = "w4a16" # "w4a16", "w8a8", "none"
    group_size: int = 128
    enable_recirculation: bool = False
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
        if self.device == "rocm":  # PyTorch exposes AMD GPUs through its HIP build as "cuda"
            if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
                raise RuntimeError("--device rocm needs a ROCm build of PyTorch and a visible AMD GPU "
                                   "(pip install torch --index-url https://download.pytorch.org/whl/rocm6.2)")
            return torch.device("cuda")
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
