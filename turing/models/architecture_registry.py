"""
Architecture Registry for Turing Engine Subspace Models: the HuggingFace model types and architecture classes the
loader converts with parity tests (tests/test_hf_parity.py). Anything else is refused rather than served wrongly.
"""

from typing import Dict, Type, Optional, Any, Tuple, List
import torch

from ..config import ModelConfig
from .causal_lm import SubspaceCausalLM

# HF model_type -> its causal-LM architecture class name (both resolve to SubspaceCausalLM)
SUPPORTED: Dict[str, str] = {
    "llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM", "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM", "gemma": "GemmaForCausalLM", "gemma2": "Gemma2ForCausalLM",
    "gemma3_text": "Gemma3ForCausalLM", "phi": "PhiForCausalLM", "phi3": "Phi3ForCausalLM",
    "gpt2": "GPT2LMHeadModel", "opt": "OPTForCausalLM", "gpt_neox": "GPTNeoXForCausalLM",
    "mixtral": "MixtralForCausalLM", "qwen2_moe": "Qwen2MoeForCausalLM", "qwen3_moe": "Qwen3MoeForCausalLM",
    "olmoe": "OlmoeForCausalLM",
}


class ArchitectureRegistry:
    """Supported HuggingFace architectures. get_model_class raises for anything not in SUPPORTED."""
    _ARCHITECTURE_MAP: Dict[str, Type[SubspaceCausalLM]] = {a: SubspaceCausalLM for a in SUPPORTED.values()}
    _MODEL_TYPE_MAP: Dict[str, Type[SubspaceCausalLM]] = {t: SubspaceCausalLM for t in SUPPORTED}

    @classmethod
    def register_architecture(cls, name: str, model_cls: Type[SubspaceCausalLM]) -> None:
        """Register a custom architecture class (its weights still need a loader mapping)."""
        cls._ARCHITECTURE_MAP[name] = model_cls

    @classmethod
    def register_model_type(cls, model_type: str, model_cls: Type[SubspaceCausalLM]) -> None:
        cls._MODEL_TYPE_MAP[model_type.lower()] = model_cls

    @classmethod
    def get_model_class(cls, arch_or_type: str) -> Type[SubspaceCausalLM]:
        """Model class for an HF architecture name or model_type; unsupported ones raise ValueError."""
        if arch_or_type in cls._ARCHITECTURE_MAP:
            return cls._ARCHITECTURE_MAP[arch_or_type]
        if arch_or_type.lower() in cls._MODEL_TYPE_MAP:
            return cls._MODEL_TYPE_MAP[arch_or_type.lower()]
        raise ValueError(f"architecture '{arch_or_type}' is not supported; supported model types: "
                         f"{', '.join(sorted(cls._MODEL_TYPE_MAP))}")

    @classmethod
    def supported_architectures(cls) -> List[str]:
        return list(cls._ARCHITECTURE_MAP.keys())

    @classmethod
    def supported_model_types(cls) -> List[str]:
        return list(cls._MODEL_TYPE_MAP.keys())


class AutoSubspaceModel:
    """Factory analogous to HuggingFace's AutoModelForCausalLM."""

    @staticmethod
    def from_config(config: ModelConfig) -> SubspaceCausalLM:
        """A randomly initialised model of this configuration (its `arch` must be supported)."""
        return ArchitectureRegistry.get_model_class(config.arch)(config)

    @staticmethod
    def from_pretrained(
        model_name_or_id: str,
        sparsity_ratio: float = 0.0,
        device: str = "auto",
        token: Optional[str] = None,
        dtype=None,
    ) -> Tuple[SubspaceCausalLM, Any]:
        """Real weights from a HuggingFace repository, a local directory, or a local .gguf file. device "auto" picks
        CUDA, then MPS, then CPU; dtype "bf16" / "fp16" / "fp32" or None (fp16/bf16 on CUDA, fp32 elsewhere)."""
        import os
        from ..config import TuringConfig
        device = str(TuringConfig(device=device).resolve_device())
        if model_name_or_id.endswith(".gguf") and os.path.isfile(model_name_or_id):
            from .gguf_loader import GGUFModelLoader
            return GGUFModelLoader(model_name_or_id).load(sparsity_ratio=sparsity_ratio, device=device)

        from .hf_loader import RealHuggingFaceLoader
        return RealHuggingFaceLoader.load_hf_model_into_turing(
            hf_model_id=model_name_or_id, sparsity_ratio=sparsity_ratio, device=device, token=token, dtype=dtype)
