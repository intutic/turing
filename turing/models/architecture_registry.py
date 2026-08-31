"""
Architecture Registry for Turing Engine Subspace Models.
Maps Hugging Face model architecture families (e.g. LlamaForCausalLM, Qwen2ForCausalLM,
DeepseekV3ForCausalLM) to Turing Engine Subspace implementations.
"""

from typing import Dict, Type, Optional, Any, Tuple, List
import torch
import torch.nn as nn

from ..config import ModelConfig
from .causal_lm import SubspaceCausalLM

class ArchitectureRegistry:
    """
    Registry of supported transformer model architecture families.
    Enables dynamic, architecture-based dispatch rather than hardcoding model checkpoints.
    """
    _ARCHITECTURE_MAP: Dict[str, Type[SubspaceCausalLM]] = {
        "LlamaForCausalLM": SubspaceCausalLM,
        "Qwen2ForCausalLM": SubspaceCausalLM,
        "Qwen2MoeForCausalLM": SubspaceCausalLM,
        "DeepseekForCausalLM": SubspaceCausalLM,
        "DeepseekV2ForCausalLM": SubspaceCausalLM,
        "DeepseekV3ForCausalLM": SubspaceCausalLM,
        "MistralForCausalLM": SubspaceCausalLM,
        "MixtralForCausalLM": SubspaceCausalLM,
        "GemmaForCausalLM": SubspaceCausalLM,
        "Gemma2ForCausalLM": SubspaceCausalLM,
        "GPT2LMHeadModel": SubspaceCausalLM,
        "OPTForCausalLM": SubspaceCausalLM,
        "PhiForCausalLM": SubspaceCausalLM,
        "Phi3ForCausalLM": SubspaceCausalLM,
        "InternLM2ForCausalLM": SubspaceCausalLM,
    }

    _MODEL_TYPE_MAP: Dict[str, Type[SubspaceCausalLM]] = {
        "llama": SubspaceCausalLM,
        "qwen2": SubspaceCausalLM,
        "qwen2_moe": SubspaceCausalLM,
        "deepseek": SubspaceCausalLM,
        "deepseek_v2": SubspaceCausalLM,
        "deepseek_v3": SubspaceCausalLM,
        "mistral": SubspaceCausalLM,
        "mixtral": SubspaceCausalLM,
        "gemma": SubspaceCausalLM,
        "gemma2": SubspaceCausalLM,
        "gpt2": SubspaceCausalLM,
        "opt": SubspaceCausalLM,
        "phi": SubspaceCausalLM,
        "phi3": SubspaceCausalLM,
        "internlm2": SubspaceCausalLM,
    }

    @classmethod
    def register_architecture(cls, name: str, model_cls: Type[SubspaceCausalLM]) -> None:
        """Register a new custom architecture family."""
        cls._ARCHITECTURE_MAP[name] = model_cls

    @classmethod
    def register_model_type(cls, model_type: str, model_cls: Type[SubspaceCausalLM]) -> None:
        """Register a new Hugging Face model_type."""
        cls._MODEL_TYPE_MAP[model_type.lower()] = model_cls

    @classmethod
    def get_model_class(cls, arch_or_type: str) -> Type[SubspaceCausalLM]:
        """
        Resolves model class by architecture name or model_type string.
        Defaults to SubspaceCausalLM for standard decoder-only transformers.
        """
        if arch_or_type in cls._ARCHITECTURE_MAP:
            return cls._ARCHITECTURE_MAP[arch_or_type]
        if arch_or_type.lower() in cls._MODEL_TYPE_MAP:
            return cls._MODEL_TYPE_MAP[arch_or_type.lower()]
        return SubspaceCausalLM

    @classmethod
    def supported_architectures(cls) -> List[str]:
        return list(cls._ARCHITECTURE_MAP.keys())

    @classmethod
    def supported_model_types(cls) -> List[str]:
        return list(cls._MODEL_TYPE_MAP.keys())


class AutoSubspaceModel:
    """
    Dynamic factory class (analogous to Hugging Face AutoModelForCausalLM)
    that instantiates Turing Engine Subspace models from any configuration.
    """
    @staticmethod
    def from_config(config: ModelConfig) -> SubspaceCausalLM:
        """Instantiate a subspace model from a ModelConfig."""
        model_cls = ArchitectureRegistry.get_model_class(config.name)
        return model_cls(config)

    @staticmethod
    def from_pretrained(
        model_name_or_id: str,
        sparsity_ratio: float = 0.5,
        device: str = "cpu",
        token: Optional[str] = None
    ) -> Tuple[SubspaceCausalLM, Any]:
        """
        Dynamically loads pretrained weights from any Hugging Face repository
        or local directory directly into Turing Engine Subspace format.
        """
        from .hf_loader import RealHuggingFaceLoader
        return RealHuggingFaceLoader.load_hf_model_into_turing(
            hf_model_id=model_name_or_id,
            sparsity_ratio=sparsity_ratio,
            device=device,
            token=token
        )
