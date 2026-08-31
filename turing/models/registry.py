"""
Model Architecture & Sizing Registry Interface for Turing Engine.
Provides unified access to dynamic architecture discovery (via ModelConfig.from_pretrained)
and pre-calibrated offline FLOP simulation profiles (from sizing_profiles.py).
"""

from typing import Dict, Optional, List
from ..config import ModelConfig
from .sizing_profiles import SIZING_PROFILES
from .resolver import ModelResolver, ResolvedModelSpec
from .architecture_registry import ArchitectureRegistry, AutoSubspaceModel

# Re-export SIZING_PROFILES as MODEL_REGISTRY for backward compatibility
MODEL_REGISTRY: Dict[str, ModelConfig] = SIZING_PROFILES


def get_model_config(model_name_or_id: str) -> ModelConfig:
    """
    Unified model architecture configuration resolver.
    
    1. Checks offline sizing catalog (e.g. for offline benchmarking & sizing).
    2. Uses ModelResolver to parse provider namespaces, reasoning effort, and CLI aliases.
    3. Dynamically derives architecture parameters from Hugging Face Hub config.json (zero hardcoding).
    """
    # 1. Check offline sizing catalog first (instant, zero network latency)
    raw_key = model_name_or_id.lower().replace("_", "-")
    if raw_key in SIZING_PROFILES:
        return SIZING_PROFILES[raw_key]

    # 2. Parse model identifier using ModelResolver
    spec = ModelResolver.parse(model_name_or_id)
    key = spec.repo_id.lower().replace("_", "-")
    if key in SIZING_PROFILES:
        return SIZING_PROFILES[key]
    
    # 3. Dynamic AutoConfig derivation from Hugging Face Hub
    try:
        return ModelConfig.from_pretrained(spec.repo_id)
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve model '{model_name_or_id}'. "
        f"Pass a valid Hugging Face repository ID (e.g. 'meta-llama/Llama-3.3-70B-Instruct', 'deepseek-ai/DeepSeek-R1'), "
        f"a local checkpoint directory, or one of the offline sizing presets: {list(SIZING_PROFILES.keys())}"
    )


__all__ = [
    "MODEL_REGISTRY",
    "SIZING_PROFILES",
    "get_model_config",
    "ModelResolver",
    "ResolvedModelSpec",
    "ArchitectureRegistry",
    "AutoSubspaceModel",
]
