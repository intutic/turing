"""
Turing Engine: Edge-to-Cloud Distributed Subspace LLM Inference Runtime.
"""

__version__ = "1.0.0"


__author__ = "Ishan Gupta"





from .config import ModelConfig, TuringConfig
from .core.subspace import SubspaceManager, SubspaceRecirculation
from .core.router import SubspaceStructuredRouter, DynamicEntropyRouter
from .core.paging import HierarchicalVirtualPageManager
from .models.causal_lm import SubspaceCausalLM
from .models.registry import MODEL_REGISTRY, get_model_config
from .dsl import chain, gen, fork, join, select, constrain

__all__ = [
    "ModelConfig",
    "TuringConfig",
    "SubspaceManager",
    "SubspaceRecirculation",
    "SubspaceStructuredRouter",
    "DynamicEntropyRouter",
    "HierarchicalVirtualPageManager",
    "SubspaceCausalLM",
    "MODEL_REGISTRY",
    "get_model_config",
    "chain",
    "gen",
    "fork",
    "join",
    "select",
    "constrain",
]
