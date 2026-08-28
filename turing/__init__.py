"""
Turing Engine: Subspace-Compressed High-Performance LLM Inference & Serving Engine
"""

__version__ = "0.3.0"
__author__ = "Ishan Gupta"



from .config import ModelConfig, TuringConfig
from .core.subspace import SubspaceManager, SubspaceRecirculation
from .core.router import SubspaceStructuredRouter, DynamicEntropyRouter
from .core.paging import HierarchicalVirtualPageManager
from .models.causal_lm import SubspaceCausalLM
from .models.registry import MODEL_REGISTRY, get_model_config

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
]
