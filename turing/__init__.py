"""
Turing Engine: Edge-to-Cloud Distributed Subspace LLM Inference Runtime.
"""

__version__ = "1.7.0"


__author__ = "Ishan Gupta"

import importlib

# Imported on first use (PEP 562): `import turing` loads neither torch nor the model code.
_EXPORTS = {
    ".config": ["ModelConfig", "TuringConfig"],
    ".core.subspace": ["SubspaceRecirculation"],
    ".core.router": ["SubspaceStructuredRouter", "DynamicEntropyRouter"],
    ".core.paging": ["HierarchicalVirtualPageManager"],
    ".models.causal_lm": ["SubspaceCausalLM"],
    ".models.registry": ["MODEL_REGISTRY", "get_model_config"],
    ".dsl": ["chain", "gen", "fork", "join", "select", "constrain"],
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = list(_WHERE)


def __getattr__(name):
    module = _WHERE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
