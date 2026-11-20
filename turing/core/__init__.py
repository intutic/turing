"""
Core primitives used by the serving path: the paged KV cache, RoPE variants and MoE expert offload.
Experimental components live in `turing.experimental`.

Names are imported on first use (PEP 562), so `import turing.core` loads no submodule.
"""

import importlib

_EXPORTS = {
    "subspace": ["SubspaceRecirculation"],
    "rope": ["RotaryEmbedding"],
    "heterogeneous_moe": ["BandwidthAdaptiveDecider", "HostExpertBank", "HeterogeneousMoERunner"],
    "expert_cache": ["GPULRUExpertCache", "RoutingHistoryTracker"],
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = list(_WHERE)


def __getattr__(name):
    module = _WHERE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
