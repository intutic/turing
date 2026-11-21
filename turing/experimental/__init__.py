"""
Experimental components: research code that `turing serve` does not use.

These modules have unit tests (kernels have parity tests), and some were evaluated on real models; the results are in
docs/experimental.md. Their APIs may change or be removed in any release.

Names are imported on first use (PEP 562), so `import turing.experimental` loads no submodule.
"""

import importlib

_EXPORTS = {
    "paging": ["HierarchicalVirtualPageManager", "PageTier"],
    "cross_model_kv": ["RoPEContentDecoupler", "ClosedFormRidgeMapper", "SVDNullSpaceProjector", "CrossModelKVPipeline"],
    "hierarchical_compression": ["HCAChunkCompressor", "CSAChunkCompressor", "CrossLayerKVSharingManager"],
    "hybrid_attention": ["LinearRecurrentAttention", "ChunkContextScorer", "HybridAttentionLayerRouter"],
    "expert_prefetch": ["OneLayerAheadExpertPrefetcher"],
    "elastic_memory": ["ElasticMemoryBudgetManager"],
    "lineage": ["CleanBaseLineageBuffer"],
    "kslot_pooling": ["KSlotCachePooler"],
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
