"""
Core algorithmic, memory, and mathematical primitives for Turing Engine.

Names are imported on first use (PEP 562), so `import turing.core` loads no submodule.
"""

import importlib

_EXPORTS = {
    "subspace": ["SubspaceRecirculation"],
    "router": ["SubspaceStructuredRouter", "DynamicEntropyRouter", "AutonomicThresholdTuner"],
    "paging": ["HierarchicalVirtualPageManager", "PageTier"],
    "optimal_transport": ["OptimalTransportEviction", "sinkhorn_knopp_eviction"],
    "speculation": ["QuadtreeMRPSpeculator", "build_dag_tree_attention_mask", "EnhancedQuadtreeDraftHead",
                    "MatryoshkaDraftHead"],
    "attention_cache": ["AttentionPatternCache", "ChunkedLongPrefillEngine"],
    "rope": ["RotaryEmbedding"],
    "cross_model_kv": ["RoPEContentDecoupler", "ClosedFormRidgeMapper", "SVDNullSpaceProjector", "CrossModelKVPipeline"],
    "mhc": ["BirkhoffManifoldProjector", "ManifoldHyperConnection"],
    "hierarchical_compression": ["HCAChunkCompressor", "CSAChunkCompressor", "CrossLayerKVSharingManager"],
    "cca": ["LayerwiseHeadBudgeter", "CompressedConvolutionalAttention"],
    "hybrid_attention": ["LinearRecurrentAttention", "ChunkContextScorer", "HybridAttentionLayerRouter"],
    "heterogeneous_moe": ["BandwidthAdaptiveDecider", "HostExpertBank", "HeterogeneousMoERunner"],
    "expert_cache": ["GPULRUExpertCache", "RoutingHistoryTracker"],
    "expert_prefetch": ["OneLayerAheadExpertPrefetcher"],
    "elastic_memory": ["ElasticMemoryBudgetManager"],
    "hybrid_mesh": ["TensorSerializer"],
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
