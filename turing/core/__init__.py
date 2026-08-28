"""
Core algorithmic, memory, and mathematical primitives for Turing Engine.
"""

from .subspace import SubspaceManager, SubspaceRecirculation
from .router import SubspaceStructuredRouter, DynamicEntropyRouter, AutonomicThresholdTuner
from .paging import HierarchicalVirtualPageManager, PageTier
from .optimal_transport import OptimalTransportEviction, sinkhorn_knopp_eviction
from .speculation import QuadtreeMRPSpeculator, build_dag_tree_attention_mask, EnhancedQuadtreeDraftHead, MatryoshkaDraftHead
from .attention_cache import AttentionPatternCache, ChunkedLongPrefillEngine
from .rope import NTKDynamicRoPEScaling
from .radix_svd import SpectralRadixSVDForest
from .pcie_swapper import DoubleBufferedAsyncRingSwapper
from .cross_model_kv import RoPEContentDecoupler, ClosedFormRidgeMapper, SVDNullSpaceProjector, CrossModelKVPipeline
from .mhc import BirkhoffManifoldProjector, ManifoldHyperConnection
from .hierarchical_compression import HCAChunkCompressor, CSAChunkCompressor, CrossLayerKVSharingManager
from .cca import LayerwiseHeadBudgeter, CompressedConvolutionalAttention
from .heterogeneous_moe import BandwidthAdaptiveDecider, HostExpertBank, HeterogeneousMoERunner
from .expert_cache import GPULRUExpertCache
from .elastic_memory import ElasticMemoryBudgetManager
from .hybrid_mesh import (
    HybridMeshConfig,
    HybridMeshCoordinator,
    TensorSerializer,
    LocalPipelineStage,
    RemotePipelineStage,
    CascadedPrefillAndDraftSpeculator,
    DistributedMoEExpertMesh
)

__all__ = [
    "SubspaceManager",
    "SubspaceRecirculation",
    "SubspaceStructuredRouter",
    "DynamicEntropyRouter",
    "AutonomicThresholdTuner",

    "HierarchicalVirtualPageManager",
    "PageTier",
    "OptimalTransportEviction",
    "sinkhorn_knopp_eviction",
    "QuadtreeMRPSpeculator",
    "build_dag_tree_attention_mask",
    "EnhancedQuadtreeDraftHead",
    "MatryoshkaDraftHead",
    "AttentionPatternCache",
    "ChunkedLongPrefillEngine",
    "NTKDynamicRoPEScaling",
    "SpectralRadixSVDForest",
    "DoubleBufferedAsyncRingSwapper",
    "RoPEContentDecoupler",
    "ClosedFormRidgeMapper",
    "SVDNullSpaceProjector",
    "CrossModelKVPipeline",
    "BirkhoffManifoldProjector",
    "ManifoldHyperConnection",
    "HCAChunkCompressor",
    "CSAChunkCompressor",
    "CrossLayerKVSharingManager",
    "LayerwiseHeadBudgeter",
    "CompressedConvolutionalAttention",
    "BandwidthAdaptiveDecider",
    "HostExpertBank",
    "HeterogeneousMoERunner",
    "GPULRUExpertCache",
    "ElasticMemoryBudgetManager",
    "HybridMeshConfig",
    "HybridMeshCoordinator",
    "TensorSerializer",
    "LocalPipelineStage",
    "RemotePipelineStage",
    "CascadedPrefillAndDraftSpeculator",
    "DistributedMoEExpertMesh",
]

