"""
Model architectures, tensor parallelism, converters, and upstream adapters for Turing Engine.
"""

from .causal_lm import SubspaceCausalLM, SubspaceDecoderLayer, SubspaceAttention, SubspaceMLP
from .tensor_parallel import ColumnParallelLinear, RowParallelLinear, init_tensor_parallel
from .registry import MODEL_REGISTRY, get_model_config
from .converter import TuringConverter
from .adapters import UncertaintyKnowledgeGate, TenantLoRAAdapter
from .vllm_adapter import VLLMCustomModelRunnerHarness, VLLMSubspaceQuantWrapper
from .gguf_loader import GGUFModelLoader, GGUFReader, GGUFDequantizer
from .gguf_tokenizer import GGUFTokenizer
from .ingest import TuringIngestEngine, StorageTier

__all__ = [
    "SubspaceCausalLM",
    "SubspaceDecoderLayer",
    "SubspaceAttention",
    "SubspaceMLP",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "init_tensor_parallel",
    "MODEL_REGISTRY",
    "get_model_config",
    "TuringConverter",
    "UncertaintyKnowledgeGate",
    "TenantLoRAAdapter",
    "VLLMCustomModelRunnerHarness",
    "VLLMSubspaceQuantWrapper",
    "GGUFModelLoader",
    "GGUFReader",
    "GGUFDequantizer",
    "GGUFTokenizer",
    "TuringIngestEngine",
    "StorageTier",
]
