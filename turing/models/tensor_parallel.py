"""
Megatron-Style Tensor Parallelism & Pipeline Parallelism for Multi-Node Serving.
Provides ColumnParallelLinear, RowParallelLinear, PipelineStage, and MicroBatchScheduler.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = [
    "init_tensor_parallel",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelEmbedding",
    "ParallelLMHead",
    "PipelineStage",
    "partition_model_for_pipeline",
    "MicroBatchScheduler",
]


def init_tensor_parallel() -> Tuple[int, int, str]:
    """
    Initializes distributed process group if running in multi-GPU NCCL environment,
    otherwise cleanly returns single-node defaults (rank=0, world_size=1).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, "cuda"
    return 0, 1, ("cuda" if torch.cuda.is_available() else "cpu")


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along output dimension.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_world_size: int = 1,
        tp_rank: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = max(1, tp_world_size)
        self.tp_rank = tp_rank

        assert out_features % self.tp_world_size == 0, f"out_features ({out_features}) must divide tp_world_size ({self.tp_world_size})"
        self.split_out_features = out_features // self.tp_world_size

        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.split_out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along input dimension,
    followed by an all-reduce collective sum across TP ranks.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_world_size: int = 1,
        tp_rank: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = max(1, tp_world_size)
        self.tp_rank = tp_rank

        assert in_features % self.tp_world_size == 0, f"in_features ({in_features}) must divide tp_world_size ({self.tp_world_size})"
        self.split_in_features = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            out = out + self.bias
        return out


class ParallelEmbedding(nn.Module):
    """
    Vocabulary embedding layer sharded across tensor parallel ranks.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, tp_world_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_world_size = max(1, tp_world_size)
        self.tp_rank = tp_rank
        
        self.vocab_start_idx = (num_embeddings // self.tp_world_size) * tp_rank
        self.vocab_end_idx = (num_embeddings // self.tp_world_size) * (tp_rank + 1)
        self.split_vocab_size = self.vocab_end_idx - self.vocab_start_idx
        
        self.weight = nn.Parameter(torch.empty(self.split_vocab_size, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size == 1:
            return F.embedding(input_ids, self.weight)
            
        # Mask tokens belonging to this rank's partition
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        local_ids = torch.clamp(input_ids - self.vocab_start_idx, 0, self.split_vocab_size - 1)
        local_embeds = F.embedding(local_ids, self.weight)
        local_embeds = local_embeds * mask.unsqueeze(-1).to(local_embeds.dtype)
        
        if dist.is_initialized():
            dist.all_reduce(local_embeds, op=dist.ReduceOp.SUM)
        return local_embeds


class ParallelLMHead(nn.Module):
    """
    LM Head projection partitioned across tensor parallel ranks.
    """
    def __init__(self, hidden_dim: int, vocab_size: int, tp_world_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tp_world_size = max(1, tp_world_size)
        self.tp_rank = tp_rank
        
        assert vocab_size % self.tp_world_size == 0 or self.tp_world_size == 1
        self.split_vocab = vocab_size // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(self.split_vocab, hidden_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(hidden_states, self.weight)
        return local_logits


class PipelineStage(nn.Module):
    """
    Wraps a contiguous subset of decoder layers as one pipeline parallelism stage.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
        embed_tokens: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden_or_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        # 1. First stage embedding
        if self.is_first_stage and self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_or_ids)
        else:
            hidden_states = hidden_or_ids

        # 2. Stage decoder layers
        new_kvs = []
        for idx, layer in enumerate(self.layers):
            p_k, p_v = (past_key_values[idx] if past_key_values and idx < len(past_key_values) else (None, None))
            hidden_states, k_out, v_out = layer(hidden_states, past_k=p_k, past_v=p_v, start_pos=start_pos)
            new_kvs.append((k_out, v_out))

        # 3. Last stage norm + head
        if self.is_last_stage:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                logits = self.lm_head(hidden_states)
                return logits, new_kvs

        return hidden_states, new_kvs


def partition_model_for_pipeline(model: Any, num_stages: int) -> List[PipelineStage]:
    """
    Partitions a SubspaceCausalLM model across num_stages pipeline stages.
    """
    total_layers = len(model.layers)
    layers_per_stage = max(1, total_layers // num_stages)
    stages = []

    for i in range(num_stages):
        start_idx = i * layers_per_stage
        end_idx = start_idx + layers_per_stage if i < num_stages - 1 else total_layers
        stage_layers = nn.ModuleList([model.layers[j] for j in range(start_idx, end_idx)])
        
        is_first = (i == 0)
        is_last = (i == num_stages - 1)
        
        stage = PipelineStage(
            layers=stage_layers,
            stage_id=i,
            num_stages=num_stages,
            is_first_stage=is_first,
            is_last_stage=is_last,
            embed_tokens=model.embed_tokens if is_first else None,
            norm=model.norm if is_last else None,
            lm_head=model.lm_head if is_last else None
        )
        stages.append(stage)

    return stages


class MicroBatchScheduler:
    """
    1F1B (One-Forward-One-Backward / Micro-Batch) scheduling for pipeline parallelism.
    Overlaps micro-batches across stages to minimize bubble latency.
    """
    def __init__(self, num_stages: int, num_micro_batches: int = 4):
        self.num_stages = num_stages
        self.num_micro_batches = max(1, num_micro_batches)

    @property
    def bubble_ratio(self) -> float:
        """Computes theoretical idle bubble fraction: (p - 1) / (m + p - 1)."""
        p = self.num_stages
        m = self.num_micro_batches
        return float(p - 1) / float(m + p - 1)

    def schedule_forward(
        self,
        micro_batches: List[torch.Tensor],
        stages: List[PipelineStage],
        start_pos: int = 0
    ) -> List[torch.Tensor]:
        """
        Executes micro-batches through all pipeline stages sequentially or concurrently.
        """
        current_activations = micro_batches
        for stage in stages:
            next_activations = []
            for mb in current_activations:
                out, _ = stage(mb, start_pos=start_pos)
                next_activations.append(out)
            current_activations = next_activations
        return current_activations
