"""
Distributed Multi-Node & Multi-GPU Inference Driver for Turing Engine.
Coordinates Tensor Parallelism (TP) and Pipeline Parallelism (PP) across physical GPU clusters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist

from ..config import ModelConfig
from ..models.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    PipelineStage,
    partition_model_for_pipeline,
    MicroBatchScheduler,
    init_tensor_parallel,
)

__all__ = [
    "DistributedConfig",
    "PlacementPolicy",
    "DistributedInferenceDriver",
]


@dataclass
class DistributedConfig:
    """
    Configuration for distributed multi-GPU & multi-node inference.
    """
    tp_size: int = 1
    pp_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return (self.tp_size > 1 or self.pp_size > 1) and self.world_size > 1

    @classmethod
    def from_env(cls, tp_size: int = 1, pp_size: int = 1) -> "DistributedConfig":
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", tp_size * pp_size))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", 29500))
        
        return cls(
            tp_size=tp_size,
            pp_size=pp_size,
            master_addr=master_addr,
            master_port=master_port,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
            local_rank=local_rank
        )


class PlacementPolicy:
    """
    Auto-determines optimal Tensor Parallel (TP) and Pipeline Parallel (PP)
    decomposition based on available GPU count and model parameter scale.
    """
    @staticmethod
    def auto_decompose(num_gpus: int, model_params_billion: float = 7.0) -> Tuple[int, int]:
        """
        Returns (tp_size, pp_size) such that tp_size * pp_size <= num_gpus.
        """
        if num_gpus <= 1:
            return (1, 1)
        elif num_gpus == 2:
            return (2, 1)
        elif num_gpus == 4:
            if model_params_billion >= 30.0:
                return (2, 2)
            return (4, 1)
        elif num_gpus == 8:
            if model_params_billion >= 70.0:
                return (4, 2)
            return (8, 1)
        elif num_gpus >= 16:
            pp = max(2, num_gpus // 8)
            tp = min(8, num_gpus // pp)
            return (tp, pp)
        else:
            # Fallback: largest power of 2 for TP, rest for PP
            tp = 2
            while tp * 2 <= num_gpus and tp < 8:
                tp *= 2
            pp = max(1, num_gpus // tp)
            return (tp, pp)


class DistributedInferenceDriver:
    """
    Driver managing multi-GPU Tensor Parallelism and Pipeline Parallelism.
    """
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        dist_config: Optional[DistributedConfig] = None
    ):
        self.model = model
        self.config = config
        self.dist_config = dist_config or DistributedConfig.from_env()
        
        # Partition pipeline stages
        if self.dist_config.pp_size > 1:
            self.stages = partition_model_for_pipeline(self.model, self.dist_config.pp_size)
            self.scheduler = MicroBatchScheduler(num_stages=self.dist_config.pp_size)
        else:
            self.stages = [
                PipelineStage(
                    layers=self.model.layers,
                    stage_id=0,
                    num_stages=1,
                    is_first_stage=True,
                    is_last_stage=True,
                    embed_tokens=self.model.embed_tokens,
                    norm=self.model.norm,
                    lm_head=self.model.lm_head
                )
            ]
            self.scheduler = MicroBatchScheduler(num_stages=1)

    def forward_distributed(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Executes distributed forward pass through pipeline stages and TP layers.
        """
        if self.dist_config.pp_size == 1:
            # Single stage: full forward pass
            stage = self.stages[0]
            return stage(input_ids, past_key_values=past_key_values, start_pos=start_pos)

        # Multi-stage pipeline execution
        activations = input_ids
        all_new_kvs = []
        
        for stage in self.stages:
            activations, stage_kvs = stage(activations, past_key_values=past_key_values, start_pos=start_pos)
            if stage_kvs:
                all_new_kvs.extend(stage_kvs)

        return activations, all_new_kvs

    def all_gather_logits(self, local_logits: torch.Tensor) -> torch.Tensor:
        """
        Gathers TP-partitioned logits across all ranks in tensor parallel group.
        """
        if not dist.is_initialized() or self.dist_config.tp_size == 1:
            return local_logits
            
        gathered = [torch.empty_like(local_logits) for _ in range(self.dist_config.tp_size)]
        dist.all_gather(gathered, local_logits)
        return torch.cat(gathered, dim=-1)
