"""
Megatron-Style Tensor Parallelism (ColumnParallelLinear, RowParallelLinear) with Auto-Fallback.
"""

from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
