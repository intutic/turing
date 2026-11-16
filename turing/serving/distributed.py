"""
Multi-process inference over torch.distributed: tensor parallelism (TP) inside each pipeline stage, pipeline
parallelism (PP) across stages.

Ranks are laid out stage-major: rank = stage * tp + tp_rank. Every rank builds the same full model, and
DistributedInferenceDriver keeps its shard (turing/models/tensor_parallel.py) and its own KV pool for its heads and
layers. Rank 0 (stage 0, TP rank 0) runs the engine and the HTTP server; every other rank runs `worker_loop()`.

One step: rank 0 broadcasts a header (the micro-batches' token ids and cache ids, plus the ids of caches freed since
the last step) and every rank runs it. Within a stage the TP ranks compute in lockstep (all-reduces inside the
layers). Between stages, TP rank t of stage s sends its hidden states to TP rank t of stage s + 1; the last stage's
TP rank 0 sends the logits to rank 0. Several micro-batches (`forward_many`, e.g. the prompts of several requests)
flow through the stages forward-only 1F1B style: a stage runs micro-batch m as soon as it arrives and sends it on
asynchronously, so stage s works on m + 1 while stage s + 1 works on m.

Launch with torchrun (one process per GPU), e.g. `torchrun --nproc_per_node 4 -m turing.cli serve --model ...
--tensor-parallel 2 --pipeline-parallel 2`; see scripts/launch_distributed.sh.
"""

import datetime
import itertools
import os
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from ..core.kv_cache import KVCache
from ..models.tensor_parallel import keep_pipeline_stage, shard_tensor_parallel, stage_forward

__all__ = ["DistributedConfig", "PlacementPolicy", "DistributedInferenceDriver", "init_distributed"]


@dataclass
class DistributedConfig:
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
        return self.world_size > 1

    @classmethod
    def from_env(cls, tp_size: int = 1, pp_size: int = 1) -> "DistributedConfig":
        """From torchrun's environment (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)."""
        return cls(tp_size=tp_size, pp_size=pp_size,
                   master_addr=os.environ.get("MASTER_ADDR", "localhost"),
                   master_port=int(os.environ.get("MASTER_PORT", 29500)),
                   backend="nccl" if torch.cuda.is_available() else "gloo",
                   rank=int(os.environ.get("RANK", 0)),
                   world_size=int(os.environ.get("WORLD_SIZE", tp_size * pp_size)),
                   local_rank=int(os.environ.get("LOCAL_RANK", 0)))

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.local_rank) if self.backend == "nccl" else torch.device("cpu")


def init_distributed(dc: DistributedConfig, timeout_s: float = 600.0) -> None:
    """Join the process group (once per process). NCCL ranks use GPU local_rank."""
    if dc.tp_size * dc.pp_size != dc.world_size:
        raise ValueError(f"tensor parallel {dc.tp_size} x pipeline parallel {dc.pp_size} must equal the world size "
                         f"{dc.world_size}")
    if dc.backend == "nccl":
        torch.cuda.set_device(dc.local_rank)
    if not dist.is_initialized():
        # under torchrun, env:// reuses the store its agent already serves on MASTER_PORT
        init = "env://" if "TORCHELASTIC_RUN_ID" in os.environ else f"tcp://{dc.master_addr}:{dc.master_port}"
        dist.init_process_group(dc.backend, init_method=init, rank=dc.rank, world_size=dc.world_size,
                                timeout=datetime.timedelta(seconds=timeout_s))


class PlacementPolicy:
    """A (tp, pp) split for a GPU count and model size: TP within a node first, PP once a model needs more."""

    @staticmethod
    def auto_decompose(num_gpus: int, model_params_billion: float = 7.0) -> Tuple[int, int]:
        if num_gpus <= 1:
            return (1, 1)
        elif num_gpus == 2:
            return (2, 1)
        elif num_gpus == 4:
            return (2, 2) if model_params_billion >= 30.0 else (4, 1)
        elif num_gpus == 8:
            return (4, 2) if model_params_billion >= 70.0 else (8, 1)
        elif num_gpus >= 16:
            pp = max(2, num_gpus // 8)
            return (min(8, num_gpus // pp), pp)
        tp = 2
        while tp * 2 <= num_gpus and tp < 8:
            tp *= 2
        return (tp, max(1, num_gpus // tp))


Caches = Union[KVCache, List[KVCache]]


class DistributedInferenceDriver:
    """Shards `model` for this rank (in place; load it on the CPU so no rank holds the full model on its GPU), moves
    it to this rank's device, and runs steps across ranks. Every rank must pass identical weights. On rank 0, `forward_distributed` is what
    ContinuousBatchEngine calls (`dist_driver`); every other rank calls `worker_loop()`. Rank 0's caches are the
    engine's, in this rank's pool (model.kv_pool()); the other ranks keep a cache per rank-0 cache, created on its
    first step and freed when rank 0 frees or drops its cache."""

    def __init__(self, model: nn.Module, config=None, dist_config: Optional[DistributedConfig] = None,
                 device: Optional[torch.device] = None):
        dc = dist_config or DistributedConfig.from_env()
        self.dist_config = dc
        self.tp, self.pp = dc.tp_size, dc.pp_size
        if dc.world_size > 1:
            init_distributed(dc)
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        self.stage, self.tp_rank = divmod(self.rank, self.tp)
        self.tp_group = None
        if self.tp > 1:
            for s in range(self.pp):  # every rank creates every group, in the same order
                g = dist.new_group(list(range(s * self.tp, (s + 1) * self.tp)))
                if s == self.stage:
                    self.tp_group = g
        shard_tensor_parallel(model, self.tp_rank, self.tp, self.tp_group)
        keep_pipeline_stage(model, self.stage, self.pp)
        if device is None:  # under torchrun each rank takes its own GPU (cuda:LOCAL_RANK)
            device = dc.device if dc.world_size > 1 else next(model.parameters()).device
        self.model = model.to(device).eval()
        self.first, self.last = self.stage == 0, self.stage == self.pp - 1
        self._ids = itertools.count()
        self._live: Dict[int, weakref.ref] = {}   # rank 0: id -> its cache
        self._caches: Dict[int, KVCache] = {}     # other ranks: id -> this rank's cache

    # ------------------------------------------------------------------ rank 0

    def forward_distributed(self, input_ids: torch.Tensor, past_key_values: Optional[Caches] = None,
                            start_pos: int = 0, logits_to_keep: int = 0) -> Tuple[torch.Tensor, Caches]:
        """SubspaceCausalLM.forward across ranks: input_ids [1, L] with one request's KVCache (None makes one), or
        [B, 1] with a list of B caches (a batched decode step). Positions come from the caches (start_pos unused)."""
        caches = past_key_values if past_key_values is not None else KVCache(self.model.kv_pool())
        return self.forward_many([(input_ids, caches)], logits_to_keep)[0], caches

    def forward_many(self, micro_batches: List[Tuple[torch.Tensor, Caches]], logits_to_keep: int = 0) -> List[torch.Tensor]:
        """Several (input_ids, caches) micro-batches in one pipelined pass (e.g. the prompts of several requests);
        returns each one's logits."""
        if self.rank != 0:
            raise RuntimeError("only rank 0 drives steps; other ranks run worker_loop()")
        mbs, header = [], []
        for ids, caches in micro_batches:
            rows = caches if isinstance(caches, list) else [caches]
            for c in rows:
                if not hasattr(c, "_dist_id"):
                    if c.length:
                        raise ValueError("a distributed request's cache must start empty (its other shards are "
                                         "built step by step)")
                    c._dist_id = next(self._ids)
                    self._live[c._dist_id] = weakref.ref(c)
            cids = [c._dist_id for c in rows] if isinstance(caches, list) else caches._dist_id
            header.append((ids.tolist(), cids))
            mbs.append((ids, caches))
        freed = [i for i, ref in self._live.items() if ref() is None or ref().pages is None]
        for i in freed:
            del self._live[i]
        if self.dist_config.world_size > 1:
            dist.broadcast_object_list([{"op": "step", "mbs": header, "freed": freed, "keep": logits_to_keep}], src=0)
        return self._run(mbs, logits_to_keep)

    def shutdown(self) -> None:
        """Rank 0: end every worker_loop."""
        if self.rank == 0 and self.dist_config.world_size > 1:
            dist.broadcast_object_list([{"op": "stop"}], src=0)

    # ------------------------------------------------------------------ other ranks

    def worker_loop(self) -> None:
        """Run rank 0's steps until it calls shutdown()."""
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            while True:
                msg = [None]
                dist.broadcast_object_list(msg, src=0)
                msg = msg[0]
                if msg["op"] == "stop":
                    return
                for i in msg["freed"]:
                    c = self._caches.pop(i, None)
                    if c is not None:
                        c.free()
                mbs = [(torch.tensor(ids, dtype=torch.long, device=device),
                        [self._cache(i) for i in cids] if isinstance(cids, list) else self._cache(cids))
                       for ids, cids in msg["mbs"]]
                self._run(mbs, msg["keep"])

    def _cache(self, cid: int) -> KVCache:
        if cid not in self._caches:
            self._caches[cid] = KVCache(self.model.kv_pool())
        return self._caches[cid]

    # ------------------------------------------------------------------ every rank

    def _run(self, mbs: List[Tuple[torch.Tensor, Caches]], keep: int) -> List[torch.Tensor]:
        if self.pp == 1:  # the model's own forward (fused CUDA decode paths); TP collectives run inside it
            return [self.model(ids, past_key_values=c, logits_to_keep=keep)[0] for ids, c in mbs]
        tp, model = self.tp, self.model
        w = model.embed_tokens.weight
        prev, nxt = (self.stage - 1) * tp + self.tp_rank, (self.stage + 1) * tp + self.tp_rank
        sends = []  # (work, tensor): a tensor must outlive its asynchronous send
        for ids, caches in mbs:
            if self.first:
                x = ids
            else:
                x = torch.empty(*ids.shape, model.config.hidden_dim, dtype=w.dtype, device=w.device)
                dist.recv(x, src=prev)
            y = stage_forward(model, x, caches, self.first, self.last, keep).contiguous()
            if not self.last:
                sends.append((dist.isend(y, dst=nxt), y))
            elif self.tp_rank == 0:
                sends.append((dist.isend(y, dst=0), y))
        outs = []
        if self.rank == 0:
            src = (self.pp - 1) * tp
            vocab = model.config.vocab_size
            for ids, _ in mbs:
                logits = torch.empty(ids.shape[0], keep or ids.shape[1], vocab, dtype=w.dtype, device=w.device)
                dist.recv(logits, src=src)
                outs.append(logits)
        for work, _ in sends:
            work.wait()
        return outs
