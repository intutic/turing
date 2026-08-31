"""
Multi-Turn Cache Lineage Controller for the Turing Engine.

This module implements the `clean_base` multi-turn invariant adapted from kvloom (XKV arXiv:2608.20617). 
Key finding: naive re-injection of translated KV residuals onto a drifted cache collapses quality 
to 0 F1 by turn 3; `clean_base` (always recomputing the residual against the frozen original cache) 
maintains full quality indefinitely.
"""

import hashlib
import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
from typing_extensions import runtime_checkable, Protocol

__all__ = [
    "CacheLineageEntry",
    "CacheLineage",
    "LineageDriftError",
    "CleanBaseLineageBuffer",
    "TurnStrategy",
    "CleanBaseStrategy",
    "AppendOnlyStrategy",
    "NaiveStrategy",
]

def _hash_kv_tensors(kv_tensors: List[torch.Tensor]) -> str:
    """Computes deterministic hex digest over tensor buffers with fast In-VRAM and C++ pointer paths."""
    try:
        from ..kernels.triton_vram_hash import compute_fast_tensor_hash
        return compute_fast_tensor_hash(kv_tensors)
    except Exception:
        pass

    h = hashlib.blake2b()
    for tensor in kv_tensors:
        tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        h.update(tensor_bytes)
    return h.hexdigest()

@dataclass(frozen=True)
class CacheLineageEntry:
    turn_index: int
    strategy: str
    read_hash: str
    wrote_hash: str
    residual_norm: float
    timestamp: float

class LineageDriftError(Exception):
    def __init__(self, turn_index: int, expected_hash: str, actual_hash: str):
        self.turn_index = turn_index
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(f"Lineage drift at turn {turn_index}: expected {expected_hash}, got {actual_hash}")

class CacheLineage:
    def __init__(self) -> None:
        self._entries: List[CacheLineageEntry] = []
        
    def record(self, turn_index: int, strategy: str, read_kv: List[torch.Tensor], wrote_kv: List[torch.Tensor], residual_norm: float = 0.0) -> CacheLineageEntry:
        if turn_index != len(self._entries):
            raise LineageDriftError(turn_index, "N/A", "N/A")
            
        read_hash = _hash_kv_tensors(read_kv)
        wrote_hash = _hash_kv_tensors(wrote_kv)
        
        entry = CacheLineageEntry(
            turn_index=turn_index,
            strategy=strategy,
            read_hash=read_hash,
            wrote_hash=wrote_hash,
            residual_norm=residual_norm,
            timestamp=time.time()
        )
        self._entries.append(entry)
        return entry
        
    def verify_read(self, turn_index: int, cache_kv: List[torch.Tensor]) -> None:
        if turn_index >= len(self._entries):
            raise ValueError(f"Turn index {turn_index} not recorded yet")
            
        entry = self._entries[turn_index]
        actual_hash = _hash_kv_tensors(cache_kv)
        if actual_hash != entry.read_hash:
            raise LineageDriftError(turn_index, entry.read_hash, actual_hash)
            
    def entries(self) -> List[CacheLineageEntry]:
        return list(self._entries)
        
    def __len__(self) -> int:
        return len(self._entries)

class CleanBaseLineageBuffer:
    peak_live_caches: ClassVar[int] = 2
    
    def __init__(self, original_kv: List[torch.Tensor], use_svd_compression: bool = False):
        self._original_kv = [t.detach().clone() for t in original_kv]
        self.use_svd_compression = use_svd_compression
        
    def get_clean_base(self) -> List[torch.Tensor]:
        return [t.detach().clone() for t in self._original_kv]

@runtime_checkable
class TurnStrategy(Protocol):
    name: ClassVar[str]
    peak_live_caches: ClassVar[int]
    
    def cache_for_turn(self, turn_index: int, original: List[torch.Tensor], previous: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        ...
        
    def translates_on_turn(self, turn_index: int) -> bool:
        ...

class CleanBaseStrategy:
    name: ClassVar[str] = "clean_base"
    peak_live_caches: ClassVar[int] = 2
    
    def cache_for_turn(self, turn_index: int, original: List[torch.Tensor], previous: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        return original
        
    def translates_on_turn(self, turn_index: int) -> bool:
        return True

class AppendOnlyStrategy:
    name: ClassVar[str] = "append_only"
    peak_live_caches: ClassVar[int] = 1
    
    def cache_for_turn(self, turn_index: int, original: List[torch.Tensor], previous: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        if turn_index == 0:
            return original
        if previous is None:
            raise ValueError("previous must be provided for turns > 0")
        return previous
        
    def translates_on_turn(self, turn_index: int) -> bool:
        return turn_index == 0

class NaiveStrategy:
    name: ClassVar[str] = "naive"
    peak_live_caches: ClassVar[int] = 1
    
    def cache_for_turn(self, turn_index: int, original: List[torch.Tensor], previous: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        if turn_index == 0:
            return original
        if previous is None:
            raise ValueError("previous must be provided for turns > 0")
        return previous
        
    def translates_on_turn(self, turn_index: int) -> bool:
        return True
