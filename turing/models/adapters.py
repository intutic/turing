"""
Epistemic Uncertainty Knowledge Gate & Multi-Tenant Rank-8 LoRA Adapters.
"""

from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn

class UncertaintyKnowledgeGate(nn.Module):
    """
    Control-plane governance evaluating epistemic uncertainty for online knowledge ingestion.
    """
    def __init__(self, hidden_dim: int = 8192, threshold: float = 0.80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def evaluate_fact_confidence(self, hidden_representation: torch.Tensor) -> Tuple[bool, float]:
        """
        Evaluates context hidden representation.
        Returns: (is_admissible, confidence_score)
        """
        score = self.classifier(hidden_representation.mean(dim=0)).item()
        return (score >= self.threshold), score

class TenantLoRAAdapter(nn.Module):
    """
    Multi-Tenant Rank-8 LoRA Adapter for request-level tenant isolation.
    y = x + alpha * (x @ W_A @ W_B)
    """
    def __init__(self, hidden_dim: int = 8192, rank: int = 8, alpha: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha

        self.lora_a = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if base_weight is not None:
            if x.is_cuda:
                try:
                    from ..kernels.triton_fused_lora import fused_lora_gemv_cuda
                    return fused_lora_gemv_cuda(x, base_weight, self.lora_a.weight.t(), self.lora_b.weight.t(), alpha=self.alpha)
                except Exception:
                    pass
            else:
                try:
                    import turing.turing_csrc as turing_csrc
                    x_cpu = x.detach().to(torch.float32).cpu().contiguous().numpy()
                    wb_cpu = base_weight.detach().to(torch.float32).cpu().contiguous().numpy()
                    wa_cpu = self.lora_a.weight.t().detach().to(torch.float32).cpu().contiguous().numpy()
                    wb2_cpu = self.lora_b.weight.t().detach().to(torch.float32).cpu().contiguous().numpy()
                    out_np = turing_csrc.fused_lora_gemv_cpu(x_cpu, wb_cpu, wa_cpu, wb2_cpu, float(self.alpha))
                    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)
                except Exception:
                    pass
            return torch.matmul(x, base_weight) + self.alpha * self.lora_b(self.lora_a(x))

        return x + self.alpha * self.lora_b(self.lora_a(x))



class GPULRUAdapterCache:
    """
    Intrusive On-GPU LRU Cache for Multi-Tenant LoRA Adapters.
    - Holds N resident adapter slots on GPU VRAM.
    - Backed by pinned host DRAM pool for 100+ tenant adapters.
    - Adapters copied in from host memory on a miss.
    - Zero base model weight duplication.
    """
    def __init__(
        self,
        hidden_dim: int = 8192,
        rank: int = 8,
        alpha: float = 0.5,
        capacity: int = 32,
        device: Union[str, torch.device] = "cpu"
    ):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.capacity = capacity
        self.device = torch.device(device)

        # On-GPU resident active slots: tenant_id -> TenantLoRAAdapter (OrderedDict for O(1) LRU)
        from collections import OrderedDict
        self.gpu_slots: OrderedDict[str, TenantLoRAAdapter] = OrderedDict()

        # Pinned host DRAM backing store: tenant_id -> (weight_a_pinned, weight_b_pinned)
        self.host_store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def register_host_adapter(
        self,
        tenant_id: str,
        lora_a_weight: torch.Tensor,
        lora_b_weight: torch.Tensor
    ) -> None:
        """
        Registers an adapter's weights (A [rank, hidden], B [hidden, rank]) into pinned host DRAM.
        """
        if lora_a_weight.shape != (self.rank, self.hidden_dim) or lora_b_weight.shape != (self.hidden_dim, self.rank):
            raise ValueError(f"LoRA weights must be A [{self.rank}, {self.hidden_dim}] and B [{self.hidden_dim}, "
                             f"{self.rank}], got {tuple(lora_a_weight.shape)} and {tuple(lora_b_weight.shape)}")

        # Pin memory on host for high-speed DMA over PCIe
        if not lora_a_weight.is_pinned() and torch.cuda.is_available() and self.device.type == "cuda":
            try:
                lora_a_weight = lora_a_weight.pin_memory()
                lora_b_weight = lora_b_weight.pin_memory()
            except Exception:
                pass

        self.host_store[tenant_id] = (lora_a_weight, lora_b_weight)

    def get_adapter(self, tenant_id: str) -> TenantLoRAAdapter:
        """
        Retrieves adapter from on-GPU LRU cache in O(1) time.
        On a miss, evicts the LRU adapter and copies the registered weights in from host memory.
        """
        if tenant_id in self.gpu_slots:
            self.hits += 1
            self.gpu_slots.move_to_end(tenant_id)
            return self.gpu_slots[tenant_id]

        self.misses += 1

        # Check if eviction needed (O(1) popitem)
        if len(self.gpu_slots) >= self.capacity:
            evicted_tenant, _ = self.gpu_slots.popitem(last=False)
            self.evictions += 1

        # Instantiate on GPU
        adapter = TenantLoRAAdapter(
            hidden_dim=self.hidden_dim,
            rank=self.rank,
            alpha=self.alpha
        ).to(self.device)

        if tenant_id not in self.host_store:
            raise KeyError(f"no LoRA adapter registered for tenant '{tenant_id}'")
        w_a, w_b = self.host_store[tenant_id]
        adapter.lora_a.weight.data.copy_(w_a, non_blocking=True)
        adapter.lora_b.weight.data.copy_(w_b, non_blocking=True)

        self.gpu_slots[tenant_id] = adapter
        return adapter

    def apply_adapter(self, x: torch.Tensor, tenant_id: str) -> torch.Tensor:
        """
        Applies tenant adapter in-place on hidden states x.
        """
        adapter = self.get_adapter(tenant_id)
        return adapter(x)

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 1.0


class MultiTenantAdapterManager:
    """
    Manages dynamic loading, hot-swapping, and routing of multi-tenant LoRA adapters.
    """
    def __init__(
        self,
        hidden_dim: int = 8192,
        rank: int = 8,
        alpha: float = 0.5,
        cache_capacity: int = 32,
        device: Union[str, torch.device] = "cpu"
    ):
        self.cache = GPULRUAdapterCache(
            hidden_dim=hidden_dim,
            rank=rank,
            alpha=alpha,
            capacity=cache_capacity,
            device=device
        )

    def get_or_create_adapter(self, tenant_id: str, device: Optional[torch.device] = None) -> TenantLoRAAdapter:
        """The tenant's registered adapter (cache.register_host_adapter); KeyError if it has none."""
        return self.cache.get_adapter(tenant_id)

    def forward_tenant(self, x: torch.Tensor, tenant_id: str) -> torch.Tensor:
        return self.cache.apply_adapter(x, tenant_id)

