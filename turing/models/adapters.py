"""
Epistemic Uncertainty Knowledge Gate & Multi-Tenant Rank-8 LoRA Adapters.
"""

from typing import Dict, Optional, Tuple
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.lora_b(self.lora_a(x))

class MultiTenantAdapterManager:
    """
    Manages dynamic loading and routing of tenant LoRA adapters.
    """
    def __init__(self, hidden_dim: int = 8192, rank: int = 8, alpha: float = 0.5):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.adapters: Dict[str, TenantLoRAAdapter] = {}

    def get_or_create_adapter(self, tenant_id: str, device: torch.device) -> TenantLoRAAdapter:
        if tenant_id not in self.adapters:
            self.adapters[tenant_id] = TenantLoRAAdapter(
                hidden_dim=self.hidden_dim, rank=self.rank, alpha=self.alpha
            ).to(device)
        return self.adapters[tenant_id]
