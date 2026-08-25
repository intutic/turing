"""
Epistemic Uncertainty & Exploration Gate.
Calculates token predictive entropy to distinguish between certain execution vs. epistemic exploration.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class EpistemicUncertaintyGate:
    """
    Epistemic Uncertainty Gate:
    Evaluates Shannon entropy H(P_t) across model logits during agent deliberation.
    If entropy exceeds the threshold, flags high epistemic uncertainty to prevent hallucinations
    and encourage active inquiry/verification against the world model.
    """
    def __init__(self, uncertainty_threshold: float = 2.5):
        self.uncertainty_threshold = uncertainty_threshold

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Calculates Shannon entropy in nats from output logits: H(p) = - sum(p * log(p)).
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # Take last token

        try:
            import turing.turing_csrc as turing_csrc
            HAS_CSRC = True
        except ImportError:
            try:
                import turing_csrc
                HAS_CSRC = True
            except ImportError:
                HAS_CSRC = False

        if HAS_CSRC and not logits.is_cuda:
            l_np = logits.detach().to(torch.float32).cpu().contiguous().numpy()
            ent_np = turing_csrc.compute_shannon_entropy(l_np)
            return float(ent_np.mean())

        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        return entropy

    def evaluate_step_uncertainty(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        Returns uncertainty diagnostics and gating decision.
        """
        entropy = self.calculate_entropy(logits)
        is_uncertain = entropy > self.uncertainty_threshold
        return {
            "entropy": round(entropy, 4),
            "threshold": self.uncertainty_threshold,
            "is_uncertain": is_uncertain,
            "action": "TRIGGER_EPISTEMIC_EXPLORATION" if is_uncertain else "CONFIDENT_EXECUTION"
        }
