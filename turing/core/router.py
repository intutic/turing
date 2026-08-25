"""
Differentiable Gumbel-Softmax routers, dynamic Top-k entropy scaling, and DARE-O activation reuse.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubspaceStructuredRouter(nn.Module):
    """
    Differentiable Subspace Structured Router with Straight-Through Gumbel-Softmax
    and dynamic Top-k tile capacity scaling (16 to 64 tiles).
    """
    def __init__(
        self,
        hidden_dim: int = 8192,
        total_tiles: int = 112,
        min_tiles: int = 16,
        max_tiles: int = 64,
        tau: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_tiles = total_tiles
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.tau = tau

        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, total_tiles * 2)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_j: torch.Tensor, top_k_override: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Routes activations from layer boundary L/3.
        Returns:
            tile_mask: [Batch, total_tiles] binary mask (0.0 or 1.0)
            uncertainty: [Batch, 1] epistemic entropy scalar
        """
        # Sequence-pooling to extract global context
        if h_j.dim() == 3:
            ctx = torch.mean(h_j, dim=1)
        else:
            ctx = h_j
        ctx = self.norm(ctx)

        logits = self.gate_proj(ctx).view(-1, self.total_tiles, 2)
        uncertainty = torch.sigmoid(self.uncertainty_head(ctx)) # [Batch, 1]

        if self.training:
            # Straight-Through Gumbel-Softmax estimator
            mask_softmax = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            tile_mask = mask_softmax[:, :, 1]
        else:
            if top_k_override is not None:
                k_val = min(max(top_k_override, self.min_tiles), self.total_tiles)
                tile_scores = logits[:, :, 1] - logits[:, :, 0]
                _, topk_idx = torch.topk(tile_scores, k=k_val, dim=-1)
                tile_mask = torch.zeros(ctx.shape[0], self.total_tiles, device=ctx.device, dtype=torch.float32)
                tile_mask.scatter_(1, topk_idx, 1.0)
            else:
                # Dynamic top-k scaling based on sequence uncertainty
                mean_uncertainty = uncertainty.mean().item()
                k_dynamic = int(round(self.min_tiles + mean_uncertainty * (self.max_tiles - self.min_tiles)))
                k_dynamic = max(self.min_tiles, min(self.max_tiles, k_dynamic))

                tile_scores = logits[:, :, 1] - logits[:, :, 0]
                _, topk_idx = torch.topk(tile_scores, k=k_dynamic, dim=-1)
                tile_mask = torch.zeros(ctx.shape[0], self.total_tiles, device=ctx.device, dtype=torch.float32)
                tile_mask.scatter_(1, topk_idx, 1.0)

        return tile_mask, uncertainty

    def anneal_temperature(self, step: int, max_steps: int):
        """Anneals Gumbel temperature: tau(t) = max(0.2, 1.0 - 0.8 * (t / T))"""
        progress = min(1.0, step / max(1, max_steps))
        self.tau = max(0.2, 1.0 - progress * 0.8)


class DynamicEntropyRouter(nn.Module):
    """
    Evaluates epistemic entropy to switch between baseline sparse bitmask (e.g. 5 tiles)
    and burst expanded allocation (e.g. 8-12 tiles).
    """
    def __init__(self, hidden_dim: int = 768, threshold: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.entropy_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, SeqLen, HiddenDim]
        Returns (is_burst: torch.Tensor, confidence: torch.Tensor)
        """
        confidence = torch.sigmoid(self.entropy_proj(x.mean(dim=1)))
        is_burst = (confidence < self.threshold)
        return is_burst, confidence


class DAREOActivationReuse(nn.Module):
    """
    Drop And REscale Optimization (DARE-O) for Token-Level Activation Reuse.
    When token cosine similarity with previous step representation exceeds threshold (0.95),
    the heavy MLP execution is bypassed, reusing the prior layer representation.
    """
    def __init__(self, cosine_threshold: float = 0.95):
        super().__init__()
        self.cosine_threshold = cosine_threshold
        self.last_input: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None

    def should_reuse(self, current_input: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor]]:
        if self.last_input is None or self.last_output is None:
            return False, None

        # Check cosine similarity
        in_flat = current_input.view(-1)
        last_flat = self.last_input.view(-1)

        sim = F.cosine_similarity(in_flat.unsqueeze(0), last_flat.unsqueeze(0)).item()
        if sim >= self.cosine_threshold:
            return True, self.last_output
        return False, None

    def update_cache(self, current_input: torch.Tensor, current_output: torch.Tensor):
        self.last_input = current_input.detach().clone()
        self.last_output = current_output.detach().clone()

    def reset(self):
        self.last_input = None
        self.last_output = None


class AutonomicThresholdTuner(nn.Module):
    """
    Autonomic Threshold & Temperature Tuner using Fused In-SRAM Adam (Fused High-Performance Kernel).
    Dynamically tunes sparsity threshold and Gumbel temperature to maintain target serving latency.
    """
    def __init__(
        self,
        target_latency_ms: float = 6.5,
        initial_threshold: float = 0.5,
        initial_tau: float = 1.0,
        lr: float = 0.01
    ):
        super().__init__()
        self.target_latency_ms = target_latency_ms
        self.lr = lr
        self.timestep = 0

        # Parameters to tune: [threshold, tau]
        self.params = torch.tensor([initial_threshold, initial_tau], dtype=torch.float32)
        self.exp_avg_m = torch.zeros(2, dtype=torch.float32)
        self.exp_avg_v = torch.zeros(2, dtype=torch.float32)

    def update_from_latency_observation(self, observed_latency_ms: float):
        """
        Updates parameters based on error between observed and target latency.
        """
        self.timestep += 1
        error = observed_latency_ms - self.target_latency_ms
        # Gradient: higher latency requires higher threshold (more sparsity) and lower temperature
        grad = torch.tensor([-error * 0.1, error * 0.05], dtype=torch.float32)

        try:
            from turing import turing_csrc
            p_np = self.params.numpy()
            g_np = grad.numpy()
            m_np = self.exp_avg_m.numpy()
            v_np = self.exp_avg_v.numpy()
            turing_csrc.fused_adam_step(
                p_np, g_np, m_np, v_np,
                self.lr, 0.9, 0.999, 1e-8, self.timestep
            )
        except Exception:
            # Fallback scalar Adam update
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            bc1 = 1.0 - (beta1 ** self.timestep)
            bc2 = 1.0 - (beta2 ** self.timestep)

            self.exp_avg_m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            self.exp_avg_v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            m_hat = self.exp_avg_m / bc1
            v_hat = self.exp_avg_v / bc2
            self.params.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-self.lr)

        # Clamp parameters to valid ranges
        self.params[0].clamp_(0.1, 0.9)
        self.params[1].clamp_(0.2, 2.0)

    @property
    def current_threshold(self) -> float:
        return float(self.params[0].item())

    @property
    def current_tau(self) -> float:
        return float(self.params[1].item())

