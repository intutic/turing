"""
Deep-to-shallow activation recirculation through a low-rank INT8 projection (experimental, off by default).
"""

from typing import Optional
import torch
import torch.nn as nn

class SubspaceRecirculation(nn.Module):
    """
    Adds a deep layer's hidden state, projected to `rank` dimensions, INT8-rounded and projected back, to a shallow
    layer's hidden state. The projection basis is the top right singular vectors of a synthetic random matrix with
    decaying column scale, not of the model's activations, and the model was not trained with this path: enabling it
    (ModelConfig.enable_recirculation) changes the model's output. use_birkhoff clamps the energy ratio at 1.
    """
    def __init__(self, hidden_dim: int = 8192, rank: int = 64, alpha: float = 0.15, use_birkhoff: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.use_birkhoff = use_birkhoff
        self.register_buffer("u_proj", nn.Parameter(torch.empty(hidden_dim, rank), requires_grad=False))
        self._init_weights()

    def _init_weights(self):
        decay = torch.exp(-3.5 * torch.arange(self.hidden_dim, dtype=torch.float32) / self.hidden_dim)
        calib = torch.randn(min(4096, self.hidden_dim * 2), self.hidden_dim, dtype=torch.float32) * decay
        _, _, v_t = torch.linalg.svd(calib, full_matrices=False)
        self.u_proj.copy_(v_t[:self.rank, :].t().contiguous())

    def forward(self, h_shallow: torch.Tensor, h_deep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes recirculation:
        h_shallow <- h_shallow + alpha * (||h_deep||_2 / (||h_shallow||_2 + eps)) * (h_deep @ U @ U^T)
        """
        if h_deep is None:
            return h_shallow

        dtype = h_shallow.dtype
        eps = 1e-6

        # Compute L2 norm ratio for energy matching
        norm_shallow = torch.linalg.vector_norm(h_shallow, dim=-1, keepdim=True) + eps
        norm_deep = torch.linalg.vector_norm(h_deep, dim=-1, keepdim=True)
        scale_ratio = norm_deep / norm_shallow

        if self.use_birkhoff:
            # Constrain energy scale to non-expansive <= 1.0
            scale_ratio = torch.clamp(scale_ratio, max=1.0)

        # Project deep belief state into Rank-64 subspace
        u = self.u_proj.to(dtype=dtype, device=h_shallow.device)
        s_sub = torch.matmul(h_deep, u) # [..., rank]

        # Symmetric INT8 compression simulation for bandwidth conservation
        abs_max = torch.amax(torch.abs(s_sub), dim=-1, keepdim=True).clamp(min=1e-5)
        s_scale = abs_max / 127.0
        s_int8 = torch.clamp(torch.round(s_sub / s_scale), -128, 127)
        s_dequant = s_int8 * s_scale

        # Reconstruct back to full dimension
        h_reconstructed = torch.matmul(s_dequant, u.t())

        # Recirculation mixing
        return h_shallow + self.alpha * scale_ratio * h_reconstructed
