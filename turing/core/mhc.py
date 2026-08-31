"""
Manifold-Constrained Hyper-Connections (mHC) for Subspace Recirculation (DeepSeek V4 & arXiv:2512.24880).
Projects multi-stream residual mappings onto the Birkhoff Polytope of Doubly Stochastic Matrices
using Sinkhorn-Knopp normalization for provably non-expansive, stable belief state recurrence.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

from turing.kernels.birkhoff_cuda import birkhoff_project_cuda

class BirkhoffManifoldProjector:
    """
    Projects arbitrary real square matrices onto the Birkhoff Polytope of Doubly Stochastic Matrices:
    B_n = { M in R^{n x n} : M_ij >= 0, sum_i M_ij = 1, sum_j M_ij = 1 }
    Using iterative Sinkhorn-Knopp algorithm.
    """
    @staticmethod
    def project(matrix: torch.Tensor, num_iterations: int = 20, eps: float = 1e-6) -> torch.Tensor:
        """
        matrix: [..., N, N]
        Applies exp(M / tau) and alternating row/column normalizations.
        """
        if matrix.is_cuda:
            return birkhoff_project_cuda(matrix, num_iterations, eps)

        if HAS_CSRC:
            device = matrix.device
            mat_cpu = matrix.detach().to(torch.float32).cpu().contiguous()
            res_np = turing_csrc.birkhoff_project(mat_cpu.numpy(), num_iterations, eps)
            return torch.from_numpy(res_np).to(dtype=matrix.dtype, device=device)

        # Fallback to pure PyTorch
        p = torch.exp(matrix - matrix.max(dim=-1, keepdim=True)[0]) + eps

        for _ in range(num_iterations):
            # Row normalization: sum_j P_ij = 1
            row_sum = p.sum(dim=-1, keepdim=True) + eps
            p = p / row_sum

            # Column normalization: sum_i P_ij = 1
            col_sum = p.sum(dim=-2, keepdim=True) + eps
            p = p / col_sum

        return p


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Block:
    Maintains n parallel residual streams (default n=4) to widen information throughput
    without increasing transformer hidden dimension.

    1. Pre-Mapping: Combines n residual streams into 1 layer input hidden state:
       x_layer = sum_i (alpha_i * stream_i), where alpha >= 0, sum(alpha) = 1.
    2. Res-Mapping: Doubly stochastic Birkhoff mixing matrix H_res in B_n across parallel streams:
       streams_mixed = streams @ H_res.
    3. Post-Mapping: Distributes layer update back across n residual streams:
       stream_i = stream_i + beta_i * layer_update, where beta_i in [0, 1].
    """
    def __init__(self, hidden_dim: int, num_streams: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_streams = num_streams

        # Unconstrained parameter matrix for Doubly Stochastic Res-Mapping
        # Shape: [num_streams, num_streams]
        self.raw_res_map = nn.Parameter(torch.eye(num_streams) + 0.1 * torch.randn(num_streams, num_streams))

        # Pre-mapping mixing coefficients (softmax-normalized)
        self.raw_pre_weights = nn.Parameter(torch.ones(num_streams) / num_streams)

        # Post-mapping gating coefficients (sigmoid-normalized)
        self.raw_post_weights = nn.Parameter(torch.zeros(num_streams))

    def get_doubly_stochastic_res_map(self) -> torch.Tensor:
        """
        Returns projected doubly stochastic mixing matrix.
        """
        return BirkhoffManifoldProjector.project(self.raw_res_map)

    def pre_map(self, streams: torch.Tensor) -> torch.Tensor:
        """
        streams: [Batch, SeqLen, num_streams, hidden_dim]
        Returns: [Batch, SeqLen, hidden_dim]
        """
        weights = F.softmax(self.raw_pre_weights, dim=0) # [num_streams]
        # Weighted sum across streams: [Batch, SeqLen, hidden_dim]
        return torch.einsum('bsnd,n->bsd', streams, weights)

    def res_map(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Mixes parallel residual streams via doubly stochastic matrix.
        streams: [Batch, SeqLen, num_streams, hidden_dim]
        Returns: [Batch, SeqLen, num_streams, hidden_dim]
        """
        h_res = self.get_doubly_stochastic_res_map() # [num_streams, num_streams]
        # streams_mixed_j = sum_i (streams_i * H_ij)
        return torch.einsum('bsid,ij->bsjd', streams, h_res)

    def post_map(self, streams_mixed: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """
        Distributes layer update back across mixed streams.
        streams_mixed: [Batch, SeqLen, num_streams, hidden_dim]
        layer_output: [Batch, SeqLen, hidden_dim]
        Returns: [Batch, SeqLen, num_streams, hidden_dim]
        """
        beta = torch.sigmoid(self.raw_post_weights) # [num_streams]
        return streams_mixed + torch.einsum('bsd,n->bsnd', layer_output, beta)

    def forward(
        self,
        streams: torch.Tensor,
        layer_fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one full mHC step around a transformer layer.
        streams: [Batch, SeqLen, num_streams, hidden_dim]
        layer_fn: Callable taking [Batch, SeqLen, hidden_dim] -> [Batch, SeqLen, hidden_dim]
        Returns: (new_streams, layer_output)
        """
        # Step 1: Pre-map streams to layer input
        x_in = self.pre_map(streams)

        # Step 2: Evaluate transformer layer
        layer_out = layer_fn(x_in)

        if streams.is_cuda and self.num_streams == 4:
            try:
                from ..kernels.triton_mhc_fuse import mhc_stream_mix_cuda
                h_res = self.get_doubly_stochastic_res_map()
                new_streams = mhc_stream_mix_cuda(streams, layer_out, h_res, self.raw_post_weights)
                return new_streams, layer_out
            except Exception:
                pass

        if not streams.is_cuda and self.num_streams == 4:
            try:
                import turing.turing_csrc as turing_csrc
                batch, seq_len, _, hidden_dim = streams.shape
                s_np = streams.view(-1, 4, hidden_dim).detach().to(torch.float32).cpu().contiguous().numpy()
                lup_np = layer_out.view(-1, hidden_dim).detach().to(torch.float32).cpu().contiguous().numpy()
                alpha_np = F.softmax(self.raw_pre_weights, dim=0).detach().to(torch.float32).cpu().contiguous().numpy()
                h_res = self.get_doubly_stochastic_res_map()
                h_np = h_res.detach().to(torch.float32).cpu().contiguous().numpy()
                beta_np = torch.sigmoid(self.raw_post_weights).detach().to(torch.float32).cpu().contiguous().numpy()

                _, s_out_np = turing_csrc.mhc_4stream_simd_cpu(s_np, lup_np, alpha_np, h_np, beta_np)
                new_streams = torch.from_numpy(s_out_np).to(device=streams.device, dtype=streams.dtype).view(batch, seq_len, 4, hidden_dim)
                return new_streams, layer_out
            except Exception:
                pass

        # Step 3: Doubly stochastic residual stream mixing
        streams_mixed = self.res_map(streams)

        # Step 4: Post-map layer output into streams
        new_streams = self.post_map(streams_mixed, layer_out)

        return new_streams, layer_out


