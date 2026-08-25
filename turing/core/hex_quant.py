"""
Hexagonal Spatial Codebook Quantizer & Parallel BMU Search.
High-Performance Hexagonal Lattice Quantization & Metric Evaluation.
Enables non-Euclidean 6-neighbor manifold clustering for high-fidelity codebook generation.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

class HexagonalSubspaceQuantizer:
    """
    Quantizes activation vectors onto a 2D Hexagonal Topological Codebook.
    Coordinates on hexagonal grid: (u, v) with distance d_hex = sqrt(du^2 + dv^2 + du*dv).
    """
    def __init__(
        self,
        codebook_dim: int = 64,
        grid_width: int = 8,
        grid_height: int = 8,
        device: torch.device = torch.device("cpu")
    ):
        self.codebook_dim = codebook_dim
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.total_cells = grid_width * grid_height
        self.device = device

        # Initialize codebook prototypes [total_cells, codebook_dim]
        self.codebook = torch.randn(self.total_cells, codebook_dim, device=device)
        self.codebook = nn.functional.normalize(self.codebook, p=2, dim=-1)

        # Precompute hexagonal grid coordinates
        self.u_coords = torch.zeros(self.total_cells, device=device)
        self.v_coords = torch.zeros(self.total_cells, device=device)
        for r in range(grid_height):
            for c in range(grid_width):
                idx = r * grid_width + c
                self.u_coords[idx] = float(c)
                self.v_coords[idx] = float(r)

    def find_bmu_parallel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel Best Matching Unit (BMU) search:
        x: [batch, codebook_dim]
        Returns: (bmu_indices [batch], min_distances [batch])
        """
        x_norm = nn.functional.normalize(x, p=2, dim=-1)
        
        # Parallel cosine similarity: [batch, total_cells]
        sim = torch.matmul(x_norm, self.codebook.t())
        dists = 1.0 - sim

        min_dists, bmu_indices = torch.min(dists, dim=-1)
        return bmu_indices, min_dists

    def quantize_subspace(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes activations to nearest codebook prototype.
        """
        try:
            import turing.turing_csrc as turing_csrc
            HAS_CSRC = True
        except ImportError:
            try:
                import turing_csrc
                HAS_CSRC = True
            except ImportError:
                HAS_CSRC = False

        if HAS_CSRC and not activations.is_cuda:
            act_np = activations.detach().to(torch.float32).cpu().contiguous().numpy()
            cd_np = self.codebook.detach().to(torch.float32).cpu().contiguous().numpy()
            bmu_np, q_np = turing_csrc.hex_quantize_activations(act_np, cd_np)
            dev = activations.device
            return (
                torch.from_numpy(q_np).to(device=dev, dtype=activations.dtype),
                torch.from_numpy(bmu_np).to(device=dev, dtype=torch.long)
            )

        bmu_indices, _ = self.find_bmu_parallel(activations)
        quantized = self.codebook[bmu_indices]
        return quantized, bmu_indices

    def hex_neighborhood_distance(self, bmu_idx1: int, bmu_idx2: int) -> float:
        """
        Hexagonal metric distance: d = sqrt(du^2 + dv^2 + du*dv)
        """
        du = self.u_coords[bmu_idx1] - self.u_coords[bmu_idx2]
        dv = self.v_coords[bmu_idx1] - self.v_coords[bmu_idx2]
        return float(torch.sqrt(du*du + dv*dv + du*dv).item())

