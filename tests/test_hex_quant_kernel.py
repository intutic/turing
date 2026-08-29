"""
Unit tests for Hexagonal Codebook Quantizer and Kernels.
Verifies BMU search, codebook distance metrics, and quantization integrity.
"""

import pytest
import torch
import torch.nn.functional as F

from turing.core.hex_quant import HexagonalSubspaceQuantizer


def test_hex_quantizer_bmu_search():
    torch.manual_seed(42)
    quantizer = HexagonalSubspaceQuantizer(codebook_dim=64, grid_width=8, grid_height=8)

    # Input activations
    x = torch.randn(16, 64, dtype=torch.float32)
    quantized, bmu_indices = quantizer.quantize_subspace(x)

    assert quantized.shape == (16, 64)
    assert bmu_indices.shape == (16,)
    assert bmu_indices.min().item() >= 0
    assert bmu_indices.max().item() < 64

    # Quantized vectors should match codebook prototypes at selected indices
    torch.testing.assert_close(quantized, quantizer.codebook[bmu_indices])


def test_hex_neighborhood_distance():
    quantizer = HexagonalSubspaceQuantizer(codebook_dim=64, grid_width=8, grid_height=8)
    d = quantizer.hex_neighborhood_distance(0, 1) # Adjacent cells
    assert d > 0.0
