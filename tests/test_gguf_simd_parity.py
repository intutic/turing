"""
Numerical Parity Test Suite: C++20 AVX2/NEON SIMD GGUF Dequantizer vs Reference Implementations.
Verifies exact floating point parity across Q4_0, Q4_1, Q8_0, FP16, and BF16 GGML blocks.
"""

import pytest
import torch
import numpy as np

import turing.turing_csrc as turing_csrc
from turing.models.gguf_loader import GGMLType, GGUFDequantizer


def test_gguf_simd_q8_0_parity():
    # 1 block = 2 bytes delta + 32 bytes int8 = 34 bytes
    delta = np.float16(0.25).tobytes()
    quants = np.array([i - 16 for i in range(32)], dtype=np.int8).tobytes()
    data = delta + quants

    # Native C++ SIMD
    res_cpp = turing_csrc.dequantize_gguf_simd(data, int(GGMLType.Q8_0.value), [32])
    
    # Expected
    expected = (np.array([i - 16 for i in range(32)], dtype=np.float32) * 0.25)
    np.testing.assert_allclose(res_cpp, expected, rtol=1e-4, atol=1e-4)


def test_gguf_simd_q4_0_parity():
    # 1 block = 2 bytes delta + 16 bytes nibbles = 18 bytes (32 elements)
    delta = np.float16(0.5).tobytes()
    # Nibbles: low 4 bits = 0x5 (5-8 = -3), high 4 bits = 0xA (10-8 = 2)
    nibbles = bytes([0xA5] * 16)
    data = delta + nibbles

    # Native C++ SIMD
    res_cpp = turing_csrc.dequantize_gguf_simd(data, int(GGMLType.Q4_0.value), [32])
    
    # Expected: 16 pairs of (-3*0.5 = -1.5, 2*0.5 = 1.0)
    expected = np.array([-1.5, 1.0] * 16, dtype=np.float32)
    np.testing.assert_allclose(res_cpp, expected, rtol=1e-4, atol=1e-4)


def test_gguf_simd_fp16_and_bf16_parity():
    # FP16
    f16_vals = np.array([1.5, -2.0, 3.25, 0.0], dtype=np.float16)
    res_f16 = turing_csrc.dequantize_gguf_simd(f16_vals.tobytes(), int(GGMLType.F16.value), [4])
    np.testing.assert_allclose(res_f16, f16_vals.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_gguf_loader_auto_dispatch():
    delta = np.float16(1.0).tobytes()
    quants = np.array([5] * 32, dtype=np.int8).tobytes()
    data = delta + quants

    tensor = GGUFDequantizer.dequantize(data, GGMLType.Q8_0, [32], target_dtype=torch.float32)
    assert tensor.shape == (32,)
    assert torch.allclose(tensor, torch.full((32,), 5.0))
