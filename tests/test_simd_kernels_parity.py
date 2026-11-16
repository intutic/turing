"""
Parity of the turing_simd.hpp kernels (AVX2+FMA on x86_64, NEON on arm64, scalar elsewhere) against NumPy, at
lengths that exercise every vector width and the scalar tails.
"""

import platform

import numpy as np
import pytest

csrc = pytest.importorskip("turing.turing_csrc")

SIZES = [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 257, 1024]
rng = np.random.default_rng(7)


def test_backend_matches_machine():
    machine = platform.machine().lower()
    expected = {"arm64": "neon", "aarch64": "neon", "x86_64": "avx2", "amd64": "avx2"}.get(machine)
    if expected is None:
        pytest.skip(f"no SIMD path for {machine}")
    assert csrc.simd_backend() == expected


@pytest.mark.parametrize("n", SIZES)
def test_dot_and_gemv(n):
    a, b = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    assert abs(csrc.simd_dot_f32(a, b) - ref) <= 1e-5 * np.abs(a * b).sum() + 1e-6

    w = rng.standard_normal((5, n)).astype(np.float32)
    np.testing.assert_allclose(csrc.simd_gemv_f32(w, a), w.astype(np.float64) @ a, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(csrc.simd_gemv_fp32_tile(a, w), w.astype(np.float64) @ a, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", SIZES)
def test_softmax_rmsnorm_silu(n):
    x = (rng.standard_normal(n) * 30).astype(np.float32)  # wide spread: exp underflows for most entries
    e = np.exp(x.astype(np.float64) - x.max())
    np.testing.assert_allclose(csrc.simd_softmax_f32(x), e / e.sum(), rtol=1e-5, atol=1e-7)

    w = rng.standard_normal(n).astype(np.float32)
    x64 = x.astype(np.float64)
    ref = x64 / np.sqrt(np.mean(x64 ** 2) + 1e-5) * w
    np.testing.assert_allclose(csrc.simd_rmsnorm_f32(x, w, 1e-5), ref, rtol=1e-5, atol=1e-5)

    g = np.linspace(-120, 120, n).astype(np.float32)  # past the exp clamp at +-88 on both sides
    up = rng.standard_normal(n).astype(np.float32)
    g64 = g.astype(np.float64)
    np.testing.assert_allclose(csrc.simd_silu_mul_f32(g, up), g64 / (1 + np.exp(-g64)) * up, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_uint8_code_kernels(n):
    a = rng.standard_normal(n).astype(np.float32)
    codes = rng.integers(0, 256, n, dtype=np.uint8)
    ref = float(np.dot(a.astype(np.float64), codes))
    assert abs(csrc.simd_dot_f32_u8(a, codes) - ref) <= 1e-5 * np.abs(a * codes).sum() + 1e-5

    y = rng.standard_normal(n).astype(np.float32)
    np.testing.assert_allclose(csrc.simd_axpby_u8_f32(0.25, -1.5, codes, y), y + 0.25 * codes.astype(np.float64) - 1.5,
                               rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("n", SIZES)
def test_gemv_int8_tile_exact(n):
    # |u8 * s8| pairs stay below int16 saturation, where every path (AVX2 maddubs included) is exact
    x = rng.integers(0, 128, n, dtype=np.uint8)
    w = rng.integers(-127, 128, (6, n), dtype=np.int8)
    np.testing.assert_array_equal(csrc.simd_gemv_int8_tile(x, w), w.astype(np.int64) @ x.astype(np.int64))
