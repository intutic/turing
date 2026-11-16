#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <memory>

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
#include <immintrin.h>
#define TURING_HAS_AVX2 1
#if defined(__FMA__) || defined(_MSC_VER)
#define TURING_HAS_AVX2_FMA 1
#endif
#endif
#endif

// NEON is baseline on every 64-bit ARM target (Apple Silicon, Graviton, Ampere), so no flag is needed there.
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define TURING_HAS_NEON 1
#endif

namespace turing {

constexpr int DEFAULT_TILE_SIZE = 256;
constexpr size_t CACHE_LINE_ALIGNMENT = 64;

// 64-byte aligned memory allocation helper
inline void* aligned_alloc_64(size_t bytes) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, CACHE_LINE_ALIGNMENT);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, CACHE_LINE_ALIGNMENT, bytes) != 0) {
        throw std::bad_alloc();
    }
#endif
    return ptr;
}

inline void aligned_free_64(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// FP32 Sparse Pointer-Skipping GEMV Kernel
// ============================================================================
inline void gemv_fp32_tile(
    const float* __restrict__ input_tile,
    const float* __restrict__ weight_tile,
    float* __restrict__ output,
    int out_features,
    int tile_size
) {
#if defined(TURING_HAS_AVX2_FMA)
    for (int r = 0; r < out_features; ++r) {
        const float* w_row = weight_tile + (r * tile_size);
        __m256 accum = _mm256_setzero_ps();

        int c = 0;
        for (; c + 31 < tile_size; c += 32) {
            accum = _mm256_fmadd_ps(_mm256_loadu_ps(input_tile + c),      _mm256_loadu_ps(w_row + c),      accum);
            accum = _mm256_fmadd_ps(_mm256_loadu_ps(input_tile + c + 8),  _mm256_loadu_ps(w_row + c + 8),  accum);
            accum = _mm256_fmadd_ps(_mm256_loadu_ps(input_tile + c + 16), _mm256_loadu_ps(w_row + c + 16), accum);
            accum = _mm256_fmadd_ps(_mm256_loadu_ps(input_tile + c + 24), _mm256_loadu_ps(w_row + c + 24), accum);
        }
        for (; c + 7 < tile_size; c += 8) {
            accum = _mm256_fmadd_ps(_mm256_loadu_ps(input_tile + c), _mm256_loadu_ps(w_row + c), accum);
        }

        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, accum);
        float sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

        for (; c < tile_size; ++c) {
            sum += input_tile[c] * w_row[c];
        }
        output[r] += sum;
    }
#elif defined(TURING_HAS_NEON)
    for (int r = 0; r < out_features; ++r) {
        const float* w_row = weight_tile + (r * tile_size);
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        int c = 0;
        for (; c + 15 < tile_size; c += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(input_tile + c),      vld1q_f32(w_row + c));
            a1 = vfmaq_f32(a1, vld1q_f32(input_tile + c + 4),  vld1q_f32(w_row + c + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(input_tile + c + 8),  vld1q_f32(w_row + c + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(input_tile + c + 12), vld1q_f32(w_row + c + 12));
        }
        for (; c + 3 < tile_size; c += 4) {
            a0 = vfmaq_f32(a0, vld1q_f32(input_tile + c), vld1q_f32(w_row + c));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; c < tile_size; ++c) {
            sum += input_tile[c] * w_row[c];
        }
        output[r] += sum;
    }
#else
    // Portable fallback
    for (int r = 0; r < out_features; ++r) {
        const float* w_row = weight_tile + (r * tile_size);
        float sum = 0.0f;
        for (int c = 0; c < tile_size; ++c) {
            sum += input_tile[c] * w_row[c];
        }
        output[r] += sum;
    }
#endif
}

// ============================================================================
// INT8 Quantized Sparse GEMV Kernel (_mm256_maddubs_epi16)
// ============================================================================
inline void gemv_int8_tile(
    const uint8_t* __restrict__ input_tile,
    const int8_t* __restrict__ weight_tile,
    int32_t* __restrict__ accum_out,
    int out_features,
    int tile_size
) {
#if defined(TURING_HAS_AVX2)
    const __m256i ones_16 = _mm256_set1_epi16(1);

    for (int r = 0; r < out_features; ++r) {
        const int8_t* w_row = weight_tile + (r * tile_size);
        __m256i accum_32 = _mm256_setzero_si256();

        int c = 0;
        for (; c + 31 < tile_size; c += 32) {
            __m256i a_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_tile + c));
            __m256i w_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row + c));
            __m256i dot_16 = _mm256_maddubs_epi16(a_u8, w_i8);
            __m256i dot_32 = _mm256_madd_epi16(dot_16, ones_16);
            accum_32 = _mm256_add_epi32(accum_32, dot_32);
        }

        alignas(32) int32_t buf[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), accum_32);
        int32_t sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

        for (; c < tile_size; ++c) {
            sum += static_cast<int32_t>(input_tile[c]) * static_cast<int32_t>(w_row[c]);
        }
        accum_out[r] += sum;
    }
#elif defined(TURING_HAS_NEON)
    // Exact u8 x s8 products (no pairwise int16 saturation): widen both to int16, multiply-accumulate into int32.
    for (int r = 0; r < out_features; ++r) {
        const int8_t* w_row = weight_tile + (r * tile_size);
        int32x4_t acc = vdupq_n_s32(0);
        int c = 0;
        for (; c + 15 < tile_size; c += 16) {
            uint8x16_t a = vld1q_u8(input_tile + c);
            int8x16_t w = vld1q_s8(w_row + c);
            int16x8_t a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
            int16x8_t a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
            int16x8_t w_lo = vmovl_s8(vget_low_s8(w));
            int16x8_t w_hi = vmovl_s8(vget_high_s8(w));
            acc = vmlal_s16(acc, vget_low_s16(a_lo), vget_low_s16(w_lo));
            acc = vmlal_s16(acc, vget_high_s16(a_lo), vget_high_s16(w_lo));
            acc = vmlal_s16(acc, vget_low_s16(a_hi), vget_low_s16(w_hi));
            acc = vmlal_s16(acc, vget_high_s16(a_hi), vget_high_s16(w_hi));
        }
        int32_t sum = vaddvq_s32(acc);
        for (; c < tile_size; ++c) {
            sum += static_cast<int32_t>(input_tile[c]) * static_cast<int32_t>(w_row[c]);
        }
        accum_out[r] += sum;
    }
#else
    // Portable fallback
    for (int r = 0; r < out_features; ++r) {
        const int8_t* w_row = weight_tile + (r * tile_size);
        int32_t sum = 0;
        for (int c = 0; c < tile_size; ++c) {
            sum += static_cast<int32_t>(input_tile[c]) * static_cast<int32_t>(w_row[c]);
        }
        accum_out[r] += sum;
    }
#endif
}

// ============================================================================
// Engine kernels: dot, gemv, axpy, softmax, RMSNorm, SiLU*up, and uint8-code variants for the int8 KV cache.
// Each has an AVX2+FMA path, a NEON path, and the scalar loop (which also finishes the tail of the SIMD paths).
// ============================================================================

inline const char* simd_backend() {
#if defined(TURING_HAS_AVX2_FMA)
    return "avx2";
#elif defined(TURING_HAS_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

struct AlignedDeleter {
    void operator()(void* p) const { if (p) aligned_free_64(p); }
};
template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedDeleter>;

// 64-byte aligned, zero-filled array of n elements (at least one, so the pointer is never null).
template <typename T>
inline AlignedArray<T> make_aligned(size_t n) {
    size_t bytes = std::max<size_t>(n, 1) * sizeof(T);
    T* p = static_cast<T*>(aligned_alloc_64(bytes));
    std::memset(static_cast<void*>(p), 0, bytes);
    return AlignedArray<T>(p);
}

#if defined(TURING_HAS_AVX2_FMA)
inline float hsum_f32x8(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    s = _mm_add_ss(s, _mm_movehl_ps(sh, s));
    return _mm_cvtss_f32(s);
}

// Cephes-style expf: x = n ln2 + r, degree-5 polynomial for e^r, 2^n through the exponent bits.
inline __m256 exp_f32x8(__m256 x) {
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f)), _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_fmadd_ps(y, z, x), _mm256_set1_ps(1.0f));
    __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

inline __m256 load_u8x8_f32(const uint8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
#elif defined(TURING_HAS_NEON)
inline float32x4_t exp_f32x4(float32x4_t x) {
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.3762626647949f)), vdupq_n_f32(-88.3762626647949f));
    float32x4_t fx = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f)));
    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vaddq_f32(vfmaq_f32(x, y, z), vdupq_n_f32(1.0f));
    int32x4_t n = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(n));
}

// 16 uint8 codes -> four float32x4 vectors.
inline void load_u8x16_f32(const uint8_t* p, float32x4_t out[4]) {
    uint8x16_t b = vld1q_u8(p);
    uint16x8_t lo = vmovl_u8(vget_low_u8(b)), hi = vmovl_u8(vget_high_u8(b));
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    out[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    out[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
}
#endif

// sum_i a[i] * b[i]
inline float dot_f32(const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    size_t i = 0;
    float s = 0.0f;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    s = hsum_f32x8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
#elif defined(TURING_HAS_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i),      vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y[M] = W[M, K] @ x[K]
inline void gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                     size_t M, size_t K) {
    for (size_t r = 0; r < M; ++r) y[r] = dot_f32(W + r * K, x, K);
}

// y += a * x
inline void axpy_f32(float a, const float* __restrict__ x, float* __restrict__ y, size_t n) {
    size_t i = 0;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#elif defined(TURING_HAS_NEON)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
#endif
    for (; i < n; ++i) y[i] += a * x[i];
}

// In-place numerically stable softmax over x[n].
inline void softmax_f32(float* x, size_t n) {
    if (n == 0) return;
    size_t i = 0;
    float mx = x[0];
#if defined(TURING_HAS_AVX2_FMA)
    if (n >= 8) {
        __m256 vm = _mm256_loadu_ps(x);
        for (i = 8; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));
        alignas(32) float buf[8];
        _mm256_store_ps(buf, vm);
        mx = *std::max_element(buf, buf + 8);
    }
#elif defined(TURING_HAS_NEON)
    if (n >= 4) {
        float32x4_t vm = vld1q_f32(x);
        for (i = 4; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(x + i));
        mx = vmaxvq_f32(vm);
    }
#endif
    for (; i < n; ++i) mx = std::max(mx, x[i]);

    i = 0;
    float sum = 0.0f;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 vmx = _mm256_set1_ps(mx), vs = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 e = exp_f32x8(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx));
        _mm256_storeu_ps(x + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    sum = hsum_f32x8(vs);
#elif defined(TURING_HAS_NEON)
    float32x4_t vmx = vdupq_n_f32(mx), vs = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(x + i), vmx));
        vst1q_f32(x + i, e);
        vs = vaddq_f32(vs, e);
    }
    sum = vaddvq_f32(vs);
#endif
    for (; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }

    float inv = 1.0f / sum;  // sum >= 1: the max element contributes exp(0)
    i = 0;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
#elif defined(TURING_HAS_NEON)
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_n_f32(vld1q_f32(x + i), inv));
#endif
    for (; i < n; ++i) x[i] *= inv;
}

// out = x / sqrt(mean(x^2) + eps) * w   (the order PyTorch's RMSNorm uses: normalise, then scale)
inline void rmsnorm_f32(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                        size_t n, float eps) {
    float inv = 1.0f / std::sqrt(dot_f32(x, x, n) / static_cast<float>(n) + eps);
    size_t i = 0;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vinv), _mm256_loadu_ps(w + i)));
#elif defined(TURING_HAS_NEON)
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vmulq_n_f32(vld1q_f32(x + i), inv), vld1q_f32(w + i)));
#endif
    for (; i < n; ++i) out[i] = (x[i] * inv) * w[i];
}

// gate = silu(gate) * up = gate / (1 + exp(-gate)) * up
inline void silu_mul_f32(float* __restrict__ gate, const float* __restrict__ up, size_t n) {
    size_t i = 0;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 s = _mm256_div_ps(g, _mm256_add_ps(one, exp_f32x8(_mm256_sub_ps(zero, g))));
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(s, _mm256_loadu_ps(up + i)));
    }
#elif defined(TURING_HAS_NEON)
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t s = vdivq_f32(g, vaddq_f32(one, exp_f32x4(vnegq_f32(g))));
        vst1q_f32(gate + i, vmulq_f32(s, vld1q_f32(up + i)));
    }
#endif
    for (; i < n; ++i) gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

// sum_i a[i] * float(codes[i])
inline float dot_f32_u8(const float* __restrict__ a, const uint8_t* __restrict__ codes, size_t n) {
    size_t i = 0;
    float s = 0.0f;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     load_u8x8_f32(codes + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), load_u8x8_f32(codes + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load_u8x8_f32(codes + i), a0);
    s = hsum_f32x8(_mm256_add_ps(a0, a1));
#elif defined(TURING_HAS_NEON)
    float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    float32x4_t c[4];
    for (; i + 16 <= n; i += 16) {
        load_u8x16_f32(codes + i, c);
        for (int j = 0; j < 4; ++j) acc[j] = vfmaq_f32(acc[j], vld1q_f32(a + i + 4 * j), c[j]);
    }
    s = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
#endif
    for (; i < n; ++i) s += a[i] * static_cast<float>(codes[i]);
    return s;
}

// y += a * float(codes) + b
inline void axpby_u8_f32(float a, float b, const uint8_t* __restrict__ codes, float* __restrict__ y, size_t n) {
    size_t i = 0;
#if defined(TURING_HAS_AVX2_FMA)
    __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(b);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_fmadd_ps(va, load_u8x8_f32(codes + i), vb)));
#elif defined(TURING_HAS_NEON)
    float32x4_t va = vdupq_n_f32(a), vb = vdupq_n_f32(b);
    float32x4_t c[4];
    for (; i + 16 <= n; i += 16) {
        load_u8x16_f32(codes + i, c);
        for (int j = 0; j < 4; ++j)
            vst1q_f32(y + i + 4 * j, vaddq_f32(vld1q_f32(y + i + 4 * j), vfmaq_f32(vb, va, c[j])));
    }
#endif
    for (; i < n; ++i) y[i] += a * static_cast<float>(codes[i]) + b;
}

} // namespace turing
