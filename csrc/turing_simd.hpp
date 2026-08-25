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

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#define __restrict__
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
#include <immintrin.h>
#define TURING_HAS_AVX2 1
#endif
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
#if defined(TURING_HAS_AVX2)
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

} // namespace turing
