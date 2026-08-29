#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 Zero-Copy SVD Wire Codec.
 * Fuses Subspace Projection (K @ U), AbsMax scale extraction, INT8 quantization,
 * and binary payload construction for high-speed network KV transfer.
 */
inline void svd_wire_project_quantize_avx2(
    const float* __restrict__ input_tensor, // [N, Heads, HeadDim]
    const float* __restrict__ u_proj,       // [HeadDim, Rank]
    int8_t* __restrict__ out_int8,          // [N, Heads, Rank]
    float* __restrict__ out_scale,          // [N, Heads, 1]
    int N,
    int Heads,
    int HeadDim,
    int Rank
) {
    int total_vectors = N * Heads;

    for (int v = 0; v < total_vectors; ++v) {
        const float* in_vec = input_tensor + v * HeadDim;
        int8_t* int8_vec = out_int8 + v * Rank;
        float* scale_ptr = out_scale + v;

        std::vector<float> proj_buf(Rank);
        float max_abs = 1e-5f;

        // Step 1: Project in_vec (HeadDim) @ u_proj (HeadDim x Rank) -> proj_buf (Rank)
        for (int r = 0; r < Rank; ++r) {
            float dot = 0.0f;
            int d = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
            __m256 accum_v = _mm256_setzero_ps();
            for (; d + 7 < HeadDim; d += 8) {
                __m256 in_vals = _mm256_loadu_ps(in_vec + d);
                // u_proj layout: [HeadDim, Rank], so element at [d, r] is u_proj[d * Rank + r]
                // Load stride of Rank or gather
                float u_temp[8];
                for (int i = 0; i < 8; ++i) {
                    u_temp[i] = u_proj[(d + i) * Rank + r];
                }
                __m256 u_vals = _mm256_loadu_ps(u_temp);
                accum_v = _mm256_fmadd_ps(in_vals, u_vals, accum_v);
            }
            float temp[8];
            _mm256_storeu_ps(temp, accum_v);
            for (int i = 0; i < 8; ++i) dot += temp[i];
#endif
            for (; d < HeadDim; ++d) {
                dot += in_vec[d] * u_proj[d * Rank + r];
            }
            proj_buf[r] = dot;
            float abs_val = std::abs(dot);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        // Step 2: Scale and Quantize to INT8
        float scale = max_abs / 127.0f;
        *scale_ptr = scale;
        float inv_scale = 1.0f / (scale > 1e-8f ? scale : 1.0f);

        int r = 0;
#if defined(TURING_HAS_AVX2)
        __m256 inv_sc_v = _mm256_set1_ps(inv_scale);
        for (; r + 7 < Rank; r += 8) {
            __m256 p_vals = _mm256_loadu_ps(proj_buf.data() + r);
            __m256 scaled = _mm256_mul_ps(p_vals, inv_sc_v);
            __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i int32_vals = _mm256_cvtps_epi32(rounded);

            // Pack int32 -> int16 -> int8 with saturation
            __m128i lo = _mm256_castsi256_si128(int32_vals);
            __m128i hi = _mm256_extracti128_si256(int32_vals, 1);
            __m128i int16_vals = _mm_packs_epi32(lo, hi);
            __m128i int8_vals = _mm_packs_epi16(int16_vals, int16_vals);

            int64_t packed64 = _mm_cvtsi128_si64(int8_vals);
            std::memcpy(int8_vec + r, &packed64, 8);
        }
#endif
        for (; r < Rank; ++r) {
            float val = proj_buf[r] * inv_scale;
            float clamped = std::clamp(std::round(val), -128.0f, 127.0f);
            int8_vec[r] = static_cast<int8_t>(clamped);
        }
    }
}

/**
 * Native C++20 AVX2 SVD Wire Reconstruction (Dequantize & Up-Project).
 * input_int8: [N, Heads, Rank]
 * scale: [N, Heads, 1]
 * u_proj: [HeadDim, Rank] (Reconstructs Full KV: Subspace @ U^T)
 * out_tensor: [N, Heads, HeadDim]
 */
inline void svd_wire_dequantize_reconstruct_avx2(
    const int8_t* __restrict__ input_int8,
    const float* __restrict__ scale,
    const float* __restrict__ u_proj,
    float* __restrict__ out_tensor,
    int N,
    int Heads,
    int HeadDim,
    int Rank
) {
    int total_vectors = N * Heads;

    for (int v = 0; v < total_vectors; ++v) {
        const int8_t* int8_vec = input_int8 + v * Rank;
        float sc = scale[v];
        float* out_vec = out_tensor + v * HeadDim;

        std::vector<float> dequant_buf(Rank);
        for (int r = 0; r < Rank; ++r) {
            dequant_buf[r] = static_cast<float>(int8_vec[r]) * sc;
        }

        // Project dequant_buf (Rank) @ u_proj^T (Rank x HeadDim) -> out_vec (HeadDim)
        for (int d = 0; d < HeadDim; ++d) {
            float dot = 0.0f;
            int r = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
            __m256 accum_v = _mm256_setzero_ps();
            for (; r + 7 < Rank; r += 8) {
                __m256 dq_vals = _mm256_loadu_ps(dequant_buf.data() + r);
                __m256 u_vals = _mm256_loadu_ps(u_proj + d * Rank + r);
                accum_v = _mm256_fmadd_ps(dq_vals, u_vals, accum_v);
            }
            float temp[8];
            _mm256_storeu_ps(temp, accum_v);
            for (int i = 0; i < 8; ++i) dot += temp[i];
#endif
            for (; r < Rank; ++r) {
                dot += dequant_buf[r] * u_proj[d * Rank + r];
            }
            out_vec[d] = dot;
        }
    }
}

} // namespace turing
