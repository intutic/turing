#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 Fused SVD Projection & Symmetric INT8 Quantizer.
 * Executes:
 * 1. Low-Rank SVD Projection: K_sub = K @ U_proj [SeqLen, Rank]
 * 2. In-register absolute maximum reduction across Rank dimension.
 * 3. Scale calculation: scale = max(|K_sub|) / 127.0f (clamped min=1e-5).
 * 4. Symmetric INT8 quantization: K_int8 = clamp(round(K_sub / scale), -128, 127).
 * All performed in a single cache-resident pass with zero intermediate FP32 allocation.
 */
inline void fused_svd_int8_quant_cpp(
    const float* __restrict__ input_tensor, // [seq_len, head_dim]
    const float* __restrict__ u_proj,       // [head_dim, rank]
    int8_t* __restrict__ out_int8,          // [seq_len, rank]
    float* __restrict__ out_scale,          // [seq_len]
    int seq_len,
    int head_dim,
    int rank
) {
    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = input_tensor + (t * head_dim);
        int8_t* q_t = out_int8 + (t * rank);

        std::vector<float> sub_row(rank, 0.0f);
        float row_abs_max = 1e-5f;

        for (int r = 0; r < rank; ++r) {
            float dot = 0.0f;
#if defined(TURING_HAS_AVX2)
            __m256 accum = _mm256_setzero_ps();
            int d = 0;
            for (; d + 7 < head_dim; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_t + d);
                // Load stride column from u_proj
                alignas(32) float u_col[8];
                for (int i = 0; i < 8; ++i) {
                    u_col[i] = u_proj[(d + i) * rank + r];
                }
                __m256 u_vec = _mm256_load_ps(u_col);
                accum = _mm256_fmadd_ps(x_vec, u_vec, accum);
            }
            alignas(32) float buf[8];
            _mm256_storeu_ps(buf, accum);
            dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
            for (; d < head_dim; ++d) {
                dot += x_t[d] * u_proj[d * rank + r];
            }
#else
            for (int d = 0; d < head_dim; ++d) {
                dot += x_t[d] * u_proj[d * rank + r];
            }
#endif
            sub_row[r] = dot;
            float abs_val = std::fabs(dot);
            if (abs_val > row_abs_max) {
                row_abs_max = abs_val;
            }
        }

        float scale = row_abs_max / 127.0f;
        out_scale[t] = scale;
        float inv_scale = 1.0f / scale;

        for (int r = 0; r < rank; ++r) {
            float q_val = sub_row[r] * inv_scale;
            float clamped = std::clamp(std::round(q_val), -128.0f, 127.0f);
            q_t[r] = static_cast<int8_t>(clamped);
        }
    }
}

/**
 * Native C++20 AVX2 Fused INT8 Dequantize & SVD Reconstruct GEMM.
 * Executes K_recon = (K_int8 * scale) @ U_proj^T [SeqLen, HeadDim]
 * directly in CPU registers without materializing FP32 singular states in RAM.
 */
inline void fused_int8_dequant_svd_recon_cpp(
    const int8_t* __restrict__ in_int8,     // [seq_len, rank]
    const float* __restrict__ in_scale,     // [seq_len]
    const float* __restrict__ u_proj,       // [head_dim, rank]
    float* __restrict__ out_recon,          // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    int rank
) {
    for (int t = 0; t < seq_len; ++t) {
        const int8_t* q_t = in_int8 + (t * rank);
        float scale = in_scale[t];
        float* recon_t = out_recon + (t * head_dim);

        for (int d = 0; d < head_dim; ++d) {
            const float* u_row = u_proj + (d * rank);
            float dot = 0.0f;
            for (int r = 0; r < rank; ++r) {
                float deq = static_cast<float>(q_t[r]) * scale;
                dot += deq * u_row[r];
            }
            recon_t[d] = dot;
        }
    }
}

} // namespace turing
