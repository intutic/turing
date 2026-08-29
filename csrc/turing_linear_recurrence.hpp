#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Linear Recurrent Attention Step (Decode L=1).
 * Performs:
 *   S_t = alpha * S_{t-1} + V_t * K_t^T   (D x D matrix update)
 *   O_t = S_t * Q_t                       (D vector emission)
 *
 * Layout:
 *   q: [B, H, D]
 *   k: [B, H, D]
 *   v: [B, H, D]
 *   state: [B, H, D, D] (in-out)
 *   out: [B, H, D] (out)
 */
inline void linear_recurrence_step_avx2(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ state,
    float* __restrict__ out,
    int B,
    int H,
    int D,
    float decay
) {
    int total_heads = B * H;

    for (int bh = 0; bh < total_heads; ++bh) {
        const float* q_vec = q + bh * D;
        const float* k_vec = k + bh * D;
        const float* v_vec = v + bh * D;
        float* s_mat = state + bh * D * D;
        float* out_vec = out + bh * D;

        // Step 1: Update state S_t = decay * S_{t-1} + v_t * k_t^T
        for (int r = 0; r < D; ++r) {
            float v_val = v_vec[r];
            float* s_row = s_mat + r * D;

            int c = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
            __m256 decay_v = _mm256_set1_ps(decay);
            __m256 v_splat = _mm256_set1_ps(v_val);

            for (; c + 7 < D; c += 8) {
                __m256 s_vals = _mm256_loadu_ps(s_row + c);
                __m256 k_vals = _mm256_loadu_ps(k_vec + c);
                __m256 s_new = _mm256_fmadd_ps(s_vals, decay_v, _mm256_mul_ps(v_splat, k_vals));
                _mm256_storeu_ps(s_row + c, s_new);
            }
#endif
            for (; c < D; ++c) {
                s_row[c] = decay * s_row[c] + v_val * k_vec[c];
            }
        }

        // Step 2: Compute output O_t = S_t * Q_t
        for (int r = 0; r < D; ++r) {
            const float* s_row = s_mat + r * D;
            float dot = 0.0f;

            int c = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
            __m256 accum_v = _mm256_setzero_ps();
            for (; c + 7 < D; c += 8) {
                __m256 s_vals = _mm256_loadu_ps(s_row + c);
                __m256 q_vals = _mm256_loadu_ps(q_vec + c);
                accum_v = _mm256_fmadd_ps(s_vals, q_vals, accum_v);
            }
            float temp[8];
            _mm256_storeu_ps(temp, accum_v);
            for (int i = 0; i < 8; ++i) dot += temp[i];
#endif
            for (; c < D; ++c) {
                dot += s_row[c] * q_vec[c];
            }
            out_vec[r] = dot;
        }
    }
}

} // namespace turing
