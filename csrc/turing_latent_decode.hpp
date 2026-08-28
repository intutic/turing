#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Latent Flash-Decode (Mode-B) Routine.
 * Directly evaluates attention in the Rank-R subspace against INT8 singular coordinates.
 */
inline void latent_decode_avx2(
    const float* __restrict__ qp,     // [B * NKV * GRP, R]
    const int8_t* __restrict__ ck,    // [B * N, R]
    const float* __restrict__ sk,     // [B * N]
    const int8_t* __restrict__ cv,    // [B * N, R]
    const float* __restrict__ sv,     // [B * N]
    float* __restrict__ out,          // [B * NKV * GRP, R]
    int B,
    int NKV,
    int GRP,
    int R,
    int N,
    float scale
) {
    std::vector<float> scores(N);

    for (int b = 0; b < B; ++b) {
        for (int kv = 0; kv < NKV; ++kv) {
            for (int g = 0; g < GRP; ++g) {
                int q_idx = (b * NKV + kv) * GRP + g;
                const float* q_vec = qp + q_idx * R;
                float* out_vec = out + q_idx * R;
                std::fill(out_vec, out_vec + R, 0.0f);

                float max_score = -1e30f;

                // Step 1: Compute Q' @ K_t'^T across all tokens N
                for (int t = 0; t < N; ++t) {
                    const int8_t* ck_t = ck + (b * N + t) * R;
                    float sk_t = sk[b * N + t];

                    float dot = 0.0f;
                    int r = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
                    __m256 accum_v = _mm256_setzero_ps();
                    __m256 sk_v = _mm256_set1_ps(sk_t);

                    for (; r + 7 < R; r += 8) {
                        __m256 q_vals = _mm256_loadu_ps(q_vec + r);
                        __m128i raw_int8 = _mm_loadu_si64(ck_t + r);
                        __m256i int32_vals = _mm256_cvtepi8_epi32(raw_int8);
                        __m256 k_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(int32_vals), sk_v);

                        accum_v = _mm256_fmadd_ps(q_vals, k_vals, accum_v);
                    }
                    float temp[8];
                    _mm256_storeu_ps(temp, accum_v);
                    for (int i = 0; i < 8; ++i) dot += temp[i];
#endif
                    for (; r < R; ++r) {
                        dot += q_vec[r] * (static_cast<float>(ck_t[r]) * sk_t);
                    }

                    float s = dot * scale;
                    scores[t] = s;
                    if (s > max_score) {
                        max_score = s;
                    }
                }

                // Step 2: Softmax over scores
                float sum_exp = 0.0f;
                for (int t = 0; t < N; ++t) {
                    float p = std::exp(scores[t] - max_score);
                    scores[t] = p;
                    sum_exp += p;
                }
                float inv_sum = 1.0f / std::max(sum_exp, 1e-8f);

                // Step 3: Aggregate Softmax(P) @ V'
                for (int t = 0; t < N; ++t) {
                    float weight = scores[t] * inv_sum;
                    const int8_t* cv_t = cv + (b * N + t) * R;
                    float sv_t = sv[b * N + t];
                    float eff_scale = weight * sv_t;

                    int r = 0;
#if defined(TURING_HAS_AVX2) && defined(__FMA__)
                    __m256 eff_v = _mm256_set1_ps(eff_scale);
                    for (; r + 7 < R; r += 8) {
                        __m256 cur_out = _mm256_loadu_ps(out_vec + r);
                        __m128i raw_int8 = _mm_loadu_si64(cv_t + r);
                        __m256i int32_vals = _mm256_cvtepi8_epi32(raw_int8);
                        __m256 v_vals = _mm256_cvtepi32_ps(int32_vals);

                        cur_out = _mm256_fmadd_ps(v_vals, eff_v, cur_out);
                        _mm256_storeu_ps(out_vec + r, cur_out);
                    }
#endif
                    for (; r < R; ++r) {
                        out_vec[r] += static_cast<float>(cv_t[r]) * eff_scale;
                    }
                }
            }
        }
    }
}

} // namespace turing
