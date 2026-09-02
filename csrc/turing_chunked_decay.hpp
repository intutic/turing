#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace turing {

class NativeChunkedDecay {
public:
    /**
     * Single-step learned-decay linear recurrence update & readout:
     * S_t = b_decay * S_{t-1} + K_t^T V_t
     * O_t = (Q_t S_t) / sqrt(D)
     */
    static void step_forward(
        const float* q,       // [D]
        const float* k,       // [D]
        const float* v,       // [D]
        float* state,         // [D * D] (in-place update)
        float* out,           // [D]
        float b_decay,
        size_t D
    ) {
        float scale = 1.0f / std::sqrt(static_cast<float>(D));

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) && defined(__FMA__)
        __m256 v_decay = _mm256_set1_ps(b_decay);

        // 1. Update State matrix: S[i, j] = b_decay * S[i, j] + k[i] * v[j]
        for (size_t i = 0; i < D; ++i) {
            __m256 v_ki = _mm256_set1_ps(k[i]);
            float* s_row = state + i * D;

            size_t j = 0;
            for (; j + 7 < D; j += 8) {
                __m256 v_s = _mm256_loadu_ps(s_row + j);
                __m256 v_vj = _mm256_loadu_ps(v + j);
                // s_new = b_decay * s + k_i * v_j
                __m256 s_new = _mm256_fmadd_ps(v_ki, v_vj, _mm256_mul_ps(v_decay, v_s));
                _mm256_storeu_ps(s_row + j, s_new);
            }
            for (; j < D; ++j) {
                s_row[j] = b_decay * s_row[j] + k[i] * v[j];
            }
        }

        // 2. Compute Out: out[j] = sum_i (q[i] * S[i, j]) * scale
        for (size_t j = 0; j < D; ++j) {
            float sum = 0.0f;
            __m256 v_sum = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 7 < D; i += 8) {
                __m256 v_qi = _mm256_loadu_ps(q + i);
                float s_col[8];
                for (size_t k_idx = 0; k_idx < 8; ++k_idx) {
                    s_col[k_idx] = state[(i + k_idx) * D + j];
                }
                __m256 v_scol = _mm256_loadu_ps(s_col);
                v_sum = _mm256_fmadd_ps(v_qi, v_scol, v_sum);
            }

            __m128 lo = _mm256_castps256_ps128(v_sum);
            __m128 hi = _mm256_extractf128_ps(v_sum, 1);
            __m128 v128 = _mm_add_ps(lo, hi);
            v128 = _mm_hadd_ps(v128, v128);
            v128 = _mm_hadd_ps(v128, v128);
            sum += _mm_cvtss_f32(v128);

            for (; i < D; ++i) {
                sum += q[i] * state[i * D + j];
            }
            out[j] = sum * scale;
        }
#else
        // ARM64 NEON / Apple Silicon / Portable scalar loop
        for (size_t i = 0; i < D; ++i) {
            float ki = k[i];
            float* s_row = state + i * D;
            for (size_t j = 0; j < D; ++j) {
                s_row[j] = b_decay * s_row[j] + ki * v[j];
            }
        }

        for (size_t j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < D; ++i) {
                sum += q[i] * state[i * D + j];
            }
            out[j] = sum * scale;
        }
#endif
    }
};

} // namespace turing
