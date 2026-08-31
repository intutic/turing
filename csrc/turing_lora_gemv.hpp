#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#define TURING_HAS_AVX2 1
#endif

namespace turing {

/**
 * @brief Fused Base GEMV + Multi-Tenant LoRA Rank-8 contraction.
 * Computes y[b, o] = sum_{i} x[b, i] * W_base[i, o] + alpha * sum_{r=1}^R (sum_{i} x[b, i] * W_A[i, r]) * W_B[r, o]
 * Accumulates intermediate rank-8 projections directly in CPU registers without DRAM writes.
 */
inline void fused_lora_gemv_simd(
    const float* __restrict__ x,           // [Batch, InDim]
    const float* __restrict__ w_base,      // [InDim, OutDim]
    const float* __restrict__ w_a,         // [InDim, Rank]
    const float* __restrict__ w_b,         // [Rank, OutDim]
    float alpha,
    float* __restrict__ out,              // [Batch, OutDim]
    int batch,
    int in_dim,
    int out_dim,
    int rank
) {
    for (int b = 0; b < batch; ++b) {
        const float* x_b = x + b * in_dim;
        float* out_b = out + b * out_dim;

        // 1. Compute LoRA intermediate vector z = x @ W_A (Rank <= 32, typically 8 or 16)
        alignas(32) float z_local[64] = {0.0f};
        int effective_rank = std::min(rank, 64);

        for (int r = 0; r < effective_rank; ++r) {
            float sum_r = 0.0f;
            int i = 0;
#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= in_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_b + i);
                alignas(32) float wa_tmp[8];
                for (int j = 0; j < 8; ++j) {
                    wa_tmp[j] = w_a[(i + j) * rank + r];
                }
                __m256 wa_vec = _mm256_load_ps(wa_tmp);
                acc = _mm256_fmadd_ps(x_vec, wa_vec, acc);
            }
            alignas(32) float acc_tmp[8];
            _mm256_store_ps(acc_tmp, acc);
            for (int j = 0; j < 8; ++j) sum_r += acc_tmp[j];
#endif
            for (; i < in_dim; ++i) {
                sum_r += x_b[i] * w_a[i * rank + r];
            }
            z_local[r] = sum_r * alpha;
        }

        // 2. Fused Base GEMV + LoRA W_B contribution: y_o = (x @ W_base)_o + (z @ W_B)_o
        for (int o = 0; o < out_dim; ++o) {
            float sum_base = 0.0f;
            int i = 0;
#if defined(__AVX2__)
            __m256 acc_base = _mm256_setzero_ps();
            for (; i + 8 <= in_dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_b + i);
                alignas(32) float wb_tmp[8];
                for (int j = 0; j < 8; ++j) {
                    wb_tmp[j] = w_base[(i + j) * out_dim + o];
                }
                __m256 wb_vec = _mm256_load_ps(wb_tmp);
                acc_base = _mm256_fmadd_ps(x_vec, wb_vec, acc_base);
            }
            alignas(32) float acc_b_tmp[8];
            _mm256_store_ps(acc_b_tmp, acc_base);
            for (int j = 0; j < 8; ++j) sum_base += acc_b_tmp[j];
#endif
            for (; i < in_dim; ++i) {
                sum_base += x_b[i] * w_base[i * out_dim + o];
            }

            // Add LoRA component
            float sum_lora = 0.0f;
            for (int r = 0; r < effective_rank; ++r) {
                sum_lora += z_local[r] * w_b[r * out_dim + o];
            }

            out_b[o] = sum_base + sum_lora;
        }
    }
}

} // namespace turing
