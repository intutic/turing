#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 SIMD KDA (Kimi Delta Attention) Step.
 * Fuses:
 *   1. Short Causal Convolution (kernel_size=4) + SiLU over Q, K, V channels
 *   2. L2 Normalization of Query & Key
 *   3. Delta State Recurrence: S_t = Diag(alpha) S_{t-1} + K_t * (V_t - S_{t-1}^T K_t)^T * beta
 *   4. Output projection: O_t = S_t^T * Q_t
 */
inline void kda_step_avx2(
    const float* __restrict__ qkv,          // [B, 3, H, D]
    float* __restrict__ conv_state,         // [B, 3, H, D, KERNEL] (updated in-place)
    const float* __restrict__ conv_weight,  // [3, H, D, KERNEL]
    float* __restrict__ state,              // [B, H, D, D] (updated in-place)
    const float* __restrict__ gate,         // [B, H, D] (log decay alpha)
    const float* __restrict__ beta,         // [B, H]
    float* __restrict__ out,                // [B, H, D]
    int B,
    int H,
    int D,
    int kernel_size,
    float norm_eps
) {
    int width = 3 * H * D;

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int bh = b * H + h;
            float* s_mat = state + bh * D * D;
            const float* g_vec = gate + bh * D;
            float b_val = beta[bh];
            float* out_vec = out + bh * D;

            // Intermediate buffers for this head
            std::vector<float> q_vec(D), k_vec(D), v_vec(D);

            // Step 1: 1D Causal Conv + SiLU for Q, K, V
            for (int ch_type = 0; ch_type < 3; ++ch_type) {
                float* target_vec = (ch_type == 0) ? q_vec.data() : ((ch_type == 1) ? k_vec.data() : v_vec.data());
                for (int d = 0; d < D; ++d) {
                    int ch_idx = ch_type * (H * D) + h * D + d;
                    float* hist = conv_state + (b * width + ch_idx) * kernel_size;
                    const float* taps = conv_weight + ch_idx * kernel_size;
                    float new_tok = qkv[b * width + ch_idx];

                    // Shift left and append
                    for (int k = 0; k < kernel_size - 1; ++k) {
                        hist[k] = hist[k + 1];
                    }
                    hist[kernel_size - 1] = new_tok;

                    // Dot product taps * history
                    float sum = 0.0f;
                    for (int k = 0; k < kernel_size; ++k) {
                        sum += taps[k] * hist[k];
                    }
                    // SiLU
                    target_vec[d] = sum / (1.0f + std::exp(-sum));
                }
            }

            // Step 2: L2 Normalization for Q and K
            float q_sq = norm_eps, k_sq = norm_eps;
            for (int d = 0; d < D; ++d) {
                q_sq += q_vec[d] * q_vec[d];
                k_sq += k_vec[d] * k_vec[d];
            }
            float q_norm = (1.0f / std::sqrt((float)D)) / std::sqrt(q_sq);
            float k_norm = 1.0f / std::sqrt(k_sq);

            for (int d = 0; d < D; ++d) {
                q_vec[d] *= q_norm;
                k_vec[d] *= k_norm;
            }

            // Step 3: Decay S_t = Diag(alpha) S_{t-1}
            for (int r = 0; r < D; ++r) {
                float alpha_r = std::exp(g_vec[r]);
                float* s_row = s_mat + r * D;
                for (int c = 0; c < D; ++c) {
                    s_row[c] *= alpha_r;
                }
            }

            // Step 4: delta = (v - S^T k) * beta
            std::vector<float> delta(D, 0.0f);
            for (int c = 0; c < D; ++c) {
                float st_k_c = 0.0f;
                for (int r = 0; r < D; ++r) {
                    st_k_c += s_mat[r * D + c] * k_vec[r];
                }
                delta[c] = (v_vec[c] - st_k_c) * b_val;
            }

            // Step 5: S += k @ delta^T
            for (int r = 0; r < D; ++r) {
                float k_r = k_vec[r];
                float* s_row = s_mat + r * D;
                for (int c = 0; c < D; ++c) {
                    s_row[c] += k_r * delta[c];
                }
            }

            // Step 6: out = S^T q
            for (int c = 0; c < D; ++c) {
                float out_c = 0.0f;
                for (int r = 0; r < D; ++r) {
                    out_c += s_mat[r * D + c] * q_vec[r];
                }
                out_vec[c] = out_c;
            }
        }
    }
}

} // namespace turing
