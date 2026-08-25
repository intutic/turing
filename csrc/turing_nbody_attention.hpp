#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Softened All-to-All Attention Kernel.
 * Adapted from High-Performance Compute Engine (N-Body Multi-Block Gravitational Potential with Softening).
 * Formulation: Attn(Q_i, K_j) = exp((Q_i @ K_j^T) * scale - softening_sq)
 */
inline void softened_nbody_attention_forward(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    float softening_sq,
    float scale
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int head_offset_q = (b * num_heads + h) * seq_len_q * head_dim;
            int head_offset_k = (b * num_heads + h) * seq_len_k * head_dim;
            int head_offset_v = (b * num_heads + h) * seq_len_k * head_dim;
            int head_offset_out = (b * num_heads + h) * seq_len_q * head_dim;

            const float* q_head = query + head_offset_q;
            const float* k_head = key + head_offset_k;
            const float* v_head = value + head_offset_v;
            float* out_head = output + head_offset_out;

            std::vector<float> scores(seq_len_k, 0.0f);

            for (int qi = 0; qi < seq_len_q; ++qi) {
                const float* q_vec = q_head + qi * head_dim;
                float max_score = -1e9f;

                // 1. All-to-all dot-product with softening parameter
                for (int kj = 0; kj < seq_len_k; ++kj) {
                    const float* k_vec = k_head + kj * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    float s = (dot * scale) - softening_sq;
                    scores[kj] = s;
                    if (s > max_score) max_score = s;
                }

                // 2. Softmax normalization
                float sum_exp = 0.0f;
                for (int kj = 0; kj < seq_len_k; ++kj) {
                    scores[kj] = std::exp(scores[kj] - max_score);
                    sum_exp += scores[kj];
                }
                float inv_sum = 1.0f / (sum_exp > 1e-8f ? sum_exp : 1.0f);
                for (int kj = 0; kj < seq_len_k; ++kj) {
                    scores[kj] *= inv_sum;
                }

                // 3. Value aggregation
                float* out_vec = out_head + qi * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int kj = 0; kj < seq_len_k; ++kj) {
                        acc += scores[kj] * v_head[kj * head_dim + d];
                    }
                    out_vec[d] = acc;
                }
            }
        }
    }
}

} // namespace turing
