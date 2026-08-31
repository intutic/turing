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
 * @brief 4-Stream Birkhoff Manifold-Constrained Hyper-Connection SIMD step.
 * 1. Pre-mapping: x_layer = sum_{i=0}^3 alpha_i * stream_i
 * 2. Res-mapping: stream_mixed_j = sum_{i=0}^3 stream_i * H_{res, i, j} (H_res in B_4)
 * 3. Post-mapping: stream_out_j = stream_mixed_j + beta_j * layer_update
 * Evaluates all 3 stages in a single pass over hidden dimension with AVX2 registers.
 */
inline void mhc_4stream_simd(
    const float* __restrict__ streams_in,      // [Batch * SeqLen, 4, HiddenDim]
    const float* __restrict__ layer_update,    // [Batch * SeqLen, HiddenDim]
    const float* __restrict__ alpha_weights,   // [4] (softmax normalized)
    const float* __restrict__ h_res_matrix,    // [4, 4] (doubly stochastic)
    const float* __restrict__ beta_weights,    // [4] (sigmoid normalized)
    float* __restrict__ layer_input_out,       // [Batch * SeqLen, HiddenDim]
    float* __restrict__ streams_out,           // [Batch * SeqLen, 4, HiddenDim]
    int total_tokens,
    int hidden_dim
) {
    for (int t = 0; t < total_tokens; ++t) {
        const float* s0 = streams_in + (t * 4 + 0) * hidden_dim;
        const float* s1 = streams_in + (t * 4 + 1) * hidden_dim;
        const float* s2 = streams_in + (t * 4 + 2) * hidden_dim;
        const float* s3 = streams_in + (t * 4 + 3) * hidden_dim;

        const float* lup = layer_update + t * hidden_dim;
        float* l_in = layer_input_out + t * hidden_dim;

        float* so0 = streams_out + (t * 4 + 0) * hidden_dim;
        float* so1 = streams_out + (t * 4 + 1) * hidden_dim;
        float* so2 = streams_out + (t * 4 + 2) * hidden_dim;
        float* so3 = streams_out + (t * 4 + 3) * hidden_dim;

        int d = 0;
#if defined(__AVX2__)
        __m256 a0 = _mm256_set1_ps(alpha_weights[0]);
        __m256 a1 = _mm256_set1_ps(alpha_weights[1]);
        __m256 a2 = _mm256_set1_ps(alpha_weights[2]);
        __m256 a3 = _mm256_set1_ps(alpha_weights[3]);

        __m256 b0 = _mm256_set1_ps(beta_weights[0]);
        __m256 b1 = _mm256_set1_ps(beta_weights[1]);
        __m256 b2 = _mm256_set1_ps(beta_weights[2]);
        __m256 b3 = _mm256_set1_ps(beta_weights[3]);

        for (; d + 8 <= hidden_dim; d += 8) {
            __m256 v0 = _mm256_loadu_ps(s0 + d);
            __m256 v1 = _mm256_loadu_ps(s1 + d);
            __m256 v2 = _mm256_loadu_ps(s2 + d);
            __m256 v3 = _mm256_loadu_ps(s3 + d);
            __m256 v_lup = _mm256_loadu_ps(lup + d);

            // 1. Pre-mapping
            __m256 v_pre = _mm256_mul_ps(v0, a0);
            v_pre = _mm256_fmadd_ps(v1, a1, v_pre);
            v_pre = _mm256_fmadd_ps(v2, a2, v_pre);
            v_pre = _mm256_fmadd_ps(v3, a3, v_pre);
            _mm256_storeu_ps(l_in + d, v_pre);

            // 2. Res-mapping & 3. Post-mapping
            // so0 = sum_i(vi * H[i,0]) + b0 * lup
            __m256 out0 = _mm256_mul_ps(v0, _mm256_set1_ps(h_res_matrix[0 * 4 + 0]));
            out0 = _mm256_fmadd_ps(v1, _mm256_set1_ps(h_res_matrix[1 * 4 + 0]), out0);
            out0 = _mm256_fmadd_ps(v2, _mm256_set1_ps(h_res_matrix[2 * 4 + 0]), out0);
            out0 = _mm256_fmadd_ps(v3, _mm256_set1_ps(h_res_matrix[3 * 4 + 0]), out0);
            out0 = _mm256_fmadd_ps(v_lup, b0, out0);
            _mm256_storeu_ps(so0 + d, out0);

            // so1 = sum_i(vi * H[i,1]) + b1 * lup
            __m256 out1 = _mm256_mul_ps(v0, _mm256_set1_ps(h_res_matrix[0 * 4 + 1]));
            out1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(h_res_matrix[1 * 4 + 1]), out1);
            out1 = _mm256_fmadd_ps(v2, _mm256_set1_ps(h_res_matrix[2 * 4 + 1]), out1);
            out1 = _mm256_fmadd_ps(v3, _mm256_set1_ps(h_res_matrix[3 * 4 + 1]), out1);
            out1 = _mm256_fmadd_ps(v_lup, b1, out1);
            _mm256_storeu_ps(so1 + d, out1);

            // so2 = sum_i(vi * H[i,2]) + b2 * lup
            __m256 out2 = _mm256_mul_ps(v0, _mm256_set1_ps(h_res_matrix[0 * 4 + 2]));
            out2 = _mm256_fmadd_ps(v1, _mm256_set1_ps(h_res_matrix[1 * 4 + 2]), out2);
            out2 = _mm256_fmadd_ps(v2, _mm256_set1_ps(h_res_matrix[2 * 4 + 2]), out2);
            out2 = _mm256_fmadd_ps(v3, _mm256_set1_ps(h_res_matrix[3 * 4 + 2]), out2);
            out2 = _mm256_fmadd_ps(v_lup, b2, out2);
            _mm256_storeu_ps(so2 + d, out2);

            // so3 = sum_i(vi * H[i,3]) + b3 * lup
            __m256 out3 = _mm256_mul_ps(v0, _mm256_set1_ps(h_res_matrix[0 * 4 + 3]));
            out3 = _mm256_fmadd_ps(v1, _mm256_set1_ps(h_res_matrix[1 * 4 + 3]), out3);
            out3 = _mm256_fmadd_ps(v2, _mm256_set1_ps(h_res_matrix[2 * 4 + 3]), out3);
            out3 = _mm256_fmadd_ps(v3, _mm256_set1_ps(h_res_matrix[3 * 4 + 3]), out3);
            out3 = _mm256_fmadd_ps(v_lup, b3, out3);
            _mm256_storeu_ps(so3 + d, out3);
        }
#endif

        for (; d < hidden_dim; ++d) {
            float x0 = s0[d], x1 = s1[d], x2 = s2[d], x3 = s3[d];
            float u = lup[d];

            l_in[d] = x0 * alpha_weights[0] + x1 * alpha_weights[1] + x2 * alpha_weights[2] + x3 * alpha_weights[3];

            so0[d] = (x0 * h_res_matrix[0] + x1 * h_res_matrix[4] + x2 * h_res_matrix[8] + x3 * h_res_matrix[12]) + beta_weights[0] * u;
            so1[d] = (x0 * h_res_matrix[1] + x1 * h_res_matrix[5] + x2 * h_res_matrix[9] + x3 * h_res_matrix[13]) + beta_weights[1] * u;
            so2[d] = (x0 * h_res_matrix[2] + x1 * h_res_matrix[6] + x2 * h_res_matrix[10] + x3 * h_res_matrix[14]) + beta_weights[2] * u;
            so3[d] = (x0 * h_res_matrix[3] + x1 * h_res_matrix[7] + x2 * h_res_matrix[11] + x3 * h_res_matrix[15]) + beta_weights[3] * u;
        }
    }
}

} // namespace turing
