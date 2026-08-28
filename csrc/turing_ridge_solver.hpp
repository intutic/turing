#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 AVX2 Fused Ridge Representation Projection Kernel.
 * Executes: Out = (X @ W) + B
 * with 256-bit SIMD FMA acceleration.
 */
inline void fused_ridge_forward_cpp(
    const float* __restrict__ x_source, // [n_tokens, in_dim]
    const float* __restrict__ w_flat,   // [in_dim, out_features]
    const float* __restrict__ b_flat,   // [out_features]
    float* __restrict__ out_flat,       // [n_tokens, out_features]
    int n_tokens,
    int in_dim,
    int out_features
) {
    for (int t = 0; t < n_tokens; ++t) {
        const float* x_t = x_source + (t * in_dim);
        float* out_t = out_flat + (t * out_features);

        for (int o = 0; o < out_features; ++o) {
            float bias_val = b_flat ? b_flat[o] : 0.0f;
            float sum = 0.0f;

#if defined(TURING_HAS_AVX2)
            __m256 accum = _mm256_setzero_ps();
            int d = 0;
            for (; d + 7 < in_dim; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_t + d);
                alignas(32) float w_col[8];
                for (int i = 0; i < 8; ++i) {
                    w_col[i] = w_flat[(d + i) * out_features + o];
                }
                __m256 w_vec = _mm256_load_ps(w_col);
                accum = _mm256_fmadd_ps(x_vec, w_vec, accum);
            }
            alignas(32) float buf[8];
            _mm256_storeu_ps(buf, accum);
            sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
            for (; d < in_dim; ++d) {
                sum += x_t[d] * w_flat[d * out_features + o];
            }
#else
            for (int d = 0; d < in_dim; ++d) {
                sum += x_t[d] * w_flat[d * out_features + o];
            }
#endif
            out_t[o] = sum + bias_val;
        }
    }
}

} // namespace turing
