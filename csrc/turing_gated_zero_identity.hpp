#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace turing {

/**
 * @brief Fast SIMD Sigmoid approximation: sigma(x) = 1 / (1 + exp(-x)).
 */
inline float fast_sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Fast SIMD Gated Zero-Identity Head Forward Pass.
 * Evaluates gate sigmoid, scales raw key and value predictions, and writes output:
 * dk = sigmoid(gate_k) * raw_k
 * dv = sigmoid(gate_v) * raw_v
 */
inline void gated_zero_identity_simd(
    const float* gate_ptr,      // [2 * num_heads]
    const float* raw_k_ptr,     // [num_heads, head_dim]
    const float* raw_v_ptr,     // [num_heads, head_dim]
    float* out_k_ptr,           // [num_heads, head_dim]
    float* out_v_ptr,           // [num_heads, head_dim]
    size_t num_heads,
    size_t head_dim
) {
    for (size_t h = 0; h < num_heads; ++h) {
        float gk = fast_sigmoid_scalar(gate_ptr[h]);
        float gv = fast_sigmoid_scalar(gate_ptr[num_heads + h]);

        const float* rk = raw_k_ptr + h * head_dim;
        const float* rv = raw_v_ptr + h * head_dim;
        float* ok = out_k_ptr + h * head_dim;
        float* ov = out_v_ptr + h * head_dim;

        size_t d = 0;
#if defined(__AVX2__)
        __m256 v_gk = _mm256_set1_ps(gk);
        __m256 v_gv = _mm256_set1_ps(gv);

        for (; d + 8 <= head_dim; d += 8) {
            __m256 rk_vec = _mm256_loadu_ps(rk + d);
            __m256 rv_vec = _mm256_loadu_ps(rv + d);

            __m256 ok_vec = _mm256_mul_ps(v_gk, rk_vec);
            __m256 ov_vec = _mm256_mul_ps(v_gv, rv_vec);

            _mm256_storeu_ps(ok + d, ok_vec);
            _mm256_storeu_ps(ov + d, ov_vec);
        }
#endif
        for (; d < head_dim; ++d) {
            ok[d] = gk * rk[d];
            ov[d] = gv * rv[d];
        }
    }
}

} // namespace turing
