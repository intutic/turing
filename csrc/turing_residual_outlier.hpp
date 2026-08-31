#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#define TURING_HAS_AVX2 1
#endif

namespace turing {

/**
 * @brief 1-Pass Subspace Residual Outlier Extraction with AVX2 SIMD.
 * Given residual vector r = x - x_recon (dim = hidden_dim), finds:
 * top_idx = argmax_{i} |r_i|
 * top_val = r[top_idx]
 * Performs in-register reduction in a single linear pass over the vector without memory sorting.
 */
inline void find_residual_outlier_simd(
    const float* __restrict__ residual,   // [Batch, HiddenDim]
    int* __restrict__ out_top_indices,   // [Batch]
    float* __restrict__ out_top_values,  // [Batch]
    int batch,
    int hidden_dim
) {
    for (int b = 0; b < batch; ++b) {
        const float* r_b = residual + b * hidden_dim;

        float max_abs_val = -1.0f;
        int max_idx = 0;
        float actual_val = 0.0f;

        int i = 0;
#if defined(__AVX2__)
        // AVX2 pass to find block with maximum absolute value
        for (; i + 8 <= hidden_dim; i += 8) {
            __m256 v = _mm256_loadu_ps(r_b + i);
            __m256 v_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); // fast fabs

            alignas(32) float vals[8];
            alignas(32) float abs_vals[8];
            _mm256_store_ps(vals, v);
            _mm256_store_ps(abs_vals, v_abs);

            for (int j = 0; j < 8; ++j) {
                if (abs_vals[j] > max_abs_val) {
                    max_abs_val = abs_vals[j];
                    max_idx = i + j;
                    actual_val = vals[j];
                }
            }
        }
#endif

        for (; i < hidden_dim; ++i) {
            float val = r_b[i];
            float abs_val = std::fabs(val);
            if (abs_val > max_abs_val) {
                max_abs_val = abs_val;
                max_idx = i;
                actual_val = val;
            }
        }

        out_top_indices[b] = max_idx;
        out_top_values[b] = actual_val;
    }
}

} // namespace turing
