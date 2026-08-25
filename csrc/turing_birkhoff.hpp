#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Birkhoff Polytope Manifold Projector (Sinkhorn-Knopp Algorithm).
 * Projects arbitrary real square matrices onto the set of doubly stochastic matrices.
 */
inline void birkhoff_manifold_project(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int n,
    int num_iterations = 20,
    float eps = 1e-6f
) {
    int mat_size = n * n;
    std::vector<float> row_sums(n, 0.0f);
    std::vector<float> col_sums(n, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        const float* in_mat = input + (b * mat_size);
        float* out_mat = output + (b * mat_size);

        // 1. Exponentiate with numerical stability: exp(M - max(M)) + eps
        for (int i = 0; i < n; ++i) {
            float max_val = in_mat[i * n];
            for (int j = 1; j < n; ++j) {
                max_val = std::max(max_val, in_mat[i * n + j]);
            }
            for (int j = 0; j < n; ++j) {
                out_mat[i * n + j] = std::exp(in_mat[i * n + j] - max_val) + eps;
            }
        }

        // 2. Iterative alternating row & column normalizations
        for (int it = 0; it < num_iterations; ++it) {
            // Row normalization
            for (int i = 0; i < n; ++i) {
                float r_sum = eps;
                for (int j = 0; j < n; ++j) {
                    r_sum += out_mat[i * n + j];
                }
                float inv_r = 1.0f / r_sum;
                for (int j = 0; j < n; ++j) {
                    out_mat[i * n + j] *= inv_r;
                }
            }

            // Column normalization
            for (int j = 0; j < n; ++j) {
                float c_sum = eps;
                for (int i = 0; i < n; ++i) {
                    c_sum += out_mat[i * n + j];
                }
                float inv_c = 1.0f / c_sum;
                for (int i = 0; i < n; ++i) {
                    out_mat[i * n + j] *= inv_c;
                }
            }
        }
    }
}

} // namespace turing
