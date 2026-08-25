#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Hexagonal Coordinate Distance.
 * Distance metric on non-Euclidean 6-neighbor hexagonal coordinate grid:
 * d = sqrt(du^2 + dv^2 + du*dv)
 */
inline float hexagonal_distance(float u1, float v1, float u2, float v2) {
    float du = u1 - u2;
    float dv = v1 - v2;
    return std::sqrt(du * du + dv * dv + du * dv);
}

/**
 * Parallel Best Matching Unit (BMU) Reduction.
 * Finds nearest codebook prototype for batch of activation vectors.
 */
inline void hexagonal_bmu_search(
    const float* __restrict__ activations,
    const float* __restrict__ codebook,
    int64_t* __restrict__ out_bmu_indices,
    float* __restrict__ out_min_distances,
    int batch_size,
    int codebook_dim,
    int total_cells
) {
    for (int b = 0; b < batch_size; ++b) {
        const float* act_vec = activations + (b * codebook_dim);
        
        // Calculate L2 norm of activation vector
        float act_norm = 0.0f;
        for (int d = 0; d < codebook_dim; ++d) {
            act_norm += act_vec[d] * act_vec[d];
        }
        act_norm = std::sqrt(act_norm > 1e-8f ? act_norm : 1.0f);

        float min_dist = 1e9f;
        int best_idx = 0;

        for (int c = 0; c < total_cells; ++c) {
            const float* proto = codebook + (c * codebook_dim);
            float dot = 0.0f;
            for (int d = 0; d < codebook_dim; ++d) {
                dot += (act_vec[d] / act_norm) * proto[d];
            }
            float dist = 1.0f - dot;
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = c;
            }
        }

        out_bmu_indices[b] = best_idx;
        out_min_distances[b] = min_dist;
    }
}

} // namespace turing
