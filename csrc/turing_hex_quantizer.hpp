#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Hexagonal Spatial Codebook Quantizer (Spatial HPC Stencil Engine).
 */
inline void hex_quantize_activations_cpp(
    const float* __restrict__ input_vectors, // [NumVectors, Dim]
    const float* __restrict__ codebook,      // [NumCells, Dim]
    int32_t* __restrict__ bmu_indices_out,   // [NumVectors]
    float* __restrict__ quantized_out,       // [NumVectors, Dim]
    int num_vectors,
    int num_cells,
    int dim
) {
    for (int v = 0; v < num_vectors; ++v) {
        const float* vec = input_vectors + (v * dim);
        int best_idx = 0;
        float min_dist_sq = 1e9f;

        for (int c = 0; c < num_cells; ++c) {
            const float* cell = codebook + (c * dim);
            float dist_sq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = vec[d] - cell[d];
                dist_sq += diff * diff;
            }
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_idx = c;
            }
        }

        bmu_indices_out[v] = best_idx;
        const float* best_cell = codebook + (best_idx * dim);
        float* out_vec = quantized_out + (v * dim);
        for (int d = 0; d < dim; ++d) {
            out_vec[d] = best_cell[d];
        }
    }
}

} // namespace turing
