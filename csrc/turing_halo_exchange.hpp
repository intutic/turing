#pragma once

#include <vector>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 2D Spatial Mesh Halo Exchange Engine (Spatial HPC Stencil Engine).
 * Performs double-buffered asynchronous boundary row exchange across 2D Tensor Parallel processor grids.
 */
inline void halo_exchange_step_cpp(
    const float* __restrict__ local_grid,    // [Height, Width]
    float* __restrict__ top_halo_out,        // [Width]
    float* __restrict__ bottom_halo_out,     // [Width]
    const float* __restrict__ top_halo_in,   // [Width]
    const float* __restrict__ bottom_halo_in,// [Width]
    float* __restrict__ next_grid,           // [Height, Width]
    int height,
    int width,
    float diffusion_alpha = 0.25f
) {
    // 1. Pack boundary halos for asynchronous neighbor transmission
    std::memcpy(top_halo_out, local_grid, width * sizeof(float));
    std::memcpy(bottom_halo_out, local_grid + ((height - 1) * width), width * sizeof(float));

    // 2. Compute 5-point stencil on interior & boundary cells using received neighbor halos
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float center = local_grid[y * width + x];

            // Top neighbor
            float top = (y > 0) ? local_grid[(y - 1) * width + x]
                                : (top_halo_in ? top_halo_in[x] : center);

            // Bottom neighbor
            float bottom = (y < height - 1) ? local_grid[(y + 1) * width + x]
                                            : (bottom_halo_in ? bottom_halo_in[x] : center);

            // Left neighbor
            float left = (x > 0) ? local_grid[y * width + (x - 1)] : center;

            // Right neighbor
            float right = (x < width - 1) ? local_grid[y * width + (x + 1)] : center;

            // Update cell with diffusion / smoothing
            next_grid[y * width + x] = center + diffusion_alpha * (top + bottom + left + right - (4.0f * center));
        }
    }
}

} // namespace turing
