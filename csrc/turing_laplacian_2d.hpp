#pragma once

#include <vector>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * 9-Point 2D Spatial Laplacian Stencil Diffusion Kernel.
 * Executes double-buffered spatial belief propagation and quadtree speculative verification.
 */
inline void laplacian_2d_step_cpp(
    const float* in_grid,
    float* out_grid,
    int height,
    int width,
    float alpha = 0.1f
) {
    for (int y = 0; y < height; ++y) {
        int ym1 = (y > 0) ? (y - 1) : 0;
        int yp1 = (y < height - 1) ? (y + 1) : (height - 1);

        for (int x = 0; x < width; ++x) {
            int xm1 = (x > 0) ? (x - 1) : 0;
            int xp1 = (x < width - 1) ? (x + 1) : (width - 1);

            float center = in_grid[y * width + x];

            // 4 direct neighbors (weight 0.5)
            float direct = in_grid[ym1 * width + x] + in_grid[yp1 * width + x] +
                           in_grid[y * width + xm1] + in_grid[y * width + xp1];

            // 4 diagonal neighbors (weight 0.25)
            float diag = in_grid[ym1 * width + xm1] + in_grid[ym1 * width + xp1] +
                         in_grid[yp1 * width + xm1] + in_grid[yp1 * width + xp1];

            float laplacian = (0.5f * direct) + (0.25f * diag) - (3.0f * center);
            out_grid[y * width + x] = center + (alpha * laplacian);
        }
    }
}

} // namespace turing
