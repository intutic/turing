#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace turing {

/**
 * C++ Accelerated Particle Swarm Optimization Kernel.
 * Adapted from High-Performance Compute Engine (Asynchronous PSO for Non-Convex Policy Optimization).
 */
inline void pso_step_cpp(
    float* __restrict__ positions,      // [pop_size, dim]
    float* __restrict__ velocities,     // [pop_size, dim]
    float* __restrict__ best_positions, // [pop_size, dim]
    float* __restrict__ best_fitness,   // [pop_size]
    float* __restrict__ global_best_pos,// [dim]
    float* __restrict__ global_best_fit,// scalar
    int pop_size,
    int dim,
    float w,
    float c1,
    float c2,
    float min_bound,
    float max_bound
) {
    (void)best_fitness;
    (void)global_best_fit;
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int p = 0; p < pop_size; ++p) {
        float* pos = positions + (p * dim);
        float* vel = velocities + (p * dim);
        float* pbest = best_positions + (p * dim);

        for (int d = 0; d < dim; ++d) {
            float r1 = dis(gen);
            float r2 = dis(gen);

            float cog = c1 * r1 * (pbest[d] - pos[d]);
            float soc = c2 * r2 * (global_best_pos[d] - pos[d]);
            vel[d] = w * vel[d] + cog + soc;

            pos[d] = std::clamp(pos[d] + vel[d], min_bound, max_bound);
        }
    }
}

} // namespace turing
