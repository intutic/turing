#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Softened N-Body Multi-Agent Belief Recirculator (Spatial HPC Stencil Engine).
 * Computes all-to-all belief state interaction forces with softening factor epsilon^2.
 */
inline void nbody_belief_recirculate_cpp(
    const float* __restrict__ belief_states, // [NumAgents, StateDim]
    float* __restrict__ updated_states,      // [NumAgents, StateDim]
    int num_agents,
    int state_dim,
    float softening_sq = 1e-4f,
    float step_size = 0.05f
) {
    for (int i = 0; i < num_agents; ++i) {
        const float* s_i = belief_states + (i * state_dim);
        float* out_i = updated_states + (i * state_dim);

        std::vector<float> force(state_dim, 0.0f);

        for (int j = 0; j < num_agents; ++j) {
            if (i == j) continue;
            const float* s_j = belief_states + (j * state_dim);

            float dist_sq = softening_sq;
            for (int d = 0; d < state_dim; ++d) {
                float diff = s_j[d] - s_i[d];
                dist_sq += diff * diff;
            }

            float inv_dist3 = 1.0f / (dist_sq * std::sqrt(dist_sq));
            for (int d = 0; d < state_dim; ++d) {
                force[d] += (s_j[d] - s_i[d]) * inv_dist3;
            }
        }

        for (int d = 0; d < state_dim; ++d) {
            out_i[d] = s_i[d] + (step_size * force[d]);
        }
    }
}

} // namespace turing
