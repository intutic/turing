#pragma once

#include "turing_threadpool.hpp"
#include <vector>
#include <cmath>

namespace turing {

/**
 * Parallel CPU MoE Expert GEMV Dispatcher.
 * Uses persistent ThreadPool to execute multi-expert projections with zero thread spawning latency.
 */
inline void parallel_cpu_moe_gemv(
    const float* __restrict__ input,       // [batch, in_features]
    const float* __restrict__ expert_weights,// [num_experts, out_features, in_features]
    const int32_t* __restrict__ expert_indices, // [batch, top_k]
    const float* __restrict__ routing_weights,  // [batch, top_k]
    float* __restrict__ output,            // [batch, out_features]
    int batch_size,
    int in_features,
    int out_features,
    int top_k
) {
    auto& pool = get_global_threadpool();

    // Parallelize across batch items
    pool.parallel_for(0, batch_size, [&](size_t /*worker_id*/, size_t start_b, size_t end_b) {
        for (size_t b = start_b; b < end_b; ++b) {
            const float* in_vec = input + (b * static_cast<size_t>(in_features));
            float* out_vec = output + (b * static_cast<size_t>(out_features));

            // Initialize output vector to zero
            for (int oc = 0; oc < out_features; ++oc) {
                out_vec[oc] = 0.0f;
            }

            // Accumulate contributions from top_k routed experts
            for (int k = 0; k < top_k; ++k) {
                int exp_idx = expert_indices[b * static_cast<size_t>(top_k) + k];
                float gate_w = routing_weights[b * static_cast<size_t>(top_k) + k];

                const float* w_exp = expert_weights + (static_cast<size_t>(exp_idx) * static_cast<size_t>(out_features) * static_cast<size_t>(in_features));

                for (int oc = 0; oc < out_features; ++oc) {
                    const float* w_row = w_exp + (static_cast<size_t>(oc) * static_cast<size_t>(in_features));
                    float acc = 0.0f;
                    for (int ic = 0; ic < in_features; ++ic) {
                        acc += in_vec[ic] * w_row[ic];
                    }
                    out_vec[oc] += gate_w * acc;
                }
            }
        }
    });
}

} // namespace turing
