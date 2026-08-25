#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

namespace turing {

/**
 * Native C++20 DAG Tree Attention Mask Engine.
 * Constructs an [N, N] additive DAG tree attention mask using 64-bit ancestor bitboards.
 */
inline void build_dag_tree_mask_cpp(
    const int32_t* __restrict__ parent_indices,
    float* __restrict__ output_mask,
    int num_nodes
) {
    std::vector<uint64_t> ancestors(num_nodes, 0ULL);
    float neg_inf = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_nodes; ++i) {
        int p = parent_indices[i];
        if (p >= 0 && p < num_nodes) {
            ancestors[i] = ancestors[p] | (1ULL << p) | (1ULL << i);
        } else {
            ancestors[i] = (1ULL << i);
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        uint64_t anc_i = ancestors[i];
        for (int j = 0; j < num_nodes; ++j) {
            output_mask[i * num_nodes + j] = ((anc_i >> j) & 1ULL) ? 0.0f : neg_inf;
        }
    }
}

} // namespace turing
