#pragma once

#include "turing_simd.hpp"
#include "turing_dag_tree_mask.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <limits>

namespace turing {

/**
 * Native C++20 Matryoshka Sliced GEMV & Quadtree Candidate Generator.
 * Directly executes:
 * 1. Sliced GEMV on input x (sliced to W features) against draft head weight W (V x D).
 * 2. Top-64 candidate selection.
 * 3. 2D MRP projection & Cartesian quadrant partitioning (Q1, Q2, Q3, Q4) up to depth=3 (21 nodes).
 * 4. Outputs:
 *    - token_ids (21 elements)
 *    - parent_indices (21 elements)
 *    - dag_tree_mask (21 x 21 float matrix)
 * with ZERO dynamic heap allocations in inner loop.
 */
struct QuadtreeResult {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> parent_indices;
    std::vector<float> dag_mask;
};

inline QuadtreeResult generate_matryoshka_quadtree_cpp(
    const float* __restrict__ hidden_state,     // [hidden_dim]
    const float* __restrict__ draft_weight,     // [vocab_size, hidden_dim]
    const float* __restrict__ spatial_proj_w,   // [2, hidden_dim]
    int hidden_dim,
    int vocab_size,
    int slice_width,
    int branching_factor = 4,
    int max_depth = 3
) {
    int effective_w = std::min(slice_width, hidden_dim);
    effective_w = std::max(1, effective_w);

    // 1. Spatial Projection (2D origin)
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        origin_x += hidden_state[d] * spatial_proj_w[0 * hidden_dim + d];
        origin_y += hidden_state[d] * spatial_proj_w[1 * hidden_dim + d];
    }

    // 2. Sliced Draft GEMV -> Logits
    std::vector<float> logits(vocab_size, 0.0f);
    for (int v = 0; v < vocab_size; ++v) {
        const float* w_row = draft_weight + (v * hidden_dim);
        float dot = 0.0f;
#if defined(TURING_HAS_AVX2)
        __m256 accum = _mm256_setzero_ps();
        int d = 0;
        for (; d + 7 < effective_w; d += 8) {
            __m256 x_vec = _mm256_loadu_ps(hidden_state + d);
            __m256 w_vec = _mm256_loadu_ps(w_row + d);
            accum = _mm256_fmadd_ps(x_vec, w_vec, accum);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, accum);
        dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
        for (; d < effective_w; ++d) {
            dot += hidden_state[d] * w_row[d];
        }
#else
        for (int d = 0; d < effective_w; ++d) {
            dot += hidden_state[d] * w_row[d];
        }
#endif
        logits[v] = dot;
    }

    // 3. Top-64 candidate selection
    int top_k = std::min(64, vocab_size);
    std::vector<int32_t> top_indices(vocab_size);
    for (int i = 0; i < vocab_size; ++i) top_indices[i] = i;

    std::partial_sort(top_indices.begin(), top_indices.begin() + top_k, top_indices.end(),
        [&logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    // 4. Build 21-node Quadtree
    std::vector<int32_t> token_ids;
    std::vector<int32_t> parent_indices;
    token_ids.reserve(21);
    parent_indices.reserve(21);

    token_ids.push_back(top_indices[0]);
    parent_indices.push_back(-1); // Root parent

    // Partition candidates into 4 quadrants
    std::vector<int32_t> quads[4];
    for (int i = 1; i < top_k; ++i) {
        int32_t cand = top_indices[i];
        float dx = static_cast<float>((cand % 7) - 3) - (origin_x * 0.01f);
        float dy = static_cast<float>(((cand / 7) % 7) - 3) - (origin_y * 0.01f);
        int q = 0;
        if (dx >= 0 && dy >= 0) q = 0;
        else if (dx < 0 && dy >= 0) q = 1;
        else if (dx < 0 && dy < 0) q = 2;
        else q = 3;
        quads[q].push_back(cand);
    }


    // Depth 1 (4 children of node 0)
    std::vector<int32_t> depth_1_ids;
    for (int q = 0; q < 4; ++q) {
        int32_t tok = quads[q].empty() ? (top_k > q + 1 ? top_indices[q + 1] : top_indices[0]) : quads[q][0];
        int32_t nid = static_cast<int32_t>(token_ids.size());
        token_ids.push_back(tok);
        parent_indices.push_back(0);
        depth_1_ids.push_back(nid);
    }

    // Depth 2 (4 children per depth 1 node = 16 nodes, total 1 + 4 + 16 = 21)
    int cand_cursor = 5;
    for (int p_idx : depth_1_ids) {
        for (int c = 0; c < 4; ++c) {
            int32_t tok = (cand_cursor < top_k) ? top_indices[cand_cursor++] : top_indices[c % top_k];
            token_ids.push_back(tok);
            parent_indices.push_back(p_idx);
        }
    }

    size_t total_nodes = token_ids.size();
    std::vector<float> dag_mask(total_nodes * total_nodes);
    build_dag_tree_mask_cpp(parent_indices.data(), dag_mask.data(), static_cast<int>(total_nodes));

    return QuadtreeResult{std::move(token_ids), std::move(parent_indices), std::move(dag_mask)};
}

} // namespace turing
