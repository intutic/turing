#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace turing {

/**
 * Native C++20 Entropic Optimal Transport (OT) KV Cache Eviction Kernel.
 * Computes Gibbs cost matrix, Sinkhorn-Knopp scaling, and top-k marginal key mass extraction.
 */
inline void sinkhorn_ot_eviction_cpp(
    const float* __restrict__ query,       // [M, D]
    const float* __restrict__ key,         // [N, D]
    int32_t* __restrict__ retained_indices,// [budget]
    float* __restrict__ key_mass_out,      // [N]
    int m_queries,
    int n_keys,
    int head_dim,
    int budget,
    float epsilon = 0.05f,
    int num_iters = 15
) {
    if (n_keys <= budget) {
        for (int i = 0; i < n_keys; ++i) {
            retained_indices[i] = i;
            key_mass_out[i] = 1.0f / static_cast<float>(std::max(1, n_keys));
        }
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int total_elements = m_queries * n_keys;
    std::vector<float> gibbs(total_elements, 0.0f);

    // 1. Cost Matrix: C_ij = - (q_i . k_j) * scale
    float min_cost = 1e9f;
    for (int i = 0; i < m_queries; ++i) {
        const float* q_vec = query + (i * head_dim);
        for (int j = 0; j < n_keys; ++j) {
            const float* k_vec = key + (j * head_dim);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float c = -dot * scale;
            gibbs[i * n_keys + j] = c;
            if (c < min_cost) min_cost = c;
        }
    }

    // 2. Gibbs Kernel: K_ij = exp(-(C_ij - min_C) / epsilon)
    float inv_eps = 1.0f / (epsilon > 1e-6f ? epsilon : 1e-6f);
    for (int idx = 0; idx < total_elements; ++idx) {
        gibbs[idx] = std::exp(-(gibbs[idx] - min_cost) * inv_eps);
    }

    // 3. Sinkhorn iterations
    std::vector<float> u(m_queries, 1.0f / static_cast<float>(m_queries));
    std::vector<float> v(n_keys, 1.0f / static_cast<float>(n_keys));
    std::vector<float> kv(m_queries, 0.0f);
    std::vector<float> ktu(n_keys, 0.0f);

    for (int it = 0; it < num_iters; ++it) {
        // kv = K @ v
        for (int i = 0; i < m_queries; ++i) {
            float sum_val = 0.0f;
            const float* row = gibbs.data() + (i * n_keys);
            for (int j = 0; j < n_keys; ++j) {
                sum_val += row[j] * v[j];
            }
            kv[i] = std::max(1e-8f, sum_val);
            u[i] = (1.0f / static_cast<float>(m_queries)) / kv[i];
        }

        // ktu = K.T @ u
        for (int j = 0; j < n_keys; ++j) {
            float sum_val = 0.0f;
            for (int i = 0; i < m_queries; ++i) {
                sum_val += gibbs[i * n_keys + j] * u[i];
            }
            ktu[j] = std::max(1e-8f, sum_val);
            v[j] = (1.0f / static_cast<float>(n_keys)) / ktu[j];
        }
    }

    // 4. Marginal key mass: m_j = sum_i (u_i * K_ij * v_j)
    for (int j = 0; j < n_keys; ++j) {
        float mass = 0.0f;
        for (int i = 0; i < m_queries; ++i) {
            mass += u[i] * gibbs[i * n_keys + j] * v[j];
        }
        key_mass_out[j] = mass;
    }

    // 5. Top-K selection
    std::vector<std::pair<float, int32_t>> score_indices(n_keys);
    for (int j = 0; j < n_keys; ++j) {
        score_indices[j] = {key_mass_out[j], static_cast<int32_t>(j)};
    }
    std::partial_sort(
        score_indices.begin(),
        score_indices.begin() + budget,
        score_indices.end(),
        [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
            return a.first > b.first;
        }
    );

    std::vector<int32_t> chosen;
    chosen.reserve(budget);
    for (int k = 0; k < budget; ++k) {
        chosen.push_back(score_indices[k].second);
    }
    std::sort(chosen.begin(), chosen.end());

    for (int k = 0; k < budget; ++k) {
        retained_indices[k] = chosen[k];
    }
}

} // namespace turing
