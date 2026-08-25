#pragma once

#include "turing_simd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace turing {

class TuringPagedAttentionEngine {
private:
    int num_heads;
    int head_dim;
    int block_size;
    int total_blocks;
    size_t page_bytes;
    uint8_t* kv_pool_raw;

public:
    TuringPagedAttentionEngine(int heads, int h_dim, int b_size, int n_blocks)
        : num_heads(heads), head_dim(h_dim), block_size(b_size), total_blocks(n_blocks), kv_pool_raw(nullptr) {
        page_bytes = 2 * static_cast<size_t>(num_heads) * static_cast<size_t>(block_size) * static_cast<size_t>(head_dim) * sizeof(float);
        size_t total_bytes = static_cast<size_t>(total_blocks) * page_bytes;
        kv_pool_raw = static_cast<uint8_t*>(aligned_alloc_64(total_bytes));
        std::memset(kv_pool_raw, 0, total_bytes);
    }

    ~TuringPagedAttentionEngine() {
        if (kv_pool_raw) {
            aligned_free_64(kv_pool_raw);
            kv_pool_raw = nullptr;
        }
    }

    float* get_k_block(int physical_block_idx, int head_idx) {
        uint8_t* page_base = kv_pool_raw + (static_cast<size_t>(physical_block_idx) * page_bytes);
        float* k_base = reinterpret_cast<float*>(page_base);
        return k_base + (static_cast<size_t>(head_idx) * static_cast<size_t>(block_size) * static_cast<size_t>(head_dim));
    }

    float* get_v_block(int physical_block_idx, int head_idx) {
        uint8_t* page_base = kv_pool_raw + (static_cast<size_t>(physical_block_idx) * page_bytes);
        float* v_base = reinterpret_cast<float*>(page_base + (page_bytes / 2));
        return v_base + (static_cast<size_t>(head_idx) * static_cast<size_t>(block_size) * static_cast<size_t>(head_dim));
    }

    void forward_selective_attention(
        const float* query,             // [num_heads, head_dim]
        const int* block_table,         // [num_logical_pages]
        int num_logical_pages,
        uint32_t active_page_mask,      // Bitmask for selective page skipping
        float* output_context           // [num_heads, head_dim]
    ) {
        std::memset(output_context, 0, static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim) * sizeof(float));
        float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < num_heads; ++h) {
            const float* q_head = query + (static_cast<size_t>(h) * static_cast<size_t>(head_dim));
            float* out_head = output_context + (static_cast<size_t>(h) * static_cast<size_t>(head_dim));

            std::vector<float> scores(static_cast<size_t>(num_logical_pages) * static_cast<size_t>(block_size), -1e9f);
            float max_score = -1e9f;

            // Phase 1: Sparse QK^T
            for (int lp = 0; lp < num_logical_pages; ++lp) {
                if (!(active_page_mask & (1U << lp))) {
                    continue; // Skip inactive memory page
                }
                int pb = block_table[lp];
                const float* k_block = get_k_block(pb, h);

                for (int t = 0; t < block_size; ++t) {
                    const float* k_token = k_block + (static_cast<size_t>(t) * static_cast<size_t>(head_dim));
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_head[d] * k_token[d];
                    }
                    float score = dot * sm_scale;
                    scores[static_cast<size_t>(lp) * static_cast<size_t>(block_size) + static_cast<size_t>(t)] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }
            }

            // Phase 2: Softmax Normalization
            float sum_exp = 0.0f;
            for (int lp = 0; lp < num_logical_pages; ++lp) {
                if (!(active_page_mask & (1U << lp))) continue;
                for (int t = 0; t < block_size; ++t) {
                    float s = scores[static_cast<size_t>(lp) * static_cast<size_t>(block_size) + static_cast<size_t>(t)];
                    float exp_val = std::exp(s - max_score);
                    scores[static_cast<size_t>(lp) * static_cast<size_t>(block_size) + static_cast<size_t>(t)] = exp_val;
                    sum_exp += exp_val;
                }
            }
            float inv_sum = (sum_exp > 1e-8f) ? (1.0f / sum_exp) : 0.0f;

            // Phase 3: Sparse Value Gathering
            for (int lp = 0; lp < num_logical_pages; ++lp) {
                if (!(active_page_mask & (1U << lp))) continue;
                int pb = block_table[lp];
                const float* v_block = get_v_block(pb, h);

                for (int t = 0; t < block_size; ++t) {
                    float weight = scores[static_cast<size_t>(lp) * static_cast<size_t>(block_size) + static_cast<size_t>(t)] * inv_sum;
                    const float* v_token = v_block + (static_cast<size_t>(t) * static_cast<size_t>(head_dim));
                    for (int d = 0; d < head_dim; ++d) {
                        out_head[d] += weight * v_token[d];
                    }
                }
            }
        }
    }
};

} // namespace turing
