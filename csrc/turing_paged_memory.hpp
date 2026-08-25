#pragma once

#include <vector>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace turing {

/**
 * Native C++20 Hierarchical Virtual Memory Page Allocator.
 * Fast multi-tier compaction (512 Huge, 64 Medium, 16 Small) for KV cache tensors.
 */
class HierarchicalBitmapAllocator {
public:
    HierarchicalBitmapAllocator(int num_huge = 64, int num_medium = 128, int num_small = 256)
        : total_huge_(num_huge), total_medium_(num_medium), total_small_(num_small) {
        huge_free_.resize(num_huge, true);
        medium_free_.resize(num_medium, true);
        small_free_.resize(num_small, true);
    }

    // Returns vector of tuples: (tier_code [512, 64, 16], physical_block_id, valid_tokens)
    std::vector<std::tuple<int, int, int>> allocate_prompt(int prompt_len) {
        std::vector<std::tuple<int, int, int>> entries;
        int rem = prompt_len;

        // 1. Huge blocks (512 tokens)
        int h_idx = 0;
        while (rem >= 512 && h_idx < total_huge_) {
            if (huge_free_[h_idx]) {
                huge_free_[h_idx] = false;
                entries.emplace_back(512, h_idx, 512);
                rem -= 512;
            }
            h_idx++;
        }

        // 2. Medium blocks (64 tokens)
        int m_idx = 0;
        while (rem >= 64 && m_idx < total_medium_) {
            if (medium_free_[m_idx]) {
                medium_free_[m_idx] = false;
                entries.emplace_back(64, m_idx, 64);
                rem -= 64;
            }
            m_idx++;
        }

        // 3. Small blocks (16 tokens)
        int s_idx = 0;
        while (rem > 0 && s_idx < total_small_) {
            if (small_free_[s_idx]) {
                small_free_[s_idx] = false;
                int valid = std::min(rem, 16);
                entries.emplace_back(16, s_idx, valid);
                rem -= valid;
            }
            s_idx++;
        }

        return entries;
    }

    void free_block(int tier, int block_id) {
        if (tier == 512 && block_id >= 0 && block_id < total_huge_) {
            huge_free_[block_id] = true;
        } else if (tier == 64 && block_id >= 0 && block_id < total_medium_) {
            medium_free_[block_id] = true;
        } else if (tier == 16 && block_id >= 0 && block_id < total_small_) {
            small_free_[block_id] = true;
        }
    }

    int get_num_free(int tier) const {
        int count = 0;
        if (tier == 512) {
            for (bool f : huge_free_) if (f) count++;
        } else if (tier == 64) {
            for (bool f : medium_free_) if (f) count++;
        } else if (tier == 16) {
            for (bool f : small_free_) if (f) count++;
        }
        return count;
    }

private:
    int total_huge_;
    int total_medium_;
    int total_small_;
    std::vector<bool> huge_free_;
    std::vector<bool> medium_free_;
    std::vector<bool> small_free_;
};

} // namespace turing
