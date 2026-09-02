#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <numeric>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace turing {

/**
 * @brief Fast Zero-Copy Variable-Length Token Packing Engine.
 * Vectorizes the concatenation of variable-length token lists and builds cu_seqlens.
 */
struct VarlenPackResult {
    std::vector<int32_t> flat_tokens;
    std::vector<int32_t> cu_seqlens;
    size_t total_tokens;
    size_t num_sequences;
};

inline VarlenPackResult pack_varlen_tokens_simd(const std::vector<std::vector<int32_t>>& token_lists) {
    VarlenPackResult result;
    result.num_sequences = token_lists.size();
    result.cu_seqlens.resize(result.num_sequences + 1, 0);

    // 1. Single-pass cumulative length calculation
    size_t total_len = 0;
    for (size_t i = 0; i < token_lists.size(); ++i) {
        total_len += token_lists[i].size();
        result.cu_seqlens[i + 1] = static_cast<int32_t>(total_len);
    }
    result.total_tokens = total_len;
    result.flat_tokens.resize(total_len);

    // 2. Direct memory copy per sequence into contiguous destination buffer
    int32_t* dest_ptr = result.flat_tokens.data();
    for (const auto& tokens : token_lists) {
        if (!tokens.empty()) {
            std::memcpy(dest_ptr, tokens.data(), tokens.size() * sizeof(int32_t));
            dest_ptr += tokens.size();
        }
    }

    return result;
}

} // namespace turing
