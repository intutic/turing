#pragma once

#include "turing_simd.hpp"
#include <cstdint>
#include <cstddef>
#include <tuple>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Speculative Parity Verifier.
 * Compares 8 token IDs per instruction cycle using _mm256_cmpeq_epi32
 * with 0 dynamic allocations and instantaneous divergence detection.
 */
struct SpecParityResult {
    bool passed;
    size_t num_compared;
    int divergence_index; // -1 if passed
};

inline SpecParityResult verify_greedy_parity_simd_cpp(
    const int32_t* __restrict__ spec_tokens,
    const int32_t* __restrict__ plain_tokens,
    size_t count
) {
    size_t i = 0;

#if defined(TURING_HAS_AVX2)
    for (; i + 7 < count; i += 8) {
        __m256i v_spec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(spec_tokens + i));
        __m256i v_plain = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(plain_tokens + i));
        __m256i cmp = _mm256_cmpeq_epi32(v_spec, v_plain);
        int mask = _mm256_movemask_epi8(cmp);

        // If all 32 bytes match, mask == -1 (0xFFFFFFFF)
        if (mask != static_cast<int>(0xFFFFFFFF)) {
            for (size_t k = 0; k < 8; ++k) {
                if (spec_tokens[i + k] != plain_tokens[i + k]) {
                    return {false, i + k + 1, static_cast<int>(i + k)};
                }
            }
        }
    }
#endif

    for (; i < count; ++i) {
        if (spec_tokens[i] != plain_tokens[i]) {
            return {false, i + 1, static_cast<int>(i)};
        }
    }

    return {true, count, -1};
}

} // namespace turing
