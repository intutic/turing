#pragma once

#include "turing_simd.hpp"
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace turing {

/**
 * Fast SIMD Common Token Prefix Matcher.
 * Compares two integer token ID arrays (int32) at line rate.
 * Returns the length of the longest matching common prefix.
 */
inline int find_common_token_prefix(
    const int32_t* __restrict__ tokens_a,
    int len_a,
    const int32_t* __restrict__ tokens_b,
    int len_b
) {
    if (!tokens_a || !tokens_b || len_a <= 0 || len_b <= 0) {
        return 0;
    }

    int limit = std::min(len_a, len_b);
    int match_len = 0;

#if defined(TURING_HAS_AVX2)
    while (match_len + 7 < limit) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tokens_a + match_len));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tokens_b + match_len));
        __m256i eq = _mm256_cmpeq_epi32(a, b);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq));
        if (mask == 0xFF) {
            match_len += 8;
        } else {
            // First mismatch within these 8 integers
            for (int i = 0; i < 8; ++i) {
                if (tokens_a[match_len] != tokens_b[match_len]) {
                    return match_len;
                }
                match_len++;
            }
            return match_len;
        }
    }
#endif

    while (match_len < limit && tokens_a[match_len] == tokens_b[match_len]) {
        match_len++;
    }

    return match_len;
}

} // namespace turing
