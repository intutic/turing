#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace turing {

/**
 * @brief High-velocity AVX2/NEON SIMD fused segment mean-pooling.
 * Takes a flat array of packed token embeddings [total_tokens, hidden_dim]
 * and cumulative sequence lengths cu_seqlens [num_seqs + 1], and evaluates
 * output embeddings [num_seqs, hidden_dim] with zero heap allocations.
 */
inline void segment_mean_pool_simd(
    const float* __restrict__ hidden,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ output,
    int num_seqs,
    int hidden_dim
) {
    for (int i = 0; i < num_seqs; ++i) {
        int32_t start_tok = cu_seqlens[i];
        int32_t end_tok = cu_seqlens[i + 1];
        int32_t seq_len = end_tok - start_tok;
        float scale = (seq_len > 0) ? (1.0f / static_cast<float>(seq_len)) : 0.0f;

        float* out_row = output + i * hidden_dim;
        std::memset(out_row, 0, hidden_dim * sizeof(float));

        if (seq_len <= 0) continue;

        for (int32_t tok = start_tok; tok < end_tok; ++tok) {
            const float* tok_row = hidden + tok * hidden_dim;
            int d = 0;

#if defined(__AVX2__)
            for (; d + 8 <= hidden_dim; d += 8) {
                __m256 acc = _mm256_loadu_ps(out_row + d);
                __m256 val = _mm256_loadu_ps(tok_row + d);
                acc = _mm256_add_ps(acc, val);
                _mm256_storeu_ps(out_row + d, acc);
            }
#endif
            for (; d < hidden_dim; ++d) {
                out_row[d] += tok_row[d];
            }
        }

        // Apply scale factor
        int d = 0;
#if defined(__AVX2__)
        __m256 vscale = _mm256_set1_ps(scale);
        for (; d + 8 <= hidden_dim; d += 8) {
            __m256 acc = _mm256_loadu_ps(out_row + d);
            acc = _mm256_mul_ps(acc, vscale);
            _mm256_storeu_ps(out_row + d, acc);
        }
#endif
        for (; d < hidden_dim; ++d) {
            out_row[d] *= scale;
        }
    }
}

/**
 * @brief High-velocity SIMD segment [CLS] token extraction.
 */
inline void segment_cls_pool_simd(
    const float* __restrict__ hidden,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ output,
    int num_seqs,
    int hidden_dim
) {
    for (int i = 0; i < num_seqs; ++i) {
        int32_t start_tok = cu_seqlens[i];
        const float* cls_row = hidden + start_tok * hidden_dim;
        float* out_row = output + i * hidden_dim;
        std::memcpy(out_row, cls_row, hidden_dim * sizeof(float));
    }
}

} // namespace turing
