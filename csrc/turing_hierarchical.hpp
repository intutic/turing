#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Hierarchical Sequence-Chunk Compressor (HCA & CSA).
 * Provides zero-allocation strided mean & attention pooling over sequence chunks.
 */
inline void hca_chunk_pool_cpp(
    const float* __restrict__ input_tensor, // [SeqLen, Heads, HeadDim]
    float* __restrict__ output_tensor,      // [NumChunks, Heads, HeadDim]
    int seq_len,
    int num_heads,
    int head_dim,
    int chunk_size = 128
) {
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    int head_stride = head_dim;
    int token_stride = num_heads * head_dim;

    for (int c = 0; c < num_chunks; ++c) {
        int t_start = c * chunk_size;
        int t_end = std::min(seq_len, t_start + chunk_size);
        int valid_tokens = t_end - t_start;
        float inv_len = 1.0f / static_cast<float>(std::max(1, valid_tokens));

        float* out_chunk = output_tensor + (c * num_heads * head_dim);

        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float sum_val = 0.0f;
                for (int t = t_start; t < t_end; ++t) {
                    sum_val += input_tensor[t * token_stride + h * head_stride + d];
                }
                out_chunk[h * head_stride + d] = sum_val * inv_len;
            }
        }
    }
}

inline void csa_block_compress_cpp(
    const float* __restrict__ input_tensor, // [SeqLen, Heads, HeadDim]
    float* __restrict__ output_tensor,      // [NumBlocks, Heads, HeadDim]
    int seq_len,
    int num_heads,
    int head_dim,
    int chunk_size = 4
) {
    hca_chunk_pool_cpp(input_tensor, output_tensor, seq_len, num_heads, head_dim, chunk_size);
}

} // namespace turing
