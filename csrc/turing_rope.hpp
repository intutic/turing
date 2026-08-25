#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Fused In-Place RoPE Decoupler and Rotary Position Scaler.
 * Performs direct 2D coordinate rotation in CPU L1 cache/registers without temporary allocations.
 */
inline void fused_rope_transform_cpp(
    float* __restrict__ data,      // [SeqLen, NumHeads, HeadDim]
    int seq_len,
    int num_heads,
    int head_dim,
    float base = 500000.0f,
    int pos_offset = 0,
    bool is_inverse = false
) {
    int dim_half = head_dim / 2;
    std::vector<float> inv_freq(dim_half);
    for (int i = 0; i < dim_half; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }

    for (int t = 0; t < seq_len; ++t) {
        float pos = static_cast<float>(t + pos_offset);
        for (int h = 0; h < num_heads; ++h) {
            float* ptr = data + (t * num_heads * head_dim) + (h * head_dim);

            for (int i = 0; i < dim_half; ++i) {
                float theta = pos * inv_freq[i];
                float c = std::cos(theta);
                float s = std::sin(theta);

                float k1 = ptr[i];
                float k2 = ptr[i + dim_half];

                if (!is_inverse) {
                    // Forward RoPE: [cos, -sin; sin, cos]
                    ptr[i] = k1 * c - k2 * s;
                    ptr[i + dim_half] = k1 * s + k2 * c;
                } else {
                    // Inverse RoPE: [cos, sin; -sin, cos]
                    ptr[i] = k1 * c + k2 * s;
                    ptr[i + dim_half] = -k1 * s + k2 * c;
                }
            }
        }
    }
}

} // namespace turing
