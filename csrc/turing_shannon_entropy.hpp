#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Fused Online Shannon Entropy Kernel.
 * Computes exact entropy H(P) = -\sum P_i \ln P_i from unnormalized logits in a single numerical pass.
 */
inline float compute_shannon_entropy_single(const float* logits, int vocab_size) {
    if (vocab_size <= 0) return 0.0f;

    // 1. Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // 2. Accumulate sum(exp(x - m)) and sum(exp(x - m) * x)
    double sum_exp = 0.0;
    double sum_exp_x = 0.0;

    for (int i = 0; i < vocab_size; ++i) {
        double e = std::exp(static_cast<double>(logits[i] - max_val));
        sum_exp += e;
        sum_exp_x += e * static_cast<double>(logits[i]);
    }

    if (sum_exp <= 0.0) return 0.0f;

    double log_z = max_val + std::log(sum_exp);
    double entropy = log_z - (sum_exp_x / sum_exp);
    return static_cast<float>(std::max(0.0, entropy));
}

inline std::vector<float> compute_shannon_entropy_batch_cpp(
    const float* logits,
    int batch_size,
    int vocab_size
) {
    std::vector<float> entropies(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        entropies[b] = compute_shannon_entropy_single(logits + (b * vocab_size), vocab_size);
    }
    return entropies;
}

} // namespace turing
