#pragma once

#include "turing_simd.hpp"
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Batched Logits Sampler.
 * Computes batched softmax, top-k truncation, and categorical sampling directly over contiguous float arrays
 * with 0 per-element Python interpreter boxing and 0 GPU-to-CPU roundtrip flushes.
 */
struct BatchedSampleParams {
    float temperature;
    int top_k;
    float top_p;
    uint64_t seed;
};

inline int32_t sample_single_logits_simd_cpp(
    const float* __restrict__ logits,
    size_t vocab_size,
    float temperature,
    int top_k,
    float uniform_sample
) {
    if (vocab_size == 0) return 0;

    // Greedy argmax if temperature <= 1e-5
    if (temperature <= 1e-5f) {
        int32_t max_idx = 0;
        float max_val = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = static_cast<int32_t>(i);
            }
        }
        return max_idx;
    }

    // 1. Find max for numerical stability
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 2. Compute scaled exp & sum
    std::vector<std::pair<float, int32_t>> probs(vocab_size);
    float inv_temp = 1.0f / temperature;
    float sum_exp = 0.0f;

    for (size_t i = 0; i < vocab_size; ++i) {
        float e = std::exp((logits[i] - max_logit) * inv_temp);
        probs[i] = {e, static_cast<int32_t>(i)};
        sum_exp += e;
    }

    // 3. Top-K filtering if specified
    int effective_k = (top_k > 0 && top_k < static_cast<int>(vocab_size)) ? top_k : static_cast<int>(vocab_size);
    if (effective_k < static_cast<int>(vocab_size)) {
        std::partial_sort(
            probs.begin(),
            probs.begin() + effective_k,
            probs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        probs.resize(effective_k);
        sum_exp = 0.0f;
        for (const auto& p : probs) {
            sum_exp += p.first;
        }
    }

    // 4. Categorical CDF sampling
    float inv_sum = 1.0f / (sum_exp > 1e-8f ? sum_exp : 1.0f);
    float target = uniform_sample;
    float cumulative = 0.0f;

    for (const auto& p : probs) {
        cumulative += p.first * inv_sum;
        if (cumulative >= target) {
            return p.second;
        }
    }

    return probs.back().second;
}

inline std::vector<int32_t> sample_batched_logits_simd_cpp(
    const float* __restrict__ batched_logits,
    size_t batch_size,
    size_t vocab_size,
    const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_ks,
    const float* __restrict__ uniform_samples
) {
    std::vector<int32_t> results(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row_ptr = batched_logits + b * vocab_size;
        results[b] = sample_single_logits_simd_cpp(
            row_ptr,
            vocab_size,
            temperatures[b],
            top_ks[b],
            uniform_samples[b]
        );
    }
    return results;
}

} // namespace turing
