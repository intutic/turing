#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(_WIN32) || defined(_MSC_VER)
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#define TURING_HAS_AVX2 1
#endif

namespace turing {

/**
 * @brief 1D Depthwise Causal Dilated Convolution with circular history buffer.
 * Performs y[b, t, c] = sum_{k=0}^{K-1} w[c, k] * x[b, t - k * dilation, c]
 * Causal padding ensures t - k * dilation < 0 fetches 0.0f.
 * Uses AVX2 _mm256_fmadd_ps when available with 8 channels per SIMD lane.
 */
inline void dilated_causal_conv1d_simd(
    const float* __restrict__ input,      // [Batch, SeqLen, Channels]
    const float* __restrict__ weights,    // [Channels, KernelSize]
    float* __restrict__ output,           // [Batch, SeqLen, Channels]
    int batch,
    int seq_len,
    int channels,
    int kernel_size,
    int dilation
) {
    for (int b = 0; b < batch; ++b) {
        const float* in_batch = input + b * seq_len * channels;
        float* out_batch = output + b * seq_len * channels;

        for (int t = 0; t < seq_len; ++t) {
            float* out_t = out_batch + t * channels;

            // Zero-initialize output for this time step
            std::memset(out_t, 0, channels * sizeof(float));

            for (int k = 0; k < kernel_size; ++k) {
                int src_t = t - (kernel_size - 1 - k) * dilation;
                if (src_t < 0) continue; // Causal zero-padding


                const float* in_k = in_batch + src_t * channels;
                int c = 0;

#if defined(__AVX2__)
                for (; c + 8 <= channels; c += 8) {
                    __m256 in_vec = _mm256_loadu_ps(in_k + c);
                    
                    // Load weights for these 8 channels at tap k
                    alignas(32) float w_tmp[8];
                    for (int i = 0; i < 8; ++i) {
                        w_tmp[i] = weights[(c + i) * kernel_size + k];
                    }
                    __m256 w_vec = _mm256_load_ps(w_tmp);

                    __m256 out_vec = _mm256_loadu_ps(out_t + c);
                    out_vec = _mm256_fmadd_ps(in_vec, w_vec, out_vec);
                    _mm256_storeu_ps(out_t + c, out_vec);
                }
#endif

                // Scalar cleanup
                for (; c < channels; ++c) {
                    float w = weights[c * kernel_size + k];
                    out_t[c] += in_k[c] * w;
                }
            }
        }
    }
}

} // namespace turing
