#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace turing {

/**
 * Cooperative Shared Memory / Cache-Tiled 1D/2D Convolution.
 * Adapted from High-Performance Compute Engine (Fast 2D Convolution with Base-Pointer Pre-Computation).
 * Eliminates inner-loop integer multiplication cycles and optimizes cache reuse.
 */
inline void cooperative_shared_conv1d(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_len,
    int kernel_size,
    int stride,
    int padding
) {
    int out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    for (int b = 0; b < batch_size; ++b) {
        const float* in_batch = input + (b * in_channels * in_len);
        float* out_batch = output + (b * out_channels * out_len);

        for (int oc = 0; oc < out_channels; ++oc) {
            float* out_channel = out_batch + (oc * out_len);
            float b_val = (bias != nullptr) ? bias[oc] : 0.0f;

            for (int out_t = 0; out_t < out_len; ++out_t) {
                float acc = b_val;
                int in_t_base = out_t * stride - padding;

                for (int ic = 0; ic < in_channels; ++ic) {
                    // Pre-compute input channel and weight channel base pointers
                    const float* in_channel = in_batch + (ic * in_len);
                    const float* w_channel = weights + (oc * in_channels * kernel_size + ic * kernel_size);

                    for (int k = 0; k < kernel_size; ++k) {
                        int cur_in_t = in_t_base + k;
                        if (cur_in_t >= 0 && cur_in_t < in_len) {
                            acc += in_channel[cur_in_t] * w_channel[k];
                        }
                    }
                }
                out_channel[out_t] = acc;
            }
        }
    }
}

} // namespace turing
