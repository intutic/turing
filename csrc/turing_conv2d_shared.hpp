#pragma once

#include <vector>
#include <cstring>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Cooperative 2D Spatial Convolution (Spatial HPC Stencil Engine).
 */
inline void cooperative_conv2d_shared_cpp(
    const float* __restrict__ input,   // [InChannels, InH, InW]
    const float* __restrict__ weights, // [OutChannels, InChannels, KernelH, KernelW]
    const float* __restrict__ bias,    // [OutChannels] or nullptr
    float* __restrict__ output,        // [OutChannels, OutH, OutW]
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int kernel_h,
    int kernel_w,
    int stride = 1,
    int padding = 0
) {
    int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    for (int oc = 0; oc < out_channels; ++oc) {
        float b_val = (bias != nullptr) ? bias[oc] : 0.0f;
        const float* oc_weights = weights + (oc * in_channels * kernel_h * kernel_w);

        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = b_val;
                int ih_base = oh * stride - padding;
                int iw_base = ow * stride - padding;

                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* ic_weights = oc_weights + (ic * kernel_h * kernel_w);
                    const float* ic_input = input + (ic * in_h * in_w);

                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int ih = ih_base + kh;
                        if (ih < 0 || ih >= in_h) continue;

                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int iw = iw_base + kw;
                            if (iw < 0 || iw >= in_w) continue;

                            acc += ic_input[ih * in_w + iw] * ic_weights[kh * kernel_w + kw];
                        }
                    }
                }

                output[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
}

} // namespace turing
