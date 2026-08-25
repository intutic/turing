#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace turing {

/**
 * Fused In-SRAM Adam Optimizer Kernel.
 * Adapted from High-Performance HPC Suite (lr_threadpool.cc:275-291).
 * Updates parameters, first moments (m), and second moments (v) in a single pass.
 */
inline void fused_adam_step(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg_m,
    float* __restrict__ exp_avg_v,
    int dim,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep
) {
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

    for (int i = 0; i < dim; ++i) {
        float g = grad[i];
        
        // 1. Update biased first moment
        float m = beta1 * exp_avg_m[i] + (1.0f - beta1) * g;
        exp_avg_m[i] = m;

        // 2. Update biased second moment
        float v = beta2 * exp_avg_v[i] + (1.0f - beta2) * (g * g);
        exp_avg_v[i] = v;

        // 3. Bias corrections
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;

        // 4. Parameter update
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

} // namespace turing
