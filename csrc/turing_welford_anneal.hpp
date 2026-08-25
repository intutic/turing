#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace turing {

/**
 * Online Streaming Welford Statistical Accumulator.
 * Computes exact running mean and variance in a single streaming pass with zero numerical cancellation.
 */
class StreamingWelford {
public:
    StreamingWelford() : count_(0), mean_(0.0), m2_(0.0) {}

    void update(double x) {
        count_++;
        double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        double delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    void reset() {
        count_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
    }

    int64_t get_count() const { return count_; }
    double get_mean() const { return mean_; }
    double get_variance() const { return (count_ > 1) ? m2_ / static_cast<double>(count_ - 1) : 0.0; }
    double get_stdev() const { return std::sqrt(get_variance()); }

private:
    int64_t count_;
    double mean_;
    double m2_;
};

/**
 * Exponential Parameter Annealing Generator for Router Temperature and Sparsity.
 */
inline double evaluate_exponential_decay(double init_val, double min_val, int64_t current_step, int64_t max_steps) {
    if (max_steps <= 0) return min_val;
    if (current_step <= 0) return init_val;
    if (current_step >= max_steps) return min_val;

    double ratio = min_val / init_val;
    double progress = static_cast<double>(current_step) / static_cast<double>(max_steps);
    return init_val * std::pow(ratio, progress);
}

} // namespace turing
