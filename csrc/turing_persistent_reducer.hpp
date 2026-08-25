#pragma once

#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace turing {

/**
 * Cache-Aligned Persistent Thread-Local Reduction Buffer.
 * Eliminates dynamic thread allocation and lock contention during parallel CPU GEMV and logit reductions.
 */
class PersistentThreadReducer {
public:
    PersistentThreadReducer(int num_threads = 4, int dim = 256)
        : num_threads_(num_threads), dim_(dim) {
        // Allocate contiguous per-thread aligned memory
        buffer_.resize(static_cast<size_t>(num_threads_) * static_cast<size_t>(dim_), 0.0f);
    }

    float* get_thread_buffer(int tid) {
        if (tid < 0 || tid >= num_threads_) return nullptr;
        return buffer_.data() + (tid * dim_);
    }

    void clear(int tid) {
        float* ptr = get_thread_buffer(tid);
        if (ptr) {
            std::memset(ptr, 0, dim_ * sizeof(float));
        }
    }

    void clear_all() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    }

    void reduce_sum(float* out_target) const {
        if (!out_target) return;
        std::memset(out_target, 0, dim_ * sizeof(float));

        for (int t = 0; t < num_threads_; ++t) {
            const float* src = buffer_.data() + (t * dim_);
            for (int i = 0; i < dim_; ++i) {
                out_target[i] += src[i];
            }
        }
    }

    int get_num_threads() const { return num_threads_; }
    int get_dim() const { return dim_; }

private:
    int num_threads_;
    int dim_;
    std::vector<float> buffer_;
};

inline std::vector<float> parallel_reduce_sum_cpp(
    const float* thread_data,
    int num_threads,
    int dim
) {
    std::vector<float> out(dim, 0.0f);
    for (int t = 0; t < num_threads; ++t) {
        const float* src = thread_data + (t * dim);
        for (int i = 0; i < dim; ++i) {
            out[i] += src[i];
        }
    }
    return out;
}

} // namespace turing
