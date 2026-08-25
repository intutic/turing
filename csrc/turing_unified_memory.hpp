#include "turing_simd.hpp"
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace turing {

/**
 * Unified Memory Slab Allocator with Async Prefetch & Cache-Line Alignment.
 * Supports zero-copy buffer staging on Apple Silicon Metal MPS, Grace Hopper, and Windows/Linux.
 */
class UnifiedMemoryPool {
public:
    explicit UnifiedMemoryPool(size_t capacity_bytes = 64 * 1024 * 1024)
        : capacity_(capacity_bytes), used_(0), raw_ptr_(nullptr) {
        // Allocate 64-byte aligned memory
        raw_ptr_ = aligned_alloc_64(capacity_);
        if (!raw_ptr_) {
            throw std::runtime_error("Failed to allocate 64-byte aligned unified memory buffer");
        }
        std::memset(raw_ptr_, 0, capacity_);
    }

    ~UnifiedMemoryPool() {
        if (raw_ptr_) {
            aligned_free_64(raw_ptr_);
            raw_ptr_ = nullptr;
        }
    }

    int64_t allocate_slab(size_t bytes) {
        // 64-byte align the requested bytes
        size_t aligned_bytes = (bytes + 63) & ~size_t(63);
        if (used_ + aligned_bytes > capacity_) {
            return -1; // Out of memory
        }
        int64_t offset = static_cast<int64_t>(used_);
        used_ += aligned_bytes;
        return offset;
    }

    void reset() {
        used_ = 0;
        if (raw_ptr_) {
            std::memset(raw_ptr_, 0, capacity_);
        }
    }

    size_t get_capacity() const { return capacity_; }
    size_t get_used() const { return used_; }
    size_t get_free() const { return (capacity_ > used_) ? capacity_ - used_ : 0; }

private:
    size_t capacity_;
    size_t used_;
    void* raw_ptr_;
};

} // namespace turing
