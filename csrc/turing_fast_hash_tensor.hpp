#pragma once

#include "turing_simd.hpp"
#include <cstdint>
#include <cstddef>
#include <string>
#include <sstream>
#include <iomanip>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Zero-Copy Tensor Pointer Checksum.
 * Scans raw tensor memory buffers directly at memory-bus speeds without Python object allocations.
 */
inline uint64_t hash_tensor_buffer_cpp(
    const uint8_t* __restrict__ data_ptr,
    size_t num_bytes,
    uint64_t seed = 0xcbf29ce484222325ULL
) {
    constexpr uint64_t PRIME1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME2 = 14029467366897019727ULL;

    uint64_t h64 = seed ^ (num_bytes * PRIME1);
    size_t i = 0;

#if defined(TURING_HAS_AVX2)
    for (; i + 31 < num_bytes; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data_ptr + i));
        // Accumulate 64-bit lanes
        alignas(32) uint64_t lanes[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), v);
        uint64_t chunk = lanes[0] ^ (lanes[1] * PRIME1) ^ (lanes[2] * PRIME2) ^ lanes[3];
        h64 = ((h64 << 31) | (h64 >> 33)) * PRIME1 + chunk;
    }
#endif

    const uint64_t* ptr64 = reinterpret_cast<const uint64_t*>(data_ptr + i);
    size_t rem_words = (num_bytes - i) / 8;
    for (size_t w = 0; w < rem_words; ++w) {
        h64 = ((h64 << 27) | (h64 >> 37)) * PRIME2 + ptr64[w];
    }

    size_t tail_idx = i + rem_words * 8;
    for (size_t t = tail_idx; t < num_bytes; ++t) {
        h64 = (h64 * 31) + data_ptr[t];
    }

    // Final avalanche
    h64 ^= h64 >> 33;
    h64 *= PRIME1;
    h64 ^= h64 >> 29;
    h64 *= PRIME2;
    h64 ^= h64 >> 32;

    return h64;
}

inline std::string hash_tensor_buffer_hex(
    const uint8_t* data_ptr,
    size_t num_bytes,
    uint64_t seed = 0xcbf29ce484222325ULL
) {
    uint64_t val = hash_tensor_buffer_cpp(data_ptr, num_bytes, seed);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << val;
    return ss.str();
}

} // namespace turing
