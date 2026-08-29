#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace turing {

/**
 * Native C++20 High-Throughput Deterministic Token Block Hasher.
 * Provides 64-bit non-cryptographic avalanche hashing (xxHash64 / Murmur3 mix)
 * directly over contiguous uint32_t token buffers with 0 allocations.
 */
inline uint64_t deterministic_token_hash_cpp(
    const uint32_t* __restrict__ token_ids,
    size_t num_tokens,
    uint64_t seed = 0
) {
    // 64-bit mixing primes (xxHash64 constants)
    constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
    constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
    constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
    constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

    uint64_t h64 = seed + PRIME64_5 + (num_tokens * sizeof(uint32_t));

    size_t i = 0;
    // Process pairs of uint32 as uint64
    for (; i + 1 < num_tokens; i += 2) {
        uint64_t k1 = static_cast<uint64_t>(token_ids[i]) | (static_cast<uint64_t>(token_ids[i + 1]) << 32);
        k1 *= PRIME64_2;
        k1 = (k1 << 31) | (k1 >> 33);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = ((h64 << 27) | (h64 >> 37)) * PRIME64_1 + PRIME64_4;
    }

    // Trailing token
    if (i < num_tokens) {
        uint64_t k1 = static_cast<uint64_t>(token_ids[i]) * PRIME64_1;
        k1 = (k1 << 23) | (k1 >> 41);
        k1 *= PRIME64_2;
        h64 ^= k1;
        h64 = ((h64 << 11) | (h64 >> 53)) * PRIME64_1;
    }

    // Final avalanche mix
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

} // namespace turing
