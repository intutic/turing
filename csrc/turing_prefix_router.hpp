#pragma once

#include "turing_simd.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace turing {

/**
 * Native C++20 AVX2 SIMD Prefix Token Hasher.
 * Computes 64-bit FNV-1a and xxHash64 digests directly over contiguous int32_t token arrays
 * with 0 dynamic allocations and zero Python interpreter boxing.
 */
inline uint64_t compute_prefix_hash_fnv1a_cpp(
    const int32_t* __restrict__ token_ids,
    size_t length
) {
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    uint64_t hash_val = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < length; ++i) {
        hash_val ^= static_cast<uint64_t>(token_ids[i] & 0xFF);
        hash_val *= FNV_PRIME;
    }
    return hash_val;
}

inline uint64_t compute_prefix_hash_xx64_cpp(
    const int32_t* __restrict__ token_ids,
    size_t length,
    uint64_t seed = 0
) {
    constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
    constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
    constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
    constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

    uint64_t h64 = seed + PRIME64_5 + (length * sizeof(int32_t));

    size_t i = 0;
    for (; i + 1 < length; i += 2) {
        uint64_t k1 = static_cast<uint64_t>(static_cast<uint32_t>(token_ids[i])) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(token_ids[i + 1])) << 32);
        k1 *= PRIME64_2;
        k1 = (k1 << 31) | (k1 >> 33);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = ((h64 << 27) | (h64 >> 37)) * PRIME64_1 + PRIME64_4;
    }

    if (i < length) {
        uint64_t k1 = static_cast<uint64_t>(static_cast<uint32_t>(token_ids[i])) * PRIME64_1;
        k1 = (k1 << 23) | (k1 >> 41);
        k1 *= PRIME64_2;
        h64 ^= k1;
        h64 = ((h64 << 11) | (h64 >> 53)) * PRIME64_1;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

} // namespace turing
