#pragma once

#include <cstdint>
#include <cstddef>

namespace turing {

/**
 * Native C++20 64-bit MurmurHash3 for Attention Pattern Cache (APC) & Sparsity Bitmasks.
 */
inline uint64_t apc_murmurhash64A(const void* key, size_t len, uint64_t seed = 0x9747b28c) {
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t* data = static_cast<const uint64_t*>(key);
    const uint64_t* end = data + (len / 8);

    while (data != end) {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const uint8_t* data2 = reinterpret_cast<const uint8_t*>(data);

    switch (len & 7) {
        case 7: h ^= static_cast<uint64_t>(data2[6]) << 48; [[fallthrough]];
        case 6: h ^= static_cast<uint64_t>(data2[5]) << 40; [[fallthrough]];
        case 5: h ^= static_cast<uint64_t>(data2[4]) << 32; [[fallthrough]];
        case 4: h ^= static_cast<uint64_t>(data2[3]) << 24; [[fallthrough]];
        case 3: h ^= static_cast<uint64_t>(data2[2]) << 16; [[fallthrough]];
        case 2: h ^= static_cast<uint64_t>(data2[1]) << 8;  [[fallthrough]];
        case 1: h ^= static_cast<uint64_t>(data2[0]);
                h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

} // namespace turing
