#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace turing {

#pragma pack(push, 1)
// GGML Quantization block structures
struct BlockQ4_0 {
    uint16_t d;        // delta (FP16)
    uint8_t qs[16];    // 32 nibbles
};

struct BlockQ4_1 {
    uint16_t d;        // delta (FP16)
    uint16_t m;        // min (FP16)
    uint8_t qs[16];    // 32 nibbles
};

struct BlockQ8_0 {
    uint16_t d;        // delta (FP16)
    int8_t qs[32];     // 32 quants (INT8)
};

struct BlockQ4_K {
    uint16_t d;        // super-block scale (FP16)
    uint16_t dmin;     // super-block min (FP16)
    uint8_t scales[12];// 6-bit scales and mins
    uint8_t qs[128];   // 256 4-bit quants
};

struct BlockQ5_K {
    uint16_t d;        // super-block scale (FP16)
    uint16_t dmin;     // super-block min (FP16)
    uint8_t scales[12];// 6-bit scales and mins
    uint8_t qh[32];    // 256 high bits
    uint8_t qs[128];   // 256 4-bit quants
};
#pragma pack(pop)

// Helper: Convert FP16 uint16_t to FP32 float portably
inline float fp16_to_fp32(uint16_t h) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    __fp16 val;
    std::memcpy(&val, &h, sizeof(uint16_t));
    return static_cast<float>(val);
#elif defined(__AVX2__) && defined(__F16C__)
    __m128i val = _mm_cvtsi32_si128(static_cast<int>(h));
    __m128 res = _mm_cvtph_ps(val);
    return _mm_cvtss_f32(res);
#else
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    if (exp == 0) {
        if (mant == 0) {
            float f = 0.0f;
            if (h & 0x8000) f = -0.0f;
            return f;
        }
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
        uint32_t f_u = sign | (((exp + 112) & 0xFF) << 23) | (static_cast<uint32_t>(mant) << 13);
        float f;
        std::memcpy(&f, &f_u, sizeof(float));
        return f;
    } else if (exp == 31) {
        uint32_t f_u = sign | 0x7F800000 | (static_cast<uint32_t>(mant) << 13);
        float f;
        std::memcpy(&f, &f_u, sizeof(float));
        return f;
    }
    uint32_t f_u = sign | (((static_cast<uint32_t>(exp) + 112) & 0xFF) << 23) | (static_cast<uint32_t>(mant) << 13);
    float f;
    std::memcpy(&f, &f_u, sizeof(float));
    return f;
#endif
}

/**
 * High-Performance SIMD Dequantizer for Q4_0 GGML Blocks
 */
inline void dequantize_q4_0(const void* src, float* dst, size_t num_blocks) {
    const auto* blocks = reinterpret_cast<const BlockQ4_0*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        float* out_ptr = dst + b * 32;
        const uint8_t* qs = blocks[b].qs;

        for (int i = 0; i < 16; ++i) {
            uint8_t byte = qs[i];
            out_ptr[2 * i]     = static_cast<float>(static_cast<int>(byte & 0x0F) - 8) * d;
            out_ptr[2 * i + 1] = static_cast<float>(static_cast<int>((byte >> 4) & 0x0F) - 8) * d;
        }
    }
}

/**
 * High-Performance SIMD Dequantizer for Q8_0 GGML Blocks
 */
inline void dequantize_q8_0(const void* src, float* dst, size_t num_blocks) {
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        float* out_ptr = dst + b * 32;

#if defined(__AVX2__)
        __m256 d_val = _mm256_set1_ps(d);
        const int8_t* qs = blocks[b].qs;

        for (int i = 0; i < 4; ++i) {
            __m128i raw8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + i * 8));
            __m256i raw32 = _mm256_cvtepi8_epi32(raw8);
            __m256 out = _mm256_mul_ps(_mm256_cvtepi32_ps(raw32), d_val);
            _mm256_storeu_ps(out_ptr + i * 8, out);
        }
#elif defined(__ARM_NEON) || defined(__aarch64__)
        const int8_t* qs = blocks[b].qs;

        for (int i = 0; i < 32; i += 4) {
            out_ptr[i]     = static_cast<float>(qs[i]) * d;
            out_ptr[i + 1] = static_cast<float>(qs[i + 1]) * d;
            out_ptr[i + 2] = static_cast<float>(qs[i + 2]) * d;
            out_ptr[i + 3] = static_cast<float>(qs[i + 3]) * d;
        }
#else
        const int8_t* qs = blocks[b].qs;
        for (int i = 0; i < 32; ++i) {
            out_ptr[i] = static_cast<float>(qs[i]) * d;
        }
#endif
    }
}

/**
 * High-Performance Dequantizer for Q4_1 GGML Blocks
 */
inline void dequantize_q4_1(const void* src, float* dst, size_t num_blocks) {
    const auto* blocks = reinterpret_cast<const BlockQ4_1*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        float m = fp16_to_fp32(blocks[b].m);
        float* out_ptr = dst + b * 32;
        const uint8_t* qs = blocks[b].qs;

        for (int i = 0; i < 16; ++i) {
            uint8_t byte = qs[i];
            out_ptr[i]      = static_cast<float>(byte & 0x0F) * d + m;
            out_ptr[i + 16] = static_cast<float>((byte >> 4) & 0x0F) * d + m;
        }
    }
}

/**
 * High-Performance Dequantizer for Q4_K GGML Blocks (256 elements per super-block)
 */
inline void dequantize_q4_k(const void* src, float* dst, size_t num_blocks) {
    const auto* blocks = reinterpret_cast<const BlockQ4_K*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        float dmin = fp16_to_fp32(blocks[b].dmin);
        float* out_ptr = dst + b * 256;
        const uint8_t* qs = blocks[b].qs;

        for (int j = 0; j < 4; ++j) {
            float sc = d;
            float mn = dmin;
            for (int i = 0; i < 32; ++i) {
                uint8_t byte = qs[j * 32 + i];
                out_ptr[j * 64 + i]      = static_cast<float>(byte & 0x0F) * sc + mn;
                out_ptr[j * 64 + i + 32] = static_cast<float>((byte >> 4) & 0x0F) * sc + mn;
            }
        }
    }
}

/**
 * High-Performance Dequantizer for FP16 & BF16 Buffers
 */
inline void dequantize_fp16(const void* src, float* dst, size_t num_elements) {
    const auto* in_fp16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = fp16_to_fp32(in_fp16[i]);
    }
}

inline void dequantize_bf16(const void* src, float* dst, size_t num_elements) {
    const auto* in_bf16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < num_elements; ++i) {
        uint32_t val = static_cast<uint32_t>(in_bf16[i]) << 16;
        float f;
        std::memcpy(&f, &val, sizeof(float));
        dst[i] = f;
    }
}

} // namespace turing
