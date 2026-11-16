#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
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

// Dequantization matches ggml's scalar reference (and gguf-py) bit for bit, so no fused multiply-add:
// clang takes the pragma inside each function body, GCC an optimize attribute over the whole block.
#if defined(__clang__)
#define TURING_NO_FMA _Pragma("clang fp contract(off)")
#else
#define TURING_NO_FMA
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace turing {

constexpr int GGUF_QK_K = 256;

#pragma pack(push, 1)
// GGML quantization blocks (ggml-common.h)
struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };
struct BlockQ4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; };
struct BlockQ5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct BlockQ5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
struct BlockQ2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; };
struct BlockQ3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct BlockQ4_K { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ5_K { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct BlockQ6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct BlockQ8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
#pragma pack(pop)

static_assert(sizeof(BlockQ4_0) == 18 && sizeof(BlockQ4_1) == 20 && sizeof(BlockQ5_0) == 22 &&
              sizeof(BlockQ5_1) == 24 && sizeof(BlockQ8_0) == 34, "GGML 32-value block sizes");
static_assert(sizeof(BlockQ2_K) == 84 && sizeof(BlockQ3_K) == 110 && sizeof(BlockQ4_K) == 144 &&
              sizeof(BlockQ5_K) == 176 && sizeof(BlockQ6_K) == 210 && sizeof(BlockQ8_K) == 292,
              "GGML 256-value block sizes");

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

// Q4_0: low nibbles are values 0-15 of the block, high nibbles 16-31.
inline void dequantize_q4_0(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ4_0*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        float* y = dst + b * 32;
        for (int j = 0; j < 16; ++j) {
            y[j]      = static_cast<float>((blocks[b].qs[j] & 0x0F) - 8) * d;
            y[j + 16] = static_cast<float>((blocks[b].qs[j] >> 4) - 8) * d;
        }
    }
}

inline void dequantize_q4_1(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ4_1*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float m = fp16_to_fp32(blocks[b].m);
        float* y = dst + b * 32;
        for (int j = 0; j < 16; ++j) {
            y[j]      = static_cast<float>(blocks[b].qs[j] & 0x0F) * d + m;
            y[j + 16] = static_cast<float>(blocks[b].qs[j] >> 4) * d + m;
        }
    }
}

// Q5_0 / Q5_1: bit j of the 32-bit qh is the fifth bit of value j.
inline void dequantize_q5_0(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ5_0*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        float* y = dst + b * 32;
        for (int j = 0; j < 16; ++j) {
            const int x0 = ((blocks[b].qs[j] & 0x0F) | (((qh >> j) & 1) << 4)) - 16;
            const int x1 = ((blocks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
            y[j] = static_cast<float>(x0) * d;
            y[j + 16] = static_cast<float>(x1) * d;
        }
    }
}

inline void dequantize_q5_1(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ5_1*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float m = fp16_to_fp32(blocks[b].m);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        float* y = dst + b * 32;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (blocks[b].qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
            const int x1 = (blocks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            y[j] = static_cast<float>(x0) * d + m;
            y[j + 16] = static_cast<float>(x1) * d + m;
        }
    }
}

inline void dequantize_q8_0(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const int8_t* qs = blocks[b].qs;
        float* y = dst + b * 32;
#if defined(__AVX2__)
        const __m256 d_val = _mm256_set1_ps(d);
        for (int i = 0; i < 4; ++i) {
            __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i * 8));
            __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw8));
            _mm256_storeu_ps(y + i * 8, _mm256_mul_ps(q, d_val));
        }
#elif defined(__ARM_NEON) || defined(__aarch64__)
        const float32x4_t d_val = vdupq_n_f32(d);
        for (int i = 0; i < 32; i += 8) {
            int16x8_t q16 = vmovl_s8(vld1_s8(qs + i));
            vst1q_f32(y + i, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))), d_val));
            vst1q_f32(y + i + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16))), d_val));
        }
#else
        for (int i = 0; i < 32; ++i) y[i] = static_cast<float>(qs[i]) * d;
#endif
    }
}

inline void dequantize_q2_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ2_K*>(src);
    float* y = dst;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float mn = fp16_to_fp32(blocks[b].dmin);
        const uint8_t* q = blocks[b].qs;
        int is = 0;
        for (int n = 0; n < GGUF_QK_K; n += 128) {
            for (int shift = 0; shift < 8; shift += 2) {
                for (int half = 0; half < 2; ++half) {
                    const uint8_t sc = blocks[b].scales[is++];
                    const float dl = d * static_cast<float>(sc & 0xF);
                    const float ml = mn * static_cast<float>(sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((q[l + 16 * half] >> shift) & 3) - ml;
                }
            }
            q += 32;
        }
    }
}

inline void dequantize_q3_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ3_K*>(src);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    float* y = dst;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d_all = fp16_to_fp32(blocks[b].d);
        const uint8_t* q = blocks[b].qs;
        const uint8_t* hm = blocks[b].hmask;
        uint32_t aux[4];
        std::memcpy(aux, blocks[b].scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16];
        std::memcpy(scales, aux, 16);
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < GGUF_QK_K; n += 128) {
            for (int shift = 0; shift < 8; shift += 2) {
                for (int half = 0; half < 2; ++half) {
                    const float dl = d_all * static_cast<float>(scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        const int k = l + 16 * half;
                        const int v = ((q[k] >> shift) & 3) - ((hm[k] & m) ? 0 : 4);
                        *y++ = dl * static_cast<float>(v);
                    }
                }
                m <<= 1;
            }
            q += 32;
        }
    }
}

// 6-bit scale and min j of a Q4_K / Q5_K block
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline void dequantize_q4_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ4_K*>(src);
    float* y = dst;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float mn = fp16_to_fp32(blocks[b].dmin);
        const uint8_t* q = blocks[b].qs;
        for (int is = 0; is < 8; is += 2) {
            uint8_t sc, m;
            get_scale_min_k4(is, blocks[b].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc), m1 = mn * static_cast<float>(m);
            get_scale_min_k4(is + 1, blocks[b].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc), m2 = mn * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
            q += 32;
        }
    }
}

inline void dequantize_q5_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ5_K*>(src);
    float* y = dst;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float mn = fp16_to_fp32(blocks[b].dmin);
        const uint8_t* ql = blocks[b].qs;
        const uint8_t* qh = blocks[b].qh;
        uint8_t u1 = 1, u2 = 2;
        for (int is = 0; is < 8; is += 2) {
            uint8_t sc, m;
            get_scale_min_k4(is, blocks[b].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc), m1 = mn * static_cast<float>(m);
            get_scale_min_k4(is + 1, blocks[b].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc), m2 = mn * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

inline void dequantize_q6_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ6_K*>(src);
    float* y = dst;
    for (size_t b = 0; b < num_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t* sc = blocks[b].scales;
        for (int n = 0; n < GGUF_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

inline void dequantize_q8_k(const void* src, float* dst, size_t num_blocks) {
    TURING_NO_FMA
    const auto* blocks = reinterpret_cast<const BlockQ8_K*>(src);
    for (size_t b = 0; b < num_blocks; ++b) {
        float d;
        std::memcpy(&d, &blocks[b].d, sizeof(float));
        for (int i = 0; i < GGUF_QK_K; ++i) dst[b * GGUF_QK_K + i] = d * static_cast<float>(blocks[b].qs[i]);
    }
}

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

// (values per block, bytes per block) of a GGML type this file dequantizes; throws for any other type.
inline std::pair<size_t, size_t> gguf_block_layout(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return {1, 4};      // F32
        case 1: return {1, 2};      // F16
        case 30: return {1, 2};     // BF16
        case 2: return {32, 18};    // Q4_0
        case 3: return {32, 20};    // Q4_1
        case 6: return {32, 22};    // Q5_0
        case 7: return {32, 24};    // Q5_1
        case 8: return {32, 34};    // Q8_0
        case 10: return {256, 84};  // Q2_K
        case 11: return {256, 110}; // Q3_K
        case 12: return {256, 144}; // Q4_K
        case 13: return {256, 176}; // Q5_K
        case 14: return {256, 210}; // Q6_K
        case 15: return {256, 292}; // Q8_K
        default:
            throw std::runtime_error("GGML type " + std::to_string(ggml_type) + " is not supported (supported: F32, F16, "
                                     "BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)");
    }
}

// Bytes that num_elements values of ggml_type take; throws for unsupported types or a partial block.
inline size_t gguf_tensor_nbytes(uint32_t ggml_type, size_t num_elements) {
    const auto [per_block, block_bytes] = gguf_block_layout(ggml_type);
    if (num_elements % per_block != 0) {
        throw std::runtime_error("GGML type " + std::to_string(ggml_type) + ": " + std::to_string(num_elements) +
                                 " values are not a whole number of " + std::to_string(per_block) + "-value blocks");
    }
    return num_elements / per_block * block_bytes;
}

// Dequantizes num_elements values (gguf_tensor_nbytes(ggml_type, num_elements) bytes at src) into dst.
inline void dequantize_gguf(uint32_t ggml_type, const void* src, float* dst, size_t num_elements) {
    const size_t nb32 = num_elements / 32, nb256 = num_elements / GGUF_QK_K;
    gguf_tensor_nbytes(ggml_type, num_elements);  // validates the type and block count
    switch (ggml_type) {
        case 0: std::memcpy(dst, src, num_elements * sizeof(float)); break;
        case 1: dequantize_fp16(src, dst, num_elements); break;
        case 30: dequantize_bf16(src, dst, num_elements); break;
        case 2: dequantize_q4_0(src, dst, nb32); break;
        case 3: dequantize_q4_1(src, dst, nb32); break;
        case 6: dequantize_q5_0(src, dst, nb32); break;
        case 7: dequantize_q5_1(src, dst, nb32); break;
        case 8: dequantize_q8_0(src, dst, nb32); break;
        case 10: dequantize_q2_k(src, dst, nb256); break;
        case 11: dequantize_q3_k(src, dst, nb256); break;
        case 12: dequantize_q4_k(src, dst, nb256); break;
        case 13: dequantize_q5_k(src, dst, nb256); break;
        case 14: dequantize_q6_k(src, dst, nb256); break;
        case 15: dequantize_q8_k(src, dst, nb256); break;
    }
}

} // namespace turing

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
