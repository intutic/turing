#pragma once
// One packed 4-bit MoE expert on the CPU, for a few input rows:
//   y = down @ (silu(gate @ x) * (up @ x))        gate, up = the two halves of gate_up
// A token routed to an expert that is not on the GPU (SparseMoE offload) is computed here from host memory instead of
// copying the expert in (92 MB for Mixtral-8x7B, several times per token).
//
// Layout as turing/models/quant.py: codes [N, K / 2] bytes (low nibble = even column), scales fp16 and zeros uint8
// [N, K / G], w[n, k] = (code - zero[n, k / G]) * scale[n, k / G]. So
//   y[n] = sum_g scale[n, g] * (sum_{k in g} code[n, k] * x[k] - zero[n, g] * sum_{k in g} x[k])
// and the weights are never widened in memory: each row of codes is read once for all input rows.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "turing_simd.hpp"

namespace turing {

inline float w4_half_to_float(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            int e = -1;
            uint32_t m = mant;
            do { ++e; m <<= 1; } while (!(m & 0x400u));
            bits = sign | ((112 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TURING_W4_AVX512 1
// One output row for one input row with 512-bit registers, where the CPU has them (checked when the product starts;
// the build itself stays AVX2). The 256-bit loop is bound by the two multiply-add units, not by memory: 3.2 ms an
// expert of Mixtral-8x7B on 16 threads of a Cascade Lake VM whose memory copies at 40 GB/s (29 GB/s read).
__attribute__((target("avx512f,fma,f16c")))
inline float w4_row_avx512(const uint8_t* row, const uint16_t* sc, const uint8_t* zr, size_t groups, size_t gh,
                           const float* xe, const float* xo, const float* xs) {
    const __m512i mask = _mm512_set1_epi32(0x0F);
    __m512 racc = _mm512_setzero_ps();
    float zsum = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        const uint8_t* b = row + g * gh;
        const float* e = xe + g * gh;
        const float* o = xo + g * gh;
        const float scale = _cvtsh_ss(sc[g]);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (size_t i = 0; i < gh; i += 16) {
            const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
            a0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(v, mask)), _mm512_loadu_ps(e + i), a0);
            a1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(v, 4)), _mm512_loadu_ps(o + i), a1);
        }
        racc = _mm512_fmadd_ps(_mm512_add_ps(a0, a1), _mm512_set1_ps(scale), racc);
        zsum += scale * static_cast<float>(zr[g]) * xs[g];
    }
    return _mm512_reduce_add_ps(racc) - zsum;
}

inline bool w4_has_avx512() {
    static const bool has = __builtin_cpu_supports("avx512f");
    return has;
}
#endif

// xe, xo [R, K / 2]: the even and odd columns of each input row; xs [R, K / G]: each group's sum.
// y [R, N] (row-major) = x @ W^T for output rows [n0, n1). R <= W4_MAX_ROWS.
constexpr size_t W4_MAX_ROWS = 8;

inline void w4_gemm_range(const uint8_t* q, const uint16_t* scales, const uint8_t* zeros, size_t K, size_t G,
                          const float* xe, const float* xo, const float* xs, size_t R, size_t N, float* y,
                          size_t n0, size_t n1) {
    const size_t half = K / 2, groups = K / G, gh = G / 2;
#if defined(TURING_W4_AVX512)
    if (R == 1 && gh % 16 == 0 && w4_has_avx512()) {
        for (size_t n = n0; n < n1; ++n)
            y[n] = w4_row_avx512(q + n * half, scales + n * groups, zeros + n * groups, groups, gh, xe, xo, xs);
        return;
    }
#endif
    for (size_t n = n0; n < n1; ++n) {
        const uint8_t* row = q + n * half;
        const uint16_t* sc = scales + n * groups;
        const uint8_t* zr = zeros + n * groups;
        float zsum[W4_MAX_ROWS] = {0};  // sum_g scale * zero * (the group's sum of x)
#if defined(TURING_HAS_AVX2)
        // Each group's vector of partial sums is scaled into one accumulator per input row: one horizontal sum per
        // output row. (FMA where the build has it; the two forms round differently, like every kernel here.)
#if defined(TURING_HAS_AVX2_FMA)
#define W4_MADD(a, b, c) _mm256_fmadd_ps(a, b, c)
#else
#define W4_MADD(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif
        const __m256i mask = _mm256_set1_epi32(0x0F);
        __m256 racc[W4_MAX_ROWS];
        for (size_t r = 0; r < R; ++r) racc[r] = _mm256_setzero_ps();
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t* b = row + g * gh;
            const float scale = w4_half_to_float(sc[g]), sz = scale * static_cast<float>(zr[g]);
            const __m256 vs = _mm256_set1_ps(scale);
            if (R == 1) {  // a decode step of one request: from the registers, nothing stored
                const float* e = xe + g * gh;
                const float* o = xo + g * gh;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                for (size_t i = 0; i < gh; i += 8) {
                    const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i)));
                    a0 = W4_MADD(_mm256_cvtepi32_ps(_mm256_and_si256(v, mask)), _mm256_loadu_ps(e + i), a0);
                    a1 = W4_MADD(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4)), _mm256_loadu_ps(o + i), a1);
                }
                racc[0] = W4_MADD(_mm256_add_ps(a0, a1), vs, racc[0]);
                zsum[0] += sz * xs[g];
                continue;
            }
            // G / 2 bytes -> low and high nibbles as floats, 8 at a time; reused by every input row
            alignas(32) float lo[64], hi[64];  // G <= 128
            for (size_t i = 0; i < gh; i += 8) {
                const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i)));
                _mm256_store_ps(lo + i, _mm256_cvtepi32_ps(_mm256_and_si256(v, mask)));
                _mm256_store_ps(hi + i, _mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4)));
            }
            for (size_t r = 0; r < R; ++r) {
                const float* e = xe + r * half + g * gh;
                const float* o = xo + r * half + g * gh;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                for (size_t i = 0; i < gh; i += 8) {
                    a0 = W4_MADD(_mm256_load_ps(lo + i), _mm256_loadu_ps(e + i), a0);
                    a1 = W4_MADD(_mm256_load_ps(hi + i), _mm256_loadu_ps(o + i), a1);
                }
                racc[r] = W4_MADD(_mm256_add_ps(a0, a1), vs, racc[r]);
                zsum[r] += sz * xs[r * groups + g];
            }
        }
        for (size_t r = 0; r < R; ++r) {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(racc[r]), _mm256_extractf128_ps(racc[r], 1));
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            y[r * N + n] = _mm_cvtss_f32(s) - zsum[r];
        }
#undef W4_MADD
#else
        float acc[W4_MAX_ROWS] = {0};
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t* b = row + g * gh;
            const float scale = w4_half_to_float(sc[g]), sz = scale * static_cast<float>(zr[g]);
            for (size_t r = 0; r < R; ++r) {
                const float* e = xe + r * half + g * gh;
                const float* o = xo + r * half + g * gh;
                float a = 0.0f;
                for (size_t i = 0; i < gh; ++i)
                    a += static_cast<float>(b[i] & 0x0F) * e[i] + static_cast<float>(b[i] >> 4) * o[i];
                acc[r] += scale * a;
                zsum[r] += sz * xs[r * groups + g];
            }
        }
        for (size_t r = 0; r < R; ++r) y[r * N + n] = acc[r] - zsum[r];
#endif
    }
}

// Threads that live for the process. A decode step runs dozens of these products, and starting sixteen threads for
// each one cost about as much as the product. Workers spin for a moment after a job (the next product of the same
// expert, or the next expert, follows within microseconds), then sleep.
class W4Pool {
  public:
    explicit W4Pool(size_t threads) {
        for (size_t i = 1; i < threads; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~W4Pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
            epoch_.fetch_add(1);
        }
        wake_.notify_all();
        for (auto& t : workers_) t.join();
    }
    size_t size() const { return workers_.size() + 1; }

    // fn(i) for every i in [0, n), on the workers and the calling thread; returns when all are done. One caller at
    // a time (w4_expert_rows holds a lock).
    void run(size_t n, const std::function<void(size_t)>& fn) {
        {
            std::lock_guard<std::mutex> lock(m_);
            fn_ = &fn;
            n_ = n;
            next_.store(0);
            pending_ = workers_.size();
            epoch_.fetch_add(1);
        }
        wake_.notify_all();
        for (size_t i; (i = next_.fetch_add(1)) < n;) fn(i);
        std::unique_lock<std::mutex> lock(m_);  // every worker checks in, so none reads fn after this returns
        done_.wait(lock, [this] { return pending_ == 0; });
    }

  private:
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            for (int spin = 0; spin < 4000 && epoch_.load(std::memory_order_relaxed) == seen; ++spin) {
#if defined(TURING_HAS_AVX2)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [&] { return epoch_.load() != seen; });
                seen = epoch_.load();
                if (stop_) return;
            }
            for (size_t i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i);
            std::lock_guard<std::mutex> lock(m_);
            if (--pending_ == 0) done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable wake_, done_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<size_t> next_{0};
    const std::function<void(size_t)>* fn_ = nullptr;
    size_t n_ = 0, pending_ = 0;
    bool stop_ = false;
};

inline long w4_pid() {
#if defined(_WIN32)
    return 1;
#else
    return static_cast<long>(getpid());
#endif
}

// The pool for `threads` threads, made on first use and again in a forked child (the parent's threads are not there).
// It is never destroyed at exit: a child of fork that inherited it would join threads it does not have.
inline W4Pool& w4_pool(size_t threads) {
    static W4Pool* pool = nullptr;
    static long owner = 0;
    const long pid = w4_pid();
    if (pool == nullptr || owner != pid || pool->size() != threads) {
        if (pool != nullptr && owner == pid) delete pool;  // another thread count; a parent's pool is left alone
        pool = new W4Pool(threads);
        owner = pid;
    }
    return *pool;
}

// One product at a time. A child of fork gets a mutex of its own: the parent's may have been copied while held.
inline std::mutex& w4_one_caller() {
    static std::mutex* m = new std::mutex;
    static long owner = w4_pid();
    if (owner != w4_pid()) {
        m = new std::mutex;
        owner = w4_pid();
    }
    return *m;
}

// y [R, N] = x [R, K] @ W^T for a packed matrix, on `threads` threads (the caller keeps R small).
inline void w4_gemm_rows(const uint8_t* q, const uint16_t* scales, const uint8_t* zeros, size_t N, size_t K, size_t G,
                         const float* x, size_t R, float* y, int threads) {
    const size_t half = K / 2, groups = K / G;
    std::vector<float> xe(R * half), xo(R * half), xs(R * groups, 0.0f);
    for (size_t r = 0; r < R; ++r)
        for (size_t k = 0; k < K; ++k) {
            (k & 1 ? xo : xe)[r * half + k / 2] = x[r * K + k];
            xs[r * groups + k / G] += x[r * K + k];
        }
    // four pieces a thread, taken as threads come free: two threads of one core do not finish together
    const size_t T = static_cast<size_t>(std::max(threads, 1)), pieces = std::min(N / 64 + 1, 4 * T);
    if (T == 1 || pieces == 1) {
        w4_gemm_range(q, scales, zeros, K, G, xe.data(), xo.data(), xs.data(), R, N, y, 0, N);
        return;
    }
    const std::function<void(size_t)> piece = [&](size_t p) {
        w4_gemm_range(q, scales, zeros, K, G, xe.data(), xo.data(), xs.data(), R, N, y, N * p / pieces,
                      N * (p + 1) / pieces);
    };
    w4_pool(T).run(pieces, piece);
}

// out [R, H] = down @ (silu(gate @ x) * (up @ x)); gate_up packed [2F, H / 2], down packed [H, F / 2].
inline void w4_expert_rows(const uint8_t* gu_q, const uint16_t* gu_s, const uint8_t* gu_z,
                           const uint8_t* dn_q, const uint16_t* dn_s, const uint8_t* dn_z,
                           size_t H, size_t F, size_t G, const float* x, size_t R, float* out, int threads) {
    std::lock_guard<std::mutex> lock(w4_one_caller());  // the pool serves one product at a time
    std::vector<float> gu(R * 2 * F), h(R * F);
    w4_gemm_rows(gu_q, gu_s, gu_z, 2 * F, H, G, x, R, gu.data(), threads);
    for (size_t r = 0; r < R; ++r) {
        const float* gate = gu.data() + r * 2 * F;
        const float* up = gate + F;
        for (size_t f = 0; f < F; ++f) h[r * F + f] = gate[f] / (1.0f + std::exp(-gate[f])) * up[f];
    }
    w4_gemm_rows(dn_q, dn_s, dn_z, H, F, G, h.data(), R, out, threads);
}

}  // namespace turing
