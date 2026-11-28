// Wave 5 go / no-go (bmad-output/cpu-first/plan.md): is a column-sparse 4-bit matrix-vector product, which skips the
// weight columns of inputs TEAL zeroes, faster than ggml's dense 4-bit product on this CPU at the sparsities TEAL keeps
// quality with (25-50%)? Decode is bandwidth bound, so it pays only if skipped columns are skipped bytes: the sparse
// layout stores each input column's weights contiguously (ggml Q4_0 blocks along the output dimension, 4.5 bits per
// weight like Q4_K) and reads only the active ones. Weights are --layers copies so they come from DRAM, not the caches.
//   turing-teal-bench [--threads 16] [--layers 16] [--K 4096] [--N 14336] [--reps 20]
// One JSON line per row: ms per layer and GB/s of weights read; the sparse rows also their speedup over dense Q4_K.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using clk = std::chrono::steady_clock;
struct BlockQ4 {  // ggml's block_q4_0: 32 weights, w = d * (q - 8); qs[i] low nibble = weight i, high = weight i + 16
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ4) == 18, "block_q4_0 layout");

// y[b*32 .. b*32+31] += s * dequant(block) for the blocks [b0, b1) of one column
static inline void axpy_blocks(float* y, const BlockQ4* blk, int b0, int b1, float x) {
    for (int b = b0; b < b1; ++b) {
        const float s = x * ggml_fp16_to_fp32(blk[b].d);
        float* o = y + static_cast<size_t>(b) * 32;
#if defined(__ARM_NEON)
        const uint8x16_t q = vld1q_u8(blk[b].qs);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));
        const float32x4_t vs = vdupq_n_f32(s);
        const int8x16_t halves[2] = {lo, hi};
        for (int h = 0; h < 2; ++h) {
            const int16x8_t w0 = vmovl_s8(vget_low_s8(halves[h])), w1 = vmovl_s8(vget_high_s8(halves[h]));
            const int16x8_t w[2] = {w0, w1};
            for (int k = 0; k < 2; ++k) {
                float* p = o + h * 16 + k * 8;
                vst1q_f32(p, vfmaq_f32(vld1q_f32(p), vs, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w[k])))));
                vst1q_f32(p + 4, vfmaq_f32(vld1q_f32(p + 4), vs, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w[k])))));
            }
        }
#else
        for (int i = 0; i < 16; ++i) {
            o[i] += s * static_cast<float>((blk[b].qs[i] & 0x0F) - 8);
            o[i + 16] += s * static_cast<float>((blk[b].qs[i] >> 4) - 8);
        }
#endif
    }
}

int main(int argc, char** argv) {
    int threads = 16, layers = 16, K = 4096, N = 14336, reps = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const int v = std::stoi(argv[i + 1]);
        if (a == "--threads") threads = v;
        else if (a == "--layers") layers = v;
        else if (a == "--K") K = v;
        else if (a == "--N") N = v;
        else if (a == "--reps") reps = v;
    }
    if (N % 32 || K % 256) {
        std::fprintf(stderr, "N must be a multiple of 32 and K of 256\n");
        return 2;
    }
    std::mt19937 rng(0);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> W(static_cast<size_t>(N) * K), Wt(static_cast<size_t>(K) * N), x(K);
    for (auto& w : W) w = g(rng) / std::sqrt(static_cast<float>(K));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) Wt[static_cast<size_t>(k) * N + n] = W[static_cast<size_t>(n) * K + k];
    for (auto& v : x) v = g(rng);

    // dense: ggml mul_mat over `layers` copies of W [K in, N out], Q4_0 and Q4_K, on ggml's CPU backend
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, threads);
    auto dense = [&](ggml_type type) {
        ggml_init_params ip{ggml_tensor_overhead() * (2 * layers + 8) + ggml_graph_overhead(), nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        std::vector<ggml_tensor*> ws(layers), ys(layers);
        ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        for (int l = 0; l < layers; ++l) ws[l] = ggml_new_tensor_2d(ctx, type, K, N);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        for (int l = 0; l < layers; ++l) ggml_build_forward_expand(gf, ys[l] = ggml_mul_mat(ctx, ws[l], xt));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<uint8_t> q(ggml_row_size(type, K) * N);
        ggml_quantize_chunk(type, W.data(), q.data(), 0, N, K, nullptr);
        for (int l = 0; l < layers; ++l) ggml_backend_tensor_set(ws[l], q.data(), 0, q.size());
        ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_graph_compute(be, gf);  // warm-up
        const auto t0 = clk::now();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, gf);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps / layers;
        std::printf("{\"kernel\": \"ggml dense %s\", \"K\": %d, \"N\": %d, \"threads\": %d, \"ms_per_layer\": %.4f, "
                    "\"GB_s\": %.1f}\n", ggml_type_name(type), K, N, threads, ms, q.size() / ms / 1e6);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ms;
    };
    dense(GGML_TYPE_Q4_0);
    const double dense_q4k = dense(GGML_TYPE_Q4_K);

    // sparse: W transposed, each input column a row of N/32 Q4_0 blocks; threads own slices of the output blocks
    const int nb = N / 32;
    const size_t col_bytes = ggml_row_size(GGML_TYPE_Q4_0, N);
    std::vector<uint8_t> colq(col_bytes * K);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, Wt.data(), colq.data(), 0, K, N, nullptr);
    std::vector<std::vector<uint8_t>> copies(layers, colq);
    std::vector<float> mag(K);
    for (int k = 0; k < K; ++k) mag[k] = std::fabs(x[k]);
    std::vector<float> sorted = mag;
    std::sort(sorted.begin(), sorted.end());
    std::vector<float> y(static_cast<size_t>(N));
    for (double sparsity : {0.0, 0.25, 0.40, 0.50}) {
        const float thr = sparsity > 0 ? sorted[static_cast<size_t>(sparsity * K)] : -1.0f;
        std::vector<int> active;
        for (int k = 0; k < K; ++k)
            if (mag[k] >= thr) active.push_back(k);
        auto run_layer = [&](const std::vector<uint8_t>& w, int b0, int b1) {
            std::fill(y.begin() + static_cast<long>(b0) * 32, y.begin() + static_cast<long>(b1) * 32, 0.0f);
            for (int k : active)
                axpy_blocks(y.data(), reinterpret_cast<const BlockQ4*>(w.data() + col_bytes * k), b0, b1, x[k]);
        };
        // self-check against a scalar reference of the same dequantization (one layer, all threads' slices)
        run_layer(copies[0], 0, nb);
        double err = 0;
        for (int n = 0; n < N; n += 97) {
            double ref = 0;
            for (int k : active) {
                const BlockQ4& blk = reinterpret_cast<const BlockQ4*>(copies[0].data() + col_bytes * k)[n / 32];
                const int i = n % 32;
                const int q = i < 16 ? (blk.qs[i] & 0x0F) : (blk.qs[i - 16] >> 4);
                ref += x[k] * ggml_fp16_to_fp32(blk.d) * (q - 8);
            }
            err = std::max(err, std::fabs(ref - y[n]) / (1e-3 + std::fabs(ref)));
        }
        std::barrier sync(threads);
        auto worker = [&](int t) {
            const int b0 = static_cast<int>(static_cast<long>(nb) * t / threads);
            const int b1 = static_cast<int>(static_cast<long>(nb) * (t + 1) / threads);
            for (int r = 0; r <= reps; ++r)
                for (int l = 0; l < layers; ++l) {
                    run_layer(copies[l], b0, b1);
                    sync.arrive_and_wait();  // one barrier per layer, as ggml has per mul_mat node
                }
        };
        std::vector<std::thread> pool;
        const auto t0 = clk::now();
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / (reps + 1) / layers;
        const double bytes = static_cast<double>(col_bytes) * active.size();
        std::printf("{\"kernel\": \"column-sparse Q4_0\", \"sparsity\": %.2f, \"K\": %d, \"N\": %d, \"threads\": %d, "
                    "\"ms_per_layer\": %.4f, \"GB_s\": %.1f, \"speedup_vs_dense_q4_k\": %.2f, \"max_rel_err\": %.2e}\n",
                    sparsity, K, N, threads, ms, bytes / ms / 1e6, dense_q4k / ms, err);
    }
    ggml_backend_free(be);
    return 0;
}
