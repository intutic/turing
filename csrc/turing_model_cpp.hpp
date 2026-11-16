#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>
#include <iostream>
#include <stdexcept>
#include "turing_simd.hpp"
#include "turing_sampler.hpp"
#include "turing_gguf_cpp.hpp"

namespace turing {

struct ModelConfigCpp {
    std::string name = "turing-model";
    uint32_t hidden_dim = 64;
    uint32_t ffn_dim = 128;
    uint32_t num_layers = 2;
    uint32_t num_heads = 4;
    uint32_t num_kv_heads = 4;
    uint32_t head_dim = 16;
    uint32_t vocab_size = 128;
    uint32_t max_pos = 512;          // KV cache length: GGUF {arch}.context_length, capped by the caller (--ctx)
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    int32_t eos_token_id = -1;       // GGUF tokenizer.ggml.eos_token_id; -1 = none (generation stops on length only)
    bool int8_kv = false;            // per-token asymmetric int8 K/V, fp32 scale and zero per KV head
};

struct TransformerLayerWeightsCpp {
    AlignedArray<float> attn_norm, ffn_norm;
    AlignedArray<float> w_q;     // [num_heads * head_dim, hidden_dim]
    AlignedArray<float> w_k;     // [num_kv_heads * head_dim, hidden_dim]
    AlignedArray<float> w_v;     // [num_kv_heads * head_dim, hidden_dim]
    AlignedArray<float> w_o;     // [hidden_dim, num_heads * head_dim]
    AlignedArray<float> w_gate;  // [ffn_dim, hidden_dim]
    AlignedArray<float> w_up;    // [ffn_dim, hidden_dim]
    AlignedArray<float> w_down;  // [hidden_dim, ffn_dim]
};

/**
 * Single-sequence Llama-style decoder (GQA, interleaved-pair RoPE on GGUF's q/k layout, SwiGLU).
 * Weights, KV cache and every per-token scratch buffer are allocated once, 64-byte aligned; the hot loops run on
 * the AVX2+FMA / NEON kernels of turing_simd.hpp.
 */
class TransformerModelCpp {
public:
    ModelConfigCpp config;
    AlignedArray<float> w_embed;  // [vocab_size, hidden_dim]
    AlignedArray<float> w_norm;   // [hidden_dim]
    AlignedArray<float> w_head;   // [vocab_size, hidden_dim]
    std::vector<TransformerLayerWeightsCpp> layers;

    explicit TransformerModelCpp(const ModelConfigCpp& cfg) : config(cfg), rng_(0) {
        layers.resize(config.num_layers);
        const size_t H = config.hidden_dim, D = config.head_dim, kv_dim = config.num_kv_heads * D;
        const size_t cache_elems = static_cast<size_t>(config.num_layers) * config.max_pos * kv_dim;
        if (config.int8_kv) {
            k_codes_ = make_aligned<uint8_t>(cache_elems);
            v_codes_ = make_aligned<uint8_t>(cache_elems);
            const size_t scale_elems = static_cast<size_t>(config.num_layers) * config.max_pos * config.num_kv_heads;
            k_scale_ = make_aligned<float>(scale_elems);
            k_zero_ = make_aligned<float>(scale_elems);
            v_scale_ = make_aligned<float>(scale_elems);
            v_zero_ = make_aligned<float>(scale_elems);
        } else {
            k_cache_ = make_aligned<float>(cache_elems);
            v_cache_ = make_aligned<float>(cache_elems);
        }
        x_ = make_aligned<float>(H);
        x_norm_ = make_aligned<float>(H);
        q_ = make_aligned<float>(config.num_heads * D);
        k_ = make_aligned<float>(kv_dim);
        v_ = make_aligned<float>(kv_dim);
        attn_out_ = make_aligned<float>(config.num_heads * D);
        proj_ = make_aligned<float>(H);
        gate_ = make_aligned<float>(config.ffn_dim);
        up_ = make_aligned<float>(config.ffn_dim);
        scores_ = make_aligned<float>(config.max_pos);
        rope_cos_ = make_aligned<float>(D / 2);
        rope_sin_ = make_aligned<float>(D / 2);
        inv_freq_ = make_aligned<float>(D / 2);
        for (size_t i = 0; i < D / 2; ++i) {
            inv_freq_[i] = 1.0f / std::pow(config.rope_theta, static_cast<float>(2 * i) / static_cast<float>(D));
        }
    }

    TransformerModelCpp(const TransformerModelCpp&) = delete;
    TransformerModelCpp& operator=(const TransformerModelCpp&) = delete;

    void seed(uint32_t s) { rng_.seed(s); }

    /** ctx_cap bounds the KV cache (and so the context) below the GGUF's trained context_length. */
    static std::unique_ptr<TransformerModelCpp> load_from_gguf(const GGUFReaderCpp& reader, size_t ctx_cap = 4096,
                                                               bool int8_kv = false) {
        auto meta_int = [&](const std::string& key, int64_t fallback) {
            auto it = reader.metadata_ints.find(key);
            return it != reader.metadata_ints.end() ? it->second : fallback;
        };
        auto meta_float = [&](const std::string& key, float fallback) {
            auto it = reader.metadata_floats.find(key);
            return it != reader.metadata_floats.end() ? it->second : fallback;
        };
        ModelConfigCpp cfg;
        std::string arch = "llama";
        if (reader.metadata_strings.count("general.architecture")) arch = reader.metadata_strings.at("general.architecture");
        if (reader.metadata_strings.count("general.name")) cfg.name = reader.metadata_strings.at("general.name");
        cfg.hidden_dim = static_cast<uint32_t>(meta_int(arch + ".embedding_length", cfg.hidden_dim));
        cfg.num_layers = static_cast<uint32_t>(meta_int(arch + ".block_count", cfg.num_layers));
        cfg.num_heads = static_cast<uint32_t>(meta_int(arch + ".attention.head_count", cfg.num_heads));
        cfg.num_kv_heads = static_cast<uint32_t>(meta_int(arch + ".attention.head_count_kv", cfg.num_heads));
        cfg.head_dim = static_cast<uint32_t>(meta_int(arch + ".attention.key_length", cfg.hidden_dim / cfg.num_heads));
        cfg.ffn_dim = static_cast<uint32_t>(meta_int(arch + ".feed_forward_length", int64_t(cfg.hidden_dim) * 4));
        cfg.rope_theta = meta_float(arch + ".rope.freq_base", cfg.rope_theta);
        cfg.rms_norm_eps = meta_float(arch + ".attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);
        cfg.eos_token_id = static_cast<int32_t>(meta_int("tokenizer.ggml.eos_token_id", -1));
        if (!reader.tokenizer_tokens.empty()) {
            cfg.vocab_size = static_cast<uint32_t>(reader.tokenizer_tokens.size());
        } else {
            cfg.vocab_size = static_cast<uint32_t>(meta_int(arch + ".vocab_size", cfg.vocab_size));
        }
        int64_t trained_ctx = meta_int(arch + ".context_length", 4096);
        cfg.max_pos = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(trained_ctx, static_cast<int64_t>(ctx_cap))));
        cfg.int8_kv = int8_kv;
        if (cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.num_heads % cfg.num_kv_heads != 0 || cfg.head_dim % 2 != 0) {
            throw std::runtime_error("unsupported attention shape in GGUF metadata");
        }

        auto model = std::make_unique<TransformerModelCpp>(cfg);
        // A tensor the file lacks is filled (1 for norms, 0 for projections) rather than left uninitialised.
        auto load = [&](const std::string& name, size_t n, float fill) {
            AlignedArray<float> t = make_aligned<float>(n);
            if (reader.has_tensor(name)) {
                reader.read_tensor_fp32(name, t.get(), n);
            } else if (fill != 0.0f) {
                std::fill(t.get(), t.get() + n, fill);
            }
            return t;
        };
        const size_t H = cfg.hidden_dim, q_dim = size_t(cfg.num_heads) * cfg.head_dim;
        const size_t kv_dim = size_t(cfg.num_kv_heads) * cfg.head_dim, embed = size_t(cfg.vocab_size) * H;
        model->w_embed = load("token_embd.weight", embed, 0.0f);
        model->w_norm = load(reader.has_tensor("output_norm.weight") ? "output_norm.weight" : "norm.weight", H, 1.0f);
        if (reader.has_tensor("output.weight")) {
            model->w_head = load("output.weight", embed, 0.0f);
        } else {  // tied embeddings
            model->w_head = make_aligned<float>(embed);
            std::memcpy(model->w_head.get(), model->w_embed.get(), embed * sizeof(float));
        }
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            auto& lay = model->layers[l];
            lay.attn_norm = load(p + "attn_norm.weight", H, 1.0f);
            lay.ffn_norm = load(p + "ffn_norm.weight", H, 1.0f);
            lay.w_q = load(p + "attn_q.weight", q_dim * H, 0.0f);
            lay.w_k = load(p + "attn_k.weight", kv_dim * H, 0.0f);
            lay.w_v = load(p + "attn_v.weight", kv_dim * H, 0.0f);
            lay.w_o = load(p + "attn_output.weight", H * q_dim, 0.0f);
            lay.w_gate = load(p + "ffn_gate.weight", size_t(cfg.ffn_dim) * H, 0.0f);
            lay.w_up = load(p + "ffn_up.weight", size_t(cfg.ffn_dim) * H, 0.0f);
            lay.w_down = load(p + "ffn_down.weight", H * cfg.ffn_dim, 0.0f);
        }
        return model;
    }

    // Evaluates 1 token at pos, writes its K/V into the cache, and produces logits [vocab_size]
    void forward_token(int32_t token_id, size_t pos, float* logits_out) {
        // Attention reads cache rows 0..pos; past max_pos those rows do not exist.
        if (pos >= config.max_pos) {
            throw std::out_of_range("position " + std::to_string(pos) + " exceeds the model's context of " +
                                    std::to_string(config.max_pos) + " tokens");
        }
        const size_t H = config.hidden_dim, V = config.vocab_size, D = config.head_dim;
        const size_t n_heads = config.num_heads, n_kv = config.num_kv_heads, ffn = config.ffn_dim;
        const size_t kv_dim = n_kv * D, group = n_heads / n_kv;
        float* x = x_.get();
        float* xn = x_norm_.get();

        if (token_id >= 0 && static_cast<size_t>(token_id) < V) {
            std::memcpy(x, w_embed.get() + static_cast<size_t>(token_id) * H, H * sizeof(float));
        } else {
            std::fill(x, x + H, 0.0f);
        }
        for (size_t i = 0; i < D / 2; ++i) {
            float angle = static_cast<float>(pos) * inv_freq_[i];
            rope_cos_[i] = std::cos(angle);
            rope_sin_[i] = std::sin(angle);
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        for (uint32_t l = 0; l < config.num_layers; ++l) {
            const auto& lay = layers[l];
            rmsnorm_f32(x, lay.attn_norm.get(), xn, H, config.rms_norm_eps);
            gemv_f32(lay.w_q.get(), xn, q_.get(), n_heads * D, H);
            gemv_f32(lay.w_k.get(), xn, k_.get(), kv_dim, H);
            gemv_f32(lay.w_v.get(), xn, v_.get(), kv_dim, H);
            apply_rope(q_.get(), n_heads);
            apply_rope(k_.get(), n_kv);

            const size_t row = (static_cast<size_t>(l) * config.max_pos + pos);  // this token's cache row
            if (config.int8_kv) {
                for (size_t h = 0; h < n_kv; ++h) {
                    quantize_u8(k_.get() + h * D, D, k_codes_.get() + row * kv_dim + h * D, k_scale_[row * n_kv + h], k_zero_[row * n_kv + h]);
                    quantize_u8(v_.get() + h * D, D, v_codes_.get() + row * kv_dim + h * D, v_scale_[row * n_kv + h], v_zero_[row * n_kv + h]);
                }
            } else {
                std::memcpy(k_cache_.get() + row * kv_dim, k_.get(), kv_dim * sizeof(float));
                std::memcpy(v_cache_.get() + row * kv_dim, v_.get(), kv_dim * sizeof(float));
            }

            const size_t base = static_cast<size_t>(l) * config.max_pos;  // first cache row of this layer
            float* scores = scores_.get();
            std::fill(attn_out_.get(), attn_out_.get() + n_heads * D, 0.0f);
            for (size_t h = 0; h < n_heads; ++h) {
                const size_t kh = h / group;
                const float* qh = q_.get() + h * D;
                float* out_h = attn_out_.get() + h * D;
                if (config.int8_kv) {
                    // q . (codes * s + z) = s * (q . codes) + z * sum(q)
                    float q_sum = 0.0f;
                    for (size_t d = 0; d < D; ++d) q_sum += qh[d];
                    for (size_t t = 0; t <= pos; ++t) {
                        const size_t r = base + t;
                        scores[t] = (k_scale_[r * n_kv + kh] * dot_f32_u8(qh, k_codes_.get() + r * kv_dim + kh * D, D)
                                     + k_zero_[r * n_kv + kh] * q_sum) * scale;
                    }
                    softmax_f32(scores, pos + 1);
                    for (size_t t = 0; t <= pos; ++t) {
                        const size_t r = base + t;
                        axpby_u8_f32(scores[t] * v_scale_[r * n_kv + kh], scores[t] * v_zero_[r * n_kv + kh],
                                     v_codes_.get() + r * kv_dim + kh * D, out_h, D);
                    }
                } else {
                    for (size_t t = 0; t <= pos; ++t) {
                        scores[t] = dot_f32(qh, k_cache_.get() + (base + t) * kv_dim + kh * D, D) * scale;
                    }
                    softmax_f32(scores, pos + 1);
                    for (size_t t = 0; t <= pos; ++t) {
                        axpy_f32(scores[t], v_cache_.get() + (base + t) * kv_dim + kh * D, out_h, D);
                    }
                }
            }

            gemv_f32(lay.w_o.get(), attn_out_.get(), proj_.get(), H, n_heads * D);
            axpy_f32(1.0f, proj_.get(), x, H);

            rmsnorm_f32(x, lay.ffn_norm.get(), xn, H, config.rms_norm_eps);
            gemv_f32(lay.w_gate.get(), xn, gate_.get(), ffn, H);
            gemv_f32(lay.w_up.get(), xn, up_.get(), ffn, H);
            silu_mul_f32(gate_.get(), up_.get(), ffn);
            gemv_f32(lay.w_down.get(), gate_.get(), proj_.get(), H, ffn);
            axpy_f32(1.0f, proj_.get(), x, H);
        }

        rmsnorm_f32(x, w_norm.get(), xn, H, config.rms_norm_eps);
        gemv_f32(w_head.get(), xn, logits_out, V, H);
    }

    /** Prompt plus up to max_new_tokens sampled tokens; stops early on the GGUF's EOS token or the context end. */
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        size_t max_new_tokens = 32,
        float temperature = 0.7f
    ) {
        std::vector<int32_t> result = prompt_tokens;
        if (prompt_tokens.empty() || max_new_tokens == 0) return result;

        std::vector<float> logits(config.vocab_size);
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            forward_token(prompt_tokens[i], i, logits.data());
        }

        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        size_t cur_pos = prompt_tokens.size();
        for (size_t step = 0; step < max_new_tokens; ++step) {
            int32_t next_tok = sample_single_logits_simd_cpp(logits.data(), config.vocab_size, temperature, 0, uniform(rng_));
            result.push_back(next_tok);
            if (next_tok == config.eos_token_id || step + 1 == max_new_tokens || cur_pos >= config.max_pos) break;
            forward_token(next_tok, cur_pos++, logits.data());
        }
        return result;
    }

private:
    AlignedArray<float> k_cache_, v_cache_;                    // fp32: [layer][pos][kv_head * head_dim]
    AlignedArray<uint8_t> k_codes_, v_codes_;                  // int8: same layout, uint8 codes
    AlignedArray<float> k_scale_, k_zero_, v_scale_, v_zero_;  // int8: [layer][pos][kv_head]
    AlignedArray<float> x_, x_norm_, q_, k_, v_, attn_out_, proj_, gate_, up_, scores_;
    AlignedArray<float> rope_cos_, rope_sin_, inv_freq_;
    std::mt19937 rng_;

    // Interleaved pairs (2i, 2i + 1): GGUF stores llama q/k rows permuted for exactly this rotation.
    void apply_rope(float* vec, size_t n_heads) const {
        const size_t D = config.head_dim;
        for (size_t h = 0; h < n_heads; ++h) {
            float* p = vec + h * D;
            for (size_t i = 0; i < D / 2; ++i) {
                float v0 = p[2 * i], v1 = p[2 * i + 1];
                p[2 * i]     = v0 * rope_cos_[i] - v1 * rope_sin_[i];
                p[2 * i + 1] = v0 * rope_sin_[i] + v1 * rope_cos_[i];
            }
        }
    }

    // Asymmetric round-to-nearest-even over one head's vector, as turing/core/kv_cache.py's int8 mode does.
    static void quantize_u8(const float* x, size_t n, uint8_t* codes, float& scale, float& zero) {
        float lo = x[0], hi = x[0];
        for (size_t i = 1; i < n; ++i) { lo = std::min(lo, x[i]); hi = std::max(hi, x[i]); }
        scale = std::max((hi - lo) / 255.0f, 1e-8f);
        zero = lo;
        for (size_t i = 0; i < n; ++i) {
            float c = std::nearbyint((x[i] - lo) / scale);
            codes[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c)));
        }
    }
};

} // namespace turing
