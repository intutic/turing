#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>
#include "turing_simd.hpp"
#include "turing_rope.hpp"
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
    uint32_t max_pos = 512;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
};

struct TransformerLayerWeightsCpp {
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
    float* w_q = nullptr;     // [num_heads * head_dim, hidden_dim]
    float* w_k = nullptr;     // [num_kv_heads * head_dim, hidden_dim]
    float* w_v = nullptr;     // [num_kv_heads * head_dim, hidden_dim]
    float* w_o = nullptr;     // [hidden_dim, num_heads * head_dim]
    float* w_gate = nullptr;  // [ffn_dim, hidden_dim]
    float* w_up = nullptr;    // [ffn_dim, hidden_dim]
    float* w_down = nullptr;  // [hidden_dim, ffn_dim]
};

class TransformerModelCpp {
public:
    ModelConfigCpp config;
    float* w_embed = nullptr;  // [vocab_size, hidden_dim]
    float* w_norm = nullptr;   // [hidden_dim]
    float* w_head = nullptr;   // [vocab_size, hidden_dim]
    std::vector<TransformerLayerWeightsCpp> layers;

    // KV cache per layer: [max_pos, num_kv_heads * head_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    TransformerModelCpp(const ModelConfigCpp& cfg) : config(cfg) {
        layers.resize(config.num_layers);
        size_t kv_dim = config.num_kv_heads * config.head_dim;
        k_cache.resize(config.num_layers, std::vector<float>(config.max_pos * kv_dim, 0.0f));
        v_cache.resize(config.num_layers, std::vector<float>(config.max_pos * kv_dim, 0.0f));
    }

    ~TransformerModelCpp() {
        if (w_embed) aligned_free_64(w_embed);
        if (w_norm) aligned_free_64(w_norm);
        if (w_head) aligned_free_64(w_head);
        for (auto& l : layers) {
            if (l.attn_norm) aligned_free_64(l.attn_norm);
            if (l.ffn_norm) aligned_free_64(l.ffn_norm);
            if (l.w_q) aligned_free_64(l.w_q);
            if (l.w_k) aligned_free_64(l.w_k);
            if (l.w_v) aligned_free_64(l.w_v);
            if (l.w_o) aligned_free_64(l.w_o);
            if (l.w_gate) aligned_free_64(l.w_gate);
            if (l.w_up) aligned_free_64(l.w_up);
            if (l.w_down) aligned_free_64(l.w_down);
        }
    }

    static std::unique_ptr<TransformerModelCpp> load_from_gguf(const GGUFReaderCpp& reader) {
        ModelConfigCpp cfg;
        std::string arch = "llama";
        if (reader.metadata_strings.find("general.architecture") != reader.metadata_strings.end()) {
            arch = reader.metadata_strings.at("general.architecture");
        }
        if (reader.metadata_strings.find("general.name") != reader.metadata_strings.end()) {
            cfg.name = reader.metadata_strings.at("general.name");
        }
        if (reader.metadata_ints.find(arch + ".embedding_length") != reader.metadata_ints.end()) {
            cfg.hidden_dim = static_cast<uint32_t>(reader.metadata_ints.at(arch + ".embedding_length"));
        }
        if (reader.metadata_ints.find(arch + ".block_count") != reader.metadata_ints.end()) {
            cfg.num_layers = static_cast<uint32_t>(reader.metadata_ints.at(arch + ".block_count"));
        }
        if (reader.metadata_ints.find(arch + ".attention.head_count") != reader.metadata_ints.end()) {
            cfg.num_heads = static_cast<uint32_t>(reader.metadata_ints.at(arch + ".attention.head_count"));
        }
        if (reader.metadata_ints.find(arch + ".attention.head_count_kv") != reader.metadata_ints.end()) {
            cfg.num_kv_heads = static_cast<uint32_t>(reader.metadata_ints.at(arch + ".attention.head_count_kv"));
        } else {
            cfg.num_kv_heads = cfg.num_heads;
        }
        cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
        if (reader.metadata_ints.find(arch + ".feed_forward_length") != reader.metadata_ints.end()) {
            cfg.ffn_dim = static_cast<uint32_t>(reader.metadata_ints.at(arch + ".feed_forward_length"));
        } else {
            cfg.ffn_dim = cfg.hidden_dim * 4;
        }
        if (!reader.tokenizer_tokens.empty()) {
            cfg.vocab_size = static_cast<uint32_t>(reader.tokenizer_tokens.size());
        }

        auto model = std::make_unique<TransformerModelCpp>(cfg);

        // Load embeddings
        size_t embed_elems = cfg.vocab_size * cfg.hidden_dim;
        model->w_embed = static_cast<float*>(aligned_alloc_64(embed_elems * sizeof(float)));
        if (reader.has_tensor("token_embd.weight")) {
            reader.read_tensor_fp32("token_embd.weight", model->w_embed, embed_elems);
        } else {
            std::memset(model->w_embed, 0, embed_elems * sizeof(float));
        }

        // Load norm
        model->w_norm = static_cast<float*>(aligned_alloc_64(cfg.hidden_dim * sizeof(float)));
        if (reader.has_tensor("output_norm.weight")) {
            reader.read_tensor_fp32("output_norm.weight", model->w_norm, cfg.hidden_dim);
        } else if (reader.has_tensor("norm.weight")) {
            reader.read_tensor_fp32("norm.weight", model->w_norm, cfg.hidden_dim);
        } else {
            for (size_t i = 0; i < cfg.hidden_dim; ++i) model->w_norm[i] = 1.0f;
        }

        // Load output / lm_head
        model->w_head = static_cast<float*>(aligned_alloc_64(embed_elems * sizeof(float)));
        if (reader.has_tensor("output.weight")) {
            reader.read_tensor_fp32("output.weight", model->w_head, embed_elems);
        } else {
            // Tied weights
            std::memcpy(model->w_head, model->w_embed, embed_elems * sizeof(float));
        }

        // Load layers
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            std::string prefix = "blk." + std::to_string(l) + ".";
            auto& lay = model->layers[l];

            lay.attn_norm = static_cast<float*>(aligned_alloc_64(cfg.hidden_dim * sizeof(float)));
            if (reader.has_tensor(prefix + "attn_norm.weight")) {
                reader.read_tensor_fp32(prefix + "attn_norm.weight", lay.attn_norm, cfg.hidden_dim);
            } else {
                for (size_t i = 0; i < cfg.hidden_dim; ++i) lay.attn_norm[i] = 1.0f;
            }

            lay.ffn_norm = static_cast<float*>(aligned_alloc_64(cfg.hidden_dim * sizeof(float)));
            if (reader.has_tensor(prefix + "ffn_norm.weight")) {
                reader.read_tensor_fp32(prefix + "ffn_norm.weight", lay.ffn_norm, cfg.hidden_dim);
            } else {
                for (size_t i = 0; i < cfg.hidden_dim; ++i) lay.ffn_norm[i] = 1.0f;
            }

            size_t q_elems = cfg.num_heads * cfg.head_dim * cfg.hidden_dim;
            lay.w_q = static_cast<float*>(aligned_alloc_64(q_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "attn_q.weight")) {
                reader.read_tensor_fp32(prefix + "attn_q.weight", lay.w_q, q_elems);
            }

            size_t kv_elems = cfg.num_kv_heads * cfg.head_dim * cfg.hidden_dim;
            lay.w_k = static_cast<float*>(aligned_alloc_64(kv_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "attn_k.weight")) {
                reader.read_tensor_fp32(prefix + "attn_k.weight", lay.w_k, kv_elems);
            }
            lay.w_v = static_cast<float*>(aligned_alloc_64(kv_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "attn_v.weight")) {
                reader.read_tensor_fp32(prefix + "attn_v.weight", lay.w_v, kv_elems);
            }

            lay.w_o = static_cast<float*>(aligned_alloc_64(q_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "attn_output.weight")) {
                reader.read_tensor_fp32(prefix + "attn_output.weight", lay.w_o, q_elems);
            }

            size_t ffn_elems = cfg.ffn_dim * cfg.hidden_dim;
            lay.w_gate = static_cast<float*>(aligned_alloc_64(ffn_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "ffn_gate.weight")) {
                reader.read_tensor_fp32(prefix + "ffn_gate.weight", lay.w_gate, ffn_elems);
            }
            lay.w_up = static_cast<float*>(aligned_alloc_64(ffn_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "ffn_up.weight")) {
                reader.read_tensor_fp32(prefix + "ffn_up.weight", lay.w_up, ffn_elems);
            }
            lay.w_down = static_cast<float*>(aligned_alloc_64(ffn_elems * sizeof(float)));
            if (reader.has_tensor(prefix + "ffn_down.weight")) {
                reader.read_tensor_fp32(prefix + "ffn_down.weight", lay.w_down, ffn_elems);
            }
        }

        return model;
    }

    // Evaluates 1 token forward step at pos, updates KV caches, and produces logits [vocab_size]
    void forward_token(int32_t token_id, size_t pos, float* logits_out) {
        size_t H = config.hidden_dim;
        size_t V = config.vocab_size;
        size_t n_heads = config.num_heads;
        size_t n_kv_heads = config.num_kv_heads;
        size_t head_dim = config.head_dim;
        size_t ffn_dim = config.ffn_dim;

        std::vector<float> x(H);
        std::vector<float> x_norm(H);
        std::vector<float> q(n_heads * head_dim);
        std::vector<float> k(n_kv_heads * head_dim);
        std::vector<float> v(n_kv_heads * head_dim);
        std::vector<float> attn_out(n_heads * head_dim);
        std::vector<float> o_out(H);
        std::vector<float> gate(ffn_dim);
        std::vector<float> up(ffn_dim);
        std::vector<float> down(H);

        // 1. Embedding lookup
        if (token_id >= 0 && static_cast<size_t>(token_id) < V && w_embed) {
            std::memcpy(x.data(), w_embed + token_id * H, H * sizeof(float));
        } else {
            std::fill(x.begin(), x.end(), 0.0f);
        }

        // 2. Transformer layers
        for (uint32_t l = 0; l < config.num_layers; ++l) {
            const auto& lay = layers[l];

            // RMSNorm 1
            rmsnorm(x.data(), lay.attn_norm, x_norm.data(), H, config.rms_norm_eps);

            // QKV GEMV
            gemv(lay.w_q, x_norm.data(), q.data(), n_heads * head_dim, H);
            gemv(lay.w_k, x_norm.data(), k.data(), n_kv_heads * head_dim, H);
            gemv(lay.w_v, x_norm.data(), v.data(), n_kv_heads * head_dim, H);

            // RoPE on Q and K
            apply_rope(q.data(), n_heads, head_dim, pos, config.rope_theta);
            apply_rope(k.data(), n_kv_heads, head_dim, pos, config.rope_theta);

            // Store in KV cache
            size_t kv_dim = n_kv_heads * head_dim;
            size_t cache_offset = pos * kv_dim;
            if (cache_offset + kv_dim <= k_cache[l].size()) {
                std::memcpy(k_cache[l].data() + cache_offset, k.data(), kv_dim * sizeof(float));
                std::memcpy(v_cache[l].data() + cache_offset, v.data(), kv_dim * sizeof(float));
            }

            // GQA Attention
            size_t kv_group = n_heads / n_kv_heads;
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            std::fill(attn_out.begin(), attn_out.end(), 0.0f);

            for (size_t h = 0; h < n_heads; ++h) {
                size_t kv_h = h / kv_group;
                const float* q_h = q.data() + h * head_dim;

                // Compute scores for past positions 0..pos
                std::vector<float> scores(pos + 1);
                float max_score = -1e9f;
                for (size_t t = 0; t <= pos; ++t) {
                    const float* k_t = k_cache[l].data() + t * kv_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) dot += q_h[d] * k_t[d];
                    scores[t] = dot * scale;
                    if (scores[t] > max_score) max_score = scores[t];
                }

                // Softmax
                float sum_exp = 0.0f;
                for (size_t t = 0; t <= pos; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (size_t t = 0; t <= pos; ++t) scores[t] *= inv_sum;

                // Weighted sum of V
                float* out_h = attn_out.data() + h * head_dim;
                for (size_t t = 0; t <= pos; ++t) {
                    const float* v_t = v_cache[l].data() + t * kv_dim + kv_h * head_dim;
                    float weight = scores[t];
                    for (size_t d = 0; d < head_dim; ++d) {
                        out_h[d] += weight * v_t[d];
                    }
                }
            }

            // Output projection
            gemv(lay.w_o, attn_out.data(), o_out.data(), H, n_heads * head_dim);

            // Residual 1
            for (size_t i = 0; i < H; ++i) x[i] += o_out[i];

            // RMSNorm 2
            rmsnorm(x.data(), lay.ffn_norm, x_norm.data(), H, config.rms_norm_eps);

            // SwiGLU MLP
            gemv(lay.w_gate, x_norm.data(), gate.data(), ffn_dim, H);
            gemv(lay.w_up, x_norm.data(), up.data(), ffn_dim, H);

            // SiLU(gate) * up
            for (size_t i = 0; i < ffn_dim; ++i) {
                float g = gate[i];
                float silu_g = g / (1.0f + std::exp(-g));
                gate[i] = silu_g * up[i];
            }

            // Down projection
            gemv(lay.w_down, gate.data(), down.data(), H, ffn_dim);

            // Residual 2
            for (size_t i = 0; i < H; ++i) x[i] += down[i];
        }

        // Final Norm
        rmsnorm(x.data(), w_norm, x_norm.data(), H, config.rms_norm_eps);

        // LM Head GEMV
        gemv(w_head, x_norm.data(), logits_out, V, H);
    }

    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        size_t max_new_tokens = 32,
        float temperature = 0.7f
    ) {
        std::vector<int32_t> result = prompt_tokens;
        if (prompt_tokens.empty()) return result;

        std::vector<float> logits(config.vocab_size);

        // Prefill
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            forward_token(prompt_tokens[i], i, logits.data());
        }

        // Autoregressive decode
        size_t cur_pos = prompt_tokens.size();
        for (size_t step = 0; step < max_new_tokens && cur_pos < config.max_pos; ++step) {
            int32_t next_tok = sample_logits(logits.data(), config.vocab_size, temperature);
            result.push_back(next_tok);
            if (next_tok == 2) break; // EOS

            forward_token(next_tok, cur_pos, logits.data());
            cur_pos++;
        }

        return result;
    }

private:
    static inline void rmsnorm(const float* x, const float* weight, float* out, size_t dim, float eps) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (size_t i = 0; i < dim; ++i) out[i] = x[i] * inv_rms * weight[i];
    }

    static inline void gemv(const float* W, const float* x, float* y, size_t M, size_t K) {
        if (!W) {
            std::fill(y, y + M, 0.0f);
            return;
        }
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            const float* row = W + i * K;
            for (size_t j = 0; j < K; ++j) {
                sum += row[j] * x[j];
            }
            y[i] = sum;
        }
    }

    static inline void apply_rope(float* vec, size_t n_heads, size_t head_dim, size_t pos, float theta) {
        for (size_t h = 0; h < n_heads; ++h) {
            float* h_ptr = vec + h * head_dim;
            for (size_t i = 0; i < head_dim / 2; ++i) {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float v0 = h_ptr[2 * i];
                float v1 = h_ptr[2 * i + 1];
                h_ptr[2 * i]     = v0 * cos_a - v1 * sin_a;
                h_ptr[2 * i + 1] = v0 * sin_a + v1 * cos_a;
            }
        }
    }

    static inline int32_t sample_logits(const float* logits, size_t vocab_size, float temperature) {
        if (temperature <= 0.0f) {
            // Greedy
            int32_t best_idx = 0;
            float max_val = logits[0];
            for (size_t i = 1; i < vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    best_idx = static_cast<int32_t>(i);
                }
            }
            return best_idx;
        }

        // Softmax with temperature
        float max_l = *std::max_element(logits, logits + vocab_size);
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp((logits[i] - max_l) / temperature);
            sum += probs[i];
        }

        float r = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) * sum;
        float acc = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            acc += probs[i];
            if (acc >= r) return static_cast<int32_t>(i);
        }
        return static_cast<int32_t>(vocab_size - 1);
    }
};

} // namespace turing
