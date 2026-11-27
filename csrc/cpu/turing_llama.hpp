#pragma once
// Turing's CPU backend on libllama (third_party/llama.cpp, pinned): one model, one sequence, the KV cache kept between
// calls so a prompt that shares a prefix with the last one only evaluates the rest. Used by the Python module
// (csrc/cpu/llama_bindings.cpp) and turing-cpu (csrc/cpu/turing_cpu_main.cpp). Decision CF1a in
// bmad-output/cpu-first/decision-log.md.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <memory>

#include "llama.h"
#if !defined(_WIN32)
#include "cpu/turing_expert_readahead.hpp"
#endif

namespace turing {

struct LlamaOptions {
    int n_ctx = 4096;
    int n_batch = 2048;      // prompt tokens per llama_process call
    int n_threads = 0;       // 0: all hardware threads
    bool use_mmap = true;    // map the GGUF instead of reading it into memory (models larger than RAM need it)
    bool repack = true;      // ggml's repacked weight layouts (ARM, AMX); copies weights out of the mapping
    bool exact = false;      // fp32 KV cache and no flash attention (libllama's defaults round K/V to fp16): parity tests
    bool verbose = false;    // libllama's own log lines
    // Wave 2: read each MoE layer's chosen experts from the file before the expert products fault them in
    // (csrc/cpu/turing_expert_readahead.hpp). Needs use_mmap and repack off (a repacked copy is already in memory).
    bool readahead = false;
    int readahead_threads = 16;
    size_t readahead_chunk = size_t(4) << 20;
    bool readahead_check_resident = true;  // false reads every chosen expert again (tests)
    bool readahead_wait = true;            // false: reads run beside the expert products instead of before them
};

class TuringLlama {
public:
    TuringLlama(const std::string& path, const LlamaOptions& opt) : opt_(opt) {
        static const bool once = [] { llama_backend_init(); return true; }();
        (void)once;
        llama_log_set(opt.verbose ? nullptr : &quiet_log, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = opt.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        mp.use_extra_bufts = opt.repack;
        model_ = llama_model_load_from_file(path.c_str(), mp);
        if (!model_) throw std::runtime_error("libllama could not load " + path);
        vocab_ = llama_model_get_vocab(model_);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(opt.n_ctx);
        cp.n_batch = static_cast<uint32_t>(opt.n_batch);
        cp.n_ubatch = static_cast<uint32_t>(std::min(opt.n_batch, 512));
        const int nt = opt.n_threads > 0 ? opt.n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        cp.n_threads = cp.n_threads_batch = nt;
        cp.no_perf = true;
        if (opt.exact) {
            cp.type_k = cp.type_v = GGML_TYPE_F32;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        if (opt.readahead) {
#if defined(_WIN32)
            llama_model_free(model_);
            throw std::runtime_error("expert read-ahead needs pread and mincore (not on Windows)");
#else
            if (!opt.use_mmap || opt.repack) {
                llama_model_free(model_);
                throw std::invalid_argument("expert read-ahead reads the mapped file: use_mmap on, repack off");
            }
            readahead_ = std::make_unique<ExpertReadAhead>(path, opt.readahead_threads, opt.readahead_chunk,
                                                           opt.readahead_check_resident, opt.readahead_wait);
            cp.cb_eval = &ExpertReadAhead::callback;
            cp.cb_eval_user_data = readahead_.get();
#endif
        }
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) {
            llama_model_free(model_);
            throw std::runtime_error("libllama could not create a context for " + path);
        }
        batch_ = llama_batch_ext_init(ctx_);
    }
    ~TuringLlama() {
        if (batch_) llama_batch_ext_free(batch_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_model_free(model_);
    }
    TuringLlama(const TuringLlama&) = delete;
    TuringLlama& operator=(const TuringLlama&) = delete;

    int n_vocab() const { return llama_vocab_n_tokens(vocab_); }
    int n_ctx() const { return static_cast<int>(llama_n_ctx(ctx_)); }
    int bos() const { return llama_vocab_bos(vocab_); }
    int eos() const { return llama_vocab_eos(vocab_); }
    bool is_eog(int token) const { return llama_vocab_is_eog(vocab_, token); }
    int n_past() const { return static_cast<int>(cached_.size()); }
    uint64_t n_params() const { return llama_model_n_params(model_); }
    uint64_t size_bytes() const { return llama_model_size(model_); }
    std::string desc() const {
        char buf[256];
        llama_model_desc(model_, buf, sizeof(buf));
        return buf;
    }
#if !defined(_WIN32)
    // Expert read-ahead counters (all zero when it is off) and the MoE layers it found in the file.
    ReadAheadStats readahead_stats() const { return readahead_ ? readahead_->stats() : ReadAheadStats{}; }
    size_t readahead_layers() const { return readahead_ ? readahead_->moe_layers() : 0; }
#endif

    std::vector<int> tokenize(const std::string& text, bool add_special = true, bool parse_special = true) const {
        int n = -llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
        std::vector<int> out(static_cast<size_t>(std::max(n, 0)));
        if (n > 0 && llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()), out.data(), n, add_special,
                                    parse_special) < 0)
            throw std::runtime_error("tokenize failed");
        return out;
    }
    std::string piece(int token) const {
        char buf[256];
        int n = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, true);
        if (n < 0) throw std::runtime_error("token_to_piece failed");
        return std::string(buf, static_cast<size_t>(n));
    }

    // Drop the KV cache beyond the first n tokens.
    void truncate(int n) {
        n = std::max(0, std::min(n, n_past()));
        if (n < n_past()) {
            llama_memory_seq_rm(llama_get_memory(ctx_), 0, n, -1);
            cached_.resize(static_cast<size_t>(n));
        }
    }

    // Append tokens after the cached ones; logits of the last token ([n_vocab]) or of every token ([n, n_vocab]).
    std::vector<float> eval(const std::vector<int>& tokens, bool all_logits = false) {
        if (tokens.empty()) throw std::invalid_argument("eval needs at least one token");
        if (n_past() + static_cast<int>(tokens.size()) > n_ctx()) throw std::length_error("context is full");
        const size_t V = static_cast<size_t>(n_vocab());
        const size_t step = llama_n_batch(ctx_);  // libllama caps the requested n_batch at n_ctx
        std::vector<float> out;
        out.reserve(all_logits ? tokens.size() * V : V);
        for (size_t c0 = 0; c0 < tokens.size(); c0 += step) {
            const size_t c1 = std::min(tokens.size(), c0 + step);
            llama_batch_ext_clear(batch_);
            for (size_t i = c0; i < c1; ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch_, 0, tokens[i]);
                const llama_pos pos = static_cast<llama_pos>(n_past() + (i - c0));
                llama_batch_ext_set_pos(batch_, idx, &pos);
                if (all_logits || i + 1 == tokens.size()) llama_batch_ext_set_output_logits(batch_, idx, true);
            }
            const int32_t rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_);
            if (rc != 0) throw std::runtime_error("llama_process returned " + std::to_string(rc));
            cached_.insert(cached_.end(), tokens.begin() + static_cast<long>(c0), tokens.begin() + static_cast<long>(c1));
            for (size_t i = c0; i < c1; ++i) {
                if (!(all_logits || i + 1 == tokens.size())) continue;
                const float* row = llama_get_logits_ith(ctx_, static_cast<int32_t>(i - c0));
                if (!row) throw std::runtime_error("no logits for an output token");
                out.insert(out.end(), row, row + V);
            }
        }
        return out;
    }

    // Make the cache hold exactly `prompt`, keeping the prefix it shares with the cached tokens and evaluating the rest;
    // the last prompt token's logits [n_vocab]. How many prompt tokens came from the cache: cached_prompt_tokens().
    std::vector<float> prefill(const std::vector<int>& prompt) {
        if (prompt.empty()) throw std::invalid_argument("prefill needs a prompt");
        size_t common = 0;
        while (common < cached_.size() && common < prompt.size() && cached_[common] == prompt[common]) ++common;
        if (common == prompt.size()) --common;  // re-evaluate the last prompt token to get its logits
        truncate(static_cast<int>(common));
        reused_ = static_cast<int>(common);
        return eval(std::vector<int>(prompt.begin() + static_cast<long>(common), prompt.end()));
    }
    int cached_prompt_tokens() const { return reused_; }

    // Generate after `prompt`, reusing the cached prefix it shares with the last call. temperature <= 0 is greedy.
    // on_token returns false to stop. Stops at an end-of-generation token (not returned) or after max_new tokens.
    std::vector<int> generate(const std::vector<int>& prompt, int max_new, float temperature = 0.0f, float top_p = 1.0f,
                              int top_k = 0, uint32_t seed = 0, const std::function<bool(int)>& on_token = {}) {
        if (prompt.empty()) throw std::invalid_argument("generate needs a prompt");
        llama_sampler* smpl = make_sampler(temperature, top_p, top_k, seed);
        std::vector<int> out;
        try {
            prefill(prompt);
            for (int i = 0; i < max_new && n_past() < n_ctx(); ++i) {
                const int tok = llama_sampler_sample(smpl, ctx_, -1);
                if (is_eog(tok)) break;
                out.push_back(tok);
                if (on_token && !on_token(tok)) break;
                if (i + 1 < max_new) eval({tok});
            }
        } catch (...) {
            llama_sampler_free(smpl);
            throw;
        }
        llama_sampler_free(smpl);
        return out;
    }

    // Greedy generation with prompt-lookup drafts (Wave 3): tokens that followed the latest n-gram where it last appeared
    // in the prompt or output are verified in one evaluation and kept while each is the argmax (greedy in exact
    // arithmetic; libllama's batched kernels round differently from its one-token kernels, so a near-tie can flip).
    // A cost gate decides how many to draft: on a CPU, verifying k drafts costs several one-token steps (2.5x for k=4,
    // Qwen3-8B Q4_K_M on Neoverse-V2: results_cpu_waves123_c4a.json), so it drafts the k that maximizes expected tokens
    // per unit of measured cost, and none when no k pays, probing now and then to notice text that starts to repeat.
    std::vector<int> generate_lookup(const std::vector<int>& prompt, int max_new, int n_draft,
                                     const std::function<bool(int)>& on_token = {}) {
        std::vector<int> hist(prompt), out;
        int cur = argmax(prefill(prompt).data());
        bool stop = false;
        while (!stop && !is_eog(cur) && static_cast<int>(out.size()) < max_new) {
            out.push_back(cur);
            hist.push_back(cur);
            if ((on_token && !on_token(cur)) || static_cast<int>(out.size()) >= max_new || n_past() + 1 >= n_ctx()) break;
            const int room = std::min({n_draft, kMaxDraft, max_new - static_cast<int>(out.size()), n_ctx() - n_past() - 1});
            std::vector<int> drafts = lookup(hist, std::max(0, room));
            const int k = choose_draft(static_cast<int>(drafts.size()));
            drafts.resize(static_cast<size_t>(k));
            std::vector<int> feed{cur};
            feed.insert(feed.end(), drafts.begin(), drafts.end());
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<float> rows = eval(feed, /*all_logits=*/true);  // row i predicts the token after feed[i]
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double& t = k == 0 ? t1_ : tk_[k];
            t = t == 0.0 ? dt : 0.8 * t + 0.2 * dt;
            const size_t V = static_cast<size_t>(n_vocab());
            size_t a = 0;
            int pred = argmax(rows.data());
            while (a < drafts.size() && pred == drafts[a]) {
                if (is_eog(pred)) {
                    stop = true;
                    break;
                }
                out.push_back(pred);
                hist.push_back(pred);
                ++a;
                if ((on_token && !on_token(pred)) || static_cast<int>(out.size()) >= max_new) {
                    stop = true;
                    break;
                }
                pred = argmax(rows.data() + a * V);
            }
            truncate(n_past() - static_cast<int>(drafts.size() - a));  // drop the rejected drafts' K/V
            spec_.steps += 1;
            spec_.drafted += drafts.size();
            spec_.accepted += a;
            if (k > 0) {  // per-draft acceptance: a successes, then one failure unless every draft was kept
                succ_ = 0.98 * succ_ + static_cast<double>(a);
                fail_ = 0.98 * fail_ + (a < drafts.size() ? 1.0 : 0.0);
            }
            cur = pred;
        }
        return out;
    }
    struct SpecStats {
        uint64_t steps = 0, drafted = 0, accepted = 0;
    };
    SpecStats spec_stats() const { return spec_; }

private:
    static constexpr int kMaxDraft = 16;
    // Drafts worth verifying out of `available`: the k maximizing (1 + p + ... + p^k) / cost(k), where p is the measured
    // per-draft acceptance and cost(k) the measured time of a k-draft step over a one-token step (prior 1 + 0.35 k until
    // measured); 0 when no k beats a one-token step, except one probe every 32 steps.
    int choose_draft(int available) {
        if (available <= 0) return 0;
        const double p = succ_ / (succ_ + fail_);
        int best_k = 0;
        double best = 1.0, gain = 1.0, pk = 1.0;
        for (int k = 1; k <= available; ++k) {
            pk *= p;
            gain += pk;
            const double cost = (tk_[k] > 0.0 && t1_ > 0.0) ? tk_[k] / t1_ : 1.0 + 0.35 * k;
            if (gain / cost > best) best = gain / cost, best_k = k;
        }
        if (best_k == 0 && ++since_probe_ >= 32) since_probe_ = 0, best_k = std::min(available, 2);
        return best_k;
    }
    int argmax(const float* row) const {
        const int V = n_vocab();
        return static_cast<int>(std::max_element(row, row + V) - row);
    }
    // Up to n tokens that followed the most recent earlier occurrence of the last 3, 2 or 1 tokens of hist.
    static std::vector<int> lookup(const std::vector<int>& hist, int n) {
        if (n <= 0) return {};
        for (int g = 3; g >= 1; --g) {
            if (static_cast<int>(hist.size()) <= g) continue;
            const auto tail = hist.end() - g;
            for (auto p = hist.end() - g - 1; p >= hist.begin(); --p) {
                if (std::equal(tail, hist.end(), p)) {
                    const auto from = p + g, to = std::min(hist.end(), from + n);
                    if (from < to) return std::vector<int>(from, to);
                    break;
                }
                if (p == hist.begin()) break;
            }
        }
        return {};
    }

    static void quiet_log(ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }
    static llama_sampler* make_sampler(float temperature, float top_p, int top_k, uint32_t seed) {
        llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (temperature <= 0.0f) {
            llama_sampler_chain_add(s, llama_sampler_init_greedy());
            return s;
        }
        if (top_k > 0) llama_sampler_chain_add(s, llama_sampler_init_top_k(top_k));
        if (top_p < 1.0f) llama_sampler_chain_add(s, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(seed));
        return s;
    }

    LlamaOptions opt_;
#if !defined(_WIN32)
    std::unique_ptr<ExpertReadAhead> readahead_;  // declared before ctx_: the context's callback points at it
#endif
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_batch_ext* batch_ = nullptr;
    std::vector<int> cached_;  // tokens whose K/V are in the cache, in order
    int reused_ = 0;           // prompt tokens the last prefill took from the cache
    SpecStats spec_;           // prompt-lookup drafts verified and accepted
    double succ_ = 4.0, fail_ = 1.0;          // decayed draft successes / failures (prior p = 0.8: explore first)
    double t1_ = 0.0, tk_[kMaxDraft + 1] = {}; // measured seconds of a one-token step and of a k-draft step
    int since_probe_ = 0;
};

}  // namespace turing
