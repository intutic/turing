// turing-cpu: the libllama CPU backend from the command line.
//   turing-cpu bench    --model m.gguf [--threads N] [--prompt-tokens 512] [--gen-tokens 128] [--reps 3] [--no-mmap] [--no-repack]
//                       [--seqs N] (N sequences decoded together: total and per-sequence tokens/s)
//   turing-cpu generate --model m.gguf --prompt "text" [--max-tokens 64] [--temp 0] [--threads N] [--draft N] (prompt lookup)
// bench prints one JSON line: prompt and decode tokens/s (median of --reps), as llama-bench's pp/tg rows measure them.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cpu/turing_llama.hpp"

using clk = std::chrono::steady_clock;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: turing-cpu bench|generate --model m.gguf [options]\n";
        return 2;
    }
    const std::string cmd = argv[1];
    std::string model, prompt = "The capital of France is";
    int threads = 0, n_prompt = 512, n_gen = 128, reps = 3, max_tokens = 64, n_draft = 0, n_seqs = 1;
    float temp = 0.0f;
    turing::LlamaOptions opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--prompt-tokens") n_prompt = std::stoi(next());
        else if (a == "--gen-tokens") n_gen = std::stoi(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--prompt") prompt = next();
        else if (a == "--max-tokens") max_tokens = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--draft") n_draft = std::stoi(next());
        else if (a == "--seqs") n_seqs = std::stoi(next());
        else if (a == "--no-mmap") opt.use_mmap = false;
        else if (a == "--no-repack") opt.repack = false;
        else if (a == "--readahead") opt.readahead = true, opt.repack = false;  // reads the mapped file: no repacked copy
        else if (a == "--readahead-threads") opt.readahead_threads = std::stoi(next());
        else if (a == "--readahead-async") opt.readahead = true, opt.repack = false, opt.readahead_wait = false;
        else if (a == "--readahead-always") opt.readahead = true, opt.repack = false;  // the default now; kept for scripts
        else if (a == "--readahead-adaptive") opt.readahead = true, opt.repack = false, opt.readahead_adaptive = true;
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::cerr << "unknown option " << a << "\n";
            return 2;
        }
    }
    if (model.empty()) {
        std::cerr << "--model is required\n";
        return 2;
    }
    opt.n_threads = threads;
    opt.n_seq = n_seqs;
    opt.n_ctx = std::max(4096, n_seqs * (n_prompt + n_gen + 8));
    try {
        const auto t_load = clk::now();
        turing::TuringLlama llm(model, opt);
        const double load_s = std::chrono::duration<double>(clk::now() - t_load).count();
        if (cmd == "generate") {
            const auto ids = llm.tokenize(prompt);
            const auto t0 = clk::now();
            const auto print = [&](int t) {
                std::cout << llm.piece(t) << std::flush;
                return true;
            };
            const auto out = n_draft > 0 && temp <= 0.0f ? llm.generate_lookup(ids, max_tokens, n_draft, print)
                                                         : llm.generate(ids, max_tokens, temp, 1.0f, 0, 0, print);
            const double s = std::chrono::duration<double>(clk::now() - t0).count();
            const auto sp = llm.spec_stats();
            std::cerr << "\n(" << out.size() << " tokens in " << s << " s, " << out.size() / s << " tok/s incl. prompt";
            if (n_draft > 0) std::cerr << "; drafts " << sp.drafted << " accepted " << sp.accepted << " in " << sp.steps << " steps";
            std::cerr << ")\n";
            return 0;
        }
        if (cmd != "bench") {
            std::cerr << "unknown command " << cmd << "\n";
            return 2;
        }
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> pick(0, llm.n_vocab() - 1);
        if (n_seqs > 1) {  // Wave 7: n_seqs sequences prefilled together, then decoded together, one token each per step
            std::vector<int> seqs(static_cast<size_t>(n_seqs)), pos(static_cast<size_t>(n_seqs), 0);
            std::vector<std::vector<int>> prompts(static_cast<size_t>(n_seqs));
            for (int i = 0; i < n_seqs; ++i) {
                seqs[i] = i;
                prompts[i].resize(static_cast<size_t>(n_prompt));
                for (auto& t : prompts[i]) t = pick(rng);
            }
            std::vector<double> pp, tg;
            for (int r = 0; r <= reps; ++r) {  // rep 0 warms up
                for (int i = 0; i < n_seqs; ++i) llm.seq_clear(i);
                auto t0 = clk::now();
                llm.step(seqs, prompts, std::vector<int>(static_cast<size_t>(n_seqs), 0));
                const double p_s = std::chrono::duration<double>(clk::now() - t0).count();
                for (int i = 0; i < n_seqs; ++i) pos[i] = n_prompt;
                t0 = clk::now();
                for (int g = 0; g < n_gen; ++g) {
                    std::vector<std::vector<int>> one(static_cast<size_t>(n_seqs), std::vector<int>{0});
                    for (auto& o : one) o[0] = pick(rng);
                    llm.step(seqs, one, pos);
                    for (auto& q : pos) ++q;
                }
                const double g_s = std::chrono::duration<double>(clk::now() - t0).count();
                if (r > 0) pp.push_back(n_seqs * n_prompt / p_s), tg.push_back(n_seqs * n_gen / g_s);
            }
            std::printf("{\"model\": \"%s\", \"desc\": \"%s\", \"threads\": %d, \"seqs\": %d, \"repack\": %s, "
                        "\"pp%d_total_tok_s\": %.2f, \"tg%d_total_tok_s\": %.2f, \"tg_per_seq_tok_s\": %.2f, \"reps\": %d}\n",
                        model.c_str(), llm.desc().c_str(), threads, n_seqs, opt.repack ? "true" : "false", n_prompt,
                        median(pp), n_gen, median(tg), median(tg) / n_seqs, reps);
            return 0;
        }
        std::vector<int> toks(static_cast<size_t>(n_prompt));
        for (auto& t : toks) t = pick(rng);
        llm.truncate(0);
        llm.eval(std::vector<int>(toks.begin(), toks.begin() + std::min(n_prompt, 16)));  // warm-up
        llm.eval({pick(rng)});
        std::vector<double> pp, tg;
        for (int r = 0; r < reps; ++r) {
            llm.truncate(0);
            auto t0 = clk::now();
            llm.eval(toks);
            pp.push_back(n_prompt / std::chrono::duration<double>(clk::now() - t0).count());
            llm.truncate(0);
            llm.eval({llm.bos() >= 0 ? llm.bos() : pick(rng)});
            t0 = clk::now();
            for (int i = 0; i < n_gen; ++i) llm.eval({pick(rng)});
            tg.push_back(n_gen / std::chrono::duration<double>(clk::now() - t0).count());
        }
        const turing::ReadAheadStats ra = llm.readahead_stats();
        std::printf("{\"model\": \"%s\", \"desc\": \"%s\", \"threads\": %d, \"mmap\": %s, \"repack\": %s, \"load_s\": %.2f, "
                    "\"pp%d_tok_s\": %.2f, \"tg%d_tok_s\": %.2f, \"reps\": %d, \"readahead\": %s, "
                    "\"readahead_bytes_read\": %llu, \"readahead_bytes_missing\": %llu, \"readahead_bytes_wanted\": %llu, "
                    "\"readahead_active_calls\": %llu, \"readahead_calls\": %llu, \"readahead_s\": %.3f}\n",
                    model.c_str(), llm.desc().c_str(), threads, opt.use_mmap ? "true" : "false",
                    opt.repack ? "true" : "false", load_s, n_prompt, median(pp), n_gen, median(tg), reps,
                    opt.readahead ? "true" : "false", static_cast<unsigned long long>(ra.bytes_read),
                    static_cast<unsigned long long>(ra.bytes_missing), static_cast<unsigned long long>(ra.bytes_wanted),
                    static_cast<unsigned long long>(ra.active_calls), static_cast<unsigned long long>(ra.calls), ra.seconds);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
