#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include "turing_gguf_cpp.hpp"
#include "turing_tokenizer_cpp.hpp"
#include "turing_model_cpp.hpp"
#include "turing_http_server.hpp"

using namespace turing;

void print_banner() {
    std::cout << "======================================================================\n";
    std::cout << "⚡ Turing Engine Standalone C++20 Runtime (turing-cli v0.8.0)\n";
    std::cout << "   Zero Python Dependency | " << simd_backend() << " SIMD kernels | Direct GGUF mmap\n";
    std::cout << "======================================================================\n\n";
}

void print_usage() {
    std::cout << "Usage: turing-cli <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  generate    Run single-prompt text generation\n";
    std::cout << "  chat        Start interactive terminal chat session\n";
    std::cout << "  serve       Launch OpenAI-compatible HTTP serving endpoint\n";
    std::cout << "  logits      Print the logits of every position of --tokens as JSON (debug / parity)\n";
    std::cout << "  info        Display hardware SIMD and runtime capabilities\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>      Path to local .gguf binary file (required)\n";
    std::cout << "  --prompt <text>     Input prompt string (default: 'Hello')\n";
    std::cout << "  --tokens <ids>      Comma-separated token ids (logits command)\n";
    std::cout << "  --max-tokens <int>  Maximum new tokens to generate (default: 32)\n";
    std::cout << "  --temp <float>      Sampling temperature (default: 0.7)\n";
    std::cout << "  --ctx <int>         KV cache length, capped by the GGUF context_length (default: 4096)\n";
    std::cout << "  --kv-cache-dtype <native|int8>  KV cache storage (default: native fp32)\n";
    std::cout << "  --host <addr>       HTTP server bind address (default: 127.0.0.1)\n";
    std::cout << "  --port <int>        HTTP server port (default: 8000)\n";
}

static int32_t meta_token(const GGUFReaderCpp& reader, const std::string& key) {
    auto it = reader.metadata_ints.find(key);
    return it != reader.metadata_ints.end() ? static_cast<int32_t>(it->second) : -1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_banner();
        print_usage();
        return 0;
    }

    std::string command = argv[1];

    if (command == "info") {
        print_banner();
        std::cout << "[*] Hardware Capabilities:\n";
        std::cout << "    [+] SIMD backend: " << simd_backend() << "\n";
#if defined(TURING_HAS_AVX2_FMA)
        std::cout << "    [+] AVX2 SIMD 256-bit + FMA3: ENABLED\n";
#elif defined(TURING_HAS_NEON)
        std::cout << "    [+] ARM NEON 128-bit + FMA: ENABLED\n";
#else
        std::cout << "    [-] Scalar fallback (build with -mavx2 -mfma on x86_64)\n";
#endif
        std::cout << "    [+] Memory Alignment: 64-byte Cache-Line Aligned\n";
        std::cout << "    [+] OS Interface: POSIX/Win32 mmap Zero-Copy\n";
        return 0;
    }

    std::string model_path = "";
    std::string prompt = "Hello, world!";
    std::string tokens_arg = "";
    std::string host = "127.0.0.1";
    std::string kv_dtype = "native";
    int max_tokens = 32;
    float temp = 0.7f;
    int port = 8000;
    long ctx = 4096;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            tokens_arg = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--ctx" && i + 1 < argc) {
            ctx = std::atol(argv[++i]);
        } else if (arg == "--kv-cache-dtype" && i + 1 < argc) {
            kv_dtype = argv[++i];
        }
    }

    if (model_path.empty()) {
        std::cerr << "[!] Error: --model <path_to_gguf> is required.\n";
        return 1;
    }
    if (kv_dtype != "native" && kv_dtype != "int8") {
        std::cerr << "[!] Error: --kv-cache-dtype must be native or int8.\n";
        return 1;
    }
    if (ctx < 1) {
        std::cerr << "[!] Error: --ctx must be positive.\n";
        return 1;
    }

    try {
        GGUFReaderCpp reader(model_path);
        TokenizerCpp tokenizer(reader.tokenizer_tokens);
        // Special tokens come from the GGUF; a token the file does not name is not special.
        tokenizer.bos_token_id = meta_token(reader, "tokenizer.ggml.bos_token_id");
        tokenizer.eos_token_id = meta_token(reader, "tokenizer.ggml.eos_token_id");
        tokenizer.pad_token_id = meta_token(reader, "tokenizer.ggml.padding_token_id");
        auto model = TransformerModelCpp::load_from_gguf(reader, static_cast<size_t>(ctx), kv_dtype == "int8");
        const std::string chat_template = reader.metadata_strings.count("tokenizer.chat_template")
                                              ? reader.metadata_strings.at("tokenizer.chat_template") : "";

        auto complete = [&](const std::string& text, int n_tok, float temperature) {
            GenerationResult r;
            auto p_tokens = tokenizer.encode(text, true);
            auto out_t = model->generate(p_tokens, static_cast<size_t>(std::max(0, n_tok)), temperature);
            std::vector<int32_t> gen_t(out_t.begin() + static_cast<long>(p_tokens.size()), out_t.end());
            r.prompt_tokens = p_tokens.size();
            r.completion_tokens = gen_t.size();
            r.finish_reason = (!gen_t.empty() && gen_t.back() == model->config.eos_token_id) ? "stop" : "length";
            r.text = tokenizer.decode(gen_t);
            return r;
        };

        if (command == "generate") {
            auto prompt_tokens = tokenizer.encode(prompt, true);
            auto start_t = std::chrono::high_resolution_clock::now();
            auto out_tokens = model->generate(prompt_tokens, static_cast<size_t>(std::max(0, max_tokens)), temp);
            auto end_t = std::chrono::high_resolution_clock::now();

            std::vector<int32_t> new_tokens(out_tokens.begin() + static_cast<long>(prompt_tokens.size()), out_tokens.end());
            std::string response = tokenizer.decode(new_tokens);
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
            double tok_s = (new_tokens.size() / std::max(1e-4, elapsed_ms / 1000.0));

            std::cout << response << "\n";
            std::cerr << "(Generated " << new_tokens.size() << " tokens in " << elapsed_ms << "ms — " << tok_s << " tok/s)\n";
        }
        else if (command == "logits") {
            std::vector<int32_t> ids;
            std::stringstream ss(tokens_arg);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) ids.push_back(static_cast<int32_t>(std::atol(item.c_str())));
            }
            if (ids.empty()) {
                std::cerr << "[!] Error: logits needs --tokens <id,id,...>\n";
                return 1;
            }
            std::vector<float> logits(model->config.vocab_size);
            std::printf("{\"simd\": \"%s\", \"kv_cache_dtype\": \"%s\", \"logits\": [", simd_backend(), kv_dtype.c_str());
            for (size_t p = 0; p < ids.size(); ++p) {
                model->forward_token(ids[p], p, logits.data());
                std::printf("%s[", p ? ", " : "");
                for (size_t v = 0; v < logits.size(); ++v) std::printf("%s%.9g", v ? ", " : "", logits[v]);
                std::printf("]");
            }
            std::printf("]}\n");
        }
        else if (command == "chat") {
            print_banner();
            std::cout << "[*] Loaded model: " << model->config.name << " (Layers: " << model->config.num_layers << ", Hidden: " << model->config.hidden_dim << ")\n";
            std::cout << "Type your message and press Enter. Type 'exit' to quit.\n\n";

            std::vector<ChatMessage> history;
            while (true) {
                std::cout << "User > ";
                std::string input_text;
                if (!std::getline(std::cin, input_text) || input_text == "exit") {
                    std::cout << "\nGoodbye!\n";
                    break;
                }
                if (input_text.empty()) continue;
                history.push_back({"user", input_text});
                try {
                    GenerationResult r = complete(apply_chat_template(chat_template, history), max_tokens, temp);
                    history.push_back({"assistant", r.text});
                    std::cout << "\nAssistant > " << r.text << "\n\n";
                } catch (const std::out_of_range& e) {
                    std::cout << "\n[!] " << e.what() << "; starting a new conversation.\n\n";
                    history.clear();
                }
            }
        }
        else if (command == "serve") {
            print_banner();
            HttpServerCpp server(host, port, complete);
            server.model_name = model->config.name;
            server.chat_template = chat_template;
            server.default_max_tokens = max_tokens;
            server.default_temperature = temp;
            if (!server.run()) return 1;
        }
        else {
            std::cerr << "[!] Unknown command: " << command << "\n";
            print_usage();
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[!] Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
