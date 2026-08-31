#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include "turing_gguf_cpp.hpp"
#include "turing_tokenizer_cpp.hpp"
#include "turing_model_cpp.hpp"
#include "turing_http_server.hpp"

using namespace turing;

void print_banner() {
    std::cout << "======================================================================\n";
    std::cout << "⚡ Turing Engine Standalone C++20 Runtime (turing-cli v0.8.0)\n";
    std::cout << "   Zero Python Dependency | Bare-Metal AVX2 SIMD | Direct GGUF mmap\n";
    std::cout << "======================================================================\n\n";
}

void print_usage() {
    std::cout << "Usage: turing-cli <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  generate    Run single-prompt text generation\n";
    std::cout << "  chat        Start interactive terminal chat session\n";
    std::cout << "  serve       Launch OpenAI-compatible HTTP serving endpoint\n";
    std::cout << "  info        Display hardware SIMD and runtime capabilities\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>      Path to local .gguf binary file (required)\n";
    std::cout << "  --prompt <text>     Input prompt string (default: 'Hello')\n";
    std::cout << "  --max-tokens <int>  Maximum new tokens to generate (default: 32)\n";
    std::cout << "  --temp <float>      Sampling temperature (default: 0.7)\n";
    std::cout << "  --port <int>        HTTP server port (default: 8000)\n";
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
#if defined(__AVX2__)
        std::cout << "    [+] AVX2 SIMD 256-bit Vector Extensions: ENABLED\n";
#else
        std::cout << "    [-] AVX2 SIMD: Scalar Fallback\n";
#endif
#if defined(__FMA__)
        std::cout << "    [+] Fused Multiply-Add (FMA3): ENABLED\n";
#endif
        std::cout << "    [+] Memory Alignment: 64-byte Cache-Line Aligned\n";
        std::cout << "    [+] OS Interface: POSIX/Win32 mmap Zero-Copy\n";
        return 0;
    }

    std::string model_path = "";
    std::string prompt = "Hello, world!";
    int max_tokens = 32;
    float temp = 0.7f;
    int port = 8000;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        std::cerr << "[!] Error: --model <path_to_gguf> is required.\n";
        return 1;
    }

    try {
        // Load GGUF model and tokenizer
        GGUFReaderCpp reader(model_path);
        TokenizerCpp tokenizer(reader.tokenizer_tokens);
        auto model = TransformerModelCpp::load_from_gguf(reader);

        if (command == "generate") {
            auto prompt_tokens = tokenizer.encode(prompt, true);
            auto start_t = std::chrono::high_resolution_clock::now();
            auto out_tokens = model->generate(prompt_tokens, max_tokens, temp);
            auto end_t = std::chrono::high_resolution_clock::now();

            std::vector<int32_t> new_tokens;
            if (out_tokens.size() > prompt_tokens.size()) {
                new_tokens.assign(out_tokens.begin() + prompt_tokens.size(), out_tokens.end());
            } else {
                new_tokens = out_tokens;
            }

            std::string response = tokenizer.decode(new_tokens);
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
            double tok_s = (new_tokens.size() / std::max(1e-4, elapsed_ms / 1000.0));

            std::cout << response << "\n";
            std::cerr << "(Generated " << new_tokens.size() << " tokens in " << elapsed_ms << "ms — " << tok_s << " tok/s)\n";
        }
        else if (command == "chat") {
            print_banner();
            std::cout << "[*] Loaded model: " << model->config.name << " (Layers: " << model->config.num_layers << ", Hidden: " << model->config.hidden_dim << ")\n";
            std::cout << "Type your message and press Enter. Type 'exit' to quit.\n\n";

            while (true) {
                std::cout << "User > ";
                std::string input_text;
                if (!std::getline(std::cin, input_text) || input_text == "exit") {
                    std::cout << "\nGoodbye!\n";
                    break;
                }
                if (input_text.empty()) continue;

                std::string formatted = tokenizer.format_chat_prompt(input_text);
                auto prompt_tokens = tokenizer.encode(formatted);
                auto out_tokens = model->generate(prompt_tokens, max_tokens, temp);

                std::vector<int32_t> new_tokens;
                if (out_tokens.size() > prompt_tokens.size()) {
                    new_tokens.assign(out_tokens.begin() + prompt_tokens.size(), out_tokens.end());
                } else {
                    new_tokens = out_tokens;
                }

                std::string response = tokenizer.decode(new_tokens);
                std::cout << "\nAssistant > " << response << "\n\n";
            }
        }
        else if (command == "serve") {
            print_banner();
            HttpServerCpp server(port, [&](const std::string& p, int n_tok) -> std::string {
                auto p_tokens = tokenizer.encode(p);
                auto out_t = model->generate(p_tokens, n_tok, temp);
                std::vector<int32_t> gen_t;
                if (out_t.size() > p_tokens.size()) {
                    gen_t.assign(out_t.begin() + p_tokens.size(), out_t.end());
                } else {
                    gen_t = out_t;
                }
                return tokenizer.decode(gen_t);
            });
            server.run();
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
