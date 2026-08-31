#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

namespace turing {

class TokenizerCpp {
public:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t bos_token_id = 1;
    int32_t eos_token_id = 2;
    int32_t pad_token_id = 2;
    int32_t unk_token_id = 0;

    TokenizerCpp() = default;

    explicit TokenizerCpp(const std::vector<std::string>& tokens) {
        id_to_token = tokens;
        for (size_t i = 0; i < tokens.size(); ++i) {
            token_to_id[tokens[i]] = static_cast<int32_t>(i);
        }
    }

    size_t vocab_size() const {
        return id_to_token.size();
    }

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = false) const {
        std::vector<int32_t> result;
        if (add_special_tokens && bos_token_id >= 0) {
            result.push_back(bos_token_id);
        }
        if (text.empty()) return result;

        // Direct full text match check
        auto it = token_to_id.find(text);
        if (it != token_to_id.end()) {
            result.push_back(it->second);
            return result;
        }

        // Greedy word / whitespace tokenizer
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::string spaced_word = " " + word;
            auto it_sp = token_to_id.find(spaced_word);
            if (it_sp != token_to_id.end()) {
                result.push_back(it_sp->second);
            } else {
                auto it_w = token_to_id.find(word);
                if (it_w != token_to_id.end()) {
                    result.push_back(it_w->second);
                } else {
                    // Character fallback
                    for (char c : word) {
                        std::string char_str(1, c);
                        auto it_c = token_to_id.find(char_str);
                        if (it_c != token_to_id.end()) {
                            result.push_back(it_c->second);
                        } else {
                            if (vocab_size() > 0) {
                                result.push_back(static_cast<uint8_t>(c) % vocab_size());
                            } else {
                                result.push_back(unk_token_id);
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = true) const {
        std::string out;
        for (int32_t t : tokens) {
            if (skip_special && (t == bos_token_id || t == eos_token_id || t == pad_token_id)) {
                continue;
            }
            if (t >= 0 && static_cast<size_t>(t) < id_to_token.size()) {
                std::string tok = id_to_token[t];
                // Replace SPM   with space
                size_t pos = 0;
                while ((pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos) {
                    tok.replace(pos, 3, " ");
                    pos += 1;
                }
                out += tok;
            } else {
                if (t >= 32 && t <= 126) {
                    out += static_cast<char>(t);
                } else {
                    out += " ";
                }
            }
        }
        return out;
    }

    std::string format_chat_prompt(const std::string& user_message) const {
        return "<|im_start|>user\n" + user_message + "<|im_end|>\n<|im_start|>assistant\n";
    }
};

} // namespace turing
