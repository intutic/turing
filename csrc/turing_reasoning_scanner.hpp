#pragma once

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

namespace turing {

/**
 * Native C++20 Fast Streaming Reasoning & Tag Filter.
 * Performs zero-copy state-machine parsing over incoming token streams to separate
 * <think>...</think> chains and regular content chunks for SSE streaming.
 */
class NativeReasoningFilter {
public:
    NativeReasoningFilter()
        : in_thinking_block_(false),
          buffered_text_(""),
          thinking_content_(""),
          final_content_("") {}

    /**
     * Processes incoming token text chunk.
     * Returns std::pair<std::string, std::string> where:
     * - first: reasoning_chunk (or empty string if none)
     * - second: content_chunk (or empty string if none)
     */
    std::pair<std::string, std::string> process_token(const std::string& token_text) {
        buffered_text_ += token_text;

        constexpr std::string_view THINK_OPEN = "<think>";
        constexpr std::string_view THINK_CLOSE = "</think>";

        // 1. Check entry into <think>
        if (!in_thinking_block_) {
            size_t open_pos = buffered_text_.find(THINK_OPEN);
            if (open_pos != std::string::npos) {
                in_thinking_block_ = true;
                std::string before_think = buffered_text_.substr(0, open_pos);
                buffered_text_ = buffered_text_.substr(open_pos + THINK_OPEN.size());

                if (!before_think.empty()) {
                    final_content_ += before_think;
                }
                return { "", before_think };
            }
        }

        // 2. Check exit from </think>
        if (in_thinking_block_) {
            size_t close_pos = buffered_text_.find(THINK_CLOSE);
            if (close_pos != std::string::npos) {
                in_thinking_block_ = false;
                std::string think_part = buffered_text_.substr(0, close_pos);
                std::string after_think = buffered_text_.substr(close_pos + THINK_CLOSE.size());

                thinking_content_ += think_part;
                final_content_ += after_think;
                buffered_text_.clear();

                return { think_part, after_think };
            }
        }

        // 3. Normal streaming within or outside thinking block
        if (in_thinking_block_) {
            std::string chunk = buffered_text_;
            thinking_content_ += chunk;
            buffered_text_.clear();
            return { chunk, "" };
        } else {
            std::string chunk = buffered_text_;
            final_content_ += chunk;
            buffered_text_.clear();
            return { "", chunk };
        }
    }

    const std::string& get_thinking_content() const { return thinking_content_; }
    const std::string& get_final_content() const { return final_content_; }
    bool is_in_thinking_block() const { return in_thinking_block_; }

    void reset() {
        in_thinking_block_ = false;
        buffered_text_.clear();
        thinking_content_.clear();
        final_content_.clear();
    }

private:
    bool in_thinking_block_;
    std::string buffered_text_;
    std::string thinking_content_;
    std::string final_content_;
};

} // namespace turing
