#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <string>
#include <shared_mutex>
#include <mutex>

namespace turing {

/**
 * Native C++20 Thread-Safe Spectral Radix-SVD Trie Node & Matcher.
 */
struct RadixTrieNode {
    std::vector<int32_t> token_ids;
    std::unordered_map<int32_t, std::shared_ptr<RadixTrieNode>> children;
    int32_t node_id;

    explicit RadixTrieNode(std::vector<int32_t> tokens, int32_t id = 0)
        : token_ids(std::move(tokens)), node_id(id) {}
};

class RadixTrieIndex {
public:
    RadixTrieIndex() : root_(std::make_shared<RadixTrieNode>(std::vector<int32_t>{}, 0)), next_id_(1) {}

    int32_t match_longest_prefix(const std::vector<int32_t>& query_tokens, int32_t& matched_length) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        matched_length = 0;
        if (query_tokens.empty()) return 0;

        auto curr = root_;
        size_t q_idx = 0;

        while (q_idx < query_tokens.size()) {
            int32_t first_tok = query_tokens[q_idx];
            auto it = curr->children.find(first_tok);
            if (it == curr->children.end()) break;

            auto child = it->second;
            size_t m = 0;
            while (m < child->token_ids.size() && (q_idx + m) < query_tokens.size() &&
                   child->token_ids[m] == query_tokens[q_idx + m]) {
                m++;
            }

            matched_length += static_cast<int32_t>(m);
            q_idx += m;

            if (m == child->token_ids.size()) {
                curr = child;
            } else {
                break;
            }
        }
        return curr->node_id;
    }

    int32_t insert(const std::vector<int32_t>& tokens) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (tokens.empty()) return 0;

        auto curr = root_;
        size_t idx = 0;

        while (idx < tokens.size()) {
            int32_t first_tok = tokens[idx];
            auto it = curr->children.find(first_tok);

            if (it == curr->children.end()) {
                std::vector<int32_t> rem(tokens.begin() + idx, tokens.end());
                int32_t new_id = next_id_++;
                auto new_node = std::make_shared<RadixTrieNode>(rem, new_id);
                curr->children[first_tok] = new_node;
                return new_id;
            }

            auto child = it->second;
            size_t m = 0;
            while (m < child->token_ids.size() && (idx + m) < tokens.size() &&
                   child->token_ids[m] == tokens[idx + m]) {
                m++;
            }

            if (m == child->token_ids.size()) {
                idx += m;
                curr = child;
            } else {
                // Split child
                std::vector<int32_t> split_tokens(child->token_ids.begin() + m, child->token_ids.end());
                auto split_node = std::make_shared<RadixTrieNode>(split_tokens, child->node_id);
                split_node->children = std::move(child->children);

                child->token_ids.resize(m);
                child->node_id = next_id_++;
                child->children.clear();
                child->children[split_tokens[0]] = split_node;

                idx += m;
                if (idx < tokens.size()) {
                    std::vector<int32_t> rem(tokens.begin() + idx, tokens.end());
                    int32_t new_id = next_id_++;
                    auto new_node = std::make_shared<RadixTrieNode>(rem, new_id);
                    child->children[rem[0]] = new_node;
                    return new_id;
                }
                return child->node_id;
            }
        }
        return curr->node_id;
    }

    void set_anchor(const std::string& key, int32_t node_id) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        anchors_[key] = node_id;
    }

    int32_t get_anchor(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = anchors_.find(key);
        if (it != anchors_.end()) {
            return it->second;
        }
        return -1;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return static_cast<size_t>(next_id_ - 1);
    }

private:
    std::shared_ptr<RadixTrieNode> root_;
    int32_t next_id_;
    std::unordered_map<std::string, int32_t> anchors_;
    mutable std::shared_mutex mutex_;
};

} // namespace turing
