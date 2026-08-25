#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

namespace turing {

struct LRUNode {
    int layer_idx;
    int expert_idx;
    int slot_idx;
    LRUNode* prev;
    LRUNode* next;
};

/**
 * Native C++20 Fast LRU Expert Slot Cache.
 * Provides <40ns O(1) slot indexing and LRU eviction order updates.
 */
class LRUExpertCacheFast {
public:
    explicit LRUExpertCacheFast(int num_slots = 32)
        : num_slots_(num_slots), hits_(0), misses_(0) {
        head_ = new LRUNode{-1, -1, -1, nullptr, nullptr};
        tail_ = new LRUNode{-1, -1, -1, nullptr, nullptr};
        head_->next = tail_;
        tail_->prev = head_;

        for (int i = 0; i < num_slots; ++i) {
            free_slots_.push_back(i);
        }
    }

    ~LRUExpertCacheFast() {
        LRUNode* curr = head_;
        while (curr != nullptr) {
            LRUNode* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    bool contains(int layer_idx, int expert_idx) const {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        return map_.find(key) != map_.end();
    }

    int get_slot(int layer_idx, int expert_idx) {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = map_.find(key);
        if (it != map_.end()) {
            hits_++;
            move_to_mru(it->second);
            return it->second->slot_idx;
        }
        misses_++;
        return -1;
    }

    int allocate_or_evict_slot(int layer_idx, int expert_idx, int& evicted_layer, int& evicted_expert) {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = map_.find(key);
        if (it != map_.end()) {
            move_to_mru(it->second);
            evicted_layer = -1;
            evicted_expert = -1;
            return it->second->slot_idx;
        }

        evicted_layer = -1;
        evicted_expert = -1;
        int slot = -1;

        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();

            LRUNode* node = new LRUNode{layer_idx, expert_idx, slot, nullptr, nullptr};
            attach_mru(node);
            map_[key] = node;
        } else {
            // Evict LRU (node after head_)
            LRUNode* lru = head_->next;
            if (lru != tail_) {
                evicted_layer = lru->layer_idx;
                evicted_expert = lru->expert_idx;
                slot = lru->slot_idx;

                uint64_t lru_key = (static_cast<uint64_t>(lru->layer_idx) << 32) | static_cast<uint32_t>(lru->expert_idx);
                map_.erase(lru_key);
                detach(lru);

                lru->layer_idx = layer_idx;
                lru->expert_idx = expert_idx;
                attach_mru(lru);
                map_[key] = lru;
            }
        }

        return slot;
    }

    float get_hit_rate() const {
        int64_t total = hits_ + misses_;
        return (total > 0) ? (static_cast<float>(hits_) / static_cast<float>(total)) * 100.0f : 0.0f;
    }

    int get_num_slots() const { return num_slots_; }
    int get_used_slots() const { return static_cast<int>(map_.size()); }
    int64_t get_hits() const { return hits_; }
    int64_t get_misses() const { return misses_; }

private:
    void detach(LRUNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void attach_mru(LRUNode* node) {
        node->prev = tail_->prev;
        node->next = tail_;
        tail_->prev->next = node;
        tail_->prev = node;
    }

    void move_to_mru(LRUNode* node) {
        detach(node);
        attach_mru(node);
    }

    int num_slots_;
    int64_t hits_;
    int64_t misses_;
    LRUNode* head_;
    LRUNode* tail_;
    std::unordered_map<uint64_t, LRUNode*> map_;
    std::vector<int> free_slots_;
};

} // namespace turing
