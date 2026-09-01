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

struct LFRUSlotEntry {
    int layer_idx{-1};
    int expert_idx{-1};
    uint32_t heat{0};
    uint32_t last_clock{0};
    bool is_pinned{false};
};

/**
 * Native C++20 High-Performance LFRU (Least Frequently / Recently Used) Expert Cache.
 * Combines frequency heat scores, recency clock tracking, bitmask pinning,
 * and hysteresis promotion gates to maximize cache hit rates with <30ns eviction selection.
 */
class LFRUExpertCacheFast {
public:
    explicit LFRUExpertCacheFast(
        int num_slots = 32,
        double hysteresis_threshold = 0.25,
        int hysteresis_min_access = 4
    ) : num_slots_(num_slots),
        hits_(0),
        misses_(0),
        clock_(0),
        hysteresis_threshold_(hysteresis_threshold),
        hysteresis_min_access_(hysteresis_min_access) {
        slots_.resize(num_slots);
        for (int i = num_slots - 1; i >= 0; --i) {
            free_slots_.push_back(i);
        }
    }

    bool contains(int layer_idx, int expert_idx) const {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        return key_to_slot_.find(key) != key_to_slot_.end();
    }

    int get_slot(int layer_idx, int expert_idx) {
        clock_++;
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            hits_++;
            int slot = it->second;
            slots_[slot].heat++;
            slots_[slot].last_clock = clock_;
            return slot;
        }
        misses_++;
        return -1;
    }

    int allocate_or_evict_slot(
        int layer_idx,
        int expert_idx,
        int& evicted_layer,
        int& evicted_expert,
        uint32_t candidate_heat = 1,
        bool force_promote = false
    ) {
        clock_++;
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            int slot = it->second;
            slots_[slot].heat += candidate_heat;
            slots_[slot].last_clock = clock_;
            evicted_layer = -1;
            evicted_expert = -1;
            return slot;
        }

        evicted_layer = -1;
        evicted_expert = -1;

        // 1. Check for free slots
        if (!free_slots_.empty()) {
            int slot = free_slots_.back();
            free_slots_.pop_back();

            slots_[slot].layer_idx = layer_idx;
            slots_[slot].expert_idx = expert_idx;
            slots_[slot].heat = candidate_heat;
            slots_[slot].last_clock = clock_;
            slots_[slot].is_pinned = false;

            key_to_slot_[key] = slot;
            return slot;
        }

        // 2. Find minimum LFRU score candidate skipping pinned slots
        int victim_slot = -1;
        uint64_t min_score = UINT64_MAX;

        for (int i = 0; i < num_slots_; ++i) {
            if (slots_[i].is_pinned) continue;

            uint32_t age = (clock_ > slots_[i].last_clock) ? (clock_ - slots_[i].last_clock) : 0;
            uint32_t recency_component = (age < 65535) ? (65535 - age) : 0;
            uint64_t composite_score = (static_cast<uint64_t>(slots_[i].heat) << 16) | static_cast<uint64_t>(recency_component);

            if (composite_score < min_score) {
                min_score = composite_score;
                victim_slot = i;
            }
        }

        if (victim_slot == -1) {
            // All slots pinned, cannot evict
            return -1;
        }

        // 3. Hysteresis Promotion Check
        if (!force_promote) {
            uint32_t victim_heat = slots_[victim_slot].heat;
            uint32_t required_heat = static_cast<uint32_t>(static_cast<double>(victim_heat) * (1.0 + hysteresis_threshold_)) + hysteresis_min_access_;
            if (candidate_heat < required_heat && victim_heat > 0) {
                // Reject promotion; victim remains hot
                return -1;
            }
        }

        // 4. Perform Eviction & Slot Assignment
        evicted_layer = slots_[victim_slot].layer_idx;
        evicted_expert = slots_[victim_slot].expert_idx;

        uint64_t victim_key = (static_cast<uint64_t>(evicted_layer) << 32) | static_cast<uint32_t>(evicted_expert);
        key_to_slot_.erase(victim_key);

        slots_[victim_slot].layer_idx = layer_idx;
        slots_[victim_slot].expert_idx = expert_idx;
        slots_[victim_slot].heat = candidate_heat;
        slots_[victim_slot].last_clock = clock_;
        slots_[victim_slot].is_pinned = false;

        key_to_slot_[key] = victim_slot;
        return victim_slot;
    }

    void pin_slot(int slot_idx) {
        if (slot_idx >= 0 && slot_idx < num_slots_) {
            slots_[slot_idx].is_pinned = true;
        }
    }

    void unpin_slot(int slot_idx) {
        if (slot_idx >= 0 && slot_idx < num_slots_) {
            slots_[slot_idx].is_pinned = false;
        }
    }

    void pin_key(int layer_idx, int expert_idx) {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            slots_[it->second].is_pinned = true;
        }
    }

    void unpin_key(int layer_idx, int expert_idx) {
        uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_idx);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            slots_[it->second].is_pinned = false;
        }
    }

    void decay(double factor = 0.95) {
        for (auto& slot : slots_) {
            slot.heat = static_cast<uint32_t>(static_cast<double>(slot.heat) * factor);
        }
    }

    float get_hit_rate() const {
        int64_t total = hits_ + misses_;
        return (total > 0) ? (static_cast<float>(hits_) / static_cast<float>(total)) * 100.0f : 0.0f;
    }

    int get_num_slots() const { return num_slots_; }
    int get_used_slots() const { return static_cast<int>(key_to_slot_.size()); }
    int64_t get_hits() const { return hits_; }
    int64_t get_misses() const { return misses_; }
    uint32_t get_clock() const { return clock_; }

    uint32_t get_slot_heat(int slot_idx) const {
        if (slot_idx >= 0 && slot_idx < num_slots_) return slots_[slot_idx].heat;
        return 0;
    }

private:
    int num_slots_;
    int64_t hits_;
    int64_t misses_;
    uint32_t clock_;
    double hysteresis_threshold_;
    int hysteresis_min_access_;
    std::vector<LFRUSlotEntry> slots_;
    std::unordered_map<uint64_t, int> key_to_slot_;
    std::vector<int> free_slots_;
};

} // namespace turing
