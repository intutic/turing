#pragma once
// Wave 2 of bmad-output/cpu-first/plan.md: models larger than RAM, experts read ahead of the expert products.
//
// libllama memory-maps the GGUF, so an expert that is not in the page cache is faulted in 4-16 KB at a time by the
// threads of the expert product, in whatever order they touch it. This eval callback runs after each MoE layer's
// router (the "ffn_moe_topk-<layer>" node; ggml_backend_sched_compute_splits calls it before the next nodes run),
// reads which experts the tokens chose, and reads those experts' byte ranges from the file with large parallel pread
// calls, skipping pages mincore reports resident. The expert products then find them in the page cache. The data is
// the file's own bytes and the routing is untouched, so the model's output cannot change.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

namespace turing {

struct ReadAheadStats {
    uint64_t calls = 0;            // router outputs seen
    uint64_t experts = 0;          // (layer, expert) pairs the routers chose, counted once per call
    uint64_t bytes_wanted = 0;     // their bytes in the file
    uint64_t bytes_read = 0;       // the bytes that were not resident and were read (queued, when not waiting)
    uint64_t bytes_missing = 0;    // the bytes that were not resident, read or not
    uint64_t active_calls = 0;     // router outputs whose missing experts were read
    double seconds = 0.0;          // time inside the callback
};

class ExpertReadAhead {
public:
    // wait: the products start after the reads (true), or the reads are queued and run beside the products (false),
    // so the products' own page faults and the large reads overlap; at most kMaxQueued reads wait in the queue.
    // adaptive: read only while the missing share of the chosen experts' bytes over the last `window` router outputs is
    // at least min_miss; below it the callback only looks (mincore). gpt-oss-120b on a Neoverse-V2: 4.4% missing with a
    // 40 GB cache, where reading cost decode speed (12.8 against 17.4 tok/s), 6.7% at 28 GB and 12.7% at 20 GB, where it
    // paid 3.8x and 4.1x (benchmarks/results/results_cpu_waves23_c4a.json).
    ExpertReadAhead(const std::string& path, int io_threads = 16, size_t chunk = size_t(4) << 20, bool check_resident = true,
                    bool wait = true, bool adaptive = true, int window = 64, double min_miss = 0.05)
        : chunk_(chunk), check_resident_(check_resident), wait_(wait), adaptive_(adaptive), window_(std::max(1, window)),
          min_miss_(min_miss), active_(!adaptive) {
        ggml_context* meta = nullptr;
        gguf_init_params gp{true, &meta};
        gguf_context* g = gguf_init_from_file(path.c_str(), gp);
        if (!g) throw std::runtime_error("expert read-ahead: cannot read " + path);
        const size_t data = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const std::string name = gguf_get_tensor_name(g, i);
            int layer = -1;
            char kind[32] = {0};
            // blk.<layer>.ffn_{gate,up,down,gate_up}_exps.weight: [.., .., n_expert], one expert contiguous after another
            if (std::sscanf(name.c_str(), "blk.%d.ffn_%31[a-z_]", &layer, kind) != 2) continue;
            const std::string k(kind);
            if (k.size() < 5 || k.compare(k.size() - 5, 5, "_exps") != 0 || name.find(".weight") == std::string::npos) continue;
            const ggml_tensor* t = ggml_get_tensor(meta, name.c_str());
            if (!t || t->ne[2] < 2) continue;
            const uint64_t size = gguf_get_tensor_size(g, i);
            layers_[layer].push_back({data + gguf_get_tensor_offset(g, i), size / uint64_t(t->ne[2])});
        }
        gguf_free(g);
        ggml_free(meta);
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("expert read-ahead: cannot open " + path);
        file_size_ = static_cast<uint64_t>(::lseek(fd_, 0, SEEK_END));
        map_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);  // only for mincore: no page is touched
        if (map_ == MAP_FAILED) map_ = nullptr;
        page_ = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
        for (int i = 0; i < std::max(1, io_threads); ++i) workers_.emplace_back([this] { work(); });
    }
    ~ExpertReadAhead() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            pending_ -= queue_.size();
            queue_.clear();  // queued read-ahead is only a hint: drop it
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
        if (map_) ::munmap(map_, file_size_);
        if (fd_ >= 0) ::close(fd_);
    }
    ExpertReadAhead(const ExpertReadAhead&) = delete;
    ExpertReadAhead& operator=(const ExpertReadAhead&) = delete;

    size_t moe_layers() const { return layers_.size(); }
    ReadAheadStats stats() const {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }

    // ggml_backend_sched_eval_callback: ask for the routers' outputs, read their experts when they are computed.
    static bool callback(ggml_tensor* t, bool ask, void* self) {
        auto* ra = static_cast<ExpertReadAhead*>(self);
        if (std::strncmp(t->name, "ffn_moe_topk-", 13) != 0) return ask ? false : true;
        if (ask) return true;
        ra->on_router(t);
        return true;  // keep computing
    }

private:
    struct ExpertTensor {
        uint64_t offset;      // in the file
        uint64_t per_expert;  // bytes of one expert
    };
    struct Job {
        uint64_t offset, len;
    };

    void on_router(const ggml_tensor* t) {
        const auto t0 = std::chrono::steady_clock::now();
        const int layer = std::atoi(t->name + 13);
        auto it = layers_.find(layer);
        if (it == layers_.end() || t->type != GGML_TYPE_I32 || (t->buffer && !ggml_backend_buffer_is_host(t->buffer)))
            return;
        std::set<int32_t> chosen;
        const int64_t used = t->ne[0], tokens = t->ne[1];
        for (int64_t j = 0; j < tokens; ++j) {
            const auto* row = reinterpret_cast<const int32_t*>(static_cast<const char*>(t->data) + j * t->nb[1]);
            for (int64_t e = 0; e < used; ++e) chosen.insert(row[e]);
        }
        std::vector<Job> jobs;
        uint64_t wanted = 0;
        for (const ExpertTensor& et : it->second) {
            for (int32_t e : chosen) {
                const uint64_t off = et.offset + uint64_t(e) * et.per_expert;
                wanted += et.per_expert;
                add_missing(off, et.per_expert, jobs);
            }
        }
        uint64_t missing = 0;
        for (const Job& j : jobs) missing += j.len;
        const bool active = active_;
        if (active && !jobs.empty()) run(jobs, wait_);
        if (adaptive_) {  // decide for the next window from this one's missing share
            win_wanted_ += wanted;
            win_missing_ += missing;
            if (++win_calls_ >= window_) {
                active_ = win_wanted_ > 0 && double(win_missing_) >= min_miss_ * double(win_wanted_);
                win_calls_ = 0, win_wanted_ = win_missing_ = 0;
            }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.calls += 1;
        stats_.experts += chosen.size();
        stats_.bytes_wanted += wanted;
        stats_.bytes_read += active ? missing : 0;
        stats_.bytes_missing += missing;
        stats_.active_calls += active ? 1 : 0;
        stats_.seconds += s;
    }

    // The ranges of [off, off + len) that are not resident, page aligned, in jobs of at most chunk_ bytes.
    void add_missing(uint64_t off, uint64_t len, std::vector<Job>& jobs) const {
        const uint64_t a = off / page_ * page_, b = std::min(file_size_, (off + len + page_ - 1) / page_ * page_);
        if (b <= a) return;
        const size_t pages = static_cast<size_t>((b - a) / page_);
        std::vector<
#ifdef __APPLE__
            char
#else
            unsigned char
#endif
            >
            vec(pages, 0);
        const bool known = check_resident_ && map_ && ::mincore(static_cast<char*>(map_) + a, b - a, vec.data()) == 0;
        size_t p = 0;
        while (p < pages) {
            if (known && (vec[p] & 1)) {
                ++p;
                continue;
            }
            size_t q = p + 1;  // at least one page per job, whatever chunk_ is
            while (q < pages && !(known && (vec[q] & 1)) && (q - p + 1) * page_ <= chunk_) ++q;
            jobs.push_back({a + p * page_, (q - p) * page_});
            p = q;
        }
    }

    // Read every job (in parallel on the workers) into scratch memory: the reads fill the page cache. wait: until done.
    void run(const std::vector<Job>& jobs, bool wait) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!wait) {  // reads still queued are for a layer whose products already ran: replace them
                pending_ -= queue_.size();
                queue_.clear();
            }
            const size_t n = wait ? jobs.size() : std::min(kMaxQueued, jobs.size());
            queue_.insert(queue_.end(), jobs.begin(), jobs.begin() + static_cast<long>(n));
            pending_ += n;
        }
        cv_.notify_all();
        if (!wait) return;
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
    }

    void work() {
        std::vector<char> buf(chunk_);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                j = queue_.back();
                queue_.pop_back();
            }
            uint64_t done = 0;
            while (done < j.len) {
                const ssize_t n = ::pread(fd_, buf.data(), static_cast<size_t>(std::min<uint64_t>(j.len - done, buf.size())),
                                          static_cast<off_t>(j.offset + done));
                if (n <= 0) break;  // a failed read only loses the read-ahead: the product still faults the pages in
                done += static_cast<uint64_t>(n);
            }
            std::lock_guard<std::mutex> lk(mu_);
            if (--pending_ == 0) done_cv_.notify_all();
        }
    }

    std::map<int, std::vector<ExpertTensor>> layers_;
    static constexpr size_t kMaxQueued = 4096;
    size_t chunk_;
    bool check_resident_;
    bool wait_;
    bool adaptive_;
    int window_;
    double min_miss_;
    bool active_;
    int win_calls_ = 0;
    uint64_t win_wanted_ = 0, win_missing_ = 0;
    int fd_ = -1;
    void* map_ = nullptr;
    uint64_t file_size_ = 0, page_ = 4096;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::vector<Job> queue_;
    size_t pending_ = 0;
    bool stop_ = false;
    mutable std::mutex stats_mu_;
    ReadAheadStats stats_;
};

}  // namespace turing
