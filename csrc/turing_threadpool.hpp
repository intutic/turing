#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <algorithm>

namespace turing {

/**
 * Persistent High-Performance ThreadPool.
 * Adapted from High-Performance HPC Suite (lr_threadpool.cc).
 * Reuses worker threads across epochs to eliminate OS thread spawning overhead.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop(false) {
        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    size_t num_workers() const {
        return workers.size();
    }

    /**
     * Parallel For with Ceiling-Division Chunking and Thread-Local Buffers.
     */
    void parallel_for(size_t start, size_t end, const std::function<void(size_t, size_t, size_t)>& func) {
        size_t total = end - start;
        if (total == 0) return;

        size_t nw = std::min(workers.size(), total);
        size_t chunk = (total + nw - 1) / nw;

        std::vector<std::future<void>> futures;
        for (size_t t = 0; t < nw; ++t) {
            size_t c_start = start + t * chunk;
            size_t c_end = std::min(c_start + chunk, end);
            if (c_start >= c_end) continue;

            futures.push_back(enqueue([&func, t, c_start, c_end]() {
                func(t, c_start, c_end);
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Global singleton instance for Turing Engine CPU workers
inline ThreadPool& get_global_threadpool() {
    static ThreadPool global_pool(std::max(1u, std::thread::hardware_concurrency()));
    return global_pool;
}

} // namespace turing
