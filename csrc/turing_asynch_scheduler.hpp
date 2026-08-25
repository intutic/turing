#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <thread>

namespace turing {

/**
 * Native C++20 Asynchronous Dynamic Master-Worker Task Scheduler (Asynchronous Master-Worker Engine).
 * Non-blocking task queue eliminating synchronization barriers across heterogeneous execution nodes.
 */
class AsynchTaskScheduler {
public:
    explicit AsynchTaskScheduler(size_t num_workers = 4)
        : stop_(false), active_tasks_(0) {
        for (size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->cv_.wait(lock, [this]() {
                            return this->stop_ || !this->tasks_.empty();
                        });

                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }

                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                        this->active_tasks_++;
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->active_tasks_--;
                        this->done_cv_.notify_all();
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto schedule_task(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("AsynchTaskScheduler: cannot schedule on stopped scheduler");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        done_cv_.wait(lock, [this]() {
            return this->tasks_.empty() && this->active_tasks_ == 0;
        });
    }

    ~AsynchTaskScheduler() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::atomic<bool> stop_;
    std::atomic<int> active_tasks_;
};

inline void asynch_schedule_token_slices_cpp(
    const float* __restrict__ input_tokens, // [NumTokens, Dim]
    float* __restrict__ output_tokens,      // [NumTokens, Dim]
    int num_tokens,
    int dim,
    float scale,
    int num_workers = 4
) {
    AsynchTaskScheduler scheduler(num_workers);
    int chunk_size = (num_tokens + num_workers - 1) / num_workers;

    std::vector<std::future<void>> futures;
    for (int w = 0; w < num_workers; ++w) {
        int t_start = w * chunk_size;
        int t_end = std::min(num_tokens, t_start + chunk_size);
        if (t_start >= num_tokens) break;

        futures.push_back(scheduler.schedule_task([=]() {
            for (int t = t_start; t < t_end; ++t) {
                const float* in_ptr = input_tokens + (t * dim);
                float* out_ptr = output_tokens + (t * dim);
                for (int d = 0; d < dim; ++d) {
                    out_ptr[d] = in_ptr[d] * scale;
                }
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

} // namespace turing
