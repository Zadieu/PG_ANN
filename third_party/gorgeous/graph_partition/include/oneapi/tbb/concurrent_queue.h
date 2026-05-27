#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace oneapi::tbb {

template <typename T>
class concurrent_bounded_queue {
 public:
  concurrent_bounded_queue() = default;

  void push(const T &value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(value);
    }
    cv_.notify_one();
  }

  void pop(T &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    value = queue_.front();
    queue_.pop();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<T> empty;
    queue_.swap(empty);
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
};

}  // namespace oneapi::tbb
