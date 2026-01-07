/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file PriorityScheduler.h
 * @brief Priority-based task scheduler.
 *
 * @details
 * This header defines a small, thread-based scheduler that executes tasks in
 * priority order. A lower numeric priority is executed before a higher numeric
 * priority (min-heap behavior). Tasks return a value of type `Y`, and callers
 * can wait on results via `std::future<Y>`.
 *
 * @par Example
 * @code
 * gtsam::PriorityScheduler<int> scheduler(4);
 * auto future = scheduler.schedule(0, [] { return 42; });
 * int value = future.get();
 * @endcode
 *
 * @author Frank Dellaert
 * @date May, 2025
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>  // For std::runtime_error
#include <thread>
#include <type_traits>  // For std::is_void_v
#include <utility>      // For std::forward
#include <vector>

namespace gtsam {

/**
 * @brief Thread pool scheduler that prioritizes tasks by numeric priority.
 *
 * @details
 * - Lower numeric values are executed first.
 * - Tasks are executed by worker threads created at construction.
 * - `schedule` returns a `std::future<Y>` for the task result.
 * - Per-thread queues reduce contention; workers steal from peers when idle.
 * - External submissions are round-robin distributed across worker queues.
 * - A condition variable parks workers when no work is available.
 *
 * @tparam Y Result type returned by tasks. Use `void` for no return value.
 */
template <typename Y>
class PriorityScheduler {
  struct Task {
    int priority;
    std::function<Y()> job;
    std::promise<Y> promise;

    Task(int p, std::function<Y()> j, std::promise<Y> p_out)
        : priority(p), job(std::move(j)), promise(std::move(p_out)) {}
  };

  using TaskPtr = std::shared_ptr<Task>;

  struct Compare {
    bool operator()(const TaskPtr& a, const TaskPtr& b) const {
      return a->priority > b->priority;
    }
  };

  struct WorkerQueue {
    std::mutex mutex;
    std::priority_queue<TaskPtr, std::vector<TaskPtr>, Compare> queue;
  };

  std::vector<std::unique_ptr<WorkerQueue>> queues_;  // Per-worker queues.
  std::atomic<size_t> queuedTasks_{0};  // Tracks queued work for wakeups.
  std::vector<std::thread> workers_;
  mutable std::mutex waitMutex_;  // Dedicated mutex for condition_variable.
  std::condition_variable condition_;
  std::atomic<bool> stop_{false};
  std::atomic<int> activeTasks_{0};  // In-flight tasks (queued + running).
  inline static thread_local PriorityScheduler<Y>* currentScheduler_ = nullptr;
  inline static thread_local int workerIndex_ = -1;
  std::atomic<size_t> nextWorker_{0};  // Round-robin distributor.

  /**
   * @brief Worker loop: wait for tasks, run them, and fulfill promises.
   *
   * Uses a condition variable to avoid spinning while the queue is empty.
   * Stops once `stop_` is set and no queued work remains.
   */
  /// Worker loop that executes tasks until shutdown.
  void worker_thread(size_t index) {
    currentScheduler_ = this;
    workerIndex_ = static_cast<int>(index);
    while (true) {
      TaskPtr task;
      if (!tryPopTask(index, task)) {
        std::unique_lock<std::mutex> lock(waitMutex_);
        condition_.wait(lock, [this] {
          return stop_.load(std::memory_order_acquire) ||
                 queuedTasks_.load(std::memory_order_acquire) > 0;
        });
        if (stop_.load(std::memory_order_acquire) &&
            queuedTasks_.load(std::memory_order_acquire) == 0) {
          currentScheduler_ = nullptr;
          workerIndex_ = -1;
          return;
        }
        continue;
      }

      try {
        // Execute the task and set the promise value:
        if constexpr (std::is_void_v<Y>) {
          task->job();
          task->promise.set_value();
        } else {
          task->promise.set_value(task->job());
        }
      } catch (...) {
        // On exception, set the exception on the promise:
        try {
          task->promise.set_exception(std::current_exception());
        } catch (...) { /* ignore */
        }
      }

      // Decrement active task count and notify waiters.
      activeTasks_.fetch_sub(1, std::memory_order_release);
      condition_.notify_all();
    }
  }

  /// Attempt to get a task from local or stolen queues.
  bool tryPopTask(size_t index, TaskPtr& task) {
    // Prefer local queue, then steal to keep workers busy.
    if (tryPopLocal(index, task)) return true;
    if (trySteal(index, task)) return true;
    return false;
  }

  /// Try to pop a task from the current worker queue.
  bool tryPopLocal(size_t index, TaskPtr& task) {
    WorkerQueue& queue = *queues_[index];
    std::lock_guard<std::mutex> lock(queue.mutex);
    if (queue.queue.empty()) return false;
    task = queue.queue.top();
    queue.queue.pop();
    queuedTasks_.fetch_sub(1, std::memory_order_release);
    return true;
  }

  /// Try to steal a task from another worker queue.
  bool trySteal(size_t index, TaskPtr& task) {
    const size_t workerCount = queues_.size();
    if (workerCount <= 1) return false;
    // Steal from other workers if local queue is empty.
    for (size_t offset = 1; offset < workerCount; ++offset) {
      size_t target = (index + offset) % workerCount;
      WorkerQueue& queue = *queues_[target];
      std::unique_lock<std::mutex> lock(queue.mutex, std::try_to_lock);
      if (!lock.owns_lock() || queue.queue.empty()) continue;
      task = queue.queue.top();
      queue.queue.pop();
      queuedTasks_.fetch_sub(1, std::memory_order_release);
      return true;
    }
    return false;
  }

  /// Return true if called on a scheduler worker thread.
  bool isWorkerThread() const { return currentScheduler_ == this; }

 public:
  /**
   * @brief Construct a scheduler with a fixed number of worker threads.
   *
   * @param numThreads Number of worker threads to create. If zero, a single
   * thread is created.
   */
  /// Construct a scheduler with a fixed worker count.
  PriorityScheduler(size_t numThreads = std::thread::hardware_concurrency()) {
    if (numThreads == 0) numThreads = 1;
    queues_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
      queues_.push_back(std::make_unique<WorkerQueue>());
    }
    for (size_t i = 0; i < numThreads; ++i) {
      workers_.emplace_back(&PriorityScheduler::worker_thread, this, i);
    }
  }

  /**
   * @brief Wait for all tasks to finish, then stop worker threads.
   *
   * @note The destructor calls `waitForAllTasks` before stopping workers.
   */
  /// Wait for tasks and join worker threads on destruction.
  ~PriorityScheduler() {
    waitForAllTasks();
    stop_.store(true, std::memory_order_release);
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  /**
   * @brief Enqueue a task for execution.
   *
   * @param priority Lower values execute before higher values.
   * @param job Callable returning a `Y`.
   * @return `std::future<Y>` associated with the task.
   */
  /// Enqueue a task for execution and return its future.
  std::future<Y> schedule(int priority, std::function<Y()> job) {
    if (stop_.load(std::memory_order_acquire)) {
      std::promise<Y> err_promise;
      err_promise.set_exception(std::make_exception_ptr(
          std::runtime_error("Scheduler is stopping or stopped.")));
      return err_promise.get_future();
    }

    std::promise<Y> promise;
    std::future<Y> future = promise.get_future();
    auto task =
        std::make_shared<Task>(priority, std::move(job), std::move(promise));

    if (isWorkerThread()) {
      // Worker submissions go to the local queue for cache locality.
      WorkerQueue& queue = *queues_[static_cast<size_t>(workerIndex_)];
      {
        std::lock_guard<std::mutex> lock(queue.mutex);
        queue.queue.push(task);
      }
    } else {
      // External submissions are distributed across worker queues.
      const size_t target =
          nextWorker_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
      WorkerQueue& queue = *queues_[target];
      std::lock_guard<std::mutex> lock(queue.mutex);
      queue.queue.push(task);
    }
    queuedTasks_.fetch_add(1, std::memory_order_release);
    // Track in-flight tasks for waiters.
    activeTasks_.fetch_add(1, std::memory_order_release);
    condition_.notify_one();
    return future;
  }

  /**
   * @brief Schedule or run inline when called from a worker thread.
   *
   * Used to fuse continuations without re-entering the queues.
   */
  template <typename T = Y, typename = std::enable_if_t<std::is_void_v<T>>>
  /// Run inline on workers, otherwise enqueue normally.
  void scheduleOrRunInline(int priority, std::function<void()> job) {
    if (stop_.load(std::memory_order_relaxed)) return;
    if (!isWorkerThread()) {
      schedule(priority, std::move(job));
      return;
    }

    // Inline execution fuses continuations and avoids re-queueing.
    activeTasks_.fetch_add(1, std::memory_order_release);
    try {
      job();
    } catch (...) { /* ignore */
    }
    activeTasks_.fetch_sub(1, std::memory_order_release);
    condition_.notify_all();
  }

  /**
   * @brief Block until all queued and active tasks complete.
   *
   * @note If the scheduler is stopping, this returns early.
   */
  /// Block until all queued and active tasks complete.
  void waitForAllTasks() {
    std::unique_lock<std::mutex> lock(waitMutex_);
    condition_.wait(lock, [this] {
      return stop_.load(std::memory_order_acquire) ||
             (activeTasks_.load(std::memory_order_acquire) == 0 &&
              queuedTasks_.load(std::memory_order_acquire) == 0);
    });
  }
};

}  // namespace gtsam
