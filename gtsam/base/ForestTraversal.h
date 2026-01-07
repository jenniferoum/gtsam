/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file ForestTraversal.h
 * @brief Priority-based forest traversal helpers.
 *
 * @details
 * This header defines a mixin that provides depth-based top-down and bottom-up
 * traversals over a forest. Priorities are derived from recursion depth and
 * executed via an internal PriorityScheduler.
 *
 * @note `Forest::roots()` or `Forest::roots` must return a range of
 * pointer-like `Node` roots.
 * @note `Node::children()` or `Node::children` must return a range of
 * pointer-like `Node` children.
 *
 * @author Frank Dellaert
 * @date May, 2025
 */

#pragma once

#include <gtsam/base/PriorityScheduler.h>

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

namespace gtsam {

/**
 * @brief Mixin that provides depth-based top-down or bottom-up traversal
 * helpers on a forest owner.
 *
 * @details
 * This helper hides explicit priority management by assigning priorities based
 * on recursion depth. It is intended for method-style use in recursive solvers.
 * Tasks are scheduled on the provided scheduler and synchronized with
 * continuations to avoid blocking worker threads. Bottom-up continuations can
 * run inline when called from a worker to reduce queue traffic.
 *
 * @note The traversal helpers operate on `PriorityScheduler<void>` and rely on
 * node-local state for any computed results.
 */
template <typename Forest, typename Node>
class ForestTraversal {
 public:
  /// Run a top-down traversal (root first).
  template <typename Fn>
  void runTopDown(Fn fn) {
    // Run a top-down traversal with continuations.
    auto state = std::make_shared<TraversalState>();
    std::future<void> done = state->done.get_future();
    Forest& forest = static_cast<Forest&>(*this);
    const auto& roots = rootsOf(forest);
    if (roots.empty()) {
      state->done.set_value();
      return;
    }
    // Hold completion until all roots are scheduled.
    state->pending.fetch_add(1, std::memory_order_relaxed);
    for (const auto& root : roots) {
      topDownAsync(*root, 0, fn, state);
    }
    finishTraversal(state);
    done.get();
  }

  /// Run a bottom-up traversal (leaves first).
  template <typename Fn>
  void runBottomUp(Fn fn) {
    // Run a bottom-up traversal with continuations.
    auto state = std::make_shared<TraversalState>();
    std::future<void> done = state->done.get_future();
    Forest& forest = static_cast<Forest&>(*this);
    const auto& roots = rootsOf(forest);
    if (roots.empty()) {
      state->done.set_value();
      return;
    }
    // Hold completion until all roots are scheduled.
    state->pending.fetch_add(1, std::memory_order_relaxed);
    for (const auto& root : roots) {
      bottomUpAsync(*root, 0, fn, state, [] {});
    }
    finishTraversal(state);
    done.get();
  }

 protected:
  /// Construct the mixin with a fixed-size scheduler.
  explicit ForestTraversal(
      size_t numThreads = std::thread::hardware_concurrency())
      : scheduler_(numThreads) {}

  /// Priority for top-down traversal (root first).
  int topDownPriority(int depth) const { return depth; }

  /// Priority for bottom-up traversal (leaves first).
  int bottomUpPriority(int depth) const { return -depth; }

 private:
  PriorityScheduler<void> scheduler_;

  /// Shared traversal state across scheduled tasks.
  struct TraversalState {
    std::atomic<int> pending{0};
    std::atomic_flag exceptionClaim = ATOMIC_FLAG_INIT;
    std::atomic<bool> hasException{false};
    std::exception_ptr exception;
    std::promise<void> done;
  };
  using StatePtr = std::shared_ptr<TraversalState>;

  /// Schedule top-down work for a node and its children.
  template <typename Fn>
  void topDownAsync(Node& node, int depth, const Fn& fn,
                    const StatePtr& state) {
    auto task = [this, &node, depth, &fn, state]() {
      try {
        std::invoke(fn, node);
        // Children are scheduled after the node in top-down order.
        auto&& children = childrenOf(node);
        for (const auto& child : children) {
          topDownAsync(*child, depth + 1, fn, state);
        }
        finishTraversal(state);
      } catch (...) {
        recordException(state, std::current_exception());
        finishTraversal(state);
        return;
      }
    };

    scheduleTask(topDownPriority(depth), state, std::move(task));
  }

  /// Schedule children first, then the parent node once all children finish.
  template <typename Fn>
  void bottomUpAsync(Node& node, int depth, const Fn& fn, const StatePtr& state,
                     const std::function<void()>& onDone) {
    auto&& children = childrenOf(node);
    if (children.empty()) {
      scheduleBottomUpNode(node, depth, fn, state, onDone);
      return;
    }

    auto remaining =
        std::make_shared<std::atomic<int>>(static_cast<int>(children.size()));
    std::function<void()> childDone = [this, &node, depth, &fn, state, onDone,
                                       remaining]() {
      if (remaining->fetch_sub(1, std::memory_order_relaxed) == 1) {
        // If a child failed, skip this node and just unwind.
        if (state->hasException.load(std::memory_order_acquire)) {
          onDone();
          return;
        }
        scheduleBottomUpNode(node, depth, fn, state, onDone);
      }
    };

    for (const auto& child : children) {
      bottomUpAsync(*child, depth + 1, fn, state, childDone);
    }
  }

  /// Schedule bottom-up work for a node after its children finish.
  template <typename Fn>
  void scheduleBottomUpNode(Node& node, int depth, const Fn& fn,
                            const StatePtr& state,
                            const std::function<void()>& onDone) {
    auto task = [this, &node, &fn, state, onDone]() {
      try {
        std::invoke(fn, node);
        onDone();
        finishTraversal(state);
      } catch (...) {
        recordException(state, std::current_exception());
        onDone();
        finishTraversal(state);
        return;
      }
    };

    // Each scheduled task increments the pending counter.
    state->pending.fetch_add(1, std::memory_order_relaxed);

    // Schedule a continuation or run it inline if already on a worker thread.
    scheduler_.scheduleOrRunInline(bottomUpPriority(depth),
                                   std::function<void()>(std::move(task)));
  }

  /// Schedule a task and increment the pending counter.
  template <typename F>
  void scheduleTask(int priority, const StatePtr& state, F&& job) {
    // Each scheduled task increments the pending counter.
    state->pending.fetch_add(1, std::memory_order_relaxed);
    scheduler_.schedule(priority, std::function<void()>(std::forward<F>(job)));
  }

  /// Record the first exception and keep the traversal draining.
  void recordException(const StatePtr& state, std::exception_ptr exception) {
    // Record the first exception and let the traversal finish.
    if (!state->exceptionClaim.test_and_set(std::memory_order_acq_rel)) {
      state->exception = exception;
      state->hasException.store(true, std::memory_order_release);
    }
  }

  /// Resolve traversal completion once pending reaches zero.
  void finishTraversal(const StatePtr& state) {
    if (state->pending.fetch_sub(1, std::memory_order_relaxed) == 1) {
      if (state->hasException.load(std::memory_order_acquire)) {
        try {
          state->done.set_exception(state->exception);
        } catch (...) { /* ignore */
        }
      } else {
        try {
          state->done.set_value();
        } catch (...) { /* ignore */
        }
      }
    }
  }

  template <typename T, typename = void>
  struct HasChildrenMethod : std::false_type {};

  template <typename T>
  struct HasChildrenMethod<T,
                           std::void_t<decltype(std::declval<T&>().children())>>
      : std::true_type {};

  template <typename T, typename = void>
  struct HasRootsMethod : std::false_type {};

  template <typename T>
  struct HasRootsMethod<T, std::void_t<decltype(std::declval<T&>().roots())>>
      : std::true_type {};

  /// Return node children via method or field.
  template <typename T>
  static auto& childrenOf(T& node) {
    if constexpr (HasChildrenMethod<T>::value) {
      return node.children();
    } else {
      return node.children;
    }
  }

  /// Return forest roots via method or field.
  template <typename T>
  static auto& rootsOf(T& forest) {
    if constexpr (HasRootsMethod<T>::value) {
      return forest.roots();
    } else {
      return forest.roots;
    }
  }
};

}  // namespace gtsam
