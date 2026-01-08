/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file ForestTraversal.h
 * @brief Forest traversal helpers with optional TBB acceleration.
 *
 * @details
 * Provides top-down and bottom-up traversal helpers that either enqueue work
 * on an internal `PriorityScheduler` (for builds without TBB) or call the
 * `treeTraversal` parallel helpers when `GTSAM_USE_TBB` is enabled.
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

#include <gtsam/config.h>
#ifdef GTSAM_USE_TBB
#include <gtsam/base/treeTraversal-inst.h>
#include <gtsam/base/types.h>
#include <tbb/global_control.h>
#else
#include <gtsam/base/PriorityScheduler.h>
#endif

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
 * When TBB is present, the traversal delegates directly to
 * `treeTraversal::DepthFirstForestParallel` and
 * `treeTraversal::PostOrderForestParallel` with the configured parallel
 * thresholds. Otherwise it falls back to the priority-queued implementation
 * that mirrors `TaskMixin`.
 */
template <typename Forest, typename Node>
class ForestTraversal {
 public:
  /// Construct a helper with a fixed thread budget (used by TBB when enabled).
  explicit ForestTraversal(
      size_t numThreads = std::thread::hardware_concurrency())
      : threadCount_(numThreads == 0 ? 1 : numThreads)
#ifndef GTSAM_USE_TBB
        ,
        scheduler_(threadCount_)
#endif
  {
  }

  /// Run a top-down traversal (root first).
  template <typename Fn>
  void runTopDown(Fn fn, int parallelThreshold = 10) {
#ifdef GTSAM_USE_TBB
    runTopDownTreeTraversal(fn, parallelThreshold);
#else
    runTopDownPriorityScheduler(fn);
#endif
  }

  /// Run a bottom-up traversal (leaves first).
  template <typename Fn>
  void runBottomUp(Fn fn, int parallelThreshold = 10) {
#ifdef GTSAM_USE_TBB
    runBottomUpTreeTraversal(fn, parallelThreshold);
#else
    runBottomUpPriorityScheduler(fn);
#endif
  }

 private:
  size_t threadCount_;

#ifdef GTSAM_USE_TBB
  using SharedNode = std::shared_ptr<Node>;

  template <typename Fn>
  void runTopDownTreeTraversal(Fn fn, int parallelThreshold) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                static_cast<int>(threadCount_));
    TbbOpenMPMixedScope threadLimiter;

    struct VisitorPre {
      Fn* fn;
      int operator()(const SharedNode& node, int&) const {
        if (node) std::invoke(*fn, *node);
        return 0;
      }
    };

    int rootData = 0;
    VisitorPre visitor{&fn};
    auto visitorPost = [](const SharedNode&, int) {};
    treeTraversal::DepthFirstForestParallel(static_cast<Forest&>(*this),
                                            rootData, visitor, visitorPost,
                                            parallelThreshold);
  }

  template <typename Fn>
  void runBottomUpTreeTraversal(Fn fn, int parallelThreshold) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                static_cast<int>(threadCount_));
    TbbOpenMPMixedScope threadLimiter;

    struct VisitorPost {
      Fn* fn;
      void operator()(const SharedNode& node) const {
        if (node) std::invoke(*fn, *node);
      }
    };

    VisitorPost visitor{&fn};
    treeTraversal::PostOrderForestParallel(static_cast<Forest&>(*this), visitor,
                                           parallelThreshold);
  }
#else
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

  template <typename Fn>
  void runTopDownPriorityScheduler(Fn fn) {
    auto state = std::make_shared<TraversalState>();
    std::future<void> done = state->done.get_future();
    Forest& forest = static_cast<Forest&>(*this);
    const auto& roots = rootsOf(forest);
    if (roots.empty()) {
      state->done.set_value();
      return;
    }
    state->pending.fetch_add(1, std::memory_order_relaxed);
    for (const auto& root : roots) {
      topDownAsync(*root, 0, fn, state);
    }
    finishTraversal(state);
    done.get();
  }

  template <typename Fn>
  void runBottomUpPriorityScheduler(Fn fn) {
    auto state = std::make_shared<TraversalState>();
    std::future<void> done = state->done.get_future();
    Forest& forest = static_cast<Forest&>(*this);
    const auto& roots = rootsOf(forest);
    if (roots.empty()) {
      state->done.set_value();
      return;
    }
    state->pending.fetch_add(1, std::memory_order_relaxed);
    for (const auto& root : roots) {
      bottomUpAsync(*root, 0, fn, state, [] {});
    }
    finishTraversal(state);
    done.get();
  }

  template <typename Fn>
  void topDownAsync(Node& node, int depth, const Fn& fn,
                    const StatePtr& state) {
    auto task = [this, &node, depth, &fn, state]() {
      try {
        std::invoke(fn, node);
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

  int topDownPriority(int depth) const { return depth; }

  int bottomUpPriority(int depth) const { return -depth; }

#endif

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
