/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testPriorityScheduler.cpp
 * @brief Unit tests for PriorityScheduler scheduling behavior.
 * @author Frank Dellaert
 * @date May, 2025
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/PriorityScheduler.h>

using namespace gtsam;

/* ************************************************************************* */
// Compose child return values inside a parent task
TEST(PriorityScheduler, ReturnValueComposition) {
  // Use multiple workers so parent tasks can wait without stalling the pool.
  PriorityScheduler<size_t> scheduler(2);

  // Schedule leaves first with higher priority (lower numeric value).
  auto left = scheduler.schedule(0, [] { return size_t{3}; }).share();
  auto right = scheduler.schedule(0, [] { return size_t{4}; }).share();

  // Parent task waits on the child futures and combines their values.
  auto parent = scheduler.schedule(
      10, [left, right]() mutable { return left.get() + right.get(); });

  const size_t expectedSum = 7;
  EXPECT_LONGS_EQUAL(expectedSum, parent.get());
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
