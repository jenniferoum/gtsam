/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 *
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testLeftLinearEKF.cpp
 * @brief Unit tests for the LeftLinearEKF class
 * @date August, 2025
 * @authors Frank Dellaert
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/navigation/LeftLinearEKF.h>

#include <iostream>

using namespace std;
using namespace gtsam;

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}