/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2020, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file TrajectoryAlignerSim3.h
 * @author Akshay Krishnan
 * @date January 2026
 * @brief Aligning a trajectory of poses to a reference trajectory using a similarity transform.
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/UnaryMeasurement.h>

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace gtsam {


class GTSAM_EXPORT TrajectoryAlignerSim3 {
 public:
  using KeyPair = std::pair<Key, Key>;
  using PoseMeasurements = std::vector<UnaryMeasurement<Pose3>>;
  using ChildrenPoses = std::vector<PoseMeasurements>;

 private:

  // Parameters.
  ExpressionFactorGraph graph_;
  Values initial_;

 public:
  TrajectoryAlignerSim3(const PoseMeasurements &aTi, const ChildrenPoses &bTi_all, const std::vector<Similarity3> &aSb_all);


  Values solve() const;

};
}  // namespace gtsam
