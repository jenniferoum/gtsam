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
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/UnaryMeasurement.h>

#include <vector>

namespace gtsam {

/**
 * @brief Aligns Pose3 trajectories from multiple child coordinate frames to a
 * parent reference frame using Sim3 (similarity) transformations.
 *
 * This class solves an optimization problem to find the best Sim3 transforms
 * (rotation, translation, and scale) that align poses from one or more child
 * coordinate frames to a parent reference frame. The optimization jointly
 * refines both the parent frame poses and the child-to-parent transformations.
 *
 * The class takes as input:
 * - Parent frame poses (aTi): Pose3 measurements in the parent coordinate frame
 * - Child frame poses (bTi_all): Pose3 measurements in one or more child frames
 * - (Optional) Initial Sim3 estimates (aSb_all): Initial transforms from each
 *   child frame to the parent frame
 *
 * The output is a Values object containing:
 * - Optimized parent frame poses (with keys from the input aTi measurements)
 * - Optimized Sim3 transforms (with Symbol keys 'S' and index for each child)
 */
class GTSAM_EXPORT TrajectoryAlignerSim3 {
 public:
  using PoseMeasurements = std::vector<UnaryMeasurement<Pose3>>;
  using ChildrenPoses = std::vector<PoseMeasurements>;

 private:
  // Data members.
  ExpressionFactorGraph graph_;
  Values initial_;

 public:
  /**
   * @brief Constructs a trajectory aligner with the given measurements.
   * @param aTi Parent frame pose measurements (key-value pairs with noise)
   * @param bTi_all Vector of child frame pose measurements, one vector per child
   * @param aSb_all Initial Sim3 estimates transforming from each child to parent.
   *                If empty, initial estimates are computed automatically.
   */
  TrajectoryAlignerSim3(const PoseMeasurements &aTi, const ChildrenPoses &bTi_all, const std::vector<Similarity3> &aSb_all);

  /// Solves the optimization problem and returns optimized poses and transforms.
  Values solve() const;

};
}  // namespace gtsam
