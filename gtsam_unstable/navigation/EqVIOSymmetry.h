/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOSymmetry.h
 * @brief   VIO symmetry actions and lift helpers
 */

#pragma once

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam_unstable/navigation/EqVIOCommon.h>
#include <gtsam_unstable/navigation/EqVIOGroup.h>
#include <gtsam_unstable/navigation/EqVIOState.h>
#include <gtsam_unstable/dllexport.h>

namespace gtsam {

GTSAM_UNSTABLE_EXPORT VIOSensorState sensorStateGroupAction(
    const VIOGroup& X, const VIOSensorState& sensor);
GTSAM_UNSTABLE_EXPORT VIOState stateGroupAction(const VIOGroup& X,
                                                const VIOState& state);
GTSAM_UNSTABLE_EXPORT VisionMeasurement outputGroupAction(
    const VIOGroup& X, const VisionMeasurement& measurement);

GTSAM_UNSTABLE_EXPORT Vector liftVelocity(const VIOState& state,
                                          const IMUVelocity& velocity);
GTSAM_UNSTABLE_EXPORT VIOGroup liftVelocityDiscrete(const VIOState& state,
                                                    const IMUVelocity& velocity,
                                                    double dt);

GTSAM_UNSTABLE_EXPORT VIOState integrateSystemFunction(
    const VIOState& state, const IMUVelocity& velocity, double dt);
GTSAM_UNSTABLE_EXPORT VisionMeasurement measureSystemState(
    const VIOState& state, const std::shared_ptr<const VIOCameraModel>& camera);

/**
 * Right action phi(xi, X) = stateGroupAction(X, xi).
 * H_xi is analytic blockwise; H_X is numerical for correctness.
 */
struct GTSAM_UNSTABLE_EXPORT VIOSymmetry
    : public GroupAction<VIOSymmetry, VIOGroup, VIOState> {
  static constexpr ActionType type = ActionType::Right;

  VIOState operator()(const VIOState& xi, const VIOGroup& X,
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_xi = {},
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X = {})
      const;
};

/**
 * Right action rho(y, X) = outputGroupAction(X, y).
 * Jacobians are computed numerically when requested.
 */
struct GTSAM_UNSTABLE_EXPORT VIOOutputSymmetry
    : public GroupAction<VIOOutputSymmetry, VIOGroup, VisionMeasurement> {
  static constexpr ActionType type = ActionType::Right;

  VisionMeasurement operator()(
      const VisionMeasurement& y, const VIOGroup& X,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_y = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X = {}) const;
};

}  // namespace gtsam
