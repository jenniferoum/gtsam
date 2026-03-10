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

/**
 * Right group action on the sensor-only block.
 * @param X group element.
 * @param sensor sensor-state block.
 * @return transformed sensor-state block.
 */
GTSAM_UNSTABLE_EXPORT VIOSensorState sensorStateGroupAction(
    const VIOGroup& X, const VIOSensorState& sensor);
/**
 * Right group action on full state.
 * @param X group element.
 * @param state full VIO state.
 * @return transformed state.
 */
GTSAM_UNSTABLE_EXPORT VIOState stateGroupAction(const VIOGroup& X,
                                                const VIOState& state);
/**
 * Right group action on vision measurements.
 * @param X group element.
 * @param measurement input measurement.
 * @return transformed measurement.
 */
GTSAM_UNSTABLE_EXPORT VisionMeasurement outputGroupAction(
    const VIOGroup& X, const VisionMeasurement& measurement);

/**
 * Continuous-time lift map from IMU velocity to VIOGroup tangent.
 * @param state current state.
 * @param velocity imu velocity input.
 * @return tangent lift in VIOGroup ordering.
 */
GTSAM_UNSTABLE_EXPORT Vector liftVelocity(const VIOState& state,
                                          const IMUVelocity& velocity);
/**
 * Discrete-time lift map from IMU velocity to VIOGroup increment.
 * @param state current state.
 * @param velocity imu velocity input.
 * @param dt integration step in seconds.
 * @return group increment.
 */
GTSAM_UNSTABLE_EXPORT VIOGroup liftVelocityDiscrete(const VIOState& state,
                                                    const IMUVelocity& velocity,
                                                    double dt);

/**
 * Integrate system dynamics forward by dt.
 * @param state current state.
 * @param velocity imu velocity input.
 * @param dt integration step in seconds.
 * @return propagated state.
 */
GTSAM_UNSTABLE_EXPORT VIOState integrateSystemFunction(
    const VIOState& state, const IMUVelocity& velocity, double dt);
/**
 * Generate ideal camera measurements from state.
 * @param state state to observe.
 * @param camera camera model.
 * @return synthetic measurement.
 */
GTSAM_UNSTABLE_EXPORT VisionMeasurement measureSystemState(
    const VIOState& state, const std::shared_ptr<const VIOCameraModel>& camera);

/**
 * Right action phi(xi, X) = stateGroupAction(X, xi).
 * H_xi is analytic blockwise; H_X is numerical for correctness.
 */
struct GTSAM_UNSTABLE_EXPORT VIOSymmetry
    : public GroupAction<VIOSymmetry, VIOGroup, VIOState> {
  static constexpr ActionType type = ActionType::Right;

  /**
   * Evaluate right action phi(xi, X).
   * @param xi state argument.
   * @param X group argument.
   * @param H_xi optional derivative wrt state.
   * @param H_X optional derivative wrt group.
   * @return transformed state.
   */
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

  /**
   * Evaluate right output action rho(y, X).
   * @param y measurement argument.
   * @param X group argument.
   * @param H_y optional derivative wrt measurement.
   * @param H_X optional derivative wrt group.
   * @return transformed measurement.
   */
  VisionMeasurement operator()(
      const VisionMeasurement& y, const VIOGroup& X,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_y = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X = {}) const;
};

}  // namespace gtsam
