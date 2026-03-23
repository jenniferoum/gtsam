/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOSymmetry.h
 * @brief EqVIO symmetry actions and lift helpers.
 * @author Rohan Bansal
 */

#pragma once

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam_unstable/navigation/EqVIOCommon.h>
#include <gtsam_unstable/navigation/EqVIOState.h>
#include <gtsam_unstable/dllexport.h>

namespace gtsam {
namespace eqvio {

/// Right group action on the sensor-only block.
GTSAM_UNSTABLE_EXPORT SensorState sensorStateGroupAction(
    const VioGroup& X, const SensorState& sensor);
/// Right group action on full state.
GTSAM_UNSTABLE_EXPORT State stateGroupAction(const VioGroup& X,
                                                const State& state);

/// Discrete-time lift map from IMU velocity to VioGroup increment.
GTSAM_UNSTABLE_EXPORT VioGroup liftVelocityDiscrete(const State& state,
                                                    const IMUInput& velocity,
                                                    double dt);

/// Generate ideal camera measurements from state.
GTSAM_UNSTABLE_EXPORT VisionMeasurement measureSystemState(
    const State& state, const std::shared_ptr<const CameraModel>& camera);

/**
 * @brief EqF linearization blocks used by EqVIO.
 *
 * EqVIO uses an inverse-depth landmark error chart to preserve the
 * equivariant output linearization while remaining numerically stable for
 * distant points. Other coordinate systems (Euclid, Polar) are not supported
 * in this implementation yet. See the EqVIO paper (van Goor and Mahony,
 * arXiv:2205.01980v3).
 */
GTSAM_UNSTABLE_EXPORT Matrix EqFStateMatrixA(
    const VioGroup& X, const State& xi0, const IMUInput& imuVel);
GTSAM_UNSTABLE_EXPORT Matrix EqFInputMatrixB(
    const VioGroup& X, const State& xi0);
GTSAM_UNSTABLE_EXPORT Matrix23 EqFoutputMatrixCiStar(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera, const Point2& y);
GTSAM_UNSTABLE_EXPORT Matrix23 EqFoutputMatrixCi(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera);
GTSAM_UNSTABLE_EXPORT Matrix EqFoutputMatrixC(
    const State& xi0, const VioGroup& X, const VisionMeasurement& y,
    const std::shared_ptr<const CameraModel>& camera,
    bool useEquivariance = true);
GTSAM_UNSTABLE_EXPORT Vector liftInnovation(const Vector& totalInnovation,
                                            const State& xi0);

/// Right action phi(xi, X) = stateGroupAction(X, xi).
struct GTSAM_UNSTABLE_EXPORT Symmetry
    : public GroupAction<Symmetry, VioGroup, State> {
  static constexpr ActionType type = ActionType::Right;

  /// Evaluate right action phi(xi, X).
  State operator()(const State& xi, const VioGroup& X,
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_xi = {},
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X = {})
      const;
};

}  // namespace eqvio
}  // namespace gtsam
