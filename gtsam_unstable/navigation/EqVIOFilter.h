/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOFilter.h
 * @brief Standalone equivariant VIO filter.
 * @author Rohan Bansal
 */

#pragma once

#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/navigation/EqVIOSymmetry.h>
#include <gtsam_unstable/dllexport.h>

#include <memory>
#include <vector>

namespace gtsam {
namespace eqvio {

/**
 * @brief Runtime parameters for the standalone EqVIO filter.
 *
 * These parameters mirror the process/measurement tuning knobs used by the
 * EqVIO runtime implementation. All variances are per-axis scalar multipliers
 * applied to identity blocks in the corresponding covariance matrices.
 */
struct EqVIOFilterParams {
  /// Initial landmark depth used when a new feature is first triangulated from bearing.
  double initialPointDepth = 10.0;
  /// Initial 3x3 covariance scalar assigned to newly inserted landmarks.
  double initialPointVariance = 1.0;
  /// Default pixel-space measurement noise variance used when `correct()` is called with empty `R`.
  double measurementNoiseVariance = 1e-4;
  /// Absolute reprojection residual threshold for outlier rejection.
  double outlierThresholdAbs = 1e8;
  /// Mahalanobis-style reprojection residual threshold for outlier rejection.
  double outlierThresholdProb = 1e8;
  /// Fraction of features to keep after ranking potential outliers (in [0, 1]).
  double featureRetention = 0.3;
  double biasOmegaProcessVariance = 0.001;
  double biasAccelProcessVariance = 0.001;
  double attitudeProcessVariance = 0.001;
  double positionProcessVariance = 0.001;
  double velocityProcessVariance = 0.001;
  double cameraAttitudeProcessVariance = 0.001;
  double cameraPositionProcessVariance = 0.001;
  double pointProcessVariance = 0.001;
  /// IMU driving noise covariance in stacked order `[gyr, acc, gyr_bias_walk, acc_bias_walk]`.
  Eigen::Matrix<double, IMUInput::CompDim, IMUInput::CompDim> inputNoise =
      Eigen::Matrix<double, IMUInput::CompDim, IMUInput::CompDim>::Identity() *
      1e-3;
};

/**
 * @brief Standalone EqVIO filter built on top of `EquivariantFilter`.
 *
 * Prediction is split into covariance propagation (single averaged IMU segment)
 * and state propagation (piecewise IMU holds), preserving the original EqVIO
 * replay semantics while still using the base equivariant filter machinery.
 */
class GTSAM_UNSTABLE_EXPORT EqVIOFilter
    : public EquivariantFilter<State, Symmetry> {
 public:
  using Base = EquivariantFilter<State, Symmetry>;

 private:
  EqVIOFilterParams params_;
  bool initialized_ = false;

 public:
  EqVIOFilter();
  /// Construct with explicit parameter bundle and default identity initial state.
  explicit EqVIOFilter(const EqVIOFilterParams& params);
  /// Construct with explicit initial reference state, covariance, and parameters.
  EqVIOFilter(const State& xi_ref, const Matrix& Sigma,
              const EqVIOFilterParams& params);

  /**
   * @brief Initialize filter attitude from gravity direction in an IMU sample.
   *
   * The measured acceleration vector is treated as gravity (small-motion
   * assumption) and aligned with world +Z.
   */
  void initializeFromIMU(const IMUInput& imu);
  
  /**
   * @brief Propagate filter state across a sequence of IMU hold intervals.
   *
   * Each hold interval `(imuInputs[i], dts[i])` is propagated via the automatic
   * base-class `predict(...)` path, so mean and covariance advance together.
   *
   * @param imuInputs IMU samples defining zero-order holds.
   * @param dts Hold durations (seconds), one per sample.
   * @throws std::invalid_argument if input sizes differ.
   */
  void predict(const std::vector<IMUInput>& imuInputs,
                 const std::vector<double>& dts);
                 
  /**
   * @brief Apply one visual correction step with dynamic landmark management.
   *
   * The method removes stale landmarks, rejects outliers, inserts new
   * landmarks, performs update, and prunes numerically invalid landmarks.
   *
   * @param measurement Observed image points keyed by landmark id.
   * @param camera Camera model used for projection/linearization.
   * @param R Optional measurement covariance; if empty/wrong-sized, defaults to
   *          `measurementNoiseVariance * I`.
   */
  void update(const VisionMeasurement& measurement,
               const std::shared_ptr<const CameraModel>& camera,
               const Matrix& R = Matrix());

  /// True after IMU-based initialization.
  bool isInitialized() const { return initialized_; }

 private:
  /// Allocate identity covariance with dimension `SensorState::CompDim + 3*nLandmarks`.
  static Matrix defaultCovariance(size_t nLandmarks);

  /// Build process noise matrix for current state dimension.
  Matrix stateProcessNoise(size_t nLandmarks) const;

  /// Run innovation update on currently matched measurements.
  void innovationUpdate(const VisionMeasurement& measurement,
                        const std::shared_ptr<const CameraModel>& camera,
                        const Matrix& outputGainMatrix);

  // LANDMARK HELPERS

  /// Add unseen landmarks from current measurement and expand group/covariance.
  void addNewLandmarks(const VisionMeasurement& measurement,
                       const std::shared_ptr<const CameraModel>& camera);
  /// Remove landmark by contiguous index in `cameraLandmarks`.
  void removeLandmarkByIndex(int idx);
  /// Remove landmark by key id.
  void removeLandmarkById(Key id);
  /// Drop landmarks not present in the current measurement id set.
  void removeOldLandmarks(const std::vector<Key>& measurementIds);
  /// Detect and remove outliers according to absolute/probabilistic thresholds.
  void removeOutliers(VisionMeasurement& measurement,
                      const std::shared_ptr<const CameraModel>& camera);
  /// Remove landmarks whose scale component leaves a numerically safe range.
  void removeInvalidLandmarks();
  /// Return 3x3 covariance block for a specific landmark id.
  Matrix3 getLandmarkCovById(Key id) const;
  /// Return induced 2x2 output covariance for a specific landmark id.
  Matrix2 outputCovarianceById(
      Key id, const std::shared_ptr<const CameraModel>& camera) const;

};

}  // namespace eqvio
}  // namespace gtsam
