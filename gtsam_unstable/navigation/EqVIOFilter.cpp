/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOFilter.cpp
 * @brief Standalone equivariant VIO filter.
 * @author Rohan Bansal
 */

#include <gtsam_unstable/navigation/EqVIOFilter.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace gtsam {
namespace eqvio {

/**
 * @brief Construct filter with given parameter bundle and identity initial
 * state.
 *
 * The base `EquivariantFilter` storage is initialized directly.
 */
EqVIOFilter::EqVIOFilter(const EqVIOFilterParams& params)
    : Base(State(), defaultCovariance(0), makeVioGroupIdentity()),
      params_(params) {
  State xi0;
  xi0.sensor.inputBias = Bias::Identity();
  xi0.sensor.pose = Pose3::Identity();
  xi0.sensor.velocity.setZero();
  xi0.sensor.cameraOffset = Pose3::Identity();
  resetReferenceAndGroup(xi0, defaultCovariance(0), makeVioGroupIdentity());
  setLandmarkKeys({});
}

/**
 * @brief Construct filter from explicit reference state, covariance, and keys.
 *
 * Landmark keys are part of the runtime bookkeeping, so seeded landmark states
 * must provide a matching external key ordering.
 */
EqVIOFilter::EqVIOFilter(const State& xi_ref, const Matrix& Sigma,
                         const std::vector<Key>& landmarkKeys,
                         const EqVIOFilterParams& params)
    : Base(State(), defaultCovariance(0), makeVioGroupIdentity()),
      params_(params) {
  resetReferenceAndGroup(xi_ref, Sigma, makeVioGroupIdentity(xi_ref.n()));
  setLandmarkKeys(landmarkKeys);
  initialized_ = true;
}

/**
 * @brief Initialize reference attitude from measured gravity direction.
 *
 * This method sets bias and velocity to nominal zeros and computes the shortest
 * rotation mapping measured acceleration to world +Z.
 */
void EqVIOFilter::initializeFromIMU(const IMUInput& imu) {
  State xi_ref = referenceState();
  xi_ref.sensor.inputBias = Bias::Identity();
  xi_ref.sensor.pose = Pose3::Identity();
  xi_ref.sensor.velocity.setZero();

  Vector3 approxGravity = imu.acc;
  if (approxGravity.norm() < 1e-9) approxGravity = Vector3::UnitZ();
  // Build initial attitude that aligns measured gravity with +Z.
  Quaternion q;
  q.setFromTwoVectors(approxGravity.normalized(), Vector3::UnitZ());
  const Rot3 R0(q);
  xi_ref.sensor.pose = Pose3(R0, Point3::Zero());
  initialized_ = true;
  resetReferenceAndGroup(xi_ref, errorCovariance(), groupEstimate());
}

/**
 * @brief Propagate mean/covariance jointly over buffered IMU intervals.
 *
 * Uses the explicit base-class `predictWithJacobian(...)` path per
 * positive-duration hold.
 */
void EqVIOFilter::predict(const std::vector<IMUInput>& imuInputs,
                          const std::vector<double>& dts) {
  if (!initialized_ || imuInputs.empty()) {
    return;
  }
  if (imuInputs.size() != dts.size()) {
    throw std::invalid_argument(
        "EqVIOFilter::propagate: imuInputs and dts size mismatch");
  }

  for (size_t i = 0; i < imuInputs.size(); ++i) {
    const double dt = dts[i];
    if (dt <= 0.0) continue;

    const IMUInput imu = imuInputs[i];
    const Matrix A = EqFStateMatrixA(groupEstimate(), referenceState(), imu);
    const Lift lift_u(imu, dt);

    const Matrix B = EqFInputMatrixB(groupEstimate(), referenceState());
    const Matrix Qc = B * params_.inputNoise * B.transpose() +
                      stateProcessNoise(referenceState().n());

    Base::template predictWithJacobian<1>(lift_u, A, Qc, dt);
  }
}

/**
 * @brief Visual update entry point including feature management.
 *
 * The update sequence is:
 * 1. Drop stale landmarks,
 * 2. Reject outliers and remove them from filter state,
 * 3. Add newly observed landmarks,
 * 4. Perform EKF-like correction,
 * 5. Remove numerically invalid landmarks.
 */
void EqVIOFilter::update(const VisionMeasurement& measurement,
                         const std::shared_ptr<const CameraModel>& camera,
                         const Matrix& R) {
  if (!initialized_) {
    return;
  }

  VisionMeasurement matchedMeasurement = measurement;
  reconcileLandmarks(matchedMeasurement, camera);

  if (matchedMeasurement.empty()) {
    return;
  }

  innovationUpdate(matchedMeasurement, camera, R);

  const std::vector<Key> invalidKeys = invalidLandmarkKeys();
  if (!invalidKeys.empty()) {
    const std::unordered_set<Key> invalidKeySet(invalidKeys.begin(),
                                                invalidKeys.end());
    std::vector<size_t> retainedIndices;
    retainedIndices.reserve(landmarkKeys_.size());
    for (size_t i = 0; i < landmarkKeys_.size(); ++i) {
      if (invalidKeySet.count(landmarkKeys_[i]) == 0) {
        retainedIndices.push_back(i);
      }
    }
    applyLandmarkStructureChange(retainedIndices, {});
  }

  assert(!errorCovariance().hasNaN());
}

/// Identity covariance helper sized for current sensor + landmark dimensions.
Matrix EqVIOFilter::defaultCovariance(size_t nLandmarks) {
  const int d = SensorState::CompDim + 3 * static_cast<int>(nLandmarks);
  return Matrix::Identity(d, d);
}

/// Build block-diagonal process covariance from scalar per-component variances.
Matrix EqVIOFilter::stateProcessNoise(size_t nLandmarks) const {
  Matrix Q =
      Matrix::Identity(SensorState::CompDim + 3 * static_cast<int>(nLandmarks),
                       SensorState::CompDim + 3 * static_cast<int>(nLandmarks));
  Q.block<3, 3>(0, 0) *= params_.biasOmegaProcessVariance;
  Q.block<3, 3>(3, 3) *= params_.biasAccelProcessVariance;
  Q.block<3, 3>(6, 6) *= params_.attitudeProcessVariance;
  Q.block<3, 3>(9, 9) *= params_.positionProcessVariance;
  Q.block<3, 3>(12, 12) *= params_.velocityProcessVariance;
  Q.block<3, 3>(15, 15) *= params_.cameraAttitudeProcessVariance;
  Q.block<3, 3>(18, 18) *= params_.cameraPositionProcessVariance;
  if (nLandmarks > 0) {
    Q.block(SensorState::CompDim, SensorState::CompDim,
            3 * static_cast<int>(nLandmarks),
            3 * static_cast<int>(nLandmarks)) *= params_.pointProcessVariance;
  }
  return Q;
}

/**
 * @brief Apply innovation update for matched measurements.
 *
 * If `outputGainMatrix` is not a valid measurement covariance shape, the method
 * falls back to isotropic `measurementNoiseVariance`.
 */
void EqVIOFilter::innovationUpdate(
    const VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera,
    const Matrix& outputGainMatrix) {
  if (measurement.empty()) return;
  if (!camera) {
    throw std::invalid_argument("EqVIOFilter::innovationUpdate: camera is null");
  }

  const std::vector<Key> observedKeys = measurementIds(measurement);
  VisionMeasurement estimatedMeasurement;
  const State& estimate = state();
  for (Key key : observedKeys) {
    const auto it = landmarkIndexByKey_.find(key);
    if (it == landmarkIndexByKey_.end()) {
      throw std::invalid_argument(
          "EqVIOFilter::innovationUpdate: measurement key not in filter state");
    }
    estimatedMeasurement[key] =
        camera->project2(estimate.cameraLandmarks[it->second].p);
  }
  const Matrix Ct =
      EqFoutputMatrixC(referenceState(), landmarkKeys_, groupEstimate(),
                       measurement, camera, true);
  const Matrix Rused = (outputGainMatrix.rows() == Ct.rows() &&
                        outputGainMatrix.cols() == Ct.rows())
                           ? outputGainMatrix
                           : Matrix::Identity(Ct.rows(), Ct.rows()) *
                                 params_.measurementNoiseVariance;

  const Vector zhat = measurementVector(estimatedMeasurement);
  const Vector z = measurementVector(measurement);
  Base::updateWithVector(zhat, Ct, z, Rused,
                         [this](const Vector& delta_xi) -> Vector {
                           return liftInnovation(delta_xi, referenceState());
                         });
}

/**
 * @brief Validate/store landmark keys for the current state dimension.
 */
void EqVIOFilter::setLandmarkKeys(const std::vector<Key>& landmarkKeys) {
  if (landmarkKeys.size() != referenceState().n()) {
    throw std::invalid_argument(
        "EqVIOFilter::setLandmarkKeys: key count must match landmark count");
  }

  std::unordered_set<Key> uniqueKeys;
  uniqueKeys.reserve(landmarkKeys.size());
  for (Key key : landmarkKeys) {
    const auto [_, inserted] = uniqueKeys.insert(key);
    if (!inserted) {
      throw std::invalid_argument(
          "EqVIOFilter::setLandmarkKeys: duplicate landmark key");
    }
  }

  landmarkKeys_ = landmarkKeys;
  missedFrameCounts_.assign(landmarkKeys_.size(), 0);
  rebuildLandmarkIndex();
}

/// Refresh the O(1) lookup table aligned with `landmarkKeys_`.
void EqVIOFilter::rebuildLandmarkIndex() {
  landmarkIndexByKey_.clear();
  landmarkIndexByKey_.reserve(landmarkKeys_.size());
  for (size_t i = 0; i < landmarkKeys_.size(); ++i) {
    landmarkIndexByKey_[landmarkKeys_[i]] = i;
  }
}

/**
 * @brief Batch landmark bookkeeping around one visual update.
 *
 * Existing tracks survive one missed frame, absolute-residual outliers are
 * pruned from the current measurement/update, and new landmarks are inserted in
 * one structure rebuild.
 */
void EqVIOFilter::reconcileLandmarks(
    VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera) {
  if (!camera && !measurement.empty()) {
    throw std::invalid_argument("EqVIOFilter::reconcileLandmarks: camera is null");
  }

  std::unordered_set<Key> observedKeys;
  observedKeys.reserve(measurement.size());
  for (const auto& [key, _] : measurement) {
    observedKeys.insert(key);
  }

  for (size_t i = 0; i < landmarkKeys_.size(); ++i) {
    if (observedKeys.count(landmarkKeys_[i]) != 0) {
      missedFrameCounts_[i] = 0;
    } else {
      ++missedFrameCounts_[i];
    }
  }

  std::vector<Key> removalKeys = detectOutliers(measurement, camera);

  const std::vector<Key> invalidKeys = invalidLandmarkKeys();
  removalKeys.insert(removalKeys.end(), invalidKeys.begin(), invalidKeys.end());
  for (size_t i = 0; i < landmarkKeys_.size(); ++i) {
    if (missedFrameCounts_[i] > kMaxMissedFrames) {
      removalKeys.push_back(landmarkKeys_[i]);
    }
  }

  std::unordered_set<Key> removalKeySet(removalKeys.begin(), removalKeys.end());
  std::vector<size_t> retainedIndices;
  retainedIndices.reserve(landmarkKeys_.size());
  for (size_t i = 0; i < landmarkKeys_.size(); ++i) {
    if (removalKeySet.count(landmarkKeys_[i]) == 0) {
      retainedIndices.push_back(i);
    }
  }

  std::vector<std::pair<Key, Landmark>> newLandmarks;
  newLandmarks.reserve(measurement.size());
  for (const auto& [key, coordinate] : measurement) {
    if (landmarkIndexByKey_.count(key) != 0) continue;

    Landmark landmark{undistortPoint(*camera, coordinate) *
                      params_.initialPointDepth};
    newLandmarks.emplace_back(key, landmark);
  }

  if (retainedIndices.size() == landmarkKeys_.size() && newLandmarks.empty()) {
    return;
  }

  applyLandmarkStructureChange(retainedIndices, newLandmarks);
}

/**
 * @brief Reject outliers using only absolute reprojection residual.
 *
 * This keeps the example/runtime path cheap and avoids using a partial
 * innovation covariance proxy for gating.
 */
std::vector<Key> EqVIOFilter::detectOutliers(
    VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera) const {
  const size_t maxOutliers = static_cast<size_t>(
      (1.0 - params_.featureRetention) * measurement.size());
  if (!camera || measurement.empty() || maxOutliers == 0 ||
      landmarkKeys_.empty()) {
    return {};
  }

  const VisionMeasurement yHat =
      measureSystemState(state(), landmarkKeys_, camera);

  std::vector<std::pair<Key, double>> residuals;
  residuals.reserve(measurement.size());
  for (const auto& [lmId, y] : measurement) {
    const auto itHat = yHat.find(lmId);
    if (itHat == yHat.end()) continue;
    const double errAbs = (y - itHat->second).norm();
    if (errAbs > params_.outlierThresholdAbs) {
      residuals.emplace_back(lmId, errAbs);
    }
  }

  std::sort(residuals.begin(), residuals.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
  if (residuals.size() > maxOutliers) {
    residuals.resize(maxOutliers);
  }

  std::vector<Key> outlierKeys;
  outlierKeys.reserve(residuals.size());
  for (const auto& [lmId, _] : residuals) {
    measurement.erase(lmId);
    outlierKeys.push_back(lmId);
  }
  return outlierKeys;
}

/// Return keys whose landmark-group scale has become numerically invalid.
std::vector<Key> EqVIOFilter::invalidLandmarkKeys() const {
  const LandmarkGroup& Q = std::get<3>(decompose(groupEstimate()));
  std::vector<Key> invalidKeys;
  invalidKeys.reserve(Q.size());
  for (size_t i = 0; i < Q.size(); ++i) {
    const double scale = SOT3Scale(Q[i]);
    if (!std::isfinite(scale) || scale <= 1e-8 || scale > 1e8) {
      invalidKeys.push_back(landmarkKeys_[i]);
    }
  }
  return invalidKeys;
}

/**
 * @brief Rebuild the dynamic landmark structure in one pass.
 *
 * The physical estimate is preserved for retained landmarks, while newly
 * observed landmarks enter with identity group transform and diagonal initial
 * covariance.
 */
void EqVIOFilter::applyLandmarkStructureChange(
    const std::vector<size_t>& retainedIndices,
    const std::vector<std::pair<Key, Landmark>>& newLandmarks) {
  const State& currentReference = referenceState();
  const auto& [A, Beta, B, Q] = decompose(groupEstimate());

  State nextReference;
  nextReference.sensor = currentReference.sensor;
  nextReference.cameraLandmarks.reserve(retainedIndices.size() +
                                        newLandmarks.size());

  std::vector<Key> nextKeys;
  std::vector<size_t> nextMissedFrameCounts;
  std::vector<SOT3> nextQ;
  nextKeys.reserve(retainedIndices.size() + newLandmarks.size());
  nextMissedFrameCounts.reserve(retainedIndices.size() + newLandmarks.size());
  nextQ.reserve(retainedIndices.size() + newLandmarks.size());

  for (size_t index : retainedIndices) {
    nextReference.cameraLandmarks.push_back(
        currentReference.cameraLandmarks[index]);
    nextKeys.push_back(landmarkKeys_[index]);
    nextMissedFrameCounts.push_back(missedFrameCounts_[index]);
    nextQ.push_back(Q[index]);
  }

  for (const auto& [key, landmark] : newLandmarks) {
    nextReference.cameraLandmarks.push_back(landmark);
    nextKeys.push_back(key);
    nextMissedFrameCounts.push_back(0);
    nextQ.push_back(SOT3::Identity());
  }

  const Matrix nextCovariance =
      rebuildCovariance(retainedIndices, newLandmarks.size());
  const VioGroup nextGroup =
      makeVioGroup(A, Beta, B, LandmarkGroup(nextQ));

  resetReferenceAndGroup(nextReference, nextCovariance, nextGroup);
  landmarkKeys_ = std::move(nextKeys);
  missedFrameCounts_ = std::move(nextMissedFrameCounts);
  rebuildLandmarkIndex();
}

/// Rebuild covariance after a batch landmark add/remove operation.
Matrix EqVIOFilter::rebuildCovariance(
    const std::vector<size_t>& retainedIndices, size_t newLandmarkCount) const {
  const Matrix& currentCovariance = errorCovariance();
  const int retainedLandmarkCount = static_cast<int>(retainedIndices.size());
  const int newLandmarkBlockCount = static_cast<int>(newLandmarkCount);
  const int newDimension =
      SensorState::CompDim + 3 * (retainedLandmarkCount + newLandmarkBlockCount);

  Matrix rebuilt = Matrix::Zero(newDimension, newDimension);
  rebuilt.block(0, 0, SensorState::CompDim, SensorState::CompDim) =
      currentCovariance.block(0, 0, SensorState::CompDim, SensorState::CompDim);

  for (size_t newI = 0; newI < retainedIndices.size(); ++newI) {
    const int srcI =
        SensorState::CompDim + 3 * static_cast<int>(retainedIndices[newI]);
    const int dstI = SensorState::CompDim + 3 * static_cast<int>(newI);

    rebuilt.block(0, dstI, SensorState::CompDim, 3) =
        currentCovariance.block(0, srcI, SensorState::CompDim, 3);
    rebuilt.block(dstI, 0, 3, SensorState::CompDim) =
        currentCovariance.block(srcI, 0, 3, SensorState::CompDim);

    for (size_t newJ = 0; newJ < retainedIndices.size(); ++newJ) {
      const int srcJ =
          SensorState::CompDim + 3 * static_cast<int>(retainedIndices[newJ]);
      const int dstJ = SensorState::CompDim + 3 * static_cast<int>(newJ);
      rebuilt.block(dstI, dstJ, 3, 3) = currentCovariance.block(srcI, srcJ, 3, 3);
    }
  }

  if (newLandmarkBlockCount > 0) {
    const int newOffset = SensorState::CompDim + 3 * retainedLandmarkCount;
    rebuilt.block(newOffset, newOffset, 3 * newLandmarkBlockCount,
                  3 * newLandmarkBlockCount) =
        Matrix::Identity(3 * newLandmarkBlockCount, 3 * newLandmarkBlockCount) *
        params_.initialPointVariance;
  }

  return rebuilt;
}

}  // namespace eqvio
}  // namespace gtsam
