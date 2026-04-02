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
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace gtsam {
namespace eqvio {

/// Default constructor delegates to explicit-params constructor.
EqVIOFilter::EqVIOFilter() : EqVIOFilter(EqVIOFilterParams()) {}

/**
 * @brief Construct filter with given parameter bundle and identity initial state.
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
  landmarkKeys_.clear();
}

/**
 * @brief Construct filter from explicit reference state and covariance.
 *
 * The group estimate is initialized to identity for `xi0.n()` landmarks.
 */
EqVIOFilter::EqVIOFilter(const State& xi_ref, const Matrix& Sigma,
                         const EqVIOFilterParams& params)
    : Base(State(), defaultCovariance(0), makeVioGroupIdentity()),
      params_(params) {
  resetReferenceAndGroup(xi_ref, Sigma, makeVioGroupIdentity(xi_ref.n()));
  landmarkKeys_.resize(xi_ref.n());
  std::iota(landmarkKeys_.begin(), landmarkKeys_.end(), 0);
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
 * Uses the automatic base-class `predict(...)` path per positive-duration hold.
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
    const Matrix A_target = EqFStateMatrixA(groupEstimate(), referenceState(), imu);
    Symmetry::Orbit actionAtRef(referenceState());
    Matrix Dphi0;
    const VioGroup identity_at_g =
        traits<VioGroup>::Compose(groupEstimate().inverse(), groupEstimate());
    actionAtRef(identity_at_g, &Dphi0);
    const Matrix D_lift =
        Dphi0.completeOrthogonalDecomposition().pseudoInverse() * A_target;

    const std::tuple<IMUInput, double, Matrix> u{imu, dt, D_lift};
    const Lift lift_u(u);
    const InputAction::Orbit psi_u(u);

    const Matrix B = EqFInputMatrixB(groupEstimate(), referenceState());
    const Matrix Qc =
        B * params_.inputNoise * B.transpose() + stateProcessNoise(referenceState().n());

    Base::template predict<1>(lift_u, psi_u, Qc, dt);
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

  removeOldLandmarks(measurementIds(measurement));

  VisionMeasurement matchedMeasurement = measurement;
  removeOutliers(matchedMeasurement, camera);
  addNewLandmarks(matchedMeasurement, camera);

  if (matchedMeasurement.empty()) {
    return;
  }

  innovationUpdate(matchedMeasurement, camera, R);
  removeInvalidLandmarks();

  assert(!errorCovariance().hasNaN());
}

/// Convenience update overload with default normalized pinhole camera.
void EqVIOFilter::update(const VisionMeasurement& measurement, const Matrix& R) {
  static const auto kDefaultCamera = std::make_shared<CameraModel>(
      Pose3::Identity(), Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));
  update(measurement, kDefaultCamera, R);
}

/// Identity covariance helper sized for current sensor + landmark dimensions.
Matrix EqVIOFilter::defaultCovariance(size_t nLandmarks) {
  const int d = SensorState::CompDim + 3 * static_cast<int>(nLandmarks);
  return Matrix::Identity(d, d);
}

/// Build block-diagonal process covariance from scalar per-component variances.
Matrix EqVIOFilter::stateProcessNoise(size_t nLandmarks) const {
  Matrix Q = Matrix::Identity(
      SensorState::CompDim + 3 * static_cast<int>(nLandmarks),
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
            3 * static_cast<int>(nLandmarks), 3 * static_cast<int>(nLandmarks)) *=
        params_.pointProcessVariance;
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

  const VisionMeasurement estimatedMeasurement =
      measureSystemState(state(), landmarkKeys_, camera);
  const Matrix Ct = EqFoutputMatrixC(referenceState(), landmarkKeys_,
                                     groupEstimate(), measurement, camera, true);
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
 * @brief Insert unseen landmarks from current vision measurement.
 *
 * New landmarks are initialized from normalized bearing at `initialPointDepth`,
 * corresponding covariance is appended, and group/covariance dimensions are expanded.
 */
void EqVIOFilter::addNewLandmarks(
    const VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera) {
  if (measurement.empty()) return;
  if (!camera) throw std::invalid_argument("EqVIOFilter::addNewLandmarks: camera is null");

  std::vector<Landmark> newLandmarks;
  std::vector<Key> newKeys;
  const std::vector<Key> existingIds = landmarkKeys_;
  for (const auto& [id, coord] : measurement) {
    if (std::find(existingIds.begin(), existingIds.end(), id) != existingIds.end()) {
      continue;
    }
    const Vector3 bearing = undistortPoint(*camera, coord);
    newLandmarks.push_back(Landmark{bearing});
    newKeys.push_back(id);
  }

  if (newLandmarks.empty()) return;

  const double initialDepth = params_.initialPointDepth;
  for (Landmark& lm : newLandmarks) lm.p *= initialDepth;

  const Matrix newLandmarksCov =
      Matrix::Identity(3 * static_cast<int>(newLandmarks.size()),
                       3 * static_cast<int>(newLandmarks.size())) *
      params_.initialPointVariance;

  State xi_ref = referenceState();
  xi_ref.cameraLandmarks.insert(xi_ref.cameraLandmarks.end(), newLandmarks.begin(),
                             newLandmarks.end());
  landmarkKeys_.insert(landmarkKeys_.end(), newKeys.begin(), newKeys.end());

  const auto& [A, Beta, B, Q] = decompose(groupEstimate());
  std::vector<SOT3> q;
  q.reserve(Q.size() + newLandmarks.size());
  for (size_t i = 0; i < Q.size(); ++i) {
    q.push_back(Q[i]);
  }
  for (size_t i = 0; i < newLandmarks.size(); ++i) {
    q.push_back(SOT3::Identity());
  }

  const VioGroup X = makeVioGroup(A, Beta, B, LandmarkGroup(q));

  Matrix Sigma = errorCovariance();
  const int oldSize = Sigma.rows();
  const int newN = static_cast<int>(newLandmarks.size());
  Sigma.conservativeResize(oldSize + 3 * newN, oldSize + 3 * newN);
  Sigma.block(oldSize, 0, 3 * newN, oldSize).setZero();
  Sigma.block(0, oldSize, oldSize, 3 * newN).setZero();
  Sigma.block(oldSize, oldSize, 3 * newN, 3 * newN) = newLandmarksCov;

  resetReferenceAndGroup(xi_ref, Sigma, X);
}

/// Remove any landmarks that are absent from the current measurement id list.
void EqVIOFilter::removeOldLandmarks(const std::vector<Key>& measurementIds) {
  const std::vector<Key> existingIds = landmarkKeys_;
  std::vector<int> lostIndices(existingIds.size());
  std::iota(lostIndices.begin(), lostIndices.end(), 0);
  if (lostIndices.empty()) return;

  const auto lostIndicesEnd = std::remove_if(
      lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const Key oldId = existingIds[static_cast<size_t>(lidx)];
        return std::any_of(measurementIds.begin(), measurementIds.end(),
                           [&oldId](const Key& measId) { return measId == oldId; });
      });
  lostIndices.erase(lostIndicesEnd, lostIndices.end());

  if (lostIndices.empty()) return;

  std::reverse(lostIndices.begin(), lostIndices.end());
  for (const int idx : lostIndices) {
    removeLandmarkByIndex(idx);
  }
}

/// Remove landmark at index `idx` from state, group, and covariance.
void EqVIOFilter::removeLandmarkByIndex(int idx) {
  State xi_ref = referenceState();
  xi_ref.cameraLandmarks.erase(xi_ref.cameraLandmarks.begin() + idx);
  landmarkKeys_.erase(landmarkKeys_.begin() + idx);

  const auto& [A, Beta, B, Q] = decompose(groupEstimate());

  std::vector<SOT3> q;
  q.reserve(Q.size() - 1);
  for (size_t i = 0; i < Q.size(); ++i) {
    if (static_cast<int>(i) == idx) continue;
    q.push_back(Q[i]);
  }

  const VioGroup X = makeVioGroup(A, Beta, B, LandmarkGroup(q));

  Matrix Sigma = errorCovariance();
  removeRows(Sigma, SensorState::CompDim + 3 * idx, 3);
  removeCols(Sigma, SensorState::CompDim + 3 * idx, 3);

  resetReferenceAndGroup(xi_ref, Sigma, X);
}

/// Remove landmark with matching key id.
void EqVIOFilter::removeLandmarkById(Key id) {
  const auto it = std::find(landmarkKeys_.begin(), landmarkKeys_.end(), id);
  assert(it != landmarkKeys_.end());
  removeLandmarkByIndex(static_cast<int>(std::distance(landmarkKeys_.begin(), it)));
}

/// Return 3x3 landmark-state covariance block for the specified id.
Matrix3 EqVIOFilter::getLandmarkCovById(Key id) const {
  const auto it = std::find(landmarkKeys_.begin(), landmarkKeys_.end(), id);
  assert(it != landmarkKeys_.end());
  const int i = static_cast<int>(std::distance(landmarkKeys_.begin(), it));
  return errorCovariance().block<3, 3>(SensorState::CompDim + 3 * i,
                                       SensorState::CompDim + 3 * i);
}

/// Project landmark covariance through output Jacobian into 2x2 image-space covariance.
Matrix2 EqVIOFilter::outputCovarianceById(
    Key id, const std::shared_ptr<const CameraModel>& camera) const {
  const Matrix3 lmCov = getLandmarkCovById(id);
  const auto it = std::find(landmarkKeys_.begin(), landmarkKeys_.end(), id);
  assert(it != landmarkKeys_.end());

  const size_t i = static_cast<size_t>(std::distance(landmarkKeys_.begin(), it));
  const LandmarkGroup& Q = std::get<3>(decompose(groupEstimate()));
  const SOT3& Q_i = Q[i];
  const Point3& q0 = referenceState().cameraLandmarks[i].p;

  const Matrix23 C0i = EqFoutputMatrixCi(q0, Q_i, camera);
  return C0i * lmCov * C0i.transpose();
}

/// Remove landmarks with non-finite or extreme SOT3 scale factors.
void EqVIOFilter::removeInvalidLandmarks() {
  const LandmarkGroup& Q = std::get<3>(decompose(groupEstimate()));
  std::set<Key> invalidIds;
  for (size_t i = 0; i < Q.size(); ++i) {
    const double a = SOT3Scale(Q[i]);
    if (!std::isfinite(a) || a <= 1e-8 || a > 1e8) {
      invalidIds.insert(landmarkKeys_[i]);
    }
  }
  for (const Key id : invalidIds) {
    removeLandmarkById(id);
  }
}

/**
 * @brief Reject outliers using absolute residual and Mahalanobis-style tests.
 *
 * At most `(1 - featureRetention)` fraction of currently measured features are
 * removed, prioritized by largest residual score.
 */
void EqVIOFilter::removeOutliers(
    VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera) {
  const size_t maxOutliers = static_cast<size_t>(
      (1.0 - params_.featureRetention) * measurement.size());
  if (!camera) return;

  const VisionMeasurement yHat = measureSystemState(state(), landmarkKeys_, camera);

  std::vector<Key> proposedOutliers;
  std::map<Key, double> absoluteOutliers;
  for (const auto& [lmId, yHatI] : yHat) {
    if (measurement.count(lmId) == 0) continue;
    const double errAbs = (measurement.at(lmId) - yHatI).norm();
    if (errAbs > params_.outlierThresholdAbs) {
      absoluteOutliers[lmId] = errAbs;
      proposedOutliers.push_back(lmId);
    }
  }

  std::map<Key, double> probabilisticOutliers;
  for (const auto& [lmId, yI] : measurement) {
    if (absoluteOutliers.count(lmId)) continue;
    const auto itHat = yHat.find(lmId);
    if (itHat == yHat.end()) continue;
    const Point2 yTildeI = yI - itHat->second;

    const Matrix2 outputCov = outputCovarianceById(lmId, camera);
    const double errProb = yTildeI.transpose() * outputCov.inverse() * yTildeI;
    if (errProb > params_.outlierThresholdProb) {
      probabilisticOutliers[lmId] = errProb;
      proposedOutliers.push_back(lmId);
    }
  }

  std::sort(proposedOutliers.begin(), proposedOutliers.end(),
            [&absoluteOutliers, &probabilisticOutliers](Key lmId1, Key lmId2) {
              if (absoluteOutliers.count(lmId1)) {
                if (absoluteOutliers.count(lmId2)) {
                  return absoluteOutliers.at(lmId1) < absoluteOutliers.at(lmId2);
                }
                return false;
              }
              if (absoluteOutliers.count(lmId2)) return true;
              return probabilisticOutliers.at(lmId1) < probabilisticOutliers.at(lmId2);
            });
  std::reverse(proposedOutliers.begin(), proposedOutliers.end());
  if (proposedOutliers.size() > maxOutliers) {
    proposedOutliers.erase(proposedOutliers.begin() + maxOutliers,
                           proposedOutliers.end());
  }

  for (const Key lmId : proposedOutliers) {
    removeLandmarkById(lmId);
    measurement.erase(lmId);
  }
}

}  // namespace eqvio
}  // namespace gtsam
