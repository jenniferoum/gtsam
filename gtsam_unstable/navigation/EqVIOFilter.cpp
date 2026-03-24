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

#include <cassert>
#include <stdexcept>

namespace gtsam {
namespace eqvio {

namespace {

/**
 * @brief Stack a landmark-id-ordered measurement map into a dense vector.
 *
 * The order follows the `std::map` key ordering, which matches all other
 * EqVIO measurement matrix assembly paths.
 */
Vector _measurementVector(const VisionMeasurement& measurement) {
  Vector v = Vector::Zero(static_cast<int>(2 * measurement.size()));
  int i = 0;
  for (const auto& item : measurement) {
    v.segment<2>(2 * i) = item.second;
    ++i;
  }
  return v;
}

/**
 * @brief Compute `lhs - rhs` residual for two id-aligned measurement maps.
 * @throws std::invalid_argument if map sizes or ids do not match exactly.
 */
Vector _measurementDifference(const VisionMeasurement& lhs,
                              const VisionMeasurement& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("measurementDifference: size mismatch");
  }

  Vector diff = Vector::Zero(static_cast<int>(2 * lhs.size()));
  auto itL = lhs.begin();
  auto itR = rhs.begin();
  int i = 0;
  for (; itL != lhs.end(); ++itL, ++itR) {
    if (itL->first != itR->first) {
      throw std::invalid_argument("measurementDifference: id mismatch");
    }
    diff.segment<2>(2 * i) = itL->second - itR->second;
    ++i;
  }
  return diff;
}

}  // namespace

/// Default constructor delegates to explicit-params constructor.
EqVIOFilter::EqVIOFilter() : EqVIOFilter(EqVIOFilterParams()) {}

/**
 * @brief Construct filter with given parameter bundle and identity initial state.
 *
 * The internal snapshot is initialized and pushed into the base
 * `EquivariantFilter` storage via `syncBase(true)`.
 */
EqVIOFilter::EqVIOFilter(const EqVIOFilterParams& params)
    : Base(State(), defaultCovariance(0), makeVioGroupIdentity()),
      params_(params) {
  snapshot_.xi0 = State();
  snapshot_.xi0.sensor.inputBias = Bias::Identity();
  snapshot_.xi0.sensor.pose = Pose3::Identity();
  snapshot_.xi0.sensor.velocity.setZero();
  snapshot_.xi0.sensor.cameraOffset = Pose3::Identity();
  snapshot_.X = makeVioGroupIdentity();
  snapshot_.Sigma = defaultCovariance(0);
  syncBase(true);
}

/**
 * @brief Construct filter from explicit reference state and covariance.
 *
 * The group estimate is initialized to identity for `xi0.n()` landmarks.
 */
EqVIOFilter::EqVIOFilter(const State& xi0, const Matrix& Sigma0,
                         const EqVIOFilterParams& params)
    : Base(State(), defaultCovariance(0), makeVioGroupIdentity()),
      params_(params) {
  snapshot_.xi0 = xi0;
  snapshot_.X = makeVioGroupIdentity(snapshot_.xi0.n());
  snapshot_.Sigma = Sigma0;
  initialized_ = true;
  syncBase(true);
}

/**
 * @brief Initialize reference attitude from measured gravity direction.
 *
 * This method sets bias and velocity to nominal zeros and computes the shortest
 * rotation mapping measured acceleration to world +Z.
 */
void EqVIOFilter::initializeFromIMU(const IMUInput& imu) {
  snapshot_.xi0.sensor.inputBias = Bias::Identity();
  snapshot_.xi0.sensor.pose = Pose3::Identity();
  snapshot_.xi0.sensor.velocity.setZero();

  Vector3 approxGravity = imu.acc;
  if (approxGravity.norm() < 1e-9) approxGravity = Vector3::UnitZ();
  // Build initial attitude that aligns measured gravity with +Z.
  Quaternion q;
  q.setFromTwoVectors(approxGravity.normalized(), Vector3::UnitZ());
  const Rot3 R0(q);
  snapshot_.xi0.sensor.pose = Pose3(R0, Point3::Zero());
  initialized_ = true;
  syncBase(true);
}

/// Replace current reference state/covariance after dimension validation.
void EqVIOFilter::setReferenceState(const State& xi0, const Matrix& Sigma0) {
  snapshot_.xi0 = xi0;
  snapshot_.X = makeVioGroupIdentity(xi0.n());
  snapshot_.Sigma = Sigma0;
  syncBase(true);
}

/**
 * @brief Propagate covariance and state over buffered IMU intervals.
 *
 * Covariance uses the time-weighted averaged IMU input over all positive `dt`.
 * State integration is then applied per-segment to preserve replay semantics.
 */
void EqVIOFilter::propagate(const std::vector<IMUInput>& imuInputs,
                            const std::vector<double>& dts) {
  if (!initialized_ || imuInputs.empty()) {
    return;
  }
  if (imuInputs.size() != dts.size()) {
    throw std::invalid_argument(
        "EqVIOFilter::propagate: imuInputs and dts size mismatch");
  }

  double totalTime = 0.0;
  IMUInput averagedImu = IMUInput::Zero();
  for (size_t i = 0; i < imuInputs.size(); ++i) {
    const double dt = dts[i];
    if (dt <= 0.0) continue;
    totalTime += dt;
    averagedImu = averagedImu + imuInputs[i] * dt;
  }
  if (totalTime <= 0.0) {
    return;
  }

  averagedImu = averagedImu * (1.0 / totalTime);
  propagateCovariance(averagedImu, totalTime);

  for (size_t i = 0; i < imuInputs.size(); ++i) {
    const double dt = dts[i];
    if (dt <= 0.0) continue;
    propagateState(imuInputs[i], dt);
  }
}

/// Covariance-only propagation through linearized EqF error dynamics.
void EqVIOFilter::propagateCovariance(const IMUInput& imu, double dt) {
  if (!initialized_ || dt <= 0.0) {
    return;
  }
  const Matrix A = EqFStateMatrixA(snapshot_.X, snapshot_.xi0, imu);
  const Matrix B = EqFInputMatrixB(snapshot_.X, snapshot_.xi0);
  const Matrix Qc =
      B * params_.inputNoise * B.transpose() + stateProcessNoise(snapshot_.xi0.n());

  auto zeroLift = [this](const State&) -> Vector {
    return Vector::Zero(static_cast<int>(Dim_groupTangent(snapshot_.X)));
  };

  Base::template predictWithJacobian<1>(zeroLift, A, Qc, dt);
  syncFromBase();
}

/// State-only propagation through lifted discrete system increment.
void EqVIOFilter::propagateState(const IMUInput& imu, double dt) {
  if (!initialized_ || dt <= 0.0) {
    return;
  }
  auto liftFunctor = [imu, dt](const State& xi) -> Vector {
    return (VioGroup::Logmap(liftVelocityDiscrete(xi, imu, dt)) / dt).eval();
  };
  const Matrix A = Matrix::Zero(snapshot_.xi0.dim(), snapshot_.xi0.dim());
  const Matrix Qc = Matrix::Zero(snapshot_.xi0.dim(), snapshot_.xi0.dim());
  Base::template predictWithJacobian<1>(liftFunctor, A, Qc, dt);
  syncFromBase();
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
void EqVIOFilter::correct(const VisionMeasurement& measurement,
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
    syncBase(false);
    return;
  }

  update(matchedMeasurement, camera, R);
  removeInvalidLandmarksNow();

  syncBase(false);

  assert(!snapshot_.Sigma.hasNaN());
}

/// Return current physical-state estimate by applying group action to reference state.
State EqVIOFilter::stateEstimate() const {
  return stateGroupAction(snapshot_.X, snapshot_.xi0);
}

/// Identity covariance helper sized for current sensor + landmark dimensions.
Matrix EqVIOFilter::defaultCovariance(size_t nLandmarks) {
  const int d = SensorState::CompDim + 3 * static_cast<int>(nLandmarks);
  return Matrix::Identity(d, d);
}

/**
 * @brief Push local snapshot into the `EquivariantFilter` base.
 * @param resetReference If true, reset base reference state; otherwise only sync estimate/covariance.
 */
void EqVIOFilter::syncBase(bool resetReference) {
  if (resetReference) {
    resetReferenceAndGroup(snapshot_.xi0, snapshot_.Sigma, snapshot_.X);
    return;
  }
  // ensure referenceState().n() == N_landmarkCount(snapshot_.X))
  setGroupEstimateAndSyncState(snapshot_.X);
  setErrorCovariance(snapshot_.Sigma);
}

/// Pull group estimate and covariance from the `EquivariantFilter` base into local snapshot.
void EqVIOFilter::syncFromBase() {
  snapshot_.X = groupEstimate();
  snapshot_.Sigma = errorCovariance();
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
 * @brief Insert unseen landmarks from current vision measurement.
 *
 * New landmarks are initialized from normalized bearing at `initialPointDepth`,
 * corresponding covariance is appended, and group/covariance dimensions are expanded.
 */
void EqVIOFilter::addNewLandmarks(
    const VisionMeasurement& measurement,
    const std::shared_ptr<const CameraModel>& camera) {
  if (measurement.empty()) return;

  std::vector<Landmark> newLandmarks;
  const std::vector<int> existingIds = snapshot_.xi0.ids();
  for (const auto& [id, coord] : measurement) {
    if (std::find(existingIds.begin(), existingIds.end(), id) != existingIds.end()) {
      continue;
    }
    const Vector3 bearing = undistortPoint(*camera, coord);
    newLandmarks.push_back(Landmark{bearing, id});
  }

  if (newLandmarks.empty()) return;

  const double initialDepth = params_.initialPointDepth;
  for (Landmark& lm : newLandmarks) lm.p *= initialDepth;

  const Matrix newLandmarksCov =
      Matrix::Identity(3 * static_cast<int>(newLandmarks.size()),
                       3 * static_cast<int>(newLandmarks.size())) *
      params_.initialPointVariance;

  snapshot_.xi0.cameraLandmarks.insert(snapshot_.xi0.cameraLandmarks.end(),
                                   newLandmarks.begin(), newLandmarks.end());

  const auto& [A, Beta, B, Q] = decompose(snapshot_.X);
  std::vector<SOT3> q;
  q.reserve(Q.size() + newLandmarks.size());
  for (size_t i = 0; i < Q.size(); ++i) {
    q.push_back(Q[i]);
  }
  for (size_t i = 0; i < newLandmarks.size(); ++i) {
    q.push_back(SOT3::Identity());
  }

  snapshot_.X = makeVioGroup(A, Beta, B, LandmarkGroup(q));

  const int oldSize = snapshot_.Sigma.rows();
  const int newN = static_cast<int>(newLandmarks.size());
  snapshot_.Sigma.conservativeResize(oldSize + 3 * newN, oldSize + 3 * newN);
  snapshot_.Sigma.block(oldSize, 0, 3 * newN, oldSize).setZero();
  snapshot_.Sigma.block(0, oldSize, oldSize, 3 * newN).setZero();
  snapshot_.Sigma.block(oldSize, oldSize, 3 * newN, 3 * newN) = newLandmarksCov;

  syncBase(true);
}

/// Remove any landmarks that are absent from the current measurement id list.
void EqVIOFilter::removeOldLandmarks(const std::vector<int>& measurementIds) {
  const std::vector<int> existingIds = snapshot_.xi0.ids();
  std::vector<int> lostIndices(existingIds.size());
  std::iota(lostIndices.begin(), lostIndices.end(), 0);
  if (lostIndices.empty()) return;

  const auto lostIndicesEnd = std::remove_if(
      lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const int oldId = existingIds[static_cast<size_t>(lidx)];
        return std::any_of(measurementIds.begin(), measurementIds.end(),
                           [&oldId](const int& measId) { return measId == oldId; });
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
  snapshot_.xi0.cameraLandmarks.erase(snapshot_.xi0.cameraLandmarks.begin() + idx);

  const auto& [A, Beta, B, Q] = decompose(snapshot_.X);

  std::vector<SOT3> q;
  q.reserve(Q.size() - 1);
  for (size_t i = 0; i < Q.size(); ++i) {
    if (static_cast<int>(i) == idx) continue;
    q.push_back(Q[i]);
  }

  snapshot_.X = makeVioGroup(A, Beta, B, LandmarkGroup(q));

  removeRows(snapshot_.Sigma, SensorState::CompDim + 3 * idx, 3);
  removeCols(snapshot_.Sigma, SensorState::CompDim + 3 * idx, 3);

  syncBase(true);
}

/// Remove landmark with matching integer id.
void EqVIOFilter::removeLandmarkById(int id) {
  const auto it = std::find_if(
      snapshot_.xi0.cameraLandmarks.begin(), snapshot_.xi0.cameraLandmarks.end(),
      [&id](const Landmark& lm) { return lm.id == id; });
  assert(it != snapshot_.xi0.cameraLandmarks.end());
  removeLandmarkByIndex(
      static_cast<int>(std::distance(snapshot_.xi0.cameraLandmarks.begin(), it)));
}

/// Return 3x3 landmark-state covariance block for the specified id.
Matrix3 EqVIOFilter::getLandmarkCovById(int id) const {
  const auto it = std::find_if(
      snapshot_.xi0.cameraLandmarks.begin(), snapshot_.xi0.cameraLandmarks.end(),
      [&id](const Landmark& lm) { return lm.id == id; });
  assert(it != snapshot_.xi0.cameraLandmarks.end());
  const int i =
      static_cast<int>(std::distance(snapshot_.xi0.cameraLandmarks.begin(), it));
  return snapshot_.Sigma.block<3, 3>(SensorState::CompDim + 3 * i,
                                 SensorState::CompDim + 3 * i);
}

/// Project landmark covariance through output Jacobian into 2x2 image-space covariance.
Matrix2 EqVIOFilter::outputCovarianceById(
    int id, const std::shared_ptr<const CameraModel>& camera) const {
  const Matrix3 lmCov = getLandmarkCovById(id);
  const auto it = std::find_if(
      snapshot_.xi0.cameraLandmarks.begin(), snapshot_.xi0.cameraLandmarks.end(),
      [&id](const Landmark& lm) { return lm.id == id; });
  assert(it != snapshot_.xi0.cameraLandmarks.end());

  const size_t i =
      static_cast<size_t>(std::distance(snapshot_.xi0.cameraLandmarks.begin(), it));
  const LandmarkGroup& Q = std::get<3>(decompose(snapshot_.X));
  const SOT3& Q_i = Q[i];

  const Matrix23 C0i = EqFoutputMatrixCi(it->p, Q_i, camera);
  return C0i * lmCov * C0i.transpose();
}

/// Remove landmarks with non-finite or extreme SOT3 scale factors.
void EqVIOFilter::removeInvalidLandmarksNow() {
  const LandmarkGroup& Q = std::get<3>(decompose(snapshot_.X));
  std::set<int> invalidIds;
  for (size_t i = 0; i < Q.size(); ++i) {
    const double a = SOT3Scale(Q[i]);
    if (!std::isfinite(a) || a <= 1e-8 || a > 1e8) {
      invalidIds.insert(snapshot_.xi0.cameraLandmarks[i].id);
    }
  }
  for (const int id : invalidIds) {
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

  const VisionMeasurement yHat = measureSystemState(stateEstimate(), camera);

  std::vector<int> proposedOutliers;
  std::map<int, double> absoluteOutliers;
  for (const auto& [lmId, yHatI] : yHat) {
    if (measurement.count(lmId) == 0) continue;
    const double errAbs = (measurement.at(lmId) - yHatI).norm();
    if (errAbs > params_.outlierThresholdAbs) {
      absoluteOutliers[lmId] = errAbs;
      proposedOutliers.push_back(lmId);
    }
  }

  std::map<int, double> probabilisticOutliers;
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
            [&absoluteOutliers, &probabilisticOutliers](int lmId1, int lmId2) {
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

  for (const int lmId : proposedOutliers) {
    removeLandmarkById(lmId);
    measurement.erase(lmId);
  }
}

/**
 * @brief Apply innovation update for matched measurements.
 *
 * If `outputGainMatrix` is not a valid measurement covariance shape, the method
 * falls back to isotropic `measurementNoiseVariance`.
 */
void EqVIOFilter::update(const VisionMeasurement& measurement,
                         const std::shared_ptr<const CameraModel>& camera,
                         const Matrix& outputGainMatrix) {
  if (measurement.empty()) return;

  const VisionMeasurement estimatedMeasurement = measureSystemState(stateEstimate(), camera);
  Vector yTilde = _measurementDifference(measurement, estimatedMeasurement);
  const Matrix Ct = EqFoutputMatrixC(snapshot_.xi0, snapshot_.X, measurement, camera,
                                     true);
  const Matrix Rused =
      (outputGainMatrix.rows() == Ct.rows() && outputGainMatrix.cols() == Ct.rows())
          ? outputGainMatrix
          : Matrix::Identity(Ct.rows(), Ct.rows()) * params_.measurementNoiseVariance;

  const Vector zhat = _measurementVector(estimatedMeasurement);
  const Vector z = _measurementVector(measurement);
  Base::updateWithVector(zhat, Ct, z, Rused, [this](const Vector& delta_xi) -> Vector {
    return liftInnovation(delta_xi, snapshot_.xi0);
  });
  syncFromBase();
}

}  // namespace eqvio
}  // namespace gtsam
