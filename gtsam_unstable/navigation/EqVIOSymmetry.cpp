/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOSymmetry.cpp
 * @brief EqVIO symmetry actions and lift helpers.
 * @author Rohan Bansal
 */

#include <gtsam_unstable/navigation/EqVIOSymmetry.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace gtsam {
namespace eqvio {

namespace {

/// Extract body transform component `bTb'` from full group element.
Pose3 bTbPrimeFromGroup(const VioGroup& X) {
  const Se23& A = std::get<0>(decompose(X));
  return Pose3(A.rotation(), A.x(0));
}

/// Extract translational velocity offset `w` from Se23 block.
Vector3 AW(const VioGroup& X) {
  const Se23& A = std::get<0>(decompose(X));
  return A.x(1);
}

/// Build Se23 element from rotation, translation, and velocity columns.
Se23 MakeA(const Rot3& R, const Point3& x0, const Vector3& w) {
  Se23::Matrix3K x;
  x.col(0) = x0;
  x.col(1) = w;
  return Se23(R, x);
}

/// Construct the shortest-arc rotation that maps first vector to second vector.
Rot3 RotationFromTwoVectors(const Vector3& from, const Vector3& to) {
  gtsam::Quaternion q;
  q.setFromTwoVectors(from, to);
  return Rot3(q);
}

/// Numerical derivative of the state action with respect to group coordinates.
Matrix NumericalDerivativeActionWrtGroup(
    const std::function<State(const VioGroup&)>& f, const VioGroup& X,
    const State& y0, double h = 1e-6) {
  const int n = static_cast<int>(Dim_groupTangent(X));
  const int m = y0.dim();
  Matrix H = Matrix::Zero(m, n);

  for (int j = 0; j < n; ++j) {
    Vector dx = Vector::Zero(n);
    dx(j) = h;
    const State yPlus = f(X.retract(dx));
    dx(j) = -h;
    const State yMinus = f(X.retract(dx));

    H.col(j) =
        (y0.localCoordinates(yPlus) - y0.localCoordinates(yMinus)) / (2.0 * h);
  }

  return H;
}

/// Jacobian of stereographic chart in closed form.
Matrix23 E3ProjectSphereDiff(const Vector3& eta) {
  const double denom = 1.0 - eta.z();
  const double inv = 1.0 / denom;
  const double inv2 = inv * inv;

  Matrix23 J;
  J << inv, 0.0, eta.x() * inv2, 0.0, inv, eta.y() * inv2;
  return J;
}

/// Jacobian of inverse stereographic chart in closed form.
Matrix32 E3ProjectSphereInvDiff(const Vector2& y) {
  static const Matrix32 I32 =
      (Matrix32() << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0).finished();

  const double s = y.squaredNorm() + 1.0;
  const double alpha = 2.0 / s;
  const double beta = -4.0 / (s * s);
  const Vector3 yMinusE3(y.x(), y.y(), -1.0);

  return alpha * I32 + beta * (yMinusE3 * y.transpose());
}

/// Differential of sphere chart at the nominal point associated with `pole`.
Matrix23 SphereChartStereoDiff0(const Vector3& pole) {
  const Rot3 sphereRot = RotationFromTwoVectors(-pole, Vector3::UnitZ());
  const Vector3 etaRotated = sphereRot.matrix() * pole;
  return E3ProjectSphereDiff(etaRotated) * sphereRot.matrix();
}

/// Differential of inverse sphere chart at zero in local coordinates.
Matrix32 SphereChartStereoInvDiff0(const Vector3& pole) {
  const Rot3 sphereRot = RotationFromTwoVectors(-pole, Vector3::UnitZ());
  return sphereRot.matrix().transpose() *
         E3ProjectSphereInvDiff(Vector2::Zero());
}

/// Convert Euclidean landmark perturbation to inverse-depth local coordinates.
Matrix3 ConvEucToInvDepth(const Point3& q0) {
  const double rho0 = 1.0 / q0.norm();
  const Vector3 y0 = q0 * rho0;

  Matrix3 M;
  M.block<2, 3>(0, 0) = rho0 * SphereChartStereoDiff0(y0) *
                        (Matrix3::Identity() - y0 * y0.transpose());
  M.block<1, 3>(2, 0) = -rho0 * rho0 * y0.transpose();
  return M;
}

/// Convert inverse-depth local perturbation back to Euclidean perturbation.
Matrix3 ConvInvDepthToEuc(const Point3& q0) {
  const double rho0 = 1.0 / q0.norm();
  const Vector3 y0 = q0 * rho0;

  Matrix3 M;
  M.block<3, 2>(0, 0) = SphereChartStereoInvDiff0(y0) / rho0;
  M.block<3, 1>(0, 2) = -y0 / (rho0 * rho0);
  return M;
}

/// Base per-feature output Jacobian before inverse-depth chart conversion.
Matrix23 _EqFoutputMatrixCiStarBase(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera, const Point2& y) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixCiStar_base: null camera");
  }

  using Matrix43 = Eigen::Matrix<double, 4, 3>;
  using Matrix34 = Eigen::Matrix<double, 3, 4>;
  using Matrix24 = Eigen::Matrix<double, 2, 4>;

  const Vector3 qHat = SOT3ApplyInverse(QHat, q0);
  const Vector3 yHat = qHat.normalized();

  Matrix43 m2g = Matrix43::Zero();
  m2g.block<3, 3>(0, 0) = -Rot3::Hat(q0);
  m2g.row(3) = -q0.transpose();
  m2g /= q0.squaredNorm();

  const auto DRho = [&camera](const Vector3& yVec) -> Matrix24 {
    Matrix34 DRhoVec = Matrix34::Zero();
    DRhoVec.block<3, 3>(0, 0) = Rot3::Hat(yVec);
    return projectionJacobian(*camera, yVec) * DRhoVec;
  };

  const Vector3 yTru = undistortPoint(*camera, y);
  const Matrix24 drhoSym = 0.5 * (DRho(yTru) + DRho(yHat));
  const Matrix44 adjQInv = QHat.inverse().AdjointMap();
  return drhoSym * adjQInv * m2g;
}

}  // namespace

/// Lift innovation from chart coordinates to group tangent coordinates.
Vector liftInnovation(const Vector& totalInnovation, const State& xi0) {
  if (totalInnovation.size() != xi0.dim()) {
    throw std::invalid_argument(
        "liftInnovation: innovation dimension mismatch");
  }

  const int N = static_cast<int>(xi0.n());
  Vector lift = Vector::Zero(21 + 4 * N);

  lift.segment<6>(9) = totalInnovation.segment<6>(0);
  lift.segment<6>(0) = totalInnovation.segment<6>(6);

  const Vector3 gammaV = totalInnovation.segment<3>(12);
  lift.segment<3>(6) =
      -gammaV - Rot3::Hat(lift.segment<3>(0)) * xi0.sensor.velocity;

  lift.segment<6>(15) =
      totalInnovation.segment<6>(15) +
      xi0.sensor.cameraOffset.inverse().AdjointMap() * lift.segment<6>(0);

  for (int i = 0; i < N; ++i) {
    const Point3 qi0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    const Vector3 gammaQi0 =
        ConvInvDepthToEuc(qi0) * totalInnovation.segment<3>(21 + 3 * i);

    lift.segment<3>(21 + 4 * i) = -qi0.cross(gammaQi0) / qi0.squaredNorm();
    lift(21 + 4 * i + 3) = -qi0.dot(gammaQi0) / qi0.squaredNorm();
  }

  return lift;
}

/// EqF state matrix construction in inverse-depth coordinates.
Matrix EqFStateMatrixA(const VioGroup& X, const State& xi0,
                       const IMUInput& imuVel) {
  const int N = static_cast<int>(xi0.n());
  Matrix A0t = Matrix::Zero(xi0.dim(), xi0.dim());

  const Matrix B = EqFInputMatrixB(X, xi0);
  A0t.block(0, 0, xi0.dim(), 3) = -B.block(0, 3, xi0.dim(), 3);
  A0t.block(0, 3, xi0.dim(), 3) = -B.block(0, 0, xi0.dim(), 3);
  A0t.block<3, 3>(9, 12).setIdentity();
  A0t.block<3, 3>(12, 6) =
      -GRAVITY_CONSTANT * Rot3::Hat(xi0.sensor.gravityDir());

  const State xiHat = stateGroupAction(X, xi0);
  const IMUInput vEst = imuVel - xiHat.sensor.inputBias;
  Vector6 U_I;
  U_I << vEst.gyr, xiHat.sensor.velocity;

  const Pose3 bTbPrime = bTbPrimeFromGroup(X);
  const Vector6 commonTwist = xi0.sensor.cameraOffset.inverse().AdjointMap() *
                              bTbPrime.AdjointMap() * U_I;
  A0t.block<6, 6>(15, 15) = Pose3::adjointMap(commonTwist);

  const Matrix3 R_IC = xiHat.sensor.cameraOffset.rotation().matrix();
  const Matrix3 R_bTbPrime = bTbPrime.rotation().matrix();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    A0t.block<3, 3>(21 + 3 * i, 12) = -ConvEucToInvDepth(q0) * Qhat_i *
                                      R_IC.transpose() * R_bTbPrime.transpose();
  }

  const Matrix66 commonTerm = B_cameraExtrinsics(X).inverse().AdjointMap() *
                              Pose3::adjointMap(commonTwist);
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    Matrix36 temp;
    temp << Rot3::Hat(q0) *
                SOT3Rotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)])
                    .matrix(),
        -SOT3Scale(Q_landmarkTransforms(X)[static_cast<size_t>(i)]) *
            SOT3Rotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)])
                .matrix();
    A0t.block<3, 6>(21 + 3 * i, 15) = ConvEucToInvDepth(q0) * temp * commonTerm;
  }

  const Vector6 U_C = xiHat.sensor.cameraOffset.inverse().AdjointMap() * U_I;
  const Vector3 v_C = U_C.tail<3>();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    const Point3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)];
    const Matrix3 A_qi =
        -Qhat_i *
        (Rot3::Hat(qhat_i) * Rot3::Hat(v_C) - 2.0 * v_C * qhat_i.transpose() +
         qhat_i * v_C.transpose()) *
        Qhat_i.inverse() * (1.0 / qhat_i.squaredNorm());
    A0t.block<3, 3>(21 + 3 * i, 21 + 3 * i) =
        ConvEucToInvDepth(q0) * A_qi * ConvInvDepthToEuc(q0);
  }

  return A0t;
}

/// EqF input matrix construction in inverse-depth coordinates.
Matrix EqFInputMatrixB(const VioGroup& X, const State& xi0) {
  const int N = static_cast<int>(xi0.n());
  Matrix Bt = Matrix::Zero(xi0.dim(), 12);

  const State xiHat = stateGroupAction(X, xi0);
  const Pose3 bTbPrime = bTbPrimeFromGroup(X);

  Bt.block<3, 3>(0, 9).setIdentity();
  Bt.block<3, 3>(3, 6).setIdentity();

  const Matrix3 R_bTbPrime = bTbPrime.rotation().matrix();
  Bt.block<3, 3>(6, 0) = R_bTbPrime;
  Bt.block<3, 3>(9, 0) = Rot3::Hat(bTbPrime.translation()) * R_bTbPrime;
  Bt.block<3, 3>(12, 0) = R_bTbPrime * Rot3::Hat(xiHat.sensor.velocity);
  Bt.block<3, 3>(12, 3) = R_bTbPrime;

  const Matrix3 RT_IC =
      xiHat.sensor.cameraOffset.rotation().matrix().transpose();
  const Point3 x_IC = xiHat.sensor.cameraOffset.translation();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    const Point3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)];
    Bt.block<3, 3>(21 + 3 * i, 0) =
        ConvEucToInvDepth(q0) * Qhat_i *
        (Rot3::Hat(qhat_i) * RT_IC + RT_IC * Rot3::Hat(x_IC));
  }

  return Bt;
}

/// Per-feature equivariant output Jacobian in inverse-depth coordinates.
Matrix23 EqFoutputMatrixCiStar(const Point3& q0, const SOT3& QHat,
                               const std::shared_ptr<const CameraModel>& camera,
                               const Point2& y) {
  const double r0 = q0.norm();
  const Vector3 y0 = q0 / r0;
  Matrix3 ind2euc;
  ind2euc.block<3, 2>(0, 0) = r0 * SphereChartStereoInvDiff0(y0);
  ind2euc.block<3, 1>(0, 2) = -r0 * q0;
  return _EqFoutputMatrixCiStarBase(q0, QHat, camera, y) * ind2euc;
}

/// Compute per-feature output Jacobian from predicted measurement.
Matrix23 EqFoutputMatrixCi(const Point3& q0, const SOT3& QHat,
                           const std::shared_ptr<const CameraModel>& camera) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixCi: null camera");
  }
  const Vector3 qHat = SOT3ApplyInverse(QHat, q0);
  const Point2 yHat = camera->project2(qHat);
  return EqFoutputMatrixCiStar(q0, QHat, camera, yHat);
}

/// Assemble full stacked output matrix for currently observed landmarks.
Matrix EqFoutputMatrixC(const State& xi0, const KeyVector& landmarkIds,
                        const VioGroup& X, const VisionMeasurement& y,
                        const std::shared_ptr<const CameraModel>& camera,
                        bool useEquivariance) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixC: null camera");
  }
  const int M = static_cast<int>(xi0.n());
  if (landmarkIds.size() != xi0.n()) {
    throw std::invalid_argument("EqFoutputMatrixC: landmark id count mismatch");
  }
  const KeyVector yIds = measurementIds(y);
  const int N = static_cast<int>(yIds.size());
  const LandmarkGroup& Q = std::get<3>(decompose(X));

  FastMap<Key, int> rowIndexByKey;
  for (int j = 0; j < N; ++j) {
    rowIndexByKey[yIds[static_cast<size_t>(j)]] = j;
  }

  Matrix C = Matrix::Zero(2 * N, 21 + 3 * M);

  for (int i = 0; i < M; ++i) {
    const Key idNum = landmarkIds[static_cast<size_t>(i)];
    const auto itRow = rowIndexByKey.find(idNum);
    if (itRow == rowIndexByKey.end()) continue;

    const size_t k = static_cast<size_t>(i);
    const int j = itRow->second;
    const Point3& qi0 = xi0.cameraLandmarks[static_cast<size_t>(i)];
    const SOT3& Qk = Q[k];

    const Matrix23 Ci =
        useEquivariance ? EqFoutputMatrixCiStar(qi0, Qk, camera, y.at(idNum))
                        : EqFoutputMatrixCi(qi0, Qk, camera);
    if (!Ci.array().isFinite().all()) {
      const Point3 qHat = SOT3ApplyInverse(Qk, qi0);
      const Point2 yObs = y.at(idNum);
      const Vector3 yUnd = undistortPoint(*camera, yObs);
      throw std::runtime_error("EqFoutputMatrixC: non-finite Ci for id " +
                               std::to_string(idNum) +
                               ", qi0_norm=" + std::to_string(qi0.norm()) +
                               ", qHat_norm=" + std::to_string(qHat.norm()) +
                               ", Q_scale=" + std::to_string(SOT3Scale(Qk)) +
                               ", yUnd_norm=" + std::to_string(yUnd.norm()));
    }
    C.block<2, 3>(2 * j, 21 + 3 * i) = Ci;
  }

  if (!C.array().isFinite().all()) {
    throw std::runtime_error("EqFoutputMatrixC produced NaN/Inf");
  }
  return C;
}

/// Apply right action on sensor block only.
SensorState sensorStateGroupAction(const VioGroup& X,
                                   const SensorState& sensor) {
  const Bias& Beta = std::get<1>(decompose(X));
  const Pose3& cTcPrime = std::get<2>(decompose(X));
  const Pose3 bTbPrime = bTbPrimeFromGroup(X);
  const Vector3 w = AW(X);

  SensorState out;
  out.inputBias = sensor.inputBias + Beta;
  out.pose = sensor.pose.compose(bTbPrime);
  out.velocity = bTbPrime.rotation().unrotate(sensor.velocity - w);
  out.cameraOffset =
      bTbPrime.inverse().compose(sensor.cameraOffset).compose(cTcPrime);
  return out;
}

/// Apply right action on full state with sensor and landmark components.
State stateGroupAction(const VioGroup& X, const State& state) {
  const LandmarkGroup& Q = std::get<3>(decompose(X));
  if (Q.size() != state.n()) {
    throw std::invalid_argument(
        "stateGroupAction: group and state landmark counts do not match");
  }

  State out;
  out.sensor = sensorStateGroupAction(X, state.sensor);
  out.cameraLandmarks.resize(Q.size());

  for (size_t i = 0; i < Q.size(); ++i) {
    out.cameraLandmarks[i] = SOT3ApplyInverse(Q[i], state.cameraLandmarks[i]);
  }

  return out;
}

/// Discrete-time lifted system increment from IMU input and current state.
VioGroup liftVelocityDiscrete(const State& state, const IMUInput& velocity,
                              double dt) {
  const SensorState& sensor = state.sensor;
  const IMUInput v_est = velocity - sensor.inputBias;

  const Rot3 dR = Rot3::Expmap(dt * v_est.gyr);
  const Vector3 dXWorld = dt * (sensor.pose.rotation() * sensor.velocity) +
                          0.5 * dt * dt *
                              (sensor.pose.rotation() * v_est.acc +
                               Vector3(0, 0, -GRAVITY_CONSTANT));
  const Point3 dXBody = sensor.pose.rotation().unrotate(dXWorld);
  const Pose3 b0Tb1(dR, dXBody);

  const Vector3 bodyVelocityDiff =
      v_est.acc - sensor.gravityDir() * GRAVITY_CONSTANT;
  const Vector3 w = sensor.velocity - (sensor.velocity + dt * bodyVelocityDiff);

  const Pose3 c0Tc1 =
      sensor.cameraOffset.inverse().compose(b0Tb1).compose(sensor.cameraOffset);
  const Pose3 c1Tc0 = sensor.cameraOffset.inverse()
                          .compose(b0Tb1.inverse())
                          .compose(sensor.cameraOffset);

  // Construct the landmark transform velocities
  std::vector<SOT3> q;
  q.resize(state.n());
  for (size_t i = 0; i < state.n(); ++i) {
    const Point3 p0 = state.cameraLandmarks[i];
    const Point3 p1 = c1Tc0.transformFrom(p0);

    const Rot3 R = RotationFromTwoVectors(p1, p0);
    const double a = p0.norm() / p1.norm();
    q[i] = MakeSOT3(SO3(R.matrix()), a);
  }

  LandmarkGroup Q(q);
  const Bias beta(dt * velocity.accBiasVel, dt * velocity.gyrBiasVel);
  const Se23 b0Tb1Kinematics = MakeA(b0Tb1.rotation(), b0Tb1.translation(), w);
  return makeVioGroup(b0Tb1Kinematics, beta, c0Tc1, Q);
}

/**
 * @brief Predict ideal normalized image measurements from current state.
 * @throws std::invalid_argument if `camera` is null or id count mismatches
 * landmarks.
 */
VisionMeasurement measureSystemState(
    const State& state, const KeyVector& landmarkIds,
    const std::shared_ptr<const CameraModel>& camera) {
  if (!camera) {
    throw std::invalid_argument("measureSystemState: camera model is null");
  }
  if (landmarkIds.size() != state.n()) {
    throw std::invalid_argument(
        "measureSystemState: landmark id count mismatch");
  }

  VisionMeasurement out;
  for (size_t i = 0; i < state.n(); ++i) {
    out[landmarkIds[i]] = camera->project2(state.cameraLandmarks[i]);
  }
  return out;
}

/// Evaluate symmetry action and optionally compute Jacobians w.r.t. state and
/// group.
State Symmetry::operator()(
    const State& xi, const VioGroup& X,
    OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_xi,
    OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X) const {
  const State y = stateGroupAction(X, xi);

  if (H_xi) {
    const int d = xi.dim();
    H_xi->setZero(d, d);

    const Pose3 bTbPrime = bTbPrimeFromGroup(X);

    Matrix66 Hpose;
    xi.sensor.pose.compose(bTbPrime, Hpose, {});
    H_xi->block<6, 6>(6, 6) = Hpose;

    Matrix66 HcTbPrimeComp, Hcamera;
    const Pose3 cTbPrime =
        bTbPrime.inverse().compose(xi.sensor.cameraOffset, {}, HcTbPrimeComp);
    const Pose3& cTcPrime = std::get<2>(decompose(X));
    const LandmarkGroup& Q = std::get<3>(decompose(X));
    cTbPrime.compose(cTcPrime, Hcamera, {});
    H_xi->block<6, 6>(15, 15) = Hcamera * HcTbPrimeComp;

    H_xi->block<6, 6>(0, 0).setIdentity();
    H_xi->block<3, 3>(12, 12) = bTbPrime.rotation().matrix().transpose();

    // Compute the Jacobian of the landmark transform velocities
    for (size_t i = 0; i < xi.n(); ++i) {
      const int row = 21 + 3 * static_cast<int>(i);
      H_xi->block<3, 3>(row, row) =
          (1.0 / SOT3Scale(Q[i])) * SOT3Rotation(Q[i]).matrix().transpose();
    }
  }

  if (H_X) {
    auto f = [&xi, this](const VioGroup& g) { return this->operator()(xi, g); };
    *H_X = NumericalDerivativeActionWrtGroup(f, X, y);
  }

  return y;
}

}  // namespace eqvio
}  // namespace gtsam
