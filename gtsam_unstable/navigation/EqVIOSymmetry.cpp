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
#include <gtsam/base/numericalDerivative.h>

#include <functional>
#include <stdexcept>

namespace gtsam {
namespace eqvio {

namespace {

Pose3 bTbPrimeFromGroup(const VioGroup& X) {
  const Se23& A = std::get<0>(decompose(X));
  return Pose3(A.rotation(), A.x(0));
}

Vector3 AW(const VioGroup& X) {
  const Se23& A = std::get<0>(decompose(X));
  return A.x(1);
}

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

constexpr double kStereoChartDiffDelta = 1e-6;

Vector2 E3ProjectSphere(const Vector3& eta) {
  static const Matrix23 I23 = Matrix23::Identity();
  static const Vector3 e3 = Vector3::UnitZ();
  return I23 * (eta - e3) / (1.0 - e3.dot(eta));
}

Vector3 E3ProjectSphereInv(const Vector2& y) {
  static const Vector3 e3 = Vector3::UnitZ();
  const Vector3 yBar = (Vector3() << y, 0.0).finished();
  return e3 + 2.0 / (yBar.squaredNorm() + 1.0) * (yBar - e3);
}

Matrix23 E3ProjectSphereDiff(const Vector3& eta) {
  const std::function<Vector2(const Vector3&)> f = [](const Vector3& x) {
    return E3ProjectSphere(x);
  };
  return gtsam::numericalDerivative11<Vector2, Vector3, 3>(
      f, eta, kStereoChartDiffDelta);
}

Matrix32 E3ProjectSphereInvDiff(const Vector2& y) {
  const std::function<Vector3(const Vector2&)> f = [](const Vector2& x) {
    return E3ProjectSphereInv(x);
  };
  return gtsam::numericalDerivative11<Vector3, Vector2, 2>(
      f, y, kStereoChartDiffDelta);
}

Matrix23 SphereChartStereoDiff0(const Vector3& pole) {
  const Rot3 sphereRot = RotationFromTwoVectors(-pole, Vector3::UnitZ());
  const Vector3 etaRotated = sphereRot.matrix() * pole;
  return E3ProjectSphereDiff(etaRotated) * sphereRot.matrix();
}

Matrix32 SphereChartStereoInvDiff0(const Vector3& pole) {
  const Rot3 sphereRot = RotationFromTwoVectors(-pole, Vector3::UnitZ());
  return sphereRot.matrix().transpose() * E3ProjectSphereInvDiff(Vector2::Zero());
}

Matrix3 ConvEucToInvDepth(const Point3& q0) {
  const double rho0 = 1.0 / q0.norm();
  const Vector3 y0 = q0 * rho0;

  Matrix3 M;
  M.block<2, 3>(0, 0) = rho0 * SphereChartStereoDiff0(y0) *
                        (Matrix3::Identity() - y0 * y0.transpose());
  M.block<1, 3>(2, 0) = -rho0 * rho0 * y0.transpose();
  return M;
}

Matrix3 ConvInvDepthToEuc(const Point3& q0) {
  const double rho0 = 1.0 / q0.norm();
  const Vector3 y0 = q0 * rho0;

  Matrix3 M;
  M.block<3, 2>(0, 0) = SphereChartStereoInvDiff0(y0) / rho0;
  M.block<3, 1>(0, 2) = -y0 / (rho0 * rho0);
  return M;
}

Matrix _EqFInputMatrixB(const VioGroup& X, const State& xi0);

Matrix _EqFStateMatrixA(const VioGroup& X, const State& xi0,
                        const IMUInput& imuVel) {
  const int N = static_cast<int>(xi0.n());
  Matrix A0t = Matrix::Zero(xi0.dim(), xi0.dim());

  const Matrix B = _EqFInputMatrixB(X, xi0);
  A0t.block(0, 0, xi0.dim(), 3) = -B.block(0, 3, xi0.dim(), 3);
  A0t.block(0, 3, xi0.dim(), 3) = -B.block(0, 0, xi0.dim(), 3);
  A0t.block<3, 3>(9, 12).setIdentity();
  A0t.block<3, 3>(12, 6) = -GRAVITY_CONSTANT * Rot3::Hat(xi0.sensor.gravityDir());

  const State xiHat = stateGroupAction(X, xi0);
  const IMUInput vEst = imuVel - xiHat.sensor.inputBias;
  Vector6 U_I;
  U_I << vEst.gyr, xiHat.sensor.velocity;

  const Pose3 bTbPrime = bTbPrimeFromGroup(X);
  const Vector6 commonTwist =
      xi0.sensor.cameraOffset.inverse().AdjointMap() *
      bTbPrime.AdjointMap() * U_I;
  A0t.block<6, 6>(15, 15) = Pose3::adjointMap(commonTwist);

  const Matrix3 R_IC = xiHat.sensor.cameraOffset.rotation().matrix();
  const Matrix3 R_bTbPrime = bTbPrime.rotation().matrix();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    A0t.block<3, 3>(SensorState::CompDim + 3 * i, 12) =
        -ConvEucToInvDepth(q0) * Qhat_i * R_IC.transpose() *
        R_bTbPrime.transpose();
  }

  const Matrix66 commonTerm =
      B_cameraExtrinsics(X).inverse().AdjointMap() * Pose3::adjointMap(commonTwist);
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    Matrix36 temp;
    temp << Rot3::Hat(q0) *
                SOT3Rotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)])
                    .matrix(),
        -SOT3Scale(Q_landmarkTransforms(X)[static_cast<size_t>(i)]) *
            SOT3Rotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)])
                .matrix();
    A0t.block<3, 6>(SensorState::CompDim + 3 * i, 15) =
        ConvEucToInvDepth(q0) * temp * commonTerm;
  }

  const Vector6 U_C = xiHat.sensor.cameraOffset.inverse().AdjointMap() * U_I;
  const Vector3 v_C = U_C.tail<3>();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    const Point3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)].p;
    const Matrix3 A_qi =
        -Qhat_i *
        (Rot3::Hat(qhat_i) * Rot3::Hat(v_C) - 2.0 * v_C * qhat_i.transpose() +
         qhat_i * v_C.transpose()) *
        Qhat_i.inverse() * (1.0 / qhat_i.squaredNorm());
    A0t.block<3, 3>(SensorState::CompDim + 3 * i,
                    SensorState::CompDim + 3 * i) =
        ConvEucToInvDepth(q0) * A_qi * ConvInvDepthToEuc(q0);
  }

  return A0t;
}

Matrix _EqFInputMatrixB(const VioGroup& X, const State& xi0) {
  const int N = static_cast<int>(xi0.n());
  Matrix Bt = Matrix::Zero(xi0.dim(), IMUInput::CompDim);

  const State xiHat = stateGroupAction(X, xi0);
  const Pose3 bTbPrime = bTbPrimeFromGroup(X);

  Bt.block<3, 3>(0, 9).setIdentity();
  Bt.block<3, 3>(3, 6).setIdentity();

  const Matrix3 R_bTbPrime = bTbPrime.rotation().matrix();
  Bt.block<3, 3>(6, 0) = R_bTbPrime;
  Bt.block<3, 3>(9, 0) = Rot3::Hat(bTbPrime.translation()) * R_bTbPrime;
  Bt.block<3, 3>(12, 0) = R_bTbPrime * Rot3::Hat(xiHat.sensor.velocity);
  Bt.block<3, 3>(12, 3) = R_bTbPrime;

  const Matrix3 RT_IC = xiHat.sensor.cameraOffset.rotation().matrix().transpose();
  const Point3 x_IC = xiHat.sensor.cameraOffset.translation();
  for (int i = 0; i < N; ++i) {
    const Point3 q0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    const Matrix3 Qhat_i =
        SOT3ScaledRotation(Q_landmarkTransforms(X)[static_cast<size_t>(i)]);
    const Point3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)].p;
    Bt.block<3, 3>(SensorState::CompDim + 3 * i, 0) =
        ConvEucToInvDepth(q0) *
        Qhat_i * (Rot3::Hat(qhat_i) * RT_IC + RT_IC * Rot3::Hat(x_IC));
  }

  return Bt;
}

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

Matrix23 _EqFoutputMatrixCiStar(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera, const Point2& y) {
  const double r0 = q0.norm();
  const Vector3 y0 = q0 / r0;
  Matrix3 ind2euc;
  ind2euc.block<3, 2>(0, 0) = r0 * SphereChartStereoInvDiff0(y0);
  ind2euc.block<3, 1>(0, 2) = -r0 * q0;
  return _EqFoutputMatrixCiStarBase(q0, QHat, camera, y) * ind2euc;
}

Vector _liftInnovation(const Vector& totalInnovation, const State& xi0) {
  if (totalInnovation.size() != xi0.dim()) {
    throw std::invalid_argument(
        "liftInnovation: innovation dimension mismatch");
  }

  const int N = static_cast<int>(xi0.n());
  Vector lift = Vector::Zero(21 + 4 * N);

  lift.segment<6>(9) = totalInnovation.segment<6>(0);
  lift.segment<6>(0) = totalInnovation.segment<6>(6);

  const Vector3 gammaV = totalInnovation.segment<3>(12);
  lift.segment<3>(6) = -gammaV - Rot3::Hat(lift.segment<3>(0)) * xi0.sensor.velocity;

  lift.segment<6>(15) =
      totalInnovation.segment<6>(15) +
      xi0.sensor.cameraOffset.inverse().AdjointMap() * lift.segment<6>(0);

  for (int i = 0; i < N; ++i) {
    const Point3 qi0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    const Vector3 gammaQi0 =
        ConvInvDepthToEuc(qi0) *
        totalInnovation.segment<3>(SensorState::CompDim + 3 * i);

    lift.segment<3>(21 + 4 * i) = -qi0.cross(gammaQi0) / qi0.squaredNorm();
    lift(21 + 4 * i + 3) = -qi0.dot(gammaQi0) / qi0.squaredNorm();
  }

  return lift;
}

}  // namespace

Matrix EqFStateMatrixA(const VioGroup& X, const State& xi0,
                       const IMUInput& imuVel) {
  return _EqFStateMatrixA(X, xi0, imuVel);
}

Matrix EqFInputMatrixB(const VioGroup& X, const State& xi0) {
  return _EqFInputMatrixB(X, xi0);
}

Matrix23 EqFoutputMatrixCiStar(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera, const Point2& y) {
  return _EqFoutputMatrixCiStar(q0, QHat, camera, y);
}

Matrix23 EqFoutputMatrixCi(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixCi: null camera");
  }
  const Vector3 qHat = SOT3ApplyInverse(QHat, q0);
  const Point2 yHat = camera->project2(qHat);
  return EqFoutputMatrixCiStar(q0, QHat, camera, yHat);
}

Matrix EqFoutputMatrixC(
    const State& xi0, const VioGroup& X, const VisionMeasurement& y,
    const std::shared_ptr<const CameraModel>& camera, bool useEquivariance) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixC: null camera");
  }
  const int M = static_cast<int>(xi0.n());
  const std::vector<int> yIds = measurementIds(y);
  const int N = static_cast<int>(yIds.size());
  const LandmarkGroup& Q = std::get<3>(decompose(X));

  Matrix C = Matrix::Zero(2 * N, SensorState::CompDim + Landmark::CompDim * M);

  for (int i = 0; i < M; ++i) {
    const int idNum = xi0.cameraLandmarks[static_cast<size_t>(i)].id;
    const auto itY = std::find(yIds.begin(), yIds.end(), idNum);
    if (itY == yIds.end()) continue;

    const size_t k = static_cast<size_t>(i);
    const int j = static_cast<int>(std::distance(yIds.begin(), itY));
    const Point3& qi0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;
    const SOT3& Qk = Q[k];

    const Matrix23 Ci = useEquivariance
                            ? EqFoutputMatrixCiStar(qi0, Qk, camera, y.at(idNum))
                            : EqFoutputMatrixCi(qi0, Qk, camera);
    if (!Ci.array().isFinite().all()) {
      const Point3 qHat = SOT3ApplyInverse(Qk, qi0);
      const Point2 yObs = y.at(idNum);
      const Vector3 yUnd = undistortPoint(*camera, yObs);
      throw std::runtime_error(
          "EqFoutputMatrixC: non-finite Ci for id " +
          std::to_string(idNum) + ", qi0_norm=" + std::to_string(qi0.norm()) +
          ", qHat_norm=" + std::to_string(qHat.norm()) +
          ", Q_scale=" + std::to_string(SOT3Scale(Qk)) +
          ", yUnd_norm=" + std::to_string(yUnd.norm()));
    }
    C.block<2, 3>(2 * j, SensorState::CompDim + 3 * i) = Ci;
  }

  if (!C.array().isFinite().all()) {
    throw std::runtime_error("EqFoutputMatrixC produced NaN/Inf");
  }
  return C;
}

Vector liftInnovation(const Vector& totalInnovation, const State& xi0) {
  return _liftInnovation(totalInnovation, xi0);
}

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
    out.cameraLandmarks[i].p =
        SOT3ApplyInverse(Q[i], state.cameraLandmarks[i].p);
    out.cameraLandmarks[i].id = state.cameraLandmarks[i].id;
  }

  return out;
}

VioGroup liftVelocityDiscrete(const State& state, const IMUInput& velocity,
                              double dt) {
  const SensorState& sensor = state.sensor;
  const IMUInput v_est = velocity - sensor.inputBias;

  const Rot3 dR = Rot3::Expmap(dt * v_est.gyr);
  const Vector3 dXWorld =
      dt * (sensor.pose.rotation() * sensor.velocity) +
      0.5 * dt * dt *
          (sensor.pose.rotation() * v_est.acc + Vector3(0, 0, -GRAVITY_CONSTANT));
  const Point3 dXBody = sensor.pose.rotation().unrotate(dXWorld);
  const Pose3 b0Tb1(dR, dXBody);

  const Vector3 bodyVelocityDiff =
      v_est.acc - sensor.gravityDir() * GRAVITY_CONSTANT;
  const Vector3 w = sensor.velocity - (sensor.velocity + dt * bodyVelocityDiff);

  const Pose3 c0Tc1 = sensor.cameraOffset.inverse().compose(b0Tb1).compose(
      sensor.cameraOffset);
  const Pose3 c1Tc0 =
      sensor.cameraOffset.inverse().compose(b0Tb1.inverse()).compose(
          sensor.cameraOffset);

  // Construct the landmark transform velocities
  std::vector<SOT3> q;
  q.resize(state.n());
  for (size_t i = 0; i < state.n(); ++i) {
    const Landmark& lm0 = state.cameraLandmarks[i];
    Landmark lm1;
    lm1.id = lm0.id;
    lm1.p = c1Tc0.transformFrom(lm0.p);

    const Rot3 R = RotationFromTwoVectors(lm1.p, lm0.p);
    const double a = lm0.p.norm() / lm1.p.norm();
    q[i] = MakeSOT3(SO3(R.matrix()), a);
  }

  LandmarkGroup Q(q);
  const Bias beta(dt * velocity.accBiasVel, dt * velocity.gyrBiasVel);
  const Se23 b0Tb1Kinematics = MakeA(b0Tb1.rotation(), b0Tb1.translation(), w);
  return makeVioGroup(b0Tb1Kinematics, beta, c0Tc1, Q);
}

/// Predict ideal normalized image measurements from current landmark state.
VisionMeasurement measureSystemState(
    const State& state, const std::shared_ptr<const CameraModel>& camera) {
  if (!camera) {
    throw std::invalid_argument("measureSystemState: camera model is null");
  }

  // Project each landmark through the camera model
  VisionMeasurement out;
  for (const Landmark& lm : state.cameraLandmarks) {
    out[lm.id] = camera->project2(lm.p);
  }
  return out;
}

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
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
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
