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
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

namespace gtsam {
namespace eqvio {

namespace {

/// Extract the SE(3) pose part from the A = (R, x, w) group block.
Pose3 APose(const VioGroup& X) {
  return Pose3(A_sensorKinematics(X).rotation(), A_sensorKinematics(X).x(0));
}

/// Extract the velocity-shift part w from the A = (R, x, w) group block.
Vector3 AW(const VioGroup& X) { return A_sensorKinematics(X).x(1); }

/// Build the Se2(3) block from rotation, translation, and velocity shift.
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

/// Validate measurement cardinality and return ids in deterministic map order.
std::vector<int> QIdsForMeasurement(const VioGroup& X,
                                    const VisionMeasurement& measurement) {
  if (N_landmarkCount(X) == 0) return {};
  if (measurement.size() != N_landmarkCount(X)) {
    throw std::invalid_argument(
        "outputGroupAction: measurement count must match group landmark count");
  }
  return measurementIds(measurement);
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

}  // namespace

/// Apply group action to the sensor sub-state.
SensorState sensorStateGroupAction(const VioGroup& X,
                                   const SensorState& sensor) {
  const Pose3 i0Ti1 = APose(X);
  const Pose3 c0Tc1 = B_cameraExtrinsics(X);
  const Vector3 velocityShift = AW(X);

  SensorState out;
  out.inputBias = sensor.inputBias + Beta_biasOffset(X);
  out.pose = sensor.pose.compose(i0Ti1);
  out.velocity = i0Ti1.rotation().unrotate(sensor.velocity - velocityShift);
  out.cameraOffset =
      i0Ti1.inverse().compose(sensor.cameraOffset).compose(c0Tc1);
  return out;
}

/// Apply group action to full state (sensor + landmark blocks).
State stateGroupAction(const VioGroup& X, const State& state) {
  if (N_landmarkCount(X) != state.n()) {
    throw std::invalid_argument(
        "stateGroupAction: group and state landmark counts do not match");
  }

  State out;
  out.sensor = sensorStateGroupAction(X, state.sensor);
  out.cameraLandmarks.resize(N_landmarkCount(X));

  // Transform the body-fixed landmarks
  for (size_t i = 0; i < N_landmarkCount(X); ++i) {
    out.cameraLandmarks[i].p =
        SOT3ApplyInverse(Q_landmarkTransforms(X)[i], state.cameraLandmarks[i].p);
    out.cameraLandmarks[i].id = state.cameraLandmarks[i].id;
  }

  return out;
}

/// Apply output action by rotating each bearing through landmark SOT3 terms.
VisionMeasurement outputGroupAction(const VioGroup& X,
                                    const VisionMeasurement& measurement,
                                    const std::shared_ptr<const CameraModel>& camera) {
  VisionMeasurement out;
  if (!camera) {
    throw std::invalid_argument("outputGroupAction: camera model is null");
  }

  const std::vector<int> qIds = QIdsForMeasurement(X, measurement);
  if (qIds.size() != N_landmarkCount(X)) {
    throw std::invalid_argument(
        "outputGroupAction: invalid Q-to-id mapping cardinality");
  }

  // Rotate each bearing through landmark SOT3 terms
  for (size_t i = 0; i < N_landmarkCount(X); ++i) {
    const int id = qIds[i];
    const auto it = measurement.find(id);
    if (it == measurement.end()) continue;

    const Vector3 bearing = undistortPoint(*camera, it->second);
    const Vector3 rotated =
        SOT3Rotation(Q_landmarkTransforms(X)[i]).matrix().transpose() * bearing;
    out[id] = camera->project2(rotated);
  }

  return out;
}

/// Continuous-time lift map in tangent coordinates of the symmetry group.
Vector liftVelocity(const State& state, const IMUInput& velocity) {
  const size_t N = state.n();
  Vector lift = Vector::Zero(21 + 4 * static_cast<int>(N));

  const SensorState& sensor = state.sensor;
  const IMUInput v_est = velocity - sensor.inputBias;

  Vector6 U_A;
  U_A << v_est.gyr, sensor.velocity;
  const Vector6 U_B = sensor.cameraOffset.inverse().AdjointMap() * U_A;
  const Vector3 u_w = -v_est.acc + sensor.gravityDir() * GRAVITY_CONSTANT;

  lift.segment<6>(0) = U_A;
  lift.segment<3>(6) = u_w;
  lift.segment<6>(9) << velocity.accBiasVel, velocity.gyrBiasVel;
  lift.segment<6>(15) = U_B;

  const Vector6 U_C = sensor.cameraOffset.inverse().AdjointMap() * U_A;
  const Vector3 omegaC = U_C.head<3>();
  const Vector3 vC = U_C.tail<3>();

  // Lift the landmark transform velocities
  for (size_t i = 0; i < N; ++i) {
    const Vector3 p = state.cameraLandmarks[i].p;
    Vector4 W;
    W.head<3>() = omegaC + Rot3::Hat(p) * vC / p.squaredNorm();
    W(3) = p.dot(vC) / p.squaredNorm();
    lift.segment<4>(21 + 4 * static_cast<int>(i)) = W;
  }

  return lift;
}

/// Discrete-time lift map returning an explicit group increment.
VioGroup liftVelocityDiscrete(const State& state, const IMUInput& velocity,
                              double dt) {
  const SensorState& sensor = state.sensor;
  const Pose3& wTi0 = sensor.pose;
  const Pose3& iTc = sensor.cameraOffset;
  const IMUInput v_est = velocity - sensor.inputBias;

  const Rot3 dR = Rot3::Expmap(dt * v_est.gyr);
  const Vector3 deltaWorld =
      dt * (wTi0.rotation() * sensor.velocity) +
      0.5 * dt * dt *
          (wTi0.rotation() * v_est.acc + Vector3(0, 0, -GRAVITY_CONSTANT));
  const Point3 deltaBody = wTi0.rotation().unrotate(deltaWorld);
  const Pose3 i0Ti1(dR, deltaBody);

  const Vector3 bodyVelocityDiff =
      v_est.acc - sensor.gravityDir() * GRAVITY_CONSTANT;
  const Vector3 velocityShift =
      sensor.velocity - (sensor.velocity + dt * bodyVelocityDiff);

  const Pose3 c0Tc1 = iTc.inverse().compose(i0Ti1).compose(iTc);
  const Pose3 c1Tc0 = c0Tc1.inverse();

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
  const Se23 A = MakeA(i0Ti1.rotation(), i0Ti1.translation(), velocityShift);
  return makeVioGroup(A, beta, c0Tc1, Q);
}

/// Integrate system dynamics over dt using IMU input.
State integrateSystemFunction(const State& state, const IMUInput& velocity,
                              double dt) {
  State out;
  const SensorState& sensor = state.sensor;
  const Pose3& wTi0 = sensor.pose;
  const Pose3& iTc = sensor.cameraOffset;
  const IMUInput v_est = velocity - sensor.inputBias;

  out.sensor.inputBias = Bias(
      sensor.inputBias.accelerometer() + dt * velocity.accBiasVel,
      sensor.inputBias.gyroscope() + dt * velocity.gyrBiasVel);

  const Rot3 dR = Rot3::Expmap(dt * v_est.gyr);
  const Vector3 deltaWorld =
      dt * (wTi0.rotation() * sensor.velocity) +
      0.5 * dt * dt *
          (wTi0.rotation() * v_est.acc + Vector3(0, 0, -GRAVITY_CONSTANT));
  const Point3 deltaBody = wTi0.rotation().unrotate(deltaWorld);
  const Pose3 i0Ti1(dR, deltaBody);

  out.sensor.pose = wTi0.compose(i0Ti1);

  const Vector3 inertialVelocityDiff =
      wTi0.rotation() * v_est.acc + Vector3(0, 0, -GRAVITY_CONSTANT);
  out.sensor.velocity = out.sensor.pose.rotation().unrotate(
      wTi0.rotation() * sensor.velocity + dt * inertialVelocityDiff);

  const Pose3 c1Tc0 = iTc.inverse().compose(i0Ti1.inverse()).compose(iTc);
  out.cameraLandmarks.resize(state.n());

  // Transform the body-fixed landmarks
  for (size_t i = 0; i < state.n(); ++i) {
    out.cameraLandmarks[i].p = c1Tc0.transformFrom(state.cameraLandmarks[i].p);
    out.cameraLandmarks[i].id = state.cameraLandmarks[i].id;
  }

  out.sensor.cameraOffset = iTc;
  return out;
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

/// Group action operator used by GroupAction and EquivariantFilter.
State VIOSymmetry::operator()(
    const State& xi, const VioGroup& X,
    OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_xi,
    OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X) const {
  const State y = stateGroupAction(X, xi);

  if (H_xi) {
    const int d = xi.dim();
    H_xi->setZero(d, d);

    const Pose3 i0Ti1 = APose(X);
    const Pose3 c0Tc1 = B_cameraExtrinsics(X);

    Matrix66 Hpose;
    xi.sensor.pose.compose(i0Ti1, Hpose, {});
    H_xi->block<6, 6>(6, 6) = Hpose;

    Matrix66 HtmpCamera, Hcamera;
    const Pose3 i1Tc0 =
        i0Ti1.inverse().compose(xi.sensor.cameraOffset, {}, HtmpCamera);
    (void)i1Tc0.compose(c0Tc1, Hcamera, {});
    H_xi->block<6, 6>(15, 15) = Hcamera * HtmpCamera;

    H_xi->block<6, 6>(0, 0).setIdentity();
    H_xi->block<3, 3>(12, 12) = i0Ti1.rotation().matrix().transpose();

    // Compute the Jacobian of the landmark transform velocities
    for (size_t i = 0; i < xi.n(); ++i) {
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
      H_xi->block<3, 3>(row, row) =
          (1.0 / SOT3Scale(Q_landmarkTransforms(X)[i])) *
          SOT3Rotation(Q_landmarkTransforms(X)[i]).matrix().transpose();
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
