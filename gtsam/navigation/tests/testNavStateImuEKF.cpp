/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 *
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testNavStateImuEKF.cpp
 * @brief Unit test for NavStateImuEKF, as well as dynamics used.
 * @date April 26, 2025
 * @authors Scott Baker, Matt Kielo, Frank Dellaert
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/NavStateImuEKF.h>
#include <gtsam/navigation/PreintegrationParams.h>

using namespace gtsam;

TEST(NavStateImuEKF, DefaultProcessNoiseFromParams) {
  // GIVEN params with specific covariances
  auto params = PreintegrationParams::MakeSharedU(9.81);
  Matrix3 Cg = (Matrix3() << 0.01, 0, 0, 0, 0.02, 0, 0, 0, 0.03).finished();
  Matrix3 Ci = (Matrix3() << 0.001, 0, 0, 0, 0.002, 0, 0, 0, 0.003).finished();
  Matrix3 Ca = (Matrix3() << 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.3).finished();
  params->setGyroscopeCovariance(Cg);
  params->setIntegrationCovariance(Ci);
  params->setAccelerometerCovariance(Ca);

  Rot3 R0 = Rot3::RzRyRx(0.0, 0.0, 0.0);
  Point3 p0(0, 0, 0);
  Vector3 v0(0, 0, 0);
  NavState X0(R0, p0, v0);
  Matrix9 P0 = I_9x9 * 0.01;
  NavStateImuEKF ekf(X0, P0, params);

  Matrix9 Q = Matrix9::Zero();
  Q.block<3, 3>(0, 0) = Cg;
  Q.block<3, 3>(3, 3) = Ci;
  Q.block<3, 3>(6, 6) = Ca;
  EXPECT(assert_equal(Q, ekf.processNoise(), 1e-12));
}

TEST(NavStateImuEKF, DynamicsJacobian) {
  // GIVEN a nontrivial NavState
  Rot3 R = Rot3::RzRyRx(0.1, -0.2, 0.3);
  Point3 p(0.5, -0.4, 0.3);
  Vector3 v(0.2, -0.1, 0.05);
  NavState X(R, p, v);

  // Controls and gravity
  Vector3 gyro(0.3, -0.2, 0.1);
  Vector3 accel(0.5, -0.3, 0.2);
  auto params = PreintegrationParams::MakeSharedU(9.81);

  // Analytic Jacobian using free dynamics with explicit gravity
  Matrix9 H;
  Vector9 xi = navStateImuDynamics(X, gyro, accel, params->getGravity(), H);
  (void)xi;  // silence unused warning

  // Numerical Jacobian w.r.t. the state using a typed std::function
  std::function<Vector9(const NavState&)> f =
      [&](const NavState& Xq) -> Vector9 {
    return navStateImuDynamics(Xq, gyro, accel, params->getGravity());
  };
  Matrix9 Hnum = numericalDerivative11<Vector9, NavState>(f, X);

  EXPECT(assert_equal(Hnum, H, 1e-6));
}

TEST(NavStateImuEKF, PredictMeanJacobian) {
  // GIVEN initial state and EKF
  Rot3 R0 = Rot3::RzRyRx(0.2, -0.1, 0.3);
  Point3 p0(0.1, 0.2, -0.3);
  Vector3 v0(-0.2, 0.4, 0.1);
  NavState X0(R0, p0, v0);
  Matrix9 P0 = I_9x9 * 0.01;
  auto params = PreintegrationParams::MakeSharedU(9.81);

  // Controls
  Vector3 gyro(0.1, 0.2, -0.1);
  Vector3 accel(0.3, -0.2, 0.4);
  double dt = 0.05;

  // We compare the analytic state transition Jacobian A from predictMean
  // to a numerical derivative of the state->state mapping g(X).
  Matrix9 A;
  // Single dynamics lambda (state-dependent only), like StateAndControl test
  auto f = [&](const NavState& X_in, OptionalJacobian<9, 9> H) {
    return navStateImuDynamics(X_in, gyro, accel, params->getGravity(), H);
  };
  // g maps a NavState to its predicted NavState after dt using the same
  // dynamics
  std::function<NavState(const NavState&)> g =
      [&](const NavState& X) -> NavState {
    NavStateImuEKF ekfForX(X, P0, params);
    return ekfForX.predictMean(f, dt);
  };

  // Analytic A from predictMean at X0
  {
    NavStateImuEKF ekfAtX0(X0, P0, params);
    ekfAtX0.predictMean(f, dt, A);
  }

  Matrix9 Anum = numericalDerivative11<NavState, NavState>(g, X0);
  EXPECT(assert_equal(Anum, A, 1e-6));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
