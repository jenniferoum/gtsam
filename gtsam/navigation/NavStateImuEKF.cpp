/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file  NavStateImuEKF.cpp
 * @brief Extended Kalman Filter derived class for IMU-driven NavState.
 *
 * @date  August 2025
 * @authors Derek Benham, Frank Dellaert
 */

#include <gtsam/navigation/NavStateImuEKF.h> 

namespace gtsam {

Vector9 navStateImuDynamics(const NavState& X, const Vector3& gyro,
                            const Vector3& accel, const Vector3& n_gravity,
                            OptionalJacobian<9, 9> H) {
  Vector9 xi;
  xi.setZero();
  // Rotation, position, velocity in NavState
  const Rot3& R = X.attitude();
  const Vector3& v = X.velocity();

  // Body-frame quantities needed for left-trivialized tangent
  const Vector3 v_body = R.unrotate(v);          // R^T v
  const Vector3 g_body = R.unrotate(n_gravity);  // R^T g

  // Tangent vector components (dR,dP,dV)
  NavState::dR(xi) = gyro;            // omega (body)
  NavState::dP(xi) = v_body;          // p_dot in body frame
  NavState::dV(xi) = accel + g_body;  // v_dot in body frame

  if (H) {
    H->setZero();
    // Helper: skew-symmetric matrix
    auto skew = [](const Vector3& a) {
      Matrix3 S;
      S << 0.0, -a.z(), a.y(), a.z(), 0.0, -a.x(), -a.y(), a.x(), 0.0;
      return S;
    };

    // xi_rot = gyro -> no dependence on state
    // Rows 0:3 already zero

    // xi_trans = R^T v
    // d(xi_trans)/d(dR) = +skew(R^T v) ; d(xi_trans)/d(dV) = I ; d/d(dP) = 0
    H->template block<3, 3>(3, 0) = skew(v_body);
    H->template block<3, 3>(3, 6) = Matrix3::Identity();

    // xi_vel = a + R^T g
    // d(xi_vel)/d(dR) = +skew(R^T g)
    H->template block<3, 3>(6, 0) = skew(g_body);
  }

  return xi;
}

}  // namespace gtsam