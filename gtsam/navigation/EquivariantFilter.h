/**
 * @file ABC_EQF.h
 * @brief Header file for the Attitude-Bias-Calibration Equivariant Filter
 *
 * This file contains declarations for the Equivariant Filter (EqF) for attitude
 * estimation with both gyroscope bias and sensor extrinsic calibration, based
 * on the paper: "Overcoming Bias: Equivariant Filter Design for Biased Attitude
 * Estimation with Online Calibration" by Fornasier et al.
 *
 * @author Darshan Rajasekaran
 * @author Jennifer Oum
 * @author Rohan Bansal
 * @author Frank Dellaert
 * @date 2025
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/LieGroupEKF.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>  // For std::accumulate
#include <string>
#include <vector>

// All implementations are wrapped in this namespace to avoid conflicts
namespace gtsam {

using namespace std;
using namespace gtsam;

//========================================================================
// Equivariant Filter (EqF)
//========================================================================

/// Equivariant Filter (EqF) implementation
template <typename G, typename M>
class EqF : public LieGroupEKF<G> {
 private:
  using LGBase = LieGroupEKF<G>;

  M xi_ref;               // Origin (reference) state on the manifold
  Matrix Dphi0;           // Differential of the state action at origin
  Matrix InnovationLift;  // Innovation lift matrix

 public:
  /// Degrees of freedom of the Lie group
  static constexpr int DOF = gtsam::traits<G>::dimension;
  /// Number of calibration states (sensors), expected to be provided by G
  static constexpr int n_cal = G::numSensors;

  /**
   * Initialize EqF
   * @param X0 Initial Lie group state
   * @param x0 Reference manifold state (origin of the lifted coordinates)
   * @param Sigma Initial covariance (DOF x DOF)
   * @param m Number of direction sensors (must be at least 2)
   */
  EqF(const G& X0, const M& x0, const Matrix& Sigma, int m);

  /**
   * Return estimated physical state on manifold M.
   * Applies the group action of the current group estimate on the origin state.
   */
  M stateEstimate() const;

  /// Return the current group estimate.
  G groupEstimate() const;

  /**
   * Propagate the filter state.
   * @tparam u the input vector.
   * @param Q Process noise covariance in lifted coordinates
   * @param dt Time step
   */
  void predict(const Vector6& u, const Matrix& Q, double dt);

  /**
   * Update the filter state with a direction measurement.
   * @tparam MeasurementType Measurement type (must expose y, d, Sigma,
   * cal_idx).
   * @param y Direction measurement
   */
  template <class MeasurementType>
  void update(const MeasurementType& y);
};

//========================================================================
// Equivariant Filter (EqF) Implementation
//========================================================================
/**
 * Initializes the EqF with state dimension validation  and computes lifted
 * innovation mapping
 * @param Sigma Initial covariance
 * @param n Number of calibration states
 * @param m Number of sensors
 */
template <typename G, typename M>
EqF<G, M>::EqF(const G& X0, const M& x0, const Matrix& Sigma, int m)
    : LGBase(X0, Sigma), xi_ref(x0) {
  if (Sigma.rows() != DOF || Sigma.cols() != DOF) {
    throw std::invalid_argument(
        "Initial covariance dimensions must match the degrees of freedom");
  }

  if (m <= 1) {
    throw std::invalid_argument(
        "Number of direction sensors must be at least 2");
  }

  // Compute differential of phi
  Dphi0 = stateActionDiff(xi_ref);
  InnovationLift = Dphi0.completeOrthogonalDecomposition().pseudoInverse();
}
/**
 * Computes the internal group state to a physical state estimate
 * @return Current state estimate
 */
template <typename G, typename M>
M EqF<G, M>::stateEstimate() const {
  // Group action X * xi_ref (defined for ABC as Group * State).
  return this->X_ * xi_ref;
}

template <typename G, typename M>
G EqF<G, M>::groupEstimate() const {
  return this->X_;
}

/**
 * Implements the prediction step of the EqF using system dynamics and
 * covariance propagation and advances the filter state by symmetry-preserving
 * dynamics.Uses a Lie group integrator scheme for discrete time propagation
 * @param u Angular velocity measurements
 * @param dt time steps
 * Updated internal state and covariance
 */
template <typename G, typename M>
void EqF<G, M>::predict(const Vector6& u, const Matrix& Q, double dt) {
  // Map current group estimate to physical state on the manifold
  M state_est = stateEstimate();

  // Compute lifted tangent vector from state and input
  Vector L = state_est.lift(u);

  Matrix Phi = stateTransitionMatrix(this->X_, u, dt);
  Matrix Bt = inputMatrixBt(this->X_);

  this->X_ = traits<G>::Compose(this->X_, traits<G>::Expmap(L * dt));
  this->P_ = Phi * this->P_ * Phi.transpose() + Bt * Q * Bt.transpose() * dt;
}
/**
 * Implements the correction step of the filter using discrete measurements
 * Computes the measurement residual, Kalman gain and the updates both the state
 * and covariance
 *
 * @param y Measurements
 */
template <typename G, typename M>
template <class MeasurementType>
void EqF<G, M>::update(const MeasurementType& y) {
  if (y.cal_idx > static_cast<int>(n_cal)) {
    throw std::invalid_argument("Calibration index out of range");
  }

  // Get vector representations for checking
  Vector3 y_vec = y.y.unitVector();
  Vector3 d_vec = y.d.unitVector();

  // Skip update if any NaN values are present
  if (std::isnan(y_vec[0]) || std::isnan(y_vec[1]) || std::isnan(y_vec[2]) ||
      std::isnan(d_vec[0]) || std::isnan(d_vec[1]) || std::isnan(d_vec[2])) {
    return;  // Skip this measurement
  }

  Matrix Ct = measurementMatrixC(y.d, y.cal_idx, this->X_);

  Vector3 action_result = outputAction(this->X_.inv(), y.y, y.cal_idx);
  Vector3 delta_vec = Rot3::Hat(y.d.unitVector()) * action_result;
  Matrix Dt = outputMatrixDt(this->X_, y.cal_idx);

  Matrix S = Ct * this->P_ * Ct.transpose() + Dt * y.Sigma * Dt.transpose();
  Matrix K = this->P_ * Ct.transpose() * S.inverse();

  // Use base class to perform update with zero innovation
  Vector3 z_zero = Vector3::Zero();
  Vector3 prediction_zero = Vector3::Zero();
  auto& baseEKF = static_cast<ManifoldEKF<G>&>(
      *this);  // not sure if this is the right way to do this (although it
               // works). would be better to inherit from ManifoldEKF
  baseEKF.update(prediction_zero, Ct, z_zero, Dt * y.Sigma * Dt.transpose());
  // Apply EqF state correction with InnovationLift
  Vector Delta = InnovationLift * K * delta_vec;
  this->X_ = traits<G>::Compose(traits<G>::Expmap(Delta), this->X_);
}
}  // namespace gtsam
