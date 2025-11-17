/**
 * @file EquivariantFilter.h
 * @brief Equivariant Filter (EqF) implementation
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

/**
 * Equivariant Filter (EqF) for state estimation on Lie groups with states on
 * manifolds.
 * @tparam M Manifold type for the physical state.
 * @tparam StateAction Functor encoding the right group action on the state.
 */
template <typename M, typename StateAction>
class EqF : public LieGroupEKF<typename StateAction::G> {
 private:
  using G = typename StateAction::G;
  using Base = LieGroupEKF<G>;
  using TangentVector = typename traits<G>::TangentVector;

  M xi_ref_;                // Origin (reference) state on the manifold
  StateAction act_on_ref_;  // Group action on the reference state
  Matrix InnovationLift_;   // Innovation lift matrix

 public:
  static constexpr int Dim = Base::Dim;  ///< Compile-time dimension of G.

  /**
   * Initialize EqF
   * @param X0 Initial Lie group state
   * @param x0 Reference manifold state (origin of the lifted coordinates)
   * @param Sigma Initial covariance (Dim x Dim)
   * @param m Number of direction sensors (must be at least 2)
   */
  EqF(const G& X0, const M& x0, const Matrix& Sigma)
      : Base(X0, Sigma), xi_ref_(x0), act_on_ref_(x0) {
    if (Sigma.rows() != Dim || Sigma.cols() != Dim) {
      throw std::invalid_argument(
          "Initial covariance dimensions must match the degrees of freedom");
    }

    // Compute differential of action phi at origin
    Matrix Dphi0 = act_on_ref_.jacobianAtIdentity();
    InnovationLift_ = Dphi0.completeOrthogonalDecomposition().pseudoInverse();
  }

  /**
   * Return estimated physical state on manifold M.
   * Applies the group action of the current group estimate on the origin state.
   */
  M stateEstimate() const {
    // Group action X * xi_ref (defined for ABC as Group * State).
    return act_on_ref_(this->X_);
  }

  /// Return the current group estimate.
  const G& groupEstimate() const { return this->X_; }

  /**
   * Propagate the filter state.
   * @tparam Lift Computes the lifted tangent vector.
   * @tparam InputAction Provides system matrices derived from the input.
   * @tparam u the input vector.
   * @param Q Process noise covariance in lifted coordinates
   * @param dt Time step
   */
  template <typename Lift, typename InputAction>
  void predict(const typename InputAction::Input& u, const Matrix& Q,
               double dt) {
    // auto dynamics = [this](const G& X, const Vector6& u,
    // OptionalJacobian<Dim, Dim> Df) {
    //   M state_est = act_on_ref_(X);
    //   return state_est.lift(u);
    // };

    // Map current group estimate to physical state on the manifold
    M state_est = stateEstimate();

    // Compute lifted tangent vector from state and input
    Lift lift_u(u);
    TangentVector xi = lift_u(state_est);

    InputAction psi_u(u);
    Matrix Phi = psi_u.stateTransitionMatrix(this->X_, dt);
    Matrix Bt = psi_u.inputMatrixBt(this->X_);

    G X_next = traits<G>::Compose(this->X_, traits<G>::Expmap(xi * dt));
    Matrix Q_process = Bt * Q * Bt.transpose() * dt;
    Base::predict(X_next, Phi, Q_process);
  }

  /**
   * Update the filter state with a direction measurement.
   * @tparam OutputAction Functor encoding the action on the measurement.
   * @tparam Measurement Measurement type carrying y, d, Sigma, and cal_idx.
   * @param y Direction measurement
   */
  template <typename OutputAction>
  void update(const OutputAction& phi_y, const Matrix& R) {
    Matrix Ct = phi_y.measurementMatrixC();

    // Kalman gain
    Matrix Dt = phi_y.outputMatrixDt(this->X_);
    Matrix S = Ct * this->P_ * Ct.transpose() + Dt * R * Dt.transpose();
    Matrix K = this->P_ * Ct.transpose() * S.inverse();

    // Innovation lift
    // TODO(Frank): Why inverse ????
    Vector3 innovation = phi_y.innovation(this->X_.inverse());
    TangentVector delta_xi = InnovationLift_ * (K * innovation);

    // Update state estimate (left-multiply by exp(delta_xi))
    // TODO(Frank): try X_ = traits<M>::Retract(X_, delta_xi);
    this->X_ = traits<G>::Compose(traits<G>::Expmap(delta_xi), this->X_);

    // Update covariance
    Matrix I_n = Matrix::Identity(this->P_.rows(), this->P_.cols());
    this->P_ = (I_n - K * Ct) * this->P_;
  }
};

}  // namespace gtsam
