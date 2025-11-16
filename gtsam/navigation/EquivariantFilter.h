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
 * @tparam Geometry Class defining the group symmetry and action.
 */
template <typename M, typename Geometry>
class EqF : public LieGroupEKF<typename Geometry::GType> {
 private:
  using G = typename Geometry::GType;
  using Base = LieGroupEKF<G>;

  M xi_ref_;               // Origin (reference) state on the manifold
  Matrix InnovationLift_;  // Innovation lift matrix
  
  public:
  static constexpr int Dim = Base::Dim;  ///< Compile-time dimension of G.
  
  /// Number of calibration states (sensors), expected to be provided by G
  static constexpr int n_cal = G::numSensors;
  
  /**
   * Initialize EqF
   * @param X0 Initial Lie group state
   * @param x0 Reference manifold state (origin of the lifted coordinates)
   * @param Sigma Initial covariance (Dim x Dim)
   * @param m Number of direction sensors (must be at least 2)
   */
  EqF(const G& X0, const M& x0, const Matrix& Sigma, int m)
  : Base(X0, Sigma), xi_ref_(x0) {
    if (Sigma.rows() != Dim || Sigma.cols() != Dim) {
      throw std::invalid_argument(
        "Initial covariance dimensions must match the degrees of freedom");
      }
      
      if (m <= 1) {
        throw std::invalid_argument(
          "Number of direction sensors must be at least 2");
        }
        
    // Compute differential of action phi at origin
    Matrix Dphi0 = Geometry::stateActionDiff(xi_ref_);
    InnovationLift_ = Dphi0.completeOrthogonalDecomposition().pseudoInverse();
  }

  /**
   * Return estimated physical state on manifold M.
   * Applies the group action of the current group estimate on the origin state.
   */
  M stateEstimate() const {
    // Group action X * xi_ref (defined for ABC as Group * State).
    return Geometry::stateAction(this->X_, xi_ref_);
  }

  /// Return the current group estimate.
  G groupEstimate() const { return this->X_; }

  /**
   * Propagate the filter state.
   * @tparam u the input vector.
   * @param Q Process noise covariance in lifted coordinates
   * @param dt Time step
   */
  void predict(const Vector6& u, const Matrix& Q, double dt) {
    // auto dynamics = [this](const G& X, const Vector6& u, OptionalJacobian<Dim, Dim> Df) {
    //   M state_est = X * xi_ref_;
    //   return state_est.lift(u);
    // };

    // Map current group estimate to physical state on the manifold
    M state_est = stateEstimate();

    // Compute lifted tangent vector from state and input
    Vector xi = Geometry::lift(state_est, u);

    Matrix Phi = Geometry::stateTransitionMatrix(this->X_, u, dt);
    Matrix Bt = Geometry::inputMatrixBt(this->X_);

    this->X_ = traits<G>::Compose(this->X_, traits<G>::Expmap(xi * dt));
    this->P_ = Phi * this->P_ * Phi.transpose() + Bt * Q * Bt.transpose() * dt;
  }

  /**
   * Update the filter state with a direction measurement.
   * @tparam MeasurementType Measurement type (must expose y, d, Sigma,
   * cal_idx).
   * @param y Direction measurement
   */
  template <class MeasurementType>
  void update(const MeasurementType& y) {
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

    Matrix Ct = Geometry::measurementMatrixC(y.d, y.cal_idx);

    Vector3 action_result = Geometry::outputAction(this->X_.inverse(), y.y, y.cal_idx);
    Vector3 delta_vec = Rot3::Hat(y.d.unitVector()) * action_result;
    Matrix Dt = Geometry::outputMatrixDt(y.cal_idx, this->X_);
    // Kalman gain
    Matrix S = Ct * this->P_ * Ct.transpose() + Dt * y.Sigma * Dt.transpose();
    Matrix K = this->P_ * Ct.transpose() * S.inverse();

    // Innovation lift
    Vector Delta = InnovationLift_ * (K * delta_vec);

    // Update state estimate (left-multiply by exp(Delta))
    this->X_ = traits<G>::Compose(traits<G>::Expmap(Delta), this->X_);

    // Update covariance
    Matrix I = Matrix::Identity(this->P_.rows(), this->P_.cols());
    this->P_ = (I - K * Ct) * this->P_;
  }
};

}  // namespace gtsam
