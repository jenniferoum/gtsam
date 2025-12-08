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

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/Values.h>

#include <iostream>
#include <type_traits>

namespace gtsam {

/**
 * Equivariant Filter (EqF) for state estimation on Lie groups.
 *
 * The EqF estimates a Lie group state X ∈ G and a manifold state ξ ∈ M.
 * It uses a symmetry principle where the error dynamics are autonomous in a
 * specific frame.
 *
 * This implementation supports two modes:
 * 1. **Automatic**: The filter calculates Jacobian A using the input orbit.
 * 2. **Explicit**: You provide the Jacobian A and the manifold covariance Qc
 *    directly.
 *
 * @tparam M Manifold type for the physical state.
 * @tparam Symmetry Functor encoding the group action on the state.
 */
template <typename M, typename Symmetry>
class EquivariantFilter {
 public:
  // Manifold traits
  static constexpr int DimM = traits<M>::dimension;
  using TangentM = typename traits<M>::TangentVector;
  using MatrixM = Eigen::Matrix<double, DimM, DimM>;
  using CovarianceM = MatrixM;

  // Group traits
  using G = typename Symmetry::Group;
  static constexpr int DimG = traits<G>::dimension;
  using TangentVector = typename traits<G>::TangentVector;

  // Cross-dimension helpers
  using MatrixMG = Eigen::Matrix<double, DimM, DimG>;
  using MatrixGM = Eigen::Matrix<double, DimG, DimM>;

 private:
  M xi_ref_;  // Origin (reference) state on the manifold
  typename Symmetry::Orbit act_on_ref_;  // Orbit of the reference state
  MatrixMG Dphi0_;           // Differential of state action at identity
  MatrixGM InnovationLift_;  // Innovation lift matrix ((Dphi0)^+)

  G X_;            // Group element estimate
  CovarianceM P_;  // Covariance on the Manifold M

 public:
  /**
   * @brief Initialize the Equivariant Filter.
   *
   * @param xi_ref Reference manifold state (origin of lifted coordinates).
   * @param Sigma Initial covariance on the manifold.
   * @param X0 Initial group estimate (default: Identity).
   */
  EquivariantFilter(const M& xi_ref, const CovarianceM& Sigma,
                    const G& X0 = traits<G>::Identity())
      : xi_ref_(xi_ref), act_on_ref_(xi_ref), X_(X0), P_(Sigma) {
    // Compute differential of action phi at identity (Dphi0)
    act_on_ref_(traits<G>::Identity(), &Dphi0_);

    // Precompute the Innovation Lift matrix (pseudo-inverse of Dphi0)
    InnovationLift_ = Dphi0_.completeOrthogonalDecomposition().pseudoInverse();
  }

  /// @return Estimated physical state on the manifold M.
  M stateEstimate() const { return act_on_ref_(X_); }

  /// @return Current group estimate.
  const G& groupEstimate() const { return X_; }

  /// @return Current covariance estimate on the manifold.
  const CovarianceM& covariance() const { return P_; }

  /**
   * @brief Compute the error dynamics matrix A (Automatic).
   *
   * Calculates A = D_phi|_0 * D_lift|_u0, where u0 is the input mapped to the
   * origin.
   *
   * Concept requirements:
   * - `Lift` must be callable as `Lift(u_origin)(xi_ref, D_lift)` where
   *   D_lift is an OptionalJacobian of shape DimM x DimG.
   * - `InputOrbit` must be a group action on the input space with operator()
   *   that accepts the current group estimate X and returns the mapped input
   *   (no other methods are required by the filter).
   *
   * @tparam Lift Functor for the lift Λ(ξ, u).
   * @tparam InputOrbit Functor for the input orbit ψ_u.
   * @param psi_u Input Orbit instance.
   * @return MatrixM The calculated error dynamics matrix A.
   */
  template <typename Lift, typename InputOrbit>
  MatrixM computeErrorDynamicsMatrix(const InputOrbit& psi_u) const {
    MatrixGM D_lift;
    // Map current input to origin: u_origin = psi_u(X^-1)
    auto u_origin = psi_u(X_.inverse());

    // Lift at origin: D_lift = d(Lambda(xi_ref, u_origin))/dxi
    Lift lift_u_origin(u_origin);
    lift_u_origin(xi_ref_, &D_lift);

    return Dphi0_ * D_lift;
  }

  /**
   * @brief Discretize continuous-time error dynamics δ̇ = A δ over dt.
   *
   * On manifolds (unlike Lie groups) the error stays in a fixed tangent space
   * at the chosen origin, so discretization is just the matrix exponential of
   * A. K mirrors LieGroupEKF: K=1 gives Euler, K>1 calls expm(A*dt, K).
   */
  template <size_t K = 1>
  MatrixM transitionMatrix(const MatrixM& A, double dt) const {
    if constexpr (K == 1) {
      return MatrixM::Identity() + A * dt;
    } else {
      return MatrixM(expm(A * dt, K));
    }
  }

  /**
   * @brief Propagate the filter state (Automatic).
   *
   * Automatically computes the error dynamics matrix A.
   *
   * Concept requirements:
   * - `Lift` is used as `Lift(u_origin)(xi_ref_, D_lift)` to obtain the lift
   *   and its Jacobian w.r.t. the manifold state.
   * - `InputOrbit` is only used via `psi_u(X_.inverse())` to map the current
   *   input to the origin; no other methods are needed.
   *
   * @tparam K Truncation order for discretization (1 = first order Euler,
   *         >1 uses matrix exponential expm(A*dt, K)).
   * @tparam Lift Functor for the lift Λ(ξ, u).
   * @tparam InputOrbit Functor for the input orbit ψ_u.
   * @param lift_u Lift functor for the current input.
   * @param psi_u Input Orbit for the current input.
   * @param A Error dynamics matrix (DimM x DimM).
   * @param Qc Process noise covariance on the manifold (continuous-time).
   * @param dt Time step.
   */
  template <size_t K = 1, typename Lift, typename InputOrbit>
  void predict(const Lift& lift_u, const InputOrbit& psi_u, const MatrixM& Qc,
               double dt) {
    // 1. Compute A automatically
    MatrixM A = computeErrorDynamicsMatrix<Lift>(psi_u);

    // 2. Delegate to explicit predict with manifold Qc
    predictWithJacobian<K>(lift_u, A, Qc, dt);
  }

  /**
   * @brief Propagate the filter state (Explicit).
   *
   * Uses provided Jacobian A and manifold covariance Qc. This allows `psi_u`
   * to be a pure Orbit without needing to implement `inputMatrixB`.
   *
   * Concept requirements:
   * - `Lift` is only used via `Lift(xi_est)` to produce a tangent vector.
   *   No additional methods are needed for this overload.
   *
   * @tparam Lift Functor for the lift Λ(ξ, u).
   * @param lift_u Lift functor for the current input.
   * @param A Error dynamics matrix (DimM x DimM).
   * @param Qc Process noise covariance on the manifold (continuous-time).
   * @param dt Time step.
   */
  template <size_t K = 1, typename Lift>
  void predictWithJacobian(const Lift& lift_u, const MatrixM& A,
                           const MatrixM& Qc, double dt) {
    // 1. Mean Propagation on Group
    M xi_est = act_on_ref_(X_);             // Pure action
    TangentVector Lambda = lift_u(xi_est);  // Pure lift

    X_ = traits<G>::Compose(X_, traits<G>::Expmap(Lambda * dt));

    // 2. Covariance Propagation on Manifold
    MatrixM Phi = transitionMatrix<K>(A, dt);

    // Qc is manifold continuous-time covariance: Q_M = Qc * dt
    CovarianceM Q_manifold = Qc * dt;

    P_ = Phi * P_ * Phi.transpose() + Q_manifold;
  }

  /**
   * @brief Helper to compute the measurement matrix C for testing or analysis.
   *
   * @param phi_y Output Orbit instance.
   * @param xi_hat Linearization point on the manifold.
   * @return The measurement Jacobian C (DimZ x DimM).
   */
  template <typename OutputOrbit>
  auto computeMeasurementMatrix(const OutputOrbit& phi_y,
                                const M& xi_hat) const {
    using Measurement = typename OutputOrbit::Manifold;
    static constexpr int DimZ = traits<Measurement>::dimension;

    if constexpr (DimZ == Eigen::Dynamic) {
      auto innovation = phi_y.innovation(xi_hat);
      const int m = innovation.rows();
      Eigen::Matrix<double, Eigen::Dynamic, DimM> C(m, DimM);
      phi_y.innovation(xi_hat, &C);
      return C;
    } else {
      Eigen::Matrix<double, DimZ, DimM> C;
      phi_y.innovation(xi_hat, &C);
      return C;
    }
  }

  /**
   * @brief Update with measurement (Automatic).
   *
   * Computes the measurement Jacobian C (H internally) automatically.
   *
   * Concept requirements:
   * - `OutputOrbit` must implement the group action on outputs via
   *   `phi_y.innovation(xi_hat, H)`. No other methods are required.
   *
   * @tparam OutputOrbit Functor for the output orbit ρ_y.
   * @param phi_y Output Orbit instance.
   * @param R Measurement noise covariance.
   */
  template <typename OutputOrbit>
  void update(const OutputOrbit& phi_y, const Eigen::MatrixXd& R) {
    M xi_hat = stateEstimate();

    using Measurement = typename OutputOrbit::Manifold;
    static constexpr int DimZ = traits<Measurement>::dimension;

    if constexpr (DimZ == Eigen::Dynamic) {
      auto innovation = phi_y.innovation(xi_hat);
      const int m = innovation.rows();
      Eigen::Matrix<double, Eigen::Dynamic, DimM> H(m, DimM);
      phi_y.innovation(xi_hat, &H);
      updateInternal(innovation, H, R);
    } else {
      Eigen::Matrix<double, DimZ, 1> innovation;
      Eigen::Matrix<double, DimZ, DimM> H;
      innovation = phi_y.innovation(xi_hat, &H);
      updateInternal(innovation, H, R);
    }
  }

  /**
   * @brief Update with measurement (Explicit).
   *
   * Uses the provided measurement Jacobian C.
   *
   * @tparam OutputOrbit Functor for the output orbit ρ_y.
   * @param phi_y Output Orbit instance.
   * @param R Measurement noise covariance.
   * @param C Measurement Jacobian (DimZ x DimM).
   */
  template <typename OutputOrbit>
  void update(const OutputOrbit& phi_y, const Eigen::MatrixXd& R,
              const Eigen::MatrixXd& C) {
    // Compute innovation without Jacobian
    M xi_hat = stateEstimate();
    auto innovation = phi_y.innovation(xi_hat);

    // Delegate using provided C as H
    updateInternal(innovation, C, R);
  }

 protected:
  /**
   * @brief Internal update implementation.
   *
   * Uses standard EKF equations:
   * K = P * H^T * (H * P * H^T + R)^-1
   * P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
   */
  template <typename VectorZ, typename MatrixZM, typename MatrixZ>
  void updateInternal(const VectorZ& innovation, const MatrixZM& H,
                      const MatrixZ& R) {
    using MatrixMZ = Eigen::Matrix<double, DimM, VectorZ::RowsAtCompileTime>;

    // Kalman Gain
    // Evaluated strictly to avoid lazy evaluation issues when P_ updates
    auto Ht = H.transpose();
    MatrixZ S = H * P_ * Ht + R;
    MatrixMZ K = P_ * Ht * S.inverse();

    // Update Covariance (Joseph form)
    MatrixM I_KH = MatrixM::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();

    // Correction in Manifold tangent space
    // K matches dimensions with innovation, so result is TangentM
    TangentM delta_xi = K * innovation;

    // Lift correction to Group tangent space
    TangentVector delta_x = InnovationLift_ * delta_xi;

    // Update Group estimate
    X_ = traits<G>::Compose(traits<G>::Expmap(delta_x), X_);
  }
};

}  // namespace gtsam
