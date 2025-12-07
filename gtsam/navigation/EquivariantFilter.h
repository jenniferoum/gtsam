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
 * 1. **Automatic**: The filter calculates Jacobians (A, B, C) using auxiliary
 *    methods on the Orbit objects (`inputMatrixB`, `measurementMatrixC`) or
 *    the system structure (for A).
 * 2. **Explicit**: You provide the Jacobians (A, B, C) directly. This allows
 *    using "pure" group actions/orbits that don't need to know about system
 *    linearization matrices.
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

  // SFINAE helpers to detect optional matrix methods
  struct internal {
    template <typename T, typename Group, typename = void>
    struct has_inputMatrixB : std::false_type {};
    template <typename T, typename Group>
    struct has_inputMatrixB<T, Group,
                            std::void_t<decltype(std::declval<T>().inputMatrixB(
                                std::declval<Group>()))>> : std::true_type {};
  };

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
   * @brief Propagate the filter state (Automatic).
   *
   * Automatically computes the error dynamics matrix A and asks `psi_u` for
   * the input noise matrix B.
   *
   * @tparam Lift Functor for the lift Λ(ξ, u).
   * @tparam InputOrbit Functor for the input orbit ψ_u.
   * @param lift_u Lift functor for the current input.
   * @param psi_u Input Orbit for the current input.
   * @param Q Process noise covariance in the Input space.
   * @param dt Time step.
   */
  template <typename Lift, typename InputOrbit>
  void predict(const Lift& lift_u, const InputOrbit& psi_u, const Matrix& Q,
               double dt) {
    // 1. Compute A automatically
    MatrixM A = computeErrorDynamicsMatrix<Lift>(psi_u);

    // 2. Get B from InputOrbit (Must exist in Automatic mode)
    static_assert(
        internal::template has_inputMatrixB<InputOrbit, G>::value,
        "InputOrbit must provide inputMatrixB(Group) for automatic "
        "predict. Use explicit predict(...) overload for pure Orbits.");
    Matrix B = psi_u.inputMatrixB(X_);

    // 3. Delegate to explicit predict
    predict(lift_u, Q, dt, A, B);
  }

  /**
   * @brief Propagate the filter state (Explicit).
   *
   * Uses provided Jacobians A and B. This allows `psi_u` to be a pure Orbit
   * without needing to implement `inputMatrixB`.
   *
   * @tparam Lift Functor for the lift Λ(ξ, u).
   * @param lift_u Lift functor for the current input.
   * @param Q Process noise covariance in the Input space.
   * @param dt Time step.
   * @param A Error dynamics matrix (DimM x DimM).
   * @param B Input noise mapping matrix (DimM x DimInput).
   */
  template <typename Lift>
  void predict(const Lift& lift_u, const Matrix& Q, double dt, const MatrixM& A,
               const Matrix& B) {
    // 1. Mean Propagation on Group
    // We don't need Jacobians here, just the value
    M xi_est = act_on_ref_(X_);             // Pure action
    TangentVector Lambda = lift_u(xi_est);  // Pure lift

    X_ = traits<G>::Compose(X_, traits<G>::Expmap(Lambda * dt));

    // 2. Covariance Propagation on Manifold
    // Phi ≈ I + A * dt
    MatrixM Phi = MatrixM::Identity() + A * dt;

    // Map noise: Q_M = B * Q * B^T * dt
    CovarianceM Q_manifold = B * Q * B.transpose() * dt;

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
  Eigen::Matrix<double, OutputOrbit::dimZ, DimM> computeMeasurementMatrix(
      const OutputOrbit& phi_y, const M& xi_hat) const {
    using MatrixZM = Eigen::Matrix<double, OutputOrbit::dimZ, DimM>;

    MatrixZM C;
    // Expect innovation to provide Jacobian via pointer
    phi_y.innovation(xi_hat, &C);
    return C;
  }

  /**
   * @brief Update with measurement (Automatic).
   *
   * Computes the measurement Jacobian C (H internally) automatically.
   *
   * @tparam OutputOrbit Functor for the output orbit ρ_y.
   * @param phi_y Output Orbit instance.
   * @param R Measurement noise covariance.
   */
  template <typename OutputOrbit>
  void update(
      const OutputOrbit& phi_y,
      const Eigen::Matrix<double, OutputOrbit::dimZ, OutputOrbit::dimZ>& R) {
    static constexpr int DimZ = OutputOrbit::dimZ;
    using MatrixZM = Eigen::Matrix<double, DimZ, DimM>;
    using VectorZ = Eigen::Matrix<double, DimZ, 1>;

    M xi_hat = stateEstimate();
    VectorZ innovation;
    MatrixZM H;

    // Expect innovation to provide Jacobian
    innovation = phi_y.innovation(xi_hat, &H);

    updateInternal(innovation, H, R);
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
  void update(
      const OutputOrbit& phi_y,
      const Eigen::Matrix<double, OutputOrbit::dimZ, OutputOrbit::dimZ>& R,
      const Eigen::Matrix<double, OutputOrbit::dimZ, DimM>& C) {
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