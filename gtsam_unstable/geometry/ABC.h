/**
 * @file ABC.h
 * @brief Core components for Attitude-Bias-Calibration systems
 *
 * This file contains fundamental components and utilities for the ABC system
 * based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 *
 * @author Darshan Rajasekaran
 * @author Jennifer Oum
 * @author Rohan Bansal
 * @author Frank Dellaert
 * @date 2025
 */

#pragma once

/**
 * @file ABC.h
 * @brief Core components for Attitude-Bias-Calibration systems
 *
 * This file contains fundamental components and utilities for the ABC system
 * based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */

#include <gtsam/base/Matrix.h>
#include <gtsam/base/MatrixLieGroup.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gtsam {
namespace abc {

/// Convert angular velocity vector to mathematical input (ω, 0)
inline Vector6 toInputVector(const Vector3& w) {
  return (Vector6() << w, Z_3x1).finished();
}

/// Bundle of calibration rotations modeled as a Lie group
template <size_t N>
using Calibrations = PowerLieGroup<Rot3, N>;

//========================================================================
// State Manifold
//========================================================================

/// State class representing the state of the Biased Attitude System
template <size_t N>
struct State {
  Rot3 R;             // Attitude rotation matrix R
  Vector3 b;          // Gyroscope bias b
  Calibrations<N> S;  // Sensor calibrations S

  static constexpr int dimension = 6 + 3 * N;

  /// Constructor
  State(const Rot3& R = Rot3(), const Vector3& b = Z_3x1,
        const Calibrations<N>& S = Calibrations<N>())
      : R(R), b(b), S(S) {}

  /// Identity function
  static State identity() { return State(Rot3(), Z_3x1, Calibrations<N>()); }

  /**
   * Compute Local coordinates in the state relative to another state.
   * @param other The other state
   * @return Local coordinates in the tangent space
   */
  Vector localCoordinates(const State<N>& other) const {
    Vector eps(dimension);

    // First 3 elements - attitude
    eps.head<3>() = R.logmap(other.R);
    // Next 3 elements - bias
    eps.segment<3>(3) = other.b - b;

    // Remaining elements - calibrations
    eps.template segment<3 * N>(6) = S.logmap(other.S);

    return eps;
  }

  /**
   * Retract from tangent space back to the manifold
   * @param v Vector in the tangent space
   * @return New state
   */
  State retract(const Vector& v) const {
    if (v.size() != static_cast<Eigen::Index>(dimension)) {
      throw std::invalid_argument(
          "Vector size does not match state dimensions");
    }
    Rot3 newR = R.expmap(v.head<3>());
    Vector3 newB = b + v.segment<3>(3);
    typename Calibrations<N>::TangentVector deltaS;
    deltaS = v.template segment<3 * N>(6);
    Calibrations<N> newS = S.expmap(deltaS);
    return State(newR, newB, newS);
  }

  void print(const std::string& s = "") const {
    if (!s.empty()) std::cout << s << " ";
    std::cout << "State<" << N << ">" << std::endl;
    R.print("  R");
    std::cout << "  b: " << b.transpose() << std::endl;
    for (size_t i = 0; i < N; ++i) {
      const std::string label = "  S[" + std::to_string(i) + "]";
      S[i].print(label);
    }
  }

  bool equals(const State<N>& other, double tol = 1e-9) const {
    if (!R.equals(other.R, tol)) return false;
    if (!equal_with_abs_tol(b, other.b, tol)) return false;
    return traits<Calibrations<N>>::Equals(S, other.S, tol);
  }
};

//========================================================================
// Symmetry Group
//========================================================================

/**
 * Symmetry group defined as the product Pose3 x Calibrations<n>
 * The Pose3 component models the (SO(3) ⋉ R^3) part acting on attitude/bias,
 * while Calibrations<n> captures the N sensor calibration rotations.
 */
template <size_t n>
struct Group : public ProductLieGroup<Pose3, Calibrations<n>> {
  using Base = ProductLieGroup<Pose3, Calibrations<n>>;
  using typename Base::ChartJacobian;
  using typename Base::Jacobian;
  using typename Base::TangentVector;

  static constexpr int dimension = Base::dimension;
  static constexpr size_t numSensors = n;

  Group() : Base() {}
  Group(const Pose3& pose, const Calibrations<n>& calibrations)
      : Base(pose, calibrations) {}
  Group(const Base& base) : Base(base) {}
  Group(const Rot3& A, const Matrix3& a, const Calibrations<n>& B)
      : Group(Pose3(A, Point3(Rot3::Vee(a))), B) {}

  static Group Identity() { return Group(); }

  Group operator*(const Group& other) const {
    return Group(Base::operator*(other));
  }

  Group compose(const Group& other, ChartJacobian H1 = ChartJacobian(),
                ChartJacobian H2 = ChartJacobian()) const {
    return Group(Base::compose(other, H1, H2));
  }

  Group between(const Group& other, ChartJacobian H1 = ChartJacobian(),
                ChartJacobian H2 = ChartJacobian()) const {
    return Group(Base::between(other, H1, H2));
  }

  Group inverse(ChartJacobian D = ChartJacobian()) const {
    return Group(Base::inverse(D));
  }

  Group retract(const TangentVector& v, ChartJacobian H1 = ChartJacobian(),
                ChartJacobian H2 = ChartJacobian()) const {
    return Group(Base::retract(v, H1, H2));
  }

  Group expmap(const TangentVector& v) const { return Group(Base::expmap(v)); }

  TangentVector logmap(const Group& g) const { return Base::logmap(g); }

  static Group Expmap(const TangentVector& v,
                      ChartJacobian Hv = ChartJacobian()) {
    return Group(Base::Expmap(v, Hv));
  }

  static TangentVector Logmap(const Group& g,
                              ChartJacobian Hg = ChartJacobian()) {
    return Base::Logmap(g, Hg);
  }

  const Pose3& pose() const { return this->first; }
  Rot3 A() const { return this->first.rotation(); }
  Vector3 a() const { return this->first.translation(); }
  const Calibrations<n>& calibrations() const { return this->second; }

  void print(const std::string& s = "") const {
    if (!s.empty()) std::cout << s << " ";
    std::cout << "Group<" << n << ">" << std::endl;
    pose().print("  Pose");
    for (size_t i = 0; i < n; ++i) {
      const std::string label = "  S[" + std::to_string(i) + "]";
      calibrations()[i].print(label);
    }
  }

  bool equals(const Group& other, double tol = 1e-9) const {
    if (!pose().equals(other.pose(), tol)) return false;
    return traits<Calibrations<n>>::Equals(calibrations(), other.calibrations(),
                                           tol);
  }
};

//========================================================================
// Group Action on State Manifold
//========================================================================

/**
 * The symmetry group G is defined as the product Pose3 x Calibrations<n>.
 * The Pose3 component models the (SO(3) ⋉ R^3) part acting on attitude/bias,
 * while Calibrations<n> captures the N sensor calibration rotations.
 *
 * The group action is defined as a function object below,
 * applied to a given state x, specified in constructor.
 */
template <size_t N>
struct StateAction {
  using M = State<N>;
  using G = Group<N>;

  const M xi_;  // Reference state

  /**
   * Construct action functor with reference state
   * @param xi State object
   */
  StateAction(const M& xi) : xi_(xi) {}

  /**
   * Implements group actions on the states
   * @param g An element of the symmetry group G.
   * @return Transformed state
   */
  M operator()(const G& g) const {
    const Rot3 new_R = xi_.R * g.A();
    const Rot3 At = g.A().inverse();
    const Vector3 new_b = At * (xi_.b - g.a());
    Calibrations<N> new_S;
    const Calibrations<N>& B = g.calibrations();
    for (size_t i = 0; i < N; i++) new_S[i] = At * xi_.S[i] * B[i];
    return {new_R, new_b, new_S};
  }

  /**
   * The Jacobian of the action at the identity of the symmetry group G
   * @return A matrix representing the jacobian of the state action
   */
  Matrix JacobianAtIdentity() const {
    Matrix H = Matrix::Zero(M::dimension, G::dimension);

    // Rotation block: δθ maps directly to the state's rotational tangent.
    H.block<3, 3>(0, 0) = I_3x3;

    // Bias block: δb = -δa + xi_.b × δθ.
    H.block<3, 3>(3, 0) = Rot3::Hat(xi_.b);
    H.block<3, 3>(3, 3) = -I_3x3;

    // Calibration blocks: δs_i = δσ_i - S_i^{-1} δθ.
    for (size_t i = 0; i < N; ++i) {
      const Matrix3 S_inv = xi_.S[i].inverse().matrix();
      const size_t row = 6 + 3 * i;
      const size_t col = 6 + 3 * i;
      H.block<3, 3>(row, 0) = -S_inv; // B[i] is identity for G::Identity
      H.block<3, 3>(row, col) = I_3x3;
    }

    return H;
  }
};

/**
 * Functor encoding the right group action on the mathematical input u.
 * For a fixed u = (ω, 0), applying X = (A, a, B) ∈ G yields
 * φ_u(X) = (A^{-1}(ω - a), 0).
 */
template <size_t N>
struct InputAction {
  using G = Group<N>;
  using Input = Vector6;

  const Input u_;

  explicit InputAction(const Input& u) : u_(u) {}

  Input operator()(const G& X) const {
    const Rot3 A = X.A();
    const Vector3 a = X.a();
    Input result;
    result.head<3>() = A.unrotate(u_.head<3>() - a);
    result.tail<3>() = Z_3x1;
    return result;
  }

  Matrix JacobianAtIdentity() const {
    Matrix H = Matrix::Zero(Input::RowsAtCompileTime, G::dimension);
    H.block<3, 3>(0, 0) = Rot3::Hat(u_.head<3>());
    H.block<3, 3>(0, 3) = -I_3x3;
    // Remaining blocks stay zero: the virtual input is unaffected by
    // calibrations.
    return H;
  }
};

/**
 * Geometry class encapsulating the ABC system dynamics and measurement models
 */
template <size_t N>
struct Geometry {
  using M = State<N>;
  using G = Group<N>;
  using InputType = Vector6;  // Mathematical input (ω, 0)

  /**
   * Transforms the Direction measurements based on the calibration type ( Eqn
   * 6)
   * @param X Group element X
   * @param y Direction measurement y
   * @param idx Calibration index
   * @return Transformed direction
   * Uses Rot3 inverse, matrix and Unit3 unitvector functions
   */
  static Vector3 outputAction(const G& X, const Unit3& y, int idx) {
    const Rot3 A = X.A();
    const Calibrations<N>& X_cal = X.calibrations();
    if (idx == -1) {
      return A.inverse().matrix() * y.unitVector();
    } else {
      if (idx >= static_cast<int>(N)) {
        throw std::out_of_range("Calibration index out of range");
      }
      return X_cal[idx].inverse().matrix() * y.unitVector();
    }
  }

  /**
   * Compute the lifted tangent vector from state and input.
   * This implements the lift operation from the equivariant filter paper.
   * @param xi Current state on the manifold (including orientation, bias, and
   * sensor rotations).
   * @param u Mathematical input vector (ω, 0)
   * @return TangentVector Lifted vector in the Lie algebra used for
   * propagation.
   */
  static typename G::TangentVector lift(const M& xi, const InputType& u) {
    Vector3 w = u.head<3>();
    Vector3 corrected_w = w - xi.b;

    Vector L = Vector::Zero(6 + 3 * N);
    L.head<3>() = corrected_w;
    L.segment<3>(3) = -Rot3::Hat(w) * xi.b;
    for (size_t i = 0; i < N; i++) {
      L.segment<3>(6 + 3 * i) = xi.S[i].unrotate(corrected_w);
    }

    return L;
  }

  /**
   * Computes the discrete time state transition matrix
   * @param u Angular velocity
   * @param dt time step
   * @return State transition matrix in discrete time
   */
  static Matrix stateTransitionMatrix(G X_hat, const Vector6& u, double dt) {
    const InputAction<N> psi_u(u);
    const Vector3 omega_tilde = psi_u(X_hat.inverse()).template head<3>();
    Matrix3 W0 = Rot3::Hat(omega_tilde);
    Matrix Phi1 = Matrix::Zero(6, 6);
    Matrix3 W0_sq = W0 * W0;
    Matrix3 Phi12 = -dt * (I_3x3 + 0.5 * dt * W0 + (dt * dt / 6.0) * W0_sq);
    Matrix3 Phi22 = I_3x3 + dt * W0 + 0.5 * dt * dt * W0_sq;
    Phi1.block<3, 3>(0, 0) = I_3x3;
    Phi1.block<3, 3>(0, 3) = Phi12;
    Phi1.block<3, 3>(3, 3) = Phi22;

    std::vector<Matrix> blocks;
    blocks.push_back(Phi1);
    blocks.insert(blocks.end(), N, Phi22);
    return gtsam::diag(blocks);
  }

  /**
   * Computes linearized continuous time state matrix
   * @param data Input data
   * @return Linearized state matrix
   */
  static Matrix stateMatrixA(const G& X_hat, const Vector6& u) {
    const InputAction<N> psi_u(u);
    const Vector3 omega_tilde = psi_u(X_hat.inverse()).template head<3>();
    Matrix3 W0 = Rot3::Hat(omega_tilde);

    Matrix A1 = Matrix::Zero(6, 6);
    A1.block<3, 3>(0, 3) = -I_3x3;
    A1.block<3, 3>(3, 3) = W0;

    std::vector<Matrix> blocks{A1};
    blocks.insert(blocks.end(), N, W0);
    return gtsam::diag(blocks);
  }

  /// Computes the input uncertainty propagation matrix
  static Matrix inputMatrix(G X_hat) {
    const Matrix3 A_matrix = X_hat.A().matrix();
    Matrix B1 = gtsam::diag({A_matrix, A_matrix});
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.calibrations()[i].matrix();
    }

    return gtsam::diag({B1, B2});
  }

  /// The continuous-time process noise covariance in lifted coordinates
  static Matrix processNoise(const Matrix& Sigma) {
    std::vector<Matrix> blocks{Sigma};
    blocks.insert(blocks.end(), N, 1e-9 * I_3x3);
    return gtsam::diag(blocks);
  }

  /// Computes the input uncertainty propagation matrix
  static Matrix inputMatrixBt(G X_hat) {
    const Matrix3 A_matrix = X_hat.A().matrix();
    Matrix B1 = gtsam::diag({A_matrix, A_matrix});
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.calibrations()[i].matrix();
    }

    return gtsam::diag({B1, B2});
  }

  /**
   * Computes the linearized measurement matrix. The structure depends on
   * whether the sensor has a calibration state
   * @param d reference direction
   * @param idx Calibration index
   * @return Measurement matrix
   * Uses the matrix zero, Rot3 hat and the Unitvector functions
   */
  static Matrix measurementMatrixC(const Unit3& d, int idx) {
    Matrix Cc = Matrix::Zero(3, 3 * N);

    // If the measurement is related to a sensor that has a calibration state
    if (idx >= 0) {  // Set the correct 3x3 block in Cc
      Cc.block<3, 3>(0, 3 * idx) = Rot3::Hat(d.unitVector());
    }

    Matrix3 wedge_d = Rot3::Hat(d.unitVector());

    // Build the combined matrix
    Matrix temp(3, 6 + 3 * N);
    temp.block<3, 3>(0, 0) = wedge_d;
    temp.block<3, 3>(0, 3) = Matrix3::Zero();
    temp.block(0, 6, 3, 3 * N) = Cc;

    return wedge_d * temp;
  }

  /**
   * Computes the measurement uncertainty propagation matrix
   * @param idx Calibration index
   * @return Returns B[idx] for calibrated sensors, A for uncalibrated
   */
  static Matrix outputMatrixDt(int idx, Group<N> X_hat) {
    // If the measurement is related to a sensor that has a calibration state
    if (idx >= 0) {
      if (idx >= static_cast<int>(N)) {
        throw std::out_of_range("Calibration index out of range");
      }
      return X_hat.calibrations()[idx].matrix();
    } else {
      return X_hat.A().matrix();
    }
  }

  static constexpr int n_cal = N;
};

}  // namespace abc

template <size_t N>
struct traits<abc::State<N>> : public internal::Manifold<abc::State<N>> {};

template <size_t N>
struct traits<const abc::State<N>> : public internal::Manifold<abc::State<N>> {
};

template <size_t N>
struct traits<abc::Group<N>> : internal::LieGroup<abc::Group<N>> {};

template <size_t N>
struct traits<const abc::Group<N>> : internal::LieGroup<abc::Group<N>> {};

}  // namespace gtsam
