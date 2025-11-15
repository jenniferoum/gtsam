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
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gtsam {
namespace abc {

/// Convert angular velocity vector to mathematical input (ω, 0)
inline Vector6 toInputVector(const Vector3& w) {
  return (Vector6() << w, Z_3x1).finished();
}

//========================================================================
// State Manifold
//========================================================================

/// State class representing the state of the Biased Attitude System
template <size_t N>
class State {
 public:
  Rot3 R;                 // Attitude rotation matrix R
  Vector3 b;              // Gyroscope bias b
  std::array<Rot3, N> S;  // Sensor calibrations S

  /// Constructor
  State(const Rot3& R = Rot3(), const Vector3& b = Z_3x1,
        const std::array<Rot3, N>& S = std::array<Rot3, N>{})
      : R(R), b(b), S(S) {}

  /// Identity function
  static State identity() {
    std::array<Rot3, N> S_id{};
    S_id.fill(Rot3());
    return State(Rot3(), Z_3x1, S_id);
  }

  /**
   * Compute Local coordinates in the state relative to another state.
   * @param other The other state
   * @return Local coordinates in the tangent space
   */
  Vector localCoordinates(const State<N>& other) const {
    Vector eps(6 + 3 * N);

    // First 3 elements - attitude
    eps.head<3>() = R.logmap(other.R);
    // Next 3 elements - bias
    eps.segment<3>(3) = other.b - b;

    // Remaining elements - calibrations
    for (size_t i = 0; i < N; i++) {
      eps.segment<3>(6 + 3 * i) = S[i].logmap(other.S[i]);
    }

    return eps;
  }

  /**
   * Retract from tangent space back to the manifold
   * @param v Vector in the tangent space
   * @return New state
   */
  State retract(const Vector& v) const {
    if (v.size() != static_cast<Eigen::Index>(6 + 3 * N)) {
      throw std::invalid_argument(
          "Vector size does not match state dimensions");
    }
    Rot3 newR = R.expmap(v.head<3>());
    Vector3 newB = b + v.segment<3>(3);
    std::array<Rot3, N> newS;
    for (size_t i = 0; i < N; i++) {
      newS[i] = S[i].expmap(v.segment<3>(6 + 3 * i));
    }
    return State(newR, newB, newS);
  }

  /**
   * Compute the lifted tangent vector from state and input.
   * This implements the lift operation from the equivariant filter paper.
   * @param u Input vector (ω, 0) where first 3 are angular velocity
   * @return Vector Lifted vector in the Lie algebra used for propagation.
   */
  Vector lift(const Vector6& u) const {
    Vector3 w = u.head<3>();
    Vector3 corrected_w = w - b;

    Vector L = Vector::Zero(6 + 3 * N);
    L.head<3>() = corrected_w;
    L.segment<3>(3) = -Rot3::Hat(w) * b;
    for (size_t i = 0; i < N; i++) {
      L.segment<3>(6 + 3 * i) = S[i].inverse().matrix() * corrected_w;
    }

    return L;
  }
};

//========================================================================
// Symmetry Group
//========================================================================

/**
 * Symmetry group (SO(3) |x so(3)) x SO(3) x ... x SO(3)
 * Each element of the B list is associated with a calibration state
 */
template <size_t n>
struct Group : public MatrixLieGroup<Group<n>, 6 + 3 * n, 4 + 3 * n> {
  using Base = MatrixLieGroup<Group<n>, 6 + 3 * n, 4 + 3 * n>;
  using typename Base::ChartJacobian;
  using typename Base::TangentVector;

  static constexpr int dimension = Base::dimension;
  static constexpr int matrixDim = 4 + 3 * n;
  using MatrixType = Eigen::Matrix<double, matrixDim, matrixDim>;
  using LieAlgebra = MatrixType;
  static constexpr int numSensors = n;

  Rot3 A;                 /// First SO(3) element
  Matrix3 a;              /// so(3) element (skew-symmetric matrix)
  std::array<Rot3, n> B;  /// List of SO(3) elements for calibration

  /// Default-initialize to identity
  Group() : A(), a(Matrix3::Zero()) { B.fill(Rot3()); }

  Group(const Rot3& A_in, const Matrix3& a_in, const std::array<Rot3, n>& B_in)
      : A(A_in), a(a_in), B(B_in) {}

  static Group Identity() { return Group(); }

  /// Matrix representation used by MatrixLieGroup machinery.
  MatrixType matrix() const {
    MatrixType result = MatrixType::Zero();
    result.template block<3, 3>(0, 0) = A.matrix();
    result.template block<3, 1>(0, 3) = Rot3::Vee(a);
    result(3, 3) = 1.0;
    for (size_t i = 0; i < n; ++i) {
      const int offset = 4 + static_cast<int>(3 * i);
      result.template block<3, 3>(offset, offset) = B[i].matrix();
    }
    return result;
  }

  /// hat operator for the Lie algebra
  static LieAlgebra Hat(const TangentVector& xi) {
    LieAlgebra result = LieAlgebra::Zero();
    result.template block<3, 3>(0, 0) = Rot3::Hat(xi.template head<3>());
    result.template block<3, 1>(0, 3) = xi.template segment<3>(3);
    for (size_t i = 0; i < n; ++i) {
      const int offset = 4 + static_cast<int>(3 * i);
      result.template block<3, 3>(offset, offset) =
          Rot3::Hat(xi.template segment<3>(6 + 3 * i));
    }
    return result;
  }

  /// vee operator for the Lie algebra
  static TangentVector Vee(const LieAlgebra& X) {
    TangentVector xi;
    xi.template head<3>() = Rot3::Vee(X.template block<3, 3>(0, 0));
    xi.template segment<3>(3) = X.template block<3, 1>(0, 3);
    for (size_t i = 0; i < n; ++i) {
      const int offset = 4 + static_cast<int>(3 * i);
      xi.template segment<3>(6 + 3 * i) =
          Rot3::Vee(X.template block<3, 3>(offset, offset));
    }
    return xi;
  }

  struct ChartAtOrigin {
    static Group Retract(const TangentVector& xi,
                         ChartJacobian Hxi = ChartJacobian()) {
      return Group::Expmap(xi, Hxi);
    }

    static TangentVector Local(const Group& g,
                               ChartJacobian Hg = ChartJacobian()) {
      return Group::Logmap(g, Hg);
    }
  };

  /// Group multiplication
  Group operator*(const Group<n>& other) const {
    std::array<Rot3, n> newB;
    for (size_t i = 0; i < n; i++) {
      newB[i] = B[i] * other.B[i];
    }
    return Group(A * other.A, a + Rot3::Hat(A.matrix() * Rot3::Vee(other.a)),
                 newB);
  }

  /// Group inverse
  Group inverse() const {
    Matrix3 inverseA = A.inverse().matrix();
    std::array<Rot3, n> inverseB;
    for (size_t i = 0; i < n; i++) {
      inverseB[i] = B[i].inverse();
    }
    return Group(A.inverse(), -Rot3::Hat(inverseA * Rot3::Vee(a)), inverseB);
  }

  /// Exponential map of the tangent space elements to the group
  static Group Expmap(const TangentVector& x,
                      OptionalJacobian<dimension, dimension> H =
                          OptionalJacobian<dimension, dimension>()) {
    if (x.size() != static_cast<Eigen::Index>(6 + 3 * n)) {
      throw std::invalid_argument("Vector size mismatch for group exponential");
    }
    Rot3 A = Rot3::Expmap(x.template head<3>());
    Vector3 a_vee = Rot3::ExpmapDerivative(-x.template head<3>()) *
                    x.template segment<3>(3);
    Matrix3 a = Rot3::Hat(a_vee);
    std::array<Rot3, n> B;
    for (size_t i = 0; i < n; i++) {
      B[i] = Rot3::Expmap(x.template segment<3>(6 + 3 * i));
    }
    if (H) *H = Eigen::Matrix<double, dimension, dimension>::Zero();
    return Group(A, a, B);
  }

  static TangentVector Logmap(const Group& g,
                              OptionalJacobian<dimension, dimension> H = {}) {
    // 1) Create the identity state and apply group action to it.
    //    We assume State<N>::identity() exists and operator*(Group, State) is
    //    defined as the group action (or provide a groupAction(g, xi) helper).
    State<n> xi0 = State<n>::identity();

    // If you have a group action function (g * state) available:
    State<n> xi_transformed = g * xi0;  // or groupAction(g, xi0)

    // 2) Compute local coordinates between identity and transformed state:
    TangentVector result = xi0.localCoordinates(xi_transformed);

    // 3) If Jacobian requested, compute numeric Jacobian of the map Group ->
    // Vector
    if (H) {
      // lambda: maps Group -> Vector
      auto mapGtoVec = [&xi0](const Group& gg) {
        State<n> x_trans = gg * xi0;  // group action
        return xi0.localCoordinates(x_trans);
      };

      // Use gtsam numerical derivative helper (type-deduction)
      *H = gtsam::numericalDerivative11<TangentVector, Group>(
          std::function<TangentVector(const Group&)>(mapGtoVec), g);
    }

    return result;
  }

  void print(const std::string& s = "") const {
    std::cout << s << "\nA:\n"
              << A.matrix() << "\na (vee): " << Rot3::Vee(a).transpose()
              << std::endl;
    for (size_t i = 0; i < n; ++i) {
      std::cout << "B[" << i << "]:\n" << B[i].matrix() << std::endl;
    }
  }

  bool equals(const Group& other, double tol = 1e-9) const {
    if (!A.equals(other.A, tol)) return false;
    if (!BEquals(other, tol)) return false;
    return (Rot3::Vee(a) - Rot3::Vee(other.a)).norm() <= tol;
  }

 private:
  bool BEquals(const Group& other, double tol) const {
    for (size_t i = 0; i < n; ++i) {
      if (!B[i].equals(other.B[i], tol)) return false;
    }
    return true;
  }
};

//========================================================================
// Helper Functions Implementation
//========================================================================

/**
 * Implements group actions on the states
 * @param X A symmetry group element Group consisting of the attitude, bias and
 * the calibration components X.a -> Rotation matrix containing the attitude X.b
 * -> A skew-symmetric matrix representing bias X.B -> A vector of Rotation
 * matrices for the calibration components
 * @param xi State object
 * xi.R -> Attitude (Rot3)
 * xi.b -> Gyroscope Bias(Vector 3)
 * xi.S -> Vector of calibration matrices(Rot3)
 * @return Transformed state
 * Uses the Rot3 inverse and Vee functions
 */
template <size_t N>
State<N> operator*(const Group<N>& X, const State<N>& xi) {
  std::array<Rot3, N> new_S;

  for (size_t i = 0; i < N; i++) {
    new_S[i] = X.A.inverse() * xi.S[i] * X.B[i];
  }

  return State<N>(xi.R * X.A, X.A.inverse().matrix() * (xi.b - Rot3::Vee(X.a)),
                  new_S);
}

/**
 * Transforms the mathematical input (ω, 0) between frames
 * @param X A symmetry group element X with the components
 * @param u Mathematical input vector (ω, 0)
 * @return Transformed input vector
 * Uses Rot3 Inverse, matrix and Vee functions and is critical for maintaining
 * the input equivariance
 */
template <size_t N>
Vector6 velocityAction(const Group<N>& X, const Vector6& u) {
  Vector6 result;
  result.head<3>() = X.A.inverse().matrix() * (u.head<3>() - Rot3::Vee(X.a));
  result.tail<3>() = Z_3x1;  // Virtual input remains zero
  return result;
}

/**
 * Transforms the Direction measurements based on the calibration type ( Eqn 6)
 * @param X Group element X
 * @param y Direction measurement y
 * @param idx Calibration index
 * @return Transformed direction
 * Uses Rot3 inverse, matrix and Unit3 unitvector functions
 */
template <size_t N>
Vector3 outputAction(const Group<N>& X, const Unit3& y, int idx) {
  if (idx == -1) {
    return X.A.inverse().matrix() * y.unitVector();
  } else {
    if (idx >= static_cast<int>(N)) {
      throw std::out_of_range("Calibration index out of range");
    }
    return X.B[idx].inverse().matrix() * y.unitVector();
  }
}

/**
 * Computes the differential of a state action at the identity of the symmetry
 * group
 * @param xi State object Xi representing the point at which to evaluate the
 * differential
 * @return A matrix representing the jacobian of the state action
 */
template <size_t N>
Matrix stateActionDiff(const State<N>& xi) {
  return gtsam::numericalDerivative11<Vector, Group<N>>(
      [&xi](const Group<N>& g) { return xi.localCoordinates(g * xi); },
      gtsam::traits<Group<N>>::Identity());
}

/**
 * Geometry class encapsulating the ABC system dynamics and measurement models
 */
template <size_t N>
struct Geometry {
  using InputType = Vector6;  // Mathematical input (ω, 0)
  using GType = Group<N>;
  using MType = State<N>;
  using TangentVector = typename GType::TangentVector;
  static MType identityState() { return MType::identity(); }
  static MType groupAction(const GType& g, const MType& x) { return g * x; }

  /**
   * Compute the lifted tangent vector from state and input.
   * @param xi Current state on the manifold (including orientation, bias, and
   * sensor rotations).
   * @param u Mathematical input vector (ω, 0)
   * @return TangentVector Lifted vector in the Lie algebra used for
   * propagation.
   */
  static TangentVector lift(const MType& xi, const InputType& u) {
    return xi.lift(u);
  }

  /**
   * Computes the discrete time state transition matrix
   * @param u Angular velocity
   * @param dt time step
   * @return State transition matrix in discrete time
   */
  static Matrix stateTransitionMatrix(const Vector6& u, double dt,
                                      GType X_hat) {
    Matrix3 W0 =
        Rot3::Hat(velocityAction(X_hat.inverse(), u).template head<3>());
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
   * Uses Matrix zero and Identity functions
   */
  static Matrix stateMatrixA(const GType& X_hat, const Vector6& u) {
    Matrix3 W0 =
        Rot3::Hat(velocityAction(X_hat.inverse(), u).template head<3>());

    Matrix A1 = Matrix::Zero(6, 6);
    A1.block<3, 3>(0, 3) = -I_3x3;
    A1.block<3, 3>(3, 3) = W0;

    std::vector<Matrix> blocks{A1};
    blocks.insert(blocks.end(), N, W0);
    return gtsam::diag(blocks);
  }

  /// Computes the input uncertainty propagation matrix
  static Matrix inputMatrix(GType X_hat) {
    Matrix B1 = gtsam::diag({X_hat.A.matrix(), X_hat.A.matrix()});
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
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
  static Matrix inputMatrixBt(GType X_hat) {
    Matrix B1 = gtsam::diag({X_hat.A.matrix(), X_hat.A.matrix()});
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
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
      return X_hat.B[idx].matrix();
    } else {
      return X_hat.A.matrix();
    }
  }

  static constexpr int n_cal = N;
};

//========================================================================
// Free-function adapters for EqF and generic EKF code
//========================================================================

/**
 * @brief Discrete-time state transition matrix for the EqF.
 *
 * Thin wrapper around Geometry<N>::stateTransitionMatrix
 */
template <size_t N>
Matrix stateTransitionMatrix(const Group<N>& X_hat, const Vector6& u,
                             double dt) {
  return Geometry<N>::stateTransitionMatrix(u, dt, X_hat);
}

/**
 * @brief Input uncertainty propagation matrix Bt for the EqF.
 *
 * Wraps Geometry<N>::inputMatrixBt
 */
template <size_t N>
Matrix inputMatrixBt(const Group<N>& X_hat) {
  return Geometry<N>::inputMatrixBt(X_hat);
}

/**
 * @brief Linearized measurement matrix C for a direction measurement.
 *
 * Wraps Geometry<N>::measurementMatrixC
 */
template <size_t N>
Matrix measurementMatrixC(const Unit3& d, int idx, const Group<N>& /*X_hat*/) {
  return Geometry<N>::measurementMatrixC(d, idx);
}

/**
 * @brief Measurement uncertainty propagation matrix Dt.
 *
 * Wraps Geometry<N>::outputMatrixDt
 */
template <size_t N>
Matrix outputMatrixDt(const Group<N>& X_hat, int idx) {
  return Geometry<N>::outputMatrixDt(idx, X_hat);
}
}  // namespace abc

template <size_t N>
struct traits<abc::Group<N>>
    : internal::MatrixLieGroup<abc::Group<N>, 4 + 3 * N> {};

template <size_t N>
struct traits<const abc::Group<N>>
    : internal::MatrixLieGroup<abc::Group<N>, 4 + 3 * N> {};
}  // namespace gtsam
