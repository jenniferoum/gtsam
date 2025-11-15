/**
 * @file ABC.h
 * @brief Core components for Attitude-Bias-Calibration systems
 *
 * This file contains fundamental components and utilities for the ABC system
 * based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
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
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

namespace gtsam {
namespace abc {

//========================================================================
// Core Data Types
//========================================================================

/// Input data struct for the Biased Attitude System (stores sensor data and
/// noise)
struct InputData {
  Vector3 w;                  /// Angular velocity (3-vector)
  Matrix Sigma;               /// Noise covariance (6x6 matrix)
  static InputData random();  /// Random input
  Matrix3 W() const {         /// Return w as a skew symmetric matrix
    return Rot3::Hat(w);
  }

  /// Convert to mathematical input vector (ω, 0)
  Vector6 toInputVector() const {
    Vector6 u;
    u.head<3>() = w;
    u.tail<3>() = Vector3::Zero();  // Virtual input
    return u;
  }
};

/// Measurement struct
struct Measurement {
  Unit3 y;           /// Measurement direction in sensor frame
  Unit3 d;           /// Known direction in global frame
  Matrix3 Sigma;     /// Covariance matrix of the measurement
  int cal_idx = -1;  /// Calibration index (-1 for calibrated sensor)
};

/// State class representing the state of the Biased Attitude System
template <size_t N>
class State {
 public:
  Rot3 R;                 // Attitude rotation matrix R
  Vector3 b;              // Gyroscope bias b
  std::array<Rot3, N> S;  // Sensor calibrations S

  /// Constructor
  State(const Rot3& R = Rot3::Identity(), const Vector3& b = Vector3::Zero(),
        const std::array<Rot3, N>& S = std::array<Rot3, N>{})
      : R(R), b(b), S(S) {}

  /// Identity function
  static State identity() {
    std::array<Rot3, N> S_id{};
    S_id.fill(Rot3::Identity());
    return State(Rot3::Identity(), Vector3::Zero(), S_id);
  }
  /**
   * Compute Local coordinates in the state relative to another state.
   * @param other The other state
   * @return Local coordinates in the tangent space
   */
  Vector localCoordinates(const State<N>& other) const {
    Vector eps(6 + 3 * N);

    // First 3 elements - attitude
    eps.head<3>() = Rot3::Logmap(R.between(other.R));
    // Next 3 elements - bias
    // Next 3 elements - bias
    eps.segment<3>(3) = other.b - b;

    // Remaining elements - calibrations
    for (size_t i = 0; i < N; i++) {
      eps.segment<3>(6 + 3 * i) = Rot3::Logmap(S[i].between(other.S[i]));
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
    Rot3 newR = R * Rot3::Expmap(v.head<3>());
    Vector3 newB = b + v.segment<3>(3);
    std::array<Rot3, N> newS;
    for (size_t i = 0; i < N; i++) {
      newS[i] = S[i] * Rot3::Expmap(v.segment<3>(6 + 3 * i));
    }
    return State(newR, newB, newS);
  }

  /**
   * Compute the lifted tangent vector from state and input.
   * This implements the lift operation from the equivariant filter paper.
   * @param u Mathematical input vector (ω, 0) where first 3 are angular
   * velocity
   * @return Vector Lifted vector in the Lie algebra used for propagation.
   */
  Vector lift(const Vector6& u) const {
    Vector L = Vector::Zero(6 + 3 * N);

    Vector3 w = u.head<3>();

    L.head<3>() = w - b;

    L.segment<3>(3) = -Rot3::Hat(w) * b;

    Vector3 corrected_w = w - b;
    for (size_t i = 0; i < N; i++) {
      L.segment<3>(6 + 3 * i) = S[i].inverse().matrix() * corrected_w;
    }

    return L;
  }

  /**
   * Convenience overload that accepts InputData
   */
  Vector lift(const InputData& data) const {
    return lift(data.toInputVector());
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
struct Group {
  Rot3 A;                 /// First SO(3) element
  Matrix3 a;              /// so(3) element (skew-symmetric matrix)
  std::array<Rot3, n> B;  /// List of SO(3) elements for calibration
  static constexpr int dimension = 6 + 3 * n;
  using TangentVector = Eigen::Matrix<double, dimension, 1>;
  static constexpr int numSensors = n;
  /// Initialize the symmetry Group
  Group(const Rot3& A = Rot3::Identity(), const Matrix3& a = Matrix3::Zero(),
        const std::array<Rot3, n>& B = std::array<Rot3, n>{})
      : A(A), a(a), B(B) {}

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
  Group inv() const {
    Matrix3 Ainv = A.inverse().matrix();
    std::array<Rot3, n> Binv;
    for (size_t i = 0; i < n; i++) {
      Binv[i] = B[i].inverse();
    }
    return Group(A.inverse(), -Rot3::Hat(Ainv * Rot3::Vee(a)), Binv);
  }

  Group inverse() const { return inv(); }

  /// Identity element
  static Group identity(int N) {  // todo: N is not used here, possibly remove
    std::array<Rot3, n> B;
    B.fill(Rot3::Identity());
    return Group(Rot3::Identity(), Matrix3::Zero(), B);
  }

  /// Exponential map of the tangent space elements to the group
  static Group exp(const Vector& x) {
    if (x.size() != static_cast<Eigen::Index>(6 + 3 * n)) {
      throw std::invalid_argument("Vector size mismatch for group exponential");
    }
    Rot3 A = Rot3::Expmap(x.head<3>());
    Vector3 a_vee = Rot3::ExpmapDerivative(-x.head<3>()) * x.segment<3>(3);
    Matrix3 a = Rot3::Hat(a_vee);
    std::array<Rot3, n> B;
    for (size_t i = 0; i < n; i++) {
      B[i] = Rot3::Expmap(x.segment<3>(6 + 3 * i));
    }
    return Group(A, a, B);
  }

  /// Retract a tangent vector back to the manifold using Expmap
  Group retract(const TangentVector& v,
                OptionalJacobian<dimension, dimension> H = {},
                OptionalJacobian<dimension, dimension> Hv = {}) const {
    return gtsam::traits<Group>::Compose(*this,
                                         gtsam::traits<Group>::Expmap(v));
  }

  // Adjoint matrix of this group element (for SE(3) or similar)
  Eigen::Matrix<double, dimension, dimension> AdjointMap() const {
    // TODO: implement properly for your group structure.
    // Placeholder: identity matrix compiles but is mathematically wrong.
    return Eigen::Matrix<double, dimension, dimension>::Identity();
  }

  static Eigen::Matrix<double, dimension, 1> Logmap(
      const Group& g, OptionalJacobian<dimension, dimension> H = {}) {
    // 1) Create the identity state and apply group action to it.
    //    We assume State<N>::identity() exists and operator*(Group, State) is
    //    defined as the group action (or provide a groupAction(g, xi) helper).
    State<n> xi0 = State<n>::identity();

    // If you have a group action function (g * state) available:
    State<n> xi_transformed = g * xi0;  // or groupAction(g, xi0)

    // 2) Compute local coordinates between identity and transformed state:
    Vector logv = xi0.localCoordinates(xi_transformed);

    // 3) If Jacobian requested, compute numeric Jacobian of the map Group ->
    // Vector
    if (H) {
      // lambda: maps Group -> Vector
      auto mapGtoVec = [&xi0](const Group& gg) {
        State<n> x_trans = gg * xi0;           // group action
        return xi0.localCoordinates(x_trans);  // returns Vector dimension x 1
      };

      // Use gtsam numerical derivative helper (type-deduction)
      *H = gtsam::numericalDerivative11<Vector, Group>(
          std::function<Vector(const Group&)>(mapGtoVec), g);
    }

    return logv;
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
  result.tail<3>() = Vector3::Zero();  // Virtual input remains zero
  return result;
}
/**
 * Transforms the Direction measurements based on the calibration type ( Eqn 6)
 * @param X Group element X
 * @param y Direction measurement y
 * @param idx Calibration index
 * @return Transformed direction
 * Uses Rot3 inverse, matric and Unit3 unitvector functions
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
 * @brief Calculates the Jacobian matrix using central difference approximation
 * @param f Vector function f
 * @param x The point at which Jacobian is evaluated
 * @return Matrix containing numerical partial derivatives of f at x
 * Uses Vector's size() and Zero(), Matrix's Zero() and col() methods
 */
Matrix numericalDifferential(std::function<Vector(const Vector&)> f,
                             const Vector& x) {
  double h = 1e-6;
  Vector fx = f(x);
  int n = fx.size();
  int m = x.size();
  Matrix Df = Matrix::Zero(n, m);

  for (int j = 0; j < m; j++) {
    Vector ej = Vector::Zero(m);
    ej(j) = 1.0;

    Vector fplus = f(x + h * ej);
    Vector fminus = f(x - h * ej);

    Df.col(j) = (fplus - fminus) / (2 * h);
  }

  return Df;
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

template <size_t N>
struct Geometry {
  using InputType = Vector6;  // Mathematical input (ω, 0)
  using Measurement = abc::Measurement;
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
  // TODO: new version of this function reduces precision, fails
  // Geometry_stateTransitionMatrix test case as a result
  static Matrix stateTransitionMatrix(const Vector6& u, double dt,
                                      GType X_hat) {
    Matrix A = stateMatrixA(X_hat, u);
    Matrix I = Matrix::Identity(A.rows(), A.cols());

    // Use truncated exponential for numerical stability if dt is small
    Matrix A2 = A * A;
    Matrix A3 = A2 * A;
    return I + dt * A + 0.5 * dt * dt * A2 + (1.0 / 6.0) * dt * dt * dt * A3;
    // return (A * dt).exp().eval();
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

  /// Computes the continuous-time process noise covariance in lifted
  /// coordinates
  static Matrix processNoise(const Matrix& Sigma) {
    std::vector<Matrix> blocks{Sigma};
    blocks.insert(blocks.end(), N, 1e-9 * I_3x3);
    return gtsam::diag(blocks);
  }

  /**
   * Computes the input uncertainty propagation matrix
   * @return
   * Uses the blockdiag matrix
   */
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
 * @brief Continuous-time process noise covariance in lifted coordinates.
 *
 * Wraps Geometry<N>::processNoise
 */
template <size_t N>
Matrix processNoise(const Group<N>& /*X_hat*/, const InputData& data) {
  return Geometry<N>::processNoise(data);
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
struct traits<abc::Group<N>> : internal::LieGroupTraits<abc::Group<N>> {
  using GType = abc::Group<N>;
  // dimension should exist on GType; if not, set to compile-time constant
  static constexpr int dimension = GType::dimension;

  // Useful aliases for Jacobian optionals and Matrix
  using OptionalJac = OptionalJacobian<dimension, dimension>;
  using MatrixDim = Eigen::Matrix<double, dimension, dimension>;

  // Identity
  static GType Identity() { return GType::identity(N); }

  // Compose with optional Jacobian outputs:
  static GType Compose(const GType& g1, const GType& g2,
                       OptionalJac Hg = OptionalJac(),
                       OptionalJac Hh = OptionalJac()) {
    // If caller requested Jacobians (Hg/Hh) we should fill them.
    // For now provide zero matrices as placeholders.
    if (Hg) *Hg = MatrixDim::Zero();
    if (Hh) *Hh = MatrixDim::Zero();
    return g1 * g2;  // your operator* already implements group multiplication
  }

  // Between (g1^{-1} * g2) with optional Jacobians
  static GType Between(const GType& g1, const GType& g2,
                       OptionalJac H1 = OptionalJac(),
                       OptionalJac H2 = OptionalJac()) {
    if (H1) *H1 = MatrixDim::Zero();
    if (H2) *H2 = MatrixDim::Zero();
    return g1.inv() * g2;  // or use g1.inverse() if that's your API
  }

  // Inverse with optional Jacobian
  static GType Inverse(const GType& g, OptionalJac H = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    return g.inv();
  }

  // Expmap (v -> Group) with optional Jacobian dExp/dv
  static GType Expmap(const Vector& v, OptionalJac H = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    return GType::exp(v);
  }

  // Local (Logmap): returns tangent vector (g1^-1 * g2). Optionally fill
  // jacobians.
  static Vector Local(const GType& g1, const GType& g2,
                      OptionalJac H1 = OptionalJac(),
                      OptionalJac H2 = OptionalJac()) {
    // If you already have a member or free function that returns local
    // coordinates, use it, e.g., g1.localCoordinates(g2) or State-based
    // approach.
    if (H1) *H1 = MatrixDim::Zero();
    if (H2) *H2 = MatrixDim::Zero();
    // Implement a sensible default: Logmap(Between(g1,g2))
    GType between = g1.inv() * g2;
    // If GType has a log/ln method use it; else, call static log if available:
    return GType::Logmap(between);  // replace with your actual logmap call
  }

  // Retract: move point g along tangent v
  static GType Retract(const GType& g, const Vector& v,
                       OptionalJac H = OptionalJac(),
                       OptionalJac Hv = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    if (Hv) *Hv = MatrixDim::Zero();
    // You can either use traits compose/exp or call a member
    // Use composition: g * Expmap(v)
    return Compose(g, Expmap(v));  // will call Compose and Expmap above
  }

  static void Print(const GType& g, const std::string& s = "") {
    std::cout << s << std::endl;
    std::cout << "A = " << g.A << std::endl;
    std::cout << "a = " << g.a << std::endl;
    for (size_t i = 0; i < GType::N; ++i) {
      std::cout << "B[" << i << "] = " << g.B[i] << std::endl;
    }
  }

  static bool Equals(const GType& g1, const GType& g2, double tol = 1e-9) {
    if (!g1.A.equals(g2.A, tol)) return false;
    if (!gtsam::assert_equal<Matrix3>(g1.a, g2.a, tol)) return false;
    for (size_t i = 0; i < GType::numSensors; ++i) {
      if (!g1.B[i].equals(g2.B[i], tol)) return false;
    }
    return true;
  }
};
}  // namespace gtsam
