/**
 * @file ABC.h
 * @brief Core components for Attitude-Bias-Calibration systems
 *
 * This file contains fundamental components and utilities for the ABC system
 * based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */
#ifndef ABC_H
#define ABC_H
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
namespace abc_eqf_lib {
using namespace std;
using namespace gtsam;

//========================================================================
// Utility Functions
//========================================================================

/**
 * @brief Creates a block diagonal matrix from input matrices
 * @param A Matrix A
 * @param B Matrix B
 * @return A single consolidated matrix with A in the top left and B in the
 * bottom right
 */
Matrix blockDiag(const Matrix& A, const Matrix& B) {
  if (A.size() == 0) {
    return B;
  } else if (B.size() == 0) {
    return A;
  } else {
    Matrix result(A.rows() + B.rows(), A.cols() + B.cols());
    result.setZero();
    result.block(0, 0, A.rows(), A.cols()) = A;
    result.block(A.rows(), A.cols(), B.rows(), B.cols()) = B;
    return result;
  }
}

/**
 * @brief Creates a block diagonal matrix by repeating a matrix 'n' times
 * @param A A matrix
 * @param n Number of times to be repeated
 * @return Block diag matrix with A repeated 'n' times
 */
Matrix repBlock(const Matrix& A, int n) {
  if (n <= 0) return Matrix();

  Matrix result = A;
  for (int i = 1; i < n; i++) {
    result = blockDiag(result, A);
  }
  return result;
}

//========================================================================
// Core Data Types
//========================================================================

/// Input struct for the Biased Attitude System

struct Input {
  Vector3 w;              /// Angular velocity (3-vector)
  Matrix Sigma;           /// Noise covariance (6x6 matrix)
  static Input random();  /// Random input
  Matrix3 W() const {     /// Return w as a skew symmetric matrix
    return Rot3::Hat(w);
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
};

//========================================================================
// Symmetry Group
//========================================================================

/**
 * Symmetry group (SO(3) |x so(3)) x SO(3) x ... x SO(3)
 * Each element of the B list is associated with a calibration state
 */
template <size_t N>
struct G {
  Rot3 A;                 /// First SO(3) element
  Matrix3 a;              /// so(3) element (skew-symmetric matrix)
  std::array<Rot3, N> B;  /// List of SO(3) elements for calibration
  static constexpr int dimension = 6 + 3 * N;
  using TangentVector = Eigen::Matrix<double, dimension, 1>;

  /// Initialize the symmetry group G
  G(const Rot3& A = Rot3::Identity(), const Matrix3& a = Matrix3::Zero(),
    const std::array<Rot3, N>& B = std::array<Rot3, N>{})
      : A(A), a(a), B(B) {}

  /// Group multiplication
  G operator*(const G<N>& other) const {
    std::array<Rot3, N> newB;
    for (size_t i = 0; i < N; i++) {
      newB[i] = B[i] * other.B[i];
    }
    return G(A * other.A, a + Rot3::Hat(A.matrix() * Rot3::Vee(other.a)), newB);
  }

  /// Group inverse
  G inv() const {
    Matrix3 Ainv = A.inverse().matrix();
    std::array<Rot3, N> Binv;
    for (size_t i = 0; i < N; i++) {
      Binv[i] = B[i].inverse();
    }
    return G(A.inverse(), -Rot3::Hat(Ainv * Rot3::Vee(a)), Binv);
  }

  /// Identity element
  static G identity(int n) {
    std::array<Rot3, N> B;
    B.fill(Rot3::Identity());
    return G(Rot3::Identity(), Matrix3::Zero(), B);
  }

  /// Exponential map of the tangent space elements to the group
  static G exp(const Vector& x) {
    if (x.size() != static_cast<Eigen::Index>(6 + 3 * N)) {
      throw std::invalid_argument("Vector size mismatch for group exponential");
    }
    Rot3 A = Rot3::Expmap(x.head<3>());
    Vector3 a_vee = Rot3::ExpmapDerivative(-x.head<3>()) * x.segment<3>(3);
    Matrix3 a = Rot3::Hat(a_vee);
    std::array<Rot3, N> B;
    for (size_t i = 0; i < N; i++) {
      B[i] = Rot3::Expmap(x.segment<3>(6 + 3 * i));
    }
    return G(A, a, B);
  }

  /// Retract a tangent vector back to the manifold using Expmap
  G retract(const TangentVector& v,
            OptionalJacobian<dimension, dimension> H = boost::none,
            OptionalJacobian<dimension, dimension> Hv = boost::none) const {
    return gtsam::traits<G>::Compose(*this, gtsam::traits<G>::Expmap(v));
  }
};

//========================================================================
// Helper Functions Implementation
//========================================================================
/**
 * Implements group actions on the states
 * @param X A symmetry group element G consisting of the attitude, bias and the
 * calibration components X.a -> Rotation matrix containing the attitude X.b ->
 * A skew-symmetric matrix representing bias X.B -> A vector of Rotation
 * matrices for the calibration components
 * @param xi State object
 * xi.R -> Attitude (Rot3)
 * xi.b -> Gyroscope Bias(Vector 3)
 * xi.S -> Vector of calibration matrices(Rot3)
 * @return Transformed state
 * Uses the Rot3 inverse and Vee functions
 */
template <size_t N>
State<N> operator*(const G<N>& X, const State<N>& xi) {
  std::array<Rot3, N> new_S;

  for (size_t i = 0; i < N; i++) {
    new_S[i] = X.A.inverse() * xi.S[i] * X.B[i];
  }

  return State<N>(xi.R * X.A, X.A.inverse().matrix() * (xi.b - Rot3::Vee(X.a)),
                  new_S);
}
/**
 * Transforms the angular velocity measurements b/w frames
 * @param X A symmetry group element X with the components
 * @param u Inputs
 * @return Transformed inputs
 * Uses Rot3 Inverse, matrix and Vee functions and is critical for maintaining
 * the input equivariance
 */
template <size_t N>
Input velocityAction(const G<N>& X, const Input& u) {
  return Input{X.A.inverse().matrix() * (u.w - Rot3::Vee(X.a)), u.Sigma};
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
Vector3 outputAction(const G<N>& X, const Unit3& y, int idx) {
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
  return gtsam::numericalDerivative11<Vector, G<N>>(
      [&xi](const G<N>& g) { return xi.localCoordinates(g * xi); },
      gtsam::traits<G<N>>::Identity());
}

template <size_t N>
struct ABCGeometry {
  using Input = abc_eqf_lib::Input;
  using Measurement = abc_eqf_lib::Measurement;
  using GType = G<N>;
  using MType = State<N>;
  using TangentVector = typename GType::TangentVector;
  static MType identityState() { return MType::identity(); }
  static MType groupAction(const GType& g, const MType& x) { return g * x; }

  /**
   * Compute the lifted tangent vector from state and input.
   * @param xi Current state on the manifold (including orientation, bias, and
   * sensor rotations).
   * @param u Input measurement containing angular velocity and its covariance.
   * @return TangentVector Lifted vector in the Lie algebra used for
   * propagation.
   */
  static TangentVector lift(const MType& xi, const Input& u) {
    TangentVector L = TangentVector::Zero();

    // First 3 elements
    L.template head<3>() = u.w - xi.b;

    // Next 3 elements
    L.template segment<3>(3) = -u.W() * xi.b;

    // Remaining elements
    for (size_t i = 0; i < N; i++) {
      L.template segment<3>(6 + 3 * i) =
          xi.S[i].inverse().matrix() * L.template head<3>();
    }

    return L;
  }

  /**
   * Computes the discrete time state transition matrix
   * @param u Angular velocity
   * @param dt time step
   * @return State transition matrix in discrete time
   */
  static Matrix stateTransitionMatrix(const Input& u, double dt, GType X_hat) {
    Matrix3 W0 = velocityAction(X_hat.inv(), u).W();
    Matrix Phi1 = Matrix::Zero(6, 6);

    Matrix3 Phi12 = -dt * (I_3x3 + (dt / 2) * W0 + ((dt * dt) / 6) * W0 * W0);
    Matrix3 Phi22 = I_3x3 + dt * W0 + ((dt * dt) / 2) * W0 * W0;

    Phi1.block<3, 3>(0, 0) = I_3x3;
    Phi1.block<3, 3>(0, 3) = Phi12;
    Phi1.block<3, 3>(3, 3) = Phi22;
    Matrix Phi2 = repBlock(Phi22, N);
    return blockDiag(Phi1, Phi2);
  }
  /**
   * Computes linearized continuous time state matrix
   * @param u Angular velocity
   * @return Linearized state matrix
   * Uses Matrix zero and Identity functions
   */
  static Matrix stateMatrixA(const GType& X_hat, const Input& u) {
    Matrix3 W0 = velocityAction(X_hat.inverse(), u).W();

    Matrix A1 = Matrix::Zero(6, 6);
    A1.block<3, 3>(0, 3) = -I_3x3;
    A1.block<3, 3>(3, 3) = W0;

    Matrix A2 = repBlock(W0, N);
    return blockDiag(A1, A2);  // Now valid inside geometry
  }
  static Matrix inputMatrix(GType X_hat) {
    Matrix B1 = blockDiag(X_hat.A.matrix(), X_hat.A.matrix());
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
    }

    return blockDiag(B1, B2);
  }

  static Matrix processNoise(const Input& u) {
    return blockDiag(u.Sigma, repBlock(1e-9 * I_3x3, N));
  }

  /**
   * Computes the input uncertainty propagation matrix
   * @return
   * Uses the blockdiag matrix
   */
  static Matrix inputMatrixBt(GType X_hat) {
    Matrix B1 = blockDiag(X_hat.A.matrix(), X_hat.A.matrix());
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
    }

    return blockDiag(B1, B2);
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
    if (idx >= 0) {
      // Set the correct 3x3 block in Cc
      Cc.block<3, 3>(0, 3 * idx) = Rot3::Hat(d.unitVector());
    }

    Matrix3 wedge_d = Rot3::Hat(d.unitVector());

    // Create the combined matrix
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
  static Matrix outputMatrixDt(int idx, G<N> X_hat) {
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

}  // namespace abc_eqf_lib

template <size_t N>
struct traits<abc_eqf_lib::G<N>> : internal::LieGroupTraits<abc_eqf_lib::G<N>> {
  using GType = abc_eqf_lib::G<N>;

  static GType Identity() { return GType::identity(N); }

  static GType Compose(const GType& g1, const GType& g2) { return g1 * g2; }

  static GType Expmap(const Vector& v) { return GType::exp(v); }
};
}  // namespace gtsam

#endif  // ABC_H
