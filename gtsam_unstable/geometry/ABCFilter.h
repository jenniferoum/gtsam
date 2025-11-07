/**
 * @file ABCFilter.h
 * @brief Filter infrastructure for Attitude-Bias-Calibration Equivariant Filter
 *
 * This file contains the filter-specific infrastructure (measurements, inputs,
 * geometry adapter) for the ABC Equivariant Filter. The pure geometric structures
 * are in ABC.h.
 * 
 * Based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */
#ifndef ABC_FILTER_H
#define ABC_FILTER_H

#include <gtsam_unstable/geometry/ABC.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

namespace gtsam {
namespace abc_eqf_lib {

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
inline Matrix blockDiag(const Matrix& A, const Matrix& B) {
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
inline Matrix repBlock(const Matrix& A, int n) {
  if (n <= 0) return Matrix();

  Matrix result = A;
  for (int i = 1; i < n; i++) {
    result = blockDiag(result, A);
  }
  return result;
}

//========================================================================
// Filter Data Types
//========================================================================

/// Input data struct for the Biased Attitude System (stores sensor data and noise)
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
  int cal_idx = -1;  /// Calibration index (-1 for uncalibrated sensor)
};

//========================================================================
// Filter-specific State Extensions
//========================================================================

/**
 * Adds filter-specific methods to State<N> via helper functions
 */
template <size_t N>
struct StateFilterOps {
  /**
   * Compute the lifted tangent vector from state and input.
   * This implements the lift operation from the equivariant filter paper.
   * @param state The current state
   * @param u Mathematical input vector (ω, 0) where first 3 are angular velocity
   * @return Vector Lifted vector in the Lie algebra used for propagation.
   */
  static Vector lift(const State<N>& state, const Vector6& u) {
    Vector L = Vector::Zero(6 + 3 * N);

    Vector3 w = u.head<3>();
    
    // First 3 elements: ω - b (corrected angular velocity)
    L.head<3>() = w - state.b;

    // Next 3 elements: -ω^ * b (bias dynamics)
    L.segment<3>(3) = -Rot3::Hat(w) * state.b;

    // Remaining elements: S_i^{-1} * (ω - b) for each calibration
    Vector3 corrected_w = w - state.b;
    for (size_t i = 0; i < N; i++) {
      L.segment<3>(6 + 3 * i) = state.S[i].inverse().matrix() * corrected_w;
    }

    return L;
  }
  
  /**
   * Convenience overload that accepts InputData
   */
  static Vector lift(const State<N>& state, const InputData& data) {
    return lift(state, data.toInputVector());
  }

  /**
   * Computes the linearized measurement matrix for a direction measurement
   * @param d reference direction
   * @param idx Calibration index (-1 for uncalibrated sensor)
   * @return Measurement matrix
   */
  static Matrix measurementMatrix(const Unit3& d, int idx) {
    Matrix Cc = Matrix::Zero(3, 3 * N);

    // If the measurement is related to a sensor that has a calibration state
    if (idx >= 0 && idx < static_cast<int>(N)) {
      Cc.block<3, 3>(0, 3 * idx) = Rot3::Hat(d.unitVector());
    }

    Matrix3 wedge_d = Rot3::Hat(d.unitVector());

    // Build the combined matrix
    Matrix temp(3, 6 + 3 * N);
    temp.block<3, 3>(0, 0) = wedge_d;
    temp.block<3, 3>(0, 3) = Matrix3::Zero();
    temp.block(0, 6, 3, 3 * N) = Cc;

    return temp;
  }
};

//========================================================================
// Geometry Adapter for Equivariant Filter
//========================================================================

template <size_t N>
struct ABCGeometry {
  using InputType = Vector6;  // Mathematical input (ω, 0)
  using InputDataType = abc_eqf_lib::InputData;  // Data with noise params
  using Measurement = abc_eqf_lib::Measurement;
  using GType = Group<N>;
  using MType = State<N>;
  using TangentVector = typename GType::TangentVector;
  
  static constexpr int n_cal = N;
  static const Matrix3 I_3x3;
  
  static MType identityState() { return MType::identity(); }
  static MType groupAction(const GType& g, const MType& x) { return g * x; }

  /**
   * Compute the lifted tangent vector from state and input.
   */
  static TangentVector lift(const MType& xi, const InputType& u) {
    return StateFilterOps<N>::lift(xi, u);
  }

  /**
   * Computes the discrete time state transition matrix
   */
  static Matrix stateTransitionMatrix(const InputDataType& data, double dt, GType X_hat) {
    InputType u = data.toInputVector();
    Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inv(), u).template head<3>());
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
   */
  static Matrix stateMatrixA(const GType& X_hat, const InputDataType& data) {
    InputType u = data.toInputVector();
    Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inverse(), u).template head<3>());

    Matrix A1 = Matrix::Zero(6, 6);
    A1.block<3, 3>(0, 3) = -I_3x3;
    A1.block<3, 3>(3, 3) = W0;

    Matrix A2 = repBlock(W0, N);
    return blockDiag(A1, A2);
  }
  
  static Matrix inputMatrix(GType X_hat) {
    Matrix B1 = blockDiag(X_hat.A.matrix(), X_hat.A.matrix());
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
    }

    return blockDiag(B1, B2);
  }

  static Matrix processNoise(const InputDataType& data) {
    return blockDiag(data.Sigma, repBlock(1e-9 * I_3x3, N));
  }

  static Matrix inputMatrixBt(GType X_hat) {
    Matrix B1 = blockDiag(X_hat.A.matrix(), X_hat.A.matrix());
    Matrix B2(3 * N, 3 * N);

    for (size_t i = 0; i < N; ++i) {
      B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
    }

    return blockDiag(B1, B2);
  }
  
  /**
   * Computes the linearized measurement matrix
   */
  static Matrix measurementMatrixC(const Unit3& d, int idx) {
    Matrix Cc = Matrix::Zero(3, 3 * N);

    if (idx >= 0) {
      Cc.block<3, 3>(0, 3 * idx) = Rot3::Hat(d.unitVector());
    }

    Matrix3 wedge_d = Rot3::Hat(d.unitVector());

    Matrix temp(3, 6 + 3 * N);
    temp.block<3, 3>(0, 0) = wedge_d;
    temp.block<3, 3>(0, 3) = Matrix3::Zero();
    temp.block(0, 6, 3, 3 * N) = Cc;

    return wedge_d * temp;
  }
  
  /**
   * Computes the measurement uncertainty propagation matrix
   */
  static Matrix outputMatrixDt(int idx, Group<N> X_hat) {
    if (idx >= 0) {
      if (idx >= static_cast<int>(N)) {
        throw std::out_of_range("Calibration index out of range");
      }
      return X_hat.B[idx].matrix();
    } else {
      return X_hat.A.matrix();
    }
  }
};

template <size_t N>
const Matrix3 ABCGeometry<N>::I_3x3 = Matrix3::Identity();

/**
 * @brief Calculates the Jacobian matrix using central difference approximation
 * @param f Vector function f
 * @param x The point at which Jacobian is evaluated
 * @return Matrix containing numerical partial derivatives of f at x
 */
inline Matrix numericalDifferential(std::function<Vector(const Vector&)> f,
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
 * Computes the differential of a state action at the identity of the symmetry group
 */
template <size_t N>
inline Matrix stateActionDiff(const State<N>& xi) {
  return gtsam::numericalDerivative11<Vector, Group<N>>(
      [&xi](const Group<N>& g) { return xi.localCoordinates(g * xi); },
      gtsam::traits<Group<N>>::Identity());
}

}  // namespace abc_eqf_lib
}  // namespace gtsam

#endif  // ABC_FILTER_H

