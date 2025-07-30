/**
 * @file ABC_EQF.h
 * @brief Header file for the Attitude-Bias-Calibration Equivariant Filter
 *
 * This file contains declarations for the Equivariant Filter (EqF) for attitude
 * estimation with both gyroscope bias and sensor extrinsic calibration, based
 * on the paper: "Overcoming Bias: Equivariant Filter Design for Biased Attitude
 * Estimation with Online Calibration" by Fornasier et al. Authors: Darshan
 * Rajasekaran & Jennifer Oum
 */

#ifndef EQUIVARIANTFILTER_H
#define EQUIVARIANTFILTER_H
#pragma once
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
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
namespace abc_eqf_lib {

using namespace std;
using namespace gtsam;

//========================================================================
// Equivariant Filter (EqF) Template Function Verification
//========================================================================

template <typename Geometry, typename M, typename Input>
class has_lift {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::lift(std::declval<M>(), std::declval<Input>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_groupAction {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::groupAction(std::declval<typename G::GType>(),
                                 std::declval<typename G::MType>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_stateTransitionMatrix {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::stateTransitionMatrix(std::declval<typename G::Input>(),
                                           std::declval<double>(),
                                           std::declval<typename G::GType>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_stateMatrixA {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::stateMatrixA(std::declval<const typename G::GType&>(),
                                  std::declval<const typename G::Input&>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_inputMatrix {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::inputMatrix(std::declval<typename G::GType>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_processNoise {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::processNoise(std::declval<const typename G::Input&>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_inputMatrixBt {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::inputMatrixBt(std::declval<typename G::GType>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_measurementMatrixC {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::measurementMatrixC(std::declval<const Unit3&>(),
                                        std::declval<int>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

template <typename Geometry>
class has_outputMatrixDt {
 private:
  template <typename G>
  static auto test(int)
      -> decltype(G::outputMatrixDt(std::declval<int>(),
                                    std::declval<typename G::GType>()),
                  std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<Geometry>(0))::value;
};

//========================================================================
// Equivariant Filter (EqF)
//========================================================================

/// Equivariant Filter (EqF) implementation
template <typename G, typename M, typename Geometry>
class EqF {
 private:
  G X_hat;                // Filter state
  Matrix Sigma;           // Error covariance
  M xi_0;                 // Origin state
  Matrix Dphi0;           // Differential of phi at origin
  Matrix InnovationLift;  // Innovation lift matrix

 public:
  /**
   * Initialize EqF
   * @param Sigma Initial covariance
   * @param m Number of sensors
   */
  EqF(const G& X0, const M& x0, const Matrix& Sigma, int m);

  /**
   * Return estimated state
   * @return Current state estimate
   */
  M stateEstimate() const;

  /**
   * Propagate the filter state
   * @param u Angular velocity measurement
   * @param dt Time step
   */
  void propagation(const typename Geometry::Input& u, double dt);

  /**
   * Update the filter state with a measurement
   * @param y Direction measurement
   */
  void update(const typename Geometry::Measurement& y);

  static constexpr int DOF = gtsam::traits<G>::dimension;
  static constexpr int n_cal = Geometry::n_cal;
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
 * Uses SelfAdjointSolver, completeOrthoganalDecomposition().pseudoInverse()
 */
template <typename G, typename M, typename Geometry>
EqF<G, M, Geometry>::EqF(const G& X0, const M& x0, const Matrix& Sigma, int m)
    : X_hat(X0), Sigma(Sigma), xi_0(x0) {
  if (Sigma.rows() != DOF || Sigma.cols() != DOF) {
    throw std::invalid_argument(
        "Initial covariance dimensions must match the degrees of freedom");
  }

  if (n_cal < 0) {
    throw std::invalid_argument(
        "Number of calibration states must be non-negative");
  }

  if (m <= 1) {
    throw std::invalid_argument(
        "Number of direction sensors must be at least 2");
  }

  static_assert(has_lift<Geometry, M, abc_eqf_lib::Input>::value,
                "Geometry must implement static lift(const M&, const Input&)");
  static_assert(has_stateTransitionMatrix<Geometry>::value,
                "Geometry must define static stateTransitionMatrix(Input, "
                "double, GType)");
  static_assert(has_groupAction<Geometry>::value,
                "Geometry must define groupAction(GType, MType)");
  static_assert(
      has_stateMatrixA<Geometry>::value,
      "Geometry must define static stateMatrixA(const GType&, const Input&)");
  static_assert(has_inputMatrix<Geometry>::value,
                "Geometry must define static inputMatrix(GType)");
  static_assert(has_processNoise<Geometry>::value,
                "Geometry must define static processNoise(const Input&)");
  static_assert(has_inputMatrixBt<Geometry>::value,
                "Geometry must define static inputMatrixBt(GType)");
  static_assert(
      has_measurementMatrixC<Geometry>::value,
      "Geometry must define static measurementMatrixC(const Unit3&, int)");
  static_assert(has_outputMatrixDt<Geometry>::value,
                "Geometry must define static outputMatrixDt(int, GType)");

  // Compute differential of phi
  Dphi0 = stateActionDiff(xi_0);
  InnovationLift = Dphi0.completeOrthogonalDecomposition().pseudoInverse();
}
/**
 * Computes the internal group state to a physical state estimate
 * @return Current state estimate
 */
template <typename G, typename M, typename Geometry>
M EqF<G, M, Geometry>::stateEstimate() const {
  return Geometry::groupAction(X_hat, xi_0);
}
/**
 * Implements the prediction step of the EqF using system dynamics and
 * covariance propagation and advances the filter state by symmtery-preserving
 * dynamics.Uses a Lie group integrator scheme for discrete time propagation
 * @param u Angular velocity measurements
 * @param dt time steps
 * Updated internal state and covariance
 */
template <typename G, typename M, typename Geometry>
void EqF<G, M, Geometry>::propagation(const typename Geometry::Input& u,
                                      double dt) {
  M state_est = stateEstimate();            // General manifold
  Vector L = Geometry::lift(state_est, u);  // Lifted vector in tangent space

  Matrix Phi = Geometry::stateTransitionMatrix(u, dt, X_hat);
  Matrix Bt = Geometry::inputMatrixBt(X_hat);

  Matrix Q = Geometry::processNoise(u);  // Replaces blockDiag(...)

  X_hat = traits<G>::Compose(X_hat, traits<G>::Expmap(L * dt));
  Sigma = Phi * Sigma * Phi.transpose() + Bt * Q * Bt.transpose() * dt;
}
/**
 * Implements the correction step of the filter using discrete measurements
 * Computes the measurement residual, Kalman gain and the updates both the state
 * and covariance
 *
 * @param y Measurements
 */
template <typename G, typename M, typename Geometry>
void EqF<G, M, Geometry>::update(const typename Geometry::Measurement& y) {
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

  Vector3 action_result = outputAction(X_hat.inv(), y.y, y.cal_idx);
  Vector3 delta_vec = Rot3::Hat(y.d.unitVector()) * action_result;
  Matrix Dt = Geometry::outputMatrixDt(y.cal_idx, X_hat);

  Matrix S = Ct * Sigma * Ct.transpose() + Dt * y.Sigma * Dt.transpose();
  Matrix K = Sigma * Ct.transpose() * S.inverse();
  Vector Delta = InnovationLift * K * delta_vec;
  X_hat = traits<G>::Compose(traits<G>::Expmap(Delta), X_hat);
  Sigma = (Matrix::Identity(DOF, DOF) - K * Ct) * Sigma;
}
}  // namespace abc_eqf_lib
}  // namespace gtsam

#endif  // EQUIVARIANTFILTER_H