/**
 * @file ABCEqFWrapper.h
 * @brief Wrapper-friendly ABC EqF helper class for Python bindings.
 *
 * This class mirrors the logic in examples/AbcEquivariantFilterExample.cpp
 * but exposes a simple, non-templated API for wrapping.
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/geometry/ABC.h>

namespace gtsam {
namespace abc {

class AbcEquivariantFilter1 {
 public:
  static constexpr size_t N = 1;
  using State1 = State<N>;
  using Group1 = Group<N>;
  using Symmetry1 = Symmetry<N>;
  using EqFilter1 = gtsam::EquivariantFilter<State1, Symmetry1>;

  AbcEquivariantFilter1() : AbcEquivariantFilter1(Matrix::Identity(6 + 3 * N, 6 + 3 * N)) {}

  explicit AbcEquivariantFilter1(const Matrix& Sigma0)
      : filter_(State1::identity(), Sigma0) {}

  /// Propagate filter using the explicit matrices API from ABC.
  void predict(const Vector3& omega, const Matrix6& inputCovariance, double dt) {
    const Matrix Q = inputProcessNoise<N>(inputCovariance);
    const Vector6 u = toInputVector(omega);
    const Lift<N> lift_u(u);
    const typename InputAction<N>::Orbit psi_u(u);

    const Group1 X_hat = filter_.groupEstimate();
    const Matrix A = stateMatrixA<N>(psi_u, X_hat);
    const Matrix B = inputMatrixB<N>(X_hat);
    const Matrix Qc = B * Q * B.transpose();

    filter_.predictWithJacobian<2>(lift_u, A, Qc, dt);
  }

  /// Update filter with a direction measurement.
  void update(const Unit3& y, const Unit3& d, const Matrix3& R, int cal_idx) {
    const Innovation<N> innovation(y, d, cal_idx);
    const Group1 X_hat = filter_.groupEstimate();
    const Matrix3 D = outputMatrixD<N>(X_hat, cal_idx);
    const Matrix3 R_adjusted = D * R * D.transpose();
    filter_.update<Vector3>(innovation, Z_3x1, R_adjusted);
  }

  /// Accessors for current estimate.
  Rot3 attitude() const { return filter_.state().R; }
  Vector3 bias() const { return filter_.state().b; }
  Rot3 calibration(size_t i) const { return filter_.state().S[i]; }
  Matrix errorCovariance() const { return filter_.errorCovariance(); }

 private:
  EqFilter1 filter_;
};

}  // namespace abc
}  // namespace gtsam


