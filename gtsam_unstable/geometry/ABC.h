/**
 * @file ABC.h
 * @brief Core components for Attitude-Bias-Calibration systems
 *
 * This file contains fundamental components and utilities for the ABC system
 * based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 *
 * We follow the paper's notation: the physical state is ξ = ((R,b), C) ∈ M, the
 * symmetry group element is X = ((A,a), B) ∈ G, and the right actions φ_ξ(X)
 * and ρ_y(X) drive the equivariant filter update. See Eqs. (4), (7), (14b), and
 * (23)–(24) in Fornasier et al. (2022) for the continuous-time dynamics, lift
 * Λ(ξ,u), output action, and EqF update.
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

#include <gtsam/base/GroupAction.h>
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
  using TangentVector = Eigen::Matrix<double, dimension, 1>;

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
  TangentVector localCoordinates(const State<N>& other) const {
    TangentVector eps(dimension);

    // First 3 elements - attitude
    eps.template head<3>() = R.logmap(other.R);
    // Next 3 elements - bias
    eps.template segment<3>(3) = other.b - b;

    // Remaining elements - calibrations
    eps.template segment<3 * N>(6) = S.logmap(other.S);

    return eps;
  }

  /**
   * Retract from tangent space back to the manifold
   * @param v Vector in the tangent space
   * @return New state
   */
  State retract(const TangentVector& v) const {
    Rot3 newR = R.expmap(v.template head<3>());
    Vector3 newB = b + v.template segment<3>(3);
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

  /**
   * Matrix representation of the Lie-algebra adjoint operator ad_xi on g.
   * For this direct product group it is block-diagonal with Pose3 and Rot3
   * blocks.
   */
  static Jacobian adjointMap(const TangentVector& xi) {
    Jacobian result = Jacobian::Zero();
    result.template block<6, 6>(0, 0) =
        Pose3::adjointMap(xi.template head<6>());
    for (size_t i = 0; i < n; ++i) {
      result.template block<3, 3>(6 + 3 * i, 6 + 3 * i) =
          Rot3::adjointMap(xi.template segment<3>(6 + 3 * i));
    }
    return result;
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
// Group Actions on State, Input, and Output Manifolds
//========================================================================

/**
 * The symmetry group G is defined as the product Pose3 x Calibrations<n>.
 * The Pose3 component models the (SO(3) ⋉ R^3) part acting on attitude/bias,
 * while Calibrations<n> captures the N sensor calibration rotations.
 *
 * The group action is defined as a function object below,
 * applied to a given state x, specified in constructor.
 * Implements the right action φ_{ξ}(X) = (R A, Aᵀ(b − a), Aᵀ C B), where
 * ξ=(R,b,C). This is the discrete version of the homogeneous-space action in
 * Eq. (4) of Fornasier et al. (2022).
 */
template <size_t N>
struct Symmetry : public GroupAction<Symmetry<N>, Group<N>, State<N>> {
  using M = State<N>;
  using G = gtsam::abc::Group<N>;
  static constexpr ActionType type = ActionType::Right;

  /**
   * Implements group actions on the states
   * @param g An element of the symmetry group G.
   * @return Transformed state
   */
  M operator()(const M& xi, const G& g,
               OptionalJacobian<M::dimension, M::dimension> Hm = {},
               OptionalJacobian<M::dimension, G::dimension> Hg = {}) const {
    const Rot3 new_R = xi.R * g.A();
    Matrix3 skew_p, At;  // derivatives of unrotate
    const Point3 p = xi.b - g.a();
    const Vector3 new_b =
        g.A().unrotate(p, Hg ? &skew_p : nullptr, (Hg || Hm) ? &At : nullptr);
    Calibrations<N> new_S;
    const Calibrations<N>& B = g.calibrations();
    Rot3 invA = g.A().inverse();  // derivative is (- A)
    for (size_t i = 0; i < N; i++) {
      Rot3 SB = xi.S[i].compose(B[i]);  // derivative in B[i] is identity
      new_S[i] = invA.compose(SB);      // derivative in invA is SB^{-1}.
    }

    if (Hm) {
      Hm->setZero();

      // d(R*A)/dR: right-multiplication by A maps a tangent vector δθ to
      // Ad_{A^{-1}} δθ = Aᵀ δθ.
      Hm->template block<3, 3>(0, 0) = At;

      // d(At*(b - a))/db = At.
      Hm->template block<3, 3>(3, 3) = At;

      // d(At * S[i] * B[i]) / dS[i] = Ad_{B[i]^{-1}} on so(3), which is just
      // multiplication by B[i]ᵀ in vector form.
      for (size_t i = 0; i < N; ++i) {
        const size_t row = 6 + 3 * i;
        const Matrix3 BiT = g.calibrations()[i].matrix().transpose();
        Hm->template block<3, 3>(row, row) = BiT;
      }
    }
    if (Hg) {
      Hg->setZero();
      // Rotation block: δθ maps directly to the state's rotational tangent.
      Hg->template block<3, 3>(0, 0) = I_3x3;

      // Bias block:
      Hg->template block<3, 3>(3, 0) = skew_p;
      Hg->template block<3, 3>(3, 3) = -I_3x3;  // - At * A (from translation()

      // Calibration blocks:
      Matrix3 A = g.A().matrix();
      for (size_t i = 0; i < N; ++i) {
        Rot3 SB = xi.S[i].compose(B[i]);
        const size_t row = 6 + 3 * i;
        const size_t col = 6 + 3 * i;
        Hg->template block<3, 3>(row, 0) = -SB.transpose() * A;
        Hg->template block<3, 3>(row, col) = I_3x3;
      }
    }
    return {new_R, new_b, new_S};
  }

  struct Orbit : public group_action::Orbit<Symmetry<N>> {
    using Base = group_action::Orbit<Symmetry<N>>;
    using Base::Base;  // Inherit constructors
  };
};

/**
 * Continuous-time dynamics on the manifold M = SO(3) x R^3 x SO(3)^N, as in
 * Eq. (2) of the paper. Given state ξ = (R, b, S) and body angular velocity
 * ω, the state derivative is:
 *   Ṙ = R (ω - b)^
 *   ḃ = 0
 *   Ṡ_i = 0
 * We represent the tangent as a vector with components (δθ, δb, δσ_i).
 */
template <size_t N>
inline typename State<N>::TangentVector dynamics(const Vector3& omega,
                                                 const State<N>& xi) {
  typename State<N>::TangentVector xi_dot;
  xi_dot.setZero();
  xi_dot.template head<3>() = omega - xi.b;
  // Remaining components are already zero.
  return xi_dot;
}

/**
 * /// Implements the lift Λ(ξ,u) from the paper: Λ encodes the lifted dynamics
 * on G induced by the biased gyroscope input u = (ω,0). Functor computing the
 * lifted tangent vector from a state and fixed input. In the notation of
 * Fornasier et al., this corresponds to Eq. (7), written in the so(3)≃ℝ³
 * identification.
 */
template <size_t N>
struct Lift {
  using M = State<N>;
  using G = Group<N>;

  explicit Lift(const Vector6& u) : u_(u) {}

  typename G::TangentVector operator()(
      const M& xi, OptionalJacobian<G::dimension, M::dimension> H = {}) const {
    typename G::TangentVector L;
    Vector3 w = u_.head<3>();  // w = omega
    Vector3 corrected_w = w - xi.b;
    L.template head<3>() = corrected_w;
    L.template segment<3>(3) = -Rot3::Hat(w) * xi.b;
    if (H) {
      H->setZero();
      // corrected_w / xi:
      H->template block<3, 3>(0, 3) = -I_3x3;

      // next segment:
      H->template block<3, 3>(3, 3) = -Rot3::Hat(w);
    }
    for (size_t i = 0; i < N; i++) {
      DenseIndex k = 6 + 3 * i;
      Vector3 v_i = xi.S[i].unrotate(corrected_w);
      L.template segment<3>(k) = v_i;
      if (H) {
        H->template block<3, 3>(k, 3) = -xi.S[i].transpose();
        H->template block<3, 3>(k, k) = Rot3::Hat(v_i);
      }
    }

    return L;
  }

 private:
  Vector6 u_;
};

/**
 * Encodes the partially applied input action ψ_u(X) = Aᵀ(ω − a), used to
 * compute A(X,u) and Φ(X,u). Functor encoding the right group action on the
 * mathematical input u. For a fixed u = (ω, 0), applying X = (A, a, B) ∈ G
 * yields φ_u(X) = (A^{-1}(ω - a), 0). The matrices A(X,u) and Φ(X,u) here match
 * the linearization in Eqs. (20) and (21), using ω̃ = Aᵀ(ω − a).
 */
template <size_t N>
struct InputAction : public GroupAction<InputAction<N>, Group<N>, Vector6> {
  using G = Group<N>;
  static constexpr ActionType type = ActionType::Right;

  Vector6 operator()(const Vector6& u, const G& X,
                     OptionalJacobian<6, 6> H_u = {},
                     OptionalJacobian<6, G::dimension> H_X = {}) const {
    const Rot3 A = X.A();
    const Vector3 a = X.a();
    Vector6 result;
    result.head<3>() = A.unrotate(u.head<3>() - a);
    result.tail<3>() = Z_3x1;
    return result;
  }
};

// Embed a 6x6 Sigma into full DimU by appending small calibration noise.
template <size_t N>
inline Matrix inputProcessNoise(const Matrix& Sigma6) {
  std::vector<Matrix> blocks{Sigma6};
  blocks.insert(blocks.end(), N, 1e-9 * I_3x3);
  return gtsam::diag(blocks);
}

/// Compute the state matrix A(X_hat).
template <size_t N>
inline Matrix stateMatrixA(const typename InputAction<N>::Orbit& psi_u,
                           const Group<N>& X_hat) {
  const Vector6 u0 = psi_u(X_hat.inverse());  // ψ_u(X)^ω (omega, 0)
  Matrix3 W0 = Rot3::Hat(u0.template head<3>());

  Matrix A1 = Matrix::Zero(6, 6);
  A1.block<3, 3>(0, 3) = -I_3x3;
  A1.block<3, 3>(3, 3) = W0;

  std::vector<Matrix> blocks{A1};
  blocks.insert(blocks.end(), N, W0);
  return gtsam::diag(blocks);
}

/// Compute the input matrix B(X_hat).
template <size_t N>
inline Matrix inputMatrixB(const Group<N>& X_hat) {
  const Matrix3 A_matrix = X_hat.A().matrix();
  Matrix B1 = gtsam::diag({A_matrix, A_matrix});
  Matrix B2(3 * N, 3 * N);
  B2.setZero();
  for (size_t i = 0; i < N; ++i) {
    B2.block<3, 3>(3 * i, 3 * i) = X_hat.calibrations()[i].matrix();
  }
  return gtsam::diag({B1, B2});
}

/**
 * Functor encoding the right action ρ_y(X) on direction measurements y,
 * parameterized by the sensor index.
 */
template <size_t N>
struct OutputAction : public GroupAction<OutputAction<N>, Group<N>, Vector3> {
  using G = Group<N>;
  static constexpr ActionType type = ActionType::Right;

  explicit OutputAction(int index = -1) : index_(index) {}

  Vector3 operator()(const Vector3& y, const G& X,
                     OptionalJacobian<3, 3> H_y = {},
                     OptionalJacobian<3, G::dimension> H_X = {}) const {
    if (H_X) H_X->setZero();
    if (index_ == -1) {
      Matrix3 H_rot;
      Vector3 res = X.A().unrotate(y, H_X ? &H_rot : nullptr, H_y);
      if (H_X) H_X->template block<3, 3>(0, 0) = H_rot;
      return res;
    } else {
      Matrix3 H_rot;
      Vector3 res =
          X.calibrations()[index_].unrotate(y, H_X ? &H_rot : nullptr, H_y);
      if (H_X) H_X->template block<3, 3>(0, 6 + 3 * index_) = H_rot;
      return res;
    }
  }

  int index_;
};

/// Compute the measurement matrix C(φ_y).
template <size_t N>
inline Matrix measurementMatrixC(const Unit3& d, int index) {
  Matrix Cc = Matrix::Zero(3, 3 * N);

  Matrix3 wedge_d = Rot3::Hat(d.unitVector());
  if (index >= 0) {
    Cc.block<3, 3>(0, 3 * index) = wedge_d;
  }

  Matrix temp(3, 6 + 3 * N);
  temp.block<3, 3>(0, 0) = wedge_d;
  temp.block<3, 3>(0, 3) = Matrix3::Zero();
  temp.block(0, 6, 3, 3 * N) = Cc;

  return wedge_d * temp;
}

template <size_t N>
struct Innovation {
  using M = State<N>;

  Innovation(const Unit3& y, const Unit3& d, int index)
      : y_(y), d_(d), xi_ref_(M::identity()), index_(index) {}

  Innovation(const Unit3& y, const Unit3& d, int index, const M& xi_ref)
      : y_(y), d_(d), xi_ref_(xi_ref), index_(index) {}

  Vector3 operator()(const M& xi_hat,
                     OptionalJacobian<3, M::dimension> H = {}) const {
    // Recover A and B_i from (xi_ref_, xi_hat) using the symmetry formulas:
    //   R_hat = R0 * A,   S_hat[i] = Aᵀ * S0[i] * B[i]
    const Rot3 R0 = xi_ref_.R;
    const Rot3 Rhat = xi_hat.R;
    const Rot3 A = R0.inverse() * Rhat;

    Vector3 transformed_y;
    if (index_ == -1) {
      // Uncalibrated sensor: transformed_y = A * y.
      transformed_y = A.rotate(y_.unitVector());
    } else {
      // Calibrated sensor i: B_i = S0[i]^{-1} * A * S_hat[i]
      const Rot3& S0i = xi_ref_.S[index_];
      const Rot3& Shati = xi_hat.S[index_];
      const Rot3 Bi = S0i.inverse() * A * Shati;
      transformed_y = Bi.rotate(y_.unitVector());
    }

    const Matrix3 wedge_d = Rot3::Hat(d_.unitVector());
    const Vector3 nu = wedge_d * transformed_y;

    if (H) {
      *H = measurementMatrixC<N>(d_, index_);
    }
    return nu;
  }

  Unit3 y_;   // measured direction
  Unit3 d_;   // reference direction
  M xi_ref_;  // reference state on the manifold
  int index_;
};

template <size_t N>
inline Matrix3 outputMatrixD(const Group<N>& X_hat, int index) {
  if (index >= 0) {
    return X_hat.calibrations()[index].matrix();
  } else {
    return X_hat.A().matrix();
  }
}

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
