/**
 * @file ABC.h
 * @brief Pure geometric structures for Attitude-Bias-Calibration systems
 *
 * This file contains only the core geometric structures (State manifold and
 * Group symmetry) for the ABC system. Filter infrastructure is in ABCFilter.h.
 * 
 * Based on the paper "Overcoming Bias: Equivariant Filter Design for Biased
 * Attitude Estimation with Online Calibration" by Fornasier et al.
 * Authors: Darshan Rajasekaran & Jennifer Oum
 */
#ifndef ABC_H
#define ABC_H

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

namespace gtsam {
namespace abc_eqf_lib {

//========================================================================
// State Manifold
//========================================================================

/**
 * State manifold for the Biased Attitude System
 * Represents the physical state: (R, b, S_1, ..., S_N)
 * - R: Attitude (SO(3))
 * - b: Gyroscope bias (R^3)
 * - S_i: Sensor calibration rotations (SO(3))
 */
template <size_t N>
class State {
 public:
  Rot3 R;                 /// Attitude rotation
  Vector3 b;              /// Gyroscope bias
  std::array<Rot3, N> S;  /// Sensor calibrations

  /// Constructor
  State(const Rot3& R = Rot3::Identity(), const Vector3& b = Vector3::Zero(),
        const std::array<Rot3, N>& S = std::array<Rot3, N>{})
      : R(R), b(b), S(S) {}

  /// Identity element
  static State identity() {
    std::array<Rot3, N> S_id{};
    S_id.fill(Rot3::Identity());
    return State(Rot3::Identity(), Vector3::Zero(), S_id);
  }
  
  /**
   * Compute local coordinates (tangent space) relative to another state
   * @param other The other state
   * @return Local coordinates in the tangent space (dimension: 6 + 3N)
   */
  Vector localCoordinates(const State<N>& other) const {
    Vector eps(6 + 3 * N);

    // First 3 elements - attitude error
    eps.head<3>() = Rot3::Logmap(R.between(other.R));
    
    // Next 3 elements - bias error
    eps.segment<3>(3) = other.b - b;

    // Remaining elements - calibration errors
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
 * Symmetry group (SO(3) ⋉ so(3)) × SO(3)^N
 * Structure: (A, a, B_1, ..., B_N)
 * - A ∈ SO(3): Rotation component
 * - a ∈ so(3): Skew-symmetric matrix (Lie algebra element)
 * - B_i ∈ SO(3): Calibration rotations
 */
template <size_t n>
struct Group {
  Rot3 A;                 /// SO(3) rotation
  Matrix3 a;              /// so(3) skew-symmetric matrix
  std::array<Rot3, n> B;  /// N calibration rotations
  
  static constexpr int dimension = 6 + 3 * n;
  static constexpr int numSensors = n;
  using TangentVector = Eigen::Matrix<double, dimension, 1>;

  /// Constructor
  Group(const Rot3& A = Rot3::Identity(), const Matrix3& a = Matrix3::Zero(),
        const std::array<Rot3, n>& B = std::array<Rot3, n>{})
      : A(A), a(a), B(B) {}

  /// Group multiplication
  Group operator*(const Group<n>& other) const {
    std::array<Rot3, n> newB;
    for (size_t i = 0; i < n; i++) {
      newB[i] = B[i] * other.B[i];
    }
    return Group(A * other.A, a + Rot3::Hat(A.matrix() * Rot3::Vee(other.a)), newB);
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
  static Group identity() {
    std::array<Rot3, n> B;
    B.fill(Rot3::Identity());
    return Group(Rot3::Identity(), Matrix3::Zero(), B);
  }

  /// Exponential map: tangent space -> group
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

  /// Retract from tangent space
  Group retract(const TangentVector& v,
                OptionalJacobian<dimension, dimension> H = {},
                OptionalJacobian<dimension, dimension> Hv = {}) const {
    return gtsam::traits<Group>::Compose(*this, gtsam::traits<Group>::Expmap(v));
  }

  /// Adjoint map (placeholder - needs proper implementation)
  Eigen::Matrix<double, dimension, dimension> AdjointMap() const {
    // TODO: implement properly for the semi-direct product structure
    return Eigen::Matrix<double, dimension, dimension>::Identity();
  }

  /// Logarithm map: group -> tangent space
  static Eigen::Matrix<double, dimension, 1>
  Logmap(const Group& g, OptionalJacobian<dimension, dimension> H = {}) {
    State<n> xi0 = State<n>::identity();
    State<n> xi_transformed = g * xi0;
    Vector logv = xi0.localCoordinates(xi_transformed);

    if (H) {
      auto mapGtoVec = [&xi0](const Group& gg) {
        State<n> x_trans = gg * xi0;
        return xi0.localCoordinates(x_trans);
      };
      *H = gtsam::numericalDerivative11<Vector, Group>(
          std::function<Vector(const Group&)>(mapGtoVec), g);
    }

    return logv;
  }
};

//========================================================================
// Group Actions
//========================================================================

/**
 * Group action on state: G × M → M
 * Transforms state xi by group element X
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
 * Velocity action: transforms input (ω, 0) between frames
 * @param X Group element
 * @param u Input vector (ω, 0) where first 3 are angular velocity
 * @return Transformed input
 */
template <size_t N>
Vector6 velocityAction(const Group<N>& X, const Vector6& u) {
  Vector6 result;
  result.head<3>() = X.A.inverse().matrix() * (u.head<3>() - Rot3::Vee(X.a));
  result.tail<3>() = Vector3::Zero();
  return result;
}

/**
 * Output action: transforms direction measurement
 * @param X Group element
 * @param y Direction measurement (Unit3)
 * @param idx Calibration index (-1 for uncalibrated)
 * @return Transformed direction
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

}  // namespace abc_eqf_lib

//========================================================================
// GTSAM Traits Specialization
//========================================================================

template <size_t N>
struct traits<abc_eqf_lib::Group<N>> : internal::LieGroupTraits<abc_eqf_lib::Group<N>> {
  using GType = abc_eqf_lib::Group<N>;
  static constexpr int dimension = GType::dimension;

  using OptionalJac = OptionalJacobian<dimension, dimension>;
  using MatrixDim = Eigen::Matrix<double, dimension, dimension>;

  static GType Identity() { return GType::identity(); }

  static GType Compose(const GType& g1, const GType& g2,
                       OptionalJac Hg = OptionalJac(), OptionalJac Hh = OptionalJac()) {
    if (Hg) *Hg = MatrixDim::Zero();
    if (Hh) *Hh = MatrixDim::Zero();
    return g1 * g2;
  }

  static GType Between(const GType& g1, const GType& g2,
                       OptionalJac H1 = OptionalJac(), OptionalJac H2 = OptionalJac()) {
    if (H1) *H1 = MatrixDim::Zero();
    if (H2) *H2 = MatrixDim::Zero();
    return g1.inv() * g2;
  }

  static GType Inverse(const GType& g, OptionalJac H = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    return g.inv();
  }

  static GType Expmap(const Vector& v, OptionalJac H = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    return GType::exp(v);
  }

  static Vector Local(const GType& g1, const GType& g2,
                      OptionalJac H1 = OptionalJac(), OptionalJac H2 = OptionalJac()) {
    if (H1) *H1 = MatrixDim::Zero();
    if (H2) *H2 = MatrixDim::Zero();
    GType between = g1.inv() * g2;
    return GType::Logmap(between);
  }

  static GType Retract(const GType& g, const Vector& v,
                       OptionalJac H = OptionalJac(), OptionalJac Hv = OptionalJac()) {
    if (H) *H = MatrixDim::Zero();
    if (Hv) *Hv = MatrixDim::Zero();
    return Compose(g, Expmap(v));
  }

  static void Print(const GType& g, const std::string& s = "") {
    std::cout << s << std::endl;
    std::cout << "A = " << g.A << std::endl;
    std::cout << "a = " << g.a << std::endl;
    for(size_t i = 0; i < GType::numSensors; ++i) {
      std::cout << "B[" << i << "] = " << g.B[i] << std::endl;
    }
  }
  
  static bool Equals(const GType& g1, const GType& g2, double tol = 1e-9) {
    if (!g1.A.equals(g2.A, tol)) return false;
    if (!gtsam::assert_equal<Matrix3>(g1.a, g2.a, tol)) return false;
    for(size_t i = 0; i < GType::numSensors; ++i) {
      if (!g1.B[i].equals(g2.B[i], tol)) return false;
    }
    return true;
  }
};

}  // namespace gtsam

#endif  // ABC_H
