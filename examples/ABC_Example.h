/**
 * @file ABC.h
 * @date March, 2025
 * @author Jennifer & Darshan
 * @brief Group and manifold for ABC Equivariant Filter
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>

namespace gtsam {
namespace abc {

/// The Manifold: (SO(3) x R^3) x SO(3)^n
template <size_t N>
struct Manifold {
  /// @name Definition
  /// @{
  Rot3 R;                 // SO(3) rotation
  Vector3 b;              // R^3 element
  std::array<Rot3, N> S;  // SO(3) rotations as a fixed-size array

  /// Default constructor yields identity
  Manifold() { throw std::runtime_error("not implemented"); }

  /// Constructor from individual elements
  Manifold(const Rot3& R, const Vector3& b, const std::array<Rot3, N>& S)
      : R(R), b(b), S(S) {}
  /// @}
  /// @name Testable
  /// @{
  /// Print function for debugging
  void print(const std::string& s = "") const {
    std::cout << s << ":" << std::endl;
    std::cout << "R       b   ";
    for (size_t i = 0; i < N; ++i) {
      std::cout << "C[" << i << "]    ";
    }
    std::cout << std::endl;
    for (size_t row = 0; row < 3; ++row) {
      std::cout << R.matrix().row(row) << "   " << b(row) << "   ";
      for (size_t i = 0; i < N; ++i) {
        std::cout << S[i].matrix().row(row) << "   ";
      }
      std::cout << std::endl;
    }
  }

  /// Equality check with tolerance
  bool equals(const Manifold& other, double tol = 1e-9) const {
    if (!R.equals(other.R, tol)) return false;
    if (!b.isApprox(other.b, tol)) return false;
    for (size_t i = 0; i < N; ++i) {
      if (!S[i].equals(other.S[i], tol)) return false;
    }
    return true;
  }
  /// @}
  /// @name Manifold
  /// @{
  inline constexpr static auto dimension = 3 + 3 + 3 * N;
  inline static size_t Dim() { return dimension; }
  inline size_t dim() const { return dimension; }

  typedef Eigen::Matrix<double, dimension, 1> TangentVector;
  typedef OptionalJacobian<dimension, dimension> ChartJacobian;

  Manifold retract(const TangentVector& v,  //
                   ChartJacobian H1 = {}, ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector localCoordinates(const Manifold& m,  //
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
};

/// The Group: (SO(3) semi so<3>) x SO(3)^n
template <size_t N>
struct Group {
  /// @name Definition
  Rot3 A;                 // SO(3) rotation
  Matrix a;               // so<3> element, let's use a matrix to make a point
  std::array<Rot3, N> B;  // SO(3) rotations as a fixed-size array

  /// Default constructor yields identity
  Group() { throw std::runtime_error("not implemented"); }

  /// Constructor from individual elements
  Group(const Rot3& A, const Matrix& a, const std::array<Rot3, N>& B)
      : A(A), a(a), B(B) {}
  /// @name Testable
  /// @{
  /// Print function for debugging
  void print(const std::string& s = "") const {
    std::cout << s << ":" << std::endl;
    std::cout << "A        a         ";
    for (size_t i = 0; i < N; ++i) {
      std::cout << "B[" << i << "]    ";
    }
    std::cout << std::endl;
    for (size_t row = 0; row < 3; ++row) {
      std::cout << A.matrix().row(row) << "   " << a.row(row) << "   ";
      for (size_t i = 0; i < N; ++i) {
        std::cout << B[i].matrix().row(row) << "   ";
      }
      std::cout << std::endl;
    }
  }

  /// Equality check with tolerance
  bool equals(const Group& other, double tol = 1e-9) const {
    if (!A.equals(other.A, tol)) return false;
    if (!a.isApprox(other.a, tol)) return false;
    for (size_t i = 0; i < N; ++i) {
      if (!B[i].equals(other.B[i], tol)) return false;
    }
    return true;
  }
  /// @}
  /// @}
  /// @name Group
  /// @{
  typedef multiplicative_group_tag group_flavor;
  static Group Identity() { throw std::runtime_error("not implemented"); }

  Group operator*(const Group& other) const {
    throw std::runtime_error("not implemented");
  }
  Group inverse() const { throw std::runtime_error("not implemented"); }
  Group compose(const Group& g) const {
    throw std::runtime_error("not implemented");
  }
  Group between(const Group& g) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
  /// @name Manifold
  /// @{
  inline constexpr static auto dimension = 3 + 3 + 3 * N;
  inline static size_t Dim() { return dimension; }
  inline size_t dim() const { return dimension; }

  typedef Eigen::Matrix<double, dimension, 1> TangentVector;
  typedef OptionalJacobian<dimension, dimension> ChartJacobian;

  Group retract(const TangentVector& v,  //
                ChartJacobian H1 = {}, ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector localCoordinates(const Group& g,  //
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
  /// @name Lie Group
  /// @{
  Group compose(const Group& other, ChartJacobian H1,
                ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  Group between(const Group& other, ChartJacobian H1,
                ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  Group inverse(ChartJacobian D) const {
    throw std::runtime_error("not implemented");
  }
  static Group Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    throw std::runtime_error("not implemented");
  }
  static TangentVector Logmap(const Group& p, ChartJacobian Hp = {}) {
    throw std::runtime_error("not implemented");
  }
  static TangentVector LocalCoordinates(const Group& p, ChartJacobian Hp = {}) {
    throw std::runtime_error("not implemented");
  }
  Group expmap(const TangentVector& v) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector logmap(const Group& g) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
};

/// The Manifold: (SO(3) x R^3) x SO(3)^n
template <typename G, typename V, size_t N>
struct Manifold2 {
  /// @name Definition
  /// @{
  G R;  // Rot3 R; // SO(3) rotation
  V b;  // Vector3 b; // R^3 element
  std::array<G, N>
      S;  // std::array<Rot3, N> S; // SO(3) rotations as a fixed-size array

  /// Default constructor yields identity
  Manifold2() { throw std::runtime_error("not implemented"); }

  /// Constructor from individual elements
  Manifold2(const G& R, const V& b, const std::array<G, N>& S)
      : R(R), b(b), S(S) {}
  /// @}
  /// @name Testable
  /// @{
  /// Print function for debugging
  void print(const std::string& s = "") const {
    std::cout << s << ":" << std::endl;
    std::cout << "R       b   ";
    for (size_t i = 0; i < N; ++i) {
      std::cout << "C[" << i << "]    ";
    }
    std::cout << std::endl;
    for (size_t row = 0; row < 3; ++row) {
      std::cout << R.matrix().row(row) << "   " << b(row) << "   ";
      for (size_t i = 0; i < N; ++i) {
        std::cout << S[i].matrix().row(row) << "   ";
      }
      std::cout << std::endl;
    }
  }

  /// Equality check with tolerance
  bool equals(const Manifold2& other, double tol = 1e-9) const {
    if (!R.equals(other.R, tol)) return false;
    if (!b.isApprox(other.b, tol)) return false;
    for (size_t i = 0; i < N; ++i) {
      if (!S[i].equals(other.S[i], tol)) return false;
    }
    return true;
  }
  /// @}
  /// @name Manifold
  /// @{
  inline constexpr static auto dimension = 3 + 3 + 3 * N;
  inline static size_t Dim() { return dimension; }
  inline size_t dim() const { return dimension; }

  typedef Eigen::Matrix<double, dimension, 1> TangentVector;
  typedef OptionalJacobian<dimension, dimension> ChartJacobian;

  Manifold2 retract(const TangentVector& v,  //
                    ChartJacobian H1 = {}, ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector localCoordinates(const Manifold2& m,  //
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
};

/// The Group: (SO(3) semi so<3>) x SO(3)^n
template <typename G, typename MAT, size_t N>
struct Group2 {
  /// @name Definition
  G A;                 // SO(3) rotation
  MAT a;               // so<3> element, let's use a matrix to make a point
  std::array<G, N> B;  // SO(3) rotations as a fixed-size array

  /// Default constructor yields identity
  Group2() { throw std::runtime_error("not implemented"); }

  /// Constructor from individual elements
  Group2(const Rot3& A, const Matrix& a, const std::array<Rot3, N>& B)
      : A(A), a(a), B(B) {}
  /// @name Testable
  /// @{
  /// Print function for debugging
  void print(const std::string& s = "") const {
    std::cout << s << ":" << std::endl;
    std::cout << "A        a         ";
    for (size_t i = 0; i < N; ++i) {
      std::cout << "B[" << i << "]    ";
    }
    std::cout << std::endl;
    for (size_t row = 0; row < 3; ++row) {
      std::cout << A.matrix().row(row) << "   " << a.row(row) << "   ";
      for (size_t i = 0; i < N; ++i) {
        std::cout << B[i].matrix().row(row) << "   ";
      }
      std::cout << std::endl;
    }
  }

  /// Equality check with tolerance
  bool equals(const Group2& other, double tol = 1e-9) const {
    if (!A.equals(other.A, tol)) return false;
    if (!a.isApprox(other.a, tol)) return false;
    for (size_t i = 0; i < N; ++i) {
      if (!B[i].equals(other.B[i], tol)) return false;
    }
    return true;
  }
  /// @}
  /// @}
  /// @name Group
  /// @{
  typedef multiplicative_group_tag group_flavor;
  static Group2 Identity() { throw std::runtime_error("not implemented"); }

  Group2 operator*(const Group2& other) const {
    throw std::runtime_error("not implemented");
  }
  Group2 inverse() const { throw std::runtime_error("not implemented"); }
  Group2 compose(const Group2& g) const {
    throw std::runtime_error("not implemented");
  }
  Group2 between(const Group2& g) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
  /// @name Manifold
  /// @{
  inline constexpr static auto dimension = 3 + 3 + 3 * N;
  inline static size_t Dim() { return dimension; }
  inline size_t dim() const { return dimension; }

  typedef Eigen::Matrix<double, dimension, 1> TangentVector;
  typedef OptionalJacobian<dimension, dimension> ChartJacobian;

  Group2 retract(const TangentVector& v,  //
                 ChartJacobian H1 = {}, ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector localCoordinates(const Group2& g,  //
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
  /// @name Lie Group
  /// @{
  Group2 compose(const Group2& other, ChartJacobian H1,
                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  Group2 between(const Group2& other, ChartJacobian H1,
                 ChartJacobian H2 = {}) const {
    throw std::runtime_error("not implemented");
  }
  Group2 inverse(ChartJacobian D) const {
    throw std::runtime_error("not implemented");
  }
  static Group2 Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    throw std::runtime_error("not implemented");
  }
  static TangentVector Logmap(const Group2& p, ChartJacobian Hp = {}) {
    throw std::runtime_error("not implemented");
  }
  static TangentVector LocalCoordinates(const Group2& p,
                                        ChartJacobian Hp = {}) {
    throw std::runtime_error("not implemented");
  }
  Group2 expmap(const TangentVector& v) const {
    throw std::runtime_error("not implemented");
  }
  TangentVector logmap(const Group2& g) const {
    throw std::runtime_error("not implemented");
  }
  /// @}
};

template <typename ManifoldT, typename GroupT>
struct EqFState {
  ManifoldT xi;  // Filter state lives on a manifold
  GroupT G;      // Symmetry group used in the filter

  // define group action as a functor or static method
  static ManifoldT act(const GroupT& g, const ManifoldT& x) {
    return groupAction(g, x);
  }
};

}  // namespace abc

// traits
template <size_t N>
struct traits<abc::Manifold<N>> : internal::LieGroupTraits<abc::Manifold<N>> {};

template <size_t N>
struct traits<abc::Group<N>> : internal::LieGroupTraits<abc::Group<N>> {};

template <typename G, typename V, size_t N>
struct traits<abc::Manifold2<G, V, N>>
    : internal::LieGroupTraits<abc::Manifold2<G, V, N>> {};

template <typename G, typename MAT, size_t N>
struct traits<abc::Group2<G, MAT, N>>
    : internal::LieGroupTraits<abc::Group2<G, MAT, N>> {};

}  // namespace gtsam
