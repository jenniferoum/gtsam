/**
 * @file  SL4.h
 * @brief Projective Special Linear Group (SL(4, R)) factor
 * @author: Hyungtae Lim
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_unstable/dllexport.h>

// To use exp(), log()
#include <string>
#include <unsupported/Eigen/MatrixFunctions>

using SL4Jacobian = gtsam::OptionalJacobian<15, 15>;

using Matrix15x15 = Eigen::Matrix<double, 15, 15>;
using Matrix16x16 = Eigen::Matrix<double, 16, 16>;

namespace gtsam {
// NOTE(hlim): Strictly speaking, it should be expressed as SL(4, ℝ),
// but for simplicity, we omit ℝ, assuming our target is over the real numbers.
// And the variable `sl4` represents SL(4, ℝ).
class GTSAM_EXPORT SL4 : public MatrixLieGroup<SL4, 15, 4> {
 public:
  static const size_t dimension = 15;

 protected:
  Matrix44 T_;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor initializes at origin
  SL4() : T_(Matrix44::Identity()) {}

  /// Copy constructor
  SL4(const Matrix44& pose);

  SL4(const SL4& pose) = default;

  SL4& operator=(const SL4& pose) = default;

  /** print with optional string */
  void print(const std::string& s = "") const;

  /** assert equality up to a tolerance */
  bool equals(const SL4& sl4, double tol = 1e-9) const;

  /** convert to 4*4 matrix */
  inline const Matrix44& matrix() const { return T_; }

  /// @}
  /// @name Manifold
  /// @{

  /// Dimensionality of tangent space = 15 DOF - used to autodetect sizes
  inline static size_t Dim() { return dimension; }

  /// Dimensionality of tangent space = 15 DOF
  inline size_t dim() const { return dimension; }

  SL4 retract(const Vector& v, SL4Jacobian H1 = {}, SL4Jacobian H2 = {}) const;

  Vector localCoordinates(const SL4& sl4, SL4Jacobian H1 = {},
                          SL4Jacobian H2 = {}) const;

  /// @}
  /// @name Group
  /// @{

  /// identity for group operation
  static SL4 Identity() { return SL4(); }

  /// inverse transformation
  SL4 inverse() const { return SL4(T_.inverse()); }

  /// Group operation
  SL4 operator*(const SL4& other) const { return SL4(T_ * other.T_); }

  /// @}
  /// @name Lie Group
  /// @{

  /// Version with derivative version added by LieGroup
  using LieGroup<SL4, 15>::inverse;

  /// Exponential map at identity - create a rotation from canonical coordinates
  static SL4 Expmap(const Vector& xi);

  /// Log map at identity - return the canonical coordinates of this rotation
  static Vector Logmap(const Matrix44& T);

  static Vector Logmap(const SL4& p);

  /// Adjoint representation of the tangent space
  Matrix15x15 AdjointMap() const;

  /// @}
  /// @name Matrix Lie Group
  /// @{

  using LieAlgebra = Matrix44;
  static Matrix44 Hat(const Vector& xi);
  static Vector Vee(const Matrix44& X);

  /// @}
  /// @name Serialization
  /// @{

  /// Serialization function
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) const {
    ar& BOOST_SERIALIZATION_NVP(T_);
  }

  /// Deserialization function
  template <class ARCHIVE>
  void deserialize(ARCHIVE& ar) {
    ar& BOOST_SERIALIZATION_NVP(T_);
  }

  /// @}

 private:
#ifdef GTSAM_ENABLE_BOOST_SERIALIZATION
  // Serialization function
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(T_);
  }
#endif
};  // \class SL4

template <>
struct traits<SL4> : public internal::MatrixLieGroup<SL4, 4> {};

template <>
struct traits<const SL4> : public internal::MatrixLieGroup<SL4, 4> {};

}  // namespace gtsam
