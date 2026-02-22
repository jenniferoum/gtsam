/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    ExtendedPose3.h
 * @brief   Extended pose Lie group SE_k(3), with static or dynamic k.
 * @author  Frank Dellaert, et al.
 */

#pragma once

#include <gtsam/base/MatrixLieGroup.h>
#include <gtsam/geometry/Kernel.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#if GTSAM_ENABLE_BOOST_SERIALIZATION
#include <boost/serialization/nvp.hpp>
#endif

namespace gtsam {

namespace internal {

/// Compile-time traits for SE_k(3) dimensions.
template <int K>
struct ExtendedPose3Traits {
  /// Manifold dimensionality for SE_k(3): 3 + 3*k.
  static constexpr int Dimension() {
    return (K == Eigen::Dynamic) ? Eigen::Dynamic : 3 + 3 * K;
  }

  /// Homogeneous matrix size for SE_k(3): (3 + k) x (3 + k).
  static constexpr int MatrixDim() {
    return (K == Eigen::Dynamic) ? Eigen::Dynamic : 3 + K;
  }
};

}  // namespace internal

/**
 * Lie group SE_k(3): semidirect product of SO(3) with k copies of R^3.
 * Tangent ordering is [omega, x1, x2, ..., xk], each xi in R^3.
 *
 * Template parameter K can be fixed (K >= 1) or Eigen::Dynamic.
 */
template <int K>
class GTSAM_EXPORT ExtendedPose3
    : public MatrixLieGroup<ExtendedPose3<K>,
                            internal::ExtendedPose3Traits<K>::Dimension(),
                            internal::ExtendedPose3Traits<K>::MatrixDim()> {
 public:
  inline constexpr static auto dimension =
      internal::ExtendedPose3Traits<K>::Dimension();
  inline constexpr static auto matrix_dim =
      internal::ExtendedPose3Traits<K>::MatrixDim();

  using Base = MatrixLieGroup<ExtendedPose3<K>, dimension, matrix_dim>;
  using TangentVector = typename Base::TangentVector;
  using Jacobian = typename Base::Jacobian;
  using ChartJacobian = typename Base::ChartJacobian;
  using ComponentJacobian =
      std::conditional_t<dimension == Eigen::Dynamic,
                         OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>,
                         OptionalJacobian<3, dimension>>;
  using LieAlgebra = Eigen::Matrix<double, matrix_dim, matrix_dim>;
  using Matrix3K = Eigen::Matrix<double, 3, K>;

  static_assert(K == Eigen::Dynamic || K >= 1,
                "ExtendedPose3<K>: K should be >= 1 or Eigen::Dynamic.");

 protected:
  Rot3 R_;      ///< Rotation component.
  Matrix3K x_;  ///< K translation-like columns in world frame.

  template <int K_>
  using IsDynamic = typename std::enable_if<K_ == Eigen::Dynamic, void>::type;
  template <int K_>
  using IsFixed = typename std::enable_if<K_ >= 1, void>::type;

  static size_t RuntimeK(const TangentVector& xi);
  static void ZeroJacobian(ChartJacobian H, size_t d);

 public:
  /// @name Constructors
  /// @{

  /// Construct fixed-size identity.
  template <int K_ = K, typename = IsFixed<K_>>
  ExtendedPose3();

  /// Construct dynamic identity, optionally with runtime k.
  template <int K_ = K, typename = IsDynamic<K_>>
  explicit ExtendedPose3(size_t k = 0);

  ExtendedPose3(const ExtendedPose3&) = default;
  ExtendedPose3& operator=(const ExtendedPose3&) = default;

  /// Construct from rotation and 3xK block.
  ExtendedPose3(const Rot3& R, const Matrix3K& x);

  /// Construct from homogeneous matrix representation.
  explicit ExtendedPose3(const LieAlgebra& T);

  /// @}
  /// @name Access
  /// @{

  static size_t Dimension(size_t k);

  /// Number of R^3 columns.
  size_t k() const;

  /// Runtime manifold dimension.
  template <int K_ = K, typename = IsDynamic<K_>>
  size_t dim() const;

  /// Rotation component.
  const Rot3& rotation(ComponentJacobian H = {}) const;

  /// i-th R^3 component, returned by value.
  Point3 x(size_t i, ComponentJacobian H = {}) const;

  const Matrix3K& xMatrix() const;
  Matrix3K& xMatrix();

  /// @}
  /// @name Testable
  /// @{

  void print(const std::string& s = "") const;

  bool equals(const ExtendedPose3& other, double tol = 1e-9) const;

  /// @}
  /// @name Group
  /// @{

  template <int K_ = K, typename = IsFixed<K_>>
  static ExtendedPose3 Identity();

  template <int K_ = K, typename = IsDynamic<K_>>
  static ExtendedPose3 Identity(size_t k = 0);

  ExtendedPose3 inverse() const;

  ExtendedPose3 operator*(const ExtendedPose3& other) const;

  /// @}
  /// @name Lie Group
  /// @{

  static ExtendedPose3 Expmap(const TangentVector& xi, ChartJacobian Hxi = {});

  static TangentVector Logmap(const ExtendedPose3& pose,
                              ChartJacobian Hpose = {});

  Jacobian AdjointMap() const;

  TangentVector Adjoint(const TangentVector& xi_b, ChartJacobian H_this = {},
                        ChartJacobian H_xib = {}) const;

  static Jacobian adjointMap(const TangentVector& xi);

  static TangentVector adjoint(const TangentVector& xi, const TangentVector& y,
                               ChartJacobian Hxi = {},
                               ChartJacobian H_y = {});

  static Jacobian ExpmapDerivative(const TangentVector& xi);

  static Jacobian LogmapDerivative(const TangentVector& xi);

  static Jacobian LogmapDerivative(const ExtendedPose3& pose);

  struct ChartAtOrigin {
    static ExtendedPose3 Retract(const TangentVector& xi,
                                 ChartJacobian Hxi = {});
    static TangentVector Local(const ExtendedPose3& pose,
                               ChartJacobian Hpose = {});
  };

  using LieGroup<ExtendedPose3<K>, dimension>::inverse;

  /// @}
  /// @name Matrix Lie Group
  /// @{

  LieAlgebra matrix() const;

  static LieAlgebra Hat(const TangentVector& xi);

  static TangentVector Vee(const LieAlgebra& X);

  /// @}

  friend std::ostream& operator<<(std::ostream& os, const ExtendedPose3& p) {
    os << "R: " << p.R_ << "\n";
    os << "x: " << p.x_;
    return os;
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(R_);
    ar& BOOST_SERIALIZATION_NVP(x_);
  }
#endif
};

/// Convenience typedef for dynamic k.
using ExtendedPose3Dynamic = ExtendedPose3<Eigen::Dynamic>;

template <int K>
struct traits<ExtendedPose3<K>>
    : public internal::MatrixLieGroup<
          ExtendedPose3<K>, internal::ExtendedPose3Traits<K>::MatrixDim()> {};

template <int K>
struct traits<const ExtendedPose3<K>>
    : public internal::MatrixLieGroup<
          ExtendedPose3<K>, internal::ExtendedPose3Traits<K>::MatrixDim()> {};

}  // namespace gtsam

#include "ExtendedPose3-inl.h"
