/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file ProductLieGroup.h
 * @date May, 2015
 * @author Frank Dellaert
 * @brief Group product of two Lie Groups
 */

#pragma once

#include <gtsam/base/Lie.h>

#include <utility>  // pair

namespace gtsam {

/**
 * @brief Template to construct the product Lie group of two other Lie groups
 * Assumes Lie group structure for G and H
 */
template <typename G, typename H>
class ProductLieGroup : public std::pair<G, H> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<H>);

 public:
  /// Base pair type
  typedef std::pair<G, H> Base;

 protected:
  /// Dimensions of the two subgroups
  static constexpr size_t dimension1 = traits<G>::dimension;
  static constexpr size_t dimension2 = traits<H>::dimension;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  ProductLieGroup() : Base(traits<G>::Identity(), traits<H>::Identity()) {}

  /// Construct from two subgroup elements
  ProductLieGroup(const G& g, const H& h) : Base(g, h) {}

  /// Construct from base pair
  ProductLieGroup(const Base& base) : Base(base) {}

  /// @}
  /// @name Group Operations
  /// @{

  typedef multiplicative_group_tag group_flavor;

  /// Identity element
  static ProductLieGroup Identity() { return ProductLieGroup(); }

  /// Group multiplication
  ProductLieGroup operator*(const ProductLieGroup& other) const {
    return ProductLieGroup(traits<G>::Compose(this->first, other.first),
                           traits<H>::Compose(this->second, other.second));
  }

  /// Group inverse
  ProductLieGroup inverse() const {
    return ProductLieGroup(traits<G>::Inverse(this->first),
                           traits<H>::Inverse(this->second));
  }

  /// Compose with another element (same as operator*)
  ProductLieGroup compose(const ProductLieGroup& g) const {
    return (*this) * g;
  }

  /// Calculate relative transformation
  ProductLieGroup between(const ProductLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Manifold dimension
  static constexpr size_t dimension = dimension1 + dimension2;

  /// Return manifold dimension
  static size_t Dim() { return dimension; }

  /// Return manifold dimension
  size_t dim() const { return dimension; }

  /// Tangent vector type
  using TangentVector = Eigen::Matrix<double, static_cast<int>(dimension), 1>;

  /// Chart Jacobian type
  using ChartJacobian = OptionalJacobian<dimension, dimension>;

  /// Retract to manifold
  ProductLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                          ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "ProductLieGroup::retract derivatives not implemented yet");
    }
    G g = traits<G>::Retract(this->first, v.template head<dimension1>());
    H h = traits<H>::Retract(this->second, v.template tail<dimension2>());
    return ProductLieGroup(g, h);
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const ProductLieGroup& g,
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "ProductLieGroup::localCoordinates derivatives not implemented yet");
    }
    typename traits<G>::TangentVector v1 =
        traits<G>::Local(this->first, g.first);
    typename traits<H>::TangentVector v2 =
        traits<H>::Local(this->second, g.second);
    TangentVector v;
    v << v1, v2;
    return v;
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

 protected:
  /// Jacobian types for internal use
  using Jacobian = Eigen::Matrix<double, static_cast<int>(dimension), static_cast<int>(dimension)>;
  using Jacobian1 = Eigen::Matrix<double, static_cast<int>(dimension1), static_cast<int>(dimension1)>;
  using Jacobian2 = Eigen::Matrix<double, static_cast<int>(dimension2), static_cast<int>(dimension2)>;

 public:
  /// Compose with Jacobians
  ProductLieGroup compose(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Compose(this->first, other.first, H1 ? &D_g_first : 0);
    H h = traits<H>::Compose(this->second, other.second, H1 ? &D_h_second : 0);
    if (H1) {
      H1->setZero();
      H1->template topLeftCorner<dimension1, dimension1>() = D_g_first;
      H1->template bottomRightCorner<dimension2, dimension2>() = D_h_second;
    }
    if (H2) *H2 = Jacobian::Identity();
    return ProductLieGroup(g, h);
  }

  /// Between with Jacobians
  ProductLieGroup between(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Between(this->first, other.first, H1 ? &D_g_first : 0);
    H h = traits<H>::Between(this->second, other.second, H1 ? &D_h_second : 0);
    if (H1) {
      H1->setZero();
      H1->template topLeftCorner<dimension1, dimension1>() = D_g_first;
      H1->template bottomRightCorner<dimension2, dimension2>() = D_h_second;
    }
    if (H2) *H2 = Jacobian::Identity();
    return ProductLieGroup(g, h);
  }

  /// Inverse with Jacobian
  ProductLieGroup inverse(ChartJacobian D) const {
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Inverse(this->first, D ? &D_g_first : 0);
    H h = traits<H>::Inverse(this->second, D ? &D_h_second : 0);
    if (D) {
      D->setZero();
      D->template topLeftCorner<dimension1, dimension1>() = D_g_first;
      D->template bottomRightCorner<dimension2, dimension2>() = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map
  static ProductLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Expmap(v.template head<dimension1>(), Hv ? &D_g_first : 0);
    H h =
        traits<H>::Expmap(v.template tail<dimension2>(), Hv ? &D_h_second : 0);
    if (Hv) {
      Hv->setZero();
      Hv->template topLeftCorner<dimension1, dimension1>() = D_g_first;
      Hv->template bottomRightCorner<dimension2, dimension2>() = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Logarithmic map
  static TangentVector Logmap(const ProductLieGroup& p, ChartJacobian Hp = {}) {
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    typename traits<G>::TangentVector v1 =
        traits<G>::Logmap(p.first, Hp ? &D_g_first : 0);
    typename traits<H>::TangentVector v2 =
        traits<H>::Logmap(p.second, Hp ? &D_h_second : 0);
    TangentVector v;
    v << v1, v2;
    if (Hp) {
      Hp->setZero();
      Hp->template topLeftCorner<dimension1, dimension1>() = D_g_first;
      Hp->template bottomRightCorner<dimension2, dimension2>() = D_h_second;
    }
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const ProductLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  ProductLieGroup expmap(const TangentVector& v) const {
    return compose(ProductLieGroup::Expmap(v));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const ProductLieGroup& g) const {
    return ProductLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const auto& adjG = traits<G>::AdjointMap(this->first);
    const auto& adjH = traits<H>::AdjointMap(this->second);
    size_t d1 = adjG.rows(), d2 = adjH.rows();
    Matrix adj = Matrix::Zero(d1 + d2, d1 + d2);
    adj.block(0, 0, d1, d1) = adjG;
    adj.block(d1, d1, d2, d2) = adjH;
    return adj;
  }

  /// @}
};

/// Traits specialization for ProductLieGroup
template <typename G, typename H>
struct traits<ProductLieGroup<G, H>>
    : internal::LieGroupTraits<ProductLieGroup<G, H>> {};

}  // namespace gtsam