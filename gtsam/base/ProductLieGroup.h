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
#include <gtsam/base/Testable.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>  // pair
#include <vector>

namespace gtsam {

/**
 * @brief Template to construct the product Lie group of two other Lie groups
 * Assumes Lie group structure for G and H
 */
template <typename G, typename H>
class ProductLieGroup : public std::pair<G, H> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<H>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<H>);

 public:
  /// Base pair type
  typedef std::pair<G, H> Base;

 protected:
  /// Dimensions of the two subgroups
  inline constexpr static int dimension1 = traits<G>::dimension;
  inline constexpr static int dimension2 = traits<H>::dimension;
  inline constexpr static bool firstDynamic = dimension1 == Eigen::Dynamic;
  inline constexpr static bool secondDynamic = dimension2 == Eigen::Dynamic;

 public:
  /// Manifold dimension
  inline constexpr static int dimension =
      firstDynamic || secondDynamic ? Eigen::Dynamic : dimension1 + dimension2;

  /// Tangent vector type
  using TangentVector = std::conditional_t<dimension == Eigen::Dynamic, Vector,
                                           Eigen::Matrix<double, dimension, 1>>;

  /// Chart Jacobian type
  using ChartJacobian =
      std::conditional_t<dimension == Eigen::Dynamic,
                         OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>,
                         OptionalJacobian<dimension, dimension>>;

  /// Jacobian types for internal use
  using Jacobian =
      std::conditional_t<dimension == Eigen::Dynamic, Matrix,
                         Eigen::Matrix<double, dimension, dimension>>;
  using Jacobian1 = typename traits<G>::Jacobian;
  using Jacobian2 = typename traits<H>::Jacobian;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  ProductLieGroup() : Base(defaultIdentity<G>(), defaultIdentity<H>()) {}

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
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "operator*");
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
  inline constexpr static int manifoldDimension = dimension;

  /// Return manifold dimension
  static constexpr int Dim() { return manifoldDimension; }

  /// Return manifold dimension
  size_t dim() const { return combinedDimension(firstDim(), secondDim()); }

  /// Retract to manifold
  ProductLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                          ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "ProductLieGroup::retract derivatives not implemented yet");
    }
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    if (static_cast<size_t>(v.size()) !=
        combinedDimension(firstDimension, secondDimension)) {
      throw std::invalid_argument(
          "ProductLieGroup::retract tangent dimension does not match product "
          "dimension");
    }
    G g = traits<G>::Retract(this->first,
                             tangentSegment<G>(v, 0, firstDimension));
    H h = traits<H>::Retract(
        this->second, tangentSegment<H>(v, firstDimension, secondDimension));
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
    checkMatchingDimensions(firstDim(), secondDim(), g.firstDim(),
                            g.secondDim(), "localCoordinates");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    typename traits<G>::TangentVector v1 =
        traits<G>::Local(this->first, g.first);
    typename traits<H>::TangentVector v2 =
        traits<H>::Local(this->second, g.second);
    return makeTangentVector(v1, v2, firstDimension, secondDimension);
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// Compose with Jacobians
  ProductLieGroup compose(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "compose");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g =
        traits<G>::Compose(this->first, other.first, H1 ? &D_g_first : nullptr);
    H h = traits<H>::Compose(this->second, other.second,
                             H1 ? &D_h_second : nullptr);
    if (H1) {
      *H1 = zeroJacobian(productDimension);
      H1->block(0, 0, firstDimension, firstDimension) = D_g_first;
      H1->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    if (H2) *H2 = identityJacobian(productDimension);
    return ProductLieGroup(g, h);
  }

  /// Between with Jacobians
  ProductLieGroup between(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "between");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g =
        traits<G>::Between(this->first, other.first, H1 ? &D_g_first : nullptr);
    H h = traits<H>::Between(this->second, other.second,
                             H1 ? &D_h_second : nullptr);
    if (H1) {
      *H1 = zeroJacobian(productDimension);
      H1->block(0, 0, firstDimension, firstDimension) = D_g_first;
      H1->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    if (H2) *H2 = identityJacobian(productDimension);
    return ProductLieGroup(g, h);
  }

  /// Inverse with Jacobian
  ProductLieGroup inverse(ChartJacobian D) const {
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Inverse(this->first, D ? &D_g_first : nullptr);
    H h = traits<H>::Inverse(this->second, D ? &D_h_second : nullptr);
    if (D) {
      *D = zeroJacobian(productDimension);
      D->block(0, 0, firstDimension, firstDimension) = D_g_first;
      D->block(firstDimension, firstDimension, secondDimension,
               secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map
  static ProductLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    if constexpr (firstDynamic && secondDynamic) {
      if (v.size() == 0) {
        if (Hv) *Hv = Matrix::Zero(0, 0);
        return ProductLieGroup();
      }
      throw std::invalid_argument(
          "ProductLieGroup::Expmap requires split tangent vectors when both "
          "factors are dynamic");
    } else if constexpr (firstDynamic) {
      if (v.size() < dimension2) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap tangent dimension is too small for the "
            "fixed second factor");
      }
      const size_t firstDimension = static_cast<size_t>(v.size() - dimension2);
      return expmapWithDimensions(v, firstDimension,
                                  static_cast<size_t>(dimension2), Hv);
    } else if constexpr (secondDynamic) {
      if (v.size() < dimension1) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap tangent dimension is too small for the "
            "fixed first factor");
      }
      const size_t secondDimension = static_cast<size_t>(v.size() - dimension1);
      return expmapWithDimensions(v, static_cast<size_t>(dimension1),
                                  secondDimension, Hv);
    } else {
      return expmapWithDimensions(v, static_cast<size_t>(dimension1),
                                  static_cast<size_t>(dimension2), Hv);
    }
  }

  /// Exponential map from subgroup tangent vectors
  static ProductLieGroup Expmap(
      const Eigen::Ref<const typename traits<G>::TangentVector>& v1,
      const Eigen::Ref<const typename traits<H>::TangentVector>& v2,
      ChartJacobian Hv) {
    const size_t firstDimension = static_cast<size_t>(v1.size());
    const size_t secondDimension = static_cast<size_t>(v2.size());
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Expmap(v1, Hv ? &D_g_first : nullptr);
    H h = traits<H>::Expmap(v2, Hv ? &D_h_second : nullptr);
    if (Hv) {
      *Hv = zeroJacobian(combinedDimension(firstDimension, secondDimension));
      Hv->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hv->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map from subgroup tangent vectors
  static ProductLieGroup Expmap(
      const Eigen::Ref<const typename traits<G>::TangentVector>& v1,
      const Eigen::Ref<const typename traits<H>::TangentVector>& v2) {
    return Expmap(v1, v2, {});
  }

  /// Logarithmic map
  static TangentVector Logmap(const ProductLieGroup& p, ChartJacobian Hp = {}) {
    const size_t firstDimension = p.firstDim();
    const size_t secondDimension = p.secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    typename traits<G>::TangentVector v1 =
        traits<G>::Logmap(p.first, Hp ? &D_g_first : nullptr);
    typename traits<H>::TangentVector v2 =
        traits<H>::Logmap(p.second, Hp ? &D_h_second : nullptr);
    TangentVector v =
        makeTangentVector(v1, v2, firstDimension, secondDimension);
    if (Hp) {
      *Hp = zeroJacobian(productDimension);
      Hp->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hp->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
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
    return compose(expmapWithDimensions(v, firstDim(), secondDim()));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const ProductLieGroup& g) const {
    return ProductLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const auto adjG = traits<G>::AdjointMap(this->first);
    const auto adjH = traits<H>::AdjointMap(this->second);
    const size_t d1 = static_cast<size_t>(adjG.rows());
    const size_t d2 = static_cast<size_t>(adjH.rows());
    Jacobian adj = zeroJacobian(d1 + d2);
    adj.block(0, 0, d1, d1) = adjG;
    adj.block(d1, d1, d2, d2) = adjH;
    return adj;
  }

  /// @}

 protected:
  template <typename T>
  static T defaultIdentity() {
    if constexpr (traits<T>::dimension == Eigen::Dynamic) {
      return T();
    } else {
      return traits<T>::Identity();
    }
  }

  size_t firstDim() const { return traits<G>::GetDimension(this->first); }
  size_t secondDim() const { return traits<H>::GetDimension(this->second); }

  static size_t combinedDimension(size_t d1, size_t d2) { return d1 + d2; }

  template <typename T, int Dim = traits<T>::dimension>
  static typename traits<T>::TangentVector tangentSegment(
      const TangentVector& v, size_t start, size_t runtimeDimension) {
    const int startIndex = static_cast<int>(start);
    const int runtimeIndex = static_cast<int>(runtimeDimension);
    if constexpr (Dim == Eigen::Dynamic) {
      return v.segment(startIndex, runtimeIndex);
    } else {
      static_cast<void>(runtimeDimension);
      return v.template segment<Dim>(startIndex);
    }
  }

  static TangentVector makeTangentVector(
      const typename traits<G>::TangentVector& v1,
      const typename traits<H>::TangentVector& v2, size_t firstDimension,
      size_t secondDimension) {
    const int firstIndex = static_cast<int>(firstDimension);
    const int secondIndex = static_cast<int>(secondDimension);
    if constexpr (dimension == Eigen::Dynamic) {
      TangentVector v(combinedDimension(firstDimension, secondDimension));
      v.segment(0, firstIndex) = v1;
      v.segment(firstIndex, secondIndex) = v2;
      return v;
    } else {
      static_cast<void>(firstDimension);
      static_cast<void>(secondDimension);
      TangentVector v;
      v << v1, v2;
      return v;
    }
  }

  static Jacobian zeroJacobian(size_t productDimension) {
    if constexpr (dimension == Eigen::Dynamic) {
      return Jacobian::Zero(productDimension, productDimension);
    } else {
      static_cast<void>(productDimension);
      return Jacobian::Zero();
    }
  }

  static Jacobian identityJacobian(size_t productDimension) {
    if constexpr (dimension == Eigen::Dynamic) {
      return Jacobian::Identity(productDimension, productDimension);
    } else {
      static_cast<void>(productDimension);
      return Jacobian::Identity();
    }
  }

  static void checkMatchingDimensions(size_t first1, size_t second1,
                                      size_t first2, size_t second2,
                                      const char* operation) {
    if (first1 != first2 || second1 != second2) {
      throw std::invalid_argument(std::string("ProductLieGroup::") + operation +
                                  " requires matching component dimensions");
    }
  }

  static ProductLieGroup expmapWithDimensions(const TangentVector& v,
                                              size_t firstDimension,
                                              size_t secondDimension,
                                              ChartJacobian Hv = {}) {
    if (static_cast<size_t>(v.size()) !=
        combinedDimension(firstDimension, secondDimension)) {
      throw std::invalid_argument(
          "ProductLieGroup::Expmap tangent dimension does not match requested "
          "component dimensions");
    }
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Expmap(tangentSegment<G>(v, 0, firstDimension),
                            Hv ? &D_g_first : nullptr);
    H h =
        traits<H>::Expmap(tangentSegment<H>(v, firstDimension, secondDimension),
                          Hv ? &D_h_second : nullptr);
    if (Hv) {
      *Hv = zeroJacobian(combinedDimension(firstDimension, secondDimension));
      Hv->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hv->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

 public:
  /// @name Testable interface
  /// @{
  void print(const std::string& s = "") const {
    std::cout << s << "ProductLieGroup" << std::endl;
    traits<G>::Print(this->first, "  first");
    traits<H>::Print(this->second, "  second");
  }

  bool equals(const ProductLieGroup& other, double tol = 1e-9) const {
    return traits<G>::Equals(this->first, other.first, tol) &&
           traits<H>::Equals(this->second, other.second, tol);
  }
  /// @}
};

/**
 * @brief Shared implementation for fixed-size and dynamic-count PowerLieGroup
 */
template <typename T, int N>
struct PowerLieGroupJacobianStorage {
  using type = std::array<T, N>;
};

template <typename T>
struct PowerLieGroupJacobianStorage<T, Eigen::Dynamic> {
  using type = std::vector<T>;
};

template <typename G, int N, typename Derived>
class PowerLieGroupBase {
 protected:
  static constexpr bool isDynamic = (N == Eigen::Dynamic);
  static constexpr int baseDimension = traits<G>::dimension;

 public:
  typedef multiplicative_group_tag group_flavor;

  static constexpr int dimension =
      isDynamic ? Eigen::Dynamic : N * baseDimension;

  typedef std::conditional_t<isDynamic, Vector,
                             Eigen::Matrix<double, dimension, 1>>
      TangentVector;

  typedef std::conditional_t<isDynamic,
                             OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>,
                             OptionalJacobian<dimension, dimension>>
      ChartJacobian;

  typedef std::conditional_t<isDynamic, Matrix,
                             Eigen::Matrix<double, dimension, dimension>>
      Jacobian;

  using BaseJacobian = typename traits<G>::Jacobian;
  using JacobianStorage =
      typename PowerLieGroupJacobianStorage<BaseJacobian, N>::type;

 protected:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  Derived& derived() { return static_cast<Derived&>(*this); }

  static size_t totalDimension(size_t count) {
    return count * static_cast<size_t>(baseDimension);
  }

  static Eigen::Index offset(size_t i) {
    return static_cast<Eigen::Index>(i * static_cast<size_t>(baseDimension));
  }

  size_t componentCount() const {
    if constexpr (isDynamic) {
      return derived().size();
    } else {
      return N;
    }
  }

  static void checkDynamicTangentSize(const TangentVector& v, size_t count,
                                      const char* operation) {
    if constexpr (isDynamic) {
      if (static_cast<size_t>(v.size()) != totalDimension(count)) {
        throw std::invalid_argument(
            std::string("PowerLieGroup::") + operation +
            " tangent dimension does not match group dimension");
      }
    } else {
      static_cast<void>(v);
      static_cast<void>(count);
      static_cast<void>(operation);
    }
  }

  void checkMatchingCounts(const Derived& other, const char* operation) const {
    if constexpr (isDynamic) {
      if (derived().size() != other.size()) {
        throw std::invalid_argument(std::string("PowerLieGroup::") + operation +
                                    " requires matching component counts");
      }
    } else {
      static_cast<void>(other);
      static_cast<void>(operation);
    }
  }

  static typename traits<G>::TangentVector tangentSegment(
      const TangentVector& v, size_t i) {
    if constexpr (isDynamic) {
      return v.segment(offset(i), baseDimension);
    } else {
      return v.template segment<baseDimension>(i * baseDimension);
    }
  }

  static Derived makeResult(size_t count) {
    if constexpr (isDynamic) {
      return Derived(count);
    } else {
      static_cast<void>(count);
      return Derived();
    }
  }

  static JacobianStorage makeJacobianStorage(size_t count) {
    if constexpr (isDynamic) {
      return JacobianStorage(count);
    } else {
      static_cast<void>(count);
      return JacobianStorage();
    }
  }

  static void assignTangentSegment(
      TangentVector& v, size_t i, const typename traits<G>::TangentVector& vi) {
    if constexpr (isDynamic) {
      v.segment(offset(i), baseDimension) = vi;
    } else {
      v.template segment<baseDimension>(i * baseDimension) = vi;
    }
  }

  template <typename MatrixType>
  static void assignJacobianBlock(MatrixType& H, size_t i,
                                  const BaseJacobian& block) {
    if constexpr (isDynamic) {
      H.block(offset(i), offset(i), baseDimension, baseDimension) = block;
    } else {
      H.template block<baseDimension, baseDimension>(i * baseDimension,
                                                     i * baseDimension) = block;
    }
  }

  static void fillJacobianBlocks(ChartJacobian H,
                                 const JacobianStorage& jacobians,
                                 size_t count) {
    if (!H) return;
    *H = zeroJacobian(count);
    for (size_t i = 0; i < count; ++i) {
      assignJacobianBlock(*H, i, jacobians[i]);
    }
  }

 public:
  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// Return manifold dimension
  size_t dim() const { return totalDimension(componentCount()); }

  /// Group multiplication
  Derived operator*(const Derived& other) const {
    checkMatchingCounts(other, "operator*");
    Derived result = makeResult(componentCount());
    for (size_t i = 0; i < componentCount(); ++i) {
      result[i] = traits<G>::Compose(derived()[i], other[i]);
    }
    return result;
  }

  /// Group inverse
  Derived inverse() const {
    Derived result = makeResult(componentCount());
    for (size_t i = 0; i < componentCount(); ++i) {
      result[i] = traits<G>::Inverse(derived()[i]);
    }
    return result;
  }

  /// Compose with another element (same as operator*)
  Derived compose(const Derived& g) const { return (*this) * g; }

  /// Calculate relative transformation
  Derived between(const Derived& g) const { return this->inverse() * g; }

  /// Retract to manifold
  Derived retract(const TangentVector& v, ChartJacobian H1 = {},
                  ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "PowerLieGroup::retract derivatives not implemented yet");
    }
    const size_t count = componentCount();
    checkDynamicTangentSize(v, count, "retract");
    Derived result = makeResult(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = traits<G>::Retract(derived()[i], tangentSegment(v, i));
    }
    return result;
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const Derived& g, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "PowerLieGroup::localCoordinates derivatives not implemented yet");
    }
    checkMatchingCounts(g, "localCoordinates");
    TangentVector v = zeroTangent(componentCount());
    for (size_t i = 0; i < componentCount(); ++i) {
      assignTangentSegment(v, i, traits<G>::Local(derived()[i], g[i]));
    }
    return v;
  }

  /// Compose with Jacobians
  Derived compose(const Derived& other, ChartJacobian H1,
                  ChartJacobian H2 = {}) const {
    checkMatchingCounts(other, "compose");
    const size_t count = componentCount();
    JacobianStorage jacobians = makeJacobianStorage(count);
    Derived result = makeResult(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = traits<G>::Compose(derived()[i], other[i],
                                     H1 ? &jacobians[i] : nullptr);
    }
    fillJacobianBlocks(H1, jacobians, count);
    if (H2) *H2 = identityJacobian(count);
    return result;
  }

  /// Between with Jacobians
  Derived between(const Derived& other, ChartJacobian H1,
                  ChartJacobian H2 = {}) const {
    checkMatchingCounts(other, "between");
    const size_t count = componentCount();
    JacobianStorage jacobians = makeJacobianStorage(count);
    Derived result = makeResult(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = traits<G>::Between(derived()[i], other[i],
                                     H1 ? &jacobians[i] : nullptr);
    }
    fillJacobianBlocks(H1, jacobians, count);
    if (H2) *H2 = identityJacobian(count);
    return result;
  }

  /// Inverse with Jacobian
  Derived inverse(ChartJacobian D) const {
    const size_t count = componentCount();
    JacobianStorage jacobians = makeJacobianStorage(count);
    Derived result = makeResult(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = traits<G>::Inverse(derived()[i], D ? &jacobians[i] : nullptr);
    }
    fillJacobianBlocks(D, jacobians, count);
    return result;
  }

  /// Exponential map
  static Derived Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    size_t count = 0;
    if constexpr (isDynamic) {
      if (v.size() % baseDimension != 0) {
        throw std::invalid_argument(
            "PowerLieGroup::Expmap tangent dimension must be divisible by base "
            "group dimension");
      }
      count = static_cast<size_t>(v.size() /
                                  static_cast<Eigen::Index>(baseDimension));
    } else {
      count = N;
    }
    JacobianStorage jacobians = makeJacobianStorage(count);
    Derived result = makeResult(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] =
          traits<G>::Expmap(tangentSegment(v, i), Hv ? &jacobians[i] : nullptr);
    }
    fillJacobianBlocks(Hv, jacobians, count);
    return result;
  }

  /// Logarithmic map
  static TangentVector Logmap(const Derived& p, ChartJacobian Hp = {}) {
    const size_t count = isDynamic ? p.size() : N;
    TangentVector v = zeroTangent(count);
    JacobianStorage jacobians = makeJacobianStorage(count);
    for (size_t i = 0; i < count; ++i) {
      assignTangentSegment(
          v, i, traits<G>::Logmap(p[i], Hp ? &jacobians[i] : nullptr));
    }
    fillJacobianBlocks(Hp, jacobians, count);
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const Derived& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  Derived expmap(const TangentVector& v) const { return compose(Expmap(v)); }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const Derived& g) const { return Logmap(between(g)); }

  /// Adjoint map
  Jacobian AdjointMap() const {
    Jacobian adj = zeroJacobian(componentCount());
    for (size_t i = 0; i < componentCount(); ++i) {
      assignJacobianBlock(adj, i, traits<G>::AdjointMap(derived()[i]));
    }
    return adj;
  }

  /// Print for debugging
  void print(const std::string& s = "") const {
    std::cout << s << "PowerLieGroup" << std::endl;
    for (size_t i = 0; i < componentCount(); ++i) {
      traits<G>::Print(derived()[i], "  component[" + std::to_string(i) + "]");
    }
  }

  /// Equality with tolerance
  bool equals(const Derived& other, double tol = 1e-9) const {
    if constexpr (isDynamic) {
      if (derived().size() != other.size()) {
        return false;
      }
    }
    for (size_t i = 0; i < componentCount(); ++i) {
      if (!traits<G>::Equals(derived()[i], other[i], tol)) {
        return false;
      }
    }
    return true;
  }

 protected:
  static TangentVector zeroTangent(size_t count) {
    if constexpr (isDynamic) {
      return TangentVector::Zero(totalDimension(count));
    } else {
      static_cast<void>(count);
      return TangentVector::Zero();
    }
  }

  static Jacobian zeroJacobian(size_t count) {
    if constexpr (isDynamic) {
      return Jacobian::Zero(totalDimension(count), totalDimension(count));
    } else {
      static_cast<void>(count);
      return Jacobian::Zero();
    }
  }

  static Jacobian identityJacobian(size_t count) {
    if constexpr (isDynamic) {
      return Jacobian::Identity(totalDimension(count), totalDimension(count));
    } else {
      static_cast<void>(count);
      return Jacobian::Identity();
    }
  }
};

/**
 * @brief Template to construct the N-fold power of a Lie group
 * Represents the group G^N = G x G x ... x G (N times)
 * Assumes Lie group structure for fixed-size G and fixed N >= 1
 */
template <typename G, int N>
class PowerLieGroup : public std::array<G, N>,
                      public PowerLieGroupBase<G, N, PowerLieGroup<G, N>> {
  static_assert(N >= 1, "PowerLieGroup requires N >= 1");
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);
  static_assert(traits<G>::dimension != Eigen::Dynamic,
                "PowerLieGroup requires a fixed-size base group");

 public:
  /// Base array type
  typedef std::array<G, N> Base;
  typedef PowerLieGroupBase<G, N, PowerLieGroup> Helper;
  using typename Helper::BaseJacobian;
  using typename Helper::ChartJacobian;
  using typename Helper::Jacobian;
  using typename Helper::TangentVector;
  static constexpr int dimension = Helper::dimension;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  PowerLieGroup() { this->fill(traits<G>::Identity()); }

  /// Construct from array of group elements
  PowerLieGroup(const Base& elements) : Base(elements) {}

  /// Construct from initializer list
  PowerLieGroup(const std::initializer_list<G>& elements) {
    if (elements.size() != N) {
      throw std::invalid_argument(
          "PowerLieGroup: initializer list size must equal N");
    }
    std::copy(elements.begin(), elements.end(), this->begin());
  }

  /// @}
  /// @name Group Operations
  /// @{

  /// Identity element
  static PowerLieGroup Identity() { return PowerLieGroup(); }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// @}
};

/**
 * @brief Dynamic-count specialization of PowerLieGroup
 * Represents G^N for runtime-sized N while keeping G fixed-size
 */
template <typename G>
class PowerLieGroup<G, Eigen::Dynamic>
    : public std::vector<G>,
      public PowerLieGroupBase<G, Eigen::Dynamic,
                               PowerLieGroup<G, Eigen::Dynamic>> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);
  static_assert(traits<G>::dimension != Eigen::Dynamic,
                "PowerLieGroup requires a fixed-size base group");

 public:
  /// Base vector type
  typedef std::vector<G> Base;
  typedef PowerLieGroupBase<G, Eigen::Dynamic, PowerLieGroup> Helper;
  using typename Helper::BaseJacobian;
  using typename Helper::ChartJacobian;
  using typename Helper::Jacobian;
  using typename Helper::TangentVector;
  static constexpr int dimension = Helper::dimension;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields a zero-length placeholder identity
  PowerLieGroup() = default;

  /// Construct a runtime-sized identity element
  explicit PowerLieGroup(size_t count) : Base(count, traits<G>::Identity()) {}

  /// Construct from vector of group elements
  PowerLieGroup(const Base& elements) : Base(elements) {}

  /// Construct from initializer list
  PowerLieGroup(const std::initializer_list<G>& elements) : Base(elements) {}

  /// @}
  /// @name Group Operations
  /// @{

  /// Identity element
  static PowerLieGroup Identity() { return PowerLieGroup(); }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// @}
};

/// Traits specialization for ProductLieGroup
template <typename G, typename H>
struct traits<ProductLieGroup<G, H>>
    : internal::LieGroup<ProductLieGroup<G, H>> {};

/// Traits specialization for PowerLieGroup
template <typename G, int N>
struct traits<PowerLieGroup<G, N>> : internal::LieGroup<PowerLieGroup<G, N>> {};

}  // namespace gtsam
