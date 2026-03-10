/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOGroup.h
 * @brief   Dynamic VI-SLAM symmetry group from Eq. (22)
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam_unstable/dllexport.h>

#include <string>
#include <vector>

namespace gtsam {

/** Similarity-orthogonal group SOT(3) represented as SO(3) x R (log-scale). */
using SOT3 = ProductLieGroup<SO3, Vector1>;

/**
 * Eq. (22) VI-SLAM symmetry group:
 *   G = SE_2(3) x R^6 x SE(3) x SOT(3)^n.
 *
 * The Lie structure is delegated to nested ProductLieGroup/PowerLieGroup
 * instances. This wrapper keeps the paper-native named accessors and the
 * landmark-id metadata used to align runtime landmark blocks.
 */
class GTSAM_UNSTABLE_EXPORT VIOGroup
    : public LieGroup<VIOGroup, Eigen::Dynamic> {
 public:
  static constexpr int dimension = Eigen::Dynamic;

  using TangentVector = Vector;
  using Jacobian = Matrix;
  using ChartJacobian = OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

  using SE23 = ExtendedPose3<2>;
  using BiasGroup = Vector6;
  using ExtrinsicsGroup = Pose3;
  using LandmarkGroup = PowerLieGroup<SOT3, Eigen::Dynamic>;
  using SensorCore = ProductLieGroup<SE23, BiasGroup>;
  using LandmarkCore = ProductLieGroup<ExtrinsicsGroup, LandmarkGroup>;
  using VIOGroupCore = ProductLieGroup<SensorCore, LandmarkCore>;

  /** @name Constructors */
  /** @{ */

  /** Construct identity with zero landmarks and empty ids. */
  VIOGroup();
  /**
   * Construct identity with @p n landmarks and empty ids metadata.
   * @param n number of landmark SOT(3) elements.
   */
  explicit VIOGroup(size_t n);
  /**
   * Construct identity with ids metadata.
   * @param ids ordered landmark ids; size sets landmark count.
   */
  explicit VIOGroup(const std::vector<int>& ids);
  /**
   * Construct from explicit factors.
   * @param A SE2(3) factor.
   * @param beta R^6 bias factor.
   * @param B SE(3) extrinsics factor.
   * @param Q SOT(3)^n landmark factor.
   * @param ids optional ordered landmark ids metadata.
   */
  VIOGroup(const SE23& A, const Vector6& beta, const Pose3& B,
           const LandmarkGroup& Q, const std::vector<int>& ids = {});

  /** @} */
  /** @name Factories */
  /** @{ */

  /** Identity with zero landmarks. */
  static VIOGroup Identity();
  /**
   * Identity with @p n landmarks and empty ids.
   * @param n number of landmark factors.
   */
  static VIOGroup Identity(size_t n);
  /**
   * Identity with explicit ids metadata.
   * @param ids ordered landmark ids.
   */
  static VIOGroup Identity(const std::vector<int>& ids);

  /** @} */
  /** @name Accessors */
  /** @{ */

  /** Access SE2(3) factor A. */
  const SE23& A() const;
  SE23& A();

  /** Access R^6 bias factor beta. */
  const Vector6& beta() const;
  Vector6& beta();

  /** Access SE(3) factor B. */
  const Pose3& B() const;
  Pose3& B();

  /** Access landmark SOT(3)^n factor Q. */
  const LandmarkGroup& Q() const;
  LandmarkGroup& Q();

  /** Access ordered landmark ids metadata. */
  const std::vector<int>& ids() const;

  /** Number of landmarks. */
  size_t n() const;
  size_t dim() const;

  /** @} */
  /** @name Group Operations */
  /** @{ */

  VIOGroup operator*(const VIOGroup& other) const;
  VIOGroup inverse() const;

  /** @} */
  /** @name Lie Group Interface */
  /** @{ */

  /**
   * Compose with optional Jacobians.
   * @param other right-hand group element.
   * @param H1 optional derivative wrt this element.
   * @param H2 optional derivative wrt other.
   * @return composed group element.
   */
  VIOGroup compose(const VIOGroup& other, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  /**
   * Relative transform with optional Jacobians.
   * @param other target group element.
   * @param H1 optional derivative wrt this element.
   * @param H2 optional derivative wrt other.
   * @return this^{-1} * other.
   */
  VIOGroup between(const VIOGroup& other, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  /**
   * Retraction at this element.
   * @param v tangent increment of size dim().
   * @param H1 optional derivative wrt this element.
   * @param H2 optional derivative wrt increment.
   * @return retracted element.
   */
  VIOGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  /**
   * Local coordinates from this to @p other.
   * @param other target group element.
   * @param H1 optional derivative wrt this element.
   * @param H2 optional derivative wrt other.
   * @return tangent vector.
   */
  TangentVector localCoordinates(const VIOGroup& other, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const;

  /**
   * Exponential map from tangent to group.
   * @param v tangent vector with size 21 + 4*n.
   * @param H optional derivative wrt v.
   * @return group element.
   */
  static VIOGroup Expmap(const TangentVector& v, ChartJacobian H = {});
  /**
   * Logarithm map from group to tangent.
   * @param g group element.
   * @param H optional derivative wrt g.
   * @return tangent vector.
   */
  static TangentVector Logmap(const VIOGroup& g, ChartJacobian H = {});

  /**
   * Adjoint matrix of this group element.
   * @return dim() x dim() adjoint matrix.
   */
  Jacobian AdjointMap() const;

  struct ChartAtOrigin {
    /**
     * Retraction at identity.
     * @param v tangent vector.
     * @param H optional derivative wrt v.
     * @return group element.
     */
    static VIOGroup Retract(const TangentVector& v, ChartJacobian H = {});
    /**
     * Local coordinates to identity.
     * @param g group element.
     * @param H optional derivative wrt g.
     * @return tangent vector.
     */
    static TangentVector Local(const VIOGroup& g, ChartJacobian H = {});
  };

  using LieGroup<VIOGroup, Eigen::Dynamic>::inverse;

  /** @} */
  /** @name Testable */
  /** @{ */

  void print(const std::string& s = "") const;
  bool equals(const VIOGroup& other, double tol = 1e-9) const;

  /** @} */

 private:
  VIOGroup(const VIOGroupCore& core, std::vector<int> ids);

  std::vector<int> resolvedIds(const VIOGroup& other) const;

  static size_t inferLandmarkCount(const TangentVector& v);

  VIOGroupCore core_;
  std::vector<int> ids_;
};

template <>
struct traits<VIOGroup> : internal::LieGroup<VIOGroup> {};

template <>
struct traits<const VIOGroup> : internal::LieGroup<VIOGroup> {};

}  // namespace gtsam
