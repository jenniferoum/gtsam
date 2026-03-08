/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    VIOGroup.h
 * @brief   Dynamic VI-SLAM symmetry group from Eq. (22)
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SOT3.h>
#include <gtsam_unstable/dllexport.h>

#include <string>
#include <vector>

namespace gtsam {

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

  /// @name Constructors
  /// @{

  VIOGroup();
  explicit VIOGroup(size_t n);
  explicit VIOGroup(const std::vector<int>& ids);
  VIOGroup(const SE23& A, const Vector6& beta, const Pose3& B,
           const LandmarkGroup& Q, const std::vector<int>& ids = {});

  /// @}
  /// @name Factories
  /// @{

  static VIOGroup Identity();
  static VIOGroup Identity(size_t n);
  static VIOGroup Identity(const std::vector<int>& ids);

  /// @}
  /// @name Accessors
  /// @{

  const SE23& A() const;
  SE23& A();

  const Vector6& beta() const;
  Vector6& beta();

  const Pose3& B() const;
  Pose3& B();

  const LandmarkGroup& Q() const;
  LandmarkGroup& Q();

  const std::vector<int>& ids() const;

  size_t n() const;
  size_t dim() const;

  /// @}
  /// @name Group Operations
  /// @{

  VIOGroup operator*(const VIOGroup& other) const;
  VIOGroup inverse() const;

  /// @}
  /// @name Lie Group Interface
  /// @{

  VIOGroup compose(const VIOGroup& other, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  VIOGroup between(const VIOGroup& other, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  VIOGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  TangentVector localCoordinates(const VIOGroup& other, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const;

  static VIOGroup Expmap(const TangentVector& v, ChartJacobian H = {});
  static TangentVector Logmap(const VIOGroup& g, ChartJacobian H = {});

  Jacobian AdjointMap() const;

  struct ChartAtOrigin {
    static VIOGroup Retract(const TangentVector& v, ChartJacobian H = {});
    static TangentVector Local(const VIOGroup& g, ChartJacobian H = {});
  };

  using LieGroup<VIOGroup, Eigen::Dynamic>::inverse;

  /// @}
  /// @name Testable
  /// @{

  void print(const std::string& s = "") const;
  bool equals(const VIOGroup& other, double tol = 1e-9) const;

  /// @}

 private:
  VIOGroup(const VIOGroupCore& core, std::vector<int> ids);

  void validate() const;
  void checkCompatible(const VIOGroup& other, const char* context) const;
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
