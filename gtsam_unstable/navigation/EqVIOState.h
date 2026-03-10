/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOState.h
 * @brief   Dynamic VIO manifold state for EqF foundations
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_unstable/dllexport.h>

#include <string>
#include <vector>

namespace gtsam {

struct GTSAM_UNSTABLE_EXPORT Landmark {
  static constexpr int CompDim = 3;

  Point3 p = Point3::Zero();
  int id = -1;

  void print(const std::string& s = "") const;
  bool equals(const Landmark& other, double tol = 1e-9) const;
};

struct GTSAM_UNSTABLE_EXPORT VIOSensorState {
  static constexpr int CompDim = 21;

  Vector6 inputBias = Vector6::Zero();
  Pose3 pose = Pose3::Identity();
  Vector3 velocity = Vector3::Zero();
  Pose3 cameraOffset = Pose3::Identity();

  /**
   * Unit gravity direction expressed in the body frame.
   * @return body-frame gravity direction.
   */
  Vector3 gravityDir() const;

  void print(const std::string& s = "") const;
  bool equals(const VIOSensorState& other, double tol = 1e-9) const;
};

/**
 * Dynamic VIO state manifold:
 *   xi = (sensor, landmarks_1..n), dim = 21 + 3n.
 */
class GTSAM_UNSTABLE_EXPORT VIOState {
 public:
  static constexpr int dimension = Eigen::Dynamic;

  using TangentVector = Vector;
  using Jacobian = Matrix;
  using ChartJacobian = OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

  VIOSensorState sensor;
  std::vector<Landmark> cameraLandmarks;

  /** Construct default-initialized state. */
  VIOState() = default;
  /**
   * Construct from explicit sensor and landmark blocks.
   * @param sensor_ sensor state block.
   * @param lms ordered landmark list.
   */
  VIOState(const VIOSensorState& sensor_, const std::vector<Landmark>& lms);

  /** Number of landmarks. */
  size_t n() const;
  int dim() const;
  std::vector<int> ids() const;

  /**
   * Retract in the state chart.
   * @param v tangent increment of size dim().
   * @param H1 optional derivative wrt this state.
   * @param H2 optional derivative wrt v.
   * @return updated state.
   */
  VIOState retract(const TangentVector& v, ChartJacobian H1 = {},
                   ChartJacobian H2 = {}) const;
  /**
   * Local coordinates in the state chart.
   * @param other target state with matching landmark layout.
   * @param H1 optional derivative wrt this state.
   * @param H2 optional derivative wrt other.
   * @return tangent vector from this to other.
   */
  TangentVector localCoordinates(const VIOState& other, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const;

  void print(const std::string& s = "") const;
  bool equals(const VIOState& other, double tol = 1e-9) const;
};

template <>
struct traits<VIOState> {
  static constexpr int dimension = Eigen::Dynamic;
  using TangentVector = Vector;
  using ManifoldType = VIOState;
  using structure_category = manifold_tag;

  static int GetDimension(const VIOState& xi) { return xi.dim(); }

  static VIOState Retract(const VIOState& xi, const TangentVector& v,
                          OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 =
                              {},
                          OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 =
                              {}) {
    return xi.retract(v, H1, H2);
  }

  static TangentVector Local(const VIOState& xi, const VIOState& other,
                             OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>
                                 H1 = {},
                             OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>
                                 H2 = {}) {
    return xi.localCoordinates(other, H1, H2);
  }

  static void Print(const VIOState& xi, const std::string& s = "") {
    xi.print(s);
  }

  static bool Equals(const VIOState& xi1, const VIOState& xi2,
                     double tol = 1e-9) {
    return xi1.equals(xi2, tol);
  }
};

template <>
struct traits<const VIOState> : traits<VIOState> {};

}  // namespace gtsam
