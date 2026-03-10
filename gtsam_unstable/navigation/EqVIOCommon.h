/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOCommon.h
 * @brief   Common VIO math/data types for unstable navigation
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam_unstable/dllexport.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gtsam {

/// Approximate gravitational acceleration magnitude in m/s^2.
constexpr double GRAVITY_CONSTANT = 9.80665;

/**
 * IMU reading bundle used by the VIO EqF machinery.
 * Contains gyro/accelerometer samples and optional bias velocities.
 */
struct GTSAM_UNSTABLE_EXPORT IMUVelocity {
  static constexpr int CompDim = 12;
  using Vector12 = Eigen::Matrix<double, 12, 1>;

  double stamp = -1.0;
  Vector3 gyr = Vector3::Zero();
  Vector3 acc = Vector3::Zero();
  Vector3 gyrBiasVel = Vector3::Zero();
  Vector3 accBiasVel = Vector3::Zero();

  static IMUVelocity Zero();

  IMUVelocity() = default;
  explicit IMUVelocity(const Vector6& vec);
  explicit IMUVelocity(const Vector12& vec);

  IMUVelocity operator+(const IMUVelocity& other) const;
  IMUVelocity operator-(const Vector6& vec) const;
  IMUVelocity operator-(const Vector12& vec) const;
  IMUVelocity operator*(double c) const;
};

/**
 * Abstract camera model interface required by VIO output/action math.
 */
class GTSAM_UNSTABLE_EXPORT VIOCameraModel {
 public:
  virtual ~VIOCameraModel() = default;

  virtual Point2 projectPoint(const Point3& p) const = 0;
  virtual Vector3 undistortPoint(const Point2& y) const = 0;
  virtual Matrix23 projectionJacobian(const Vector3& y) const = 0;
};

/**
 * Vision measurement keyed by landmark id.
 */
struct GTSAM_UNSTABLE_EXPORT VisionMeasurement {
  static constexpr int dimension = Eigen::Dynamic;

  using TangentVector = Vector;
  using Jacobian = Matrix;
  using ChartJacobian = OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

  double stamp = -1.0;
  std::map<int, Point2> camCoordinates;
  std::shared_ptr<const VIOCameraModel> camera;

  size_t n() const;
  int dim() const;
  std::vector<int> getIds() const;
  operator Vector() const;

  VisionMeasurement retract(const TangentVector& v, ChartJacobian H1 = {},
                            ChartJacobian H2 = {}) const;
  TangentVector localCoordinates(const VisionMeasurement& other,
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const;

  void print(const std::string& s = "") const;
  bool equals(const VisionMeasurement& other, double tol = 1e-9) const;
};

VisionMeasurement operator-(const VisionMeasurement& y1,
                            const VisionMeasurement& y2);
VisionMeasurement operator+(const VisionMeasurement& y, const Vector& eta);

template <>
struct traits<VisionMeasurement> {
  static constexpr int dimension = Eigen::Dynamic;
  using TangentVector = Vector;
  using ManifoldType = VisionMeasurement;
  using structure_category = manifold_tag;

  static int GetDimension(const VisionMeasurement& y) { return y.dim(); }

  static VisionMeasurement Retract(
      const VisionMeasurement& y, const TangentVector& v,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 = {}) {
    return y.retract(v, H1, H2);
  }

  static TangentVector Local(
      const VisionMeasurement& y, const VisionMeasurement& other,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 = {}) {
    return y.localCoordinates(other, H1, H2);
  }

  static void Print(const VisionMeasurement& y, const std::string& s = "") {
    y.print(s);
  }

  static bool Equals(const VisionMeasurement& y1, const VisionMeasurement& y2,
                     double tol = 1e-9) {
    return y1.equals(y2, tol);
  }
};

template <>
struct traits<const VisionMeasurement> : traits<VisionMeasurement> {};

}  // namespace gtsam
