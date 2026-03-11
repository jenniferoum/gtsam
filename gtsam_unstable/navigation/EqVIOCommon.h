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

/** Approximate gravitational acceleration magnitude in m/s^2. */
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

  /** Return an IMUVelocity with all components zero and an invalid timestamp. */
  static IMUVelocity Zero();

  IMUVelocity() = default;

  /** Construct from a size-6 stacked vector of angular and linear components. */
  explicit IMUVelocity(const Vector6& vec);

  /**
   * Construct from a size-12 stacked vector containing IMU and bias-velocity
   * components.
   */
  explicit IMUVelocity(const Vector12& vec);

  /** Add two IMUVelocity instances component-wise. */
  IMUVelocity operator+(const IMUVelocity& other) const;

  /**
   * Subtract a size-6 stacked vector from the IMU angular and linear
   * components.
   */
  IMUVelocity operator-(const Vector6& vec) const;

  /**
   * Subtract a size-12 stacked vector from all IMU and bias-velocity
   * components.
   */
  IMUVelocity operator-(const Vector12& vec) const;

  /** Scale all components of the IMUVelocity by a scalar factor. */
  IMUVelocity operator*(double c) const;
};

/**
 * Abstract camera model interface required by VIO output/action math.
 */
class GTSAM_UNSTABLE_EXPORT VIOCameraModel {
 public:
  virtual ~VIOCameraModel() = default;

  /**
   * Project a camera-frame 3D point to image coordinates.
   * @param p camera-frame point.
   * @return image measurement.
   */
  virtual Point2 projectPoint(const Point3& p) const = 0;
  /**
   * Convert image coordinates to an undistorted 3D bearing-like vector.
   * @param y image measurement.
   * @return undistorted vector (typically homogeneous bearing).
   */
  virtual Vector3 undistortPoint(const Point2& y) const = 0;
  /**
   * Jacobian of projection with respect to the input 3D vector.
   * @param y input 3D vector.
   * @return 2x3 projection Jacobian.
   */
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

  /** Number of landmark measurements. */
  size_t n() const;

  /** Ordered landmark ids matching camCoordinates map iteration order. */
  std::vector<int> getIds() const;

  int dim() const;

  operator Vector() const;

  /**
   * Retract in Euclidean chart.
   * @param v tangent increment with size dim().
   * @param H1 optional derivative wrt this measurement.
   * @param H2 optional derivative wrt v.
   * @return updated measurement.
   */
  VisionMeasurement retract(const TangentVector& v, ChartJacobian H1 = {},
                            ChartJacobian H2 = {}) const;
  /**
   * Local coordinates in Euclidean chart.
   * @param other target measurement with aligned ids.
   * @param H1 optional derivative wrt this measurement.
   * @param H2 optional derivative wrt other.
   * @return tangent vector from this to other.
   */
  TangentVector localCoordinates(const VisionMeasurement& other,
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const;

  void print(const std::string& s = "") const;
  bool equals(const VisionMeasurement& other, double tol = 1e-9) const;
};

GTSAM_UNSTABLE_EXPORT VisionMeasurement operator-(
    const VisionMeasurement& y1, const VisionMeasurement& y2);

GTSAM_UNSTABLE_EXPORT VisionMeasurement operator+(
    const VisionMeasurement& y, const Vector& eta);

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
