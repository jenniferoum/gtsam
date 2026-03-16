/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/// @file EqVIOCommon.h
/// @brief Common EqVIO math/data types for unstable navigation.
/// @author Rohan Bansal

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam_unstable/dllexport.h>

#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/navigation/ImuBias.h>

#include <map>
#include <memory>
#include <string>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gtsam {
namespace eqvio {

using SOT3 = ProductLieGroup<SO3, double>;

using VIOSE23 = ExtendedPose3<2>;
using VIOBias = imuBias::ConstantBias;
using VIOLandmarkGroup = PowerLieGroup<SOT3, Eigen::Dynamic>;
using VIOSensorCore = ProductLieGroup<VIOSE23, VIOBias>;
using VIOLandmarkCore = ProductLieGroup<Pose3, VIOLandmarkGroup>;
using VIOGroup = ProductLieGroup<VIOSensorCore, VIOLandmarkCore>;

/// Approximate gravitational acceleration magnitude in m/s^2.
constexpr double GRAVITY_CONSTANT = 9.80665;

/// IMU reading bundle used by EqVIO.
struct GTSAM_UNSTABLE_EXPORT IMUVelocity {
  static constexpr int CompDim = 12;
  using Vector12 = Eigen::Matrix<double, 12, 1>;

  double stamp = -1.0;
  Vector3 gyr = Vector3::Zero();
  Vector3 acc = Vector3::Zero();
  Vector3 gyrBiasVel = Vector3::Zero();
  Vector3 accBiasVel = Vector3::Zero();

  /// Return a zero-initialized reading with invalid timestamp.
  static IMUVelocity Zero() { return IMUVelocity(); }

  IMUVelocity() = default;

  /// Construct from stacked [gyr, acc].
  explicit IMUVelocity(const Vector6& vec) {
    gyr = vec.head<3>();
    acc = vec.tail<3>();
  }

  /// Construct from stacked [gyr, acc, gyrBiasVel, accBiasVel].
  explicit IMUVelocity(const Vector12& vec) {
    gyr = vec.segment<3>(0);
    acc = vec.segment<3>(3);
    gyrBiasVel = vec.segment<3>(6);
    accBiasVel = vec.segment<3>(9);
  }

  /// Component-wise addition.
  IMUVelocity operator+(const IMUVelocity& other) const {
    IMUVelocity out;
    out.stamp = stamp >= 0.0 ? stamp : other.stamp;
    out.gyr = gyr + other.gyr;
    out.acc = acc + other.acc;
    out.gyrBiasVel = gyrBiasVel + other.gyrBiasVel;
    out.accBiasVel = accBiasVel + other.accBiasVel;
    return out;
  }

  /// Subtract stacked [gyr, acc].
  IMUVelocity operator-(const Vector6& vec) const {
    IMUVelocity out(*this);
    out.gyr -= vec.head<3>();
    out.acc -= vec.tail<3>();
    return out;
  }

  /// Subtract stacked [gyr, acc, gyrBiasVel, accBiasVel].
  IMUVelocity operator-(const Vector12& vec) const {
    IMUVelocity out(*this);
    out.gyr -= vec.segment<3>(0);
    out.acc -= vec.segment<3>(3);
    out.gyrBiasVel -= vec.segment<3>(6);
    out.accBiasVel -= vec.segment<3>(9);
    return out;
  }

  /// Subtract a ConstantBias from [gyr, acc].
  IMUVelocity operator-(const VIOBias& bias) const {
    IMUVelocity out(*this);
    out.gyr -= bias.gyroscope();
    out.acc -= bias.accelerometer();
    return out;
  }

  /// Scale all components.
  IMUVelocity operator*(double c) const {
    IMUVelocity out(*this);
    out.gyr *= c;
    out.acc *= c;
    out.gyrBiasVel *= c;
    out.accBiasVel *= c;
    return out;
  }
};

/// EqVIO camera built on top of GTSAM PinholeCamera<Cal3_S2>.
class GTSAM_UNSTABLE_EXPORT VIOCameraModel
    : public PinholeCamera<Cal3_S2> {
 public:
  using Base = PinholeCamera<Cal3_S2>;

  VIOCameraModel() : Base(Pose3::Identity(), Cal3_S2()) {}
  explicit VIOCameraModel(const Cal3_S2& K) : Base(Pose3::Identity(), K) {}
  VIOCameraModel(const Pose3& pose, const Cal3_S2& K) : Base(pose, K) {}
  virtual ~VIOCameraModel() = default;

  /// Project a camera-frame 3D point to image coordinates.
  virtual Point2 projectPoint(const Point3& p) const {
    if (std::abs(p.z()) < 1e-12) {
      throw std::invalid_argument("VIOCameraModel::projectPoint: z is near zero");
    }
    const Point2 pn(p.x() / p.z(), p.y() / p.z());
    return this->calibration().uncalibrate(pn);
  }

  /// Convert image coordinates to an undistorted 3D bearing-like vector.
  virtual Vector3 undistortPoint(const Point2& y) const {
    const Point2 p = this->calibration().calibrate(y);
    return Vector3(p.x(), p.y(), 1.0);
  }

  /// Projection Jacobian with respect to the input 3D vector.
  virtual Matrix23 projectionJacobian(const Vector3& y) const {
    if (std::abs(y.z()) < 1e-12) {
      throw std::invalid_argument("VIOCameraModel::projectionJacobian: z is near zero");
    }

    const double invz = 1.0 / y.z();
    const double invz2 = invz * invz;
    const double fx = this->calibration().fx();
    const double fy = this->calibration().fy();
    const double s = this->calibration().skew();

    Matrix23 J;
    J << fx * invz, s * invz, -(fx * y.x() + s * y.y()) * invz2, 0.0,
        fy * invz, -fy * y.y() * invz2;
    return J;
  }
};

/// Vision measurement keyed by landmark id.
struct GTSAM_UNSTABLE_EXPORT VisionMeasurement {
  static constexpr int dimension = Eigen::Dynamic;

  using TangentVector = Vector;
  using Jacobian = Matrix;
  using ChartJacobian = OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;

  double stamp = -1.0;
  std::map<int, Point2> camCoordinates;
  std::shared_ptr<const VIOCameraModel> camera;

  /// Number of landmark measurements.
  size_t n() const { return camCoordinates.size(); }

  /// Ordered landmark ids matching map iteration order.
  std::vector<int> getIds() const {
    std::vector<int> ids;
    ids.reserve(camCoordinates.size());
    for (const auto& [id, _] : camCoordinates) {
      (void)_;
      ids.push_back(id);
    }
    return ids;
  }

  int dim() const { return static_cast<int>(2 * camCoordinates.size()); }

  operator Vector() const {
    Vector v = Vector::Zero(dim());
    int i = 0;
    for (const auto& [_, y] : camCoordinates) {
      (void)_;
      v.segment<2>(2 * i) = y;
      ++i;
    }
    return v;
  }

  /// Retract in Euclidean chart.
  VisionMeasurement retract(const TangentVector& v, ChartJacobian H1 = {},
                            ChartJacobian H2 = {}) const {
    if (v.size() != dim()) {
      throw std::invalid_argument(
          "VisionMeasurement::retract: unexpected tangent dimension");
    }

    VisionMeasurement out(*this);
    int i = 0;
    for (auto& [_, y] : out.camCoordinates) {
      (void)_;
      y += v.segment<2>(2 * i);
      ++i;
    }

    if (H1) *H1 = Matrix::Identity(dim(), dim());
    if (H2) *H2 = Matrix::Identity(dim(), dim());
    return out;
  }

  /// Local coordinates in Euclidean chart.
  TangentVector localCoordinates(const VisionMeasurement& other,
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    TangentVector v = Vector::Zero(dim());
    int i = 0;
    auto it1 = camCoordinates.begin();
    auto it2 = other.camCoordinates.begin();
    for (; it1 != camCoordinates.end(); ++it1, ++it2) {
      v.segment<2>(2 * i) = it2->second - it1->second;
      ++i;
    }

    if (H1) *H1 = -Matrix::Identity(dim(), dim());
    if (H2) *H2 = Matrix::Identity(dim(), dim());
    return v;
  }

  void print(const std::string& s = "") const {
    if (!s.empty()) std::cout << s << std::endl;
    std::cout << "VisionMeasurement(n=" << n() << ")" << std::endl;
    for (const auto& [id, y] : camCoordinates) {
      std::cout << "  id " << id << ": " << y.transpose() << std::endl;
    }
  }
  bool equals(const VisionMeasurement& other, double tol = 1e-9) const {
    if (stamp != other.stamp) return false;
    if (camCoordinates.size() != other.camCoordinates.size()) return false;

    auto it1 = camCoordinates.begin();
    auto it2 = other.camCoordinates.begin();
    for (; it1 != camCoordinates.end(); ++it1, ++it2) {
      if (it1->first != it2->first) return false;
      if (!equal_with_abs_tol(it1->second, it2->second, tol)) return false;
    }
    return true;
  }
};

inline VisionMeasurement operator-(const VisionMeasurement& y1,
                                   const VisionMeasurement& y2) {
  VisionMeasurement out;
  out.stamp = y1.stamp;
  out.camera = y1.camera ? y1.camera : y2.camera;

  auto it1 = y1.camCoordinates.begin();
  auto it2 = y2.camCoordinates.begin();
  for (; it1 != y1.camCoordinates.end(); ++it1, ++it2) {
    out.camCoordinates[it1->first] = it1->second - it2->second;
  }
  return out;
}

inline VisionMeasurement operator+(const VisionMeasurement& y,
                                   const Vector& eta) {
  return y.retract(eta);
}

/// Readable accessors for the composed ProductLieGroup VIOGroup.
inline const VIOSE23& groupA(const VIOGroup& X) { return X.first.first; }

inline const VIOBias& groupBeta(const VIOGroup& X) { return X.first.second; }

inline const Pose3& groupB(const VIOGroup& X) { return X.second.first; }

inline const VIOLandmarkGroup& groupQ(const VIOGroup& X) {
  return X.second.second;
}

inline size_t groupN(const VIOGroup& X) { return groupQ(X).size(); }
inline size_t groupDim(const VIOGroup& X) { return 21 + 4 * groupN(X); }

inline VIOGroup makeVIOGroup(const VIOSE23& A, const VIOBias& beta,
                             const Pose3& B, const VIOLandmarkGroup& Q) {
  return VIOGroup(VIOSensorCore(A, beta), VIOLandmarkCore(B, Q));
}

inline VIOGroup makeVIOGroupIdentity(size_t n = 0) {
  return makeVIOGroup(VIOSE23::Identity(), VIOBias::Identity(), Pose3::Identity(),
                      VIOLandmarkGroup(n));
}

}  // namespace eqvio

template <>
struct traits<eqvio::VisionMeasurement> {
  static constexpr int dimension = Eigen::Dynamic;
  using TangentVector = Vector;
  using ManifoldType = eqvio::VisionMeasurement;
  using structure_category = manifold_tag;

  static int GetDimension(const eqvio::VisionMeasurement& y) { return y.dim(); }

  static eqvio::VisionMeasurement Retract(
      const eqvio::VisionMeasurement& y, const TangentVector& v,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 = {}) {
    return y.retract(v, H1, H2);
  }

  static TangentVector Local(
      const eqvio::VisionMeasurement& y, const eqvio::VisionMeasurement& other,
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 = {},
      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 = {}) {
    return y.localCoordinates(other, H1, H2);
  }

  static void Print(const eqvio::VisionMeasurement& y,
                    const std::string& s = "") {
    y.print(s);
  }

  static bool Equals(const eqvio::VisionMeasurement& y1,
                     const eqvio::VisionMeasurement& y2, double tol = 1e-9) {
    return y1.equals(y2, tol);
  }
};

template <>
struct traits<const eqvio::VisionMeasurement>
    : traits<eqvio::VisionMeasurement> {};

}  // namespace gtsam
