/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOCommon.cpp
 * @brief   Common VIO math/data type implementations
 */

#include <gtsam_unstable/navigation/EqVIOCommon.h>

#include <iostream>
#include <stdexcept>

namespace gtsam {


IMUVelocity IMUVelocity::Zero() { return IMUVelocity(); }

IMUVelocity::IMUVelocity(const Vector6& vec) {
  gyr = vec.head<3>();
  acc = vec.tail<3>();
}

IMUVelocity::IMUVelocity(const Vector12& vec) {
  gyr = vec.segment<3>(0);
  acc = vec.segment<3>(3);
  gyrBiasVel = vec.segment<3>(6);
  accBiasVel = vec.segment<3>(9);
}

IMUVelocity IMUVelocity::operator+(const IMUVelocity& other) const {
  IMUVelocity out;
  out.stamp = stamp >= 0.0 ? stamp : other.stamp;
  out.gyr = gyr + other.gyr;
  out.acc = acc + other.acc;
  out.gyrBiasVel = gyrBiasVel + other.gyrBiasVel;
  out.accBiasVel = accBiasVel + other.accBiasVel;
  return out;
}

IMUVelocity IMUVelocity::operator-(const Vector6& vec) const {
  IMUVelocity out(*this);
  out.gyr -= vec.head<3>();
  out.acc -= vec.tail<3>();
  return out;
}

IMUVelocity IMUVelocity::operator-(const Vector12& vec) const {
  IMUVelocity out(*this);
  out.gyr -= vec.segment<3>(0);
  out.acc -= vec.segment<3>(3);
  out.gyrBiasVel -= vec.segment<3>(6);
  out.accBiasVel -= vec.segment<3>(9);
  return out;
}

IMUVelocity IMUVelocity::operator*(double c) const {
  IMUVelocity out(*this);
  out.gyr *= c;
  out.acc *= c;
  out.gyrBiasVel *= c;
  out.accBiasVel *= c;
  return out;
}

size_t VisionMeasurement::n() const { return camCoordinates.size(); }

int VisionMeasurement::dim() const {
  return static_cast<int>(2 * camCoordinates.size());
}

std::vector<int> VisionMeasurement::getIds() const {
  std::vector<int> ids;
  ids.reserve(camCoordinates.size());
  for (const auto& [id, _] : camCoordinates) {
    (void)_;
    ids.push_back(id);
  }
  return ids;
}

VisionMeasurement::operator Vector() const {
  Vector v = Vector::Zero(dim());
  int i = 0;
  for (const auto& [_, y] : camCoordinates) {
    (void)_;
    v.segment<2>(2 * i) = y;
    ++i;
  }
  return v;
}

VisionMeasurement VisionMeasurement::retract(const TangentVector& v,
                                             ChartJacobian H1,
                                             ChartJacobian H2) const {
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

VisionMeasurement::TangentVector VisionMeasurement::localCoordinates(
    const VisionMeasurement& other, ChartJacobian H1, ChartJacobian H2) const {

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

void VisionMeasurement::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << std::endl;
  std::cout << "VisionMeasurement(n=" << n() << ")" << std::endl;
  for (const auto& [id, y] : camCoordinates) {
    std::cout << "  id " << id << ": " << y.transpose() << std::endl;
  }
}

bool VisionMeasurement::equals(const VisionMeasurement& other, double tol) const {
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

VisionMeasurement operator-(const VisionMeasurement& y1,
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

VisionMeasurement operator+(const VisionMeasurement& y, const Vector& eta) {
  return y.retract(eta);
}

}  // namespace gtsam
