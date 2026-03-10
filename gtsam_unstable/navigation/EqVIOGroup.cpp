/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    EqVIOGroup.cpp
 * @brief   Dynamic VI-SLAM symmetry group from Eq. (22)
 */

#include <gtsam_unstable/navigation/EqVIOGroup.h>

#include <stdexcept>
#include <utility>

namespace gtsam {


VIOGroup::VIOGroup() : core_(), ids_() {}

VIOGroup::VIOGroup(size_t n)
    : VIOGroup(SE23::Identity(), Vector6::Zero(), Pose3::Identity(),
               LandmarkGroup(n)) {}

VIOGroup::VIOGroup(const std::vector<int>& ids)
    : VIOGroup(SE23::Identity(), Vector6::Zero(), Pose3::Identity(),
               LandmarkGroup(ids.size()), ids) {}

VIOGroup::VIOGroup(const SE23& A, const Vector6& beta, const Pose3& B,
                   const LandmarkGroup& Q, const std::vector<int>& ids)
    : core_(SensorCore(A, beta), LandmarkCore(B, Q)), ids_(ids) {
}

VIOGroup::VIOGroup(const VIOGroupCore& core, std::vector<int> ids)
    : core_(core), ids_(std::move(ids)) {
}

// helpers so we don't have to reference core objects in the public interface
VIOGroup VIOGroup::Identity() { return VIOGroup(); }

VIOGroup VIOGroup::Identity(size_t n) { return VIOGroup(n); }

VIOGroup VIOGroup::Identity(const std::vector<int>& ids) { return VIOGroup(ids); }

const VIOGroup::SE23& VIOGroup::A() const { return core_.first.first; }

VIOGroup::SE23& VIOGroup::A() { return core_.first.first; }

const Vector6& VIOGroup::beta() const { return core_.first.second; }

Vector6& VIOGroup::beta() { return core_.first.second; }

const Pose3& VIOGroup::B() const { return core_.second.first; }

Pose3& VIOGroup::B() { return core_.second.first; }

const VIOGroup::LandmarkGroup& VIOGroup::Q() const { return core_.second.second; }

VIOGroup::LandmarkGroup& VIOGroup::Q() { return core_.second.second; }

const std::vector<int>& VIOGroup::ids() const { return ids_; }

size_t VIOGroup::n() const { return Q().size(); }

size_t VIOGroup::dim() const { return 21 + 4 * n(); }

VIOGroup VIOGroup::operator*(const VIOGroup& other) const { return compose(other); }

VIOGroup VIOGroup::inverse() const { return VIOGroup(core_.inverse(), ids_); }

VIOGroup VIOGroup::compose(const VIOGroup& other, ChartJacobian H1,
                           ChartJacobian H2) const {
  return VIOGroup(core_.compose(other.core_, H1, H2), resolvedIds(other));
}

VIOGroup VIOGroup::between(const VIOGroup& other, ChartJacobian H1,
                           ChartJacobian H2) const {
  return VIOGroup(core_.between(other.core_, H1, H2), resolvedIds(other));
}

VIOGroup VIOGroup::retract(const TangentVector& v, ChartJacobian H1,
                           ChartJacobian H2) const {
  return VIOGroup(core_.retract(v, H1, H2), ids_);
}

VIOGroup::TangentVector VIOGroup::localCoordinates(const VIOGroup& other,
                                                   ChartJacobian H1,
                                                   ChartJacobian H2) const {
  return core_.localCoordinates(other.core_, H1, H2);
}

VIOGroup VIOGroup::Expmap(const TangentVector& v, ChartJacobian H) {
  inferLandmarkCount(v);
  return VIOGroup(VIOGroupCore::Expmap(v, H), {});
}

VIOGroup::TangentVector VIOGroup::Logmap(const VIOGroup& g, ChartJacobian H) {
  return VIOGroupCore::Logmap(g.core_, H);
}

VIOGroup::Jacobian VIOGroup::AdjointMap() const { return core_.AdjointMap(); }

VIOGroup VIOGroup::ChartAtOrigin::Retract(const TangentVector& v,
                                          ChartJacobian H) {
  return VIOGroup::Expmap(v, H);
}

VIOGroup::TangentVector VIOGroup::ChartAtOrigin::Local(const VIOGroup& g,
                                                       ChartJacobian H) {
  return VIOGroup::Logmap(g, H);
}

void VIOGroup::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << std::endl;
  A().print("  A");
  gtsam::print(Vector(beta()), "  beta");
  B().print("  B");
  Q().print("  Q");
  if (!ids_.empty()) {
    std::cout << "  ids:";
    for (const int id : ids_) std::cout << ' ' << id;
    std::cout << std::endl;
  }
}

bool VIOGroup::equals(const VIOGroup& other, double tol) const {
  return core_.equals(other.core_, tol);
}

std::vector<int> VIOGroup::resolvedIds(const VIOGroup& other) const {
  if (!ids_.empty()) return ids_;
  if (!other.ids_.empty()) return other.ids_;
  return {};
}

size_t VIOGroup::inferLandmarkCount(const TangentVector& v) {
  if (v.size() < 21 || (v.size() - 21) % 4 != 0) {
    throw std::invalid_argument(
        "VIOGroup::Expmap: tangent dimension must be 21 + 4*n");
  }
  return static_cast<size_t>((v.size() - 21) / 4);
}

}  // namespace gtsam
