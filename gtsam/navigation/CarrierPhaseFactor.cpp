/**
 *  @file   CarrierPhaseFactor.cpp
 *  @brief  Implementation file for GNSS Carrier Phase factors
 *  @date   March 23, 2026
 **/

#include "CarrierPhaseFactor.h"

#include <limits>

namespace {

/// Speed of light in a vacuum (m/s):
constexpr double CLIGHT = 299792458.0;

}  // namespace

namespace gtsam {

//***************************************************************************
CarrierPhaseFactor::CarrierPhaseFactor(
    const Key receiverPositionKey, const Key receiverClockBiasKey,
    const Key ambiguityKey, const double measuredCarrierPhase,
    const Point3& satellitePosition, const double satelliteClockBias,
    const SharedNoiseModel& model)
    : Base(model, receiverPositionKey, receiverClockBiasKey, ambiguityKey),
      CarrierPhaseBase{measuredCarrierPhase, satellitePosition,
                       satelliteClockBias} {}

//***************************************************************************
void CarrierPhaseFactor::print(const std::string& s,
                               const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(carrierPhase_, "carrier phase (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
}

//***************************************************************************
bool CarrierPhaseFactor::equals(const NonlinearFactor& expected,
                                double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         traits<double>::Equals(carrierPhase_, e->carrierPhase_, tol) &&
         traits<Point3>::Equals(satPos_, e->satPos_, tol) &&
         traits<double>::Equals(satClkBias_, e->satClkBias_, tol);
}

//***************************************************************************
Vector CarrierPhaseFactor::evaluateError(
    const Point3& receiverPosition, const double& receiverClockBias,
    const double& ambiguity, OptionalMatrixType HreceiverPos,
    OptionalMatrixType HreceiverClockBias,
    OptionalMatrixType Hambiguity) const {
  // error = range + c*(dt_u - dt_s) + ambiguity - measurement
  const Vector3 position_difference = receiverPosition - satPos_;
  const double range = position_difference.norm();
  const double rho =
      range + CLIGHT * (receiverClockBias - satClkBias_) + ambiguity;
  const double error = rho - carrierPhase_;

  if (HreceiverPos) {
    if (range < std::numeric_limits<double>::epsilon()) {
      *HreceiverPos = Matrix13::Zero();
    } else {
      *HreceiverPos = (position_difference / range).transpose();
    }
  }

  if (HreceiverClockBias) {
    *HreceiverClockBias = I_1x1 * CLIGHT;
  }

  if (Hambiguity) {
    *Hambiguity = I_1x1;
  }

  return Vector1(error);
}

//***************************************************************************
CarrierPhaseFactorArm::CarrierPhaseFactorArm(
    const Key poseKey, const Key receiverClockBiasKey, const Key ambiguityKey,
    const double measuredCarrierPhase, const Point3& satellitePosition,
    const Point3& leverArm, const double satelliteClockBias,
    const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey, ambiguityKey),
      CarrierPhaseBase{measuredCarrierPhase, satellitePosition,
                       satelliteClockBias},
      bL_(leverArm) {}

//***************************************************************************
CarrierPhaseFactorArm::CarrierPhaseFactorArm(
    const Key poseKey, const Key receiverClockBiasKey, const Key ambiguityKey,
    const double measuredCarrierPhase, const Point3& satellitePosition,
    const Point3& leverArm, const Pose3& ecef_T_nav,
    const double satelliteClockBias, const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey, ambiguityKey),
      CarrierPhaseBase{measuredCarrierPhase, satellitePosition,
                       satelliteClockBias},
      bL_(leverArm),
      ecef_T_nav_(ecef_T_nav) {}

//***************************************************************************
void CarrierPhaseFactorArm::print(const std::string& s,
                                  const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(carrierPhase_, "carrier phase (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
  gtsam::print(Vector(bL_), "lever arm (body frame meters): ");
  if (ecef_T_nav_) {
    ecef_T_nav_->print("ecef_T_nav:\n");
  }
}

//***************************************************************************
bool CarrierPhaseFactorArm::equals(const NonlinearFactor& expected,
                                   double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr || !Base::equals(*e, tol)) return false;
  if (!traits<double>::Equals(carrierPhase_, e->carrierPhase_, tol))
    return false;
  if (!traits<Point3>::Equals(satPos_, e->satPos_, tol)) return false;
  if (!traits<double>::Equals(satClkBias_, e->satClkBias_, tol)) return false;
  if (!traits<Point3>::Equals(bL_, e->bL_, tol)) return false;
  if (ecef_T_nav_.has_value() != e->ecef_T_nav_.has_value()) return false;
  if (ecef_T_nav_ && !ecef_T_nav_->equals(*e->ecef_T_nav_, tol)) return false;
  return true;
}

//***************************************************************************
Vector CarrierPhaseFactorArm::evaluateError(
    const Pose3& pose, const double& receiverClockBias,
    const double& ambiguity, OptionalMatrixType H_pose,
    OptionalMatrixType HreceiverClockBias,
    OptionalMatrixType Hambiguity) const {
  // Convert from local nav frame to ECEF if ecef_T_nav is provided:
  Matrix66 H_compose;
  const bool has_nav = ecef_T_nav_.has_value();
  const Pose3 ecef_T_body = has_nav
      ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
      : pose;

  // Compute antenna position in the ECEF frame:
  const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
  const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

  // error = range + c*(dt_u - dt_s) + ambiguity - measurement
  const Vector3 position_difference = antennaPos - satPos_;
  const double range = position_difference.norm();
  const double rho =
      range + CLIGHT * (receiverClockBias - satClkBias_) + ambiguity;
  const double error = rho - carrierPhase_;

  // Compute associated derivatives:
  if (H_pose) {
    H_pose->resize(1, 6);
    if (range < std::numeric_limits<double>::epsilon()) {
      H_pose->setZero();
    } else {
      const Matrix u = (position_difference / range).transpose();  // 1x3
      Matrix16 H_ecef;
      H_ecef.block<1, 3>(0, 0) =
          u * (-ecef_R_body * skewSymmetric(bL_));
      H_ecef.block<1, 3>(0, 3) = u * ecef_R_body;
      *H_pose = has_nav ? H_ecef * H_compose : H_ecef;
    }
  }

  if (HreceiverClockBias) {
    *HreceiverClockBias = I_1x1 * CLIGHT;
  }

  if (Hambiguity) {
    *Hambiguity = I_1x1;
  }

  return Vector1(error);
}

}  // namespace gtsam
