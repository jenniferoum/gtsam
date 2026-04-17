/**
 *  @file   CarrierPhaseFactor.cpp
 *  @brief  Implementation file for GNSS Carrier Phase factors
 *  @date   March 23, 2026
 **/

#include "CarrierPhaseFactor.h"

#include <limits>

namespace gtsam {

using gnss::C_LIGHT;

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
  gtsam::print(measurement_, "carrier phase (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
}

//***************************************************************************
bool CarrierPhaseFactor::equals(const NonlinearFactor& expected,
                                double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         traits<double>::Equals(measurement_, e->measurement_, tol) &&
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
      range + C_LIGHT * (receiverClockBias - satClkBias_) + ambiguity;
  const double error = rho - measurement_;

  if (HreceiverPos) {
    if (range < std::numeric_limits<double>::epsilon()) {
      *HreceiverPos = Matrix13::Zero();
    } else {
      *HreceiverPos = (position_difference / range).transpose();
    }
  }

  if (HreceiverClockBias) {
    *HreceiverClockBias = I_1x1 * C_LIGHT;
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
  gtsam::print(measurement_, "carrier phase (m): ");
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
  if (!traits<double>::Equals(measurement_, e->measurement_, tol))
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
      range + C_LIGHT * (receiverClockBias - satClkBias_) + ambiguity;
  const double error = rho - measurement_;

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
    *HreceiverClockBias = I_1x1 * C_LIGHT;
  }

  if (Hambiguity) {
    *Hambiguity = I_1x1;
  }

  return Vector1(error);
}

//***************************************************************************
DoubleDifferenceCarrierPhaseFactor::DoubleDifferenceCarrierPhaseFactor(
    const Key positionKey, const Key ambRefKey, const Key ambTargetKey,
    const double cpRovRef, const double cpBaseRef, const double cpRovTarget,
    const double cpBaseTarget, const Point3& satRefRov,
    const Point3& satTargetRov, const Point3& satRefBase,
    const Point3& satTargetBase, const Point3& basePos, const double lam,
    const SharedNoiseModel& model)
    : Base(model, positionKey, ambRefKey, ambTargetKey),
      cpRovRef_(cpRovRef),
      cpBaseRef_(cpBaseRef),
      cpRovTarget_(cpRovTarget),
      cpBaseTarget_(cpBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos),
      lam_(lam) {}

//***************************************************************************
void DoubleDifferenceCarrierPhaseFactor::print(const std::string& s,
                                 const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? "" : s + " ") << "DoubleDifferenceCarrierPhaseFactor\n";
  std::cout << "  lam: " << lam_ << "\n";
  Base::print("", keyFormatter);
}

//***************************************************************************
bool DoubleDifferenceCarrierPhaseFactor::equals(const NonlinearFactor& expected,
                                  double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         std::abs(cpRovRef_ - e->cpRovRef_) < tol &&
         std::abs(cpBaseRef_ - e->cpBaseRef_) < tol &&
         std::abs(cpRovTarget_ - e->cpRovTarget_) < tol &&
         std::abs(cpBaseTarget_ - e->cpBaseTarget_) < tol &&
         std::abs(lam_ - e->lam_) < tol &&
         traits<Point3>::Equals(satRefRov_, e->satRefRov_, tol) &&
         traits<Point3>::Equals(satTargetRov_, e->satTargetRov_, tol) &&
         traits<Point3>::Equals(satRefBase_, e->satRefBase_, tol) &&
         traits<Point3>::Equals(satTargetBase_, e->satTargetBase_, tol) &&
         traits<Point3>::Equals(basePos_, e->basePos_, tol);
}

//***************************************************************************
Vector DoubleDifferenceCarrierPhaseFactor::evaluateError(
    const Point3& pos, const double& ambRef, const double& ambTarget,
    OptionalMatrixType Hpos, OptionalMatrixType HambRef,
    OptionalMatrixType HambTarget) const {
  const double ddObs =
      (cpRovRef_ - cpBaseRef_) - (cpRovTarget_ - cpBaseTarget_);

  Point3 eRef, eTarget, dummy;
  Matrix13 H_rRovRef, H_rRovTarget;
  const double rRovRef = gnss::geodist(satRefRov_, pos, eRef, H_rRovRef);
  const double rRovTarget =
      gnss::geodist(satTargetRov_, pos, eTarget, H_rRovTarget);
  const double rBaseRef = gnss::geodist(satRefBase_, basePos_, dummy);
  const double rBaseTarget = gnss::geodist(satTargetBase_, basePos_, dummy);

  const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
  const double error = ddModel + lam_ * (ambRef - ambTarget) - ddObs;

  if (Hpos) *Hpos = H_rRovRef - H_rRovTarget;
  if (HambRef) *HambRef = (Matrix(1, 1) << lam_).finished();
  if (HambTarget) *HambTarget = (Matrix(1, 1) << -lam_).finished();

  return Vector1(error);
}

//***************************************************************************
DoubleDifferenceCarrierPhaseFactorArm::DoubleDifferenceCarrierPhaseFactorArm(
    const Key poseKey, const Key ambRefKey, const Key ambTargetKey,
    const double cpRovRef, const double cpBaseRef, const double cpRovTarget,
    const double cpBaseTarget, const Point3& satRefRov,
    const Point3& satTargetRov, const Point3& satRefBase,
    const Point3& satTargetBase, const Point3& basePos, const double lam,
    const Point3& leverArm, const SharedNoiseModel& model)
    : Base(model, poseKey, ambRefKey, ambTargetKey),
      cpRovRef_(cpRovRef),
      cpBaseRef_(cpBaseRef),
      cpRovTarget_(cpRovTarget),
      cpBaseTarget_(cpBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos),
      lam_(lam),
      bL_(leverArm) {}

//***************************************************************************
DoubleDifferenceCarrierPhaseFactorArm::DoubleDifferenceCarrierPhaseFactorArm(
    const Key poseKey, const Key ambRefKey, const Key ambTargetKey,
    const double cpRovRef, const double cpBaseRef, const double cpRovTarget,
    const double cpBaseTarget, const Point3& satRefRov,
    const Point3& satTargetRov, const Point3& satRefBase,
    const Point3& satTargetBase, const Point3& basePos, const double lam,
    const Point3& leverArm, const Pose3& ecef_T_nav,
    const SharedNoiseModel& model)
    : Base(model, poseKey, ambRefKey, ambTargetKey),
      cpRovRef_(cpRovRef),
      cpBaseRef_(cpBaseRef),
      cpRovTarget_(cpRovTarget),
      cpBaseTarget_(cpBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos),
      lam_(lam),
      bL_(leverArm),
      ecef_T_nav_(ecef_T_nav) {}

//***************************************************************************
void DoubleDifferenceCarrierPhaseFactorArm::print(const std::string& s,
                                    const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? "" : s + " ") << "DoubleDifferenceCarrierPhaseFactorArm\n";
  std::cout << "  lam: " << lam_ << "\n";
  Base::print("", keyFormatter);
}

//***************************************************************************
bool DoubleDifferenceCarrierPhaseFactorArm::equals(const NonlinearFactor& expected,
                                     double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr || !Base::equals(*e, tol)) return false;
  if (std::abs(cpRovRef_ - e->cpRovRef_) >= tol) return false;
  if (std::abs(cpBaseRef_ - e->cpBaseRef_) >= tol) return false;
  if (std::abs(cpRovTarget_ - e->cpRovTarget_) >= tol) return false;
  if (std::abs(cpBaseTarget_ - e->cpBaseTarget_) >= tol) return false;
  if (std::abs(lam_ - e->lam_) >= tol) return false;
  if (!traits<Point3>::Equals(satRefRov_, e->satRefRov_, tol)) return false;
  if (!traits<Point3>::Equals(satTargetRov_, e->satTargetRov_, tol))
    return false;
  if (!traits<Point3>::Equals(satRefBase_, e->satRefBase_, tol)) return false;
  if (!traits<Point3>::Equals(satTargetBase_, e->satTargetBase_, tol))
    return false;
  if (!traits<Point3>::Equals(basePos_, e->basePos_, tol)) return false;
  if (!traits<Point3>::Equals(bL_, e->bL_, tol)) return false;
  if (ecef_T_nav_.has_value() != e->ecef_T_nav_.has_value()) return false;
  if (ecef_T_nav_ && !ecef_T_nav_->equals(*e->ecef_T_nav_, tol)) return false;
  return true;
}

//***************************************************************************
Vector DoubleDifferenceCarrierPhaseFactorArm::evaluateError(
    const Pose3& pose, const double& ambRef, const double& ambTarget,
    OptionalMatrixType H_pose, OptionalMatrixType HambRef,
    OptionalMatrixType HambTarget) const {
  Matrix66 H_compose;
  const bool has_nav = ecef_T_nav_.has_value();
  const Pose3 ecef_T_body = has_nav
      ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
      : pose;

  const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
  const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

  const double ddObs =
      (cpRovRef_ - cpBaseRef_) - (cpRovTarget_ - cpBaseTarget_);

  Point3 eRef, eTarget, dummy;
  Matrix13 H_rRovRef, H_rRovTarget;
  const double rRovRef = gnss::geodist(satRefRov_, antennaPos, eRef, H_rRovRef);
  const double rRovTarget =
      gnss::geodist(satTargetRov_, antennaPos, eTarget, H_rRovTarget);
  const double rBaseRef = gnss::geodist(satRefBase_, basePos_, dummy);
  const double rBaseTarget = gnss::geodist(satTargetBase_, basePos_, dummy);

  const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
  const double error = ddModel + lam_ * (ambRef - ambTarget) - ddObs;

  if (H_pose) {
    H_pose->resize(1, 6);
    const Matrix13 dd_u = H_rRovRef - H_rRovTarget;
    Matrix16 H_ecef;
    H_ecef.block<1, 3>(0, 0) = dd_u * (-ecef_R_body * skewSymmetric(bL_));
    H_ecef.block<1, 3>(0, 3) = dd_u * ecef_R_body;
    *H_pose = has_nav ? H_ecef * H_compose : H_ecef;
  }
  if (HambRef) *HambRef = (Matrix(1, 1) << lam_).finished();
  if (HambTarget) *HambTarget = (Matrix(1, 1) << -lam_).finished();

  return Vector1(error);
}

}  // namespace gtsam
