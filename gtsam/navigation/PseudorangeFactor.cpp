/**
 *  @file   PseudorangeFactor.cpp
 *  @author Sammy Guo
 *  @brief  Implementation file for GNSS Pseudorange factor
 *  @date   January 18, 2026
 **/

#include "PseudorangeFactor.h"

#include <limits>

namespace gtsam {

using gnss::C_LIGHT;

//***************************************************************************
PseudorangeFactor::PseudorangeFactor(const Key receiverPositionKey,
                                     const Key receiverClockBiasKey,
                                     const double measuredPseudorange,
                                     const Point3& satellitePosition,
                                     const double satelliteClockBias,
                                     const SharedNoiseModel& model)
    : Base(model, receiverPositionKey, receiverClockBiasKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias} {}

//***************************************************************************
void PseudorangeFactor::print(const std::string& s,
                              const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(measurement_, "pseudorange (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
}

//***************************************************************************
bool PseudorangeFactor::equals(const NonlinearFactor& expected,
                               double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         traits<double>::Equals(measurement_, e->measurement_, tol) &&
         traits<Point3>::Equals(satPos_, e->satPos_, tol) &&
         traits<double>::Equals(satClkBias_, e->satClkBias_, tol);
}

//***************************************************************************
Vector PseudorangeFactor::evaluateError(
    const Point3& receiverPosition, const double& receiverClockBias,
    OptionalMatrixType HreceiverPos,
    OptionalMatrixType HreceiverClockBias) const {
  // Apply pseudorange equation: rho = range + c*[dt_u - dt^s]
  const Vector3 position_difference = receiverPosition - satPos_;
  const double range = position_difference.norm();
  const double rho = range + C_LIGHT * (receiverClockBias - satClkBias_);
  const double error = rho - measurement_;

  // Compute associated derivatives:
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

  return Vector1(error);
}

//***************************************************************************
PseudorangeFactorArm::PseudorangeFactorArm(
    const Key poseKey, const Key receiverClockBiasKey,
    const double measuredPseudorange, const Point3& satellitePosition,
    const Point3& leverArm, const double satelliteClockBias,
    const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias},
      bL_(leverArm) {}

//***************************************************************************
PseudorangeFactorArm::PseudorangeFactorArm(
    const Key poseKey, const Key receiverClockBiasKey,
    const double measuredPseudorange, const Point3& satellitePosition,
    const Point3& leverArm, const Pose3& ecef_T_nav,
    const double satelliteClockBias, const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias},
      bL_(leverArm),
      ecef_T_nav_(ecef_T_nav) {}

//***************************************************************************
void PseudorangeFactorArm::print(const std::string& s,
                                  const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(measurement_, "pseudorange (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
  gtsam::print(Vector(bL_), "lever arm (body frame meters): ");
  if (ecef_T_nav_) {
    ecef_T_nav_->print("ecef_T_nav: ");
  }
}

//***************************************************************************
bool PseudorangeFactorArm::equals(const NonlinearFactor& expected,
                                   double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr || !Base::equals(*e, tol)) return false;
  if (!traits<double>::Equals(measurement_, e->measurement_, tol)) return false;
  if (!traits<Point3>::Equals(satPos_, e->satPos_, tol)) return false;
  if (!traits<double>::Equals(satClkBias_, e->satClkBias_, tol)) return false;
  if (!traits<Point3>::Equals(bL_, e->bL_, tol)) return false;
  if (ecef_T_nav_.has_value() != e->ecef_T_nav_.has_value()) return false;
  if (ecef_T_nav_ && !ecef_T_nav_->equals(*e->ecef_T_nav_, tol)) return false;
  return true;
}

//***************************************************************************
Vector PseudorangeFactorArm::evaluateError(
    const Pose3& pose, const double& receiverClockBias,
    OptionalMatrixType H_pose,
    OptionalMatrixType HreceiverClockBias) const {
  // Convert from local nav frame to ECEF if ecef_T_nav is provided:
  Matrix66 H_compose;
  const bool has_nav = ecef_T_nav_.has_value();
  const Pose3 ecef_T_body = has_nav
      ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
      : pose;

  // Compute antenna position in the ECEF frame:
  const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
  const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

  // Apply pseudorange equation: rho = range + c*[dt_u - dt^s]
  const Vector3 position_difference = antennaPos - satPos_;
  const double range = position_difference.norm();
  const double rho = range + C_LIGHT * (receiverClockBias - satClkBias_);
  const double error = rho - measurement_;

  // Compute associated derivatives:
  if (H_pose) {
    H_pose->resize(1, 6);
    if (range < std::numeric_limits<double>::epsilon()) {
      H_pose->setZero();
    } else {
      // u = unit vector from satellite to antenna
      const Matrix u = (position_difference / range).transpose();  // 1x3
      Matrix16 H_ecef;
      H_ecef.block<1, 3>(0, 0) =
          u * (-ecef_R_body * skewSymmetric(bL_));
      H_ecef.block<1, 3>(0, 3) = u * ecef_R_body;
      // Chain rule: if ecef_T_nav is set, multiply by compose Jacobian
      *H_pose = has_nav ? H_ecef * H_compose : H_ecef;
    }
  }

  if (HreceiverClockBias) {
    *HreceiverClockBias = I_1x1 * C_LIGHT;
  }

  return Vector1(error);
}

//***************************************************************************
DifferentialPseudorangeFactor::DifferentialPseudorangeFactor(
    const Key receiverPositionKey, const Key receiverClockBiasKey,
    const Key differentialCorrectionKey, const double measuredPseudorange,
    const Point3& satellitePosition, const double satelliteClockBias,
    const SharedNoiseModel& model)
    : Base(model, receiverPositionKey, receiverClockBiasKey,
           differentialCorrectionKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias} {}

//***************************************************************************
void DifferentialPseudorangeFactor::print(
    const std::string& s, const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(measurement_, "pseudorange (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
}

//***************************************************************************
bool DifferentialPseudorangeFactor::equals(const NonlinearFactor& expected,
                                           double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         traits<double>::Equals(measurement_, e->measurement_, tol) &&
         traits<Point3>::Equals(satPos_, e->satPos_, tol) &&
         traits<double>::Equals(satClkBias_, e->satClkBias_, tol);
}

//***************************************************************************
Vector DifferentialPseudorangeFactor::evaluateError(
    const Point3& receiverPosition, const double& receiverClock_bias,
    const double& differentialCorrection, OptionalMatrixType HreceiverPos,
    OptionalMatrixType HreceiverClockBias,
    OptionalMatrixType HdifferentialCorrection) const {
  const Vector3 position_difference = receiverPosition - satPos_;
  const double range = position_difference.norm();
  const double rho = range + C_LIGHT * (receiverClock_bias - satClkBias_);
  const double error = rho - measurement_ - differentialCorrection;

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

  if (HdifferentialCorrection) {
    *HdifferentialCorrection = -I_1x1;
  }

  return Vector1(error);
}

//***************************************************************************
DifferentialPseudorangeFactorArm::DifferentialPseudorangeFactorArm(
    const Key poseKey, const Key receiverClockBiasKey,
    const Key differentialCorrectionKey, const double measuredPseudorange,
    const Point3& satellitePosition, const Point3& leverArm,
    const double satelliteClockBias, const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey, differentialCorrectionKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias},
      bL_(leverArm) {}

//***************************************************************************
DifferentialPseudorangeFactorArm::DifferentialPseudorangeFactorArm(
    const Key poseKey, const Key receiverClockBiasKey,
    const Key differentialCorrectionKey, const double measuredPseudorange,
    const Point3& satellitePosition, const Point3& leverArm,
    const Pose3& ecef_T_nav, const double satelliteClockBias,
    const SharedNoiseModel& model)
    : Base(model, poseKey, receiverClockBiasKey, differentialCorrectionKey),
      PseudorangeBase{measuredPseudorange, satellitePosition,
                      satelliteClockBias},
      bL_(leverArm),
      ecef_T_nav_(ecef_T_nav) {}

//***************************************************************************
void DifferentialPseudorangeFactorArm::print(
    const std::string& s, const KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  gtsam::print(measurement_, "pseudorange (m): ");
  gtsam::print(Vector(satPos_), "sat position (ECEF meters): ");
  gtsam::print(satClkBias_, "sat clock bias (s): ");
  gtsam::print(Vector(bL_), "lever arm (body frame meters): ");
  if (ecef_T_nav_) {
    ecef_T_nav_->print("ecef_T_nav: ");
  }
}

//***************************************************************************
bool DifferentialPseudorangeFactorArm::equals(
    const NonlinearFactor& expected, double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr || !Base::equals(*e, tol)) return false;
  if (!traits<double>::Equals(measurement_, e->measurement_, tol)) return false;
  if (!traits<Point3>::Equals(satPos_, e->satPos_, tol)) return false;
  if (!traits<double>::Equals(satClkBias_, e->satClkBias_, tol)) return false;
  if (!traits<Point3>::Equals(bL_, e->bL_, tol)) return false;
  if (ecef_T_nav_.has_value() != e->ecef_T_nav_.has_value()) return false;
  if (ecef_T_nav_ && !ecef_T_nav_->equals(*e->ecef_T_nav_, tol)) return false;
  return true;
}

//***************************************************************************
Vector DifferentialPseudorangeFactorArm::evaluateError(
    const Pose3& pose, const double& receiverClockBias,
    const double& differentialCorrection, OptionalMatrixType H_pose,
    OptionalMatrixType HreceiverClockBias,
    OptionalMatrixType HdifferentialCorrection) const {
  Matrix66 H_compose;
  const bool has_nav = ecef_T_nav_.has_value();
  const Pose3 ecef_T_body = has_nav
      ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
      : pose;

  const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
  const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

  const Vector3 position_difference = antennaPos - satPos_;
  const double range = position_difference.norm();
  const double rho = range + C_LIGHT * (receiverClockBias - satClkBias_);
  const double error = rho - measurement_ - differentialCorrection;

  if (H_pose) {
    H_pose->resize(1, 6);
    if (range < std::numeric_limits<double>::epsilon()) {
      H_pose->setZero();
    } else {
      const Matrix u = (position_difference / range).transpose();
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

  if (HdifferentialCorrection) {
    *HdifferentialCorrection = -I_1x1;
  }

  return Vector1(error);
}

//***************************************************************************
DoubleDifferencePseudorangeFactor::DoubleDifferencePseudorangeFactor(
    const Key positionKey, const double prRovRef, const double prBaseRef,
    const double prRovTarget, const double prBaseTarget,
    const Point3& satRefRov, const Point3& satTargetRov,
    const Point3& satRefBase, const Point3& satTargetBase,
    const Point3& basePos, const SharedNoiseModel& model)
    : Base(model, positionKey),
      prRovRef_(prRovRef),
      prBaseRef_(prBaseRef),
      prRovTarget_(prRovTarget),
      prBaseTarget_(prBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos) {}

//***************************************************************************
void DoubleDifferencePseudorangeFactor::print(const std::string& s,
                                const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? "" : s + " ") << "DoubleDifferencePseudorangeFactor\n";
  Base::print("", keyFormatter);
}

//***************************************************************************
bool DoubleDifferencePseudorangeFactor::equals(const NonlinearFactor& expected,
                                 double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) &&
         std::abs(prRovRef_ - e->prRovRef_) < tol &&
         std::abs(prBaseRef_ - e->prBaseRef_) < tol &&
         std::abs(prRovTarget_ - e->prRovTarget_) < tol &&
         std::abs(prBaseTarget_ - e->prBaseTarget_) < tol &&
         traits<Point3>::Equals(satRefRov_, e->satRefRov_, tol) &&
         traits<Point3>::Equals(satTargetRov_, e->satTargetRov_, tol) &&
         traits<Point3>::Equals(satRefBase_, e->satRefBase_, tol) &&
         traits<Point3>::Equals(satTargetBase_, e->satTargetBase_, tol) &&
         traits<Point3>::Equals(basePos_, e->basePos_, tol);
}

//***************************************************************************
Vector DoubleDifferencePseudorangeFactor::evaluateError(const Point3& pos,
                                          OptionalMatrixType H) const {
  const double ddObs =
      (prRovRef_ - prBaseRef_) - (prRovTarget_ - prBaseTarget_);

  // Rover: use satellite positions at rover time, with Sagnac-aware Jacobian
  Point3 eRef, eTarget;
  Matrix13 H_rRovRef, H_rRovTarget;
  const double rRovRef = gnss::geodist(satRefRov_, pos, eRef, H_rRovRef);
  const double rRovTarget =
      gnss::geodist(satTargetRov_, pos, eTarget, H_rRovTarget);

  // Base: use satellite positions at base time (no derivative needed)
  Point3 dummy;
  const double rBaseRef = gnss::geodist(satRefBase_, basePos_, dummy);
  const double rBaseTarget = gnss::geodist(satTargetBase_, basePos_, dummy);

  const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
  const double error = ddModel - ddObs;

  if (H) {
    *H = H_rRovRef - H_rRovTarget;
  }
  return Vector1(error);
}

//***************************************************************************
DoubleDifferencePseudorangeFactorArm::DoubleDifferencePseudorangeFactorArm(
    const Key poseKey, const double prRovRef, const double prBaseRef,
    const double prRovTarget, const double prBaseTarget,
    const Point3& satRefRov, const Point3& satTargetRov,
    const Point3& satRefBase, const Point3& satTargetBase,
    const Point3& basePos, const Point3& leverArm,
    const SharedNoiseModel& model)
    : Base(model, poseKey),
      prRovRef_(prRovRef),
      prBaseRef_(prBaseRef),
      prRovTarget_(prRovTarget),
      prBaseTarget_(prBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos),
      bL_(leverArm) {}

//***************************************************************************
DoubleDifferencePseudorangeFactorArm::DoubleDifferencePseudorangeFactorArm(
    const Key poseKey, const double prRovRef, const double prBaseRef,
    const double prRovTarget, const double prBaseTarget,
    const Point3& satRefRov, const Point3& satTargetRov,
    const Point3& satRefBase, const Point3& satTargetBase,
    const Point3& basePos, const Point3& leverArm, const Pose3& ecef_T_nav,
    const SharedNoiseModel& model)
    : Base(model, poseKey),
      prRovRef_(prRovRef),
      prBaseRef_(prBaseRef),
      prRovTarget_(prRovTarget),
      prBaseTarget_(prBaseTarget),
      satRefRov_(satRefRov),
      satTargetRov_(satTargetRov),
      satRefBase_(satRefBase),
      satTargetBase_(satTargetBase),
      basePos_(basePos),
      bL_(leverArm),
      ecef_T_nav_(ecef_T_nav) {}

//***************************************************************************
void DoubleDifferencePseudorangeFactorArm::print(const std::string& s,
                                   const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? "" : s + " ") << "DoubleDifferencePseudorangeFactorArm\n";
  Base::print("", keyFormatter);
}

//***************************************************************************
bool DoubleDifferencePseudorangeFactorArm::equals(const NonlinearFactor& expected,
                                    double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr || !Base::equals(*e, tol)) return false;
  if (std::abs(prRovRef_ - e->prRovRef_) >= tol) return false;
  if (std::abs(prBaseRef_ - e->prBaseRef_) >= tol) return false;
  if (std::abs(prRovTarget_ - e->prRovTarget_) >= tol) return false;
  if (std::abs(prBaseTarget_ - e->prBaseTarget_) >= tol) return false;
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
Vector DoubleDifferencePseudorangeFactorArm::evaluateError(
    const Pose3& pose, OptionalMatrixType H_pose) const {
  Matrix66 H_compose;
  const bool has_nav = ecef_T_nav_.has_value();
  const Pose3 ecef_T_body = has_nav
      ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
      : pose;

  const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
  const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

  const double ddObs =
      (prRovRef_ - prBaseRef_) - (prRovTarget_ - prBaseTarget_);

  Point3 eRef, eTarget, dummy;
  Matrix13 H_rRovRef, H_rRovTarget;
  const double rRovRef = gnss::geodist(satRefRov_, antennaPos, eRef, H_rRovRef);
  const double rRovTarget =
      gnss::geodist(satTargetRov_, antennaPos, eTarget, H_rRovTarget);
  const double rBaseRef = gnss::geodist(satRefBase_, basePos_, dummy);
  const double rBaseTarget = gnss::geodist(satTargetBase_, basePos_, dummy);

  const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
  const double error = ddModel - ddObs;

  if (H_pose) {
    H_pose->resize(1, 6);
    // d(error)/d(antennaPos) = H_rRovRef - H_rRovTarget (1x3)
    const Matrix13 dd_u = H_rRovRef - H_rRovTarget;
    Matrix16 H_ecef;
    H_ecef.block<1, 3>(0, 0) = dd_u * (-ecef_R_body * skewSymmetric(bL_));
    H_ecef.block<1, 3>(0, 3) = dd_u * ecef_R_body;
    *H_pose = has_nav ? H_ecef * H_compose : H_ecef;
  }
  return Vector1(error);
}

}  // namespace gtsam
