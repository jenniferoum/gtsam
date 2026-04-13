/**
 *  @file   CarrierPhaseFactor.h
 *  @brief  Header file for GNSS Carrier Phase factors
 *  @date   March 23, 2026
 **/
#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>
#include <string>

namespace gtsam {

/**
 * Base class storing common members for carrier phase factors.
 *
 * Clock biases are in seconds (same convention as PseudorangeFactorArm).
 * Ambiguity is in meters (= lambda * N). To recover the integer
 * ambiguity N, divide by the wavelength: N = ambiguity_meters / lambda
 */
struct CarrierPhaseBase {
  double carrierPhase_;  ///< Carrier phase measurement in meters.
  Point3 satPos_;        ///< Satellite position in WGS84 ECEF meters.
  double satClkBias_;    ///< Satellite clock bias in seconds.
};

/**
 * Undifferenced GNSS carrier phase factor for point positioning.
 *
 * The error model is:
 *   error = ||recv_pos - satPos|| + c*(dt_u - dt_s) + ambiguity - phi
 *
 * where dt_u, dt_s are in seconds and ambiguity is in meters.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT CarrierPhaseFactor
    : public NoiseModelFactorN<Point3, double, double>,
      private CarrierPhaseBase {
 private:
  typedef NoiseModelFactorN<Point3, double, double> Base;

 public:
  using Base::evaluateError;

  typedef std::shared_ptr<CarrierPhaseFactor> shared_ptr;
  typedef CarrierPhaseFactor This;

  /** default constructor - only use for serialization */
  CarrierPhaseFactor()
      : CarrierPhaseBase{0.0, Point3(0, 0, 0), 0.0} {}

  virtual ~CarrierPhaseFactor() = default;

  /**
   * Construct a CarrierPhaseFactor.
   *
   * @param receiverPositionKey Receiver gtsam::Point3 ECEF position node.
   * @param receiverClockBiasKey Receiver clock bias node (seconds).
   * @param ambiguityKey Ambiguity node (meters, = lambda * N)..
   * @param measuredCarrierPhase Carrier phase measurement in meters.
   * @param satellitePosition Satellite ECEF position in meters.
   * @param satelliteClockBias Satellite clock bias in seconds.
   * @param model 1-D noise model.
   */
  CarrierPhaseFactor(
      Key receiverPositionKey, Key receiverClockBiasKey, Key ambiguityKey,
      double measuredCarrierPhase, const Point3& satellitePosition,
      double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// print
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override;

  /// equals
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override;

  /// vector of errors
  Vector evaluateError(const Point3& receiverPosition,
                       const double& receiverClockBias,
                       const double& ambiguity,
                       OptionalMatrixType HreceiverPos,
                       OptionalMatrixType HreceiverClockBias,
                       OptionalMatrixType Hambiguity) const override;

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(CarrierPhaseFactor::Base);
    ar& BOOST_SERIALIZATION_NVP(carrierPhase_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
  }
#endif
};

/// traits
template <>
struct traits<CarrierPhaseFactor> : public Testable<CarrierPhaseFactor> {};

/**
 * Carrier phase factor with lever arm correction.
 *
 * Like CarrierPhaseFactor, but uses a Pose3 (position + attitude) as the
 * receiver state variable, allowing compensation for a lever arm offset.
 *
 * The antenna position is computed as:
 *   antenna_pos = ecef_T_body.translation() + ecef_R_body * bL_
 *
 * The error model is:
 *   error = ||antenna_pos - satPos|| + c*(dt_u - dt_s) + ambiguity - phi
 *
 * When the optional ecef_T_nav transform is provided, the pose key is
 * interpreted as a local navigation frame pose (e.g., ENU), and the factor
 * internally converts it to ECEF via ecef_T_body = ecef_T_nav * nav_T_body.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT CarrierPhaseFactorArm
    : public NoiseModelFactorN<Pose3, double, double>,
      private CarrierPhaseBase {
 private:
  typedef NoiseModelFactorN<Pose3, double, double> Base;

  Point3 bL_;  ///< Lever arm from body origin to antenna in body frame.
  std::optional<Pose3> ecef_T_nav_;  ///< Optional ECEF-from-nav transform.

 public:
  using Base::evaluateError;

  typedef std::shared_ptr<CarrierPhaseFactorArm> shared_ptr;
  typedef CarrierPhaseFactorArm This;

  /** default constructor - only use for serialization */
  CarrierPhaseFactorArm()
      : CarrierPhaseBase{0.0, Point3(0, 0, 0), 0.0}, bL_(0, 0, 0) {}

  virtual ~CarrierPhaseFactorArm() = default;

  /**
   * Construct a CarrierPhaseFactorArm (ECEF pose key).
   */
  CarrierPhaseFactorArm(
      Key poseKey, Key receiverClockBiasKey, Key ambiguityKey,
      double measuredCarrierPhase, const Point3& satellitePosition,
      const Point3& leverArm, double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  /**
   * Construct a CarrierPhaseFactorArm with ecef_T_nav (local nav frame pose).
   */
  CarrierPhaseFactorArm(
      Key poseKey, Key receiverClockBiasKey, Key ambiguityKey,
      double measuredCarrierPhase, const Point3& satellitePosition,
      const Point3& leverArm, const Pose3& ecef_T_nav,
      double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// print
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override;

  /// equals
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override;

  /// vector of errors
  Vector evaluateError(const Pose3& pose,
                       const double& receiverClockBias,
                       const double& ambiguity,
                       OptionalMatrixType H_pose,
                       OptionalMatrixType HreceiverClockBias,
                       OptionalMatrixType Hambiguity) const override;

  /// return the lever arm
  inline const Point3& leverArm() const { return bL_; }

  /// return the optional ecef_T_nav transform
  inline const std::optional<Pose3>& ecefTnav() const { return ecef_T_nav_; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(CarrierPhaseFactorArm::Base);
    ar& BOOST_SERIALIZATION_NVP(carrierPhase_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
    ar& BOOST_SERIALIZATION_NVP(bL_);
    ar& BOOST_SERIALIZATION_NVP(ecef_T_nav_);
  }
#endif
};

/// traits
template <>
struct traits<CarrierPhaseFactorArm>
    : public Testable<CarrierPhaseFactorArm> {};

/**
 * DD carrier phase factor.
 *
 * Takes four raw (undifferenced) carrier phase observations -- rover and base
 * for both reference and target satellites -- along with satellite positions at
 * both rover and base observation times, base station position, and wavelength.
 * The factor forms the double-difference internally.
 *
 * error = (cpRovRef - cpBaseRef) - (cpRovTarget - cpBaseTarget)
 *       - [(geodist(satRefRov,pos) - geodist(satRefBase,basePos))
 *        - (geodist(satTargetRov,pos) - geodist(satTargetBase,basePos))]
 *       - lam * (ambRef - ambTarget)
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT DDCarrierPhaseFactor
    : public NoiseModelFactorN<Point3, double, double> {
 private:
  typedef NoiseModelFactorN<Point3, double, double> Base;

  double cpRovRef_;
  double cpBaseRef_;
  double cpRovTarget_;
  double cpBaseTarget_;
  Point3 satRefRov_;
  Point3 satTargetRov_;
  Point3 satRefBase_;
  Point3 satTargetBase_;
  Point3 basePos_;
  double lam_;

  static constexpr double OMGE = 7.2921151467e-5;
  static constexpr double C_LIGHT = 299792458.0;

  static double geodist(const Point3& sat, const Point3& rcv, Point3& e) {
    const Point3 dr = sat - rcv;
    const double r = dr.norm();
    e = dr / r;
    return r + OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
  }

 public:
  using Base::evaluateError;
  typedef std::shared_ptr<DDCarrierPhaseFactor> shared_ptr;
  typedef DDCarrierPhaseFactor This;

  DDCarrierPhaseFactor()
      : cpRovRef_(0), cpBaseRef_(0), cpRovTarget_(0), cpBaseTarget_(0),
        satRefRov_(0,0,0), satTargetRov_(0,0,0),
        satRefBase_(0,0,0), satTargetBase_(0,0,0),
        basePos_(0,0,0), lam_(0) {}

  virtual ~DDCarrierPhaseFactor() = default;

  DDCarrierPhaseFactor(Key positionKey, Key ambRefKey, Key ambTargetKey,
                       double cpRovRef, double cpBaseRef,
                       double cpRovTarget, double cpBaseTarget,
                       const Point3& satRefRov, const Point3& satTargetRov,
                       const Point3& satRefBase, const Point3& satTargetBase,
                       const Point3& basePos, double lam,
                       const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, positionKey, ambRefKey, ambTargetKey),
        cpRovRef_(cpRovRef), cpBaseRef_(cpBaseRef),
        cpRovTarget_(cpRovTarget), cpBaseTarget_(cpBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos), lam_(lam) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override {
    std::cout << (s.empty() ? "" : s + " ") << "DDCarrierPhaseFactor\n";
    std::cout << "  lam: " << lam_ << "\n";
    Base::print("", keyFormatter);
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
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

  Vector evaluateError(const Point3& pos,
                       const double& ambRef, const double& ambTarget,
                       OptionalMatrixType Hpos,
                       OptionalMatrixType HambRef,
                       OptionalMatrixType HambTarget) const override {
    const double ddObs = (cpRovRef_ - cpBaseRef_) - (cpRovTarget_ - cpBaseTarget_);

    Point3 eRef, eTarget, dummy;
    const double rRovRef = geodist(satRefRov_, pos, eRef);
    const double rRovTarget = geodist(satTargetRov_, pos, eTarget);
    const double rBaseRef = geodist(satRefBase_, basePos_, dummy);
    const double rBaseTarget = geodist(satTargetBase_, basePos_, dummy);

    const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
    const double error = ddObs - ddModel - lam_ * (ambRef - ambTarget);

    if (Hpos) *Hpos = (Matrix(1, 3) << (eRef - eTarget).transpose()).finished();
    if (HambRef) *HambRef = (Matrix(1, 1) << -lam_).finished();
    if (HambTarget) *HambTarget = (Matrix(1, 1) << lam_).finished();

    return Vector1(error);
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(DDCarrierPhaseFactor::Base);
    ar& BOOST_SERIALIZATION_NVP(cpRovRef_);
    ar& BOOST_SERIALIZATION_NVP(cpBaseRef_);
    ar& BOOST_SERIALIZATION_NVP(cpRovTarget_);
    ar& BOOST_SERIALIZATION_NVP(cpBaseTarget_);
    ar& BOOST_SERIALIZATION_NVP(satRefRov_);
    ar& BOOST_SERIALIZATION_NVP(satTargetRov_);
    ar& BOOST_SERIALIZATION_NVP(satRefBase_);
    ar& BOOST_SERIALIZATION_NVP(satTargetBase_);
    ar& BOOST_SERIALIZATION_NVP(basePos_);
    ar& BOOST_SERIALIZATION_NVP(lam_);
  }
#endif
};

template <>
struct traits<DDCarrierPhaseFactor> : public Testable<DDCarrierPhaseFactor> {};

/**
 * DD carrier phase factor with lever arm correction.
 *
 * Like DDCarrierPhaseFactor but uses Pose3 (position + attitude) with
 * lever arm offset. Optional ecef_T_nav for local navigation frame.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT DDCarrierPhaseFactorArm
    : public NoiseModelFactorN<Pose3, double, double> {
 private:
  typedef NoiseModelFactorN<Pose3, double, double> Base;

  double cpRovRef_;
  double cpBaseRef_;
  double cpRovTarget_;
  double cpBaseTarget_;
  Point3 satRefRov_;
  Point3 satTargetRov_;
  Point3 satRefBase_;
  Point3 satTargetBase_;
  Point3 basePos_;
  double lam_;
  Point3 bL_;
  std::optional<Pose3> ecef_T_nav_;

  static constexpr double OMGE = 7.2921151467e-5;
  static constexpr double C_LIGHT = 299792458.0;

  static double geodist(const Point3& sat, const Point3& rcv, Point3& e) {
    const Point3 dr = sat - rcv;
    const double r = dr.norm();
    e = dr / r;
    return r + OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
  }

 public:
  using Base::evaluateError;
  typedef std::shared_ptr<DDCarrierPhaseFactorArm> shared_ptr;
  typedef DDCarrierPhaseFactorArm This;

  DDCarrierPhaseFactorArm()
      : cpRovRef_(0), cpBaseRef_(0), cpRovTarget_(0), cpBaseTarget_(0),
        satRefRov_(0,0,0), satTargetRov_(0,0,0),
        satRefBase_(0,0,0), satTargetBase_(0,0,0),
        basePos_(0,0,0), lam_(0), bL_(0,0,0) {}

  virtual ~DDCarrierPhaseFactorArm() = default;

  DDCarrierPhaseFactorArm(Key poseKey, Key ambRefKey, Key ambTargetKey,
                          double cpRovRef, double cpBaseRef,
                          double cpRovTarget, double cpBaseTarget,
                          const Point3& satRefRov, const Point3& satTargetRov,
                          const Point3& satRefBase, const Point3& satTargetBase,
                          const Point3& basePos, double lam,
                          const Point3& leverArm,
                          const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, poseKey, ambRefKey, ambTargetKey),
        cpRovRef_(cpRovRef), cpBaseRef_(cpBaseRef),
        cpRovTarget_(cpRovTarget), cpBaseTarget_(cpBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos), lam_(lam), bL_(leverArm) {}

  DDCarrierPhaseFactorArm(Key poseKey, Key ambRefKey, Key ambTargetKey,
                          double cpRovRef, double cpBaseRef,
                          double cpRovTarget, double cpBaseTarget,
                          const Point3& satRefRov, const Point3& satTargetRov,
                          const Point3& satRefBase, const Point3& satTargetBase,
                          const Point3& basePos, double lam,
                          const Point3& leverArm, const Pose3& ecef_T_nav,
                          const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, poseKey, ambRefKey, ambTargetKey),
        cpRovRef_(cpRovRef), cpBaseRef_(cpBaseRef),
        cpRovTarget_(cpRovTarget), cpBaseTarget_(cpBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos), lam_(lam), bL_(leverArm), ecef_T_nav_(ecef_T_nav) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override {
    std::cout << (s.empty() ? "" : s + " ") << "DDCarrierPhaseFactorArm\n";
    std::cout << "  lam: " << lam_ << "\n";
    Base::print("", keyFormatter);
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    if (e == nullptr || !Base::equals(*e, tol)) return false;
    if (std::abs(cpRovRef_ - e->cpRovRef_) >= tol) return false;
    if (std::abs(cpBaseRef_ - e->cpBaseRef_) >= tol) return false;
    if (std::abs(cpRovTarget_ - e->cpRovTarget_) >= tol) return false;
    if (std::abs(cpBaseTarget_ - e->cpBaseTarget_) >= tol) return false;
    if (std::abs(lam_ - e->lam_) >= tol) return false;
    if (!traits<Point3>::Equals(satRefRov_, e->satRefRov_, tol)) return false;
    if (!traits<Point3>::Equals(satTargetRov_, e->satTargetRov_, tol)) return false;
    if (!traits<Point3>::Equals(satRefBase_, e->satRefBase_, tol)) return false;
    if (!traits<Point3>::Equals(satTargetBase_, e->satTargetBase_, tol)) return false;
    if (!traits<Point3>::Equals(basePos_, e->basePos_, tol)) return false;
    if (!traits<Point3>::Equals(bL_, e->bL_, tol)) return false;
    if (ecef_T_nav_.has_value() != e->ecef_T_nav_.has_value()) return false;
    if (ecef_T_nav_ && !ecef_T_nav_->equals(*e->ecef_T_nav_, tol)) return false;
    return true;
  }

  Vector evaluateError(const Pose3& pose,
                       const double& ambRef, const double& ambTarget,
                       OptionalMatrixType H_pose,
                       OptionalMatrixType HambRef,
                       OptionalMatrixType HambTarget) const override {
    Matrix66 H_compose;
    const bool has_nav = ecef_T_nav_.has_value();
    const Pose3 ecef_T_body = has_nav
        ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
        : pose;

    const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
    const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

    const double ddObs = (cpRovRef_ - cpBaseRef_) - (cpRovTarget_ - cpBaseTarget_);

    Point3 eRef, eTarget, dummy;
    const double rRovRef = geodist(satRefRov_, antennaPos, eRef);
    const double rRovTarget = geodist(satTargetRov_, antennaPos, eTarget);
    const double rBaseRef = geodist(satRefBase_, basePos_, dummy);
    const double rBaseTarget = geodist(satTargetBase_, basePos_, dummy);

    const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
    const double error = ddObs - ddModel - lam_ * (ambRef - ambTarget);

    if (H_pose) {
      H_pose->resize(1, 6);
      const bool ok = rRovRef > std::numeric_limits<double>::epsilon() &&
                      rRovTarget > std::numeric_limits<double>::epsilon();
      if (!ok) {
        H_pose->setZero();
      } else {
        const Matrix13 dd_u = (eRef - eTarget).transpose();
        Matrix16 H_ecef;
        H_ecef.block<1, 3>(0, 0) = dd_u * (-ecef_R_body * skewSymmetric(bL_));
        H_ecef.block<1, 3>(0, 3) = dd_u * ecef_R_body;
        *H_pose = has_nav ? H_ecef * H_compose : H_ecef;
      }
    }
    if (HambRef) *HambRef = (Matrix(1, 1) << -lam_).finished();
    if (HambTarget) *HambTarget = (Matrix(1, 1) << lam_).finished();

    return Vector1(error);
  }

  inline const Point3& leverArm() const { return bL_; }
  inline const std::optional<Pose3>& ecefTnav() const { return ecef_T_nav_; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(DDCarrierPhaseFactorArm::Base);
    ar& BOOST_SERIALIZATION_NVP(cpRovRef_);
    ar& BOOST_SERIALIZATION_NVP(cpBaseRef_);
    ar& BOOST_SERIALIZATION_NVP(cpRovTarget_);
    ar& BOOST_SERIALIZATION_NVP(cpBaseTarget_);
    ar& BOOST_SERIALIZATION_NVP(satRefRov_);
    ar& BOOST_SERIALIZATION_NVP(satTargetRov_);
    ar& BOOST_SERIALIZATION_NVP(satRefBase_);
    ar& BOOST_SERIALIZATION_NVP(satTargetBase_);
    ar& BOOST_SERIALIZATION_NVP(basePos_);
    ar& BOOST_SERIALIZATION_NVP(lam_);
    ar& BOOST_SERIALIZATION_NVP(bL_);
    ar& BOOST_SERIALIZATION_NVP(ecef_T_nav_);
  }
#endif
};

template <>
struct traits<DDCarrierPhaseFactorArm> : public Testable<DDCarrierPhaseFactorArm> {};

}  // namespace gtsam
