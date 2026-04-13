/**
 *  @file   PseudorangeFactor.h
 *  @author Sammy Guo
 *  @brief  Header file for GNSS Pseudorange factor
 *  @date   January 18, 2026
 **/
#pragma once

#include <gtsam/base/std_optional_serialization.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>
#include <string>

namespace gtsam {

/**
 * Base class storing common members for GNSS-related pseudorange factors.
 */
struct PseudorangeBase {
  double
      pseudorange_;    ///< Receiver-reported pseudorange measurement in meters.
  Point3 satPos_;      ///< Satellite position in WGS84 ECEF meters.
  double satClkBias_;  ///< Satellite clock bias in seconds.
};

/**
 * Simplified GNSS pseudorange model for basic positioning problems.
 *
 * This factor implements a simplified version of equation 5.6 [1]
 * \rho = r + c[\delta t_u - \delta t^s] + I_{\rho} + T_{\rho} + \epsilon_{\rho}
 * where `\rho` is measured pseudorange (in meters) from the receiver,
 * `r` true range (in meters) between receiver antenna and satellite,
 * `c` is speed of light in a vacuum (m/s),
 * `\delta t_u` is receiver clock bias (seconds),
 * `\delta t_s` is satellite clock bias (seconds),
 * and `I_{\rho}`, `T_{\rho}`, and `\epsilon_{\rho}` are ionospheric,
 * tropospheric, and unmodeled errors respectively.
 *
 * Ionospheric and tropospheric terms are omitted in this simplified factor.
 * Note that this factor is also designed for code-phase measurements.
 *
 * @ingroup navigation
 *
 * REFERENCES:
 * [1] P. Misra et. al., "Global Positioning Systems: Signals, Measurements, and
 * Performance", Second Edition, 2012.
 */
class GTSAM_EXPORT PseudorangeFactor : public NoiseModelFactorN<Point3, double>,
                                       private PseudorangeBase {
 private:
  typedef NoiseModelFactorN<Point3, double> Base;

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<PseudorangeFactor> shared_ptr;

  /// Typedef to this class
  typedef PseudorangeFactor This;

  /** default constructor - only use for serialization */
  PseudorangeFactor() = default;

  virtual ~PseudorangeFactor() = default;

  /**
   * Construct a PseudorangeFactor that models the distance between a receiver
   * and a satellite.
   *
   * @param receiverPositionKey Receiver gtsam::Point3 ECEF position node.
   * @param receiverClockBiasKey Receiver clock bias node.
   * @param measuredPseudorange Receiver-measured pseudorange in meters.
   * @param satellitePosition Satellite ECEF position in meters.
   * @param satelliteClockBias Satellite clock bias in seconds.
   * @param model 1-D pseudorange noise model.
   */
  PseudorangeFactor(
      Key receiverPositionKey, Key receiverClockBiasKey,
      double measuredPseudorange, const Point3& satellitePosition,
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
                       OptionalMatrixType HreceiverPos,
                       OptionalMatrixType HreceiverClockBias) const override;

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  /// Serialization function
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PseudorangeFactor::Base);
    ar& BOOST_SERIALIZATION_NVP(pseudorange_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
  }
#endif
};

/// traits
template <>
struct traits<PseudorangeFactor> : public Testable<PseudorangeFactor> {};

/**
 * Simple differentially-corrected pseudorange factor for precise positioning.
 *
 * This factor implements the model prescribed by chapter 5.8.2 from [1],
 * where a reference GNSS receiver with known position provides differential
 * pseudorange corrections for a "user" receiver to eliminate common-mode
 * atmospheric errors. The idea being that spatially local receivers experience
 * the same atmospheric errors since their signal paths pass through the same
 * regions of Earth's atmosphere. Therefore, this factor accepts an additional
 * "differential correction" variable from a reference receiver to cancel-out
 * local-area biases from the user's pseudoranges.
 *
 * Note that this factor is designed for code-phase measurements.
 *
 * @example Please see the `DifferentialPseudorangeExample.ipynb` notebook
 * for a demonstration of this factor on CORS datasets.
 *
 * @ingroup navigation
 *
 * REFERENCES:
 * [1] P. Misra et. al., "Global Positioning Systems: Signals, Measurements, and
 * Performance", Second Edition, 2012.
 */
class GTSAM_EXPORT DifferentialPseudorangeFactor
    : public NoiseModelFactorN<Point3, double, double>,
      private PseudorangeBase {
 private:
  typedef NoiseModelFactorN<Point3, double, double> Base;

 public:
  using Base::evaluateError;
  typedef std::shared_ptr<DifferentialPseudorangeFactor> shared_ptr;
  typedef DifferentialPseudorangeFactor This;

  DifferentialPseudorangeFactor() = default;
  virtual ~DifferentialPseudorangeFactor() = default;

  DifferentialPseudorangeFactor(
      Key receiverPositionKey, Key receiverClockBiasKey,
      Key differentialCorrectionKey, double measuredPseudorange,
      const Point3& satellitePosition, double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override;
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override;
  Vector evaluateError(
      const Point3& receiverPosition, const double& receiverClock_bias,
      const double& differentialCorrection, OptionalMatrixType HreceiverPos,
      OptionalMatrixType HreceiverClockBias,
      OptionalMatrixType HdifferentialCorrection) const override;

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(
        DifferentialPseudorangeFactor::Base);
    ar& BOOST_SERIALIZATION_NVP(pseudorange_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
  }
#endif
};

/// traits
template <>
struct traits<DifferentialPseudorangeFactor>
    : public Testable<DifferentialPseudorangeFactor> {};

/**
 * GNSS pseudorange factor with lever arm correction.
 *
 * Like PseudorangeFactor, but uses a Pose3 (position + attitude) as the
 * receiver state variable, allowing compensation for a lever arm offset
 * between the body frame origin and the GNSS antenna location.
 *
 * The antenna position is computed as:
 *   antenna_pos = ecef_T_body.translation() + ecef_R_body * bL_
 *
 * where bL_ is the lever arm from the body origin to the antenna in the body
 * frame. The error model is:
 *   error = ||antenna_pos - satPos|| + c*(dt_u - dt_s) - pseudorange
 *
 * When the optional ecef_T_nav transform is provided, the pose key is
 * interpreted as a local navigation frame pose (e.g., ENU), and the factor
 * internally converts it to ECEF via ecef_T_body = ecef_T_nav * nav_T_body.
 * This allows the same Pose3 variable to be shared with ImuFactor.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT PseudorangeFactorArm
    : public NoiseModelFactorN<Pose3, double>,
      private PseudorangeBase {
 private:
  typedef NoiseModelFactorN<Pose3, double> Base;

  Point3 bL_;  ///< Lever arm from body origin to antenna in body frame.
  std::optional<Pose3> ecef_T_nav_;  ///< Optional ECEF-from-nav transform.

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<PseudorangeFactorArm> shared_ptr;

  /// Typedef to this class
  typedef PseudorangeFactorArm This;

  /** default constructor - only use for serialization */
  PseudorangeFactorArm() : PseudorangeBase{0.0, Point3(0, 0, 0), 0.0}, bL_(0, 0, 0) {}

  virtual ~PseudorangeFactorArm() = default;

  /**
   * Construct a PseudorangeFactorArm (ECEF pose key).
   *
   * @param poseKey Receiver gtsam::Pose3 key (body pose in ECEF frame).
   * @param receiverClockBiasKey Receiver clock bias node.
   * @param measuredPseudorange Receiver-measured pseudorange in meters.
   * @param satellitePosition Satellite ECEF position in meters.
   * @param leverArm Translation from body origin to antenna in body frame.
   * @param satelliteClockBias Satellite clock bias in seconds.
   * @param model 1-D pseudorange noise model.
   */
  PseudorangeFactorArm(
      Key poseKey, Key receiverClockBiasKey,
      double measuredPseudorange, const Point3& satellitePosition,
      const Point3& leverArm, double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  /**
   * Construct a PseudorangeFactorArm with ecef_T_nav (local nav frame pose key).
   *
   * @param poseKey Receiver gtsam::Pose3 key (body pose in local nav frame).
   * @param receiverClockBiasKey Receiver clock bias node.
   * @param measuredPseudorange Receiver-measured pseudorange in meters.
   * @param satellitePosition Satellite ECEF position in meters.
   * @param leverArm Translation from body origin to antenna in body frame.
   * @param ecef_T_nav Transform from local navigation frame to ECEF.
   * @param satelliteClockBias Satellite clock bias in seconds.
   * @param model 1-D pseudorange noise model.
   */
  PseudorangeFactorArm(
      Key poseKey, Key receiverClockBiasKey,
      double measuredPseudorange, const Point3& satellitePosition,
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
                       OptionalMatrixType H_pose,
                       OptionalMatrixType HreceiverClockBias) const override;

  /// return the lever arm, a position in the body frame
  inline const Point3& leverArm() const { return bL_; }

  /// return the optional ecef_T_nav transform
  inline const std::optional<Pose3>& ecefTnav() const { return ecef_T_nav_; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  /// Serialization function
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PseudorangeFactorArm::Base);
    ar& BOOST_SERIALIZATION_NVP(pseudorange_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
    ar& BOOST_SERIALIZATION_NVP(bL_);
    ar& BOOST_SERIALIZATION_NVP(ecef_T_nav_);
  }
#endif
};

/// traits
template <>
struct traits<PseudorangeFactorArm>
    : public Testable<PseudorangeFactorArm> {};

/**
 * Differentially-corrected pseudorange factor with lever arm correction.
 *
 * Combines the differential correction model of DifferentialPseudorangeFactor
 * with the lever arm compensation of PseudorangeFactorArm. Uses a Pose3
 * (position + attitude) as the receiver state variable.
 *
 * The error model is:
 *   error = ||antenna_pos - satPos|| + c*(dt_u - dt_s) - pseudorange - correction
 *
 * When the optional ecef_T_nav transform is provided, the pose key is
 * interpreted as a local navigation frame pose (e.g., ENU), and the factor
 * internally converts it to ECEF via ecef_T_body = ecef_T_nav * nav_T_body.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT DifferentialPseudorangeFactorArm
    : public NoiseModelFactorN<Pose3, double, double>,
      private PseudorangeBase {
 private:
  typedef NoiseModelFactorN<Pose3, double, double> Base;
  Point3 bL_;
  std::optional<Pose3> ecef_T_nav_;

 public:
  using Base::evaluateError;
  typedef std::shared_ptr<DifferentialPseudorangeFactorArm> shared_ptr;
  typedef DifferentialPseudorangeFactorArm This;

  DifferentialPseudorangeFactorArm() : PseudorangeBase{0.0, Point3(0, 0, 0), 0.0}, bL_(0, 0, 0) {}
  virtual ~DifferentialPseudorangeFactorArm() = default;

  DifferentialPseudorangeFactorArm(
      Key poseKey, Key receiverClockBiasKey, Key differentialCorrectionKey,
      double measuredPseudorange, const Point3& satellitePosition,
      const Point3& leverArm, double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  DifferentialPseudorangeFactorArm(
      Key poseKey, Key receiverClockBiasKey, Key differentialCorrectionKey,
      double measuredPseudorange, const Point3& satellitePosition,
      const Point3& leverArm, const Pose3& ecef_T_nav,
      double satelliteClockBias = 0.0,
      const SharedNoiseModel& model = noiseModel::Unit::Create(1));

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override;
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override;
  Vector evaluateError(const Pose3& pose,
                       const double& receiverClockBias,
                       const double& differentialCorrection,
                       OptionalMatrixType H_pose,
                       OptionalMatrixType HreceiverClockBias,
                       OptionalMatrixType HdifferentialCorrection) const override;

  inline const Point3& leverArm() const { return bL_; }
  inline const std::optional<Pose3>& ecefTnav() const { return ecef_T_nav_; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(
        DifferentialPseudorangeFactorArm::Base);
    ar& BOOST_SERIALIZATION_NVP(pseudorange_);
    ar& BOOST_SERIALIZATION_NVP(satPos_);
    ar& BOOST_SERIALIZATION_NVP(satClkBias_);
    ar& BOOST_SERIALIZATION_NVP(bL_);
    ar& BOOST_SERIALIZATION_NVP(ecef_T_nav_);
  }
#endif
};

/// traits
template <>
struct traits<DifferentialPseudorangeFactorArm>
    : public Testable<DifferentialPseudorangeFactorArm> {};

/**
 * DD pseudorange factor.
 *
 * Takes four raw (undifferenced) pseudorange observations -- rover and base for
 * both reference and target satellites -- along with satellite positions at
 * rover and base observation times, and base station position.  The factor forms
 * the double-difference internally and computes geometric distances with Sagnac
 * correction.
 *
 * Satellite positions differ between rover and base times because satellites
 * move ~3 km/s. Using the same positions for both causes ~100m DD errors
 * when rover and base observation times differ (e.g. rover 5Hz, base 1Hz).
 *
 * error = (prRovRef - prBaseRef) - (prRovTarget - prBaseTarget)
 *       - [(geodist(satRefRov,pos) - geodist(satRefBase,basePos))
 *        - (geodist(satTargetRov,pos) - geodist(satTargetBase,basePos))]
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT DDPseudorangeFactor : public NoiseModelFactorN<Point3> {
 private:
  typedef NoiseModelFactorN<Point3> Base;

  double prRovRef_;         ///< Rover pseudorange for ref satellite [m]
  double prBaseRef_;        ///< Base pseudorange for ref satellite [m]
  double prRovTarget_;      ///< Rover pseudorange for target satellite [m]
  double prBaseTarget_;     ///< Base pseudorange for target satellite [m]
  Point3 satRefRov_;        ///< Ref satellite ECEF at rover time [m]
  Point3 satTargetRov_;     ///< Target satellite ECEF at rover time [m]
  Point3 satRefBase_;       ///< Ref satellite ECEF at base time [m]
  Point3 satTargetBase_;    ///< Target satellite ECEF at base time [m]
  Point3 basePos_;          ///< Base station ECEF position [m]

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
  typedef std::shared_ptr<DDPseudorangeFactor> shared_ptr;
  typedef DDPseudorangeFactor This;

  DDPseudorangeFactor()
      : prRovRef_(0), prBaseRef_(0), prRovTarget_(0), prBaseTarget_(0),
        satRefRov_(0,0,0), satTargetRov_(0,0,0),
        satRefBase_(0,0,0), satTargetBase_(0,0,0), basePos_(0,0,0) {}

  virtual ~DDPseudorangeFactor() = default;

  /**
   * @param positionKey   Rover Point3 ECEF position.
   * @param prRovRef      Rover pseudorange for ref satellite [m].
   * @param prBaseRef     Base pseudorange for ref satellite [m].
   * @param prRovTarget   Rover pseudorange for target satellite [m].
   * @param prBaseTarget  Base pseudorange for target satellite [m].
   * @param satRefRov     Ref satellite ECEF at rover observation time [m].
   * @param satTargetRov  Target satellite ECEF at rover observation time [m].
   * @param satRefBase    Ref satellite ECEF at base observation time [m].
   * @param satTargetBase Target satellite ECEF at base observation time [m].
   * @param basePos       Base station ECEF position [m].
   * @param model         1-D noise model.
   */
  DDPseudorangeFactor(Key positionKey,
                      double prRovRef, double prBaseRef,
                      double prRovTarget, double prBaseTarget,
                      const Point3& satRefRov, const Point3& satTargetRov,
                      const Point3& satRefBase, const Point3& satTargetBase,
                      const Point3& basePos,
                      const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, positionKey),
        prRovRef_(prRovRef), prBaseRef_(prBaseRef),
        prRovTarget_(prRovTarget), prBaseTarget_(prBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override {
    std::cout << (s.empty() ? "" : s + " ") << "DDPseudorangeFactor\n";
    Base::print("", keyFormatter);
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
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

  Vector evaluateError(const Point3& pos, OptionalMatrixType H) const override {
    const double ddObs = (prRovRef_ - prBaseRef_) - (prRovTarget_ - prBaseTarget_);

    // Rover: use satellite positions at rover time
    Point3 eRef, eTarget;
    const double rRovRef = geodist(satRefRov_, pos, eRef);
    const double rRovTarget = geodist(satTargetRov_, pos, eTarget);

    // Base: use satellite positions at base time
    Point3 dummy;
    const double rBaseRef = geodist(satRefBase_, basePos_, dummy);
    const double rBaseTarget = geodist(satTargetBase_, basePos_, dummy);

    const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
    const double error = ddObs - ddModel;

    if (H) {
      *H = (Matrix(1, 3) << (eRef - eTarget).transpose()).finished();
    }
    return Vector1(error);
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(DDPseudorangeFactor::Base);
    ar& BOOST_SERIALIZATION_NVP(prRovRef_);
    ar& BOOST_SERIALIZATION_NVP(prBaseRef_);
    ar& BOOST_SERIALIZATION_NVP(prRovTarget_);
    ar& BOOST_SERIALIZATION_NVP(prBaseTarget_);
    ar& BOOST_SERIALIZATION_NVP(satRefRov_);
    ar& BOOST_SERIALIZATION_NVP(satTargetRov_);
    ar& BOOST_SERIALIZATION_NVP(satRefBase_);
    ar& BOOST_SERIALIZATION_NVP(satTargetBase_);
    ar& BOOST_SERIALIZATION_NVP(basePos_);
  }
#endif
};

template <>
struct traits<DDPseudorangeFactor> : public Testable<DDPseudorangeFactor> {};

/**
 * DD pseudorange factor with lever arm correction.
 *
 * Like DDPseudorangeFactor but uses Pose3 (position + attitude) with
 * lever arm offset. Optional ecef_T_nav for local navigation frame.
 *
 * @ingroup navigation
 */
class GTSAM_EXPORT DDPseudorangeFactorArm : public NoiseModelFactorN<Pose3> {
 private:
  typedef NoiseModelFactorN<Pose3> Base;

  double prRovRef_;
  double prBaseRef_;
  double prRovTarget_;
  double prBaseTarget_;
  Point3 satRefRov_;
  Point3 satTargetRov_;
  Point3 satRefBase_;
  Point3 satTargetBase_;
  Point3 basePos_;
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
  typedef std::shared_ptr<DDPseudorangeFactorArm> shared_ptr;
  typedef DDPseudorangeFactorArm This;

  DDPseudorangeFactorArm()
      : prRovRef_(0), prBaseRef_(0), prRovTarget_(0), prBaseTarget_(0),
        satRefRov_(0,0,0), satTargetRov_(0,0,0),
        satRefBase_(0,0,0), satTargetBase_(0,0,0),
        basePos_(0,0,0), bL_(0,0,0) {}

  virtual ~DDPseudorangeFactorArm() = default;

  DDPseudorangeFactorArm(Key poseKey,
                         double prRovRef, double prBaseRef,
                         double prRovTarget, double prBaseTarget,
                         const Point3& satRefRov, const Point3& satTargetRov,
                         const Point3& satRefBase, const Point3& satTargetBase,
                         const Point3& basePos, const Point3& leverArm,
                         const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, poseKey),
        prRovRef_(prRovRef), prBaseRef_(prBaseRef),
        prRovTarget_(prRovTarget), prBaseTarget_(prBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos), bL_(leverArm) {}

  DDPseudorangeFactorArm(Key poseKey,
                         double prRovRef, double prBaseRef,
                         double prRovTarget, double prBaseTarget,
                         const Point3& satRefRov, const Point3& satTargetRov,
                         const Point3& satRefBase, const Point3& satTargetBase,
                         const Point3& basePos, const Point3& leverArm,
                         const Pose3& ecef_T_nav,
                         const SharedNoiseModel& model = noiseModel::Unit::Create(1))
      : Base(model, poseKey),
        prRovRef_(prRovRef), prBaseRef_(prBaseRef),
        prRovTarget_(prRovTarget), prBaseTarget_(prBaseTarget),
        satRefRov_(satRefRov), satTargetRov_(satTargetRov),
        satRefBase_(satRefBase), satTargetBase_(satTargetBase),
        basePos_(basePos), bL_(leverArm), ecef_T_nav_(ecef_T_nav) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override {
    std::cout << (s.empty() ? "" : s + " ") << "DDPseudorangeFactorArm\n";
    Base::print("", keyFormatter);
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    if (e == nullptr || !Base::equals(*e, tol)) return false;
    if (std::abs(prRovRef_ - e->prRovRef_) >= tol) return false;
    if (std::abs(prBaseRef_ - e->prBaseRef_) >= tol) return false;
    if (std::abs(prRovTarget_ - e->prRovTarget_) >= tol) return false;
    if (std::abs(prBaseTarget_ - e->prBaseTarget_) >= tol) return false;
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

  Vector evaluateError(const Pose3& pose, OptionalMatrixType H_pose) const override {
    Matrix66 H_compose;
    const bool has_nav = ecef_T_nav_.has_value();
    const Pose3 ecef_T_body = has_nav
        ? ecef_T_nav_->compose(pose, {}, H_pose ? &H_compose : nullptr)
        : pose;

    const Matrix3 ecef_R_body = ecef_T_body.rotation().matrix();
    const Point3 antennaPos = ecef_T_body.translation() + ecef_R_body * bL_;

    const double ddObs = (prRovRef_ - prBaseRef_) - (prRovTarget_ - prBaseTarget_);

    Point3 eRef, eTarget, dummy;
    const double rRovRef = geodist(satRefRov_, antennaPos, eRef);
    const double rRovTarget = geodist(satTargetRov_, antennaPos, eTarget);
    const double rBaseRef = geodist(satRefBase_, basePos_, dummy);
    const double rBaseTarget = geodist(satTargetBase_, basePos_, dummy);

    const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
    const double error = ddObs - ddModel;

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
    return Vector1(error);
  }

  inline const Point3& leverArm() const { return bL_; }
  inline const std::optional<Pose3>& ecefTnav() const { return ecef_T_nav_; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(DDPseudorangeFactorArm::Base);
    ar& BOOST_SERIALIZATION_NVP(prRovRef_);
    ar& BOOST_SERIALIZATION_NVP(prBaseRef_);
    ar& BOOST_SERIALIZATION_NVP(prRovTarget_);
    ar& BOOST_SERIALIZATION_NVP(prBaseTarget_);
    ar& BOOST_SERIALIZATION_NVP(satRefRov_);
    ar& BOOST_SERIALIZATION_NVP(satTargetRov_);
    ar& BOOST_SERIALIZATION_NVP(satRefBase_);
    ar& BOOST_SERIALIZATION_NVP(satTargetBase_);
    ar& BOOST_SERIALIZATION_NVP(basePos_);
    ar& BOOST_SERIALIZATION_NVP(bL_);
    ar& BOOST_SERIALIZATION_NVP(ecef_T_nav_);
  }
#endif
};

template <>
struct traits<DDPseudorangeFactorArm> : public Testable<DDPseudorangeFactorArm> {};

}  // namespace gtsam
