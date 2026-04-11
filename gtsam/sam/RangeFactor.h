/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  RangeFactor.h
 *  @brief Serializable factor induced by a range measurement
 *  @date July 2015
 *  @author Frank Dellaert
 **/

#pragma once

#include <gtsam/nonlinear/NoiseModelFactorN.h>

namespace gtsam {

// forward declaration of Range functor, assumed partially specified
template <typename A1, typename A2>
struct Range;

/**
 * Binary factor for a range measurement
 * Works for any two types A1,A2 for which the functor Range<A1,A2>() is defined
 * @ingroup sam
 */
template <typename A1, typename A2 = A1, typename T = double>
class RangeFactor : public NoiseModelFactorN<A1, A2> {
 private:
  typedef RangeFactor<A1, A2, T> This;
  typedef NoiseModelFactorN<A1, A2> Base;

  T measured_;  ///< The measured range

 public:
  /// default constructor
  RangeFactor() = default;

  /// Construct from keys, a range measurement, and a noise model.
  RangeFactor(Key key1, Key key2, T measured, const SharedNoiseModel& model)
      : Base(model, key1, key2), measured_(measured) {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// @return the measurement
  const T& measured() const { return measured_; }

  /// Evaluate the unwhitened range error and optional Jacobians.
  Vector evaluateError(const A1& a1, const A2& a2,
                       OptionalMatrixType H1 = OptionalNone,
                       OptionalMatrixType H2 = OptionalNone) const override {
    Matrix Hpred1, Hpred2, Hlocal;
    const T predicted =
        Range<A1, A2>()(a1, a2, H1 ? &Hpred1 : nullptr, H2 ? &Hpred2 : nullptr);
    const Vector error =
        -traits<T>::Local(predicted, measured_, (H1 || H2) ? &Hlocal : nullptr);
    if (H1) *H1 = -Hlocal * Hpred1;
    if (H2) *H2 = -Hlocal * Hpred2;
    return error;
  }

  /// print
  void print(const std::string& s = "",
             const KeyFormatter& kf = DefaultKeyFormatter) const override {
    std::cout << s << "RangeFactor" << std::endl;
    Base::print(s, kf);
    traits<T>::Print(measured_, "  measured: ");
  }

  /// Check equality up to a tolerance.
  bool equals(const NonlinearFactor& f, double tol) const override {
    const This* p = dynamic_cast<const This*>(&f);
    return p != nullptr && Base::equals(f, tol) &&
           traits<T>::Equals(measured_, p->measured_, tol);
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  /// Serialize this factor.
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(measured_);
  }
#endif
};  // \ RangeFactor

/// traits
template <typename A1, typename A2, typename T>
struct traits<RangeFactor<A1, A2, T> >
    : public Testable<RangeFactor<A1, A2, T> > {};

/**
 * Binary factor for a range measurement, with a transform applied
 * @ingroup sam
 */
template <typename A1, typename A2 = A1,
          typename T = typename Range<A1, A2>::result_type>
class RangeFactorWithTransform : public NoiseModelFactorN<A1, A2> {
 private:
  typedef RangeFactorWithTransform<A1, A2, T> This;
  typedef NoiseModelFactorN<A1, A2> Base;

  T measured_;        ///< The measured range
  A1 body_T_sensor_;  ///< The pose of the sensor in the body frame

 public:
  /// Default constructor
  RangeFactorWithTransform() = default;

  /// Construct from keys, a range measurement, a noise model, and transform.
  RangeFactorWithTransform(Key key1, Key key2, T measured,
                           const SharedNoiseModel& model,
                           const A1& body_T_sensor)
      : Base(model, key1, key2),
        measured_(measured),
        body_T_sensor_(body_T_sensor) {}

  /// Destroy this factor.
  ~RangeFactorWithTransform() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// @return the measurement
  const T& measured() const { return measured_; }

  /// Evaluate the unwhitened range error and optional Jacobians.
  Vector evaluateError(const A1& a1, const A2& a2,
                       OptionalMatrixType H1 = OptionalNone,
                       OptionalMatrixType H2 = OptionalNone) const override {
    Matrix HposeCompose, HsensorRange, H2Range, Hlocal;
    const A1 nav_T_sensor =
        a1.compose(body_T_sensor_, H1 ? &HposeCompose : nullptr);
    const T predicted =
        Range<A1, A2>()(nav_T_sensor, a2, H1 ? &HsensorRange : nullptr,
                        H2 ? &H2Range : nullptr);
    const Vector error =
        -traits<T>::Local(predicted, measured_, (H1 || H2) ? &Hlocal : nullptr);
    if (H1) *H1 = -Hlocal * HsensorRange * HposeCompose;
    if (H2) *H2 = -Hlocal * H2Range;
    return error;
  }

  // An evaluateError overload to accept matrices (Matrix&) and pass it to the
  // OptionalMatrixType evaluateError overload
  Vector evaluateError(const A1& a1, const A2& a2, Matrix& H1,
                       Matrix& H2) const {
    return evaluateError(a1, a2, &H1, &H2);
  }

  /// print contents
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "RangeFactorWithTransform" << std::endl;
    this->body_T_sensor_.print("  sensor pose in body frame: ");
    Base::print(s, keyFormatter);
    traits<T>::Print(measured_, "  measured: ");
  }

  /// Check equality up to a tolerance.
  bool equals(const NonlinearFactor& f, double tol) const override {
    const This* p = dynamic_cast<const This*>(&f);
    return p != nullptr && Base::equals(f, tol) &&
           traits<T>::Equals(measured_, p->measured_, tol) &&
           traits<A1>::Equals(body_T_sensor_, p->body_T_sensor_, tol);
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  /// Serialize this factor.
  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(measured_);
    ar& BOOST_SERIALIZATION_NVP(body_T_sensor_);
  }
#endif
};  // \ RangeFactorWithTransform

/// traits
template <typename A1, typename A2, typename T>
struct traits<RangeFactorWithTransform<A1, A2, T> >
    : public Testable<RangeFactorWithTransform<A1, A2, T> > {};

}  // namespace gtsam
