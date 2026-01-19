/**
 *  @file   PseudorangeFactor.h
 *  @author Sammy Guo
 *  @brief  Header file for GNSS Pseudorange factor
 *  @date   January 18, 2026
 **/
#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <string>

namespace gtsam {

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
 *
 * @ingroup navigation
 *
 * REFERENCES:
 * [1] P. Misra et. al., "Global Positioning Systems: Signals, Measurements, and
 * Performance", Second Edition, 2012.
 */
class GTSAM_EXPORT PseudorangeFactor
    : public NoiseModelFactorN<Point3, double> {
 private:
  typedef NoiseModelFactorN<Point3, double> Base;

  const double
      pseudorange_;  ///< Receiver-reported pseudorange measurement in meters.
  const Point3 sat_pos_;       ///< Satellite position in WGS84 ECEF meters.
  const double sat_clk_bias_;  ///< Satellite clock bias in seconds.

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<PseudorangeFactor> shared_ptr;

  /// Typedef to this class
  typedef PseudorangeFactor This;

  /** default constructor - only use for serialization */
  PseudorangeFactor();

  virtual ~PseudorangeFactor() = default;

  /**
   * @brief TODO
   */
  PseudorangeFactor(Key receiver_position_key, Key receiver_clock_bias_key,
                    double measured_pseudorange,
                    const Point3& satellite_position,
                    double satellite_clock_bias, const SharedNoiseModel& model);

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// print
  //  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
  //                                            DefaultKeyFormatter) const
  //                                            override;

  /// equals
  //  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const
  //  override;

  /// vector of errors
  Vector evaluateError(const Point3& receiver_position,
                       const double& receiver_clock_bias,
                       OptionalMatrixType Hreceiver_pos,
                       OptionalMatrixType Hreceiver_clock_bias) const override;

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  /// Serialization function
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(pseudorange_);
    ar& BOOST_SERIALIZATION_NVP(sat_pos_);
    ar& BOOST_SERIALIZATION_NVP(sat_clk_bias_);
  }
#endif
};

}  // namespace gtsam
