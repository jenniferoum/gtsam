/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file NonlinearDensity.h
 * @date September 30, 2025
 * @author Frank Dellaert
 * @brief A nonlinear density, inherits from NonlinearLikelihood.
 */

#pragma once

#include <gtsam/nonlinear/NonlinearLikelihood.h>
#include <gtsam/nonlinear/Values.h>

namespace gtsam {

/**
 * A nonlinear density, inherits from NonlinearLikelihood.
 * @ingroup nonlinear
 */
template <class VALUE>
class NonlinearDensity : public NonlinearLikelihood<VALUE> {
 public:
  typedef VALUE T;
  typedef NonlinearLikelihood<VALUE> Base;

  /// @name Standard Constructors
  /// @{

  /// Default constructor for serialization.
  NonlinearDensity() {}

  /// Constructor
  NonlinearDensity(Key key, const VALUE& origin, const SharedNoiseModel& model,
                   const std::optional<Vector>& mean = {})
      : Base(key, origin, model, mean) {}

  /// @}
  /// @name Standard Destructor
  /// @{
  ~NonlinearDensity() override {}
  /// @}
  /// @name Testable
  /// @{

  /// print
  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override {
    std::cout << s << "NonlinearDensity on " << keyFormatter(this->key())
              << "\n";
    traits<T>::Print(this->origin_, "  origin: ");
    if (this->mean_) gtsam::print(*this->mean_, "  tangent space mean: ");
    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model\n";
  }

  /// equals
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const auto* e = dynamic_cast<const NonlinearDensity*>(&expected);
    return e && Base::equals(*e, tol);
  }

  /// @}
  /// @name Standard API
  /// @{

  /**
   * Calculate the log-probability of the given value.
   * error(x) as defined for a GTSAM factor already equals 0.5 * ||r(x)||^2_Σ
   * (i.e. the negative log-likelihood without the normalization constant).
   * Hence: log P(x) = log k - error(x).
   */
  double logProbability(const T& x) const {
    return -(negLogConstant() + this->error(x));
  }

  /**
   * Evaluate the probability density at the given value.
   * P(x) = exp(logProbability(x)).
   */
  double evaluate(const T& x) const { return exp(logProbability(x)); }

  /**
   * Log-probability overload taking a Values container. This mirrors the
   * linear GaussianConditional interface so densities can be queried in a
   * uniform way when only a Values is available.
   */
  double logProbability(const Values& values) const {
    const T& x = values.at<T>(this->key());
    return logProbability(x);
  }

  /// Evaluate density P(x) using a Values container.
  double evaluate(const Values& values) const {
    const T& x = values.at<T>(this->key());
    return evaluate(x);
  }

  /**
   * Calculate the normalization constant for the density.
   * For a Gaussian noise model with covariance Σ, we return
   *   - log k = 0.5 * n * log(2*pi) + 0.5 * log |Σ|
   * where n = dim().  Note: gaussian->logDeterminant() returns log|Σ|.
   * For non-Gaussian noise models this is not (straightforwardly) defined and
   * we throw.
   */
  double negLogConstant() const {
    // Get number of rows
    const size_t n = this->dim();

    // Get noise model and dynamic cast to Gaussian
    const auto& noiseModel = this->noiseModel();
    auto gaussian = std::dynamic_pointer_cast<noiseModel::Gaussian>(noiseModel);
    if (gaussian) {
      constexpr double log2pi = 1.8378770664093454835606594728112;  // log(2*pi)
      const double logDetSigma = gaussian->logDeterminant();        // log |Σ|
      return 0.5 * n * log2pi + 0.5 * logDetSigma;
    }

    // If not Gaussian, throw an error
    throw std::runtime_error(
        "NonlinearDensity::negLogConstant() is only implemented for "
        "Gaussian noise models. The noise model used is of type " +
        std::string(typeid(*noiseModel).name()));
  }
  /// @}

 private:
#ifdef GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NonlinearLikelihood", boost::serialization::base_object<Base>(*this));
  }
#endif
};

}  // namespace gtsam
