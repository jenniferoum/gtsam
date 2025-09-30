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
  NonlinearDensity(Key key, const VALUE& origin, const SharedNoiseModel& model)
      : Base(key, origin, model) {}

  /// @}
  /// @name Standard Destructor
  /// @{
  ~NonlinearDensity() override {}
  /// @}

  /**
   * Calculate the log-probability of the given value.
   * log P(x) = -0.5 * error(x) + C,
   * where C is the normalization constant.
   * The error is 0.5 * |h(x)-z|^2_Sigma, and for this factor h(x) = Local(origin,x) and z=0.
   */
  double logProbability(const T& x) const {
    const double error = this->error(x);
    return -error + this->normalizationConstant();
  }

  /**
   * Evaluate the error at the given value.
   * This is the same as the negative log-probability without the normalization constant.
   */
  double evaluate(const T& x) const { return this->error(x); }

 private:
  /**
   * Calculate the normalization constant for the density.
   * For a Gaussian noise model, this is -0.5*n*log(2*pi) + log|R|, where R is the upper-triangular factor of the precision matrix.
   * For other noise models, this is not well-defined and this method will throw an exception.
   */
  double normalizationConstant() const {
    // Get number of rows
    const size_t n = this->dim();

    // Get noise model and dynamic cast to Gaussian
    const auto& noiseModel = this->noiseModel();
    auto gaussian = std::dynamic_pointer_cast<noiseModel::Gaussian>(noiseModel);
    if (gaussian) {
      const double logdetR = gaussian->logDeterminant();
      return -0.5 * n * log(2.0 * M_PI) + logdetR;
    }

    // If not Gaussian, throw an error
    throw std::runtime_error(
        "NonlinearDensity::normalizationConstant() is only implemented for "
        "Gaussian noise models. The noise model used is of type " +
        std::string(typeid(*noiseModel).name()));
  }

  /// @name Advanced Interface
  /// @{

#ifdef GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NonlinearLikelihood",
        boost::serialization::base_object<Base>(*this));
  }
#endif

  /// @}
};

}  // namespace gtsam
