/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  NonlinearLikelihood.h
 *  @author Frank Dellaert
 **/
#pragma once

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>

namespace gtsam {

/**
 * A class for a soft prior on any Value type, but with a non-zero mean
 * in the tangent space.
 * The error is `e(x) = Local(origin, x) - mean`.
 * The loss is `0.5 * ||e(x)||^2_Sigma` for a Gaussian noise model,
 * and `rho(e(x))` for a robust noise model `rho`.
 * The likelihood is `exp(-loss)`.
 * @ingroup nonlinear
 **/
template <class VALUE>
class NonlinearLikelihood : public NoiseModelFactorN<VALUE> {
 public:
  typedef VALUE T;

  // Provide access to the Matrix& version of evaluateError:
  using NoiseModelFactor1<VALUE>::evaluateError;

 protected:
  typedef NoiseModelFactorN<VALUE> Base;

  VALUE origin_; /** The point in the manifold at which the tangent space is
                    taken. */
  std::optional<Vector>
      mean_; /** The mean in the tangent space, default is zero vector. */

  /** concept check by type */
  GTSAM_CONCEPT_TESTABLE_TYPE(T)

 public:
  /// Typedef to this class
  typedef NonlinearLikelihood<VALUE> This;

  /// @name Standard Constructors
  /// @{

  /** default constructor - only use for serialization */
  NonlinearLikelihood() {}

  /** Constructor */
  NonlinearLikelihood(Key key, const VALUE& origin,
                      const SharedNoiseModel& model,
                      const std::optional<Vector>& mean = {})
      : Base(model, key), origin_(origin), mean_(mean) {}

  /// @}
  /// @name Standard Destructor
  /// @{

  ~NonlinearLikelihood() override {}

  /// @}
  /// @name Testable
  /// @{

  /** print */
  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override {
    std::cout << s << "NonlinearLikelihood on " << keyFormatter(this->key())
              << "\n";
    traits<T>::Print(origin_, "  origin: ");
    if (mean_) {
      gtsam::print(*mean_, "  tangent space mean: ");
    }
    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
  }

  /** equals */
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    bool mean_equals =
        (!mean_ && !e->mean_) ||
        (mean_ && e->mean_ && equal_with_abs_tol(*mean_, *e->mean_, tol));
    return e != nullptr && Base::equals(*e, tol) &&
           traits<T>::Equals(origin_, e->origin_, tol) && mean_equals;
  }

  /// @}
  /// @name Factor
  /// @{

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /** vector of errors */
  Vector evaluateError(const T& x, OptionalMatrixType H) const override {
    if (H)
      (*H) = Matrix::Identity(traits<T>::GetDimension(x),
                              traits<T>::GetDimension(x));
    // manifold equivalent of (x-z)-mu -> Local(z,x)-mu
    Vector error = traits<T>::Local(origin_, x);
    if (mean_) {
      return error - *mean_;
    }
    return error;
  }

  /** Compute the likelihood of a given value */
  double likelihood(const T& x) const {
    Vector e = evaluateError(x);
    double squared_mahalanobis_distance =
        this->noiseModel_->squaredMahalanobisDistance(e);
    double loss = this->noiseModel_->loss(squared_mahalanobis_distance);
    return exp(-loss);
  }

  /// @}
  /// @name Access
  /// @{

  const VALUE& origin() const { return origin_; }
  const std::optional<Vector>& mean() const { return mean_; }

  /// @}

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    // NoiseModelFactor1 instead of NoiseModelFactorN for backward compatibility
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor1", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(origin_);
    ar& BOOST_SERIALIZATION_NVP(mean_);
  }
#endif

  // Alignment, see
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  inline constexpr static auto NeedsToAlign = (sizeof(T) % 16) == 0;

 public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
};

/// traits
template <class VALUE>
struct traits<NonlinearLikelihood<VALUE> >
    : public Testable<NonlinearLikelihood<VALUE> > {};

}  // namespace gtsam