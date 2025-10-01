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
   * Fusion operator implementing the (approximate) three-step Fusion
   * method in Ge–van Goor–Mahony (2024): choose a reference, express both
   * densities as extended concentrated Gaussians in that chart, fuse the
   * Gaussians, then reset to a zero-mean concentrated Gaussian.
   *
   * Notes/assumptions:
   *  - Only supports Gaussian noise models; throws otherwise.
   *  - Uses a full first-order Jacobian for the change of coordinates between charts
   *    via the chain rule: J_map = (∂Local(x̂,q)/∂q)|_{q=origin} · (∂Retract(origin,δ)/∂δ)|_{δ=0}.
   *  - If both inputs share the same origin and mean, this reduces to
   *    classical Gaussian fusion: Σ⁺ = (Σ₁^{-1}+Σ₂^{-1})^{-1} at the same origin.
   */
  NonlinearDensity operator*(const NonlinearDensity& other) const {
    // 0) Sanity checks
    if (this->key() != other.key())
      throw std::invalid_argument("NonlinearDensity::operator*: keys differ");

    // Extract Gaussian noise models and covariances
    auto g1 =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(this->noiseModel());
    auto g2 =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(other.noiseModel());
    if (!g1 || !g2)
      throw std::runtime_error(
          "NonlinearDensity::operator*: only Gaussian noise models are "
          "supported");

    const size_t n = this->dim();
    if (n != other.dim())
      throw std::invalid_argument(
          "NonlinearDensity::operator*: dimension mismatch");

    const Matrix Sigma1 = g1->covariance();
    const Matrix Sigma2 = g2->covariance();

    // 1) Choose a reference x̂. We use the naive-fusion reference in identity
    //    coordinates to remain generic across VALUE types.
    const Vector mu1_ref =
        traits<T>::Logmap(this->origin_);  // in chart at identity
    const Vector mu2_ref = traits<T>::Logmap(other.origin_);
    const Matrix SigmaRef = (Sigma1.inverse() + Sigma2.inverse()).inverse();
    const Vector muRef =
        SigmaRef * (Sigma1.inverse() * mu1_ref + Sigma2.inverse() * mu2_ref);
    const T xhat = traits<T>::Expmap(muRef);

    // 2) Express both densities at x̂ as extended concentrated Gaussians using Jacobians.
    // For density 1
    Matrix Hlp1, Hlq1; // d Local(xhat,q)/d p and d Local(xhat,q)/d q
    Vector r1 = traits<T>::Local(xhat, this->origin_, Hlp1, Hlq1);

    Matrix Hr_p1, Hr_v1; // d Retract(origin,delta)/d origin and /d delta at delta=0
    traits<T>::Retract(this->origin_, Vector::Zero(n), Hr_p1, Hr_v1);

    Matrix Jmap1 = Hlq1 * Hr_v1;  // chain rule mapping from δ at origin to chart at xhat
    Vector m1 = this->mean_.value_or(Vector::Zero(n));
    Vector mu1_hat = r1 + Jmap1 * m1;
    Matrix S1_hat = Jmap1 * Sigma1 * Jmap1.transpose();

    // For density 2
    Matrix Hlp2, Hlq2;
    Vector r2 = traits<T>::Local(xhat, other.origin_, Hlp2, Hlq2);
    Matrix Hr_p2, Hr_v2;
    traits<T>::Retract(other.origin_, Vector::Zero(n), Hr_p2, Hr_v2);
    Matrix Jmap2 = Hlq2 * Hr_v2;
    Vector m2 = other.mean_.value_or(Vector::Zero(n));
    Vector mu2_hat = r2 + Jmap2 * m2;
    Matrix S2_hat = Jmap2 * Sigma2 * Jmap2.transpose();

    // 3) Classical Gaussian fusion in the common tangent at x̂.
    const Matrix Sdiamond = (S1_hat.inverse() + S2_hat.inverse()).inverse();
    const Vector mu_plus =
        Sdiamond * (S1_hat.inverse() * mu1_hat + S2_hat.inverse() * mu2_hat);

    // 4) Reset: x⁺ = Retract(x̂, µ⁺); set zero-mean concentrated Gaussian with
    // Σ⁺ ≈ Sdiamond.
    const T xplus = traits<T>::Retract(xhat, mu_plus);
    const SharedNoiseModel modelPlus =
        noiseModel::Gaussian::Covariance(Sdiamond);

    return NonlinearDensity(this->key(), xplus, modelPlus);
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
