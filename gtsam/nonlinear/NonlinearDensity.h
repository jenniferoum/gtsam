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

// Simple, non-templated Gaussian fusion in a common tangent space
struct FusedGaussian {
  Vector mean;        // fused mean
  Matrix covariance;  // fused covariance
};

inline FusedGaussian FuseGaussians(const Vector& mu1, const Matrix& S1,
                                   const Vector& mu2, const Matrix& S2) {
  const Matrix iS1 = S1.inverse();
  const Matrix iS2 = S2.inverse();
  const Matrix S = (iS1 + iS2).inverse();
  const Vector m = S * (iS1 * mu1 + iS2 * mu2);
  return {m, S};
}

/**
 * A nonlinear density, inherits from NonlinearLikelihood. With Gaussian noise
 * models, models exactly a (left) extended concentrated Gaussian (L-ECG).
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

  /// Constructor with noise model and optional mean in tangent space
  NonlinearDensity(Key key, const VALUE& origin, const SharedNoiseModel& model,
                   const std::optional<Vector>& mean = {})
      : Base(key, origin, model, mean) {}

  /// Constructor with covariance matrix and optional mean in tangent space
  NonlinearDensity(Key key, const VALUE& origin, const Matrix& covariance,
                   const std::optional<Vector>& mean = {})
      : Base(key, origin, noiseModel::Gaussian::Covariance(covariance), mean) {}

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
   * Log-probability overload taking a Values container. This mirrors the
   * linear GaussianConditional interface so densities can be queried in a
   * uniform way when only a Values is available.
   */
  double logProbability(const Values& values) const {
    const T& x = values.at<T>(this->key());
    return logProbability(x);
  }

  /**
   * Evaluate the probability density at the given value.
   * P(x) = exp(logProbability(x)).
   */
  double evaluate(const T& x) const { return exp(logProbability(x)); }

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
   * For non-Gaussian noise models this is not (easily) defined and we throw.
   */
  double negLogConstant() const {
    const size_t n = this->dim();
    auto gaussian = getGaussian("negLogConstant");
    constexpr double log2pi = 1.8378770664093454835606594728112;  // log(2*pi)
    const double logDetSigma = gaussian->logDeterminant();        // log |Σ|
    return 0.5 * n * log2pi + 0.5 * logDetSigma;
  }

  /**
   * Fusion operator implementing the (approximate) three-step Fusion
   * method in:
   *   Y. Ge, P. van Goor and R. Mahony, "A Geometric Perspective on Fusing
   *   Gaussian Distributions on Lie Groups," in IEEE Control Systems Letters,
   *   vol. 8, pp. 844-849, 2024, https://ieeexplore.ieee.org/document/10539262
   *
   * We choose our origin as the reference, express other density in our chart,
   * fuse the Gaussians, then reset to a zero-mean concentrated Gaussian.
   *
   * Notes/assumptions:
   *  - Only supports Gaussian noise models; throws otherwise.
   *  - If both inputs share the same origin and mean, this reduces to
   *    classical Gaussian fusion: Σ⁺ = (Σ₁^{-1}+Σ₂^{-1})^{-1} at the same
   * origin.
   */
  NonlinearDensity operator*(const NonlinearDensity& other) const {
    // 0) Sanity checks
    if (this->key() != other.key())
      throw std::invalid_argument("NonlinearDensity::operator*: keys differ");

    // Extract Gaussian noise models and covariances
    auto g1 = getGaussian("operator*");
    auto g2 = other.getGaussian("operator*");

    const Matrix Sigma1 = g1->covariance();
    const Matrix Sigma2 = g2->covariance();

    // 1) Reference selection (info-weighted in identity chart)
    const Vector mu1_ref = traits<T>::Logmap(this->origin_);
    const Vector mu2_ref = traits<T>::Logmap(other.origin_);
    const Matrix SigRef = (Sigma1.inverse() + Sigma2.inverse()).inverse();
    const Vector muRef =
        SigRef * (Sigma1.inverse() * mu1_ref + Sigma2.inverse() * mu2_ref);
    const T xhat = traits<T>::Expmap(muRef);

    // 2) Map both ECGs to chart at x̂
    Vector mu1_hat, mu2_hat;
    Matrix S1_hat, S2_hat;
    // mapToReference_ removed

    // 3) Classical Gaussian fusion in the common tangent at x̂ (simple form)
    // We must map mu1_hat, S1_hat, mu2_hat, S2_hat before fusion: implement
    // here

    // Compute mapping for first density
    {
      const size_t n = Sigma1.rows();
      Matrix Hlp1, Hlq1;
      Vector r1 = traits<T>::Local(xhat, this->origin_, Hlp1, Hlq1);
      Matrix Hr_p, Hr_v1;
      traits<T>::Retract(this->origin_, Vector::Zero(n), Hr_p, Hr_v1);
      const Matrix Jmap1 = Hlq1 * Hr_v1;
      const Vector m1 = this->mean_.value_or(Vector::Zero(n));
      mu1_hat = r1 + Jmap1 * m1;
      S1_hat = Jmap1 * Sigma1 * Jmap1.transpose();
    }

    // Compute mapping for second density
    {
      const size_t n = Sigma2.rows();
      Matrix Hlp2, Hlq2;
      Vector r2 = traits<T>::Local(xhat, other.origin_, Hlp2, Hlq2);
      Matrix Hr_p2, Hr_v2;
      traits<T>::Retract(other.origin_, Vector::Zero(n), Hr_p2, Hr_v2);
      const Matrix Jmap2 = Hlq2 * Hr_v2;
      const Vector m2 = other.mean_.value_or(Vector::Zero(n));
      mu2_hat = r2 + Jmap2 * m2;
      S2_hat = Jmap2 * Sigma2 * Jmap2.transpose();
    }

    const FusedGaussian fg = FuseGaussians(mu1_hat, S1_hat, mu2_hat, S2_hat);

    // 4) Reset to zero-mean at fused origin (covariance form)
    return resetFrom(xhat, fg.mean, fg.covariance, /*isInformation=*/false);
  }

  /// Return T element corresponding to the mean, with optional Jacobian
  T retractMean(Matrix* xHm = nullptr) const {
    const size_t n = this->dim();
    const bool zeroMean = !this->mean_;
    if (xHm && zeroMean) xHm->setIdentity(n, n);
    return zeroMean
               ? this->origin_
               : traits<T>::Retract(this->origin_, *(this->mean_), {}, xHm);
  }

  /**
   * Transport this density to a new origin x̂, returning a density at x̂
   * with nonzero mean in that chart. Uses a full first-order Jacobian for the
   * change of coordinates between charts via the chain rule:
   *   J = ∂Local(x̂,x)/∂x · ∂Retract(origin,m)/∂m
   * @note: Only Gaussian noise models are supported.
   */
  NonlinearDensity transportTo(const T& x_hat) const {
    auto g = getGaussian("transportTo");

    Matrix xHm;
    const T x = retractMean(&xHm);

    Matrix hatHx;  // d Local(x̂,q)/d p and d Local(x̂,q)/d q
    Vector muHat = traits<T>::Local(x_hat, x, {}, hatHx);
    const Matrix hatJm = hatHx * xHm;  // chain rule
    const Matrix covHat = hatJm * g->covariance() * hatJm.transpose();

    return NonlinearDensity(this->key(), x_hat, covHat, muHat);
  }

  /**
   * Create a new NonlinearDensity with zero mean by moving the origin to x̂ =
   * Retract(origin, mean). Returns an ECG with origin=x̂, zero mean, and
   * covariance transported to x̂.
   * @note: Only Gaussian noise models are supported.
   */
  NonlinearDensity reset() const {
    if (!this->mean_) return *this;  // already zero-mean
    auto g = getGaussian("reset");

    Matrix hatJm;
    const T x_hat = retractMean(&hatJm);
    const Matrix covHat = hatJm * g->covariance() * hatJm.transpose();

    return NonlinearDensity(this->key(), x_hat, covHat);
  }

  /** Reset given a reference x̂ and a nonzero mean μ in its chart, returning
   *  a zero-mean ECG at x⁺ = Retract(x̂, μ). If isInformation=true, Σ_or_Λ is
   *  interpreted as precision Λ and we construct a Gaussian Information model.
   */
  NonlinearDensity resetFrom(const T& xhat, const Vector& mu,
                             const Matrix& Sigma_or_Lambda,
                             bool isInformation = false) const {
    const T xplus = traits<T>::Retract(xhat, mu);
    const SharedNoiseModel modelPlus =
        isInformation ? noiseModel::Gaussian::Information(Sigma_or_Lambda)
                      : noiseModel::Gaussian::Covariance(Sigma_or_Lambda);
    return NonlinearDensity(this->key(), xplus, modelPlus);
  }

  noiseModel::Gaussian::shared_ptr getGaussian(
      const std::string& method) const {
    using noiseModel::Gaussian;
    const auto& model = this->noiseModel();
    auto g = std::dynamic_pointer_cast<Gaussian>(model);
    if (!g)
      throw std::runtime_error("NonlinearDensity::" + method +
                               " is only implemented for  Gaussian noise "
                               "models. The noise model used is of type " +
                               std::string(typeid(*model).name()));
    return g;
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
