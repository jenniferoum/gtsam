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

#include <gtsam/base/utilities.h>
#include <gtsam/nonlinear/NonlinearLikelihood.h>
#include <gtsam/nonlinear/Values.h>

#include <random>  // for std::mt19937_64

namespace gtsam {

/**
 * A nonlinear density, inherits from NonlinearLikelihood. With Gaussian noise
 * models, models exactly a (left) extended concentrated Gaussian (L-ECG).
 * @ingroup nonlinear
 */
template <class T>
class NonlinearDensity : public NonlinearLikelihood<T> {
 public:
  using Base = NonlinearLikelihood<T>;
  using Gaussian = typename Base::Gaussian;

  /// @name Standard Constructors
  /// @{

  /// Default constructor for serialization.
  NonlinearDensity() {}

  /// Constructor with noise model and optional mean in tangent space
  NonlinearDensity(Key key, const T& origin, const SharedNoiseModel& model,
                   const std::optional<Vector>& mean = {})
      : Base(key, origin, model, mean) {}

  /// Constructor with covariance matrix and optional mean in tangent space
  NonlinearDensity(Key key, const T& origin, const Matrix& covariance,
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
   * Calculate the normalization constant for the density.
   * For a Gaussian noise model with covariance Σ, we return
   *   - log k = 0.5 * n * log(2*pi) + 0.5 * log |Σ|
   * where n = dim().  Note: gaussian->logDeterminant() returns log|Σ|.
   * For non-Gaussian noise models this is not (easily) defined and we throw.
   */
  double negLogConstant() const {
    const size_t n = this->dim();
    auto gaussian = this->gaussianModel("negLogConstant", /* throw */ true);
    constexpr double log2pi = 1.8378770664093454835606594728112;  // log(2*pi)
    const double logDetSigma = gaussian->logDeterminant();        // log |Σ|
    return 0.5 * n * log2pi + 0.5 * logDetSigma;
  }

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
    return exp(logProbability(values));
  }

  /// @}
  /// @name Transport and Fusion
  /// @{

  /**
   * Create a new NonlinearDensity with zero mean by moving the origin to x̂ =
   * Retract(origin, mean). Returns an ECG with origin=x̂, zero mean, and
   * covariance transported to x̂.
   * @note: Only Gaussian noise models are supported.
   */
  NonlinearDensity reset() const {
    if (!this->mean_) return *this;  // already zero-mean
    auto g = this->gaussianModel("reset", /* throw */ true);

    Matrix hatJm;
    const T x_hat = retractMean(&hatJm);
    const Matrix covHat = hatJm * g->covariance() * hatJm.transpose();

    return NonlinearDensity(this->key(), x_hat, Symmetrize(covHat));
  }

  /**
   * Transport this density to a new origin x̂, returning a density at x̂
   * with nonzero mean in that chart. Uses a full first-order Jacobian for the
   * change of coordinates between charts via the chain rule:
   *   J = ∂Local(x̂,x)/∂x · ∂Retract(origin,m)/∂m
   * @note: Only Gaussian noise models are supported.
   */
  NonlinearDensity transportTo(const T& x_hat) const {
    auto g = this->gaussianModel("transportTo", /* throw */ true);

    Matrix xHm;  // ∂Retract(origin,m)/∂m
    const T x = retractMean(&xHm);

    Matrix hatHx;  // ∂Local(x̂,x)/∂x
    Vector muHat = traits<T>::Local(x_hat, x, {}, hatHx);
    const Matrix hatJm = hatHx * xHm;  // chain rule
    const Matrix covHat = hatJm * g->covariance() * hatJm.transpose();

    return NonlinearDensity(this->key(), x_hat, Symmetrize(covHat), muHat);
  }

  /**
   * Fusion operator implementing the (approximate) three-step Fusion
   * method in:
   *   Y. Ge, P. van Goor and R. Mahony, "A Geometric Perspective on Fusing
   *   Gaussian Distributions on Lie Groups," in IEEE Control Systems Letters,
   *   vol. 8, pp. 844-849, 2024, https://ieeexplore.ieee.org/document/10539262
   *
   * We choose this->origin_ as the reference, express other density in our
   * chart, fuse the Gaussians, then reset to a zero-mean concentrated Gaussian.
   *
   * Notes/assumptions:
   *  - Only supports Gaussian noise models; throws otherwise.
   *  - If both inputs share the same origin, this reduces to
   *    classical Gaussian fusion: Σ⁺ = (Σ₁^{-1}+Σ₂^{-1})^{-1} at the same
   * origin.
   */
  NonlinearDensity operator*(const NonlinearDensity& other) const {
    // 0) Sanity checks
    if (this->key() != other.key())
      throw std::invalid_argument("NonlinearDensity::operator*: keys differ");
    if (this->dim() != other.dim())
      throw std::invalid_argument(
          "NonlinearDensity::operator*: dimension mismatch");

    // 1) Transport other to our chart
    NonlinearDensity o = other.transportTo(this->origin_);

    // 2) Fuse the Gaussians in our tangent space
    const auto g1 = this->gaussian("operator*", /* throw */ true);
    const auto g2 = o.gaussian("operator*", /* throw */ true);
    auto [m, P] = Fuse(*g1, *g2);

    // 3) Create a fused NonlinearDensity at our origin with fused mean
    NonlinearDensity ecg(this->key(), this->origin_, P, m);

    // 4) Reset to zero mean
    return ecg.reset();
  }

  /// @}

 private:
  /// Helper function to symmetrize a covariance matrix.
  static Matrix Symmetrize(const Matrix& matrix) {
    return 0.5 * (matrix + matrix.transpose());
  }

  /// Fuse two Gaussian distributions into a single Gaussian.
  static Gaussian Fuse(const Gaussian& g1, const Gaussian& g2) {
    const Matrix& P1 = g1.second;
    const Matrix& P2 = g2.second;
    const Vector& m1 = g1.first;
    const Vector& m2 = g2.first;

    const Matrix W1 = P1.inverse();
    const Matrix W2 = P2.inverse();
    const Matrix P = (W1 + W2).inverse();
    const Vector m = P * (W1 * m1 + W2 * m2);
    return {m, Symmetrize(P)};
  }

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
