/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testNonlinearDensity.cpp
 * @date September 30, 2025
 * @author Frank Dellaert
 * @brief Unit tests for NonlinearDensity
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearDensity.h>

using namespace gtsam;

//******************************************************************************
TEST(NonlinearDensity, Pose2) {
  Key key(1);
  Pose2 origin(1, 2, 0.3);
  SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.2, 0.3));
  NonlinearDensity<Pose2> factor(key, origin, model);

  // Test error
  Pose2 x = origin;
  Values values;
  values.insert(key, x);
  EXPECT_DOUBLES_EQUAL(0.0, factor.error(values), 1e-9);

  // Test logProbability and evaluate
  double expected_log_prob =
      -0.5 * 3 * log(2 * M_PI) + log(1.0 / (0.1 * 0.2 * 0.3));
  EXPECT_DOUBLES_EQUAL(expected_log_prob, factor.logProbability(x), 1e-9);
  EXPECT_DOUBLES_EQUAL(expected_log_prob, factor.logProbability(values), 1e-9);
  EXPECT_DOUBLES_EQUAL(exp(expected_log_prob), factor.evaluate(x), 1e-9);
  EXPECT_DOUBLES_EQUAL(exp(expected_log_prob), factor.evaluate(values), 1e-9);

  // Test with non-Gaussian noise model
  SharedNoiseModel robust_model = noiseModel::Robust::Create(
      noiseModel::mEstimator::Huber::Create(1.345), model);
  NonlinearDensity<Pose2> robust_factor(key, origin, robust_model);
  CHECK_EXCEPTION(robust_factor.logProbability(x), std::runtime_error);

  // Compare value vs values interface consistency
  EXPECT_DOUBLES_EQUAL(factor.logProbability(x), factor.logProbability(values),
                       1e-9);
  EXPECT_DOUBLES_EQUAL(factor.evaluate(x), factor.evaluate(values), 1e-9);
}

//******************************************************************************
TEST(NonlinearDensity, Pose2WithMean) {
  Key key(1);
  Pose2 origin(1, 2, 0.3);
  SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.2, 0.3));

  Pose2 x = origin;
  Values values;
  values.insert(key, x);

  // Non-zero mean in tangent space (dx, dy, dtheta)
  Vector3 mean(0.05, -0.02, 0.1);
  NonlinearDensity<Pose2> factor(key, origin, model, mean);

  // Compute expected error = 0.5 * ||-mean||^2_{Sigma^{-1}}
  Vector sigmas(3);
  sigmas << 0.1, 0.2, 0.3;
  Vector invSigmas = sigmas.cwiseInverse();
  double mahalanobis =
      (-mean.cwiseProduct(invSigmas)).squaredNorm();  // (r .* invSigmas)^2 sum
  double expected_error = 0.5 * mahalanobis;
  EXPECT_DOUBLES_EQUAL(expected_error, factor.error(values), 1e-9);

  // Normalization constant (same as zero-mean case)
  double logDetSigma =
      2 * log(0.1) + 2 * log(0.2) + 2 * log(0.3);  // log(σ^2) per component sum
  double expected_log_k = -0.5 * 3 * log(2 * M_PI) - 0.5 * logDetSigma;
  double expected_log_prob = expected_log_k - expected_error;
  EXPECT_DOUBLES_EQUAL(expected_log_prob, factor.logProbability(x), 1e-9);
  EXPECT_DOUBLES_EQUAL(expected_log_prob, factor.logProbability(values), 1e-9);
  EXPECT_DOUBLES_EQUAL(exp(expected_log_prob), factor.evaluate(x), 1e-9);
  EXPECT_DOUBLES_EQUAL(exp(expected_log_prob), factor.evaluate(values), 1e-9);

  // Compare value vs values interface consistency
  EXPECT_DOUBLES_EQUAL(factor.logProbability(x), factor.logProbability(values),
                       1e-9);
  EXPECT_DOUBLES_EQUAL(factor.evaluate(x), factor.evaluate(values), 1e-9);
}

//******************************************************************************
TEST(NonlinearDensity, FusionPose2Identical) {
  Key key(1);
  Pose2 origin(1.0, 2.0, 0.3);
  Matrix3 Sigma;
  Sigma << 0.04, 0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0,
      0.16;  // covariances (not sigmas)
  SharedNoiseModel model = noiseModel::Gaussian::Covariance(Sigma);

  NonlinearDensity<Pose2> a(key, origin, model);
  NonlinearDensity<Pose2> b(key, origin, model);

  NonlinearDensity<Pose2> fused = a * b;

  // Expect origin unchanged for identical inputs
  EXPECT(assert_equal(origin, fused.origin()));

  // Covariance should be halved: (Σ^{-1}+Σ^{-1})^{-1} = Σ/2
  auto gf = std::dynamic_pointer_cast<noiseModel::Gaussian>(fused.noiseModel());
  CHECK(gf);
  Matrix3 Sigma_f = gf->covariance();
  EXPECT(assert_equal<Matrix3>(0.5 * Sigma, Sigma_f, 1e-9));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
