/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testWNOAInterpFactor.cpp
 * @brief   Unit test for WNOA Interpolation Factor
 * @author  Connor Holmes
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WNOAFactor.h>
#include <gtsam/nonlinear/WNOAInterpFactor.h>
#include <gtsam/nonlinear/WNOAInterpolator.h>
#include <gtsam/slam/BetweenFactor.h>

#include <set>
#include <unordered_map>
#include <unordered_set>

using namespace std;
using namespace gtsam;

// Symbol shorthands
using symbol_shorthand::P;
using symbol_shorthand::V;

static Vector Q_p3 = Vector3::Ones();
static Vector Q_se3 = Vector6::Ones();
static double timestep = 0.1;

/**** Point3 Test Variables*****/
// First pose, moving forward in x with velocity 1
Vector3 v0_p3(1.0, 0.0, 0.0);
Point3 p0_p3(0.0, 0.0, 0.0);
// define next two poses along same trajectory
Point3 p1_p3 = p0_p3 + timestep * v0_p3;
Vector3 v1_p3 = v0_p3;
Point3 p2_p3 = p0_p3 + 2 * timestep * v0_p3;
Vector3 v2_p3 = v0_p3;

/**** SE3 Test Variables *****/
// Define First Pose  with a general velocity
static Pose3 p0_se3 = Pose3::Expmap(Vector6(0.5, 0.0, 0.0, 0.0, 0.0, 0.0));
static Vector6 v0_se3(1, 0.0, 0.5, 0.1, 0.0, 0.0);
// Define Second Pose with same velocity (to get an error)
static Pose3 p1_se3 = p0_se3.expmap(timestep * v0_se3);
// Define Third Pose with same vel
static Pose3 p2_se3 = p0_se3.expmap(2 * timestep * v0_se3);
static Vector6 v2_se3 = v0_se3;
// Define Third Pose with same vel
static Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
// Define Third Pose with same vel
static Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);

// Define interpolation parameters
// Define estimated and interpolated states for testing
static set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                         StateData(P(2), V(2), 2 * timestep),
                                         StateData(P(3), V(3), 100 * timestep)};
static set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                            StateData(P(4), V(4), timestep)};

// STATE DATA TESTS

// Constructor
TEST(StateData, Constructor) { StateData(P(0), V(0), 0.0); }

// Ordered Set
TEST(StateData, OrderedSet) {
  // define set
  set<StateData> sd_set;
  // add to set in inverse order
  sd_set.insert(StateData(P(4), V(4), 3.0));
  sd_set.insert(StateData(P(3), V(3), 2.00001));
  sd_set.insert(StateData(P(2), V(2), 2.0));
  sd_set.insert(StateData(P(1), V(1), 1.0));
  sd_set.insert(StateData(P(0), V(0), 0.0));
  // verify the order of the set
  KeyVector key_order = {P(0), P(1), P(2), P(3), P(4)};
  int i = 0;
  for (auto state : sd_set) {
    cout << DefaultKeyFormatter(state.pose) << endl;
    EXPECT(state.pose == key_order[i]);
    i++;
  }
}

// hash map
TEST(StateData, UnorderedSet) {
  // define set
  unordered_set<StateData> sd_set;
  // add to set in inverse order
  sd_set.insert(StateData(P(2), V(2), 2.0));
  sd_set.insert(StateData(P(1), V(1), 1.0));
  sd_set.insert(StateData(P(0), V(0), 0.0));
  EXPECT(sd_set.size() == 3);
  // insert new state with existing pose and vel (but mixed)
  sd_set.insert(StateData(P(0), V(1), 0.0));
  EXPECT(sd_set.size() == 4);
  // insert new state with same pose and vel but diff time (should not add)
  sd_set.insert(StateData(P(0), V(0), 0.1));
  EXPECT(sd_set.size() == 4);
}

// Constructor test
TEST(WNOAInterp, Constructor) {
  // Create a factor
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto prior = std::make_shared<PriorFactor<Point3>>(P(1), p0_p3, model);

  // wrap factor
  const auto factor = WNOAInterpFactor<Point3>(prior, estimatedStates,
                                               interpolatedStates, Q_p3);
  // get factor keys
  KeyVector outer_keys = factor.keys();
  // Make sure keys defined properly
  CHECK(outer_keys.size() == 4);
  CHECK(find(outer_keys.begin(), outer_keys.end(), P(0)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), P(2)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), V(0)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), V(2)) != outer_keys.end());
}

TEST(WNOAInterp, Print) {
  // Create a factor
  Point3 priorValue(0.5, 0.0, 0.0);
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = std::make_shared<PriorFactor<Point3>>(P(1), priorValue, model);

  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, estimatedStates,
                                               interpolatedStates, Q_p3);

  factor.print();
  cout << endl;
}

TEST(WNOAInterp, EvalErrorP3Unary) {
  // SAME POSE CASE
  // Create a prior factor and interpolated version
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  // prior at first pose
  auto prior = std::make_shared<PriorFactor<Point3>>(P(1), p0_p3, model);
  auto factor = std::make_shared<WNOAInterpFactor<Point3>>(
      prior, estimatedStates, interpolatedStates, Q_p3);
  Values values;
  values.insert(P(0), p0_p3);
  values.insert(P(2), p0_p3);
  values.insert(V(0), Vector3::Zero().eval());  // zero vel
  values.insert(V(2), Vector3::Zero().eval());
  // compute residuals
  auto residual1 = factor->unwhitenedError(values);
  auto res_actual = Vector3::Zero().eval();
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));

  // DIFFERENT POSE, ZERO VELOCITY
  // prior at pose 1 (between 0 and 2)
  prior = std::make_shared<PriorFactor<Point3>>(P(1), p1_p3, model);
  factor = std::make_shared<WNOAInterpFactor<Point3>>(prior, estimatedStates,
                                                      interpolatedStates, Q_p3);
  values.update(P(0), p0_p3);
  values.update(P(2), p2_p3);
  values.update(V(0), Vector3::Zero().eval());
  values.update(V(2), Vector3::Zero().eval());
  // Evaluate
  residual1 = factor->unwhitenedError(values);
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));

  // DIFFERENT POSE, WITH VELOCITY
  // prior at pose 1 (between 0 and 2)
  prior = std::make_shared<PriorFactor<Point3>>(P(1), p1_p3, model);
  factor = std::make_shared<WNOAInterpFactor<Point3>>(prior, estimatedStates,
                                                      interpolatedStates, Q_p3);
  values.update(P(0), p0_p3);
  values.update(P(2), p2_p3);
  values.update(V(0), v0_p3);
  values.update(V(2), v0_p3);
  // Evaluate
  residual1 = factor->unwhitenedError(values);
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3UnaryPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto prior_pose =
      std::make_shared<PriorFactor<Pose3>>(P(1), p1_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor_pose = WNOAInterpFactor<Pose3>(prior_pose, estimatedStates,
                                                   interpolatedStates, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Check pose residuals
  const auto res_zero = Vector6::Zero();
  auto residual = factor_pose.unwhitenedError(values);  // new
  CHECK(assert_equal(residual, res_zero, 1e-12));
  residual = prior_pose->evaluateError(p1_se3);  // original
  CHECK(assert_equal(residual, res_zero, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3BetweenPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  // construct relative pose
  const Pose3 p01_se3 = p0_se3.inverse().compose(p1_se3);
  const auto between_factor =
      std::make_shared<BetweenFactor<Pose3>>(P(0), P(1), p01_se3, model);
  // Construct factor for interpolated pose and velocity
  // NOTE: we set the noise model to be fixed to avoid Jacobian computations.
  const auto factor = WNOAInterpFactor<Pose3>(between_factor, estimatedStates,
                                              interpolatedStates, Q_se3, true);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Check pose residuals

  auto res_btwn = between_factor->evaluateError(p0_se3, p1_se3);  // original
  auto res_interp = factor.unwhitenedError(values);               // new
  CHECK(assert_equal(res_btwn, res_interp, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3BtwnInterp) {
  // Same as between above, but using two interpolated states with different
  // boundaries
  set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                    StateData(P(2), V(2), 2 * timestep),
                                    StateData(P(4), V(4), 4 * timestep)};
  set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                       StateData(P(3), V(3), 3 * timestep)};
  const Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
  const Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);
  Vector6 v2_se3 = v0_se3;
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto between_factor = std::make_shared<BetweenFactor<Pose3>>(
      P(1), P(3), Pose3::Identity(), model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Pose3>(between_factor, estimatedStates,
                                              interpolatedStates, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);
  values.insert(V(4), v0_se3);

  // Check that residual is the same when we interpolate
  auto res_btwn = between_factor->evaluateError(p1_se3, p3_se3);  // original
  auto res_btwn_interp = factor.unwhitenedError(values);          // new
  // Check that residuals match between
  CHECK(assert_equal(res_btwn, res_btwn_interp, 1e-12));
}

/* *************************************************************************
 */
TEST(WNOAInterp, JacobianPoint3UnaryPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto prior_factor =
      std::make_shared<PriorFactor<Point3>>(P(1), p1_p3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Point3>(prior_factor, estimatedStates,
                                               interpolatedStates, Q_p3);

  // Set up values
  Values values;
  values.insert(P(0), p0_p3);
  values.insert(P(2), p2_p3);
  values.insert(V(0), v0_p3);
  values.insert(V(2), v2_p3);

  // Create container for Jacobians
  vector<Matrix> H(factor.keys().size());
  // Get residual and Jacobian
  factor.unwhitenedError(values, &H);

  unordered_map<Key, Matrix> Jacs;
  KeyVector factor_keys = factor.keys();
  for (uint i = 0; i < factor_keys.size(); i++) {
    Jacs[factor_keys[i]] = H[i];
  }

  // define lambda function for derivatives
  auto f = [&](auto& p0, auto& v0, auto& p2, auto& v2) {
    Values values;
    values.insert(P(0), p0);
    values.insert(P(2), p2);
    values.insert(V(0), v0);
    values.insert(V(2), v2);
    return factor.unwhitenedError(values);
  };

  // Compute numerical derivatives
  double delta = 1e-6;
  Matrix J_p0_num =
      numericalDerivative41<Vector, Point3, Vector3, Point3, Vector3>(
          f, p0_p3, v0_p3, p2_p3, v2_p3, delta);
  Matrix J_v0_num =
      numericalDerivative42<Vector, Point3, Vector3, Point3, Vector3>(
          f, p0_p3, v0_p3, p2_p3, v2_p3, delta);
  Matrix J_p2_num =
      numericalDerivative43<Vector, Point3, Vector3, Point3, Vector3>(
          f, p0_p3, v0_p3, p2_p3, v2_p3, delta);
  Matrix J_v2_num =
      numericalDerivative44<Vector, Point3, Vector3, Point3, Vector3>(
          f, p0_p3, v0_p3, p2_p3, v2_p3, delta);

  double tol = 1e-4;
  EXPECT(assert_equal(Jacs[P(0)], J_p0_num, tol));
  EXPECT(assert_equal(Jacs[V(0)], J_v0_num, tol));
  EXPECT(assert_equal(Jacs[P(2)], J_p2_num, tol));
  EXPECT(assert_equal(Jacs[V(2)], J_v2_num, tol));
}

/* *************************************************************************
 */
TEST(WNOAInterp, JacobianSE3UnaryPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto prior_factor =
      std::make_shared<PriorFactor<Pose3>>(P(1), p1_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Pose3>(prior_factor, estimatedStates,
                                              interpolatedStates, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Create container for Jacobians
  vector<Matrix> H(factor.keys().size());
  // Get residual and Jacobian
  factor.unwhitenedError(values, &H);

  unordered_map<Key, Matrix> Jacs;
  KeyVector factor_keys = factor.keys();
  for (uint i = 0; i < factor_keys.size(); i++) {
    Jacs[factor_keys[i]] = H[i];
  }

  // define lambda function for derivatives
  auto f = [&](auto& p0, auto& v0, auto& p2, auto& v2) {
    Values values;
    values.insert(P(0), p0);
    values.insert(P(2), p2);
    values.insert(V(0), v0);
    values.insert(V(2), v2);
    return factor.unwhitenedError(values);
  };

  // Compute numerical derivatives
  double delta = 1e-6;
  Matrix J_p0_num =
      numericalDerivative41<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_v0_num =
      numericalDerivative42<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_p2_num =
      numericalDerivative43<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_v2_num =
      numericalDerivative44<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);

  double tol = 1e-3;
  EXPECT(assert_equal(Jacs[P(0)], J_p0_num, tol));
  EXPECT(assert_equal(Jacs[V(0)], J_v0_num, tol));
  EXPECT(assert_equal(Jacs[P(2)], J_p2_num, tol));
  EXPECT(assert_equal(Jacs[V(2)], J_v2_num, tol));
}

/* *************************************************************************
 */
TEST(WNOAInterp, PrecomputeLambdaPsiUnarySe3) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto prior_factor =
      std::make_shared<PriorFactor<Pose3>>(P(1), p1_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Pose3>(
      prior_factor, estimatedStates, interpolatedStates, Q_se3, false, false);
  // factor with precomputed interpolation matrices
  const auto factor_alt = WNOAInterpFactor<Pose3>(
      prior_factor, estimatedStates, interpolatedStates, Q_se3, false, true);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Create container for Jacobians
  vector<Matrix> H(factor.keys().size());
  vector<Matrix> H_alt(factor.keys().size());
  // Get residual and Jacobian
  auto residual = factor.unwhitenedError(values, &H);
  auto residual_alt = factor_alt.unwhitenedError(values, &H_alt);
  // check residuals
  EXPECT(assert_equal(residual, residual_alt));
  // check that derivatives are identical
  int index = 0;
  for (auto& mat : H) {
    EXPECT(assert_equal(mat, H_alt[index]));
    index++;
  }
}

/* *************************************************************************
 */

TEST(WNOAInterp, Interpolator) {
  // Create Interpolator
  Interpolator<Pose3> interpolator(Q_se3);

  // Get analytic Jacobians
  vector<Matrix> H(8);
  interpolator.interpolatePoseAndVelocity(
      TimestampedPoseVelocity<Pose3>(p0_se3, v0_se3, 0.0),
      TimestampedPoseVelocity<Pose3>(p2_se3, v2_se3, 2 * timestep), timestep,
      &H);

  // define lambda function for derivatives
  auto f = [&](auto& p0, auto& v0, auto& p2, auto& v2) {
    auto [pose, vel] = interpolator.interpolatePoseAndVelocity(
        TimestampedPoseVelocity<Pose3>(p0, v0, 0.0),
        TimestampedPoseVelocity<Pose3>(p2, v2, 2 * timestep), timestep);

    return p1_se3.logmap(pose);
  };

  // Compute numerical derivatives
  double delta = 1e-6;
  Matrix J_p0_num =
      numericalDerivative41<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_v0_num =
      numericalDerivative42<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_p2_num =
      numericalDerivative43<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);
  Matrix J_v2_num =
      numericalDerivative44<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p2_se3, v2_se3, delta);

  double tol = 1e-3;
  EXPECT(assert_equal(H[0], J_p0_num, tol));
  EXPECT(assert_equal(H[1], J_v0_num, tol));
  EXPECT(assert_equal(H[2], J_p2_num, tol));
  EXPECT(assert_equal(H[3], J_v2_num, tol));
}

/* *************************************************************************
 */

TEST(WNOAInterp, NoiseModelSE3Unary) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto prior_factor =
      std::make_shared<PriorFactor<Pose3>>(P(1), p1_se3, model);
  auto cov_prior = model->covariance();
  // Factor with changing measurement noise
  const auto factor = WNOAInterpFactor<Pose3>(prior_factor, estimatedStates,
                                              interpolatedStates, Q_se3);
  // Factor with fixed meas noise
  const auto factor_fixed = WNOAInterpFactor<Pose3>(
      prior_factor, estimatedStates, interpolatedStates, Q_se3, true);
  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);
  // Call error functions
  auto cov = factor.noiseModel(values)->covariance();
  auto cov_fixed = factor_fixed.noiseModel(values)->covariance();
  // Verify that the covariance has changed from the prior
  EXPECT(assert_equal(cov_fixed, cov_prior));
  EXPECT(assert_inequal(cov, cov_prior));
}

/* *************************************************************************
 */

TEST(WNOAInterp, NoiseModelSE3Btwn) {
  // Same as between above, but using two interpolated states with different
  // boundaries
  set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                    StateData(P(2), V(2), 2 * timestep),
                                    StateData(P(4), V(4), 4 * timestep)};
  set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                       StateData(P(3), V(3), 3 * timestep)};
  const Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
  const Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);
  Vector6 v2_se3 = v0_se3;
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto cov_inner = model->covariance();
  const auto between_factor = std::make_shared<BetweenFactor<Pose3>>(
      P(1), P(3), Pose3::Identity(), model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Pose3>(between_factor, estimatedStates,
                                              interpolatedStates, Q_se3);
  const auto factor_fixed = WNOAInterpFactor<Pose3>(
      between_factor, estimatedStates, interpolatedStates, Q_se3, true);
  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);
  values.insert(V(4), v0_se3);

  // Call error functions
  auto cov = factor.noiseModel(values)->covariance();
  auto cov_fixed = factor_fixed.noiseModel(values)->covariance();
  // Verify that the covariance has changed from the prior
  EXPECT(assert_equal(cov_fixed, cov_inner));
  EXPECT(assert_inequal(cov, cov_inner));
}

/* *************************************************************************
 */

TEST(WNOAInterp, NoiseModelP3Btwn) {
  // Same as between above, but using two interpolated states with different
  // boundaries
  set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                    StateData(P(2), V(2), 2 * timestep),
                                    StateData(P(4), V(4), 4 * timestep)};
  set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                       StateData(P(3), V(3), 3 * timestep)};
  // const Point3 p3_p3 = p0_p3 + 3 * timestep * v0_p3;
  const Point3 p4_p3 = p0_p3 + 4 * timestep * v0_p3;
  Vector3 v2_p3 = v0_p3;
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto cov_inner = model->covariance();
  const auto between_factor = std::make_shared<BetweenFactor<Point3>>(
      P(1), P(3), Point3::Identity(), model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Point3>(between_factor, estimatedStates,
                                               interpolatedStates, Q_p3);

  // Set up values
  Values values;
  values.insert(P(0), p0_p3);
  values.insert(P(2), p2_p3);
  values.insert(P(4), p4_p3);
  values.insert(V(0), v0_p3);
  values.insert(V(2), v2_p3);
  values.insert(V(4), v0_p3);

  // Get modified noise model
  auto cov = factor.noiseModel(values)->covariance();
  // subtract original covariance
  Matrix cov_diff = cov - cov_inner;
  // verify same as 2x interpolated conditional covariance
  // Note: Jacobians for Point3 are identity
  auto interpolator = Interpolator<Point3>(Q_p3);
  auto Sigma_tau = interpolator.computeConditionalCov(
      TimestampedPoseVelocity<Point3>(p0_p3, v0_p3, 0.0),
      TimestampedPoseVelocity<Point3>(p2_p3, v2_p3, 2 * timestep),
      TimestampedPoseVelocity<Point3>(p1_p3, v1_p3, timestep));
  EXPECT(assert_equal(cov_diff, 2 * Sigma_tau.block<3, 3>(0, 0)));
}

/* *************************************************************************
 */

TEST(WNOAInterp, LinearizeSE3Btwn) {
  // Same as between above, but using two interpolated states with different
  // boundaries
  set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                    StateData(P(2), V(2), 2 * timestep),
                                    StateData(P(4), V(4), 4 * timestep)};
  set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                       StateData(P(3), V(3), 3 * timestep)};
  const Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
  const Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);
  Vector6 v2_se3 = v0_se3;
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto between_factor = std::make_shared<BetweenFactor<Pose3>>(
      P(1), P(3), p1_se3.inverse() * p3_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor = WNOAInterpFactor<Pose3>(between_factor, estimatedStates,
                                              interpolatedStates, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);
  values.insert(V(4), v0_se3);

  // Call Linearize function
  auto lin_factor =
      dynamic_pointer_cast<JacobianFactor>(factor.linearize(values));
  Vector6 error = lin_factor->getb();
  EXPECT(assert_equal(error, Vector6::Zero().eval()));
}

/* *************************************************************************
 */

TEST(WNOAInterp, SE3OptimTest) {
  // Define optimization:
  //       unary
  //         |
  //  0 ---- 1 ---- 2 ----- 3 ----- 4
  //  e      i      e       i       e
  //          --- between ---
  set<StateData> estimatedStates = {StateData(P(0), V(0), 0.0),
                                    StateData(P(2), V(2), 2 * timestep),
                                    StateData(P(4), V(4), 4 * timestep)};
  set<StateData> interpolatedStates = {StateData(P(1), V(1), timestep),
                                       StateData(P(3), V(3), 3 * timestep)};
  const Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
  const Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);
  // Define nominal factors
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto between_factor = std::make_shared<BetweenFactor<Pose3>>(
      P(1), P(3), p1_se3.inverse() * p3_se3, model);
  const auto prior_pose_factor =
      std::make_shared<PriorFactor<Pose3>>(P(1), p1_se3, model);
  const auto prior_vel_factor =
      std::make_shared<PriorFactor<Vector6>>(V(1), v0_se3, model);

  // Define graph
  NonlinearFactorGraph graph;

  // Construct interpolated factors
  graph.add(WNOAInterpFactor<Pose3>(prior_pose_factor, estimatedStates,
                                    interpolatedStates, Q_se3));
  graph.add(WNOAInterpFactor<Pose3>(prior_vel_factor, estimatedStates,
                                    interpolatedStates, Q_se3));
  graph.add(WNOAInterpFactor<Pose3>(between_factor, estimatedStates,
                                    interpolatedStates, Q_se3));

  // Add WNOA factors
  graph.add(
      WNOAMotionFactor<Pose3>(P(0), V(0), P(2), V(2), 2 * timestep, Q_se3));
  graph.add(
      WNOAMotionFactor<Pose3>(P(2), V(2), P(4), V(4), 2 * timestep, Q_se3));

  // Set up values at ground truth solution
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);  // Same velocity for all points
  values.insert(V(2), v0_se3);
  values.insert(V(4), v0_se3);

  // Set up optimizer
  GaussNewtonOptimizer optimizer(graph, values);
  // We expect the initial to be zero because config is the ground truth
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);
  // Iterate once, and the config should not have changed
  optimizer.iterate();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);
  // Complete solution
  optimizer.optimize();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-6);

  // perturb solution and converge again
  Values values_pert;
  values_pert.insert(
      P(0), p0_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));

  values_pert.insert(
      P(2), p2_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));

  values_pert.insert(
      P(4), p4_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));
  values_pert.insert(V(0), v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) *
                                        0.1);  // Same velocity for all points
  values_pert.insert(V(2),
                     v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) * 0.1);
  values_pert.insert(V(4),
                     v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) * 0.1);

  // Set up optimizer
  GaussNewtonOptimizer optimizer2(graph, values_pert);
  // Check that we converge to solution
  optimizer2.optimize();
  DOUBLES_EQUAL(0.0, optimizer2.error(), 1e-4);
}

TEST(WNOAInterp, SE3InterpGraph) {
  // Test automatic interpolation
  // Define optimization:
  //       unary
  //         |
  //  0 ---- 1 ---- 2 ----- 3 ----- 4
  //  e      i      e       i       e
  //          --- between ---

  // Define nominal factors
  const auto model = noiseModel::Diagonal::Sigmas(Vector6::Ones());
  const auto between_factor =
      BetweenFactor<Pose3>(P(1), P(3), p1_se3.inverse() * p3_se3, model);
  const auto prior_pose_factor = PriorFactor<Pose3>(P(1), p1_se3, model);
  const auto prior_vel_factor = PriorFactor<Vector6>(V(1), v0_se3, model);

  // Generate original graph
  NonlinearFactorGraph graph;
  graph.add(between_factor);
  graph.add(prior_pose_factor);
  graph.add(prior_vel_factor);

  // Interpolate the graph
  set<StateData> estimatedStatesShuffled = {
      StateData(P(0), V(0), 0.0), StateData(P(4), V(4), 4 * timestep),
      StateData(P(2), V(2), 2 * timestep)};
  set<StateData> interpolatedStatesShuffled = {
      StateData(P(3), V(3), 3 * timestep), StateData(P(1), V(1), timestep)};
  auto new_graph = interpolateFactorGraph<Pose3>(
      graph, estimatedStatesShuffled, interpolatedStatesShuffled, Q_se3);

  // Set up values at ground truth solution
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);  // Same velocity for all points
  values.insert(V(2), v0_se3);
  values.insert(V(4), v0_se3);
  // Set up optimizer
  GaussNewtonOptimizer optimizer(new_graph, values);
  // We expect the initial to be zero because config is the ground truth
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);
  // Iterate once, and the config should not have changed
  optimizer.iterate();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);
  // Complete solution
  Values result = optimizer.optimize();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-6);

  // Test value interpolation
  Values result_interp =
      updateInterpValues<Pose3>(new_graph, result, estimatedStatesShuffled,
                                interpolatedStatesShuffled, Q_se3);

  auto p3_se3_est = result_interp.at<Pose3>(P(3));
  auto p1_se3_est = result_interp.at<Pose3>(P(1));

  EXPECT(assert_equal(p3_se3, p3_se3_est, 1e-6));
  EXPECT(assert_equal(p1_se3, p1_se3_est, 1e-6));

  // perturb solution and converge again
  Values values_pert;
  values_pert.insert(
      P(0), p0_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));
  values_pert.insert(
      P(2), p2_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));
  values_pert.insert(
      P(4), p4_se3.expmap(Vector6(0.001, 0.001, 0.001, 0.1, 0.1, 0.1)));
  values_pert.insert(V(0),
                     v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) * 0.1);
  // Same velocity for all points
  values_pert.insert(V(2),
                     v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) * 0.1);
  values_pert.insert(V(4),
                     v0_se3 + Vector6(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) * 0.1);
  // Set up optimizer
  GaussNewtonOptimizer optimizer2(new_graph, values_pert);
  // Check that we converge to solution
  optimizer2.optimize();
  DOUBLES_EQUAL(0.0, optimizer2.error(), 1e-4);
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}