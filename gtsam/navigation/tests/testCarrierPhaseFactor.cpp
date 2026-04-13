/**
 *  @file   testCarrierPhaseFactor.cpp
 *  @brief  Unit tests for CarrierPhaseFactor and CarrierPhaseFactorArm
 *  @date   March 23, 2026
 **/

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/CarrierPhaseFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <cmath>

using namespace gtsam;

/// GPS L1 wavelength (~0.1903 m)
static constexpr double LAMBDA_L1 = 0.190293672798365;
/// Speed of light
static constexpr double CLIGHT = 299792458.0;

/// Test helper: compute ECEF-to-ENU transform from geodetic origin (WGS84).
static Pose3 makeEcefTnav(double lat_deg, double lon_deg, double h) {
  constexpr double deg2rad = M_PI / 180.0;
  const double lat = lat_deg * deg2rad;
  const double lon = lon_deg * deg2rad;
  const double slat = std::sin(lat), clat = std::cos(lat);
  const double slon = std::sin(lon), clon = std::cos(lon);

  Matrix3 R;
  R.col(0) = Vector3(-slon, clon, 0.0);
  R.col(1) = Vector3(-slat * clon, -slat * slon, clat);
  R.col(2) = Vector3(clat * clon, clat * slon, slat);

  constexpr double a = 6378137.0;
  constexpr double f = 1.0 / 298.257223563;
  const double e2 = 2.0 * f - f * f;
  const double N = a / std::sqrt(1.0 - e2 * slat * slat);
  const Point3 t((N + h) * clat * clon,
                 (N + h) * clat * slon,
                 (N * (1.0 - e2) + h) * slat);
  return Pose3(Rot3(R), t);
}

// *************************************************************************
// CarrierPhaseFactor tests (all quantities in meters)
// *************************************************************************
TEST(TestCarrierPhaseFactor, Constructor) {
  // All zeros: error = 0 + 0 - 0 + 0 - 0 = 0
  const auto factor =
      CarrierPhaseFactor(Key(0), Key(1), Key(2), 0.0, Point3::Zero(), 0.0);
  const double error = factor.evaluateError(Point3::Zero(), 0.0, 0.0)[0];
  EXPECT_DOUBLES_EQUAL(0.0, error, 1e-9);
}

// *************************************************************************
TEST(TestCarrierPhaseFactor, AmbiguityEffect) {
  // error = range + clock - sat_clock + ambiguity - measurement
  // With range = measurement and clock = sat_clock = 0:
  //   error = ambiguity
  const Point3 satPos(0.0, 0.0, 20200000.0);
  const Point3 recvPos(0.0, 0.0, 0.0);
  const double range = 20200000.0;

  const auto factor = CarrierPhaseFactor(
      Key(0), Key(1), Key(2), range, satPos, 0.0);

  const double e0 = factor.evaluateError(recvPos, 0.0, 0.0)[0];
  EXPECT_DOUBLES_EQUAL(0.0, e0, 1e-9);

  // Adding 1.0 meter ambiguity should change error by 1.0
  const double e1 = factor.evaluateError(recvPos, 0.0, 1.0)[0];
  EXPECT_DOUBLES_EQUAL(1.0, e1, 1e-9);

  // Adding one L1 cycle in meters
  const double amb_one_cycle = LAMBDA_L1;
  const double e2 = factor.evaluateError(recvPos, 0.0, amb_one_cycle)[0];
  EXPECT_DOUBLES_EQUAL(LAMBDA_L1, e2, 1e-9);
}

// *************************************************************************
TEST(TestCarrierPhaseFactor, Jacobians) {
  const auto factor = CarrierPhaseFactor(
      Key(0), Key(1), Key(2), 24874028.989,
      Point3(-5824269.46342, -22935011.26952, -12195522.22428),
      -0.00022743876852667193);

  Values values;
  values.insert(Key(0), Point3(-3961908.12, 3348995.59, 3698211.13));
  values.insert(Key(1), 5.377885e-07);         // clock in seconds
  values.insert(Key(2), LAMBDA_L1 * 1000.0);   // ambiguity in meters
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestCarrierPhaseFactor, print) {
  const auto factor =
      CarrierPhaseFactor(Key(0), Key(1), Key(2), 0.0, Point3::Zero(), 0.0);
  factor.print("test ");
}

// *************************************************************************
TEST(TestCarrierPhaseFactor, equals) {
  const auto f1 = CarrierPhaseFactor(1, 2, 3, 100.0, Point3(1, 2, 3), 0.0);
  const auto f2 = CarrierPhaseFactor(1, 2, 3, 100.0, Point3(1, 2, 3), 0.0);
  const auto f3 = CarrierPhaseFactor(1, 2, 3, 200.0, Point3(1, 2, 3), 0.0);

  CHECK(f1.equals(f2));
  CHECK(!f1.equals(f3));
}

// *************************************************************************
// CarrierPhaseFactorArm tests
// *************************************************************************
TEST(TestCarrierPhaseFactorArm, Constructor) {
  const Point3 leverArm(0.5, -0.3, 1.0);
  const auto factor = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 0.0, Point3::Zero(), leverArm, 0.0);

  const Pose3 pose(Rot3::Identity(), Point3::Zero());
  const double error = factor.evaluateError(pose, 0.0, 0.0)[0];
  // Error should be range from lever arm to origin = ||leverArm||
  EXPECT_DOUBLES_EQUAL(leverArm.norm(), error, 1e-9);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, Jacobians) {
  const Point3 leverArm(0.1, 0.0, -0.5);
  const auto factor = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 24874028.989,
      Point3(-5824269.46342, -22935011.26952, -12195522.22428),
      leverArm, -0.00022743876852667193);

  Values values;
  values.insert(Key(0), Pose3(Rot3::RzRyRx(0.1, 0.2, 0.3),
                               Point3(-3961908.12, 3348995.59, 3698211.13)));
  values.insert(Key(1), 5.377885e-07);          // clock in seconds
  values.insert(Key(2), LAMBDA_L1 * 1000.0);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, ZeroLeverArm) {
  const Point3 pos(-3961908.12, 3348995.59, 3698211.13);
  const Point3 satPos(-5824269.46342, -22935011.26952, -12195522.22428);
  const double phi = 24874028.989;
  const double satClk = -0.00022743876852667193;
  const double clock = 1e-7;
  const double amb_m = LAMBDA_L1 * 500.0;

  const auto factorPt = CarrierPhaseFactor(
      Key(0), Key(1), Key(2), phi, satPos, satClk);
  const auto factorArm = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), phi, satPos, Point3::Zero(), satClk);

  const double errorPt = factorPt.evaluateError(pos, clock, amb_m)[0];
  const double errorArm = factorArm.evaluateError(
      Pose3(Rot3::Identity(), pos), clock, amb_m)[0];
  EXPECT_DOUBLES_EQUAL(errorPt, errorArm, 1e-9);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, EcefTnavIdentity) {
  const Point3 leverArm(0.5, -0.3, 1.0);
  const Point3 satPos(0.0, 0.0, 3.0);
  const double phi = 4.0;

  const auto factorEcef = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), phi, satPos, leverArm, 0.0);
  const auto factorNav = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), phi, satPos, leverArm, Pose3::Identity(), 0.0);

  const Pose3 pose(Rot3::RzRyRx(0.1, 0.2, 0.3), Point3(1.0, 2.0, 3.0));
  const double errorEcef = factorEcef.evaluateError(pose, 1.0, 10.0)[0];
  const double errorNav = factorNav.evaluateError(pose, 1.0, 10.0)[0];
  EXPECT_DOUBLES_EQUAL(errorEcef, errorNav, 1e-9);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, EcefTnavIdentityJacobians) {
  const Point3 leverArm(0.5, -0.3, 1.0);
  const auto factor = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 4.0, Vector3(0.0, 0.0, 3.0), leverArm,
      Pose3::Identity(), 0.0);

  Values values;
  values.insert(Key(0), Pose3(Rot3::RzRyRx(0.1, 0.2, 0.3),
                               Point3(1.0, 2.0, 3.0)));
  values.insert(Key(1), 0.0);
  values.insert(Key(2), 10.0);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, EcefTnavENUJacobians) {
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);
  const Point3 leverArm(0.1, 0.0, -0.5);
  const Point3 satPos(-5824269.46342, -22935011.26952, -12195522.22428);

  const auto factor = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 24874028.989, satPos, leverArm,
      ecef_T_nav, -0.00022743876852667193);

  Values values;
  values.insert(Key(0), Pose3(Rot3::RzRyRx(0.05, -0.03, 0.1),
                               Point3(10.0, 20.0, 5.0)));
  values.insert(Key(1), 5.377885093511699e-07);
  values.insert(Key(2), LAMBDA_L1 * 1000.0);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, EcefTnavConsistency) {
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);
  const Pose3 nav_T_body(Rot3::RzRyRx(0.05, -0.03, 0.1),
                          Point3(10.0, 20.0, 5.0));
  const Pose3 ecef_T_body = ecef_T_nav.compose(nav_T_body);

  const Point3 leverArm(0.1, 0.0, -0.5);
  const Point3 satPos(-5824269.46342, -22935011.26952, -12195522.22428);
  const double phi = 24874028.989;
  const double satClkBias = -0.00022743876852667193;
  const double clockBias = 5.377885093511699e-07;
  const double ambiguity_m = LAMBDA_L1 * 1000.0;

  const auto factorEcef = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), phi, satPos, leverArm, satClkBias);
  const auto factorNav = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), phi, satPos, leverArm,
      ecef_T_nav, satClkBias);

  const double errorEcef =
      factorEcef.evaluateError(ecef_T_body, clockBias, ambiguity_m)[0];
  const double errorNav =
      factorNav.evaluateError(nav_T_body, clockBias, ambiguity_m)[0];
  EXPECT_DOUBLES_EQUAL(errorEcef, errorNav, 1e-6);
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, EcefTnavEquals) {
  const Point3 leverArm(0.1, 0.2, 0.3);
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);

  const auto f1 = CarrierPhaseFactorArm(
      1, 2, 3, 0.0, Point3::Zero(), leverArm, 0.0);
  const auto f2 = CarrierPhaseFactorArm(
      1, 2, 3, 0.0, Point3::Zero(), leverArm, ecef_T_nav, 0.0);

  CHECK(!f1.equals(f2));
  CHECK(f2.equals(f2));
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, print) {
  const auto factor = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 0.0, Point3::Zero(), Point3(0.1, 0.2, 0.3),
      0.0);
  factor.print("test ");

  const auto factorNav = CarrierPhaseFactorArm(
      Key(0), Key(1), Key(2), 0.0, Point3::Zero(), Point3(0.1, 0.2, 0.3),
      Pose3::Identity(), 0.0);
  factorNav.print("test nav ");
}

// *************************************************************************
TEST(TestCarrierPhaseFactorArm, equals) {
  const Point3 leverArm(0.1, 0.2, 0.3);
  const auto f1 = CarrierPhaseFactorArm(
      1, 2, 3, 100.0, Point3(1, 2, 3), leverArm, 0.0);
  const auto f2 = CarrierPhaseFactorArm(
      1, 2, 3, 100.0, Point3(1, 2, 3), leverArm, 0.0);
  const auto f3 = CarrierPhaseFactorArm(
      1, 2, 3, 100.0, Point3(1, 2, 3), leverArm, 999.0);

  CHECK(f1.equals(f2));
  CHECK(!f1.equals(f3));  // different sat clock bias
}

// *************************************************************************
// DDCarrierPhaseFactor tests
// *************************************************************************

/// Shared test geometry for DD factors
static const Point3 kBasePos(-3961908.12, 3348995.59, 3698211.13);
static const Point3 kSatRefRov(-5824269.46, -22935011.27, -12195522.22);
static const Point3 kSatTargetRov(15524471.18, -6304441.44, 20851474.88);
// Satellite positions at base time (slightly different due to satellite motion)
static const Point3 kSatRefBase(-5824242.10, -22935002.50, -12195510.80);
static const Point3 kSatTargetBase(15524505.30, -6304460.20, 20851440.60);

static double computeGeodist(const Point3& sat, const Point3& rcv) {
  constexpr double OMGE = 7.2921151467e-5;
  constexpr double C_LIGHT = 299792458.0;
  const double r = (sat - rcv).norm();
  return r + OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
}

TEST(TestDDCarrierPhaseFactor, ZeroError) {
  // Build observations that give zero error at the true position
  const Point3 truePos(-3961900.00, 3349000.00, 3698215.00);
  const double ambRef = 100.0;
  const double ambTarget = 200.0;

  const double rRovRef = computeGeodist(kSatRefRov, truePos);
  const double rRovTarget = computeGeodist(kSatTargetRov, truePos);
  const double rBaseRef = computeGeodist(kSatRefBase, kBasePos);
  const double rBaseTarget = computeGeodist(kSatTargetBase, kBasePos);

  // DD model = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget)
  const double ddModel = (rRovRef - rBaseRef) - (rRovTarget - rBaseTarget);
  // DD obs = ddModel + lam * (ambRef - ambTarget) => error = 0
  const double ddObs = ddModel + LAMBDA_L1 * (ambRef - ambTarget);
  const double cpRovRef = ddObs / 2.0 + 1000.0;
  const double cpBaseRef = -ddObs / 2.0 + 1000.0;
  const double cpRovTarget = 500.0;
  const double cpBaseTarget = 500.0;
  // Adjust so DD obs is correct:
  // (cpRovRef - cpBaseRef) - (cpRovTarget - cpBaseTarget)
  // = (ddObs/2 + 1000 - (-ddObs/2 + 1000)) - (500 - 500) = ddObs

  const auto factor = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      cpRovRef, cpBaseRef, cpRovTarget, cpBaseTarget,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);

  const double error = factor.evaluateError(truePos, ambRef, ambTarget)[0];
  EXPECT_DOUBLES_EQUAL(0.0, error, 1e-4);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactor, Jacobians) {
  const auto factor = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);

  Values values;
  values.insert(Key(0), Point3(-3961900.00, 3349000.00, 3698215.00));
  values.insert(Key(1), 100.0);  // ambRef
  values.insert(Key(2), 200.0);  // ambTarget
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactor, equals) {
  const auto f1 = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);
  const auto f2 = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);
  const auto f3 = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, 0.25);  // different wavelength

  CHECK(f1.equals(f2));
  CHECK(!f1.equals(f3));
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactor, print) {
  const auto factor = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);
  factor.print("test ");
}

// *************************************************************************
// DDCarrierPhaseFactorArm tests
// *************************************************************************
TEST(TestDDCarrierPhaseFactorArm, Jacobians) {
  const Point3 leverArm(0.1, 0.0, -0.5);
  const auto factor = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm);

  Values values;
  values.insert(Key(0), Pose3(Rot3::RzRyRx(0.1, 0.2, 0.3),
                               Point3(-3961900.00, 3349000.00, 3698215.00)));
  values.insert(Key(1), 100.0);
  values.insert(Key(2), 200.0);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactorArm, EcefTnavJacobians) {
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);
  const Point3 leverArm(0.1, 0.0, -0.5);
  const auto factor = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm, ecef_T_nav);

  Values values;
  values.insert(Key(0), Pose3(Rot3::RzRyRx(0.05, -0.03, 0.1),
                               Point3(10.0, 20.0, 5.0)));
  values.insert(Key(1), 100.0);
  values.insert(Key(2), 200.0);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-3, 1e-5);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactorArm, ZeroLeverArm) {
  // DDCarrierPhaseFactorArm with zero lever arm should match DDCarrierPhaseFactor
  const Point3 pos(-3961900.00, 3349000.00, 3698215.00);

  const auto factorPt = DDCarrierPhaseFactor(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1);
  const auto factorArm = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, Point3::Zero());

  const double errorPt = factorPt.evaluateError(pos, 100.0, 200.0)[0];
  const double errorArm = factorArm.evaluateError(
      Pose3(Rot3::Identity(), pos), 100.0, 200.0)[0];
  EXPECT_DOUBLES_EQUAL(errorPt, errorArm, 1e-9);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactorArm, EcefTnavConsistency) {
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);
  const Pose3 nav_T_body(Rot3::RzRyRx(0.05, -0.03, 0.1),
                          Point3(10.0, 20.0, 5.0));
  const Pose3 ecef_T_body = ecef_T_nav.compose(nav_T_body);
  const Point3 leverArm(0.1, 0.0, -0.5);

  const auto factorEcef = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm);
  const auto factorNav = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      25000000.0, 24999500.0, 22000000.0, 21999800.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm, ecef_T_nav);

  const double errorEcef = factorEcef.evaluateError(ecef_T_body, 100.0, 200.0)[0];
  const double errorNav = factorNav.evaluateError(nav_T_body, 100.0, 200.0)[0];
  EXPECT_DOUBLES_EQUAL(errorEcef, errorNav, 1e-6);
}

// *************************************************************************
TEST(TestDDCarrierPhaseFactorArm, equals) {
  const Point3 leverArm(0.1, 0.0, -0.5);
  const Pose3 ecef_T_nav = makeEcefTnav(35.578, 139.749, 80.0);

  const auto f1 = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm);
  const auto f2 = DDCarrierPhaseFactorArm(
      Key(0), Key(1), Key(2),
      100.0, 99.0, 80.0, 79.0,
      kSatRefRov, kSatTargetRov, kSatRefBase, kSatTargetBase,
      kBasePos, LAMBDA_L1, leverArm, ecef_T_nav);

  // Different ecef_T_nav should not be equal
  CHECK(!f1.equals(f2));
  CHECK(f1.equals(f1));
  CHECK(f2.equals(f2));
}

// *************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
// *************************************************************************
