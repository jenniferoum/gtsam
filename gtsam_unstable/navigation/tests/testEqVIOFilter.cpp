/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/// @file testEqVIOFilter.cpp

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/navigation/EqVIOFilter.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio;

namespace {

State MakeState1() {
  SensorState sensor;
  sensor.inputBias = Bias(Vector3(0.03, -0.01, 0.02), Vector3(0.1, -0.2, 0.05));
  sensor.pose = Pose3(Rot3::RzRyRx(0.2, -0.1, 0.15), Point3(0.4, -0.2, 1.0));
  sensor.velocity = Vector3(0.5, -0.3, 0.2);
  sensor.cameraOffset =
      Pose3(Rot3::RzRyRx(-0.08, 0.04, -0.03), Point3(0.1, 0.0, 0.05));
  return State(sensor, {{Point3(0.3, -0.15, 4.5), 10}});
}

Se23 MakeA(const Rot3& R, const Point3& t, const Vector3& w) {
  Se23::Matrix3K x;
  x.col(0) = t;
  x.col(1) = w;
  return Se23(R, x);
}

SensorState SensorFixture() {
  SensorState s;
  s.inputBias = Bias((Vector3() << 0.01, -0.02, 0.03).finished(),
                     (Vector3() << 0.04, -0.01, 0.02).finished());
  s.pose = Pose3(Rot3::RzRyRx(0.1, -0.05, 0.2), Point3(0.3, -0.4, 1.2));
  s.velocity = Vector3(0.2, -0.1, 0.05);
  s.cameraOffset =
      Pose3(Rot3::RzRyRx(-0.02, 0.03, -0.01), Point3(0.1, 0.02, 0.08));
  return s;
}

State State0() { return State(SensorFixture(), {}); }
State State1() {
  return State(SensorFixture(), {{Point3(0.8, -0.2, 4.5), 11}});
}
State State3() {
  return State(SensorFixture(), {{Point3(0.8, -0.2, 4.5), 11},
                                 {Point3(-0.6, 0.3, 3.8), 22},
                                 {Point3(0.1, 0.7, 5.2), 33}});
}

VioGroup Group0() { return makeVioGroupIdentity(0); }
VioGroup Group1() {
  const SOT3 q1(SO3::Expmap((Vector3() << 0.02, -0.01, 0.03).finished()),
                std::log(1.1));
  return makeVioGroup(
      MakeA(Rot3::RzRyRx(0.03, -0.02, 0.01), Point3(0.05, -0.01, 0.02),
            Vector3(0.01, -0.02, 0.03)),
      Bias((Vector3() << 0.01, 0.0, -0.01).finished(),
           (Vector3() << 0.02, -0.01, 0.03).finished()),
      Pose3(Rot3::RzRyRx(-0.01, 0.02, -0.03), Point3(0.02, 0.01, -0.01)),
      LandmarkGroup({q1}));
}
VioGroup Group3() {
  const SOT3 q1(SO3::Expmap((Vector3() << 0.02, -0.01, 0.03).finished()),
                std::log(1.1));
  const SOT3 q2(SO3::Expmap((Vector3() << -0.01, 0.03, -0.02).finished()),
                std::log(0.95));
  const SOT3 q3(SO3::Expmap((Vector3() << 0.01, 0.02, 0.01).finished()),
                std::log(1.05));
  return makeVioGroup(
      MakeA(Rot3::RzRyRx(0.03, -0.02, 0.01), Point3(0.05, -0.01, 0.02),
            Vector3(0.01, -0.02, 0.03)),
      Bias((Vector3() << 0.01, 0.0, -0.01).finished(),
           (Vector3() << 0.02, -0.01, 0.03).finished()),
      Pose3(Rot3::RzRyRx(-0.01, 0.02, -0.03), Point3(0.02, 0.01, -0.01)),
      LandmarkGroup({q1, q2, q3}));
}

IMUInput ImuFixture() {
  IMUInput u;
  u.gyr = Vector3(0.02, -0.01, 0.03);
  u.acc = Vector3(0.1, -0.05, 9.7);
  u.gyrBiasVel = Vector3(0.01, 0.0, -0.01);
  u.accBiasVel = Vector3(-0.02, 0.01, 0.0);
  return u;
}

bool IsFinite(const Matrix& M) { return M.array().isFinite().all(); }

void PropagateSingle(EqVIOFilter& filter, const IMUInput& imu, double dt) {
  filter.propagate(std::vector<IMUInput>{imu}, std::vector<double>{dt});
}

}  // namespace

TEST(EqVIOFilter, DynamicLandmarksAddRemove) {
  EqVIOFilterParams params;
  params.initialPointDepth = 5.0;

  EqVIOFilter filter(params);
  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));

  IMUInput imu0;
  imu0.stamp = 0.0;
  imu0.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
  imu0.gyr = Vector3::Zero();
  filter.initializeFromIMU(imu0);

  IMUInput imu1 = imu0;
  imu1.stamp = 0.01;
  PropagateSingle(filter, imu1, 0.01);

  VisionMeasurement meas1;
  meas1[1] = camera->project2(Point3(0.2, -0.1, 3.5));
  meas1[2] = camera->project2(Point3(-0.3, 0.15, 4.0));
  PropagateSingle(filter, imu1, 0.01);
  filter.correct(meas1, camera);
  EXPECT_LONGS_EQUAL(2, filter.stateEstimate().n());

  IMUInput imu2 = imu0;
  imu2.stamp = 0.02;
  PropagateSingle(filter, imu2, 0.01);

  VisionMeasurement meas2;
  meas2[1] = meas1.at(1);
  filter.correct(meas2, camera);

  const State est = filter.stateEstimate();
  EXPECT_LONGS_EQUAL(1, est.n());
  EXPECT_LONGS_EQUAL(1, est.cameraLandmarks.front().id);
}

TEST(EqVIOFilter, InitAndPropagation) {
  EqVIOFilterParams params;

  EqVIOFilter filter(params);
  EXPECT(!filter.isInitialized());

  IMUInput imu0;
  imu0.stamp = 1.0;
  imu0.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
  imu0.gyr = Vector3::Zero();
  filter.initializeFromIMU(imu0);
  EXPECT(filter.isInitialized());

  IMUInput imu1 = imu0;
  imu1.stamp = 1.01;
  PropagateSingle(filter, imu1, 0.01);
  PropagateSingle(filter, imu1, 0.01);

  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));
  VisionMeasurement meas;
  filter.correct(meas, camera);

  EXPECT(filter.view().Sigma.array().isFinite().all());
}

TEST(EqVIOFilter, ParityShortSequence) {
  EqVIOFilterParams params;

  EqVIOFilter filter(params);
  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));

  IMUInput initImu;
  initImu.stamp = 0.0;
  initImu.gyr = Vector3::Zero();
  initImu.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
  filter.initializeFromIMU(initImu);

  State manual = filter.stateEstimate();
  double t = 0.01;
  const double dt = 0.01;
  for (int k = 0; k < 8; ++k) {
    IMUInput imu;
    imu.stamp = t;
    imu.gyr = Vector3(0.01, -0.02, 0.015);
    imu.acc = Vector3(0.02, -0.01, GRAVITY_CONSTANT - 0.03);
    imu.gyrBiasVel = Vector3(0.0005, -0.0002, 0.0001);
    imu.accBiasVel = Vector3(-0.0004, 0.0003, -0.0001);
    PropagateSingle(filter, imu, dt);

    VisionMeasurement emptyMeas;
    filter.correct(emptyMeas, camera);

    manual = integrateSystemFunction(manual, imu, dt);
    t += dt;
  }

  const State est = filter.stateEstimate();
  const Vector eps = manual.localCoordinates(est);
  EXPECT(eps.norm() < 2e-5);
}

TEST(EqVIOFilter, VisionUpdate) {
  EqVIOFilterParams params;

  const State xi0 = MakeState1();
  const Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;
  EqVIOFilter filter(xi0, Sigma0, params);

  IMUInput imu;
  imu.stamp = 0.0;
  imu.gyr = Vector3::Zero();
  imu.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
  PropagateSingle(filter, imu, 0.01);

  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));
  const VisionMeasurement meas =
      measureSystemState(filter.stateEstimate(), camera);
  filter.correct(meas, camera);

  EXPECT_LONGS_EQUAL(1, filter.stateEstimate().n());
  EXPECT(filter.view().Sigma.array().isFinite().all());
}

TEST(EqVIOFilter, Smoke) {
  EqVIOFilterParams params;

  SensorState sensor;
  sensor.inputBias = Bias::Identity();
  sensor.pose = Pose3::Identity();
  sensor.velocity.setZero();
  sensor.cameraOffset = Pose3::Identity();

  State xi0(sensor, {{Point3(0.8, -0.2, 4.5), 11}, {Point3(-0.6, 0.3, 3.8), 22}});
  Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;

  EqVIOFilter filter(xi0, Sigma0, params);
  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));

  const double dt = 0.01;
  double t = 0.0;
  for (int k = 0; k < 100; ++k) {
    t += dt;
    IMUInput imu;
    imu.stamp = t;
    imu.gyr = Vector3::Zero();
    imu.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
    PropagateSingle(filter, imu, dt);

    if (k % 5 == 0) {
      VisionMeasurement y = measureSystemState(filter.stateEstimate(), camera);
      filter.correct(y, camera);
    }
  }

  const State est = filter.stateEstimate();
  EXPECT_LONGS_EQUAL(2, est.n());
  EXPECT_LONGS_EQUAL(xi0.dim(), filter.view().Sigma.rows());
  EXPECT_LONGS_EQUAL(xi0.dim(), filter.view().Sigma.cols());
  EXPECT(filter.view().Sigma.array().isFinite().all());
}

TEST(VIOEqFMatricesInvDepth, Selector) {
  const EqFCoordinateSuite* suite = &EqFCoordinateSuite_invdepth;
  EXPECT(suite != nullptr);
}

//******************************************************************************
TEST(VIOEqFMatricesInvDepth, ShapesAndFinite) {
  const auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));
  const EqFCoordinateSuite& suite = EqFCoordinateSuite_invdepth;

  for (const auto& pair : std::vector<std::pair<State, VioGroup>>{
           {State0(), Group0()}, {State1(), Group1()}, {State3(), Group3()}}) {
    const State& xi0 = pair.first;
    const VioGroup& X = pair.second;
    const IMUInput imu = ImuFixture();
    const VisionMeasurement y =
        measureSystemState(stateGroupAction(X, xi0), camera);

    const Matrix A = suite.stateMatrixA(X, xi0, imu);
    const Matrix B = suite.inputMatrixB(X, xi0);
    const Matrix C = suite.outputMatrixC(xi0, X, y, camera, true);

    EXPECT_LONGS_EQUAL(xi0.dim(), A.rows());
    EXPECT_LONGS_EQUAL(xi0.dim(), A.cols());
    EXPECT_LONGS_EQUAL(xi0.dim(), B.rows());
    EXPECT_LONGS_EQUAL(IMUInput::CompDim, B.cols());
    EXPECT_LONGS_EQUAL(2 * static_cast<long>(y.size()), C.rows());
    EXPECT_LONGS_EQUAL(xi0.dim(), C.cols());

    EXPECT(IsFinite(A));
    EXPECT(IsFinite(B));
    EXPECT(IsFinite(C));
  }
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
