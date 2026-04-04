/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testEqVIOFilter.cpp
 * @brief Unit tests for the EqVIOFilter.
 * @author Rohan Bansal
 */

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
  return State(sensor, {{Point3(0.3, -0.15, 4.5)}});
}

std::vector<Key> Keys1() { return {42}; }
std::vector<Key> Keys2() { return {10, 20}; }

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
State State1() { return State(SensorFixture(), {{Point3(0.8, -0.2, 4.5)}}); }
State State3() {
  return State(SensorFixture(), {{Point3(0.8, -0.2, 4.5)},
                                 {Point3(-0.6, 0.3, 3.8)},
                                 {Point3(0.1, 0.7, 5.2)}});
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
  filter.predict(std::vector<IMUInput>{imu}, std::vector<double>{dt});
}

State integrateSystemFunction(const State& state, const IMUInput& velocity,
                              double dt) {
  State out;
  const SensorState& sensor = state.sensor;
  const IMUInput v_est = velocity - sensor.inputBias;

  out.sensor.inputBias =
      Bias(sensor.inputBias.accelerometer() + dt * velocity.accBiasVel,
           sensor.inputBias.gyroscope() + dt * velocity.gyrBiasVel);

  const Rot3 dR = Rot3::Expmap(dt * v_est.gyr);
  const Vector3 dXWorld = dt * (sensor.pose.rotation() * sensor.velocity) +
                          0.5 * dt * dt *
                              (sensor.pose.rotation() * v_est.acc +
                               Vector3(0, 0, -GRAVITY_CONSTANT));
  const Point3 dXBody = sensor.pose.rotation().unrotate(dXWorld);
  const Pose3 b0Tb1(dR, dXBody);

  out.sensor.pose = sensor.pose.compose(b0Tb1);

  const Vector3 inertialVelocityDiff =
      sensor.pose.rotation() * v_est.acc + Vector3(0, 0, -GRAVITY_CONSTANT);
  out.sensor.velocity = out.sensor.pose.rotation().unrotate(
      sensor.pose.rotation() * sensor.velocity + dt * inertialVelocityDiff);

  const Pose3 c1Tc0 = sensor.cameraOffset.inverse()
                          .compose(b0Tb1.inverse())
                          .compose(sensor.cameraOffset);
  out.cameraLandmarks.resize(state.n());

  for (size_t i = 0; i < state.n(); ++i) {
    out.cameraLandmarks[i].p = c1Tc0.transformFrom(state.cameraLandmarks[i].p);
  }

  out.sensor.cameraOffset = sensor.cameraOffset;
  return out;
}

}  // namespace

// Verifies landmark insertion/removal logic respects the stale-track grace
// period.
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
  filter.update(meas1, camera);
  EXPECT_LONGS_EQUAL(2, filter.state().n());

  IMUInput imu2 = imu0;
  imu2.stamp = 0.02;
  PropagateSingle(filter, imu2, 0.01);

  VisionMeasurement meas2;
  meas2[1] = meas1.at(1);
  filter.update(meas2, camera);
  EXPECT_LONGS_EQUAL(2, filter.state().n());

  filter.update(meas2, camera);
  EXPECT_LONGS_EQUAL(1, filter.state().n());
}

// Verifies IMU-based initialization and basic propagation produce finite
// covariance.
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
  filter.update(meas, camera);

  EXPECT(filter.errorCovariance().array().isFinite().all());
}

// Verifies short propagation sequence matches direct system-function
// integration.
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

  State manual = filter.state();
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
    filter.update(emptyMeas, camera);

    manual = integrateSystemFunction(manual, imu, dt);
    t += dt;
  }

  const State est = filter.state();
  const Vector eps = manual.localCoordinates(est);
  EXPECT(eps.norm() < 2e-5);
}

// Verifies vision update keeps landmark count and covariance numerically valid.
TEST(EqVIOFilter, VisionUpdate) {
  EqVIOFilterParams params;

  const State xi0 = MakeState1();
  const Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;
  EqVIOFilter filter(xi0, Sigma0, Keys1(), params);

  IMUInput imu;
  imu.stamp = 0.0;
  imu.gyr = Vector3::Zero();
  imu.acc = Vector3(0.0, 0.0, GRAVITY_CONSTANT);
  PropagateSingle(filter, imu, 0.01);

  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));
  const VisionMeasurement meas = measureSystemState(filter.state(), Keys1(), camera);
  filter.update(meas, camera);

  EXPECT_LONGS_EQUAL(1, filter.state().n());
  EXPECT(filter.errorCovariance().array().isFinite().all());
}

// End-to-end smoke test for repeated propagate/correct cycles.
TEST(EqVIOFilter, Smoke) {
  EqVIOFilterParams params;

  SensorState sensor;
  sensor.inputBias = Bias::Identity();
  sensor.pose = Pose3::Identity();
  sensor.velocity.setZero();
  sensor.cameraOffset = Pose3::Identity();

  State xi0(sensor, {{Point3(0.8, -0.2, 4.5)}, {Point3(-0.6, 0.3, 3.8)}});
  Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;

  EqVIOFilter filter(xi0, Sigma0, Keys2(), params);
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
      VisionMeasurement y = measureSystemState(filter.state(), Keys2(), camera);
      filter.update(y, camera);
    }
  }

  const State est = filter.state();
  EXPECT_LONGS_EQUAL(2, est.n());
  EXPECT_LONGS_EQUAL(xi0.dim(), filter.errorCovariance().rows());
  EXPECT_LONGS_EQUAL(xi0.dim(), filter.errorCovariance().cols());
  EXPECT(filter.errorCovariance().array().isFinite().all());
}

// Verifies seeded landmark states use the explicit keyed constructor.
TEST(EqVIOFilter, SeededLandmarkKeys) {
  EqVIOFilterParams params;
  const State xi0 = MakeState1();
  const Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;

  EqVIOFilter filter(xi0, Sigma0, Keys1(), params);
  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));
  const VisionMeasurement meas = measureSystemState(filter.state(), Keys1(), camera);
  filter.update(meas, camera);

  EXPECT_LONGS_EQUAL(1, filter.state().n());
  EXPECT(filter.errorCovariance().array().isFinite().all());
}

// Verifies multiple landmark additions/removals are reconciled cleanly.
TEST(EqVIOFilter, BatchLandmarkStructureChange) {
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

  VisionMeasurement meas1;
  meas1[1] = camera->project2(Point3(0.2, -0.1, 3.5));
  meas1[2] = camera->project2(Point3(-0.3, 0.15, 4.0));
  filter.update(meas1, camera);
  EXPECT_LONGS_EQUAL(2, filter.state().n());

  VisionMeasurement meas2;
  meas2[2] = meas1.at(2);
  meas2[3] = camera->project2(Point3(0.4, 0.05, 4.2));
  meas2[4] = camera->project2(Point3(-0.2, -0.2, 3.7));
  filter.update(meas2, camera);
  EXPECT_LONGS_EQUAL(4, filter.state().n());
  EXPECT_LONGS_EQUAL(SensorState::CompDim + 12, filter.errorCovariance().rows());

  filter.update(meas2, camera);
  EXPECT_LONGS_EQUAL(3, filter.state().n());
  EXPECT_LONGS_EQUAL(SensorState::CompDim + 9, filter.errorCovariance().rows());
}

// Verifies supplied R does not override the absolute-residual outlier gate.
TEST(EqVIOFilter, AbsoluteOutlierGateWithSuppliedR) {
  EqVIOFilterParams params;
  params.outlierThresholdAbs = 1e-3;
  params.featureRetention = 0.0;

  const State xi0 = MakeState1();
  const Matrix Sigma0 = Matrix::Identity(xi0.dim(), xi0.dim()) * 1e-3;
  EqVIOFilter filter(xi0, Sigma0, Keys1(), params);
  auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));

  VisionMeasurement meas = measureSystemState(filter.state(), Keys1(), camera);
  const Key key = Keys1().front();
  meas[key] = meas.at(key) + Point2(0.1, 0.1);

  const Matrix R = Matrix::Identity(2, 2) * 1e6;
  filter.update(meas, camera, R);

  EXPECT_LONGS_EQUAL(0, filter.state().n());
}

// Verifies EqF linearization matrices have expected shapes and finite entries.
TEST(VIOEqFMatrices, ShapesAndFinite) {
  const auto camera =
      std::make_shared<CameraModel>(Pose3::Identity(), Cal3_S2(1, 1, 0, 0, 0));

  for (const auto& pair : std::vector<std::pair<State, VioGroup>>{
           {State0(), Group0()}, {State1(), Group1()}, {State3(), Group3()}}) {
    const State& xi0 = pair.first;
    const VioGroup& X = pair.second;
    const IMUInput imu = ImuFixture();
    const VisionMeasurement y =
        measureSystemState(stateGroupAction(X, xi0), camera);

    const Matrix A = EqFStateMatrixA(X, xi0, imu);
    const Matrix B = EqFInputMatrixB(X, xi0);
    const Matrix C =
        EqFoutputMatrixC(xi0, measurementIds(y), X, y, camera, true);

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
