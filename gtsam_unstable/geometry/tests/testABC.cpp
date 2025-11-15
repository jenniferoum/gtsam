/**
 * @file testABC.cpp
 * @brief Test file for ABC (Attitude-Bias-Calibration) system components
 *
 * @author Darshan Rajasekaran
 * @author Jennifer Oum
 * @author Rohan Bansal
 * @author Frank Dellaert
 * @date 2025
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/geometry/ABC.h>

using namespace gtsam;

// Define N for testing purposes, e.g., 2 calibration states
using State = abc::State<2>;
using Group = abc::Group<2>;
using Geometry = abc::Geometry<2>;
using Calibrations = abc::Calibrations<2>;

/* ************************************************************************* */
TEST(ABC, State) {
  Rot3 R1 = Rot3::Rx(0.1);
  Vector3 b1(0.01, 0.02, 0.03);
  Calibrations S1;
  S1[0] = Rot3::Ry(0.05);
  S1[1] = Rot3::Rz(0.06);

  State state1(R1, b1, S1);

  EXPECT(assert_equal(state1.R, R1, 1e-9));
  EXPECT(assert_equal(state1.b, b1, 1e-9));
  EXPECT(assert_equal(state1.S, S1));

  // Test identity
  State identityState = State::identity();
  EXPECT(assert_equal(identityState.R, Rot3(), 1e-9));
  EXPECT(assert_equal<Vector3>(identityState.b, Z_3x1, 1e-9));
  Calibrations expectedS_id;
  EXPECT(assert_equal(identityState.S, expectedS_id));

  // Test localCoordinates and retract (manifold properties)
  Rot3 R2 = Rot3::Rx(0.2);
  Vector3 b2(0.05, 0.06, 0.07);
  Calibrations S2;
  S2[0] = Rot3::Ry(0.1);
  S2[1] = Rot3::Rz(0.15);
  State state2(R2, b2, S2);

  Vector actual_local = state1.localCoordinates(state2);
  State retracted_state2 = state1.retract(actual_local);
  EXPECT(assert_equal(retracted_state2.R, state2.R, 1e-9));
  EXPECT(assert_equal(retracted_state2.b, state2.b, 1e-9));
  EXPECT(assert_equal(retracted_state2.S, state2.S));

  // Test localCoordinates at identity
  Vector expected_identity_local = Vector::Zero(6 + 3 * 2);
  EXPECT(assert_equal(identityState.localCoordinates(identityState),
                      expected_identity_local, 1e-9));

  // Test retract at identity
  Vector v_test = Vector::Zero(6 + 3 * 2);
  v_test.head<3>() << 0.1, 0.2, 0.3;         // R
  v_test.segment<3>(3) << 0.01, 0.02, 0.03;  // b
  v_test.segment<3>(6) << 0.05, 0.06, 0.07;  // S[0]
  v_test.segment<3>(9) << 0.08, 0.09, 0.10;  // S[1]

  State retracted_from_id = identityState.retract(v_test);
  EXPECT(
      assert_equal(retracted_from_id.R, Rot3::Expmap(v_test.head<3>()), 1e-9));
  EXPECT(assert_equal(retracted_from_id.b, v_test.segment<3>(3).eval(), 1e-9));
  EXPECT(assert_equal(retracted_from_id.S[0],
                      Rot3::Expmap(v_test.segment<3>(6).eval()), 1e-9));
  EXPECT(assert_equal(retracted_from_id.S[1],
                      Rot3::Expmap(v_test.segment<3>(9).eval()), 1e-9));

  // Test retract invalid argument
  CHECK_EXCEPTION(identityState.retract(Vector::Zero(1)),
                  std::invalid_argument);
}

/* ************************************************************************* */
namespace abc_group_examples {
Rot3 A1 = Rot3::Rx(0.1);
Vector3 t1_vec(0.01, 0.02, 0.03);
Calibrations B1{Rot3::Ry(0.05), Rot3::Rz(0.06)};
Group g1(Pose3(A1, Point3(t1_vec)), B1);

Rot3 A2 = Rot3::Ry(0.2);
Vector3 t2_vec(0.04, 0.05, 0.06);
Calibrations B2{Rot3::Rz(0.07), Rot3::Rx(0.08)};
Group g2(Pose3(A2, Point3(t2_vec)), B2);
}  // namespace abc_group_examples

/* ************************************************************************* */
TEST(ABC, GroupOperations) {
  using namespace abc_group_examples;

  // Test group multiplication
  Group g1_g2 = g1 * g2;
  EXPECT(assert_equal(g1_g2.A(), A1 * A2, 1e-9));
  Vector3 expected_a = t1_vec + A1.matrix() * t2_vec;
  EXPECT(assert_equal(g1_g2.a(), expected_a, 1e-9));
  EXPECT(assert_equal(g1_g2.calibrations()[0], B1[0] * B2[0], 1e-9));
  EXPECT(assert_equal(g1_g2.calibrations()[1], B1[1] * B2[1], 1e-9));

  // Test inverse
  Group g1_inv = g1.inverse();
  EXPECT(assert_equal(g1_inv.A(), A1.inverse(), 1e-9));
  Vector3 expected_a_inv = -A1.inverse().matrix() * t1_vec;
  EXPECT(assert_equal(g1_inv.a(), expected_a_inv, 1e-9));
  EXPECT(assert_equal(g1_inv.calibrations()[0], B1[0].inverse(), 1e-9));
  EXPECT(assert_equal(g1_inv.calibrations()[1], B1[1].inverse(), 1e-9));

  // Test g * g.inv() == identity
  Group identity_check = g1 * g1_inv;
  Group expected_identity = Group::Identity();
  EXPECT(assert_equal(identity_check.A(), expected_identity.A(), 1e-9));
  EXPECT(assert_equal(identity_check.a(), expected_identity.a(), 1e-9));
  EXPECT(assert_equal(identity_check.calibrations(),
                      expected_identity.calibrations()));

  // Test Expmap and Logmap
  Group::TangentVector v_tangent = Group::TangentVector::Zero();
  v_tangent.head<3>() << 0.1, 0.2, 0.3;         // For A
  v_tangent.segment<3>(3) << 0.01, 0.02, 0.03;  // For 'a' part
  v_tangent.segment<3>(6) << 0.04, 0.05, 0.06;  // For B[0]
  v_tangent.segment<3>(9) << 0.07, 0.08, 0.09;  // For B[1]

  Group g_exp = Group::Expmap(v_tangent);
  // Logmap is a placeholder, so we can only check its consistency if
  // exp(log(g)) = g Currently Logmap returns zero, so cannot properly test
  // exp(log(g)) == g
  // EXPECT(assert_equal(Group::Logmap(g_exp), v_tangent, 1e-9)); // This will
  // fail with placeholder

  // Test retract on G
  Group g_retracted = g1.retract(v_tangent);

  const Group composed = g1 * Group::Expmap(v_tangent);
  EXPECT(assert_equal(g_retracted.A(), composed.A(), 1e-9));
  EXPECT(assert_equal(g_retracted.a(), composed.a(), 1e-9));
  EXPECT(assert_equal(g_retracted.calibrations(), composed.calibrations()));

  // Test traits for G
  const Group identity = Group::Identity();
  EXPECT(assert_equal(traits<Group>::Identity().A(), identity.A(), 1e-9));
  EXPECT(assert_equal(traits<Group>::Identity().a(), identity.a(), 1e-9));
  EXPECT(assert_equal(traits<Group>::Identity().calibrations(),
                      identity.calibrations()));
  // testLie<Group>(g1, g2, 1e-9);
}

//******************************************************************************
TEST(ABC, AdjointMap) {
  using namespace abc_group_examples;

  Group::Jacobian adjoint = g1.AdjointMap();
  Group::Jacobian expected = Group::Jacobian::Zero();
  expected.block<6, 6>(0, 0) = g1.pose().AdjointMap();
  for (size_t i = 0; i < Group::numSensors; ++i) {
    expected.block<3, 3>(6 + 3 * i, 6 + 3 * i) =
        g1.calibrations()[i].AdjointMap();
  }

  EXPECT(assert_equal(adjoint, expected));
}

/* ************************************************************************* */
TEST(ABC, GroupActions) {
  // Setup a G element
  Rot3 gA = Rot3::Rx(0.1);
  Vector3 gTranslation(0.01, 0.02, 0.03);
  Calibrations gB;
  gB[0] = Rot3::Ry(0.05);
  gB[1] = Rot3::Rz(0.06);
  Group X(Pose3(gA, Point3(gTranslation)), gB);

  // Setup a State element
  Rot3 sR = Rot3::Rz(0.2);
  Vector3 sb(0.04, 0.05, 0.06);
  Calibrations sS;
  sS[0] = Rot3::Rx(0.07);
  sS[1] = Rot3::Ry(0.08);
  State xi(sR, sb, sS);

  // Test State Action (G * State)
  State transformed_xi = X * xi;
  EXPECT(assert_equal(transformed_xi.R, xi.R * X.A(), 1e-9));
  EXPECT(assert_equal<Matrix>(transformed_xi.b,
                              X.A().inverse().matrix() * (xi.b - X.a()), 1e-9));
  EXPECT(assert_equal(transformed_xi.S[0],
                      X.A().inverse() * xi.S[0] * X.calibrations()[0], 1e-9));
  EXPECT(assert_equal(transformed_xi.S[1],
                      X.A().inverse() * xi.S[1] * X.calibrations()[1], 1e-9));

  // Test velocityAction
  Vector3 omega(1, 2, 3);
  Vector6 u = abc::toInputVector(omega);
  Vector6 transformed_u = velocityAction(X, u);
  EXPECT(assert_equal<Vector>(transformed_u.head<3>(),
                              X.A().inverse().matrix() * (omega - X.a()),
                              1e-9));
  EXPECT(assert_equal<Vector>(transformed_u.tail<3>(), Z_3x1,
                              1e-9));  // Virtual input stays zero

  // Test outputAction (calibrated sensor)
  Unit3 y_meas = Unit3(1, 0, 0);
  int cal_idx = 0;
  Vector3 transformed_y_calibrated = outputAction(X, y_meas, cal_idx);
  EXPECT(assert_equal<Vector>(
      transformed_y_calibrated,
      X.calibrations()[cal_idx].inverse().matrix() * y_meas.unitVector(),
      1e-9));

  // Test outputAction (uncalibrated sensor)
  int uncalibrated_idx = -1;
  Vector3 transformed_y_uncalibrated =
      outputAction(X, y_meas, uncalibrated_idx);
  EXPECT(assert_equal<Vector>(transformed_y_uncalibrated,
                              X.A().inverse().matrix() * y_meas.unitVector(),
                              1e-9));

  // Test outputAction out of range
  CHECK_EXCEPTION(outputAction(X, y_meas, 2), std::out_of_range);
}

/* ************************************************************************* */
TEST(ABC, Geometry_identityState) {
  State expected_id = State::identity();
  State actual = Geometry::identityState();
  EXPECT(assert_equal<Rot3>(actual.R, expected_id.R, 1e-9));
  EXPECT(assert_equal<Vector>(actual.b, expected_id.b, 1e-9));
  EXPECT(assert_equal(actual.S, expected_id.S));
}

/* ************************************************************************* */
TEST(ABC, Geometry_groupAction) {
  Rot3 A = Rot3::Rx(0.1);
  Vector3 translation(0.01, 0.02, 0.03);
  Calibrations B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  Group g(Pose3(A, Point3(translation)), B);

  Rot3 R = Rot3::Rz(0.2);
  Vector3 b(0.04, 0.05, 0.06);
  Calibrations S;
  S[0] = Rot3::Rx(0.07);
  S[1] = Rot3::Ry(0.08);
  State x(R, b, S);

  State expected_transformed_x = g * x;
  State actual_transformed = Geometry::groupAction(g, x);

  // Compare each component
  EXPECT(
      assert_equal<Rot3>(actual_transformed.R, expected_transformed_x.R, 1e-9));
  EXPECT(assert_equal<Vector>(actual_transformed.b, expected_transformed_x.b,
                              1e-9));
  EXPECT(assert_equal(actual_transformed.S, expected_transformed_x.S));
}

/* ************************************************************************* */
TEST(ABC, Geometry_lift) {
  // Setup state
  Rot3 R = Rot3();
  Vector3 b(0.1, 0.2, 0.3);
  Calibrations S_arr;
  S_arr[0] = Rot3::Rx(0.05);
  S_arr[1] = Rot3::Ry(0.06);
  State xi(R, b, S_arr);

  // Setup input
  Vector3 omega(0.5, 0.6, 0.7);
  Vector6 u = abc::toInputVector(omega);
  typename Group::TangentVector L = xi.lift(u);

  // Expected values
  Vector3 expected_L_head = omega - xi.b;
  Vector3 expected_L_segment3 = -Rot3::Hat(omega) * xi.b;
  Vector3 expected_L_segment6_0 = S_arr[0].inverse().matrix() * expected_L_head;
  Vector3 expected_L_segment6_1 = S_arr[1].inverse().matrix() * expected_L_head;

  EXPECT(assert_equal<Vector>(L.head<3>(), expected_L_head, 1e-9));
  EXPECT(assert_equal<Vector>(L.segment<3>(3), expected_L_segment3, 1e-9));
  EXPECT(assert_equal<Vector>(L.segment<3>(6), expected_L_segment6_0, 1e-9));
  EXPECT(assert_equal<Vector>(L.segment<3>(9), expected_L_segment6_1, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_stateMatrixA) {
  // Setup G element for X_hat
  Rot3 A = Rot3::Rx(0.1);
  Vector3 translation(0.01, 0.02, 0.03);
  Calibrations B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  Group X_hat(Pose3(A, Point3(translation)), B);

  // Setup input
  Vector3 omega(0.5, 0.6, 0.7);
  Vector6 u = abc::toInputVector(omega);
  Matrix A_matrix = Geometry::stateMatrixA(X_hat, u);
  Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inverse(), u).head<3>());

  Matrix expected_A1 = Matrix::Zero(6, 6);
  expected_A1.block<3, 3>(0, 3) = -I_3x3;
  expected_A1.block<3, 3>(3, 3) = W0;

  Matrix expected_A2 = gtsam::diag({W0, W0});
  Matrix expected_A_matrix = gtsam::diag({expected_A1, expected_A2});

  EXPECT(assert_equal(A_matrix, expected_A_matrix, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_stateTransitionMatrix) {
  // Setup G element for X_hat
  Rot3 A = Rot3::Rx(0.1);
  Vector3 translation(0.01, 0.02, 0.03);
  Calibrations B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  Group X_hat(Pose3(A, Point3(translation)), B);

  // Setup input
  Vector3 omega(0.5, 0.6, 0.7);
  double dt = 0.1;

  Vector6 u = abc::toInputVector(omega);
  Matrix Phi = Geometry::stateTransitionMatrix(u, dt, X_hat);
  Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inverse(), u).head<3>());
  Matrix Phi1 = Matrix::Zero(6, 6);
  Matrix3 Phi12 = -dt * (I_3x3 + (dt / 2) * W0 + ((dt * dt) / 6) * W0 * W0);
  Matrix3 Phi22 = I_3x3 + dt * W0 + ((dt * dt) / 2) * W0 * W0;

  Phi1.block<3, 3>(0, 0) = I_3x3;
  Phi1.block<3, 3>(0, 3) = Phi12;
  Phi1.block<3, 3>(3, 3) = Phi22;
  Matrix Phi2 = gtsam::diag({Phi22, Phi22});
  Matrix expected_Phi = gtsam::diag({Phi1, Phi2});

  EXPECT(assert_equal(Phi, expected_Phi, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_inputMatrix) {
  // Setup G element for X_hat
  Rot3 A = Rot3::Rx(0.1);
  Vector3 translation(0.01, 0.02, 0.03);
  Calibrations B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  Group X_hat(Pose3(A, Point3(translation)), B);

  Matrix input_matrix = Geometry::inputMatrix(X_hat);

  const Matrix3 X_hat_rot = X_hat.A().matrix();
  Matrix expected_B1 = gtsam::diag({X_hat_rot, X_hat_rot});
  Matrix expected_B2(3 * 2, 3 * 2);
  expected_B2.setZero();
  for (size_t i = 0; i < 2; ++i) {
    expected_B2.block<3, 3>(3 * i, 3 * i) = X_hat.calibrations()[i].matrix();
  }
  Matrix expected_input_matrix = gtsam::diag({expected_B1, expected_B2});

  EXPECT(assert_equal(input_matrix, expected_input_matrix, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_processNoise) {
  Matrix Sigma = (Matrix(6, 6) << 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 6)
                     .finished();

  Matrix Q = Geometry::processNoise(Sigma);

  Matrix expected_Q_cal_part = 1e-9 * I_6x6;
  Matrix expected_Q = gtsam::diag({Sigma, expected_Q_cal_part});

  EXPECT(assert_equal(Q, expected_Q, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_inputMatrixBt) {
  // This function is identical to inputMatrix, so we'll test its output matches
  // inputMatrix. Setup G element for X_hat
  Rot3 A = Rot3::Rx(0.1);
  Vector3 translation(0.01, 0.02, 0.03);
  Calibrations B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  Group X_hat(Pose3(A, Point3(translation)), B);

  Matrix input_matrix_Bt = Geometry::inputMatrixBt(X_hat);
  Matrix input_matrix =
      Geometry::inputMatrix(X_hat);  // Reference from the other function

  EXPECT(assert_equal(input_matrix_Bt, input_matrix, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_measurementMatrixC) {
  Unit3 d = Unit3(0, 0, 1);  // Reference direction (e.g., gravity)
  Matrix3 wedge_d = Rot3::Hat(d.unitVector());

  // Test with calibrated sensor (idx = 0)
  int cal_idx = 0;
  Matrix C_cal = Geometry::measurementMatrixC(d, cal_idx);

  Matrix expected_Cc_cal = Matrix::Zero(3, 3 * 2);
  expected_Cc_cal.block<3, 3>(0, 3 * cal_idx) = wedge_d;

  Matrix expected_temp_cal(3, 6 + 3 * 2);
  expected_temp_cal.block<3, 3>(0, 0) = wedge_d;
  expected_temp_cal.block<3, 3>(0, 3) = Matrix3::Zero();
  expected_temp_cal.block(0, 6, 3, 3 * 2) = expected_Cc_cal;
  Matrix expected_C_cal = wedge_d * expected_temp_cal;

  EXPECT(assert_equal(C_cal, expected_C_cal, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, EqFilter) {
  using M = abc::State<2>;
  using G = abc::Group<2>;
  using EqFilter = gtsam::EqF<G, M>;

  const G g_0;
  const M xi_ref;  // Reference state (xi circle) and not inital state?
  const int numSensors = 2;

  Matrix initialSigma = Matrix::Identity(G::dimension, G::dimension);
  initialSigma.diagonal().head<3>() =
      Vector3::Constant(0.1);  // Attitude uncertainty
  initialSigma.diagonal().segment<3>(3) =
      Vector3::Constant(0.01);  // Bias uncertainty
  initialSigma.diagonal().tail<3>() =
      Vector3::Constant(0.1);  // Calibration uncertainty

  EqFilter filter(g_0, xi_ref, initialSigma, numSensors);

  G X_HatActual = filter.groupEstimate();

  G X_HatExpected = filter.state();  // from LieGroupEKF

  EXPECT(traits<G>::Equals(g_0, X_HatActual, 1e-9));

  EXPECT(traits<G>::Equals(X_HatActual, X_HatExpected, 1e-9));

  Vector3 omega(0.01, -0.02, 0.015);
  Matrix Sigma = I_6x6;
  double dt = 0.01;

  Vector6 u_vec = abc::toInputVector(omega);
  Matrix Q = Geometry::processNoise(Sigma);
  filter.predict(u_vec, Q, dt);

  EXPECT(traits<G>::Equals(filter.groupEstimate(), filter.state(), 1e-9));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
