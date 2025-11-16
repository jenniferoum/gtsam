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
namespace abc_examples {
Rot3 A1 = Rot3::Rx(0.1);
Vector3 t1(0.01, 0.02, 0.03);
Calibrations B1{Rot3::Ry(0.05), Rot3::Rz(0.06)};
State xi1(A1, t1, B1);
Group g1(Pose3(A1, t1), B1);

Rot3 A2 = Rot3::Ry(0.2);
Vector3 t2(0.04, 0.05, 0.06);
Calibrations B2{Rot3::Rz(0.07), Rot3::Rx(0.08)};
State xi2(A2, t2, B2);
Group g2(Pose3(A2, t2), B2);

Vector3 omega(1, 2, 3);
Vector6 u = abc::toInputVector(omega);
}  // namespace abc_examples

/* ************************************************************************* */
TEST(ABC, State) {
  State state1 = abc_examples::xi1;

  EXPECT(assert_equal(abc_examples::A1, state1.R));
  EXPECT(assert_equal(abc_examples::t1, state1.b));
  EXPECT(assert_equal(abc_examples::B1, state1.S));

  // Test identity
  State identityState = State::identity();
  EXPECT(assert_equal(identityState.R, Rot3()));
  EXPECT(assert_equal<Vector3>(identityState.b, Z_3x1));
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
  EXPECT(assert_equal(retracted_state2.R, state2.R));
  EXPECT(assert_equal(retracted_state2.b, state2.b));
  EXPECT(assert_equal(retracted_state2.S, state2.S));

  // Test localCoordinates at identity
  Vector expected_identity_local = Vector::Zero(6 + 3 * 2);
  EXPECT(assert_equal(identityState.localCoordinates(identityState),
                      expected_identity_local));

  // Test retract at identity
  Vector v_test = Vector::Zero(6 + 3 * 2);
  v_test.head<3>() << 0.1, 0.2, 0.3;         // R
  v_test.segment<3>(3) << 0.01, 0.02, 0.03;  // b
  v_test.segment<3>(6) << 0.05, 0.06, 0.07;  // S[0]
  v_test.segment<3>(9) << 0.08, 0.09, 0.10;  // S[1]

  State retracted_from_id = identityState.retract(v_test);
  EXPECT(assert_equal(retracted_from_id.R, Rot3::Expmap(v_test.head<3>())));
  EXPECT(assert_equal(retracted_from_id.b, v_test.segment<3>(3).eval()));
  EXPECT(assert_equal(retracted_from_id.S[0],
                      Rot3::Expmap(v_test.segment<3>(6).eval())));
  EXPECT(assert_equal(retracted_from_id.S[1],
                      Rot3::Expmap(v_test.segment<3>(9).eval())));

  // Test retract invalid argument
  CHECK_EXCEPTION(identityState.retract(Vector::Zero(1)),
                  std::invalid_argument);
}

/* ************************************************************************* */
TEST(ABC, GroupOperations) {
  using namespace abc_examples;

  // Test group multiplication
  Group g1_g2 = g1 * g2;
  EXPECT(assert_equal(g1_g2.A(), A1 * A2));
  Vector3 expected_a = t1 + A1.matrix() * t2;
  EXPECT(assert_equal(g1_g2.a(), expected_a));
  EXPECT(assert_equal(g1_g2.calibrations()[0], B1[0] * B2[0]));
  EXPECT(assert_equal(g1_g2.calibrations()[1], B1[1] * B2[1]));

  // Test inverse
  Group g1_inv = g1.inverse();
  EXPECT(assert_equal(g1_inv.A(), A1.inverse()));
  Vector3 expected_a_inv = -A1.inverse().matrix() * t1;
  EXPECT(assert_equal(g1_inv.a(), expected_a_inv));
  EXPECT(assert_equal(g1_inv.calibrations()[0], B1[0].inverse()));
  EXPECT(assert_equal(g1_inv.calibrations()[1], B1[1].inverse()));

  // Test g * g.inv() == identity
  Group identity_check = g1 * g1_inv;
  Group expected_identity = Group::Identity();
  EXPECT(assert_equal(identity_check.A(), expected_identity.A()));
  EXPECT(assert_equal(identity_check.a(), expected_identity.a()));
  EXPECT(assert_equal(identity_check.calibrations(),
                      expected_identity.calibrations()));

  // Test Expmap and Logmap
  Group::TangentVector v_tangent = Group::TangentVector::Zero();
  v_tangent.head<3>() << 0.1, 0.2, 0.3;         // For A
  v_tangent.segment<3>(3) << 0.01, 0.02, 0.03;  // For 'a' part
  v_tangent.segment<3>(6) << 0.04, 0.05, 0.06;  // For B[0]
  v_tangent.segment<3>(9) << 0.07, 0.08, 0.09;  // For B[1]

  Group g_exp = Group::Expmap(v_tangent);
  EXPECT(assert_equal(Group::Logmap(g_exp), v_tangent));

  // Test retract on G
  Group g_retracted = g1.retract(v_tangent);

  const Group composed = g1 * Group::Expmap(v_tangent);
  EXPECT(assert_equal(g_retracted.A(), composed.A()));
  EXPECT(assert_equal(g_retracted.a(), composed.a()));
  EXPECT(assert_equal(g_retracted.calibrations(), composed.calibrations()));

  // Test traits for G
  const Group identity = Group::Identity();
  EXPECT(assert_equal(traits<Group>::Identity().A(), identity.A()));
  EXPECT(assert_equal(traits<Group>::Identity().a(), identity.a()));
  EXPECT(assert_equal(traits<Group>::Identity().calibrations(),
                      identity.calibrations()));
  // testLie<Group>(g1, g2);
}

//******************************************************************************
TEST(ABC, AdjointMap) {
  using namespace abc_examples;

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
TEST(ABC, StateAction) {
  using namespace abc_examples;

  // Test State Action (G * State)
  State transformed_xi = abc::StateAction<2>(xi2)(g1);
  EXPECT(assert_equal(transformed_xi.R, xi2.R * g1.A()));
  EXPECT(assert_equal<Matrix>(transformed_xi.b,
                              g1.A().inverse().matrix() * (xi2.b - g1.a())));
  EXPECT(assert_equal(transformed_xi.S[0],
                      g1.A().inverse() * xi2.S[0] * g1.calibrations()[0],
                      1e-9));
  EXPECT(assert_equal(transformed_xi.S[1],
                      g1.A().inverse() * xi2.S[1] * g1.calibrations()[1],
                      1e-9));
}

/* ************************************************************************* */
// A right action satisfies φ(xi, g1 * g2) = φ(φ(xi, g1), g2), i.e. applying
// g1 then g2 equals applying their product, with the multiplication happening
// on the right.
TEST(ABC, StateActionIsRightAction) {
  using namespace abc_examples;

  // Create action functors
  abc::StateAction<2> phi_xi1(xi1);

  // Left side: apply composed group element (g1 * g2) to xi
  const State left_side = phi_xi1(g1 * g2);

  // Right side: apply g1 first, then g2 to the result
  const State xi1_g1 = phi_xi1(g1);
  const State right_side = abc::StateAction<2>(xi1_g1)(g2);

  // For a right action, these should be equal
  EXPECT(assert_equal(left_side, right_side));

  // Additional test with g1 and g2 reversed
  const State left_side_2 = phi_xi1(g2 * g1);

  const State xi1_g2 = phi_xi1(g2);
  const State right_side_2 = abc::StateAction<2>(xi1_g2)(g1);

  EXPECT(assert_equal(left_side_2, right_side_2));
}

/* ************************************************************************* */
TEST(ABC, StateActionJacobianAnalytic) {
  using namespace abc_examples;

  abc::StateAction<2> action_xi1(xi1);
  Matrix analytic1 = action_xi1.JacobianAtIdentity();
  Matrix numerical1 = gtsam::numericalDerivative11<State, Group>(
      [&](const Group& g) { return action_xi1(g); }, Group::Identity());
  EXPECT(assert_equal(analytic1, numerical1));

  abc::StateAction<2> action_xi2(xi2);
  Matrix analytic2 = action_xi2.JacobianAtIdentity();
  Matrix numerical2 = gtsam::numericalDerivative11<State, Group>(
      [&](const Group& g) { return action_xi2(g); }, Group::Identity());
  EXPECT(assert_equal(analytic2, numerical2));
}

/* ************************************************************************* */
TEST(ABC, InputAction) {
  using namespace abc_examples;

  abc::InputAction<2> psi_u(u);

  Vector6 transformed_u = psi_u(g1);
  EXPECT(assert_equal<Vector>(transformed_u.head<3>(),
                              g1.A().unrotate(omega - g1.a())));
  EXPECT(assert_equal<Vector>(transformed_u.tail<3>(), Z_3x1,
                              1e-9));  // Virtual input stays zero

  EXPECT(assert_equal(transformed_u, abc::InputAction<2>(u)(g1)));
}

/* ************************************************************************* */
// A right action satisfies φ(xi, g1 * g2) = φ(φ(xi, g1), g2), i.e. applying
// g1 then g2 equals applying their product, with the multiplication happening
// on the right.
TEST(ABC, InputActionIsRightAction) {
  using namespace abc_examples;

  // Create action functors
  abc::InputAction<2> psi_u(u);

  // Left side: apply composed group element (g1 * g2) to xi
  const Vector6 left_side = psi_u(g1 * g2);

  // Right side: apply g1 first, then g2 to the result
  const Vector6 u_g1 = psi_u(g1);
  const Vector6 right_side = abc::InputAction<2>(u_g1)(g2);
  // For a right action, these should be equal
  EXPECT(assert_equal(left_side, right_side));

  // Additional test with g1 and g2 reversed
  const Vector6 left_side_2 = psi_u(g2 * g1);

  const Vector6 u_g2 = psi_u(g2);
  const Vector6 right_side_2 = abc::InputAction<2>(u_g2)(g1);

  EXPECT(assert_equal(left_side_2, right_side_2));
}

/* ************************************************************************* */
TEST(ABC, InputActionJacobianAnalytic) {
  using namespace abc_examples;

  // Create action functors
  abc::InputAction<2> psi_u(u);
  Matrix analytic = psi_u.JacobianAtIdentity();
  Matrix numerical = gtsam::numericalDerivative11<Vector6, Group>(
      [&](const Group& g) { return psi_u(g); }, Group::Identity());

  EXPECT(assert_equal(analytic, numerical));
}

/* ************************************************************************* */
TEST(ABC, Geometry_lift) {
  State xi = abc_examples::xi1;

  // Setup input
  Vector3 omega(0.5, 0.6, 0.7);
  Vector6 u = abc::toInputVector(omega);
  typename Group::TangentVector L = Geometry::lift(xi, u);

  // Expected values
  Vector3 expected_L_head = omega - xi.b;
  Vector3 expected_L_segment3 = -Rot3::Hat(omega) * xi.b;
  Vector3 expected_L_segment6_0 = xi.S[0].inverse().matrix() * expected_L_head;
  Vector3 expected_L_segment6_1 = xi.S[1].inverse().matrix() * expected_L_head;

  EXPECT(assert_equal<Vector>(L.head<3>(), expected_L_head));
  EXPECT(assert_equal<Vector>(L.segment<3>(3), expected_L_segment3));
  EXPECT(assert_equal<Vector>(L.segment<3>(6), expected_L_segment6_0));
  EXPECT(assert_equal<Vector>(L.segment<3>(9), expected_L_segment6_1));
}

/* ************************************************************************* */
TEST(ABC, Geometry_stateMatrixA) {
  using namespace abc_examples;

  // Setup input
  Matrix A_matrix = Geometry::stateMatrixA(g1, u);
  abc::InputAction<2> psi_u(u);
  Matrix3 W0 = Rot3::Hat(psi_u(g1.inverse()).head<3>());

  Matrix expected_A1 = Matrix::Zero(6, 6);
  expected_A1.block<3, 3>(0, 3) = -I_3x3;
  expected_A1.block<3, 3>(3, 3) = W0;

  Matrix expected_A2 = gtsam::diag({W0, W0});
  Matrix expected_A_matrix = gtsam::diag({expected_A1, expected_A2});

  EXPECT(assert_equal(A_matrix, expected_A_matrix));
}

/* ************************************************************************* */
TEST(ABC, Geometry_stateTransitionMatrix) {
  Group X_hat = abc_examples::g1;

  // Setup input
  Vector3 omega(0.5, 0.6, 0.7);
  double dt = 0.1;

  Vector6 u = abc::toInputVector(omega);
  Matrix Phi = Geometry::stateTransitionMatrix(X_hat, u, dt);
  Matrix3 W0 = Rot3::Hat(abc::InputAction<2>(u)(X_hat.inverse()).head<3>());
  Matrix Phi1 = Matrix::Zero(6, 6);
  Matrix3 Phi12 = -dt * (I_3x3 + (dt / 2) * W0 + ((dt * dt) / 6) * W0 * W0);
  Matrix3 Phi22 = I_3x3 + dt * W0 + ((dt * dt) / 2) * W0 * W0;

  Phi1.block<3, 3>(0, 0) = I_3x3;
  Phi1.block<3, 3>(0, 3) = Phi12;
  Phi1.block<3, 3>(3, 3) = Phi22;
  Matrix Phi2 = gtsam::diag({Phi22, Phi22});
  Matrix expected_Phi = gtsam::diag({Phi1, Phi2});

  EXPECT(assert_equal(Phi, expected_Phi));
}

/* ************************************************************************* */
TEST(ABC, Geometry_inputMatrix) {
  Group X_hat = abc_examples::g1;

  Matrix input_matrix = Geometry::inputMatrix(X_hat);

  const Matrix3 X_hat_rot = X_hat.A().matrix();
  Matrix expected_B1 = gtsam::diag({X_hat_rot, X_hat_rot});
  Matrix expected_B2(3 * 2, 3 * 2);
  expected_B2.setZero();
  for (size_t i = 0; i < 2; ++i) {
    expected_B2.block<3, 3>(3 * i, 3 * i) = X_hat.calibrations()[i].matrix();
  }
  Matrix expected_input_matrix = gtsam::diag({expected_B1, expected_B2});

  EXPECT(assert_equal(input_matrix, expected_input_matrix));
}

/* ************************************************************************* */
TEST(ABC, Geometry_processNoise) {
  Matrix Sigma = (Matrix(6, 6) << 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 6)
                     .finished();

  Matrix Q = Geometry::processNoise(Sigma);

  Matrix expected_Q_cal_part = 1e-9 * I_6x6;
  Matrix expected_Q = gtsam::diag({Sigma, expected_Q_cal_part});

  EXPECT(assert_equal(Q, expected_Q));
}

/* ************************************************************************* */
TEST(ABC, Geometry_inputMatrixBt) {
  // This function is identical to inputMatrix, so we'll test its output matches
  // inputMatrix.
  Group X_hat = abc_examples::g1;

  Matrix input_matrix_Bt = Geometry::inputMatrixBt(X_hat);
  Matrix input_matrix =
      Geometry::inputMatrix(X_hat);  // Reference from the other function

  EXPECT(assert_equal(input_matrix_Bt, input_matrix));
}

/* ************************************************************************* */
TEST(ABC, OutputAction) {
  using namespace abc_examples;

  Unit3 y(1, 0, 0);

  // Test outputAction (calibrated sensor)
  int cal_idx = 0;
  abc::OutputAction<2> phi_y(y, cal_idx);
  Vector3 transformed_y_calibrated = phi_y(g1);
  EXPECT(assert_equal<Vector>(transformed_y_calibrated,
                              g1.calibrations()[0].unrotate(y.unitVector())));

  // Test outputAction (uncalibrated sensor)
  int uncalibrated_idx = -1;
  abc::OutputAction<2> uncalibrated_phi_y(y, uncalibrated_idx);
  Vector3 transformed_y_uncalibrated = uncalibrated_phi_y(g1);
  EXPECT(assert_equal<Vector>(transformed_y_uncalibrated,
                              g1.A().unrotate(y.unitVector())));

  CHECK_EXCEPTION(abc::OutputAction<2>(y, 2), std::out_of_range);
}

/* ************************************************************************* */
// A right action satisfies φ(y, g1 * g2) = φ(φ(y, g1), g2), mirroring the
// behavior tested for state and input actions.
TEST(ABC, OutputActionIsRightAction) {
  using namespace abc_examples;

  Unit3 y_meas(1, 0, 0);
  abc::OutputAction<2> phi_y(y_meas, 0);

  const Vector3 left_side = phi_y(g1 * g2);

  Unit3 y_g1(phi_y(g1));
  const Vector3 right_side = abc::OutputAction<2>(y_g1, 0)(g2);
  EXPECT(assert_equal(left_side, right_side));

  const Vector3 left_side_2 = phi_y(g2 * g1);
  Unit3 y_g2(phi_y(g2));
  const Vector3 right_side_2 = abc::OutputAction<2>(y_g2, 0)(g1);

  EXPECT(assert_equal(left_side_2, right_side_2));
}

/* ************************************************************************* */
TEST(ABC, OutputActionJacobianAnalytic) {
  using namespace abc_examples;

  Unit3 y_meas(1, 0, 0);
  abc::OutputAction<2> phi_y(y_meas, 0);
  Matrix analytic = phi_y.JacobianAtIdentity();
  Matrix numerical = gtsam::numericalDerivative11<Vector3, Group>(
      [&](const Group& g) { return phi_y(g); }, Group::Identity());

  EXPECT(assert_equal(analytic, numerical));
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

  EXPECT(assert_equal(C_cal, expected_C_cal));
}

/* ************************************************************************* */
TEST(ABC, EqFilter) {
  using G = Group;
  const State xi_ref = abc_examples::xi1;  // Reference state (xi circle)
  const int numSensors = 2;

  Matrix initialSigma = Matrix::Identity(G::dimension, G::dimension);
  initialSigma.diagonal().head<3>() =
      Vector3::Constant(0.1);  // Attitude uncertainty
  initialSigma.diagonal().segment<3>(3) =
      Vector3::Constant(0.01);  // Bias uncertainty
  initialSigma.diagonal().tail<3>() =
      Vector3::Constant(0.1);  // Calibration uncertainty

  const G g_0;
  EqF<State, abc::StateAction<2>> filter(g_0, xi_ref, initialSigma, numSensors);

  // Check initial state
  EXPECT(assert_equal(g_0, filter.state()));
  EXPECT(assert_equal(g_0, filter.groupEstimate()));

  // Perform a prediction step
  Vector3 omega(0.01, -0.02, 0.015);
  Matrix Sigma = I_6x6;
  double dt = 0.01;

  Vector6 u_vec = abc::toInputVector(omega);
  Matrix Q = Geometry::processNoise(Sigma);
  filter.predict<Geometry>(u_vec, Q, dt);

  // Regression
  Group expected({Rot3(1, 0.00015, -0.0004,  //
                       -0.00015, 1, 3e-08,   //
                       0.0004, 3e-08, 1),
                  Point3(9.00091e-06, 1.49932e-06, -3.9982e-06)},
                 {Rot3(1, 0.000149811, -0.000400001,   //
                       -0.000149814, 1, -7.46691e-06,  //
                       0.000399999, 7.52684e-06, 1),
                  Rot3(1, 0.000150005, -0.000399278,  //
                       -0.000149995, 1, 2.40155e-05,  //
                       0.000399282, -2.39557e-05, 1)});
  EXPECT(assert_equal(expected, filter.groupEstimate(), 1e-4));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
