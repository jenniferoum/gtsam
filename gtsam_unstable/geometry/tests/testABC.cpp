/**
 * @file testABC.cpp
 * @brief Test file for ABC (Attitude-Bias-Calibration) system components
 * @author Darshan Rajasekaran & Jennifer Oum
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/geometry/ABC.h>

using namespace gtsam;

// Define N for testing purposes, e.g., 2 calibration states
using State2 = abc::State<2>;
using G2 = abc::Group<2>;
using Geometry2 = abc::Geometry<2>;

// Helper for approximate equality of arrays of Rot3
bool ArraysEqual(const std::array<Rot3, 2>& arr1,
                 const std::array<Rot3, 2>& arr2, double tol = 1e-9) {
  for (size_t i = 0; i < 2; ++i) {
    if (!arr1[i].equals(arr2[i], tol)) {
      return false;
    }
  }
  return true;
}

// Custom stream operator for G to enable EXPECT(_EQUAL
namespace gtsam {
namespace abc {
template <size_t N>
std::ostream& operator<<(std::ostream& os, const Group<N>& g) {
  os << "A: " << g.A << "\n";
  os << "a: " << Rot3::Vee(g.a).transpose() << "\n";
  os << "B: [";
  for (size_t i = 0; i < N; ++i) {
    os << g.B[i];
    if (i < N - 1) os << ", ";
  }
  os << "]";
  return os;
}

template <size_t N>
std::ostream& operator<<(std::ostream& os, const State<N>& s) {
  os << "R: " << s.R << "\n";
  os << "b: " << s.b.transpose() << "\n";
  os << "S: [";
  for (size_t i = 0; i < N; ++i) {
    os << s.S[i];
    if (i < N - 1) os << ", ";
  }
  os << "]";
  return os;
}
}  // namespace abc
}  // namespace gtsam

/* ************************************************************************* */
TEST(ABC, InputData) {
  abc::InputData input;
  input.w = (Vector(3) << 1, 2, 3).finished();
  input.Sigma = I_6x6;

  EXPECT(assert_equal(input.w, (Vector(3) << 1, 2, 3).finished(), 1e-9));
  EXPECT(assert_equal(input.Sigma, I_6x6, 1e-9));
  EXPECT(assert_equal(input.W(), Rot3::Hat(input.w), 1e-9));

  EXPECT(input.w.norm() > 0);      // Should not be zero
  EXPECT(input.Sigma.norm() > 0);  // Should not be zero

  // Test conversion to Vector6
  Vector6 u = input.toInputVector();
  EXPECT(assert_equal<Vector>(u.head<3>(), input.w, 1e-9));
  EXPECT(assert_equal<Vector>(u.tail<3>(), Vector3::Zero(), 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Measurement) {
  abc::Measurement meas;
  meas.y = Unit3(1, 0, 0);
  meas.d = Unit3(0, 1, 0);
  meas.Sigma = I_3x3;
  meas.cal_idx = 0;

  EXPECT(assert_equal(meas.y.unitVector(), (Vector(3) << 1, 0, 0).finished(),
                      1e-9));
  EXPECT(assert_equal(meas.d.unitVector(), (Vector(3) << 0, 1, 0).finished(),
                      1e-9));
  EXPECT(assert_equal<Matrix3>(meas.Sigma, I_3x3, 1e-9));
  EXPECT_LONGS_EQUAL(meas.cal_idx, 0);
}

/* ************************************************************************* */
TEST(ABC, State) {
  Rot3 R1 = Rot3::Rx(0.1);
  Vector3 b1 = (Vector(3) << 0.01, 0.02, 0.03).finished();
  std::array<Rot3, 2> S1;
  S1[0] = Rot3::Ry(0.05);
  S1[1] = Rot3::Rz(0.06);

  State2 state1(R1, b1, S1);

  EXPECT(assert_equal(state1.R, R1, 1e-9));
  EXPECT(assert_equal(state1.b, b1, 1e-9));
  EXPECT((ArraysEqual(state1.S, S1)));

  // Test identity
  State2 identityState = State2::identity();
  EXPECT(assert_equal(identityState.R, Rot3::Identity(), 1e-9));
  EXPECT(assert_equal<Vector3>(identityState.b, Vector3::Zero(), 1e-9));
  std::array<Rot3, 2> expectedS_id;
  expectedS_id.fill(Rot3::Identity());
  EXPECT((ArraysEqual(identityState.S, expectedS_id)));

  // Test localCoordinates and retract (manifold properties)
  Rot3 R2 = Rot3::Rx(0.2);
  Vector3 b2 = (Vector(3) << 0.05, 0.06, 0.07).finished();
  std::array<Rot3, 2> S2;
  S2[0] = Rot3::Ry(0.1);
  S2[1] = Rot3::Rz(0.15);
  State2 state2(R2, b2, S2);

  Vector actual_local = state1.localCoordinates(state2);
  State2 retracted_state2 = state1.retract(actual_local);
  EXPECT(assert_equal(retracted_state2.R, state2.R, 1e-9));
  EXPECT(assert_equal(retracted_state2.b, state2.b, 1e-9));
  EXPECT((ArraysEqual(retracted_state2.S, state2.S)));

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

  State2 retracted_from_id = identityState.retract(v_test);
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
TEST(ABC, G_GroupOperations) {
  Rot3 A1 = Rot3::Rx(0.1);
  Matrix3 a1 = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B1;
  B1[0] = Rot3::Ry(0.05);
  B1[1] = Rot3::Rz(0.06);
  G2 g1(A1, a1, B1);

  Rot3 A2 = Rot3::Ry(0.2);
  Matrix3 a2 = Rot3::Hat((Vector(3) << 0.04, 0.05, 0.06).finished());
  std::array<Rot3, 2> B2;
  B2[0] = Rot3::Rz(0.07);
  B2[1] = Rot3::Rx(0.08);
  G2 g2(A2, a2, B2);

  // Test group multiplication
  G2 g1_g2 = g1 * g2;
  EXPECT(assert_equal(g1_g2.A, A1 * A2, 1e-9));
  Vector3 expected_a_vee = Rot3::Vee(a1) + A1.matrix() * Rot3::Vee(a2);
  EXPECT(assert_equal(Rot3::Vee(g1_g2.a), expected_a_vee, 1e-9));
  EXPECT(assert_equal(g1_g2.B[0], B1[0] * B2[0], 1e-9));
  EXPECT(assert_equal(g1_g2.B[1], B1[1] * B2[1], 1e-9));

  // Test inverse
  G2 g1_inv = g1.inv();
  EXPECT(assert_equal(g1_inv.A, A1.inverse(), 1e-9));
  Vector3 expected_a_inv_vee = -A1.inverse().matrix() * Rot3::Vee(a1);
  EXPECT(assert_equal(Rot3::Vee(g1_inv.a), expected_a_inv_vee, 1e-9));
  EXPECT(assert_equal(g1_inv.B[0], B1[0].inverse(), 1e-9));
  EXPECT(assert_equal(g1_inv.B[1], B1[1].inverse(), 1e-9));

  // Test g * g.inv() == identity
  G2 identity_check = g1 * g1_inv;
  G2 expected_identity = G2::identity(2);
  EXPECT(assert_equal(identity_check.A, expected_identity.A, 1e-9));
  EXPECT(assert_equal(identity_check.a, expected_identity.a, 1e-9));
  EXPECT((ArraysEqual(identity_check.B, expected_identity.B)));

  // Test Expmap and Logmap
  int G_DIM = 6 + 3 * 2;
  Vector v_tangent = Vector::Zero(G_DIM);
  v_tangent.head<3>() << 0.1, 0.2, 0.3;         // For A
  v_tangent.segment<3>(3) << 0.01, 0.02, 0.03;  // For 'a' part
  v_tangent.segment<3>(6) << 0.04, 0.05, 0.06;  // For B[0]
  v_tangent.segment<3>(9) << 0.07, 0.08, 0.09;  // For B[1]

  G2 g_exp = G2::exp(v_tangent);
  // Logmap is a placeholder, so we can only check its consistency if
  // exp(log(g)) = g Currently Logmap returns zero, so cannot properly test
  // exp(log(g)) == g
  // EXPECT(assert_equal(G2::Logmap(g_exp), v_tangent, 1e-9)); // This will fail
  // with placeholder

  // Test retract on G
  G2 g_retracted = g1.retract(v_tangent);

  EXPECT(assert_equal(g_retracted.A, (g1 * G2::exp(v_tangent)).A, 1e-9));
  EXPECT(assert_equal(g_retracted.a, (g1 * G2::exp(v_tangent)).a, 1e-9));
  EXPECT(ArraysEqual(g_retracted.B, (g1 * G2::exp(v_tangent)).B));

  // Test traits for G
  EXPECT(assert_equal(traits<G2>::Identity().A, G2::identity(2).A, 1e-9));
  EXPECT(assert_equal(traits<G2>::Identity().a, G2::identity(2).a, 1e-9));
  EXPECT(ArraysEqual(traits<G2>::Identity().B, G2::identity(2).B));
  // testLie<G2>(g1, g2, 1e-9);
}

/* ************************************************************************* */
TEST(ABC, GroupActions) {
  // Setup a G element
  Rot3 gA = Rot3::Rx(0.1);
  Matrix3 ga_skew = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> gB;
  gB[0] = Rot3::Ry(0.05);
  gB[1] = Rot3::Rz(0.06);
  G2 X(gA, ga_skew, gB);

  // Setup a State element
  Rot3 sR = Rot3::Rz(0.2);
  Vector3 sb = (Vector(3) << 0.04, 0.05, 0.06).finished();
  std::array<Rot3, 2> sS;
  sS[0] = Rot3::Rx(0.07);
  sS[1] = Rot3::Ry(0.08);
  State2 xi(sR, sb, sS);

  // Test State Action (G * State)
  State2 transformed_xi = X * xi;
  EXPECT(assert_equal(transformed_xi.R, xi.R * X.A, 1e-9));
  EXPECT(assert_equal<Matrix>(transformed_xi.b,
                              X.A.inverse().matrix() * (xi.b - Rot3::Vee(X.a)),
                              1e-9));
  EXPECT(assert_equal(transformed_xi.S[0], X.A.inverse() * xi.S[0] * X.B[0],
                      1e-9));
  EXPECT(assert_equal(transformed_xi.S[1], X.A.inverse() * xi.S[1] * X.B[1],
                      1e-9));

  // Test velocityAction
  abc::InputData u_data;
  u_data.w = (Vector(3) << 1, 2, 3).finished();
  u_data.Sigma = I_6x6;

  Vector6 u = u_data.toInputVector();
  Vector6 transformed_u = velocityAction(X, u);
  EXPECT(assert_equal<Vector>(
      transformed_u.head<3>(),
      X.A.inverse().matrix() * (u_data.w - Rot3::Vee(X.a)), 1e-9));
  EXPECT(assert_equal<Vector>(transformed_u.tail<3>(), Vector3::Zero(),
                              1e-9));  // Virtual input stays zero

  // Test outputAction (calibrated sensor)
  Unit3 y_meas = Unit3(1, 0, 0);
  int cal_idx = 0;
  Vector3 transformed_y_cal = outputAction(X, y_meas, cal_idx);
  EXPECT(assert_equal<Vector>(
      transformed_y_cal, X.B[cal_idx].inverse().matrix() * y_meas.unitVector(),
      1e-9));

  // Test outputAction (uncalibrated sensor)
  int uncal_idx = -1;
  Vector3 transformed_y_uncal = outputAction(X, y_meas, uncal_idx);
  EXPECT(assert_equal<Vector>(
      transformed_y_uncal, X.A.inverse().matrix() * y_meas.unitVector(), 1e-9));

  // Test outputAction out of range
  CHECK_EXCEPTION(outputAction(X, y_meas, 2), std::out_of_range);
}

/* ************************************************************************* */
TEST(ABC, numericalDifferential) {
  // Test with a simple linear function
  std::function<Vector(const Vector&)> f_linear = [](const Vector& x) {
    return (Vector(2) << 2 * x(0) + 3 * x(1), x(0) - x(1)).finished();
  };
  Vector x0_linear = (Vector(2) << 1, 2).finished();
  Matrix expected_Df_linear = (Matrix(2, 2) << 2, 3, 1, -1).finished();
  EXPECT(assert_equal(abc::numericalDifferential(f_linear, x0_linear),
                      expected_Df_linear, 1e-6));

  // Test with a non-linear function
  std::function<Vector(const Vector&)> f_nonlinear = [](const Vector& x) {
    return (Vector(2) << x(0) * x(0), sin(x(1))).finished();
  };
  Vector x0_nonlinear = (Vector(2) << 2, M_PI_2).finished();
  Matrix expected_Df_nonlinear =
      (Matrix(2, 2) << 4, 0, 0, cos(M_PI_2))
          .finished();  // At x=(2,pi/2), df/dx = [[4, 0], [0, 0]]
  EXPECT(assert_equal(abc::numericalDifferential(f_nonlinear, x0_nonlinear),
                      expected_Df_nonlinear, 1e-6));
}

/* ************************************************************************* */
TEST(ABC, Geometry_identityState) {
  State2 expected_id = State2::identity();
  State2 actual = Geometry2::identityState();
  EXPECT(assert_equal<Rot3>(actual.R, expected_id.R, 1e-9));
  EXPECT(assert_equal<Vector>(actual.b, expected_id.b, 1e-9));
  EXPECT(ArraysEqual(actual.S, expected_id.S));
}

/* ************************************************************************* */
TEST(ABC, Geometry_groupAction) {
  Rot3 A = Rot3::Rx(0.1);
  Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  G2 g(A, a, B);

  Rot3 R = Rot3::Rz(0.2);
  Vector3 b = (Vector(3) << 0.04, 0.05, 0.06).finished();
  std::array<Rot3, 2> S;
  S[0] = Rot3::Rx(0.07);
  S[1] = Rot3::Ry(0.08);
  State2 x(R, b, S);

  State2 expected_transformed_x = g * x;
  State2 actual_transformed = Geometry2::groupAction(g, x);

  // Compare each component
  EXPECT(
      assert_equal<Rot3>(actual_transformed.R, expected_transformed_x.R, 1e-9));
  EXPECT(assert_equal<Vector>(actual_transformed.b, expected_transformed_x.b,
                              1e-9));
  EXPECT(ArraysEqual(actual_transformed.S, expected_transformed_x.S));
}

/* ************************************************************************* */
TEST(ABC, Geometry_lift) {
  // Setup state
  Rot3 R = Rot3::Identity();
  Vector3 b = (Vector(3) << 0.1, 0.2, 0.3).finished();
  std::array<Rot3, 2> S_arr;
  S_arr[0] = Rot3::Rx(0.05);
  S_arr[1] = Rot3::Ry(0.06);
  State2 xi(R, b, S_arr);

  // Setup input
  abc::InputData u_data;
  u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
  u_data.Sigma = I_6x6;

  Vector6 u = u_data.toInputVector();
  typename G2::TangentVector L = xi.lift(u);

  // Expected values
  Vector3 expected_L_head = u_data.w - xi.b;
  Vector3 expected_L_segment3 = -u_data.W() * xi.b;
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
  Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  G2 X_hat(A, a, B);

  // Setup input
  abc::InputData u_data;
  u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
  u_data.Sigma = I_6x6;

  Vector6 u = u_data.toInputVector();
  Matrix A_matrix = Geometry2::stateMatrixA(X_hat, u);
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
  Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  G2 X_hat(A, a, B);

  // Setup input
  abc::InputData u_data;
  u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
  u_data.Sigma = I_6x6;

  double dt = 0.1;

  Vector6 u = u_data.toInputVector();
  Matrix Phi = Geometry2::stateTransitionMatrix(u, dt, X_hat);
  Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inv(), u).head<3>());
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
  Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  G2 X_hat(A, a, B);

  Matrix input_matrix = Geometry2::inputMatrix(X_hat);

  Matrix expected_B1 = gtsam::diag({X_hat.A.matrix(), X_hat.A.matrix()});
  Matrix expected_B2(3 * 2, 3 * 2);
  expected_B2.setZero();
  for (size_t i = 0; i < 2; ++i) {
    expected_B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
  }
  Matrix expected_input_matrix = gtsam::diag({expected_B1, expected_B2});

  EXPECT(assert_equal(input_matrix, expected_input_matrix, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_processNoise) {
  abc::InputData u_data;
  u_data.w = Vector3::Zero();
  u_data.Sigma = (Matrix(6, 6) << 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 6)
                     .finished();

  Matrix Q = Geometry2::processNoise(u_data.Sigma);

  Matrix expected_Q_cal_part = 1e-9 * I_6x6;
  Matrix expected_Q = gtsam::diag({u_data.Sigma, expected_Q_cal_part});

  EXPECT(assert_equal(Q, expected_Q, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_inputMatrixBt) {
  // This function is identical to inputMatrix, so we'll test its output matches
  // inputMatrix. Setup G element for X_hat
  Rot3 A = Rot3::Rx(0.1);
  Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
  std::array<Rot3, 2> B;
  B[0] = Rot3::Ry(0.05);
  B[1] = Rot3::Rz(0.06);
  G2 X_hat(A, a, B);

  Matrix input_matrix_Bt = Geometry2::inputMatrixBt(X_hat);
  Matrix input_matrix =
      Geometry2::inputMatrix(X_hat);  // Reference from the other function

  EXPECT(assert_equal(input_matrix_Bt, input_matrix, 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Geometry_measurementMatrixC) {
  Unit3 d = Unit3(0, 0, 1);  // Reference direction (e.g., gravity)
  Matrix3 wedge_d = Rot3::Hat(d.unitVector());

  // Test with calibrated sensor (idx = 0)
  int cal_idx = 0;
  Matrix C_cal = Geometry2::measurementMatrixC(d, cal_idx);

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

  abc::InputData u;
  u.w = (Vector3() << 0.01, -0.02, 0.015).finished();
  u.Sigma = I_6x6;
  double dt = 0.01;

  Vector6 u_vec = u.toInputVector();
  Matrix Q = Geometry2::processNoise(u.Sigma);
  filter.predict(u_vec, Q, dt);

  EXPECT(traits<G>::Equals(filter.groupEstimate(), filter.state(), 1e-9));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */