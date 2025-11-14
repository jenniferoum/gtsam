/**
 * @file testABC.cpp
 * @brief Test file for ABC (Attitude-Bias-Calibration) system components
 * @author Darshan Rajasekaran & Jennifer Oum
 */

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <CppUnitLite/TestHarness.h>

#include <gtsam_unstable/geometry/ABC.h>
#include <gtsam/navigation/EquivariantFilter.h>

using namespace gtsam;
using namespace gtsam::abc_eqf_lib;
using namespace std;

// Define N for testing purposes, e.g., 2 calibration states
static const size_t N_TEST = 2;
using StateN = State<N_TEST>;
using GN = Group<N_TEST>;
using ABCGeometryN = ABCGeometry<N_TEST>;

// Helper for approximate equality of arrays of Rot3
bool ArraysEqual(const std::array<Rot3, N_TEST>& arr1, const std::array<Rot3, N_TEST>& arr2, double tol = 1e-9) {
    for (size_t i = 0; i < N_TEST; ++i) {
        if (!arr1[i].equals(arr2[i], tol)) {
            return false;
        }
    }
    return true;
}

// Custom stream operator for G to enable EXPECT(_EQUAL
namespace gtsam {
namespace abc_eqf_lib {
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
} // namespace abc_eqf_lib
} // namespace gtsam


/* ************************************************************************* */
TEST(ABC, blockDiag) {
    Matrix A = (Matrix(2, 2) << 1, 2, 3, 4).finished();
    Matrix B = (Matrix(3, 3) << 5, 6, 7, 8, 9, 10, 11, 12, 13).finished();
    Matrix expected = (Matrix(5, 5) <<
        1, 2, 0, 0, 0,
        3, 4, 0, 0, 0,
        0, 0, 5, 6, 7,
        0, 0, 8, 9, 10,
        0, 0, 11, 12, 13).finished();
    EXPECT(assert_equal(blockDiag(A, B), expected, 1e-9));

    EXPECT(assert_equal(blockDiag(Matrix(), A), A, 1e-9));
    EXPECT(assert_equal(blockDiag(Matrix(), B), B, 1e-9));
    EXPECT(assert_equal(blockDiag(Matrix(), Matrix()), Matrix(), 1e-9));
}

/* ************************************************************************* */
TEST(ABC, repBlock) {
    Matrix A = (Matrix(2, 2) << 1, 2, 3, 4).finished();
    Matrix expected2 = (Matrix(4, 4) <<
        1, 2, 0, 0,
        3, 4, 0, 0,
        0, 0, 1, 2,
        0, 0, 3, 4).finished();
    EXPECT(assert_equal(repBlock(A, 2), expected2, 1e-9));

    Matrix expected3 = (Matrix(6, 6) <<
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 0, 0,
        0, 0, 1, 2, 0, 0,
        0, 0, 3, 4, 0, 0,
        0, 0, 0, 0, 1, 2,
        0, 0, 0, 0, 3, 4).finished();
    EXPECT(assert_equal(repBlock(A, 3), expected3, 1e-9));
    EXPECT(assert_equal(repBlock(A, 1), A, 1e-9));
    EXPECT(assert_equal(repBlock(A, 0), Matrix(), 1e-9));
}

///* ************************************************************************* */
TEST(ABC, InputData) {
    InputData input;
    input.w = (Vector(3) << 1, 2, 3).finished();
    input.Sigma = I_6x6;

    EXPECT(assert_equal(input.w, (Vector(3) << 1, 2, 3).finished(), 1e-9));
    EXPECT(assert_equal(input.Sigma, I_6x6, 1e-9));
    EXPECT(assert_equal(input.W(), Rot3::Hat(input.w), 1e-9));

    EXPECT(input.w.norm() > 0); // Should not be zero
    EXPECT(input.Sigma.norm() > 0); // Should not be zero
    
    // Test conversion to Vector6
    Vector6 u = input.toInputVector();
    EXPECT(assert_equal<Vector>(u.head<3>(), input.w, 1e-9));
    EXPECT(assert_equal<Vector>(u.tail<3>(), Vector3::Zero(), 1e-9));
}

/* ************************************************************************* */
TEST(ABC, Measurement) {
    Measurement meas;
    meas.y = Unit3(1, 0, 0);
    meas.d = Unit3(0, 1, 0);
    meas.Sigma = I_3x3;
    meas.cal_idx = 0;

    EXPECT(assert_equal(meas.y.unitVector(), (Vector(3) << 1, 0, 0).finished(), 1e-9));
    EXPECT(assert_equal(meas.d.unitVector(), (Vector(3) << 0, 1, 0).finished(), 1e-9));
    EXPECT(assert_equal<Matrix3>(meas.Sigma, I_3x3, 1e-9));
    EXPECT_LONGS_EQUAL(meas.cal_idx, 0);
}

///* ************************************************************************* */
TEST(ABC, State) {
    Rot3 R1 = Rot3::Rx(0.1);
    Vector3 b1 = (Vector(3) << 0.01, 0.02, 0.03).finished();
    std::array<Rot3, N_TEST> S1;
    S1[0] = Rot3::Ry(0.05);
    S1[1] = Rot3::Rz(0.06);

    StateN state1(R1, b1, S1);

    EXPECT(assert_equal(state1.R, R1, 1e-9));
    EXPECT(assert_equal(state1.b, b1, 1e-9));
    EXPECT((ArraysEqual(state1.S, S1)));

    // Test identity
    StateN identityState = StateN::identity();
    EXPECT(assert_equal(identityState.R, Rot3::Identity(), 1e-9));
    EXPECT(assert_equal<Vector3>(identityState.b, Vector3::Zero(), 1e-9));
    std::array<Rot3, N_TEST> expectedS_id;
    expectedS_id.fill(Rot3::Identity());
    EXPECT((ArraysEqual(identityState.S, expectedS_id)));

    // Test localCoordinates and retract (manifold properties)
    Rot3 R2 = Rot3::Rx(0.2);
    Vector3 b2 = (Vector(3) << 0.05, 0.06, 0.07).finished();
    std::array<Rot3, N_TEST> S2;
    S2[0] = Rot3::Ry(0.1);
    S2[1] = Rot3::Rz(0.15);
    StateN state2(R2, b2, S2);

    Vector actual_local = state1.localCoordinates(state2);
    StateN retracted_state2 = state1.retract(actual_local);
    EXPECT(assert_equal(retracted_state2.R, state2.R, 1e-9));
    EXPECT(assert_equal(retracted_state2.b, state2.b, 1e-9));
    EXPECT((ArraysEqual(retracted_state2.S, state2.S)));

    // Test localCoordinates at identity
    Vector expected_identity_local = Vector::Zero(6 + 3 * N_TEST);
    EXPECT(assert_equal(identityState.localCoordinates(identityState), expected_identity_local, 1e-9));

    // Test retract at identity
    Vector v_test = Vector::Zero(6 + 3 * N_TEST);
    v_test.head<3>() << 0.1, 0.2, 0.3; // R
    v_test.segment<3>(3) << 0.01, 0.02, 0.03; // b
    v_test.segment<3>(6) << 0.05, 0.06, 0.07; // S[0]
    v_test.segment<3>(9) << 0.08, 0.09, 0.10; // S[1]

    StateN retracted_from_id = identityState.retract(v_test);
    EXPECT(assert_equal(retracted_from_id.R, Rot3::Expmap(v_test.head<3>()), 1e-9));
    EXPECT(assert_equal(retracted_from_id.b, v_test.segment<3>(3).eval(), 1e-9));
    EXPECT(assert_equal(retracted_from_id.S[0], Rot3::Expmap(v_test.segment<3>(6).eval()), 1e-9));
    EXPECT(assert_equal(retracted_from_id.S[1], Rot3::Expmap(v_test.segment<3>(9).eval()), 1e-9));

    // Test retract invalid argument
    CHECK_EXCEPTION(identityState.retract(Vector::Zero(1)), std::invalid_argument);
}

/* ************************************************************************* */
TEST(ABC, G_GroupOperations) {
    Rot3 A1 = Rot3::Rx(0.1);
    Matrix3 a1 = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
    std::array<Rot3, N_TEST> B1;
    B1[0] = Rot3::Ry(0.05);
    B1[1] = Rot3::Rz(0.06);
    GN g1(A1, a1, B1);

    Rot3 A2 = Rot3::Ry(0.2);
    Matrix3 a2 = Rot3::Hat((Vector(3) << 0.04, 0.05, 0.06).finished());
    std::array<Rot3, N_TEST> B2;
    B2[0] = Rot3::Rz(0.07);
    B2[1] = Rot3::Rx(0.08);
    GN g2(A2, a2, B2);

    // Test group multiplication
    GN g1_g2 = g1 * g2;
    EXPECT(assert_equal(g1_g2.A, A1 * A2, 1e-9));
    Vector3 expected_a_vee = Rot3::Vee(a1) + A1.matrix() * Rot3::Vee(a2);
    EXPECT(assert_equal(Rot3::Vee(g1_g2.a), expected_a_vee, 1e-9));
    EXPECT(assert_equal(g1_g2.B[0], B1[0] * B2[0], 1e-9));
    EXPECT(assert_equal(g1_g2.B[1], B1[1] * B2[1], 1e-9));

    // Test inverse
    GN g1_inv = g1.inv();
    EXPECT(assert_equal(g1_inv.A, A1.inverse(), 1e-9));
    Vector3 expected_a_inv_vee = -A1.inverse().matrix() * Rot3::Vee(a1);
    EXPECT(assert_equal(Rot3::Vee(g1_inv.a), expected_a_inv_vee, 1e-9));
    EXPECT(assert_equal(g1_inv.B[0], B1[0].inverse(), 1e-9));
    EXPECT(assert_equal(g1_inv.B[1], B1[1].inverse(), 1e-9));

    // Test g * g.inv() == identity
    GN identity_check = g1 * g1_inv;
    GN expected_identity = GN::identity(N_TEST);
    EXPECT(assert_equal(identity_check.A, expected_identity.A, 1e-9));
    EXPECT(assert_equal(identity_check.a, expected_identity.a, 1e-9));
    EXPECT((ArraysEqual(identity_check.B, expected_identity.B)));

    // Test Expmap and Logmap
    int G_DIM = 6 + 3 * N_TEST;
    Vector v_tangent = Vector::Zero(G_DIM);
    v_tangent.head<3>() << 0.1, 0.2, 0.3; // For A
    v_tangent.segment<3>(3) << 0.01, 0.02, 0.03; // For 'a' part
    v_tangent.segment<3>(6) << 0.04, 0.05, 0.06; // For B[0]
    v_tangent.segment<3>(9) << 0.07, 0.08, 0.09; // For B[1]

    GN g_exp = GN::exp(v_tangent);
    // Logmap is a placeholder, so we can only check its consistency if exp(log(g)) = g
    // Currently Logmap returns zero, so cannot properly test exp(log(g)) == g
    //EXPECT(assert_equal(GN::Logmap(g_exp), v_tangent, 1e-9)); // This will fail with placeholder

    // Test retract on G
    GN g_retracted = g1.retract(v_tangent);

    EXPECT(assert_equal(g_retracted.A, (g1 * GN::exp(v_tangent)).A, 1e-9));
	EXPECT(assert_equal(g_retracted.a, (g1 * GN::exp(v_tangent)).a, 1e-9));
	EXPECT(ArraysEqual(g_retracted.B, (g1 * GN::exp(v_tangent)).B));

    // Test traits for G
    EXPECT(assert_equal(traits<GN>::Identity().A, GN::identity(N_TEST).A, 1e-9));
	EXPECT(assert_equal(traits<GN>::Identity().a, GN::identity(N_TEST).a, 1e-9));
	EXPECT(ArraysEqual(traits<GN>::Identity().B, GN::identity(N_TEST).B));
    //testLie<GN>(g1, g2, 1e-9);
}

/* ************************************************************************* */
TEST(ABC, GroupActions) {
    // Setup a G element
    Rot3 gA = Rot3::Rx(0.1);
    Matrix3 ga_skew = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
    std::array<Rot3, N_TEST> gB;
    gB[0] = Rot3::Ry(0.05);
    gB[1] = Rot3::Rz(0.06);
    GN X(gA, ga_skew, gB);

    // Setup a State element
    Rot3 sR = Rot3::Rz(0.2);
    Vector3 sb = (Vector(3) << 0.04, 0.05, 0.06).finished();
    std::array<Rot3, N_TEST> sS;
    sS[0] = Rot3::Rx(0.07);
    sS[1] = Rot3::Ry(0.08);
    StateN xi(sR, sb, sS);

    // Test State Action (G * State)
    StateN transformed_xi = X * xi;
    EXPECT(assert_equal(transformed_xi.R, xi.R * X.A, 1e-9));
    EXPECT(assert_equal<Matrix>(transformed_xi.b, X.A.inverse().matrix() * (xi.b - Rot3::Vee(X.a)), 1e-9));
    EXPECT(assert_equal(transformed_xi.S[0], X.A.inverse() * xi.S[0] * X.B[0], 1e-9));
    EXPECT(assert_equal(transformed_xi.S[1], X.A.inverse() * xi.S[1] * X.B[1], 1e-9));

    // Test velocityAction
    InputData u_data;
    u_data.w = (Vector(3) << 1, 2, 3).finished();
    u_data.Sigma = I_6x6;
    
    Vector6 u = u_data.toInputVector();
    Vector6 transformed_u = velocityAction(X, u);
    EXPECT(assert_equal<Vector>(transformed_u.head<3>(), X.A.inverse().matrix() * (u_data.w - Rot3::Vee(X.a)), 1e-9));
    EXPECT(assert_equal<Vector>(transformed_u.tail<3>(), Vector3::Zero(), 1e-9)); // Virtual input stays zero

    // Test outputAction (calibrated sensor)
    Unit3 y_meas = Unit3(1, 0, 0);
    int cal_idx = 0;
    Vector3 transformed_y_cal = outputAction(X, y_meas, cal_idx);
    EXPECT(assert_equal<Vector>(transformed_y_cal, X.B[cal_idx].inverse().matrix() * y_meas.unitVector(), 1e-9));

    // Test outputAction (uncalibrated sensor)
    int uncal_idx = -1;
    Vector3 transformed_y_uncal = outputAction(X, y_meas, uncal_idx);
    EXPECT(assert_equal<Vector>(transformed_y_uncal, X.A.inverse().matrix() * y_meas.unitVector(), 1e-9));

    // Test outputAction out of range
    CHECK_EXCEPTION(outputAction(X, y_meas, N_TEST), std::out_of_range);
}

/* ************************************************************************* */
TEST(ABC, numericalDifferential) {
    // Test with a simple linear function
    std::function<Vector(const Vector&)> f_linear = [](const Vector& x) {
        return (Vector(2) << 2 * x(0) + 3 * x(1), x(0) - x(1)).finished();
    };
    Vector x0_linear = (Vector(2) << 1, 2).finished();
    Matrix expected_Df_linear = (Matrix(2, 2) << 2, 3, 1, -1).finished();
    EXPECT(assert_equal(numericalDifferential(f_linear, x0_linear), expected_Df_linear, 1e-6));

    // Test with a non-linear function
    std::function<Vector(const Vector&)> f_nonlinear = [](const Vector& x) {
        return (Vector(2) << x(0) * x(0), sin(x(1))).finished();
    };
    Vector x0_nonlinear = (Vector(2) << 2, M_PI_2).finished();
    Matrix expected_Df_nonlinear = (Matrix(2, 2) << 4, 0, 0, cos(M_PI_2)).finished(); // At x=(2,pi/2), df/dx = [[4, 0], [0, 0]]
    EXPECT(assert_equal(numericalDifferential(f_nonlinear, x0_nonlinear), expected_Df_nonlinear, 1e-6));
}

/* ************************************************************************* */
TEST(ABC, ABCGeometry_identityState) {
    StateN expected_id = StateN::identity();
    StateN actual = ABCGeometryN::identityState();
    EXPECT(assert_equal<Rot3>(actual.R, expected_id.R, 1e-9));
	EXPECT(assert_equal<Vector>(actual.b, expected_id.b, 1e-9));
	EXPECT(ArraysEqual(actual.S, expected_id.S));
}

///* ************************************************************************* */
TEST(ABC, ABCGeometry_groupAction) {
    Rot3 A = Rot3::Rx(0.1);
    Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
    std::array<Rot3, N_TEST> B;
    B[0] = Rot3::Ry(0.05);
    B[1] = Rot3::Rz(0.06);
    GN g(A, a, B);

    Rot3 R = Rot3::Rz(0.2);
    Vector3 b = (Vector(3) << 0.04, 0.05, 0.06).finished();
    std::array<Rot3, N_TEST> S;
    S[0] = Rot3::Rx(0.07);
    S[1] = Rot3::Ry(0.08);
    StateN x(R, b, S);

    StateN expected_transformed_x = g * x;
    StateN actual_transformed = ABCGeometryN::groupAction(g, x);

    // Compare each component
    EXPECT(assert_equal<Rot3>(actual_transformed.R, expected_transformed_x.R, 1e-9));
    EXPECT(assert_equal<Vector>(actual_transformed.b, expected_transformed_x.b, 1e-9));
    EXPECT(ArraysEqual(actual_transformed.S, expected_transformed_x.S));
}

/* ************************************************************************* */
TEST(ABC, ABCGeometry_lift) {
    // Setup state
    Rot3 R = Rot3::Identity();
    Vector3 b = (Vector(3) << 0.1, 0.2, 0.3).finished();
    std::array<Rot3, N_TEST> S_arr;
    S_arr[0] = Rot3::Rx(0.05);
    S_arr[1] = Rot3::Ry(0.06);
    StateN xi(R, b, S_arr);

    // Setup input
    InputData u_data;
    u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
    u_data.Sigma = I_6x6;
    
    Vector6 u = u_data.toInputVector();
    typename GN::TangentVector L = xi.lift(u);

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

///* ************************************************************************* */
TEST(ABC, ABCGeometry_stateMatrixA) {
    // Setup G element for X_hat
    Rot3 A = Rot3::Rx(0.1);
    Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
    std::array<Rot3, N_TEST> B;
    B[0] = Rot3::Ry(0.05);
    B[1] = Rot3::Rz(0.06);
    GN X_hat(A, a, B);

    // Setup input
    InputData u_data;
    u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
    u_data.Sigma = I_6x6;

    Vector6 u = u_data.toInputVector();
    Matrix A_matrix = ABCGeometryN::stateMatrixA(X_hat, u);
    Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inverse(), u).head<3>());

    Matrix expected_A1 = Matrix::Zero(6, 6);
    expected_A1.block<3, 3>(0, 3) = -I_3x3;
    expected_A1.block<3, 3>(3, 3) = W0;

    Matrix expected_A2 = repBlock(W0, N_TEST);
    Matrix expected_A_matrix = blockDiag(expected_A1, expected_A2);

    EXPECT(assert_equal(A_matrix, expected_A_matrix, 1e-9));
}

///* ************************************************************************* */
TEST(ABC, ABCGeometry_stateTransitionMatrix) {
    // Setup G element for X_hat
    Rot3 A = Rot3::Rx(0.1);
    Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
    std::array<Rot3, N_TEST> B;
    B[0] = Rot3::Ry(0.05);
    B[1] = Rot3::Rz(0.06);
    GN X_hat(A, a, B);

    // Setup input
    InputData u_data;
    u_data.w = (Vector(3) << 0.5, 0.6, 0.7).finished();
    u_data.Sigma = I_6x6;

    double dt = 0.1;

    Vector6 u = u_data.toInputVector();
    Matrix Phi = ABCGeometryN::stateTransitionMatrix(u, dt, X_hat);
    Matrix3 W0 = Rot3::Hat(velocityAction(X_hat.inv(), u).head<3>());
    Matrix Phi1 = Matrix::Zero(6, 6);
    Matrix3 Phi12 = -dt * (I_3x3 + (dt / 2) * W0 + ((dt * dt) / 6) * W0 * W0);
    Matrix3 Phi22 = I_3x3 + dt * W0 + ((dt * dt) / 2) * W0 * W0;

    Phi1.block<3, 3>(0, 0) = I_3x3;
    Phi1.block<3, 3>(0, 3) = Phi12;
    Phi1.block<3, 3>(3, 3) = Phi22;
    Matrix Phi2 = repBlock(Phi22, N_TEST);
    Matrix expected_Phi = blockDiag(Phi1, Phi2);

    EXPECT(assert_equal(Phi, expected_Phi, 1e-9));
}


///* ************************************************************************* */
//TEST(ABC, ABCGeometry_inputMatrix) {
//    // Setup G element for X_hat
//    Rot3 A = Rot3::Rx(0.1);
//    Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
//    std::array<Rot3, N_TEST> B;
//    B[0] = Rot3::Ry(0.05);
//    B[1] = Rot3::Rz(0.06);
//    GN X_hat(A, a, B);
//
//    Matrix input_matrix = ABCGeometryN::inputMatrix(X_hat);
//
//    Matrix expected_B1 = blockDiag(X_hat.A.matrix(), X_hat.A.matrix());
//    Matrix expected_B2(3 * N_TEST, 3 * N_TEST);
//    expected_B2.setZero();
//    for (size_t i = 0; i < N_TEST; ++i) {
//        expected_B2.block<3, 3>(3 * i, 3 * i) = X_hat.B[i].matrix();
//    }
//    Matrix expected_input_matrix = blockDiag(expected_B1, expected_B2);
//
//    EXPECT(assert_equal(input_matrix, expected_input_matrix, 1e-9));
//}

///* ************************************************************************* */
TEST(ABC, ABCGeometry_processNoise) {
    InputData u_data;
    u_data.w = Vector3::Zero();
    u_data.Sigma = (Matrix(6, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 2, 0, 0, 0, 0,
        0, 0, 3, 0, 0, 0,
        0, 0, 0, 4, 0, 0,
        0, 0, 0, 0, 5, 0,
        0, 0, 0, 0, 0, 6).finished();

    Matrix Q = ABCGeometryN::processNoise(u_data.Sigma);

    Matrix expected_Q_cal_part = repBlock(1e-9 * I_3x3, N_TEST);
    Matrix expected_Q = blockDiag(u_data.Sigma, expected_Q_cal_part);

    EXPECT(assert_equal(Q, expected_Q, 1e-9));
}


/////* ************************************************************************* */
//TEST(ABC, ABCGeometry_inputMatrixBt) {
//    // This function is identical to inputMatrix, so we'll test its output matches inputMatrix.
//    // Setup G element for X_hat
//    Rot3 A = Rot3::Rx(0.1);
//    Matrix3 a = Rot3::Hat((Vector(3) << 0.01, 0.02, 0.03).finished());
//    std::array<Rot3, N_TEST> B;
//    B[0] = Rot3::Ry(0.05);
//    B[1] = Rot3::Rz(0.06);
//    GN X_hat(A, a, B);
//
//    Matrix input_matrix_Bt = ABCGeometryN::inputMatrixBt(X_hat);
//    Matrix input_matrix = ABCGeometryN::inputMatrix(X_hat); // Reference from the other function
//
//    EXPECT(assert_equal(input_matrix_Bt, input_matrix, 1e-9));
//}

///* ************************************************************************* */
//TEST(ABC, ABCGeometry_measurementMatrixC) {
//    Unit3 d = Unit3(0, 0, 1); // Reference direction (e.g., gravity)
//    Matrix3 wedge_d = Rot3::Hat(d.unitVector());
//
//    // Test with calibrated sensor (idx = 0)
//    int cal_idx = 0;
//    Matrix C_cal = ABCGeometryN::measurementMatrixC(d, cal_idx);
//
//    Matrix expected_Cc_cal = Matrix::Zero(3, 3 * N_TEST);
//    expected_Cc_cal.block<3, 3>(0, 3 * cal_idx) = wedge_d;
//
//    Matrix expected_temp_cal(3, 6 + 3 * N_TEST);
//    expected_temp_cal.block<3, 3>(0, 0) = wedge_d;
//    expected_temp_cal.block<3, 3>(0, 3) = Matrix3::Zero();
//    expected_temp_cal.block(0, 6, 3, 3 * N_TEST) = expected_Cc_cal;
//    Matrix expected_C_cal = wedge_d * expected_temp_cal;

//
//    EXPECT(assert_equal(C_cal, expected_C_cal, 1e-9));
//}
TEST(ABC, EqFilter){
    using M = abc_eqf_lib::State<N_TEST>;
    using G = abc_eqf_lib::Group<N_TEST>;
    using EqFilter = gtsam::EqF<G, M>;
    
    const G g_0;
    const M xi_ref; // Reference state (xi circle) and not inital state?
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

    G X_HatExpected = filter.state(); //from LieGroupEKF

    EXPECT(traits<G>::Equals(g_0, X_HatActual, 1e-9));

    EXPECT(traits<G>::Equals(X_HatActual, X_HatExpected, 1e-9));

    InputData u;
	u.w = (Vector3() << 0.01, -0.02, 0.015).finished();
	u.Sigma = I_6x6;
	double dt = 0.01;

    Vector6 u_vec = u.toInputVector();
    Matrix Q = ABCGeometryN::processNoise(u.Sigma);
    filter.predict(u_vec, Q, dt);

	EXPECT(traits<G>::Equals(filter.groupEstimate(), filter.state(), 1e-9));
  }
/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */