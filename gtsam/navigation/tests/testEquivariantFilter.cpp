/**
 * @file testEquivariantFilter.cpp
 * @brief Simple SO(3) equivariant filter example (attitude-only),
 *        exercising EquivariantFilter with a different M/G/Actions combo.
 *
 * This is inspired by the simple sphere / attitude example in Mahony's
 * equivariant filter tutorial, but here formulated as a pure SO(3) example:
 * - Physical state M is Rot3 (attitude).
 * - Symmetry group G is also Rot3.
 * - The group acts on the reference state by left multiplication.
 *
 * The goal is to ensure EquivariantFilter.h is generic and not tied to ABC.h.
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/EquivariantFilter.h>

using namespace gtsam;

namespace attitude_example {

//---------------------------------------------------------------------------
// Types
//---------------------------------------------------------------------------

using M = Rot3;  // physical state: attitude
using G = Rot3;  // symmetry group: SO(3) again

//---------------------------------------------------------------------------
// StateAction: group action on the state
//
// Here we treat the reference state R_ref and apply left multiplication:
//   phi(R_ref, X) = X * R_ref
// (we choose left vs right just for concreteness; only consistency matters.)
//---------------------------------------------------------------------------

struct StateAction {
  using M = attitude_example::M;
  using G = attitude_example::G;

  const M R_ref_;

  explicit StateAction(const M& R_ref) : R_ref_(R_ref) {}

  /// Group action at state R_ref by group element X.
  M operator()(const G& X) const { return X.compose(R_ref_); }

  /// Jacobian of the action at the group identity.
  ///
  /// For small perturbations Exp(delta) near identity, we have:
  ///   phi(R_ref, Exp(delta)) ≈ Exp(delta) * R_ref
  /// and the corresponding tangent is just delta itself, so the Jacobian
  /// at identity is the 3x3 identity.
  Matrix jacobianAtIdentity() const {
    Matrix H = Matrix::Zero(3, 3);
    H.block<3, 3>(0, 0) = I_3x3;
    return H;
  }
};

//---------------------------------------------------------------------------
// Lift: maps (state, input) -> group tangent
//
// For this simple example, the lift just returns the body angular velocity
// itself, independent of the state. This is enough to test the EqF plumbing.
//---------------------------------------------------------------------------

struct Lift {
  using G = attitude_example::G;
  using Input = Vector3;

  explicit Lift(const Input& u) : omega_(u) {}

  typename traits<G>::TangentVector operator()(const M& /*R*/) const {
    // Tangent space of SO(3) ~ R^3, so we simply return omega.
    return omega_;
  }

 private:
  Input omega_;
};

//---------------------------------------------------------------------------
// InputAction: group action on the input + system matrices
//
// We take the input u as a body-frame angular velocity. For this example,
// we don't use the group action on input in the prediction, but we still
// provide a reasonable definition. The important parts for EqF are:
//   - using Input = ...
//   - processNoise(Sigma): returns Q in lifted coords
//   - stateTransitionMatrix(X_hat, dt): Dim x Dim
//   - inputMatrixBt(X_hat): Dim x Dim
//---------------------------------------------------------------------------

struct InputAction {
  using G = attitude_example::G;
  using Input = Vector3;

  explicit InputAction(const Input& u) : omega_(u) {}

  Input operator()(const G& X) const {
    // Right group action on input: X^{-1} * omega.
    // This is not actually used in EqF.predict but is well-defined.
    return X.unrotate(omega_);
  }

  /// Embed process noise covariance into the lifted coordinates.
  /// For this simple SO(3)-only system, we just pass it through.
  static Matrix processNoise(const Matrix& Sigma) { return Sigma; }

  /// Linearized state transition matrix Phi. We take a simple model:
  ///   X_{k+1} = Exp(omega * dt) * X_k
  /// whose linearization in tangent coordinates is approximately identity.
  Matrix stateTransitionMatrix(const G& /*X_hat*/, double /*dt*/) const {
    return I_3x3;
  }

  /// Input matrix B^T that maps process noise to lifted coordinates.
  /// We use the identity for this example.
  Matrix inputMatrix(const G& /*X_hat*/) const { return I_3x3; }

  Matrix inputMatrixBt(const G& X_hat) const { return inputMatrix(X_hat); }

 private:
  Input omega_;
};

//---------------------------------------------------------------------------
// OutputAction: group action on the measurement + Jacobians
//
// We consider a direction measurement y with a reference direction d.
// For simplicity, we define:
//   y_body(X) = X^{-1} * y
// and the innovation as:
//   innovation(X) = d^∧ * y_body(X)
// same style as in ABC's OutputAction.
//---------------------------------------------------------------------------

struct OutputAction {
  using G = attitude_example::G;
  using Output = Vector3;

  OutputAction(const Unit3& y, const Unit3& d) : y_(y), d_(d) {}

  Output operator()(const G& X) const {
    // Transform measured direction into the "reference" frame by unrotating.
    return X.unrotate(y_.unitVector());
  }

  /// Jacobian of phi_y at the identity of G.
  Matrix jacobianAtIdentity() const {
    Matrix H = Matrix::Zero(3, 3);
    H.block<3, 3>(0, 0) = Rot3::Hat(y_.unitVector());
    return H;
  }

  /// Innovation: wedge(d) * transformed_y
  Vector3 innovation(const G& X) const {
    const Vector3 transformed_y = this->operator()(X);
    return Rot3::Hat(d_.unitVector()) * transformed_y;
  }

  /// Linearized measurement matrix C.
  ///
  /// For this simple example, we define:
  ///   C = (d^∧) * (d^∧)
  /// in lifted coordinates, independent of X.
  Matrix measurementMatrixC() const {
    const Matrix3 wedge_d = Rot3::Hat(d_.unitVector());
    Matrix3 temp = wedge_d;  // 3x3
    return wedge_d * temp;
  }

  /// Measurement noise propagation matrix D_t.
  ///
  /// Here we propagate measurement noise through the current rotation X_hat.
  Matrix outputMatrixDt(const G& X_hat) const { return X_hat.matrix(); }

 private:
  Unit3 y_;  // measured direction
  Unit3 d_;  // reference direction
};

}  // namespace attitude_example

//==============================================================================
// Tests
//==============================================================================

TEST(EquivariantFilter_Attitude, Predict) {
  using namespace attitude_example;

  // Initial group and reference state: both identity.
  const G X0;     // Rot3() == identity
  const M R_ref;  // identity
  Matrix Sigma0 = 0.01 * I_3x3;

  EqF<M, StateAction> filter(X0, R_ref, Sigma0);

  // Input: body angular velocity
  Lift::Input omega;
  omega << 0.1, -0.2, 0.3;
  const double dt = 0.01;

  // Process noise in input space, then lifted
  Matrix Sigma_u = 0.1 * I_3x3;
  Matrix Q = InputAction::processNoise(Sigma_u);

  // --- Perform prediction through EqF ---
  filter.predict<Lift, InputAction>(omega, Q, dt);

  // --- Expected result ---
  const G X_expected = Rot3::Expmap(omega * dt) * X0;

  InputAction psi_u(omega);
  Matrix Phi = psi_u.stateTransitionMatrix(X0, dt);
  Matrix Bt = psi_u.inputMatrixBt(X0);
  Matrix Q_process = Bt * Q * Bt.transpose() * dt;
  Matrix P_expected = Phi * Sigma0 * Phi.transpose() + Q_process;

  EXPECT(assert_equal(X_expected, filter.groupEstimate(), 1e-9));
  EXPECT(assert_equal(P_expected, filter.covariance(), 1e-9));

  // stateEstimate() should just be the group estimate in this example
  EXPECT(assert_equal(filter.groupEstimate(), filter.stateEstimate(), 1e-9));
}

TEST(EquivariantFilter_Attitude, Update) {
  using namespace attitude_example;

  // Initial group and reference state: both identity.
  const G X0;
  const M R_ref;
  Matrix Sigma0 = 0.01 * I_3x3;

  EqF<M, StateAction> filter(X0, R_ref, Sigma0);

  // Do a small prediction first to get a non-trivial covariance.
  Lift::Input omega;
  omega << 0.1, -0.2, 0.3;
  const double dt = 0.01;
  Matrix Sigma_u = 0.1 * I_3x3;
  Matrix Q = InputAction::processNoise(Sigma_u);
  filter.predict<Lift, InputAction>(omega, Q, dt);

  const G X_before = filter.groupEstimate();
  const Matrix P_before = filter.covariance();

  // Measurement: y is a known direction, d is the reference direction.
  const Unit3 d(0, 0, 1);  // reference direction (z-axis)
  const Unit3 y(0, 0, 1);  // measured aligned with d
  const Matrix R_meas = 0.01 * I_3x3;

  OutputAction phi_y(y, d);

  // --- Perform update through EqF ---
  filter.update<OutputAction>(phi_y, R_meas);

  const G X_after = filter.groupEstimate();
  const Matrix P_after = filter.covariance();

  // --- Manual update (should match EqF) ---

  // Innovation lift matrix (inverse of Dphi0)
  StateAction act_ref(R_ref);
  Matrix Dphi0 = act_ref.jacobianAtIdentity();
  Matrix InnovationLift =
      Dphi0.completeOrthogonalDecomposition().pseudoInverse();

  // Measurement matrices
  Matrix Ct = phi_y.measurementMatrixC();
  Matrix Dt = phi_y.outputMatrixDt(X_before);

  Matrix S = Ct * P_before * Ct.transpose() + Dt * R_meas * Dt.transpose();
  Matrix K = P_before * Ct.transpose() * S.inverse();

  // Innovation evaluated at X_before^{-1}
  Vector3 innovation = phi_y.innovation(X_before.inverse());
  Vector3 delta_xi = InnovationLift * (K * innovation);

  const G X_expected = Rot3::Expmap(delta_xi) * X_before;

  Matrix I = Matrix::Identity(P_before.rows(), P_before.cols());
  Matrix P_expected = (I - K * Ct) * P_before;

  EXPECT(assert_equal(X_expected, X_after, 1e-8));
  EXPECT(assert_equal(P_expected, P_after, 1e-8));

  // Again, stateEstimate() should coincide with groupEstimate() here.
  EXPECT(assert_equal(filter.groupEstimate(), filter.stateEstimate(), 1e-8));
}

//==============================================================================

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}