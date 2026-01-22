"""
Minimal example for gtsam.EquivariantFilterUnit3Symmetry (Unit3/Rot3 EqF).

This mirrors the attitude-only EquivariantFilter test setup.
"""

import numpy as np
import gtsam
from gtsam import Rot3, Unit3


def main() -> None:
    # Reference direction and initial covariance on S^2
    eta_ref = Unit3(0.0, 0.0, 1.0)
    sigma0 = 0.01 * np.eye(2)

    filter_eqf = gtsam.EquivariantFilterUnit3Symmetry(eta_ref, sigma0)

    # explicit lift vector (omega) and manifold Qc
    omega = np.array([0.1, -0.2, 0.3])
    A = np.zeros((2, 2))
    sigma_u = 0.1 * np.eye(3)
    B = np.eye(2, 3)
    Qc = B @ sigma_u @ B.T
    dt = 0.01

    filter_eqf.predictWithJacobianEuler(omega, A, Qc, dt)

    expected_Q = Rot3.Expmap(omega * dt)
    assert filter_eqf.groupEstimate().equals(expected_Q, 1e-9)

    expected_state = Unit3(expected_Q.unrotate(eta_ref.point3()))
    assert filter_eqf.state().equals(expected_state, 1e-9)

    print("After predict:")
    print("  Truth (eta_ref):", eta_ref.unitVector())
    print("  Estimated state:", filter_eqf.state().unitVector())

    # update by a simple linear scaling of the direction
    c_m = 1.2
    eta_hat = filter_eqf.state()
    H = c_m * eta_hat.basis()  # 3x2
    prediction = c_m * eta_hat.point3()
    z = c_m * eta_ref.point3()
    R = 0.01 * np.eye(3)

    filter_eqf.updateWithVector(prediction, H, z, R)

    assert filter_eqf.errorCovariance().shape == (2, 2)
    print("After update:")
    print("  Truth (eta_ref):", eta_ref.unitVector())
    print("  Estimated state:", filter_eqf.state().unitVector())
    print("EquivariantFilterUnit3Symmetry example ran successfully.")


if __name__ == "__main__":
    main()


