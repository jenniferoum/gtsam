"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Unit tests for White-Noise-on-Acceleration, Continuous-time, Gaussian-process factors.

Author: Connor Holmes
"""
import unittest

import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase
from gtsam.utils.numerical_derivative import (
    numericalDerivative41,
    numericalDerivative42,
    numericalDerivative43,
    numericalDerivative44,
)

from gtsam import Symbol
from gtsam import WnoaInterpFactorPose3
from gtsam import WnoaMotionFactorPose3

class TestStateData(GtsamTestCase):
    """Test StateData class."""
    def test_construction(self):
        """Test construction of StateData."""
        state_data = gtsam.StateData()
        self.assertIsInstance(state_data, gtsam.StateData)
        
        pose_key = Symbol('x', 0).key()
        velocity_key = Symbol('v', 0).key()
        time = 0.0
        state_data = gtsam.StateData(pose_key, velocity_key, time)
        self.assertEqual(state_data.pose, pose_key)
        self.assertEqual(state_data.velocity, velocity_key)
        self.assertEqual(state_data.time, time)

class TestWnoaMotionFactor(GtsamTestCase):
    """Test WnoaMotionFactor class.
    Tests are based on the more extensive C++ tests in gtsam/nonlinear/testWnoaFactor.cpp."""
    def test_construction_and_eval(self):
        """Test construction of WnoaMotionFactor."""
        # First state
        pose_key = Symbol('x', 0).key()
        velocity_key = Symbol('v', 0).key()
        time = 0.0
        # Second state
        state_data_0 = gtsam.StateData(pose_key, velocity_key, time)
        pose_key = Symbol('x', 1).key()
        velocity_key = Symbol('v', 1).key()
        time = 1.0
        state_data_1 = gtsam.StateData(pose_key, velocity_key, time)
        q_psd_diag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        factor = WnoaMotionFactorPose3(state_data_0, state_data_1, q_psd_diag)
        self.assertIsInstance(factor, WnoaMotionFactorPose3)

    def test_evaluate_error(self):
        """Test evaluateError without Jacobians."""
        pose_key_0 = Symbol('x', 0).key()
        velocity_key_0 = Symbol('v', 0).key()
        pose_key_1 = Symbol('x', 1).key()
        velocity_key_1 = Symbol('v', 1).key()

        state_data_0 = gtsam.StateData(pose_key_0, velocity_key_0, 0.0)
        state_data_1 = gtsam.StateData(pose_key_1, velocity_key_1, 0.1)

        q_psd_diag = np.ones(6)
        factor = WnoaMotionFactorPose3(state_data_0, state_data_1, q_psd_diag)

        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.2, 1.0, 0.0, 0.0]))
        v0 = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 2.0])
        p1 = p0.retract(0.1 * v0)
        v1 = 2.0 * v0

        error = factor.evaluateError(p0, v0, p1, v1)
        expected = np.hstack([np.zeros(6), v0])

        np.testing.assert_allclose(error, expected, rtol=1e-4, atol=1e-6)


class TestWnoaInterpFactorPose3(GtsamTestCase):
    """Test WnoaInterpFactorPose3 class."""

    def _make_factor_and_values(self):
        timestep = 0.1
        q_psd_diag = np.ones(6)

        pose_key_0 = Symbol('x', 0).key()
        velocity_key_0 = Symbol('v', 0).key()
        pose_key_1 = Symbol('x', 1).key()
        velocity_key_1 = Symbol('v', 1).key()
        pose_key_2 = Symbol('x', 2).key()
        velocity_key_2 = Symbol('v', 2).key()

        estimated_states = [
            gtsam.StateData(pose_key_0, velocity_key_0, 0.0),
            gtsam.StateData(pose_key_2, velocity_key_2, 2.0 * timestep),
        ]
        estimated_states = set(estimated_states)
        interpolated_states = [
            gtsam.StateData(pose_key_1, velocity_key_1, timestep)
        ]
        interpolated_states = set(interpolated_states)
        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        v0 = np.array([1.0, 0.0, 0.5, 0.1, 0.0, 0.0])
        p1 = p0.retract(timestep * v0)
        p2 = p0.retract(2.0 * timestep * v0)

        model = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        prior = gtsam.PriorFactorPose3(pose_key_1, p1, model)
        factor = WnoaInterpFactorPose3(
            prior, estimated_states, interpolated_states, q_psd_diag
        )

        values = gtsam.Values()
        values.insert(pose_key_0, p0)
        values.insert(pose_key_2, p2)
        values.insert(velocity_key_0, v0)
        values.insert(velocity_key_2, v0)

        return factor, values

    def test_construction(self):
        """Test WnoaInterpFactorPose3 construction."""
        factor, _ = self._make_factor_and_values()
        self.assertIsInstance(factor, WnoaInterpFactorPose3)

    def test_print(self):
        """Test WnoaInterpFactorPose3 print."""
        factor, _ = self._make_factor_and_values()
        factor.print()

    def test_equals(self):
        """Test WnoaInterpFactorPose3 equals."""
        factor_1, _ = self._make_factor_and_values()
        factor_2, _ = self._make_factor_and_values()
        self.assertTrue(factor_1.equals(factor_2, 1e-9))

    def test_error(self):
        """Test WnoaInterpFactorPose3 error."""
        factor, values = self._make_factor_and_values()
        error = factor.error(values)
        self.assertAlmostEqual(error, 0.0, places=9)


        
        
if __name__ == "__main__":
    unittest.main()