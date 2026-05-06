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
from gtsam import interpolateFactorGraphPose3
from gtsam import interpolateWnoaFactorGraphPose3
from gtsam import updateInterpValuesPose3
from gtsam import updateInterpValuesWithCovariancePose3

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


class TestWnoaFactorGraphPose3(GtsamTestCase):
    """Test interpolation factor graphs for Pose3."""

    def _make_se3_fixture(self):
        timestep = 0.1
        q_psd_diag = np.ones(6)

        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        v0 = np.array([1.0, 0.0, 0.5, 0.1, 0.0, 0.0])
        p1 = p0.retract(timestep * v0)
        p2 = p0.retract(2.0 * timestep * v0)
        p3 = p0.retract(3.0 * timestep * v0)
        p4 = p0.retract(4.0 * timestep * v0)

        keys = {
            "p0": Symbol("x", 0).key(),
            "p1": Symbol("x", 1).key(),
            "p2": Symbol("x", 2).key(),
            "p3": Symbol("x", 3).key(),
            "p4": Symbol("x", 4).key(),
            "v0": Symbol("v", 0).key(),
            "v1": Symbol("v", 1).key(),
            "v2": Symbol("v", 2).key(),
            "v3": Symbol("v", 3).key(),
            "v4": Symbol("v", 4).key(),
        }

        return timestep, q_psd_diag, keys, p0, p1, p2, p3, p4, v0

    def _make_se3_interp_graph(self, use_wnoa_graph=False):
        timestep, q_psd_diag, keys, p0, p1, p2, p3, p4, v0 = (
            self._make_se3_fixture()
        )

        model = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        between_factor = gtsam.BetweenFactorPose3(
            keys["p1"], keys["p3"], p1.between(p3), model
        )
        prior_pose_factor = gtsam.PriorFactorPose3(keys["p1"], p1, model)
        prior_vel_factor = gtsam.PriorFactorVector6(keys["v1"], v0, model)

        graph = gtsam.NonlinearFactorGraph()
        graph.add(between_factor)
        graph.add(prior_pose_factor)
        graph.add(prior_vel_factor)

        estimated_states = {
            gtsam.StateData(keys["p0"], keys["v0"], 0.0),
            gtsam.StateData(keys["p4"], keys["v4"], 4.0 * timestep),
            gtsam.StateData(keys["p2"], keys["v2"], 2.0 * timestep),
        }
        interpolated_states = {
            gtsam.StateData(keys["p3"], keys["v3"], 3.0 * timestep),
            gtsam.StateData(keys["p1"], keys["v1"], timestep),
        }

        if use_wnoa_graph:
            new_graph = interpolateWnoaFactorGraphPose3(
                graph, estimated_states, interpolated_states, q_psd_diag
            )
        else:
            new_graph = interpolateFactorGraphPose3(
                graph, estimated_states, interpolated_states, q_psd_diag
            )

        values = gtsam.Values()
        values.insert(keys["p0"], p0)
        values.insert(keys["p2"], p2)
        values.insert(keys["p4"], p4)
        values.insert(keys["v0"], v0)
        values.insert(keys["v2"], v0)
        values.insert(keys["v4"], v0)

        return (
            timestep,
            q_psd_diag,
            keys,
            p0,
            p1,
            p2,
            p3,
            p4,
            v0,
            new_graph,
            values,
            estimated_states,
            interpolated_states,
        )

    def _assert_covariance_map(self, covariance_map, keys):
        def is_positive_semidefinite(covariance):
            symmetric = 0.5 * (covariance + covariance.T)
            min_eigenvalue = np.min(np.linalg.eigvalsh(symmetric))
            return min_eigenvalue >= -1e-9

        self.assertIn(keys["p1"], covariance_map)
        self.assertIn(keys["v1"], covariance_map)
        self.assertIn(keys["p3"], covariance_map)
        self.assertIn(keys["v3"], covariance_map)

        for key in (keys["p1"], keys["v1"], keys["p3"], keys["v3"]):
            covariance = covariance_map[key]
            self.assertEqual(covariance.shape, (6, 6))
            self.assertTrue(is_positive_semidefinite(covariance))

    def test_se3_interp_graph(self):
        """Test interpolateFactorGraphPose3 for SE3 graphs."""
        (
            _timestep,
            _q_psd_diag,
            keys,
            p0,
            _p1,
            p2,
            _p3,
            p4,
            v0,
            new_graph,
            values,
            _estimated_states,
            _interpolated_states,
        ) = self._make_se3_interp_graph()

        optimizer = gtsam.GaussNewtonOptimizer(new_graph, values)
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.iterate()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.optimize()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=6)

        perturb = np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])
        values_pert = gtsam.Values()
        values_pert.insert(keys["p0"], p0.retract(perturb))
        values_pert.insert(keys["p2"], p2.retract(perturb))
        values_pert.insert(keys["p4"], p4.retract(perturb))
        v_pert = v0 + 0.1 * np.ones(6)
        values_pert.insert(keys["v0"], v_pert)
        values_pert.insert(keys["v2"], v_pert)
        values_pert.insert(keys["v4"], v_pert)

        optimizer2 = gtsam.GaussNewtonOptimizer(new_graph, values_pert)
        optimizer2.optimize()
        self.assertAlmostEqual(optimizer2.error(), 0.0, places=4)

    def test_se3_interp_wnoa_graph(self):
        """Test interpolateWnoaFactorGraphPose3 for SE3 graphs."""
        (
            _timestep,
            _q_psd_diag,
            keys,
            p0,
            _p1,
            p2,
            _p3,
            p4,
            v0,
            new_graph,
            values,
            _estimated_states,
            _interpolated_states,
        ) = self._make_se3_interp_graph(use_wnoa_graph=True)

        optimizer = gtsam.LevenbergMarquardtOptimizer(new_graph, values)
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.iterate()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.optimize()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=6)

        perturb = np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])
        values_pert = gtsam.Values()
        values_pert.insert(keys["p0"], p0.retract(perturb))
        values_pert.insert(keys["p2"], p2.retract(perturb))
        values_pert.insert(keys["p4"], p4.retract(perturb))
        v_pert = v0 + 0.1 * np.ones(6)
        values_pert.insert(keys["v0"], v_pert)
        values_pert.insert(keys["v2"], v_pert)
        values_pert.insert(keys["v4"], v_pert)

        optimizer2 = gtsam.GaussNewtonOptimizer(new_graph, values_pert)
        optimizer2.optimize()
        self.assertAlmostEqual(optimizer2.error(), 0.0, places=4)

    def test_se3_update_interp_values(self):
        """Test updateInterpValuesPose3 for SE3 graphs."""
        (
            _timestep,
            q_psd_diag,
            keys,
            _p0,
            p1,
            _p2,
            p3,
            _p4,
            v0,
            new_graph,
            values,
            estimated_states,
            interpolated_states,
        ) = self._make_se3_interp_graph()

        result_interp = updateInterpValuesPose3(
            new_graph, values, estimated_states, interpolated_states, q_psd_diag
        )

        p3_est = result_interp.atPose3(keys["p3"])
        p1_est = result_interp.atPose3(keys["p1"])
        v3_est = result_interp.atVector6(keys["v3"])
        v1_est = result_interp.atVector6(keys["v1"])

        self.gtsamAssertEquals(p3, p3_est, 1e-6)
        self.gtsamAssertEquals(p1, p1_est, 1e-6)
        np.testing.assert_allclose(v0, v3_est, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(v0, v1_est, rtol=1e-12, atol=1e-12)

    def test_se3_update_interp_values_with_covariance(self):
        """Test updateInterpValuesWithCovariancePose3 for SE3 graphs."""
        (
            _timestep,
            q_psd_diag,
            keys,
            _p0,
            p1,
            _p2,
            p3,
            _p4,
            v0,
            new_graph,
            values,
            estimated_states,
            interpolated_states,
        ) = self._make_se3_interp_graph()

        result_interp, covariance_map = updateInterpValuesWithCovariancePose3(
            new_graph, values, estimated_states, interpolated_states, q_psd_diag
        )

        p3_est = result_interp.atPose3(keys["p3"])
        p1_est = result_interp.atPose3(keys["p1"])
        v3_est = result_interp.atVector6(keys["v3"])
        v1_est = result_interp.atVector6(keys["v1"])

        self.gtsamAssertEquals(p3, p3_est, 1e-6)
        self.gtsamAssertEquals(p1, p1_est, 1e-6)
        np.testing.assert_allclose(v0, v3_est, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(v0, v1_est, rtol=1e-12, atol=1e-12)
        self._assert_covariance_map(covariance_map, keys)

        
if __name__ == "__main__":
    unittest.main()