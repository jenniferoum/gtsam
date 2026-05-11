"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Unit tests for White-Noise-on-Acceleration, Continuous-time, Gaussian-process factors.

Author: Connor Holmes
"""
import unittest
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from gtsam import Symbol
from gtsam import WnoaInterpFactorPose3
from gtsam import WnoaMotionFactorPose3
from gtsam import interpolateFactorGraphPose3
from gtsam import interpolateWnoaFactorGraphPose3
from gtsam import updateInterpValuesPose3
from gtsam import updateInterpValuesWithCovariancePose3


@dataclass
class Se3FixtureData:
    """Data container for SE3 fixture."""
    timeStep: float
    qPsdDiag: np.ndarray
    keys: dict
    p0: gtsam.Pose3
    p1: gtsam.Pose3
    p2: gtsam.Pose3
    p3: gtsam.Pose3
    p4: gtsam.Pose3
    v0: np.ndarray


@dataclass
class Se3InterpGraphData:
    """Data container for SE3 interpolation graph."""
    timeStep: float
    qPsdDiag: np.ndarray
    keys: dict
    p0: gtsam.Pose3
    p1: gtsam.Pose3
    p2: gtsam.Pose3
    p3: gtsam.Pose3
    p4: gtsam.Pose3
    v0: np.ndarray
    newGraph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    estimatedStates: set
    interpolatedStates: set

class TestStateData(GtsamTestCase):
    """Test StateData class."""
    def testConstruction(self):
        """Test construction of StateData."""
        stateData = gtsam.StateData()
        self.assertIsInstance(stateData, gtsam.StateData)
        
        poseKey = Symbol('x', 0).key()
        velocityKey = Symbol('v', 0).key()
        time = 0.0
        stateData = gtsam.StateData(poseKey, velocityKey, time)
        self.assertEqual(stateData.pose, poseKey)
        self.assertEqual(stateData.velocity, velocityKey)
        self.assertEqual(stateData.time, time)

class TestWnoaMotionFactor(GtsamTestCase):
    """Test WnoaMotionFactor class.
    Tests are based on the more extensive C++ tests in gtsam/nonlinear/testWnoaFactor.cpp."""
    def testConstructionAndEval(self):
        """Test construction of WnoaMotionFactor."""
        # First state
        poseKey = Symbol('x', 0).key()
        velocityKey = Symbol('v', 0).key()
        time = 0.0
        # Second state
        stateData0 = gtsam.StateData(poseKey, velocityKey, time)
        poseKey = Symbol('x', 1).key()
        velocityKey = Symbol('v', 1).key()
        time = 1.0
        stateData1 = gtsam.StateData(poseKey, velocityKey, time)
        qPsdDiag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        factor = WnoaMotionFactorPose3(stateData0, stateData1, qPsdDiag)
        self.assertIsInstance(factor, WnoaMotionFactorPose3)

    def testEvaluateError(self):
        """Test evaluateError without Jacobians."""
        poseKey0 = Symbol('x', 0).key()
        velocityKey0 = Symbol('v', 0).key()
        poseKey1 = Symbol('x', 1).key()
        velocityKey1 = Symbol('v', 1).key()

        stateData0 = gtsam.StateData(poseKey0, velocityKey0, 0.0)
        stateData1 = gtsam.StateData(poseKey1, velocityKey1, 0.1)

        qPsdDiag = np.ones(6)
        factor = WnoaMotionFactorPose3(stateData0, stateData1, qPsdDiag)

        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.2, 1.0, 0.0, 0.0]))
        v0 = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 2.0])
        p1 = p0.retract(0.1 * v0)
        v1 = 2.0 * v0

        error = factor.evaluateError(p0, v0, p1, v1)
        expected = np.hstack([np.zeros(6), v0])

        np.testing.assert_allclose(error, expected, rtol=1e-4, atol=1e-6)


class TestWnoaInterpFactorPose3(GtsamTestCase):
    """Test WnoaInterpFactorPose3 class."""

    def _makeFactorAndValues(self):
        timeStep = 0.1
        qPsdDiag = np.ones(6)

        poseKey0 = Symbol('x', 0).key()
        velocityKey0 = Symbol('v', 0).key()
        poseKey1 = Symbol('x', 1).key()
        velocityKey1 = Symbol('v', 1).key()
        poseKey2 = Symbol('x', 2).key()
        velocityKey2 = Symbol('v', 2).key()

        estimatedStates = [
            gtsam.StateData(poseKey2, velocityKey2, 2.0 * timeStep),
            gtsam.StateData(poseKey0, velocityKey0, 0.0),
        ]
        estimatedStates = set(estimatedStates)
        interpolatedStates = [
            gtsam.StateData(poseKey1, velocityKey1, timeStep)
        ]
        interpolatedStates = set(interpolatedStates)
        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        v0 = np.array([1.0, 0.0, 0.5, 0.1, 0.0, 0.0])
        p1 = p0.retract(timeStep * v0)
        p2 = p0.retract(2.0 * timeStep * v0)

        model = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        prior = gtsam.PriorFactorPose3(poseKey1, p1, model)
        factor = WnoaInterpFactorPose3(
            prior, estimatedStates, interpolatedStates, qPsdDiag
        )

        values = gtsam.Values()
        values.insert(poseKey0, p0)
        values.insert(poseKey2, p2)
        values.insert(velocityKey0, v0)
        values.insert(velocityKey2, v0)

        return factor, values

    def testConstruction(self):
        """Test WnoaInterpFactorPose3 construction."""
        factor, _ = self._makeFactorAndValues()
        self.assertIsInstance(factor, WnoaInterpFactorPose3)

    def testPrint(self):
        """Test WnoaInterpFactorPose3 print."""
        factor, _ = self._makeFactorAndValues()
        factor.print()

    def testEquals(self):
        """Test WnoaInterpFactorPose3 equals."""
        factor1, _ = self._makeFactorAndValues()
        factor2, _ = self._makeFactorAndValues()
        self.assertTrue(factor1.equals(factor2, 1e-9))

    def testError(self):
        """Test WnoaInterpFactorPose3 error."""
        factor, values = self._makeFactorAndValues()
        error = factor.error(values)
        self.assertAlmostEqual(error, 0.0, places=9)


class TestWnoaFactorGraphPose3(GtsamTestCase):
    """Test interpolation factor graphs for Pose3."""

    def _makeSe3Fixture(self):
        timeStep = 0.1
        qPsdDiag = np.ones(6)

        p0 = gtsam.Pose3.Expmap(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        v0 = np.array([1.0, 0.0, 0.5, 0.1, 0.0, 0.0])
        p1 = p0.retract(timeStep * v0)
        p2 = p0.retract(2.0 * timeStep * v0)
        p3 = p0.retract(3.0 * timeStep * v0)
        p4 = p0.retract(4.0 * timeStep * v0)

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

        return Se3FixtureData(
            timeStep=timeStep,
            qPsdDiag=qPsdDiag,
            keys=keys,
            p0=p0,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            v0=v0,
        )

    def _makeSe3InterpGraph(self, useWnoaGraph=False):
        fixture = self._makeSe3Fixture()

        model = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        betweenFactor = gtsam.BetweenFactorPose3(
            fixture.keys["p1"], fixture.keys["p3"], fixture.p1.between(fixture.p3), model
        )
        priorPoseFactor = gtsam.PriorFactorPose3(fixture.keys["p1"], fixture.p1, model)
        priorVelFactor = gtsam.PriorFactorVector6(fixture.keys["v1"], fixture.v0, model)

        graph = gtsam.NonlinearFactorGraph()
        graph.add(betweenFactor)
        graph.add(priorPoseFactor)
        graph.add(priorVelFactor)

        estimatedStates = {
            gtsam.StateData(fixture.keys["p0"], fixture.keys["v0"], 0.0),
            gtsam.StateData(fixture.keys["p4"], fixture.keys["v4"], 4.0 * fixture.timeStep),
            gtsam.StateData(fixture.keys["p2"], fixture.keys["v2"], 2.0 * fixture.timeStep),
        }
        interpolatedStates = {
            gtsam.StateData(fixture.keys["p3"], fixture.keys["v3"], 3.0 * fixture.timeStep),
            gtsam.StateData(fixture.keys["p1"], fixture.keys["v1"], fixture.timeStep),
        }

        if useWnoaGraph:
            newGraph = interpolateWnoaFactorGraphPose3(
                graph, estimatedStates, interpolatedStates, fixture.qPsdDiag
            )
        else:
            newGraph = interpolateFactorGraphPose3(
                graph, estimatedStates, interpolatedStates, fixture.qPsdDiag
            )

        values = gtsam.Values()
        values.insert(fixture.keys["p0"], fixture.p0)
        values.insert(fixture.keys["p2"], fixture.p2)
        values.insert(fixture.keys["p4"], fixture.p4)
        values.insert(fixture.keys["v0"], fixture.v0)
        values.insert(fixture.keys["v2"], fixture.v0)
        values.insert(fixture.keys["v4"], fixture.v0)

        return Se3InterpGraphData(
            timeStep=fixture.timeStep,
            qPsdDiag=fixture.qPsdDiag,
            keys=fixture.keys,
            p0=fixture.p0,
            p1=fixture.p1,
            p2=fixture.p2,
            p3=fixture.p3,
            p4=fixture.p4,
            v0=fixture.v0,
            newGraph=newGraph,
            values=values,
            estimatedStates=estimatedStates,
            interpolatedStates=interpolatedStates,
        )

    def _assertCovarianceMap(self, covarianceMap, keys):
        def isPositiveSemidefinite(covariance):
            symmetric = 0.5 * (covariance + covariance.T)
            minEigenvalue = np.min(np.linalg.eigvalsh(symmetric))
            return minEigenvalue >= -1e-9

        self.assertIn(keys["p1"], covarianceMap)
        self.assertIn(keys["v1"], covarianceMap)
        self.assertIn(keys["p3"], covarianceMap)
        self.assertIn(keys["v3"], covarianceMap)

        for key in (keys["p1"], keys["v1"], keys["p3"], keys["v3"]):
            covariance = covarianceMap[key]
            self.assertEqual(covariance.shape, (6, 6))
            self.assertTrue(isPositiveSemidefinite(covariance))

    def testSe3InterpGraph(self):
        """Test interpolateFactorGraphPose3 for SE3 graphs."""
        data = self._makeSe3InterpGraph()

        optimizer = gtsam.GaussNewtonOptimizer(data.newGraph, data.values)
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.iterate()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.optimize()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=6)

        perturb = np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])
        valuesPert = gtsam.Values()
        valuesPert.insert(data.keys["p0"], data.p0.retract(perturb))
        valuesPert.insert(data.keys["p2"], data.p2.retract(perturb))
        valuesPert.insert(data.keys["p4"], data.p4.retract(perturb))
        vPert = data.v0 + 0.1 * np.ones(6)
        valuesPert.insert(data.keys["v0"], vPert)
        valuesPert.insert(data.keys["v2"], vPert)
        valuesPert.insert(data.keys["v4"], vPert)

        optimizer2 = gtsam.GaussNewtonOptimizer(data.newGraph, valuesPert)
        optimizer2.optimize()
        self.assertAlmostEqual(optimizer2.error(), 0.0, places=4)

    def testSe3InterpWnoaGraph(self):
        """Test interpolateWnoaFactorGraphPose3 for SE3 graphs."""
        data = self._makeSe3InterpGraph(useWnoaGraph=True)

        optimizer = gtsam.LevenbergMarquardtOptimizer(data.newGraph, data.values)
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.iterate()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=9)
        optimizer.optimize()
        self.assertAlmostEqual(optimizer.error(), 0.0, places=6)

        perturb = np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])
        valuesPert = gtsam.Values()
        valuesPert.insert(data.keys["p0"], data.p0.retract(perturb))
        valuesPert.insert(data.keys["p2"], data.p2.retract(perturb))
        valuesPert.insert(data.keys["p4"], data.p4.retract(perturb))
        vPert = data.v0 + 0.1 * np.ones(6)
        valuesPert.insert(data.keys["v0"], vPert)
        valuesPert.insert(data.keys["v2"], vPert)
        valuesPert.insert(data.keys["v4"], vPert)

        optimizer2 = gtsam.GaussNewtonOptimizer(data.newGraph, valuesPert)
        optimizer2.optimize()
        self.assertAlmostEqual(optimizer2.error(), 0.0, places=4)

    def testSe3UpdateInterpValues(self):
        """Test updateInterpValuesPose3 for SE3 graphs."""
        data = self._makeSe3InterpGraph()

        resultInterp = updateInterpValuesPose3(
            data.newGraph, data.values, data.estimatedStates, data.interpolatedStates, data.qPsdDiag
        )

        p3Est = resultInterp.atPose3(data.keys["p3"])
        p1Est = resultInterp.atPose3(data.keys["p1"])
        v3Est = resultInterp.atVector6(data.keys["v3"])
        v1Est = resultInterp.atVector6(data.keys["v1"])

        self.gtsamAssertEquals(data.p3, p3Est, 1e-6)
        self.gtsamAssertEquals(data.p1, p1Est, 1e-6)
        np.testing.assert_allclose(data.v0, v3Est, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(data.v0, v1Est, rtol=1e-12, atol=1e-12)

    def testSe3UpdateInterpValuesWithCovariance(self):
        """Test updateInterpValuesWithCovariancePose3 for SE3 graphs."""
        data = self._makeSe3InterpGraph()

        resultInterp, covarianceMap = updateInterpValuesWithCovariancePose3(
            data.newGraph, data.values, data.estimatedStates, data.interpolatedStates, data.qPsdDiag
        )

        p3Est = resultInterp.atPose3(data.keys["p3"])
        p1Est = resultInterp.atPose3(data.keys["p1"])
        v3Est = resultInterp.atVector6(data.keys["v3"])
        v1Est = resultInterp.atVector6(data.keys["v1"])

        self.gtsamAssertEquals(data.p3, p3Est, 1e-6)
        self.gtsamAssertEquals(data.p1, p1Est, 1e-6)
        np.testing.assert_allclose(data.v0, v3Est, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(data.v0, v1Est, rtol=1e-12, atol=1e-12)
        self._assertCovarianceMap(covarianceMap, data.keys)

        
if __name__ == "__main__":
    unittest.main()