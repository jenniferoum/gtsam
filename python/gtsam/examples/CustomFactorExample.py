"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

CustomFactor demo that simulates a 1-D sensor fusion task.
Author: Fan Jiang, Frank Dellaert
"""

from functools import partial
from typing import List, Optional

import gtsam
import numpy as np

Id_1x1 = np.eye(1)


def simulate_car() -> List[float]:
    """Simulate a car for one second"""
    x0 = 0
    dt = 0.25  # 4 Hz, typical GPS
    v = 144 * 1000 / 3600  # 144 km/hour = 90mph, pretty fast
    x = [x0 + v * dt * i for i in range(5)]

    return x


def error_gps(
    measurement: np.ndarray,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    key = this.keys()[0]
    estimate = values.atVector(key)
    error = estimate - measurement
    if jacobians is not None:
        jacobians[0] = Id_1x1
    return error


def error_odom(
    measurement: np.ndarray,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    key1 = this.keys()[0]
    key2 = this.keys()[1]
    pos1, pos2 = values.atVector(key1), values.atVector(key2)
    error = (pos2 - pos1) - measurement
    if jacobians is not None:
        jacobians[0] = -Id_1x1
        jacobians[1] = Id_1x1
    return error


def error_lm(
    measurement: np.ndarray,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    key = this.keys()[0]
    pos = values.atVector(key)
    error = pos - measurement
    if jacobians is not None:
        jacobians[0] = Id_1x1
    return error


def main():
    X = simulate_car()
    print(f"Simulated car trajectory: {X}")

    add_noise = True

    # GPS measurements
    sigma_gps = 3.0
    gps_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_gps)
    gps_sampler = gtsam.Sampler(gps_noise_model, 42)
    gps_measurements = [
        X[k] + (gps_sampler.sample()[0] if add_noise else 0) for k in range(5)
    ]

    # Odometry measurements
    sigma_odo = 0.1
    odo_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_odo)
    odo_sampler = gtsam.Sampler(odo_noise_model, 42)
    odometry_measurements = [
        X[k + 1] - X[k] + (odo_sampler.sample()[0] if add_noise else 0)
        for k in range(4)
    ]

    # Landmark measurements:
    sigma_lm = 1
    lm_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_lm)
    lm_sampler = gtsam.Sampler(lm_noise_model, 42)

    lm_0 = 5.0
    landmark0_measurement = X[0] - lm_0 + (lm_sampler.sample()[0] if add_noise else 0)
    lm_3 = 28.0
    landmark3_measurement = X[3] - lm_3 + (lm_sampler.sample()[0] if add_noise else 0)

    unknown = [gtsam.symbol("x", k) for k in range(5)]
    print("unknowns = ", list(map(gtsam.DefaultKeyFormatter, unknown)))

    # 1. Result with only GPS
    factor_graph = gtsam.NonlinearFactorGraph()
    for k in range(5):
        factor_graph.add(
            gtsam.CustomFactor(
                gps_noise_model,
                [unknown[k]],
                partial(error_gps, np.array([gps_measurements[k]])),
            )
        )

    v = gtsam.Values()
    for i in range(5):
        v.insert(unknown[i], np.array([0.0]))

    params = gtsam.GaussNewtonParams()
    result = gtsam.GaussNewtonOptimizer(factor_graph, v, params).optimize()

    vals = [result.atVector(unknown[k])[0] for k in range(5)]
    print(
        f"Result with only GPS: {np.round(vals, 2)}, Error: {factor_graph.error(result):.6f}"
    )

    # 2. Result with GPS+Odometry
    for k in range(4):
        factor_graph.add(
            gtsam.CustomFactor(
                odo_noise_model,
                [unknown[k], unknown[k + 1]],
                partial(error_odom, np.array([odometry_measurements[k]])),
            )
        )

    result = gtsam.GaussNewtonOptimizer(factor_graph, v, params).optimize()
    vals = [result.atVector(unknown[k])[0] for k in range(5)]
    print(
        f"Result with GPS+Odometry: {np.round(vals, 2)}, Error: {factor_graph.error(result):.6f}"
    )

    # 3. Result with GPS+Odometry+Landmark
    factor_graph.add(
        gtsam.CustomFactor(
            lm_noise_model,
            [unknown[0]],
            partial(error_lm, np.array([lm_0 + landmark0_measurement])),
        )
    )
    factor_graph.add(
        gtsam.CustomFactor(
            lm_noise_model,
            [unknown[3]],
            partial(error_lm, np.array([lm_3 + landmark3_measurement])),
        )
    )

    result = gtsam.GaussNewtonOptimizer(factor_graph, v, params).optimize()
    vals = [result.atVector(unknown[k])[0] for k in range(5)]
    print(
        f"Result with GPS+Odometry+Landmark: {np.round(vals, 2)}, Error: {factor_graph.error(result):.6f}"
    )


if __name__ == "__main__":
    main()
