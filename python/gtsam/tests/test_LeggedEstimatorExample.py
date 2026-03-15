"""Smoke tests for the staircase legged-estimation Python helper."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from gtsam.examples.LeggedEstimatorExample import (
    SHOWCASE_VARIANTS,
    display_trajectory,
    first_full_contact_time,
    load_legged_csv_dataset,
    make_contact_animation,
    make_contact_side_animation,
    make_legged_estimator,
    replay_legged_estimator,
    replay_legged_variants,
    write_contact_side_animation_gif,
)


class TestLeggedEstimatorExample(unittest.TestCase):
    def test_all_wrapped_estimators_run_on_replay_prefix(self):
        dataset = load_legged_csv_dataset()
        variants = (
            "invariant_ekf",
            "invariant_graph",
            "fixed_lag_single_bias",
            "fixed_lag_combined_bias",
        )

        for variant in variants:
            estimator = make_legged_estimator(variant, dataset["foot_names"])
            result = replay_legged_estimator(
                estimator, dataset, max_duration_seconds=1.5
            )
            final = result["final"]
            estimate = estimator.estimate()
            footholds = np.asarray(estimate.xMatrix(), dtype=float)[:, 2:]
            bias = estimator.estimateBias()
            self.assertGreater(len(result["trajectory"]), 0)
            self.assertTrue(np.isfinite([final["x"], final["y"], final["z"]]).all())
            self.assertTrue(
                np.isfinite([final["roll"], final["pitch"], final["yaw"]]).all()
            )
            self.assertEqual(estimate.k(), 2 + len(dataset["foot_names"]))
            self.assertTrue(
                np.isfinite(np.asarray(bias.accelerometer(), dtype=float)).all()
            )
            self.assertTrue(
                np.isfinite(np.asarray(bias.gyroscope(), dtype=float)).all()
            )
            self.assertEqual(np.asarray(footholds).shape[0], 3)

    def test_showcase_variants_produce_nonempty_trajectories(self):
        dataset = load_legged_csv_dataset()
        results = replay_legged_variants(dataset=dataset)
        self.assertEqual(set(results.keys()), set(SHOWCASE_VARIANTS))
        for result in results.values():
            self.assertGreater(len(result["trajectory"]), 0)
            self.assertGreaterEqual(result["path_length"], 0.0)

    def test_display_trajectory_starts_at_first_full_contact(self):
        dataset = load_legged_csv_dataset()
        start_time_s = first_full_contact_time(dataset)
        estimator = make_legged_estimator("invariant_ekf", dataset["foot_names"])
        result = replay_legged_estimator(estimator, dataset, max_duration_seconds=1.0)
        displayed = display_trajectory(result["trajectory"], start_time_s)

        self.assertGreater(len(displayed), 0)
        self.assertAlmostEqual(displayed[0]["timestamp_s"], 0.0)
        self.assertGreaterEqual(result["trajectory"][-1]["timestamp_s"], start_time_s)

    def test_contact_animation_builds_from_replay_result(self):
        dataset = load_legged_csv_dataset()
        estimator = make_legged_estimator("fixed_lag_single_bias", dataset["foot_names"])
        result = replay_legged_estimator(estimator, dataset, max_duration_seconds=10.0)
        figure3d = make_contact_animation(
            result,
            "Fixed-lag smoother (single bias)",
            start_time_s=first_full_contact_time(dataset),
        )
        figure2d = make_contact_side_animation(
            result,
            "Fixed-lag smoother (single bias)",
            start_time_s=first_full_contact_time(dataset),
        )

        self.assertGreater(len(result["contact_frames"]), 0)
        self.assertTrue(
            any(len(frame["breadcrumb_positions"]) > 0 for frame in result["contact_frames"])
        )
        self.assertGreater(len(figure3d.frames), 0)
        self.assertEqual(figure3d.frames[0].data[0].type, "scatter3d")
        self.assertTrue(bool(figure3d.layout.sliders))
        self.assertTrue(bool(figure3d.layout.updatemenus))
        self.assertGreater(len(figure2d.frames), 0)
        self.assertEqual(figure2d.frames[0].data[0].type, "scatter")
        self.assertTrue(bool(figure2d.layout.sliders))
        self.assertTrue(bool(figure2d.layout.updatemenus))

    def test_contact_side_animation_gif_writes_file(self):
        dataset = load_legged_csv_dataset()
        estimator = make_legged_estimator("fixed_lag_single_bias", dataset["foot_names"])
        result = replay_legged_estimator(estimator, dataset, max_duration_seconds=2.0)
        output_dir = Path(tempfile.mkdtemp(prefix="legged_gif_test_"))
        output_path = output_dir / "stairs_side.gif"
        write_contact_side_animation_gif(
            result,
            "Fixed-lag smoother (single bias)",
            output_path,
            start_time_s=first_full_contact_time(dataset),
            speedup=5.0,
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()
