#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(rows, fields):
    for row in rows:
        for field in fields:
            row[field] = float(row[field])
        row["query_size"] = int(row["query_size"])
        row["queries"] = int(row["queries"])
        row["support_cliques"] = int(row["support_cliques"])
        row["compressed_cliques"] = int(row["compressed_cliques"])
        row["reduced_state_dim"] = int(row["reduced_state_dim"])
    return rows


def baseline_speedups(rows):
    baselines = {}
    for row in rows:
      if row["variant"] == "legacy_dense":
        baselines[(row["dataset"], row["ordering"], row["query_family"], row["mode"], row["query_size"])] = row["median_total_ms"]

    for row in rows:
      key = (row["dataset"], row["ordering"], row["query_family"], row["mode"], row["query_size"])
      baseline = baselines.get(key, row["median_total_ms"])
      row["speedup"] = baseline / row["median_total_ms"] if row["median_total_ms"] else 1.0
    return rows


def filter_rows(rows, **criteria):
    result = []
    for row in rows:
        if all(row[key] == value for key, value in criteria.items()):
            result.append(row)
    return result


def grouped(rows, *keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def dataset_label(dataset):
    return dataset.removesuffix(".graph").removesuffix(".txt")


def plot_ablation(rows, output_path):
    joint_rows = [row for row in rows if row["mode"] == "joint" and row["query_family"] == "local_window" and row["dataset"] in {"w10000.graph", "w20000.txt"}]
    variants = ["legacy_dense", "steiner_dense", "legacy_solve", "steiner_solve"]
    variant_styles = {
        "legacy_dense": {"color": "#228b22", "linestyle": "--", "marker": "s", "linewidth": 2.0},
        "steiner_dense": {"color": "#ff8c00", "linestyle": "-", "marker": "o", "linewidth": 2.0},
        "legacy_solve": {"color": "#1f77b4", "linestyle": "-.", "marker": "^", "linewidth": 2.0},
        "steiner_solve": {"color": "#b22222", "linestyle": "-", "marker": "D", "linewidth": 2.2},
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (dataset, ordering) in zip(axes.flat, [("w10000.graph", "COLAMD"), ("w10000.graph", "METIS"), ("w20000.txt", "COLAMD"), ("w20000.txt", "METIS")]):
        subset = filter_rows(joint_rows, dataset=dataset, ordering=ordering)
        for variant in variants:
            series = sorted(filter_rows(subset, variant=variant), key=lambda row: row["query_size"])
            if not series:
                continue
            style = variant_styles[variant]
            ax.plot(
                [row["query_size"] for row in series],
                [row["speedup"] for row in series],
                label=variant.replace("_", " "),
                markersize=6,
                zorder=3 if variant == "legacy_dense" else 4,
                **style,
            )
        ax.set_title(f"{dataset_label(dataset)} / {ordering}")
        ax.set_xlabel("Query size")
        ax.set_ylabel("Speedup vs legacy_dense")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path)
    plt.close(figure)


def plot_ordering(rows, output_path):
    target_rows = [
        row
        for row in rows
        if row["mode"] == "joint"
        and row["variant"] in {"legacy_dense", "steiner_solve"}
        and row["dataset"] in {"w10000.graph", "w20000.txt"}
        and row["query_family"] == "local_window"
    ]
    variant_styles = {
        ("legacy_dense", "COLAMD"): {"color": "#228b22", "linestyle": "--", "marker": "s", "linewidth": 2.0},
        ("legacy_dense", "METIS"): {"color": "#66a61e", "linestyle": ":", "marker": "s", "linewidth": 2.0},
        ("steiner_solve", "COLAMD"): {"color": "#b22222", "linestyle": "-", "marker": "D", "linewidth": 2.2},
        ("steiner_solve", "METIS"): {"color": "#ff8c00", "linestyle": "-.", "marker": "o", "linewidth": 2.2},
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset in zip(axes, ["w10000.graph", "w20000.txt"]):
        subset = [row for row in target_rows if row["dataset"] == dataset]
        query_sizes = sorted({row["query_size"] for row in subset})
        for variant, ordering in [
            ("legacy_dense", "COLAMD"),
            ("legacy_dense", "METIS"),
            ("steiner_solve", "COLAMD"),
            ("steiner_solve", "METIS"),
        ]:
            series = sorted(
                [
                    row
                    for row in subset
                    if row["variant"] == variant and row["ordering"] == ordering
                ],
                key=lambda row: row["query_size"],
            )
            if not series:
                continue
            style = variant_styles[(variant, ordering)]
            ax.plot(
                [row["query_size"] for row in series],
                [row["median_total_ms"] for row in series],
                label=f"{ordering} / {variant.replace('_', ' ')}",
                markersize=6,
                **style,
            )
        ax.set_title(dataset_label(dataset))
        ax.set_xlabel("Local-window query size")
        ax.set_xticks(query_sizes, [str(size) for size in query_sizes])
        ax.set_ylabel("Median query time (ms)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path)
    plt.close(figure)


def plot_structure(rows, output_path):
    subset = [row for row in rows if row["mode"] == "joint" and row["variant"] == "steiner_solve" and row["query_family"] == "local_window" and row["dataset"] in {"w10000.graph", "w20000.txt"} and row["ordering"] == "COLAMD"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, dataset in zip(axes, ["w10000.graph", "w20000.txt"]):
        series = sorted([row for row in subset if row["dataset"] == dataset], key=lambda row: row["query_size"])
        x = [row["query_size"] for row in series]
        ax.plot(x, [row["support_cliques"] for row in series], marker="o", label="support cliques")
        ax.plot(x, [row["compressed_cliques"] for row in series], marker="s", label="compressed cliques")
        ax.plot(x, [row["reduced_state_dim"] for row in series], marker="^", label="reduced state dim")
        ax.set_title(dataset_label(dataset))
        ax.set_xlabel("Query size")
        ax.set_ylabel("Count / dimension")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path)
    plt.close(figure)


def plot_selected_cross(rows, output_path):
    subset = [row for row in rows if row["mode"] == "cross" and row["query_family"] == "selected_cross" and row["dataset"] in {"w10000.graph", "w20000.txt"}]
    variants = ["legacy_dense", "steiner_dense", "legacy_solve", "steiner_solve"]
    variant_styles = {
        "legacy_dense": {"color": "#228b22", "linestyle": "--", "marker": "s", "linewidth": 2.0},
        "steiner_dense": {"color": "#ff8c00", "linestyle": "-", "marker": "o", "linewidth": 2.0},
        "legacy_solve": {"color": "#1f77b4", "linestyle": "-.", "marker": "^", "linewidth": 2.0},
        "steiner_solve": {"color": "#b22222", "linestyle": "-", "marker": "D", "linewidth": 2.2},
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset in zip(axes, ["w10000.graph", "w20000.txt"]):
        for variant in variants:
            series = sorted(
                [row for row in subset if row["dataset"] == dataset and row["ordering"] == "COLAMD" and row["variant"] == variant],
                key=lambda row: row["query_size"],
            )
            if not series:
                continue
            style = variant_styles[variant]
            ax.plot(
                [row["query_size"] for row in series],
                [row["speedup"] for row in series],
                label=variant.replace("_", " "),
                markersize=6,
                zorder=3 if variant == "legacy_dense" else 4,
                **style,
            )
        ax.set_title(f"{dataset_label(dataset)} / COLAMD")
        ax.set_xlabel("Query size")
        ax.set_ylabel("Speedup vs legacy_dense")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path)
    plt.close(figure)


def load_key_list(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(row["key"]) for row in reader]


def load_pose_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row["key"] = int(row["key"])
        row["x"] = float(row["x"])
        row["y"] = float(row["y"])
        row["theta"] = float(row["theta"])
        row["in_local"] = int(row["in_local"])
        row["in_wide"] = int(row["in_wide"])
    return rows


def load_edge_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row["key1"] = int(row["key1"])
        row["key2"] = int(row["key2"])
    return rows


def plot_w100_queries(data_dir, output_path):
    pose_rows = load_pose_rows(data_dir / "w100_poses.csv")
    edge_rows = load_edge_rows(data_dir / "w100_edges.csv")
    local_keys = load_key_list(data_dir / "w100_local_keys.csv")
    wide_keys = load_key_list(data_dir / "w100_wide_keys.csv")
    pose_by_key = {row["key"]: row for row in pose_rows}

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    query_specs = [
        ("Local 10-pose query", local_keys, "#b22222"),
        ("Wide-separated 10-pose query", wide_keys, "#1f5aa6"),
    ]

    for ax, (title, query_keys, color) in zip(axes, query_specs):
        for edge in edge_rows:
            pose1 = pose_by_key[edge["key1"]]
            pose2 = pose_by_key[edge["key2"]]
            ax.plot(
                [pose1["x"], pose2["x"]],
                [pose1["y"], pose2["y"]],
                color="#c8c8c8",
                linewidth=0.7,
                alpha=0.7,
                zorder=1,
            )

        ax.scatter(
            [row["x"] for row in pose_rows],
            [row["y"] for row in pose_rows],
            s=10,
            color="#666666",
            alpha=0.65,
            zorder=2,
        )

        query_points = [pose_by_key[key] for key in query_keys]
        ax.plot(
            [row["x"] for row in query_points],
            [row["y"] for row in query_points],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            [row["x"] for row in query_points],
            [row["y"] for row in query_points],
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

        ax.scatter(
            [query_points[0]["x"]],
            [query_points[0]["y"]],
            s=70,
            marker="s",
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            [query_points[-1]["x"]],
            [query_points[-1]["y"]],
            s=70,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

        for key in query_keys[:: max(1, len(query_keys) // 5)]:
            row = pose_by_key[key]
            ax.text(
                row["x"] + 0.12,
                row["y"] + 0.12,
                str(key),
                fontsize=8,
                color=color,
                zorder=6,
            )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def pose_block(covariance, keys, key):
    block_index = keys.index(key)
    start = 3 * block_index
    return covariance[start : start + 3, start : start + 3]


def representative_pose_key(keys, covariance):
    scored = []
    for key in keys:
        marginal = pose_block(covariance, keys, key)
        translation_area = np.linalg.det(marginal[:2, :2])
        scored.append((translation_area, marginal[2, 2], key))
    scored.sort(reverse=True)
    return scored[0][2]


def sample_tangent_gaussian(rng, covariance, size):
    symmetric_covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
    clamped = np.clip(eigenvalues, 0.0, None)
    sqrt_covariance = eigenvectors @ np.diag(np.sqrt(clamped))
    standard = rng.standard_normal((size, covariance.shape[0]))
    return np.einsum("nj,ij->ni", standard, sqrt_covariance)


def exp_se2(tangent):
    vx, vy, omega = tangent
    if abs(omega) < 1e-9:
        return np.array([vx, vy]), omega

    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    a_term = sin_omega / omega
    b_term = (1.0 - cos_omega) / omega
    v_matrix = np.array([[a_term, -b_term], [b_term, a_term]])
    return v_matrix @ np.array([vx, vy]), omega


def retract_pose(mean_pose, tangent):
    translation, omega = exp_se2(tangent)
    mean_theta = mean_pose["theta"]
    rotation = np.array(
        [
            [math.cos(mean_theta), -math.sin(mean_theta)],
            [math.sin(mean_theta), math.cos(mean_theta)],
        ]
    )
    point = np.array([mean_pose["x"], mean_pose["y"]]) + rotation @ translation
    return point[0], point[1], mean_theta + omega


def wrap_degrees(angle_radians):
    return ((np.rad2deg(angle_radians) + 180.0) % 360.0) - 180.0


def draw_heading_arrow(ax, pose_row, length=1.0):
    dx = length * math.cos(pose_row["theta"])
    dy = length * math.sin(pose_row["theta"])
    ax.arrow(
        pose_row["x"],
        pose_row["y"],
        dx,
        dy,
        width=0.03,
        head_width=0.22,
        head_length=0.28,
        color="black",
        length_includes_head=True,
        zorder=5,
    )


def draw_world_sample_ellipse(ax, positions):
    covariance_xy = np.cov(positions[:, :2].T)
    center = positions[:, :2].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 4.0 * np.sqrt(np.maximum(eigenvalues, 0.0))
    ellipse = Ellipse(
        xy=(center[0], center[1]),
        width=width,
        height=height,
        angle=angle,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=4,
    )
    ax.add_patch(ellipse)


def plot_pose2_marginals(data_dir, output_path):
    local_keys = load_key_list(data_dir / "w100_local_keys.csv")
    wide_keys = load_key_list(data_dir / "w100_wide_keys.csv")
    pose_rows = {row["key"]: row for row in load_pose_rows(data_dir / "w100_poses.csv")}
    local_covariance = np.loadtxt(data_dir / "w100_local_covariance.csv", delimiter=",")
    wide_covariance = np.loadtxt(data_dir / "w100_wide_covariance.csv", delimiter=",")
    local_key = representative_pose_key(local_keys, local_covariance)
    wide_key = representative_pose_key(wide_keys, wide_covariance)
    rng = np.random.default_rng(7)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.05]},
    )
    heatmap_axes = axes[:2]
    colorbar_axis = axes[2]
    specifications = [
        ("Representative local Pose2 marginal", local_key, local_keys, local_covariance),
        ("Representative wide-separated Pose2 marginal", wide_key, wide_keys, wide_covariance),
    ]

    for ax, (title, key, keys, covariance) in zip(heatmap_axes, specifications):
        marginal = pose_block(covariance, keys, key)
        mean_pose = pose_rows[key]
        samples_tangent = sample_tangent_gaussian(rng, marginal, size=4000)
        positions = np.array([retract_pose(mean_pose, tangent) for tangent in samples_tangent])
        headings = wrap_degrees(positions[:, 2])
        heatmap = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=headings,
            s=6,
            alpha=0.32,
            cmap="turbo",
            vmin=-180.0,
            vmax=180.0,
            linewidths=0.0,
            zorder=2,
        )
        draw_world_sample_ellipse(ax, positions)
        ax.scatter(
            [mean_pose["x"]],
            [mean_pose["y"]],
            color="black",
            marker="x",
            s=65,
            linewidths=2.0,
            zorder=6,
        )
        draw_heading_arrow(ax, mean_pose)
        ax.text(
            mean_pose["x"] + 0.35,
            mean_pose["y"] + 0.35,
            f"key {key}",
            fontsize=9,
            color="black",
            zorder=7,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    colorbar = figure.colorbar(heatmap, cax=colorbar_axis)
    colorbar.set_label("heading (deg)")
    figure.subplots_adjust(left=0.08, right=0.94, bottom=0.2, top=0.88, wspace=0.32)
    figure.savefig(output_path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-csv-dir", default="")
    parser.add_argument("--visual-data-dir", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = baseline_speedups(
        to_float(
            load_rows(input_path),
            ["median_total_ms", "total_total_ms", "median_reduction_ms", "median_extraction_ms"],
        )
    )

    plot_ablation(rows, output_dir / "results-ablation.pdf")
    plot_ordering(rows, output_dir / "results-ordering.pdf")
    plot_structure(rows, output_dir / "results-structure.pdf")
    plot_selected_cross(rows, output_dir / "results-cross.pdf")

    if args.visual_data_dir:
        visual_data_dir = Path(args.visual_data_dir)
        plot_w100_queries(visual_data_dir, output_dir / "results-w100-queries.pdf")
        plot_pose2_marginals(
            visual_data_dir, output_dir / "results-w100-covariance.pdf"
        )

    if args.copy_csv_dir:
        copy_dir = Path(args.copy_csv_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, copy_dir / input_path.name)


if __name__ == "__main__":
    main()
