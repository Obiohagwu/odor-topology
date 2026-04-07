from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances

from odor_topology.topology import (
    h1_run_summary,
    landmark_distance_submatrix,
    sample_indices,
)


NullModelFn = Callable[[np.ndarray, int], np.ndarray]
DistanceMatrixFn = Callable[[np.ndarray], np.ndarray]


def build_sample_index_runs(
    n_rows: int,
    max_points: int,
    runs: int,
    base_seed: int,
) -> list[np.ndarray]:
    return [
        sample_indices(n_rows=n_rows, max_points=max_points, seed=base_seed + run_index)
        for run_index in range(runs)
    ]


def summarize_runs(runs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    max_values = np.array([run["max_h1_persistence"] for run in runs], dtype=float)
    feature_counts = np.array([run["n_h1_features"] for run in runs], dtype=float)
    pooled = np.array(
        [value for run in runs for value in run["all_h1_persistence"]],
        dtype=float,
    )
    top_matrix = np.array(
        [
            run["top_h1_persistence"] + [0.0] * max(0, top_k - len(run["top_h1_persistence"]))
            for run in runs
        ],
        dtype=float,
    )

    summary: dict[str, Any] = {
        "n_runs": int(len(runs)),
        "median_max_h1_persistence": float(np.median(max_values)) if len(max_values) else 0.0,
        "mean_max_h1_persistence": float(np.mean(max_values)) if len(max_values) else 0.0,
        "std_max_h1_persistence": float(np.std(max_values)) if len(max_values) else 0.0,
        "median_h1_feature_count": float(np.median(feature_counts)) if len(feature_counts) else 0.0,
        "mean_h1_feature_count": float(np.mean(feature_counts)) if len(feature_counts) else 0.0,
        "pooled_h1_persistence_p95": float(np.quantile(pooled, 0.95)) if len(pooled) else 0.0,
        "pooled_h1_persistence_p99": float(np.quantile(pooled, 0.99)) if len(pooled) else 0.0,
        "mean_top_h1_persistence": [],
    }

    if len(top_matrix):
        summary["mean_top_h1_persistence"] = [
            float(value) for value in np.mean(top_matrix, axis=0).tolist()
        ]

    return summary


def compact_run_record(run: dict[str, Any]) -> dict[str, Any]:
    compact = dict(run)
    persistence = np.array(compact.pop("all_h1_persistence"), dtype=float)
    compact["h1_persistence_p95"] = float(np.quantile(persistence, 0.95)) if len(persistence) else 0.0
    compact["h1_persistence_p99"] = float(np.quantile(persistence, 0.99)) if len(persistence) else 0.0
    return compact


def build_robustness_summary(
    observed_runs: list[dict[str, Any]],
    null_runs_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    observed_max = np.array([run["max_h1_persistence"] for run in observed_runs], dtype=float)
    summary: dict[str, Any] = {}

    for null_name, null_runs in null_runs_by_name.items():
        null_max = np.array([run["max_h1_persistence"] for run in null_runs], dtype=float)
        null_pooled = np.array(
            [value for run in null_runs for value in run["all_h1_persistence"]],
            dtype=float,
        )
        top1_p95 = float(np.quantile(null_max, 0.95)) if len(null_max) else 0.0
        pooled_p95 = float(np.quantile(null_pooled, 0.95)) if len(null_pooled) else 0.0
        pooled_p99 = float(np.quantile(null_pooled, 0.99)) if len(null_pooled) else 0.0

        summary[null_name] = {
            "null_top1_p95": top1_p95,
            "null_feature_p95": pooled_p95,
            "null_feature_p99": pooled_p99,
            "observed_runs_exceeding_null_top1_p95": int(np.sum(observed_max > top1_p95)),
            "observed_run_fraction_exceeding_null_top1_p95": (
                float(np.mean(observed_max > top1_p95)) if len(observed_max) else 0.0
            ),
            "observed_runs_with_any_feature_exceeding_null_feature_p95": int(
                sum(
                    any(value > pooled_p95 for value in run["all_h1_persistence"])
                    for run in observed_runs
                )
            ),
            "observed_run_fraction_with_any_feature_exceeding_null_feature_p95": (
                float(
                    np.mean(
                        [
                            any(value > pooled_p95 for value in run["all_h1_persistence"])
                            for run in observed_runs
                        ]
                    )
                )
                if observed_runs
                else 0.0
            ),
        }

    return summary


def run_metric_analysis(
    matrix: np.ndarray,
    metric_spec: dict[str, Any],
    sample_index_runs: list[np.ndarray],
    top_k: int,
    null_models: dict[str, NullModelFn],
    base_seed: int,
) -> dict[str, Any]:
    observed_runs: list[dict[str, Any]] = []
    null_runs_by_name: dict[str, list[dict[str, Any]]] = {
        null_name: [] for null_name in null_models
    }

    metric_name = str(metric_spec["name"])
    ripser_metric = metric_spec.get("ripser_metric")
    distance_matrix_fn: DistanceMatrixFn | None = metric_spec.get("distance_matrix_fn")

    for run_index, indices in enumerate(sample_index_runs):
        sampled = matrix[indices]
        if distance_matrix_fn is not None:
            observed_input = distance_matrix_fn(sampled)
            observed = h1_run_summary(
                observed_input,
                metric=None,
                distance_matrix=True,
                top_k=top_k,
            )
        else:
            observed = h1_run_summary(
                sampled,
                metric=str(ripser_metric),
                distance_matrix=False,
                top_k=top_k,
            )
        observed["run_index"] = int(run_index)
        observed["sample_seed"] = int(base_seed + run_index)
        observed["metric_name"] = metric_name
        observed_runs.append(observed)

        for null_offset, (null_name, null_fn) in enumerate(null_models.items(), start=1):
            null_seed = int(base_seed + run_index + null_offset * 10_000)
            null_matrix = null_fn(sampled, null_seed)
            if distance_matrix_fn is not None:
                null_input = distance_matrix_fn(null_matrix)
                null_summary = h1_run_summary(
                    null_input,
                    metric=None,
                    distance_matrix=True,
                    top_k=top_k,
                )
            else:
                null_summary = h1_run_summary(
                    null_matrix,
                    metric=str(ripser_metric),
                    distance_matrix=False,
                    top_k=top_k,
                )
            null_summary["run_index"] = int(run_index)
            null_summary["null_seed"] = null_seed
            null_summary["metric_name"] = metric_name
            null_runs_by_name[null_name].append(null_summary)

    return {
        "observed_runs": [compact_run_record(run) for run in observed_runs],
        "observed_summary": summarize_runs(observed_runs, top_k=top_k),
        "null_models": {
            null_name: {
                "runs": [compact_run_record(run) for run in null_runs],
                "summary": summarize_runs(null_runs, top_k=top_k),
            }
            for null_name, null_runs in null_runs_by_name.items()
        },
        "robustness_against_nulls": build_robustness_summary(
            observed_runs=observed_runs,
            null_runs_by_name=null_runs_by_name,
        ),
    }


def run_landmark_metric_analysis(
    matrix: np.ndarray,
    metric_spec: dict[str, Any],
    sample_index_runs: list[np.ndarray],
    top_k: int,
    null_models: dict[str, NullModelFn],
    base_seed: int,
    n_landmarks: int,
) -> dict[str, Any]:
    observed_runs: list[dict[str, Any]] = []
    null_runs_by_name: dict[str, list[dict[str, Any]]] = {
        null_name: [] for null_name in null_models
    }

    metric_name = str(metric_spec["name"])
    ripser_metric = metric_spec.get("ripser_metric")
    distance_matrix_fn: DistanceMatrixFn | None = metric_spec.get("distance_matrix_fn")

    def build_distance_matrix(sampled_matrix: np.ndarray) -> np.ndarray:
        if distance_matrix_fn is not None:
            return np.asarray(distance_matrix_fn(sampled_matrix), dtype=float)
        return np.asarray(pairwise_distances(sampled_matrix, metric=str(ripser_metric)), dtype=float)

    for run_index, indices in enumerate(sample_index_runs):
        sampled = matrix[indices]
        landmark_seed = int(base_seed + run_index + 500_000)
        observed_distance_matrix = build_distance_matrix(sampled)
        observed_landmark_distances, landmark_indices = landmark_distance_submatrix(
            distance_matrix=observed_distance_matrix,
            n_landmarks=n_landmarks,
            seed=landmark_seed,
        )
        observed = h1_run_summary(
            observed_landmark_distances,
            metric=None,
            distance_matrix=True,
            top_k=top_k,
        )
        observed["run_index"] = int(run_index)
        observed["sample_seed"] = int(base_seed + run_index)
        observed["landmark_seed"] = landmark_seed
        observed["metric_name"] = metric_name
        observed["n_landmarks_used"] = int(len(landmark_indices))
        observed_runs.append(observed)

        for null_offset, (null_name, null_fn) in enumerate(null_models.items(), start=1):
            null_seed = int(base_seed + run_index + null_offset * 10_000)
            null_landmark_seed = int(landmark_seed + null_offset * 1_000)
            null_matrix = null_fn(sampled, null_seed)
            null_distance_matrix = build_distance_matrix(null_matrix)
            null_landmark_distances, null_landmark_indices = landmark_distance_submatrix(
                distance_matrix=null_distance_matrix,
                n_landmarks=n_landmarks,
                seed=null_landmark_seed,
            )
            null_summary = h1_run_summary(
                null_landmark_distances,
                metric=None,
                distance_matrix=True,
                top_k=top_k,
            )
            null_summary["run_index"] = int(run_index)
            null_summary["null_seed"] = null_seed
            null_summary["landmark_seed"] = null_landmark_seed
            null_summary["metric_name"] = metric_name
            null_summary["n_landmarks_used"] = int(len(null_landmark_indices))
            null_runs_by_name[null_name].append(null_summary)

    return {
        "analysis_type": "greedy_landmark_distance_matrix",
        "landmark_count_requested": int(n_landmarks),
        "observed_runs": [compact_run_record(run) for run in observed_runs],
        "observed_summary": summarize_runs(observed_runs, top_k=top_k),
        "null_models": {
            null_name: {
                "runs": [compact_run_record(run) for run in null_runs],
                "summary": summarize_runs(null_runs, top_k=top_k),
            }
            for null_name, null_runs in null_runs_by_name.items()
        },
        "robustness_against_nulls": build_robustness_summary(
            observed_runs=observed_runs,
            null_runs_by_name=null_runs_by_name,
        ),
    }
