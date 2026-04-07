from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from odor_topology.audit import infer_binary_label_columns
from odor_topology.config import ProjectConfig
from odor_topology.topology import h1_persistence_values


def resolve_label_columns(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> list[str]:
    if config.label_columns:
        return list(config.label_columns)
    exclude_columns = [config.id_column, config.smiles_column, *config.ignore_columns]
    return infer_binary_label_columns(df, exclude_columns=exclude_columns)


def build_binary_label_frame(
    df: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    if not label_columns:
        return pd.DataFrame(index=df.index)
    label_frame = df[label_columns].fillna(0)
    label_frame = label_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (label_frame > 0).astype(np.uint8)


def compute_neighborhood_graph(
    matrix: np.ndarray,
    metric_spec: dict[str, Any],
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_rows = matrix.shape[0]
    effective_neighbors = max(1, min(int(n_neighbors), n_rows - 1))
    distance_matrix_fn = metric_spec.get("distance_matrix_fn")
    ripser_metric = metric_spec.get("ripser_metric")

    if distance_matrix_fn is not None:
        global_distance_matrix = np.asarray(distance_matrix_fn(matrix), dtype=float)
        distances = global_distance_matrix.copy()
        np.fill_diagonal(distances, np.inf)
        neighbor_indices = np.argsort(distances, axis=1)[:, :effective_neighbors]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        return neighbor_indices, neighbor_distances, global_distance_matrix

    model = NearestNeighbors(
        n_neighbors=effective_neighbors + 1,
        algorithm="brute",
        metric=str(ripser_metric),
    )
    model.fit(matrix)
    raw_distances, raw_indices = model.kneighbors(matrix)

    neighbor_indices = np.empty((n_rows, effective_neighbors), dtype=int)
    neighbor_distances = np.empty((n_rows, effective_neighbors), dtype=float)
    for row_index in range(n_rows):
        mask = raw_indices[row_index] != row_index
        filtered_indices = raw_indices[row_index][mask][:effective_neighbors]
        filtered_distances = raw_distances[row_index][mask][:effective_neighbors]
        if len(filtered_indices) != effective_neighbors:
            raise ValueError(
                f"Expected {effective_neighbors} neighbors for row {row_index}, "
                f"got {len(filtered_indices)}"
            )
        neighbor_indices[row_index] = filtered_indices
        neighbor_distances[row_index] = filtered_distances

    return neighbor_indices, neighbor_distances, None


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    entropy = np.zeros_like(probs, dtype=float)
    mask = (probs > 0.0) & (probs < 1.0)
    entropy[mask] = -(
        probs[mask] * np.log2(probs[mask]) +
        (1.0 - probs[mask]) * np.log2(1.0 - probs[mask])
    )
    return entropy


def local_linear_geometry_features(
    patch_matrix: np.ndarray,
    max_components: int = 3,
) -> dict[str, float]:
    centered = np.asarray(patch_matrix, dtype=float) - np.mean(patch_matrix, axis=0, keepdims=True)
    if centered.shape[0] < 2 or centered.shape[1] == 0:
        return {
            "local_participation_ratio": 0.0,
            "local_pca_reconstruction_error_ratio": 0.0,
        }

    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variances = singular_values**2
    total_variance = float(np.sum(variances))
    if total_variance <= 0:
        return {
            "local_participation_ratio": 0.0,
            "local_pca_reconstruction_error_ratio": 0.0,
        }

    participation_ratio = float(
        (total_variance**2) / max(np.sum(variances**2), 1e-12)
    )
    n_components = min(max_components, len(variances))
    residual_variance = float(np.sum(variances[n_components:]))
    reconstruction_error_ratio = residual_variance / total_variance

    return {
        "local_participation_ratio": participation_ratio,
        "local_pca_reconstruction_error_ratio": reconstruction_error_ratio,
    }


def build_local_feature_table(
    matrix: np.ndarray,
    metric_spec: dict[str, Any],
    label_matrix: np.ndarray,
    row_index: np.ndarray,
    label_columns: list[str],
    n_neighbors: int,
    progress_every: int = 250,
) -> pd.DataFrame:
    neighbor_indices, neighbor_distances, global_distance_matrix = compute_neighborhood_graph(
        matrix=matrix,
        metric_spec=metric_spec,
        n_neighbors=n_neighbors,
    )
    ripser_metric = metric_spec.get("ripser_metric")

    records: list[dict[str, Any]] = []
    for focal_index in range(matrix.shape[0]):
        if progress_every and (focal_index + 1) % progress_every == 0:
            print(
                f"  processed {focal_index + 1}/{matrix.shape[0]} neighborhoods "
                f"for metric={metric_spec['name']}"
            )

        local_neighbor_indices = neighbor_indices[focal_index]
        patch_indices = np.concatenate(([focal_index], local_neighbor_indices))
        patch_matrix = matrix[patch_indices]

        if global_distance_matrix is not None:
            patch_distances = global_distance_matrix[np.ix_(patch_indices, patch_indices)]
        else:
            patch_distances = pairwise_distances(
                patch_matrix,
                metric=str(ripser_metric),
            )

        focal_neighbor_distances = patch_distances[0, 1:]
        upper_triangle = patch_distances[np.triu_indices_from(patch_distances, k=1)]

        h1_values = h1_persistence_values(
            patch_distances,
            metric=None,
            distance_matrix=True,
        )
        linear_features = local_linear_geometry_features(patch_matrix)

        focal_labels = label_matrix[focal_index]
        neighbor_labels = label_matrix[local_neighbor_indices]
        focal_label_count = int(np.sum(focal_labels))
        neighbor_prevalence = np.mean(neighbor_labels, axis=0) if len(label_columns) else np.array([])
        entropy_all = float(np.mean(binary_entropy(neighbor_prevalence))) if len(label_columns) else np.nan
        active_mask = (
            (neighbor_prevalence > 0) | (focal_labels > 0)
        ) if len(label_columns) else np.array([], dtype=bool)
        entropy_active = (
            float(np.mean(binary_entropy(neighbor_prevalence[active_mask])))
            if len(label_columns) and np.any(active_mask)
            else 0.0
        )

        mean_neighbor_label_jaccard = np.nan
        share_any_label_fraction = np.nan
        if len(label_columns) and focal_label_count > 0:
            intersections = np.sum(neighbor_labels * focal_labels, axis=1)
            unions = np.sum((neighbor_labels + focal_labels) > 0, axis=1)
            jaccard = np.divide(
                intersections,
                unions,
                out=np.zeros_like(intersections, dtype=float),
                where=unions > 0,
            )
            mean_neighbor_label_jaccard = float(np.mean(jaccard))
            share_any_label_fraction = float(np.mean(intersections > 0))

        record: dict[str, Any] = {
            "row_index": int(row_index[focal_index]),
            "n_neighbors": int(len(local_neighbor_indices)),
            "focal_label_count": focal_label_count,
            "mean_neighbor_distance": float(np.mean(focal_neighbor_distances)),
            "std_neighbor_distance": float(np.std(focal_neighbor_distances)),
            "p90_neighbor_distance": float(np.quantile(focal_neighbor_distances, 0.90)),
            "max_neighbor_distance": float(np.max(focal_neighbor_distances)),
            "local_patch_distance_mean": float(np.mean(upper_triangle)),
            "local_patch_distance_std": float(np.std(upper_triangle)),
            "local_density_proxy": float(1.0 / (np.mean(focal_neighbor_distances) + 1e-8)),
            "local_h1_feature_count": int(len(h1_values)),
            "local_h1_max_persistence": float(h1_values[0]) if len(h1_values) else 0.0,
            "local_h1_total_persistence": float(np.sum(h1_values)) if len(h1_values) else 0.0,
            "local_h1_mean_persistence": float(np.mean(h1_values)) if len(h1_values) else 0.0,
            "local_h1_p95_persistence": float(np.quantile(h1_values, 0.95)) if len(h1_values) else 0.0,
            "target_mean_neighbor_label_jaccard": mean_neighbor_label_jaccard,
            "target_share_any_label_fraction": share_any_label_fraction,
            "target_neighbor_label_entropy_all": entropy_all,
            "target_neighbor_label_entropy_active": entropy_active,
        }
        record.update(linear_features)
        records.append(record)

    return pd.DataFrame.from_records(records)


def evaluate_regression_models(
    feature_table: pd.DataFrame,
    geometry_columns: list[str],
    topology_columns: list[str],
    target_column: str,
    random_seed: int,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict[str, Any]:
    working = feature_table.dropna(subset=[target_column]).reset_index(drop=True)
    if len(working) < n_splits:
        raise ValueError(
            f"Not enough rows to evaluate target '{target_column}': {len(working)}"
        )

    y = working[target_column].to_numpy(dtype=float)
    feature_blocks = {
        "geometry_only": geometry_columns,
        "topology_only": topology_columns,
        "geometry_plus_topology": geometry_columns + topology_columns,
    }

    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )
    split_indices = list(splitter.split(working))

    per_model_results: dict[str, list[dict[str, float | int]]] = {
        model_name: [] for model_name in feature_blocks
    }

    for split_index, (train_idx, test_idx) in enumerate(split_indices):
        for model_name, columns in feature_blocks.items():
            x_train = working.iloc[train_idx][columns].to_numpy(dtype=float)
            x_test = working.iloc[test_idx][columns].to_numpy(dtype=float)
            y_train = y[train_idx]
            y_test = y[test_idx]

            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            spearman_value = float(spearmanr(y_test, predictions).statistic)
            if np.isnan(spearman_value):
                spearman_value = 0.0

            per_model_results[model_name].append(
                {
                    "split_index": int(split_index),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": spearman_value,
                }
            )

    summary = {
        "target_column": target_column,
        "n_rows_used": int(len(working)),
        "n_splits_total": int(len(split_indices)),
        "models": {},
        "deltas": {},
    }

    for model_name, rows in per_model_results.items():
        r2_values = np.array([row["r2"] for row in rows], dtype=float)
        spearman_values = np.array([row["spearman"] for row in rows], dtype=float)
        summary["models"][model_name] = {
            "splits": rows,
            "mean_r2": float(np.mean(r2_values)),
            "median_r2": float(np.median(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_spearman": float(np.mean(spearman_values)),
            "median_spearman": float(np.median(spearman_values)),
            "std_spearman": float(np.std(spearman_values)),
        }

    geometry_r2 = np.array([row["r2"] for row in per_model_results["geometry_only"]], dtype=float)
    full_r2 = np.array(
        [row["r2"] for row in per_model_results["geometry_plus_topology"]],
        dtype=float,
    )
    geometry_spearman = np.array(
        [row["spearman"] for row in per_model_results["geometry_only"]],
        dtype=float,
    )
    full_spearman = np.array(
        [row["spearman"] for row in per_model_results["geometry_plus_topology"]],
        dtype=float,
    )

    delta_r2 = full_r2 - geometry_r2
    delta_spearman = full_spearman - geometry_spearman
    summary["deltas"]["geometry_plus_topology_minus_geometry_only"] = {
        "mean_delta_r2": float(np.mean(delta_r2)),
        "median_delta_r2": float(np.median(delta_r2)),
        "p05_delta_r2": float(np.quantile(delta_r2, 0.05)),
        "p95_delta_r2": float(np.quantile(delta_r2, 0.95)),
        "fraction_splits_delta_r2_positive": float(np.mean(delta_r2 > 0)),
        "mean_delta_spearman": float(np.mean(delta_spearman)),
        "median_delta_spearman": float(np.median(delta_spearman)),
        "p05_delta_spearman": float(np.quantile(delta_spearman, 0.05)),
        "p95_delta_spearman": float(np.quantile(delta_spearman, 0.95)),
        "fraction_splits_delta_spearman_positive": float(np.mean(delta_spearman > 0)),
    }

    return summary


def summarize_utility_result_row(
    representation_name: str,
    metric_name: str,
    target_column: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    geometry = result["models"]["geometry_only"]
    topology = result["models"]["topology_only"]
    full = result["models"]["geometry_plus_topology"]
    delta = result["deltas"]["geometry_plus_topology_minus_geometry_only"]

    return {
        "representation": representation_name,
        "metric": metric_name,
        "target": target_column,
        "n_rows_used": int(result["n_rows_used"]),
        "n_splits_total": int(result["n_splits_total"]),
        "geometry_only_mean_r2": float(geometry["mean_r2"]),
        "topology_only_mean_r2": float(topology["mean_r2"]),
        "geometry_plus_topology_mean_r2": float(full["mean_r2"]),
        "geometry_only_mean_spearman": float(geometry["mean_spearman"]),
        "topology_only_mean_spearman": float(topology["mean_spearman"]),
        "geometry_plus_topology_mean_spearman": float(full["mean_spearman"]),
        "mean_delta_r2": float(delta["mean_delta_r2"]),
        "fraction_splits_delta_r2_positive": float(delta["fraction_splits_delta_r2_positive"]),
        "mean_delta_spearman": float(delta["mean_delta_spearman"]),
        "fraction_splits_delta_spearman_positive": float(
            delta["fraction_splits_delta_spearman_positive"]
        ),
    }
