from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from odor_topology.io import load_embeddings, load_table
from odor_topology.topology import (
    coordinate_permutation_null,
    covariance_matched_gaussian_null,
    fixed_margin_swap_null,
    generalized_tanimoto_distance_matrix,
    prevalence_matched_bernoulli_null,
    row_sum_matched_multinomial_null,
)


def prepare_pom_representation(name: str, path: str | Path) -> dict[str, Any]:
    matrix = load_embeddings(path)
    row_index = np.arange(matrix.shape[0], dtype=int)
    return {
        "name": name,
        "family": "pom",
        "source_path": str(path),
        "row_index": row_index,
        "matrix": np.asarray(matrix, dtype=float),
        "metrics": [
            {"name": "euclidean", "ripser_metric": "euclidean"},
            {"name": "cosine", "ripser_metric": "cosine"},
        ],
        "null_models": {
            "coordinate_permutation": coordinate_permutation_null,
            "covariance_matched_gaussian": covariance_matched_gaussian_null,
        },
        "preprocessing": "raw learned embedding with no rescaling",
    }


def prepare_physchem_representation(path: str | Path) -> dict[str, Any]:
    table = load_table(path)
    if "row_index" not in table.columns:
        raise KeyError(f"Expected 'row_index' column in {path}")

    feature_columns = [column for column in table.columns if column not in {"row_index", "smiles"}]
    ordered = table.sort_values("row_index").reset_index(drop=True)
    raw_matrix = ordered[feature_columns].to_numpy(dtype=float)
    scaled_matrix = StandardScaler().fit_transform(raw_matrix)

    return {
        "name": "rdkit_physchem",
        "family": "chemical_baseline",
        "source_path": str(path),
        "row_index": ordered["row_index"].to_numpy(dtype=int),
        "matrix": scaled_matrix,
        "metrics": [{"name": "euclidean", "ripser_metric": "euclidean"}],
        "null_models": {
            "coordinate_permutation": coordinate_permutation_null,
            "covariance_matched_gaussian": covariance_matched_gaussian_null,
        },
        "preprocessing": (
            f"z-scored continuous descriptors across {scaled_matrix.shape[0]} molecules; "
            f"features={feature_columns}"
        ),
    }


def prepare_bit_fingerprint_representation(
    name: str,
    path: str | Path,
    description: str,
) -> dict[str, Any]:
    data = np.load(path)
    if "fingerprints" not in data or "row_index" not in data:
        raise KeyError(f"Expected 'fingerprints' and 'row_index' arrays in {path}")

    row_index = data["row_index"].astype(int)
    order = np.argsort(row_index)
    aligned_matrix = data["fingerprints"][order].astype(bool)
    aligned_rows = row_index[order]

    return {
        "name": name,
        "family": "chemical_baseline",
        "source_path": str(path),
        "row_index": aligned_rows,
        "matrix": aligned_matrix,
        "metrics": [{"name": "jaccard", "ripser_metric": "jaccard"}],
        "null_models": {
            "fixed_margin_swap": fixed_margin_swap_null,
            "prevalence_matched_bernoulli": prevalence_matched_bernoulli_null,
            "coordinate_permutation": coordinate_permutation_null,
        },
        "preprocessing": description,
    }


def prepare_count_fingerprint_representation(path: str | Path) -> dict[str, Any]:
    data = np.load(path)
    if "fingerprints" not in data or "row_index" not in data:
        raise KeyError(f"Expected 'fingerprints' and 'row_index' arrays in {path}")

    row_index = data["row_index"].astype(int)
    order = np.argsort(row_index)
    aligned_matrix = data["fingerprints"][order].astype(int)
    aligned_rows = row_index[order]

    return {
        "name": "cfp_radius4_2048",
        "family": "chemical_baseline",
        "source_path": str(path),
        "row_index": aligned_rows,
        "matrix": aligned_matrix,
        "metrics": [
            {
                "name": "generalized_tanimoto",
                "distance_matrix_fn": generalized_tanimoto_distance_matrix,
            }
        ],
        "null_models": {
            "row_sum_matched_multinomial": row_sum_matched_multinomial_null,
            "coordinate_permutation": coordinate_permutation_null,
        },
        "preprocessing": (
            "paper-matched count-based Morgan fingerprints (radius 4, 2048 hashed bins) "
            "with generalized Tanimoto distance"
        ),
    }


def common_row_index(representations: list[dict[str, Any]]) -> np.ndarray:
    common = set(representations[0]["row_index"].tolist())
    for representation in representations[1:]:
        common &= set(representation["row_index"].tolist())
    return np.array(sorted(common), dtype=int)


def align_to_common_rows(
    matrix: np.ndarray,
    row_index: np.ndarray,
    common_rows: np.ndarray,
) -> np.ndarray:
    position_lookup = {int(row): idx for idx, row in enumerate(row_index.tolist())}
    indices = np.array([position_lookup[int(row)] for row in common_rows], dtype=int)
    return matrix[indices]


def metric_summary_row(
    representation_name: str,
    family: str,
    metric_name: str,
    preprocessing: str,
    n_rows: int,
    n_dimensions: int,
    null_model_names: list[str],
    result: dict[str, Any],
) -> dict[str, Any]:
    observed = result["observed_summary"]
    robustness = result["robustness_against_nulls"]

    strongest_top1_null_name, strongest_top1_stats = max(
        robustness.items(),
        key=lambda item: item[1]["null_top1_p95"],
    )
    strongest_feature_null_name, strongest_feature_stats = max(
        robustness.items(),
        key=lambda item: item[1]["null_feature_p95"],
    )

    strongest_top1_threshold = float(strongest_top1_stats["null_top1_p95"])
    strongest_feature_threshold = float(strongest_feature_stats["null_feature_p95"])

    summary: dict[str, Any] = {
        "representation": representation_name,
        "family": family,
        "metric": metric_name,
        "n_rows": int(n_rows),
        "n_dimensions": int(n_dimensions),
        "preprocessing": preprocessing,
        "null_models": ",".join(null_model_names),
        "observed_mean_max_h1_persistence": float(observed["mean_max_h1_persistence"]),
        "observed_median_max_h1_persistence": float(observed["median_max_h1_persistence"]),
        "observed_pooled_h1_p95": float(observed["pooled_h1_persistence_p95"]),
        "observed_pooled_h1_p99": float(observed["pooled_h1_persistence_p99"]),
        "strongest_null_top1_name": strongest_top1_null_name,
        "strongest_null_top1_p95": strongest_top1_threshold,
        "top1_signal_to_strongest_null_p95": (
            float(observed["mean_max_h1_persistence"] / strongest_top1_threshold)
            if strongest_top1_threshold > 0
            else None
        ),
        "strongest_null_feature_name": strongest_feature_null_name,
        "strongest_null_feature_p95": strongest_feature_threshold,
        "feature_p95_signal_to_strongest_null_p95": (
            float(observed["pooled_h1_persistence_p95"] / strongest_feature_threshold)
            if strongest_feature_threshold > 0
            else None
        ),
        "min_run_fraction_exceeding_null_top1_p95": float(
            min(
                stats["observed_run_fraction_exceeding_null_top1_p95"]
                for stats in robustness.values()
            )
        ),
        "min_run_fraction_exceeding_null_feature_p95": float(
            min(
                stats["observed_run_fraction_with_any_feature_exceeding_null_feature_p95"]
                for stats in robustness.values()
            )
        ),
    }

    if "analysis_type" in result:
        summary["analysis_type"] = result["analysis_type"]
    if "landmark_count_requested" in result:
        summary["landmark_count_requested"] = int(result["landmark_count_requested"])

    return summary
