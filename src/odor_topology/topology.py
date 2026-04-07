from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def sample_indices(n_rows: int, max_points: int, seed: int) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def sample_rows(matrix: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    indices = sample_indices(matrix.shape[0], max_points=max_points, seed=seed)
    return matrix[indices]


def pairwise_distance_summary(
    matrix: np.ndarray,
    metric: str,
    max_points: int,
    seed: int,
) -> dict[str, float]:
    sampled = sample_rows(matrix, max_points=max_points, seed=seed)
    distances = pairwise_distances(sampled, metric=metric)
    upper = distances[np.triu_indices_from(distances, k=1)]
    return {
        "metric": metric,
        "n_points_used": int(sampled.shape[0]),
        "q05": float(np.quantile(upper, 0.05)),
        "q50": float(np.quantile(upper, 0.50)),
        "q95": float(np.quantile(upper, 0.95)),
        "mean": float(np.mean(upper)),
    }


def pca_summary(matrix: np.ndarray, max_components: int) -> dict[str, Any]:
    n_components = min(max_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return {
        "n_components": int(n_components),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_explained_variance_ratio": [
            float(x) for x in np.cumsum(pca.explained_variance_ratio_)
        ],
    }


def ripser_h1_summary(
    matrix: np.ndarray,
    metric: str,
    max_points: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from ripser import ripser
    except ImportError:
        return {
            "available": False,
            "message": "ripser is not installed; topology computation skipped",
        }

    sampled = sample_rows(matrix, max_points=max_points, seed=seed)
    result = ripser(sampled, maxdim=1, metric=metric)
    diagrams = result["dgms"]
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
    persistence = h1[:, 1] - h1[:, 0] if len(h1) else np.array([])
    top_persistence = np.sort(persistence)[::-1][:10]
    return {
        "available": True,
        "metric": metric,
        "n_points_used": int(sampled.shape[0]),
        "n_h1_features": int(len(h1)),
        "top_h1_persistence": [float(x) for x in top_persistence.tolist()],
    }


def h1_persistence_values(
    matrix: np.ndarray,
    metric: str | None = None,
    distance_matrix: bool = False,
) -> np.ndarray:
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError("ripser is not installed; topology computation skipped") from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The input point cloud has more columns than rows; did you mean to transpose?",
        )
        if distance_matrix:
            result = ripser(matrix, maxdim=1, distance_matrix=True)
        else:
            if metric is None:
                raise ValueError("metric must be provided when distance_matrix is False")
            result = ripser(matrix, maxdim=1, metric=metric)
    diagrams = result["dgms"]
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
    if len(h1) == 0:
        return np.array([], dtype=float)

    persistence = h1[:, 1] - h1[:, 0]
    persistence = persistence[np.isfinite(persistence)]
    return np.sort(persistence)[::-1]


def h1_run_summary(
    matrix: np.ndarray,
    metric: str | None = None,
    distance_matrix: bool = False,
    top_k: int = 10,
) -> dict[str, Any]:
    persistence = h1_persistence_values(
        matrix=matrix,
        metric=metric,
        distance_matrix=distance_matrix,
    )
    return {
        "n_points_used": int(matrix.shape[0]),
        "n_h1_features": int(len(persistence)),
        "max_h1_persistence": float(persistence[0]) if len(persistence) else 0.0,
        "mean_h1_persistence": float(np.mean(persistence)) if len(persistence) else 0.0,
        "top_h1_persistence": [float(x) for x in persistence[:top_k].tolist()],
        "all_h1_persistence": [float(x) for x in persistence.tolist()],
    }


def pca_compress(
    matrix: np.ndarray,
    explained_variance_threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if not 0 < explained_variance_threshold <= 1:
        raise ValueError("explained_variance_threshold must be in (0, 1]")

    pca = PCA(n_components=explained_variance_threshold, svd_solver="full")
    reduced = pca.fit_transform(matrix)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return reduced, {
        "n_components": int(reduced.shape[1]),
        "explained_variance_ratio": explained,
    }


def coordinate_permutation_null(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        permuted[:, column] = rng.permutation(matrix[:, column])
    return permuted


def covariance_matched_gaussian_null(
    matrix: np.ndarray,
    seed: int,
    jitter: float = 1e-9,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.mean(matrix, axis=0)
    covariance = np.cov(matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues + jitter))
    standard_normal = rng.normal(size=matrix.shape)
    return standard_normal @ transform.T + mean


def prevalence_matched_bernoulli_null(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.clip(np.mean(matrix, axis=0), a_min=0.0, a_max=1.0)
    samples = rng.binomial(1, probabilities, size=matrix.shape)
    return np.asarray(samples, dtype=bool)


def row_sum_matched_multinomial_null(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    integer_matrix = np.asarray(matrix, dtype=int)
    if np.any(integer_matrix < 0):
        raise ValueError("row_sum_matched_multinomial_null expects nonnegative counts")

    column_sums = np.sum(integer_matrix, axis=0)
    total = int(np.sum(column_sums))
    if total == 0:
        return np.zeros_like(integer_matrix, dtype=int)

    probabilities = column_sums / total
    row_sums = np.sum(integer_matrix, axis=1).astype(int)
    sampled_rows = [
        rng.multinomial(row_sum, probabilities) if row_sum > 0 else np.zeros_like(column_sums)
        for row_sum in row_sums
    ]
    return np.asarray(sampled_rows, dtype=int)


def fixed_margin_swap_null(
    matrix: np.ndarray,
    seed: int,
    n_swap_factor: int = 5,
    max_attempt_factor: int = 50,
) -> np.ndarray:
    bool_matrix = np.asarray(matrix, dtype=bool).copy()
    n_rows, _ = bool_matrix.shape
    row_sets = [set(np.flatnonzero(bool_matrix[row]).tolist()) for row in range(n_rows)]
    total_ones = int(np.sum(bool_matrix))
    if total_ones == 0:
        return bool_matrix

    rng = np.random.default_rng(seed)
    target_swaps = max(total_ones * n_swap_factor, 1)
    max_attempts = max(target_swaps * max_attempt_factor, 1)
    successful_swaps = 0
    attempts = 0

    while successful_swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        row_a, row_b = rng.choice(n_rows, size=2, replace=False)
        only_a = list(row_sets[row_a] - row_sets[row_b])
        only_b = list(row_sets[row_b] - row_sets[row_a])
        if not only_a or not only_b:
            continue

        col_a = int(rng.choice(only_a))
        col_b = int(rng.choice(only_b))

        bool_matrix[row_a, col_a] = False
        bool_matrix[row_b, col_b] = False
        bool_matrix[row_a, col_b] = True
        bool_matrix[row_b, col_a] = True

        row_sets[row_a].remove(col_a)
        row_sets[row_b].remove(col_b)
        row_sets[row_a].add(col_b)
        row_sets[row_b].add(col_a)
        successful_swaps += 1

    return bool_matrix


def generalized_tanimoto_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    nonnegative = np.asarray(matrix, dtype=float)
    if np.any(nonnegative < 0):
        raise ValueError("generalized_tanimoto_distance_matrix expects nonnegative values")

    gram = nonnegative @ nonnegative.T
    squared_norms = np.sum(nonnegative * nonnegative, axis=1, keepdims=True)
    denominator = squared_norms + squared_norms.T - gram
    similarity = np.divide(
        gram,
        denominator,
        out=np.ones_like(gram),
        where=denominator > 0,
    )
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def greedy_farthest_point_landmarks(
    distance_matrix: np.ndarray,
    n_landmarks: int,
    seed: int,
) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("greedy_farthest_point_landmarks expects a square distance matrix")

    n_points = distances.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)

    n_landmarks = max(1, min(int(n_landmarks), n_points))
    rng = np.random.default_rng(seed)
    first_index = int(rng.integers(n_points))

    selected = [first_index]
    min_distance_to_selected = distances[first_index].copy()
    min_distance_to_selected[first_index] = -np.inf

    while len(selected) < n_landmarks:
        next_index = int(np.argmax(min_distance_to_selected))
        if next_index in selected:
            break
        selected.append(next_index)
        min_distance_to_selected = np.minimum(
            min_distance_to_selected,
            distances[next_index],
        )
        min_distance_to_selected[selected] = -np.inf

    return np.asarray(selected, dtype=int)


def landmark_distance_submatrix(
    distance_matrix: np.ndarray,
    n_landmarks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = greedy_farthest_point_landmarks(
        distance_matrix=distance_matrix,
        n_landmarks=n_landmarks,
        seed=seed,
    )
    return distance_matrix[np.ix_(indices, indices)], indices
