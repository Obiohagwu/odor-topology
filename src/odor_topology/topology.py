from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def sample_rows(matrix: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= max_points:
        return matrix
    rng = np.random.default_rng(seed)
    indices = rng.choice(matrix.shape[0], size=max_points, replace=False)
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
