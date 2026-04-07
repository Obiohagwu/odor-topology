from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.audit import summarize_embedding_matrix
from odor_topology.config import load_project_config
from odor_topology.io import load_embeddings
from odor_topology.robustness import build_sample_index_runs, run_metric_analysis
from odor_topology.topology import coordinate_permutation_null, covariance_matched_gaussian_null


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit H1 robustness across all extracted OpenPOM ensemble embeddings."
    )
    parser.add_argument("--config", required=True, help="Path to project config JSON")
    parser.add_argument(
        "--embeddings-glob",
        default=str(PROJECT_ROOT / "data" / "processed" / "openpom_embeddings_exp*.npy"),
        help="Glob for extracted embedding matrices",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Points per repeated subsample run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=8,
        help="Number of repeated observed subsample runs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top persistence values to retain per run",
    )
    parser.add_argument(
        "--distance-sample-size",
        type=int,
        default=1000,
        help="Sample size for checkpoint-to-checkpoint distance correlation checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)

    embedding_paths = [Path(path) for path in sorted(glob(args.embeddings_glob))]
    if not embedding_paths:
        raise FileNotFoundError(f"No embeddings matched {args.embeddings_glob}")

    embeddings: dict[str, np.ndarray] = {
        path.stem: load_embeddings(path) for path in embedding_paths
    }
    n_rows = {matrix.shape[0] for matrix in embeddings.values()}
    if len(n_rows) != 1:
        raise ValueError(f"Expected all embeddings to share row count, got {sorted(n_rows)}")

    sample_index_runs = build_sample_index_runs(
        n_rows=next(iter(n_rows)),
        max_points=args.max_points,
        runs=args.runs,
        base_seed=config.random_seed,
    )

    report: dict[str, object] = {
        "settings": {
            "max_points": int(args.max_points),
            "runs": int(args.runs),
            "top_k": int(args.top_k),
            "distance_sample_size": int(args.distance_sample_size),
            "random_seed": int(config.random_seed),
        },
        "embeddings": {},
        "pairwise_distance_agreement": [],
    }

    summary_rows: list[dict[str, object]] = []
    for stem, matrix in embeddings.items():
        metrics_report: dict[str, object] = {}
        for metric_name in ["euclidean", "cosine"]:
            result = run_metric_analysis(
                matrix=matrix,
                metric_spec={"name": metric_name, "ripser_metric": metric_name},
                sample_index_runs=sample_index_runs,
                top_k=args.top_k,
                null_models={
                    "coordinate_permutation": coordinate_permutation_null,
                    "covariance_matched_gaussian": covariance_matched_gaussian_null,
                },
                base_seed=config.random_seed,
            )
            metrics_report[metric_name] = result
            observed = result["observed_summary"]
            strongest_top1_null = max(
                stats["null_top1_p95"]
                for stats in result["robustness_against_nulls"].values()
            )
            strongest_feature_null = max(
                stats["null_feature_p95"]
                for stats in result["robustness_against_nulls"].values()
            )
            summary_rows.append(
                {
                    "embedding": stem,
                    "metric": metric_name,
                    "observed_mean_max_h1_persistence": float(observed["mean_max_h1_persistence"]),
                    "observed_pooled_h1_p95": float(observed["pooled_h1_persistence_p95"]),
                    "top1_signal_to_strongest_null_p95": (
                        float(observed["mean_max_h1_persistence"] / strongest_top1_null)
                        if strongest_top1_null > 0
                        else None
                    ),
                    "feature_p95_signal_to_strongest_null_p95": (
                        float(observed["pooled_h1_persistence_p95"] / strongest_feature_null)
                        if strongest_feature_null > 0
                        else None
                    ),
                    "min_run_fraction_exceeding_null_top1_p95": float(
                        min(
                            stats["observed_run_fraction_exceeding_null_top1_p95"]
                            for stats in result["robustness_against_nulls"].values()
                        )
                    ),
                    "min_run_fraction_exceeding_null_feature_p95": float(
                        min(
                            stats["observed_run_fraction_with_any_feature_exceeding_null_feature_p95"]
                            for stats in result["robustness_against_nulls"].values()
                        )
                    ),
                }
            )

        report["embeddings"][stem] = {
            "matrix_summary": summarize_embedding_matrix(matrix),
            "metrics": metrics_report,
        }

    rng = np.random.default_rng(config.random_seed)
    sample_size = min(args.distance_sample_size, next(iter(n_rows)))
    sampled_rows = np.sort(rng.choice(next(iter(n_rows)), size=sample_size, replace=False))
    for stem_a, stem_b in combinations(sorted(embeddings), 2):
        matrix_a = embeddings[stem_a][sampled_rows]
        matrix_b = embeddings[stem_b][sampled_rows]
        distances_a = pairwise_distances(matrix_a, metric="euclidean")
        distances_b = pairwise_distances(matrix_b, metric="euclidean")
        upper_a = distances_a[np.triu_indices_from(distances_a, k=1)]
        upper_b = distances_b[np.triu_indices_from(distances_b, k=1)]
        report["pairwise_distance_agreement"].append(
            {
                "embedding_a": stem_a,
                "embedding_b": stem_b,
                "sample_size": int(sample_size),
                "pairwise_distance_pearson": float(np.corrcoef(upper_a, upper_b)[0, 1]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["metric", "embedding"]).reset_index(drop=True)
    report_path = PROJECT_ROOT / "outputs" / "reports" / "openpom_ensemble_checkpoint_audit.json"
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "openpom_ensemble_checkpoint_summary.csv"
    report_path.write_text(json.dumps(report, indent=2))
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nWrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
