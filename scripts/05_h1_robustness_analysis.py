from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.audit import summarize_embedding_matrix
from odor_topology.config import load_project_config
from odor_topology.io import load_embeddings
from odor_topology.robustness import build_sample_index_runs, run_metric_analysis
from odor_topology.topology import (
    coordinate_permutation_null,
    covariance_matched_gaussian_null,
    pca_compress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated H1 robustness analysis with matched null models."
    )
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["euclidean", "cosine"],
        help="Distance metrics to evaluate for POM space",
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
        "--pca-variance-threshold",
        type=float,
        default=0.90,
        help="Variance ratio to keep in the mild PCA-compressed variant",
    )
    parser.add_argument(
        "--skip-pca-variant",
        action="store_true",
        help="Skip the mild PCA-compressed robustness check",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    matrix = load_embeddings(config.embedding_path, key=config.embedding_key)
    dataset_stem = Path(config.dataset_path).stem

    variants: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "original": (
            matrix,
            {
                "variant_name": "original",
                "matrix_summary": summarize_embedding_matrix(matrix),
            },
        )
    }

    if not args.skip_pca_variant:
        reduced, pca_info = pca_compress(
            matrix=matrix,
            explained_variance_threshold=args.pca_variance_threshold,
        )
        variants["pca_compressed"] = (
            reduced,
            {
                "variant_name": "pca_compressed",
                "matrix_summary": summarize_embedding_matrix(reduced),
                "pca_compression": pca_info,
            },
        )

    report: dict[str, Any] = {
        "dataset_path": str(config.dataset_path),
        "embedding_path": str(config.embedding_path),
        "settings": {
            "metrics": args.metrics,
            "max_points": int(args.max_points),
            "runs": int(args.runs),
            "top_k": int(args.top_k),
            "pca_variance_threshold": float(args.pca_variance_threshold),
            "random_seed": int(config.random_seed),
        },
        "variants": {},
    }

    for variant_name, (variant_matrix, variant_info) in variants.items():
        variant_report = dict(variant_info)
        variant_report["metrics"] = {}
        sample_index_runs = build_sample_index_runs(
            n_rows=variant_matrix.shape[0],
            max_points=args.max_points,
            runs=args.runs,
            base_seed=config.random_seed,
        )
        for metric_name in args.metrics:
            metric_spec = {
                "name": metric_name,
                "ripser_metric": metric_name,
            }
            variant_report["metrics"][metric_name] = run_metric_analysis(
                matrix=variant_matrix,
                metric_spec=metric_spec,
                sample_index_runs=sample_index_runs,
                top_k=args.top_k,
                null_models={
                    "coordinate_permutation": coordinate_permutation_null,
                    "covariance_matched_gaussian": covariance_matched_gaussian_null,
                },
                base_seed=config.random_seed,
            )
        report["variants"][variant_name] = variant_report

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{dataset_stem}_h1_robustness_analysis.json"
    )
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
