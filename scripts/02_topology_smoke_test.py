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
from odor_topology.topology import pairwise_distance_summary, pca_summary, ripser_h1_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test on the embedding matrix.")
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument("--metric", default="euclidean", help="Distance metric for the smoke test")
    parser.add_argument("--max-points", type=int, default=1500, help="Max points for distance and topology work")
    parser.add_argument("--pca-components", type=int, default=10, help="Number of PCA components to summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    matrix = load_embeddings(config.embedding_path, key=config.embedding_key)

    report = {
        "embedding_path": str(config.embedding_path),
        "matrix_summary": summarize_embedding_matrix(matrix),
        "distance_summary": pairwise_distance_summary(
            matrix=matrix,
            metric=args.metric,
            max_points=args.max_points,
            seed=config.random_seed,
        ),
        "pca_summary": pca_summary(matrix=matrix, max_components=args.pca_components),
        "ripser_summary": ripser_h1_summary(
            matrix=matrix,
            metric=args.metric,
            max_points=args.max_points,
            seed=config.random_seed,
        ),
    }

    report_path = PROJECT_ROOT / "outputs" / "reports" / "topology_smoke_test.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
