from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.config import load_project_config
from odor_topology.utility import evaluate_regression_models, summarize_utility_result_row


GEOMETRY_COLUMNS = [
    "mean_neighbor_distance",
    "std_neighbor_distance",
    "p90_neighbor_distance",
    "max_neighbor_distance",
    "local_patch_distance_mean",
    "local_patch_distance_std",
    "local_density_proxy",
    "local_participation_ratio",
    "local_pca_reconstruction_error_ratio",
]

TOPOLOGY_COLUMNS = [
    "local_h1_feature_count",
    "local_h1_max_persistence",
    "local_h1_total_persistence",
    "local_h1_mean_persistence",
    "local_h1_p95_persistence",
]

DEFAULT_TARGETS = [
    "target_mean_neighbor_label_jaccard",
    "target_share_any_label_fraction",
    "target_neighbor_label_entropy_active",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether local topology features add explanatory value beyond "
            "local geometry for odor-neighborhood targets."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument(
        "--feature-summary",
        default="",
        help="Optional path to the utility feature summary JSON",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target columns to evaluate",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds per repeat",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeated CV rounds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    dataset_stem = Path(config.dataset_path).stem
    feature_summary_path = (
        Path(args.feature_summary)
        if args.feature_summary
        else PROJECT_ROOT / "outputs" / "reports" / f"{dataset_stem}_utility_feature_summary.json"
    )

    feature_summary = json.loads(feature_summary_path.read_text())
    report: dict[str, object] = {
        "dataset_path": str(config.dataset_path),
        "feature_summary_path": str(feature_summary_path),
        "targets": args.targets,
        "settings": {
            "n_splits": int(args.n_splits),
            "n_repeats": int(args.n_repeats),
            "random_seed": int(config.random_seed),
            "geometry_columns": GEOMETRY_COLUMNS,
            "topology_columns": TOPOLOGY_COLUMNS,
        },
        "representations": {},
    }

    summary_rows: list[dict[str, object]] = []
    for entry in feature_summary["feature_tables"]:
        feature_table = pd.read_csv(entry["output_csv"])
        representation_key = f"{entry['representation']}__{entry['metric']}"
        representation_report: dict[str, object] = {
            "representation": entry["representation"],
            "metric": entry["metric"],
            "feature_csv": entry["output_csv"],
            "targets": {},
        }

        for target_column in args.targets:
            result = evaluate_regression_models(
                feature_table=feature_table,
                geometry_columns=GEOMETRY_COLUMNS,
                topology_columns=TOPOLOGY_COLUMNS,
                target_column=target_column,
                random_seed=config.random_seed,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
            )
            representation_report["targets"][target_column] = result
            summary_rows.append(
                summarize_utility_result_row(
                    representation_name=str(entry["representation"]),
                    metric_name=str(entry["metric"]),
                    target_column=target_column,
                    result=result,
                )
            )

        report["representations"][representation_key] = representation_report

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["representation", "metric", "target"]
    ).reset_index(drop=True)

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{dataset_stem}_utility_analysis.json"
    )
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{dataset_stem}_utility_analysis_summary.csv"
    )
    report_path.write_text(json.dumps(report, indent=2))
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nWrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
