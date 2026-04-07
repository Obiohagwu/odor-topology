from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.audit import summarize_embedding_matrix
from odor_topology.config import load_project_config
from odor_topology.io import load_table
from odor_topology.representations import (
    align_to_common_rows,
    common_row_index,
    metric_summary_row,
    prepare_bit_fingerprint_representation,
    prepare_count_fingerprint_representation,
    prepare_physchem_representation,
    prepare_pom_representation,
)
from odor_topology.robustness import build_sample_index_runs, run_metric_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare H1 robustness across POM and chemical baseline representations."
    )
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument(
        "--replication-embedding-path",
        default="",
        help="Path to the replication POM embedding",
    )
    parser.add_argument(
        "--physchem-path",
        default="",
        help="Path to the RDKit physicochemical descriptor table",
    )
    parser.add_argument(
        "--legacy-morgan-path",
        default="",
        help="Path to the legacy radius-2 Morgan bit fingerprint matrix",
    )
    parser.add_argument(
        "--paper-bit-path",
        default="",
        help="Path to the paper-matched radius-4 bit fingerprint matrix",
    )
    parser.add_argument(
        "--paper-count-path",
        default="",
        help="Path to the paper-matched radius-4 count fingerprint matrix",
    )
    parser.add_argument(
        "--report-prefix",
        default="",
        help="Optional prefix for report filenames; defaults to the dataset stem",
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
    return parser.parse_args()


def resolve_default_processed_path(
    user_value: str,
    dataset_stem: str,
    suffix: str,
    legacy_name: str,
) -> str:
    if user_value:
        return user_value
    dataset_specific = PROJECT_ROOT / "data" / "processed" / f"{dataset_stem}_{suffix}"
    if dataset_specific.exists():
        return str(dataset_specific)
    return str(PROJECT_ROOT / "data" / "processed" / legacy_name)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    dataset = load_table(config.dataset_path)
    dataset_stem = Path(config.dataset_path).stem
    report_prefix = args.report_prefix or dataset_stem

    replication_embedding_path = (
        args.replication_embedding_path
        or str(PROJECT_ROOT / "data" / "processed" / "openpom_embeddings_exp2.npy")
    )
    physchem_path = resolve_default_processed_path(
        user_value=args.physchem_path,
        dataset_stem=dataset_stem,
        suffix="rdkit_physchem_descriptors.csv",
        legacy_name="rdkit_physchem_descriptors.csv",
    )
    legacy_morgan_path = resolve_default_processed_path(
        user_value=args.legacy_morgan_path,
        dataset_stem=dataset_stem,
        suffix="morgan_bit_radius2_2048.npz",
        legacy_name="morgan_bit_radius2_2048.npz",
    )
    paper_bit_path = resolve_default_processed_path(
        user_value=args.paper_bit_path,
        dataset_stem=dataset_stem,
        suffix="morgan_bit_radius4_2048.npz",
        legacy_name="morgan_bit_radius4_2048.npz",
    )
    paper_count_path = resolve_default_processed_path(
        user_value=args.paper_count_path,
        dataset_stem=dataset_stem,
        suffix="morgan_count_radius4_2048.npz",
        legacy_name="morgan_count_radius4_2048.npz",
    )

    representations = [
        prepare_pom_representation("pom_exp1_primary", config.embedding_path),
        prepare_pom_representation("pom_exp2_replication", replication_embedding_path),
        prepare_physchem_representation(physchem_path),
        prepare_bit_fingerprint_representation(
            name="bfp_radius2_2048",
            path=legacy_morgan_path,
            description="legacy bit-based Morgan fingerprints (radius 2, 2048 bits) with Jaccard distance",
        ),
        prepare_bit_fingerprint_representation(
            name="bfp_radius4_2048",
            path=paper_bit_path,
            description="paper-matched bit-based Morgan fingerprints (radius 4, 2048 bits) with Jaccard distance",
        ),
        prepare_count_fingerprint_representation(paper_count_path),
    ]

    common_rows = common_row_index(representations)
    if len(common_rows) == 0:
        raise ValueError("No common rows found across representations")

    if len(common_rows) != len(dataset):
        print(
            f"Warning: only {len(common_rows)} of {len(dataset)} dataset rows are shared across representations"
        )

    sample_index_runs = build_sample_index_runs(
        n_rows=len(common_rows),
        max_points=args.max_points,
        runs=args.runs,
        base_seed=config.random_seed,
    )

    report: dict[str, Any] = {
        "dataset_path": str(config.dataset_path),
        "n_dataset_rows": int(len(dataset)),
        "n_common_rows_across_representations": int(len(common_rows)),
        "settings": {
            "max_points": int(args.max_points),
            "runs": int(args.runs),
            "top_k": int(args.top_k),
            "random_seed": int(config.random_seed),
            "sample_row_indices_are_shared_across_representations": True,
            "comparison_note": (
                "Raw persistence magnitudes are not directly comparable across different metrics. "
                "Use null-relative ratios and exceedance fractions for cross-representation comparison."
            ),
        },
        "representations": {},
    }

    summary_rows: list[dict[str, Any]] = []
    for representation in representations:
        aligned_matrix = align_to_common_rows(
            matrix=representation["matrix"],
            row_index=representation["row_index"],
            common_rows=common_rows,
        )
        representation_report: dict[str, Any] = {
            "family": representation["family"],
            "source_path": representation["source_path"],
            "preprocessing": representation["preprocessing"],
            "n_common_rows": int(aligned_matrix.shape[0]),
            "matrix_summary": summarize_embedding_matrix(np.asarray(aligned_matrix, dtype=float)),
            "metrics": {},
        }

        for metric_spec in representation["metrics"]:
            metric_name = str(metric_spec["name"])
            result = run_metric_analysis(
                matrix=aligned_matrix,
                metric_spec=metric_spec,
                sample_index_runs=sample_index_runs,
                top_k=args.top_k,
                null_models=representation["null_models"],
                base_seed=config.random_seed,
            )
            representation_report["metrics"][metric_name] = result
            summary_rows.append(
                metric_summary_row(
                    representation_name=representation["name"],
                    family=representation["family"],
                    metric_name=metric_name,
                    preprocessing=representation["preprocessing"],
                    n_rows=aligned_matrix.shape[0],
                    n_dimensions=aligned_matrix.shape[1],
                    null_model_names=list(representation["null_models"].keys()),
                    result=result,
                )
            )

        report["representations"][representation["name"]] = representation_report

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["family", "representation", "metric"]
    ).reset_index(drop=True)

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{report_prefix}_representation_robustness_comparison.json"
    )
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{report_prefix}_representation_robustness_summary.csv"
    )
    report_path.write_text(json.dumps(report, indent=2))
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nWrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
