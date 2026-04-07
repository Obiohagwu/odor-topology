from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.config import load_project_config
from odor_topology.io import load_table
from odor_topology.representations import (
    prepare_bit_fingerprint_representation,
    prepare_count_fingerprint_representation,
    prepare_physchem_representation,
    prepare_pom_representation,
)
from odor_topology.utility import (
    build_binary_label_frame,
    build_local_feature_table,
    resolve_label_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-molecule local geometry, topology, and odor-neighborhood "
            "target features for utility analysis."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument(
        "--replication-embedding-path",
        default="",
        help="Optional path to a replication POM embedding",
    )
    parser.add_argument(
        "--physchem-path",
        default="",
        help="Optional path to an RDKit physicochemical descriptor table",
    )
    parser.add_argument(
        "--paper-bit-path",
        default="",
        help="Optional path to a paper-matched radius-4 Morgan bit fingerprint matrix",
    )
    parser.add_argument(
        "--paper-count-path",
        default="",
        help="Optional path to a paper-matched radius-4 Morgan count fingerprint matrix",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=40,
        help="Neighborhood size k for local feature construction",
    )
    parser.add_argument(
        "--include-replication",
        action="store_true",
        help="Include the replication POM checkpoint in the feature build",
    )
    parser.add_argument(
        "--include-count-fingerprint",
        action="store_true",
        help="Include the paper-matched count fingerprint representation",
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

    label_columns = resolve_label_columns(dataset, config)
    label_frame = build_binary_label_frame(dataset, label_columns)
    label_matrix = label_frame.to_numpy(dtype=float)
    row_index = dataset.index.to_numpy(dtype=int)

    physchem_path = resolve_default_processed_path(
        user_value=args.physchem_path,
        dataset_stem=dataset_stem,
        suffix="rdkit_physchem_descriptors.csv",
        legacy_name="rdkit_physchem_descriptors.csv",
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

    representations: list[dict[str, object]] = [
        prepare_pom_representation("pom_exp1_primary", config.embedding_path),
        prepare_physchem_representation(physchem_path),
        prepare_bit_fingerprint_representation(
            name="bfp_radius4_2048",
            path=paper_bit_path,
            description="paper-matched bit-based Morgan fingerprints (radius 4, 2048 bits) with Jaccard distance",
        ),
    ]

    if args.include_replication:
        replication_embedding_path = (
            args.replication_embedding_path
            or str(PROJECT_ROOT / "data" / "processed" / "openpom_embeddings_exp2.npy")
        )
        representations.append(
            prepare_pom_representation("pom_exp2_replication", replication_embedding_path)
        )

    if args.include_count_fingerprint:
        representations.append(
            prepare_count_fingerprint_representation(paper_count_path)
        )

    summary: dict[str, object] = {
        "dataset_path": str(config.dataset_path),
        "n_rows": int(len(dataset)),
        "n_label_columns": int(len(label_columns)),
        "label_columns": label_columns,
        "n_neighbors": int(args.n_neighbors),
        "feature_tables": [],
    }

    for representation in representations:
        matrix = representation["matrix"]
        representation_row_index = representation["row_index"]
        representation_name = str(representation["name"])
        for metric_spec in representation["metrics"]:
            metric_name = str(metric_spec["name"])
            feature_table = build_local_feature_table(
                matrix=matrix,
                metric_spec=metric_spec,
                label_matrix=label_matrix[representation_row_index],
                row_index=representation_row_index,
                label_columns=label_columns,
                n_neighbors=args.n_neighbors,
            )
            feature_table.insert(1, "representation", representation_name)
            feature_table.insert(2, "metric", metric_name)
            feature_table.insert(3, "dataset_stem", dataset_stem)

            output_name = f"{dataset_stem}_{representation_name}_{metric_name}_local_utility_features.csv"
            output_path = PROJECT_ROOT / "data" / "processed" / output_name
            feature_table.to_csv(output_path, index=False)
            summary["feature_tables"].append(
                {
                    "representation": representation_name,
                    "metric": metric_name,
                    "source_path": str(representation["source_path"]),
                    "output_csv": str(output_path),
                    "n_rows": int(len(feature_table)),
                }
            )

    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{dataset_stem}_utility_feature_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
