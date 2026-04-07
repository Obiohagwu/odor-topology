from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overlapping subset of a source dataset and aligned embedding "
            "by excluding SMILES present in a reference dataset."
        )
    )
    parser.add_argument("--source-dataset", required=True, help="Path to the source dataset CSV")
    parser.add_argument("--source-embedding", required=True, help="Path to the source embedding NPY")
    parser.add_argument(
        "--source-metadata",
        required=True,
        help="Path to the source embedding metadata CSV",
    )
    parser.add_argument(
        "--source-smiles-column",
        default="smiles",
        help="SMILES column name in the source dataset",
    )
    parser.add_argument(
        "--exclude-dataset",
        required=True,
        help="Path to the reference dataset whose SMILES should be excluded",
    )
    parser.add_argument(
        "--exclude-smiles-column",
        default="smiles",
        help="SMILES column name in the exclude/reference dataset",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix under the project data/processed and outputs/reports directories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dataset = load_table(args.source_dataset)
    source_embedding = np.load(args.source_embedding)
    source_metadata = pd.read_csv(args.source_metadata)
    exclude_dataset = load_table(args.exclude_dataset)

    if len(source_dataset) != source_embedding.shape[0]:
        raise ValueError(
            "Source dataset row count does not match source embedding row count: "
            f"{len(source_dataset)} vs {source_embedding.shape[0]}"
        )
    if len(source_metadata) != len(source_dataset):
        raise ValueError(
            "Source metadata row count does not match source dataset row count: "
            f"{len(source_metadata)} vs {len(source_dataset)}"
        )
    if "row_index" not in source_metadata.columns:
        raise KeyError("Expected 'row_index' column in source metadata")

    exclude_smiles = set(
        exclude_dataset[args.exclude_smiles_column].dropna().astype(str).tolist()
    )
    source_smiles = source_dataset[args.source_smiles_column].astype(str)
    keep_mask = ~source_smiles.isin(exclude_smiles)
    keep_indices = np.flatnonzero(keep_mask.to_numpy())

    subset_dataset = source_dataset.iloc[keep_indices].reset_index(drop=True)
    subset_embedding = source_embedding[keep_indices]
    subset_metadata = source_metadata.iloc[keep_indices].copy().reset_index(drop=True)
    subset_metadata["row_index"] = np.arange(len(subset_metadata), dtype=int)

    output_prefix = args.output_prefix
    dataset_out = PROJECT_ROOT / "data" / "processed" / f"{output_prefix}.csv"
    embedding_out = PROJECT_ROOT / "data" / "processed" / f"{output_prefix}.npy"
    metadata_out = PROJECT_ROOT / "data" / "processed" / f"{output_prefix}_metadata.csv"
    summary_out = PROJECT_ROOT / "outputs" / "reports" / f"{output_prefix}_summary.json"

    subset_dataset.to_csv(dataset_out, index=False)
    np.save(embedding_out, subset_embedding)
    subset_metadata.to_csv(metadata_out, index=False)

    summary = {
        "source_dataset_path": str(args.source_dataset),
        "exclude_dataset_path": str(args.exclude_dataset),
        "source_rows": int(len(source_dataset)),
        "excluded_smiles_count": int(len(exclude_smiles)),
        "subset_rows": int(len(subset_dataset)),
        "subset_embedding_shape": [int(x) for x in subset_embedding.shape],
        "outputs": {
            "dataset_csv": str(dataset_out),
            "embedding_npy": str(embedding_out),
            "metadata_csv": str(metadata_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_out}")


if __name__ == "__main__":
    main()
