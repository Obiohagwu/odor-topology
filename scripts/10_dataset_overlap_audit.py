from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit overlap between two molecule tables by SMILES."
    )
    parser.add_argument("--dataset-a", required=True, help="Path to the first dataset CSV")
    parser.add_argument("--dataset-b", required=True, help="Path to the second dataset CSV")
    parser.add_argument(
        "--smiles-column-a",
        default="smiles",
        help="SMILES column name for dataset A",
    )
    parser.add_argument(
        "--smiles-column-b",
        default="smiles",
        help="SMILES column name for dataset B",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_a = load_table(args.dataset_a)
    dataset_b = load_table(args.dataset_b)

    smiles_a = set(dataset_a[args.smiles_column_a].dropna().astype(str).tolist())
    smiles_b = set(dataset_b[args.smiles_column_b].dropna().astype(str).tolist())

    intersection = smiles_a & smiles_b
    union = smiles_a | smiles_b
    only_a = smiles_a - smiles_b
    only_b = smiles_b - smiles_a

    dataset_a_name = Path(args.dataset_a).stem
    dataset_b_name = Path(args.dataset_b).stem

    report = {
        "dataset_a_path": str(args.dataset_a),
        "dataset_b_path": str(args.dataset_b),
        "dataset_a_unique_smiles": int(len(smiles_a)),
        "dataset_b_unique_smiles": int(len(smiles_b)),
        "intersection_unique_smiles": int(len(intersection)),
        "union_unique_smiles": int(len(union)),
        "dataset_a_only_unique_smiles": int(len(only_a)),
        "dataset_b_only_unique_smiles": int(len(only_b)),
        "intersection_over_dataset_a": float(len(intersection) / len(smiles_a)) if smiles_a else 0.0,
        "intersection_over_dataset_b": float(len(intersection) / len(smiles_b)) if smiles_b else 0.0,
        "jaccard_similarity": float(len(intersection) / len(union)) if union else 0.0,
    }

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{dataset_a_name}__{dataset_b_name}_overlap_audit.json"
    )
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
