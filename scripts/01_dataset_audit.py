from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.audit import build_dataset_report, infer_binary_label_columns
from odor_topology.config import load_project_config
from odor_topology.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the odor dataset table.")
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    df = load_table(config.dataset_path)

    label_columns = config.label_columns
    if not label_columns:
        exclude_columns = [config.id_column, config.smiles_column, *config.ignore_columns]
        label_columns = infer_binary_label_columns(df, exclude_columns=exclude_columns)

    report, frequency_table = build_dataset_report(
        df=df,
        id_column=config.id_column,
        smiles_column=config.smiles_column,
        label_columns=label_columns,
        family_min_count=config.family_min_count,
        top_k_families=config.top_k_families,
    )
    report["dataset_path"] = str(config.dataset_path)
    report["label_columns_inferred"] = not bool(config.label_columns)

    report_path = PROJECT_ROOT / "outputs" / "reports" / "dataset_audit.json"
    freq_path = PROJECT_ROOT / "outputs" / "reports" / "label_frequencies.csv"

    report_path.write_text(json.dumps(report, indent=2))
    frequency_table.to_csv(freq_path, index=False)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    print(f"Wrote {freq_path}")


if __name__ == "__main__":
    main()
