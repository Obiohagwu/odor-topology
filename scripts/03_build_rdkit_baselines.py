from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.chemistry import (
    build_morgan_bit_fingerprints,
    build_morgan_count_fingerprints,
    build_physchem_descriptors,
    smiles_validity_table,
)
from odor_topology.config import load_project_config
from odor_topology.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RDKit baseline features.")
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument("--legacy-radius", type=int, default=2, help="Legacy Morgan bit radius")
    parser.add_argument("--paper-radius", type=int, default=4, help="Paper-matched fingerprint radius")
    parser.add_argument("--n-bits", type=int, default=2048, help="Fingerprint bit count")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for output filenames; defaults to the dataset stem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    df = load_table(config.dataset_path)
    output_prefix = args.output_prefix or Path(config.dataset_path).stem

    if config.smiles_column not in df.columns:
        raise KeyError(f"SMILES column '{config.smiles_column}' not found in dataset")

    smiles_series = df[config.smiles_column]
    validity = smiles_validity_table(smiles_series)
    physchem_df, invalid_physchem = build_physchem_descriptors(smiles_series)
    legacy_bit_fingerprints, legacy_bit_metadata = build_morgan_bit_fingerprints(
        smiles_series=smiles_series,
        radius=args.legacy_radius,
        n_bits=args.n_bits,
    )
    paper_bit_fingerprints, paper_bit_metadata = build_morgan_bit_fingerprints(
        smiles_series=smiles_series,
        radius=args.paper_radius,
        n_bits=args.n_bits,
    )
    paper_count_fingerprints, paper_count_metadata = build_morgan_count_fingerprints(
        smiles_series=smiles_series,
        radius=args.paper_radius,
        n_bits=args.n_bits,
    )

    physchem_path = (
        PROJECT_ROOT / "data" / "processed" / f"{output_prefix}_rdkit_physchem_descriptors.csv"
    )
    legacy_fp_path = (
        PROJECT_ROOT / "data" / "processed" / f"{output_prefix}_morgan_fingerprints.npz"
    )
    legacy_fp_named_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"{output_prefix}_morgan_bit_radius{args.legacy_radius}_{args.n_bits}.npz"
    )
    paper_bit_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"{output_prefix}_morgan_bit_radius{args.paper_radius}_{args.n_bits}.npz"
    )
    paper_count_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"{output_prefix}_morgan_count_radius{args.paper_radius}_{args.n_bits}.npz"
    )
    validity_path = (
        PROJECT_ROOT / "outputs" / "reports" / f"{output_prefix}_rdkit_smiles_validity.csv"
    )
    invalid_path = (
        PROJECT_ROOT / "outputs" / "reports" / f"{output_prefix}_rdkit_invalid_smiles.csv"
    )
    summary_path = (
        PROJECT_ROOT / "outputs" / "reports" / f"{output_prefix}_rdkit_baseline_summary.json"
    )

    physchem_df.to_csv(physchem_path, index=False)
    validity.to_csv(validity_path, index=False)
    invalid_physchem.to_csv(invalid_path, index=False)
    np.savez_compressed(
        legacy_fp_path,
        fingerprints=legacy_bit_fingerprints,
        row_index=legacy_bit_metadata.loc[
            legacy_bit_metadata["is_valid"], "row_index"
        ].to_numpy(),
    )
    np.savez_compressed(
        legacy_fp_named_path,
        fingerprints=legacy_bit_fingerprints,
        row_index=legacy_bit_metadata.loc[
            legacy_bit_metadata["is_valid"], "row_index"
        ].to_numpy(),
    )
    np.savez_compressed(
        paper_bit_path,
        fingerprints=paper_bit_fingerprints,
        row_index=paper_bit_metadata.loc[
            paper_bit_metadata["is_valid"], "row_index"
        ].to_numpy(),
    )
    np.savez_compressed(
        paper_count_path,
        fingerprints=paper_count_fingerprints,
        row_index=paper_count_metadata.loc[
            paper_count_metadata["is_valid"], "row_index"
        ].to_numpy(),
    )

    summary = {
        "dataset_path": str(config.dataset_path),
        "output_prefix": output_prefix,
        "smiles_column": config.smiles_column,
        "n_rows": int(len(df)),
        "n_valid_smiles": int(validity["is_valid"].sum()),
        "n_invalid_smiles": int((~validity["is_valid"]).sum()),
        "n_physchem_rows": int(len(physchem_df)),
        "legacy_bit_fingerprint_shape": list(legacy_bit_fingerprints.shape),
        "paper_bit_fingerprint_shape": list(paper_bit_fingerprints.shape),
        "paper_count_fingerprint_shape": list(paper_count_fingerprints.shape),
        "legacy_radius": args.legacy_radius,
        "paper_radius": args.paper_radius,
        "n_bits": args.n_bits,
        "outputs": {
            "physchem_csv": str(physchem_path),
            "legacy_fingerprints_npz": str(legacy_fp_path),
            "legacy_named_fingerprints_npz": str(legacy_fp_named_path),
            "paper_bit_fingerprints_npz": str(paper_bit_path),
            "paper_count_fingerprints_npz": str(paper_count_path),
            "validity_csv": str(validity_path),
            "invalid_smiles_csv": str(invalid_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
