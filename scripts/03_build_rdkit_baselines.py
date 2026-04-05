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
    build_morgan_fingerprints,
    build_physchem_descriptors,
    smiles_validity_table,
)
from odor_topology.config import load_project_config
from odor_topology.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RDKit baseline features.")
    parser.add_argument("--config", required=True, help="Path to the project config JSON")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint bit count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    df = load_table(config.dataset_path)

    if config.smiles_column not in df.columns:
        raise KeyError(f"SMILES column '{config.smiles_column}' not found in dataset")

    smiles_series = df[config.smiles_column]
    validity = smiles_validity_table(smiles_series)
    physchem_df, invalid_physchem = build_physchem_descriptors(smiles_series)
    fingerprints, fingerprint_metadata = build_morgan_fingerprints(
        smiles_series=smiles_series,
        radius=args.radius,
        n_bits=args.n_bits,
    )

    physchem_path = PROJECT_ROOT / "data" / "processed" / "rdkit_physchem_descriptors.csv"
    fp_path = PROJECT_ROOT / "data" / "processed" / "morgan_fingerprints.npz"
    validity_path = PROJECT_ROOT / "outputs" / "reports" / "rdkit_smiles_validity.csv"
    invalid_path = PROJECT_ROOT / "outputs" / "reports" / "rdkit_invalid_smiles.csv"
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "rdkit_baseline_summary.json"

    physchem_df.to_csv(physchem_path, index=False)
    validity.to_csv(validity_path, index=False)
    invalid_physchem.to_csv(invalid_path, index=False)
    np.savez_compressed(
        fp_path,
        fingerprints=fingerprints,
        row_index=fingerprint_metadata.loc[
            fingerprint_metadata["is_valid"], "row_index"
        ].to_numpy(),
    )

    summary = {
        "dataset_path": str(config.dataset_path),
        "smiles_column": config.smiles_column,
        "n_rows": int(len(df)),
        "n_valid_smiles": int(validity["is_valid"].sum()),
        "n_invalid_smiles": int((~validity["is_valid"]).sum()),
        "n_physchem_rows": int(len(physchem_df)),
        "fingerprint_shape": list(fingerprints.shape),
        "radius": args.radius,
        "n_bits": args.n_bits,
        "outputs": {
            "physchem_csv": str(physchem_path),
            "fingerprints_npz": str(fp_path),
            "validity_csv": str(validity_path),
            "invalid_smiles_csv": str(invalid_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
