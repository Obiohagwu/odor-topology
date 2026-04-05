from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RDKit is required for chemistry baselines. Use the project venv Python."
    ) from exc


PHYS_CHEM_DESCRIPTOR_FNS = {
    "mol_wt": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_donors": Lipinski.NumHDonors,
    "h_acceptors": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "ring_count": Lipinski.RingCount,
    "aromatic_rings": Lipinski.NumAromaticRings,
    "heavy_atom_count": Lipinski.HeavyAtomCount,
    "fraction_csp3": Lipinski.FractionCSP3,
}


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def smiles_validity_table(smiles_series: pd.Series) -> pd.DataFrame:
    rows = []
    for idx, smiles in smiles_series.items():
        mol = mol_from_smiles(smiles)
        rows.append(
            {
                "row_index": idx,
                "smiles": smiles,
                "is_valid": mol is not None,
            }
        )
    return pd.DataFrame(rows)


def build_physchem_descriptors(smiles_series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    invalid_rows = []
    for idx, smiles in smiles_series.items():
        mol = mol_from_smiles(smiles)
        if mol is None:
            invalid_rows.append({"row_index": idx, "smiles": smiles})
            continue
        row = {"row_index": idx, "smiles": smiles}
        for name, fn in PHYS_CHEM_DESCRIPTOR_FNS.items():
            row[name] = float(fn(mol))
        valid_rows.append(row)
    return pd.DataFrame(valid_rows), pd.DataFrame(invalid_rows)


def build_morgan_fingerprints(
    smiles_series: pd.Series,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, pd.DataFrame]:
    fingerprint_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for idx, smiles in smiles_series.items():
        mol = mol_from_smiles(smiles)
        if mol is None:
            metadata_rows.append(
                {"row_index": idx, "smiles": smiles, "is_valid": False}
            )
            continue
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        array = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvect, array)
        fingerprint_rows.append(array)
        metadata_rows.append({"row_index": idx, "smiles": smiles, "is_valid": True})
    if fingerprint_rows:
        matrix = np.stack(fingerprint_rows, axis=0)
    else:
        matrix = np.empty((0, n_bits), dtype=np.uint8)
    return matrix, pd.DataFrame(metadata_rows)
