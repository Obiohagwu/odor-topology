from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def infer_binary_label_columns(
    df: pd.DataFrame,
    exclude_columns: Iterable[str],
) -> list[str]:
    exclude = set(exclude_columns)
    label_columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_bool_dtype(series):
            label_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            uniques = set(series.astype(float).unique().tolist())
            if uniques.issubset({0.0, 1.0}):
                label_columns.append(column)
    return label_columns


def build_dataset_report(
    df: pd.DataFrame,
    id_column: str,
    smiles_column: str,
    label_columns: list[str],
    family_min_count: int,
    top_k_families: int,
) -> tuple[dict, pd.DataFrame]:
    report: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "id_column_present": id_column in df.columns,
        "smiles_column_present": smiles_column in df.columns,
        "n_label_columns": int(len(label_columns)),
    }

    if id_column in df.columns:
        report["n_missing_id"] = int(df[id_column].isna().sum())
        report["n_duplicate_id_rows"] = int(df[id_column].duplicated().sum())

    if smiles_column in df.columns:
        report["n_missing_smiles"] = int(df[smiles_column].isna().sum())
        report["n_duplicate_smiles_rows"] = int(df[smiles_column].duplicated().sum())

    missing_fraction = df.isna().mean().sort_values(ascending=False)
    report["columns_with_any_missing"] = int((missing_fraction > 0).sum())
    report["top_missing_columns"] = [
        {"column": column, "missing_fraction": float(value)}
        for column, value in missing_fraction.head(20).items()
        if value > 0
    ]

    if not label_columns:
        report["label_density_mean"] = None
        report["molecules_with_no_labels"] = None
        report["candidate_broad_families"] = []
        return report, pd.DataFrame(columns=["label", "count", "fraction"])

    label_frame = df[label_columns].fillna(0)
    label_frame = label_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    label_frame = (label_frame > 0).astype(int)

    counts = label_frame.sum(axis=0).sort_values(ascending=False)
    fractions = counts / max(len(df), 1)
    frequency_table = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values.astype(int),
            "fraction": fractions.values.astype(float),
        }
    )

    row_label_counts = label_frame.sum(axis=1)
    report["label_density_mean"] = float(row_label_counts.mean())
    report["label_density_median"] = float(row_label_counts.median())
    report["molecules_with_no_labels"] = int((row_label_counts == 0).sum())
    report["candidate_broad_families"] = frequency_table[
        frequency_table["count"] >= family_min_count
    ].head(top_k_families).to_dict(orient="records")

    return report, frequency_table


def summarize_embedding_matrix(matrix: np.ndarray) -> dict[str, object]:
    return {
        "n_rows": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]),
        "mean_abs_value": float(np.mean(np.abs(matrix))),
        "std_mean": float(np.std(matrix, axis=0).mean()),
    }
