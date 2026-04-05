from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_embeddings(path: str | Path, key: str = "") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        return _ensure_2d(array, path)
    if suffix == ".npz":
        data = np.load(path)
        if key:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}")
            return _ensure_2d(data[key], path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"No arrays found in {path}")
        return _ensure_2d(data[keys[0]], path)
    if suffix in {".csv", ".tsv", ".txt", ".parquet"}:
        table = load_table(path)
        numeric = table.select_dtypes(include=["number", "bool"])
        return _ensure_2d(numeric.to_numpy(dtype=float), path)
    raise ValueError(f"Unsupported embedding format: {path}")


def _ensure_2d(array: np.ndarray, path: Path) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array in {path}, got shape {array.shape}")
    return np.asarray(array, dtype=float)
