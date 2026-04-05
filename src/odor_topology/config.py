from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    dataset_path: Path
    embedding_path: Path
    embedding_key: str = ""
    id_column: str = "molecule_id"
    smiles_column: str = "smiles"
    label_columns: list[str] = field(default_factory=list)
    ignore_columns: list[str] = field(default_factory=list)
    family_min_count: int = 50
    top_k_families: int = 20
    random_seed: int = 0


def load_project_config(path: str | Path) -> ProjectConfig:
    raw = json.loads(Path(path).read_text())
    return ProjectConfig(
        dataset_path=Path(raw["dataset_path"]),
        embedding_path=Path(raw["embedding_path"]),
        embedding_key=raw.get("embedding_key", ""),
        id_column=raw.get("id_column", "molecule_id"),
        smiles_column=raw.get("smiles_column", "smiles"),
        label_columns=list(raw.get("label_columns", [])),
        ignore_columns=list(raw.get("ignore_columns", [])),
        family_min_count=int(raw.get("family_min_count", 50)),
        top_k_families=int(raw.get("top_k_families", 20)),
        random_seed=int(raw.get("random_seed", 0)),
    )
