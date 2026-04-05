from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "ripser",
        "persim",
        "gudhi",
        "rdkit",
    ]
    report = {
        "project_root": str(PROJECT_ROOT),
        "python_version": platform.python_version(),
        "packages": {name: has_module(name) for name in packages},
        "expected_paths": {
            "configs": str(PROJECT_ROOT / "configs"),
            "data_raw": str(PROJECT_ROOT / "data" / "raw"),
            "data_processed": str(PROJECT_ROOT / "data" / "processed"),
            "outputs_reports": str(PROJECT_ROOT / "outputs" / "reports"),
            "outputs_figures": str(PROJECT_ROOT / "outputs" / "figures"),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
