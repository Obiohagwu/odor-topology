from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from odor_topology.config import load_project_config
from odor_topology.io import load_table
from openpom.feat.graph_featurizer import GraphConvConstants, GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings for all available OpenPOM ensemble checkpoints."
    )
    parser.add_argument("--config", required=True, help="Path to project config JSON")
    parser.add_argument(
        "--checkpoints-dir",
        default=str(PROJECT_ROOT.parent / "openpom" / "models" / "ensemble_models"),
        help="Directory containing experiments_*/checkpoint2_real.pt files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory to write embedding matrices and metadata",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Prediction batch size",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs",
    )
    return parser.parse_args()


def build_model(batch_size: int) -> MPNNPOMModel:
    return MPNNPOMModel(
        n_tasks=138,
        batch_size=batch_size,
        class_imbalance_ratio=None,
        loss_aggr_type="sum",
        node_out_feats=100,
        edge_hidden_feats=75,
        edge_out_feats=100,
        num_step_message_passing=5,
        mpnn_residual=True,
        message_aggregator_type="sum",
        mode="classification",
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1,
        readout_type="set2set",
        num_step_set2set=3,
        num_layer_set2set=2,
        ffn_hidden_list=[392, 392],
        ffn_embeddings=256,
        ffn_activation="relu",
        ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False,
        weight_decay=1e-5,
        self_loop=False,
        optimizer_name="adam",
        log_frequency=32,
        model_dir="/tmp/openpom_embedding_extract_batch",
        device_name="cpu",
    )


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    df = load_table(config.dataset_path)

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.smiles_column not in df.columns:
        raise KeyError(f"SMILES column '{config.smiles_column}' not found in dataset")

    smiles = df[config.smiles_column].tolist()
    featurizer = GraphFeaturizer()
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X, y=None)

    metadata = pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=int),
            "smiles": df[config.smiles_column].astype(str),
        }
    )
    if config.id_column in df.columns:
        metadata["molecule_id"] = df[config.id_column].astype(str)

    checkpoint_paths = sorted(checkpoints_dir.glob("experiments_*/checkpoint2_real.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint2_real.pt files found under {checkpoints_dir}")

    summary_rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        experiment_name = checkpoint_path.parent.name
        exp_number = experiment_name.split("_")[-1]
        output_npy = output_dir / f"openpom_embeddings_exp{exp_number}.npy"
        output_csv = output_dir / f"openpom_embeddings_exp{exp_number}_metadata.csv"

        if output_npy.exists() and output_csv.exists() and not args.force:
            embeddings = np.load(output_npy)
            summary_rows.append(
                {
                    "experiment": experiment_name,
                    "checkpoint_path": str(checkpoint_path),
                    "embedding_npy": str(output_npy),
                    "metadata_csv": str(output_csv),
                    "embedding_shape": list(embeddings.shape),
                    "status": "reused_existing",
                }
            )
            continue

        model = build_model(batch_size=args.batch_size)
        model.restore(str(checkpoint_path))
        embeddings = model.predict_embedding(dataset)
        np.save(output_npy, embeddings)
        metadata.to_csv(output_csv, index=False)

        summary_rows.append(
            {
                "experiment": experiment_name,
                "checkpoint_path": str(checkpoint_path),
                "embedding_npy": str(output_npy),
                "metadata_csv": str(output_csv),
                "embedding_shape": list(embeddings.shape),
                "status": "extracted",
            }
        )

    summary = {
        "dataset_path": str(config.dataset_path),
        "n_rows": int(len(df)),
        "n_checkpoints_found": int(len(checkpoint_paths)),
        "outputs": summary_rows,
    }
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "openpom_ensemble_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
