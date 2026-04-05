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
        description="Extract 256-dim POM embeddings from an OpenPOM checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to project config JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to a real checkpoint .pt file")
    parser.add_argument(
        "--output-npy",
        default=str(PROJECT_ROOT / "data" / "processed" / "openpom_embeddings_exp1.npy"),
        help="Where to save the embedding matrix",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "openpom_embeddings_exp1_metadata.csv"),
        help="Where to save per-row metadata aligned to the embedding matrix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Prediction batch size",
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
        model_dir="/tmp/openpom_embedding_extract",
        device_name="cpu",
    )


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    df = load_table(config.dataset_path)

    if config.smiles_column not in df.columns:
        raise KeyError(f"SMILES column '{config.smiles_column}' not found in dataset")

    smiles = df[config.smiles_column].tolist()
    featurizer = GraphFeaturizer()
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X, y=None)

    model = build_model(batch_size=args.batch_size)
    model.restore(args.checkpoint)
    embeddings = model.predict_embedding(dataset)

    output_npy = Path(args.output_npy)
    output_csv = Path(args.output_csv)
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / f"{output_npy.stem}_summary.json"
    )

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy, embeddings)

    metadata = pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=int),
            "smiles": df[config.smiles_column].astype(str),
        }
    )
    if config.id_column in df.columns:
        metadata["molecule_id"] = df[config.id_column].astype(str)
    metadata.to_csv(output_csv, index=False)

    summary = {
        "dataset_path": str(config.dataset_path),
        "checkpoint_path": str(args.checkpoint),
        "n_rows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "smiles_column": config.smiles_column,
        "id_column": config.id_column,
        "outputs": {
            "embedding_npy": str(output_npy),
            "metadata_csv": str(output_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
