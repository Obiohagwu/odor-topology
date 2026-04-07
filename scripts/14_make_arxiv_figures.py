from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "arxiv" / "figures"

REPRESENTATION_LABELS = {
    ("pom_exp1_primary", "euclidean"): "POM exp1\nEuclidean",
    ("pom_exp1_primary", "cosine"): "POM exp1\nCosine",
    ("pom_exp2_replication", "euclidean"): "POM exp2\nEuclidean",
    ("pom_exp2_replication", "cosine"): "POM exp2\nCosine",
    ("bfp_radius4_2048", "jaccard"): "Morgan bit r4\nJaccard",
    ("bfp_radius2_2048", "jaccard"): "Morgan bit r2\nJaccard",
    ("cfp_radius4_2048", "generalized_tanimoto"): "Morgan count r4\nGen. Tanimoto",
    ("rdkit_physchem", "euclidean"): "RDKit physchem\nEuclidean",
}

TARGET_LABELS = {
    "target_mean_neighbor_label_jaccard": "Mean neighbor\nlabel Jaccard",
    "target_neighbor_label_entropy_active": "Neighbor label\nentropy (active)",
    "target_share_any_label_fraction": "Share any\nlabel fraction",
}

REPRESENTATION_ORDER = [
    "POM exp1\nCosine",
    "POM exp1\nEuclidean",
    "POM exp2\nCosine",
    "POM exp2\nEuclidean",
    "Morgan bit r4\nJaccard",
    "Morgan bit r2\nJaccard",
    "Morgan count r4\nGen. Tanimoto",
    "RDKit physchem\nEuclidean",
]

UTILITY_REPRESENTATION_ORDER = [
    "POM exp1\nCosine",
    "POM exp1\nEuclidean",
    "RDKit physchem\nEuclidean",
    "Morgan bit r4\nJaccard",
    "Morgan count r4\nGen. Tanimoto",
]

TARGET_ORDER = [
    "Neighbor label\nentropy (active)",
    "Share any\nlabel fraction",
    "Mean neighbor\nlabel Jaccard",
]

COLORS = {
    "POM exp1\nCosine": "#74a9cf",
    "POM exp1\nEuclidean": "#045a8d",
    "POM exp2\nCosine": "#a6bddb",
    "POM exp2\nEuclidean": "#2b8cbe",
    "Morgan bit r4\nJaccard": "#d95f02",
    "Morgan bit r2\nJaccard": "#fdb863",
    "Morgan count r4\nGen. Tanimoto": "#7b3294",
    "RDKit physchem\nEuclidean": "#636363",
}


def short_label(row: pd.Series) -> str:
    key = (str(row["representation"]), str(row["metric"]))
    return REPRESENTATION_LABELS.get(key, str(row["representation"]))


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )


def plot_threshold_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    ratio_column: str = "top1_signal_to_strongest_null_p95",
) -> None:
    working = df.copy()
    working["label"] = working.apply(short_label, axis=1)
    working = working.sort_values(ratio_column, ascending=True).reset_index(drop=True)

    y_positions = np.arange(len(working))
    values = working[ratio_column].to_numpy(dtype=float)
    labels = working["label"].tolist()
    colors = [COLORS.get(label, "#888888") for label in labels]

    ax.barh(y_positions, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Observed / strongest-null 95th percentile")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    xmax = max(1.1, float(np.max(values)) * 1.18)
    ax.set_xlim(0, xmax)
    for y_pos, value in zip(y_positions, values, strict=True):
        ax.text(
            value + (0.02 * xmax),
            y_pos,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )


def make_direct_robustness_figure() -> None:
    curated = pd.read_csv(REPORT_DIR / "representation_robustness_summary.csv")
    nonoverlap = pd.read_csv(
        REPORT_DIR / "gslf_5862_excluding_curated_4983_representation_robustness_summary.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=False)
    plot_threshold_panel(
        axes[0],
        curated,
        title="Curated 4,983-row GS/LF table\nDirect repeated-subsample H1",
    )
    plot_threshold_panel(
        axes[1],
        nonoverlap,
        title="1,600-molecule non-overlap subset\nDirect repeated-subsample H1",
    )
    fig.suptitle("Topological signal is robust in POM, but not unique to POM", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure1_direct_robustness")
    plt.close(fig)


def make_landmark_robustness_figure() -> None:
    curated = pd.read_csv(REPORT_DIR / "representation_robustness_summary_landmark.csv")
    nonoverlap = pd.read_csv(
        REPORT_DIR / "gslf_5862_excluding_curated_4983_representation_robustness_summary_landmark.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharex=False)
    plot_threshold_panel(
        axes[0],
        curated,
        title="Curated 4,983-row GS/LF table\nGreedy-landmark distance-matrix H1",
    )
    plot_threshold_panel(
        axes[1],
        nonoverlap,
        title="1,600-molecule non-overlap subset\nGreedy-landmark distance-matrix H1",
    )
    fig.suptitle("Landmark analyses preserve the main cautionary result", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure2_landmark_robustness")
    plt.close(fig)


def make_ensemble_stability_figure() -> None:
    summary = pd.read_csv(REPORT_DIR / "openpom_ensemble_checkpoint_summary.csv")
    audit = json.loads((REPORT_DIR / "openpom_ensemble_checkpoint_audit.json").read_text())

    summary["exp_number"] = summary["embedding"].str.extract(r"exp(\d+)").astype(int)
    summary = summary.sort_values(["metric", "exp_number"]).reset_index(drop=True)

    embeddings = sorted(
        {row["embedding_a"] for row in audit["pairwise_distance_agreement"]}
        | {row["embedding_b"] for row in audit["pairwise_distance_agreement"]},
        key=lambda name: int(str(name).split("exp")[-1]),
    )
    index_lookup = {name: idx for idx, name in enumerate(embeddings)}
    agreement_matrix = np.ones((len(embeddings), len(embeddings)), dtype=float)
    for row in audit["pairwise_distance_agreement"]:
        i = index_lookup[row["embedding_a"]]
        j = index_lookup[row["embedding_b"]]
        agreement_matrix[i, j] = float(row["pairwise_distance_pearson"])
        agreement_matrix[j, i] = float(row["pairwise_distance_pearson"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)

    metric_colors = {"cosine": "#2b8cbe", "euclidean": "#045a8d"}
    for metric_name in ("cosine", "euclidean"):
        metric_df = summary[summary["metric"] == metric_name].sort_values("exp_number")
        axes[0].plot(
            metric_df["exp_number"],
            metric_df["top1_signal_to_strongest_null_p95"],
            marker="o",
            linewidth=2.0,
            label=metric_name.capitalize(),
            color=metric_colors[metric_name],
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(summary["exp_number"].sort_values().unique())
    axes[0].set_xlabel("OpenPOM ensemble checkpoint")
    axes[0].set_ylabel("Observed / strongest-null 95th percentile")
    axes[0].set_title("Direct H1 signal remains above null across all 10 checkpoints")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    heatmap = axes[1].imshow(
        agreement_matrix,
        cmap="viridis",
        vmin=float(np.min(agreement_matrix)),
        vmax=float(np.max(agreement_matrix)),
        aspect="auto",
    )
    short_ticks = [name.replace("openpom_embeddings_", "") for name in embeddings]
    axes[1].set_xticks(np.arange(len(embeddings)))
    axes[1].set_xticklabels(short_ticks, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(embeddings)))
    axes[1].set_yticklabels(short_ticks)
    axes[1].set_title("Pairwise Pearson agreement of sampled distance matrices")
    colorbar = fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")

    fig.suptitle("Checkpoint variation changes scale, not the main qualitative conclusion", fontsize=14)
    save_figure(fig, "figure3_ensemble_stability")
    plt.close(fig)


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
) -> plt.AxesImage:
    image = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=12, pad=8)

    midpoint = (vmin + vmax) / 2.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            text_color = "white" if value >= midpoint else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    return image


def utility_heatmap_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df.apply(short_label, axis=1)
    df["target_label"] = df["target"].map(TARGET_LABELS)
    matrix = df.pivot(index="target_label", columns="label", values="mean_delta_r2")
    matrix = matrix.reindex(index=TARGET_ORDER, columns=UTILITY_REPRESENTATION_ORDER)
    return matrix


def make_utility_figure() -> None:
    curated = utility_heatmap_matrix(REPORT_DIR / "curated_GS_LF_merged_4983_utility_analysis_summary.csv")
    nonoverlap = utility_heatmap_matrix(
        REPORT_DIR / "gslf_5862_excluding_curated_4983_utility_analysis_summary.csv"
    )

    vmax = float(max(np.nanmax(curated.to_numpy()), np.nanmax(nonoverlap.to_numpy())))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    image = draw_heatmap(
        axes[0],
        curated.to_numpy(dtype=float),
        row_labels=curated.index.tolist(),
        col_labels=curated.columns.tolist(),
        title="Curated 4,983-row GS/LF table\nMean delta R^2 from adding topology",
        vmin=0.0,
        vmax=vmax,
    )
    draw_heatmap(
        axes[1],
        nonoverlap.to_numpy(dtype=float),
        row_labels=nonoverlap.index.tolist(),
        col_labels=nonoverlap.columns.tolist(),
        title="1,600-molecule non-overlap subset\nMean delta R^2 from adding topology",
        vmin=0.0,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label("Geometry + topology minus geometry-only mean R^2")
    fig.suptitle("Topology features help in some tasks, but gains are modest and representation-dependent", fontsize=14)
    save_figure(fig, "figure4_utility_deltas")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    make_direct_robustness_figure()
    make_landmark_robustness_figure()
    make_ensemble_stability_figure()
    make_utility_figure()
    print(f"Wrote arXiv figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
