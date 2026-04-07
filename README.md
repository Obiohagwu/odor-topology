# Odor Topology Robustness

This project tests whether apparent topological structure in learned olfactory representation space is reproducible, representation-specific, and scientifically useful.

The immediate goal is not to "discover the topology of odor space." The immediate goal is to build a clean pipeline that can answer a narrower question well:

> Does any topological signal in a fixed Principal Odor Map embedding survive metric choice, subsampling, null-model comparison, and baseline-representation controls?

## Current Focus

- stress-test POM H1 under multiple datasets, nulls, and checkpoints
- compare learned odor space against matched chemical baselines
- document exactly where topology appears robust and where it does not
- avoid overclaiming representation-specific structure before the evidence supports it

## Current Evidence

- direct repeated-subsample H1 is robust in the primary OpenPOM embedding on the curated 4,983-row table, the broader 5,862-row GS/LF table, and the 1,600-molecule non-overlap subset
- the same Euclidean H1 story is broadly stable across all 10 released OpenPOM ensemble checkpoints
- paper-matched Morgan bit fingerprints also show strong H1 under matched direct comparisons, so robust topology is not currently unique to POM
- paper-matched count fingerprints and simple physicochemical descriptors are weaker on the strict top-1 null criterion
- on the 1,600-molecule non-overlap subset, direct and landmark comparisons still show robust POM H1, but paper-matched Morgan bit fingerprints remain at least as strong or stronger
- a second greedy-landmark route supports Euclidean POM H1 more clearly than cosine POM H1 on the curated table, while the non-overlap subset gives a somewhat healthier cosine result; the strongest current claim is still metric-sensitive rather than universal
- a first utility analysis shows that local topology can add explanatory value beyond local geometry, but the gains are target-dependent and not obviously unique to POM

Key reports:

- `/Users/oboh/bio-research/odor-topology/outputs/reports/curated_GS_LF_merged_4983_h1_robustness_analysis.json`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/chemprop_gs_lf_filtered_h1_robustness_analysis.json`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/gslf_5862_excluding_curated_4983_h1_robustness_analysis.json`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/gslf_5862_excluding_curated_4983_representation_robustness_summary.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/gslf_5862_excluding_curated_4983_representation_robustness_summary_landmark.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/representation_robustness_summary.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/representation_robustness_summary_landmark.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/openpom_ensemble_checkpoint_summary.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/curated_GS_LF_merged_4983__chemprop_gs_lf_filtered_overlap_audit.json`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/curated_GS_LF_merged_4983_utility_analysis_summary.csv`
- `/Users/oboh/bio-research/odor-topology/outputs/reports/gslf_5862_excluding_curated_4983_utility_analysis_summary.csv`

## arXiv Draft

- manuscript source: `/Users/oboh/bio-research/odor-topology/arxiv/main.tex`
- figure generation script: `/Users/oboh/bio-research/odor-topology/scripts/14_make_arxiv_figures.py`
- generated figure bundle: `/Users/oboh/bio-research/odor-topology/arxiv/figures/`
- applicability and positioning note: `/Users/oboh/bio-research/odor-topology/APPLICABILITY_AND_POSITIONING_NOTE.md`

The current arXiv package is source-ready but was not compiled locally because this machine does not currently have a TeX toolchain installed.
It now includes inline references, populated author metadata, and a clean local upload bundle at `/Users/oboh/bio-research/odor-topology/arxiv.zip`.

## Project Layout

- `configs/`: local configuration files
- `data/raw/`: raw tables and embedding files
- `data/processed/`: cleaned outputs
- `outputs/reports/`: JSON and CSV reports
- `outputs/figures/`: plots
- `arxiv/`: manuscript draft and submission-oriented figures
- `scripts/`: runnable entrypoints
- `src/odor_topology/`: reusable code

## Quick Start

1. Copy the example config:

```bash
cp /Users/oboh/bio-research/odor-topology/configs/project_config.example.json \
   /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

2. Edit `project_config.json` so `dataset_path` and `embedding_path` point at the real files.

3. Check the local environment:

```bash
python3 /Users/oboh/bio-research/odor-topology/scripts/00_environment_check.py
```

4. Run the dataset audit:

```bash
python3 /Users/oboh/bio-research/odor-topology/scripts/01_dataset_audit.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

5. Run the embedding smoke test:

```bash
python3 /Users/oboh/bio-research/odor-topology/scripts/02_topology_smoke_test.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

6. Build RDKit baselines from the molecule table:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/03_build_rdkit_baselines.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

7. Extract POM embeddings from an official OpenPOM checkpoint:

```bash
/Users/oboh/bio-research/odor-topology/.venv311/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/04_extract_openpom_embeddings.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json \
  --checkpoint /Users/oboh/bio-research/openpom/models/ensemble_models/experiments_1/checkpoint2_real.pt
```

8. Run the first repeated H1 robustness analysis:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/05_h1_robustness_analysis.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json \
  --max-points 1000 \
  --runs 8 \
  --top-k 10
```

9. Run the matched representation comparison across POM and chemical baselines:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/06_representation_robustness_comparison.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json \
  --max-points 1000 \
  --runs 8 \
  --top-k 10
```

10. Extract all released OpenPOM ensemble embeddings:

```bash
/Users/oboh/bio-research/odor-topology/.venv311/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/07_extract_openpom_ensemble_embeddings.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

11. Audit robustness across the full OpenPOM ensemble:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/08_openpom_ensemble_checkpoint_audit.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

12. Run the second, greedy-landmark comparison route:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/09_landmark_robustness_comparison.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json \
  --max-points 1000 \
  --landmark-count 250 \
  --runs 8 \
  --top-k 10
```

13. Audit overlap between related GS/LF tables:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/10_dataset_overlap_audit.py \
  --dataset-a /Users/oboh/bio-research/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv \
  --smiles-column-a nonStereoSMILES \
  --dataset-b /Users/oboh/bio-research/olfaction/data/datasets/chemprop_gs_lf_filtered.csv \
  --smiles-column-b smiles
```

14. Create the non-overlap GS/LF subset and aligned embeddings:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/11_make_nonoverlap_subset.py \
  --source-dataset /Users/oboh/bio-research/olfaction/data/datasets/chemprop_gs_lf_filtered.csv \
  --source-embedding /Users/oboh/bio-research/odor-topology/data/processed/openpom_embeddings_exp1_gslf5862.npy \
  --source-metadata /Users/oboh/bio-research/odor-topology/data/processed/openpom_embeddings_exp1_gslf5862_metadata.csv \
  --source-smiles-column smiles \
  --exclude-dataset /Users/oboh/bio-research/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv \
  --exclude-smiles-column nonStereoSMILES \
  --output-prefix gslf_5862_excluding_curated_4983
```

15. Build local geometry/topology feature tables for utility analysis:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/12_build_local_utility_features.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json \
  --include-count-fingerprint \
  --n-neighbors 40
```

16. Evaluate geometry-only versus geometry-plus-topology utility:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/13_run_utility_analysis.py \
  --config /Users/oboh/bio-research/odor-topology/configs/project_config.json
```

## Expected Table Shape

The dataset audit assumes one row per molecule and works best when the table contains:

- one molecule identifier column
- one SMILES column
- binary odor descriptor columns

If `label_columns` is empty in the config, the audit script will try to infer binary label columns automatically.

## Dependencies

A local virtual environment is set up at `/Users/oboh/bio-research/odor-topology/.venv`.

The topology and chemistry packages are installed there:

- `ripser`
- `persim`
- `gudhi`
- `rdkit`

Use the venv Python for the full pipeline:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python ...
```

For OpenPOM checkpoint restore and embedding extraction, use the Python 3.11 environment:

```bash
/Users/oboh/bio-research/odor-topology/.venv311/bin/python ...
```

That env exists because the OpenPOM stack depends on `dgl`, which needed a Python/Torch combination different from the lighter topology env.

Most analysis scripts now default to dataset-stem-specific outputs. That means running the same script on a second dataset will produce a parallel report set instead of overwriting the curated GS/LF outputs.

## First Deliverables

- audited descriptor frequency table
- duplicate and missing-data report
- candidate broad-family shortlist
- embedding geometry smoke test
- topology smoke test with H1 gated behind package availability
- RDKit baseline features for Aim 2
- first official POM embedding export from a restored OpenPOM checkpoint
- first repeated H1 robustness report with Euclidean and cosine metrics plus matched nulls
- first matched POM-versus-baseline robustness comparison with shared subsampling schedules
- full released-checkpoint ensemble audit
- second-route greedy-landmark robustness comparison
- broader GS/LF stress test and explicit overlap audit
- non-overlap 1,600-molecule GS/LF subset for cleaner validation
- first local-neighborhood utility analysis comparing geometry-only versus geometry-plus-topology models

## License

This repository is released under the MIT License. See [LICENSE](/Users/oboh/bio-research/odor-topology/LICENSE).
