# Odor Topology Robustness

This project tests whether apparent topological structure in learned olfactory representation space is reproducible, representation-specific, and scientifically useful.

The immediate goal is not to "discover the topology of odor space." The immediate goal is to build a clean pipeline that can answer a narrower question well:

> Does any topological signal in a fixed Principal Odor Map embedding survive metric choice, subsampling, null-model comparison, and baseline-representation controls?

## Current Focus

- lock one primary POM representation
- audit the OpenPOM GoodScents-Leffingwell table
- define descriptor and family subsets cleanly
- run geometry and topology smoke tests on the embedding
- compare against matched chemical baselines once RDKit is available

## Project Layout

- `configs/`: local configuration files
- `data/raw/`: raw tables and embedding files
- `data/processed/`: cleaned outputs
- `outputs/reports/`: JSON and CSV reports
- `outputs/figures/`: plots
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

## First Deliverables

- audited descriptor frequency table
- duplicate and missing-data report
- candidate broad-family shortlist
- embedding geometry smoke test
- topology smoke test with H1 gated behind package availability
- RDKit baseline features for Aim 2
- first official POM embedding export from a restored OpenPOM checkpoint

## License

This repository is released under the MIT License. See [LICENSE](/Users/oboh/bio-research/odor-topology/LICENSE).
