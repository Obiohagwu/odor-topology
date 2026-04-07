# Task List

## Done

- [x] Point `configs/project_config.json` at the curated OpenPOM 4,983-row GS/LF dataset
- [x] Run `00_environment_check.py`
- [x] Run `01_dataset_audit.py`
- [x] Run `03_build_rdkit_baselines.py`
- [x] Build a compatible Python 3.11 OpenPOM environment at `.venv311`
- [x] Restore an official OpenPOM checkpoint and verify `predict_embedding()` works on a test batch
- [x] Extract the full `(4983, 256)` primary POM embedding matrix from official checkpoint `experiments_1/checkpoint2`
- [x] Update `configs/project_config.json` with the extracted embedding path
- [x] Run `02_topology_smoke_test.py` against the extracted POM matrix
- [x] Fetch a second official checkpoint for representation-stability checks
- [x] Extract embeddings from `experiments_2/checkpoint2_real.pt` for a replication representation
- [x] Add repeated subsampling H1 robustness analysis
- [x] Add null model generation
- [x] Add POM metric comparison: Euclidean vs cosine
- [x] Run the first full H1 robustness pass on the primary POM embedding
- [x] Add checkpoint-to-checkpoint topology stability comparison
- [x] Compare primary POM robustness against RDKit baselines under matched settings
- [x] Rebuild chemical baselines with paper-matched radius-4 bit and count fingerprints
- [x] Add stricter fingerprint nulls including fixed-margin swap
- [x] Extract all 10 released OpenPOM ensemble checkpoint embeddings
- [x] Audit H1 robustness across the full OpenPOM ensemble
- [x] Add greedy-landmark distance-matrix H1 comparison as a second route
- [x] Audit the broader 5,862-row GS/LF table
- [x] Audit overlap between curated 4,983-row and broader 5,862-row GS/LF tables
- [x] Create and audit the 1,600-molecule non-overlap GS/LF subset
- [x] Run repeated H1 robustness analysis on the broader 5,862-row GS/LF table
- [x] Run repeated H1 robustness analysis on the 1,600-molecule non-overlap GS/LF subset
- [x] Build subset-specific RDKit baselines for the 1,600-molecule non-overlap subset
- [x] Extract the replication checkpoint embedding for the 1,600-molecule non-overlap subset
- [x] Run direct POM-vs-baseline robustness comparison on the 1,600-molecule non-overlap subset
- [x] Run landmark POM-vs-baseline robustness comparison on the 1,600-molecule non-overlap subset
- [x] Build local geometry/topology feature tables for curated utility analysis
- [x] Run the first curated utility analysis
- [x] Build local geometry/topology feature tables for the 1,600-molecule non-overlap subset
- [x] Run the first non-overlap utility analysis
- [x] Make dataset-audit and H1 outputs dataset-specific to prevent report overwrites
- [x] Make baseline and comparison outputs dataset-specific to prevent report overwrites
- [x] Write a short results memo distinguishing "robust topology in POM" from "topology unique to POM"
- [x] Write a short results memo distinguishing "topology exists" from "topology adds utility" and "topology is unique to POM"
- [x] Draft an arXiv-oriented manuscript and figure bundle

## Now

- [ ] Replace the placeholder author line in `arxiv/main.tex`
- [ ] Add external literature citations before upload
- [ ] Compile the TeX bundle once a TeX toolchain is available
- [ ] Add feature-matching or persistence-image style stability beyond top-k persistence magnitudes
- [ ] Decide whether to add a second utility model family beyond ridge regression
- [ ] Decide whether to add a block-permutation test for the topology feature set
- [ ] Decide whether to extend the non-overlap subset analysis from 2 checkpoints to all 10 OpenPOM ensemble checkpoints

## Next

- [ ] Add mild PCA-compression decision rule to the written spec
- [ ] Add dimension-matched or bottleneck-matched chemical baselines to separate compression from odor relevance
- [ ] Test whether the POM-baseline contrast holds under larger subsamples or alternate run counts
- [ ] Sweep landmark counts to see whether the cosine-landmark weakness is stable or only a small-landmark artifact
- [ ] Add a representation comparison on the non-overlap subset if replication embeddings and baselines are built
- [ ] Extend utility analysis to the replication checkpoint and/or all 10 checkpoints

## After That

- [ ] Add family-level size-matched analysis
- [ ] Add family-level utility analysis after overlap control
- [ ] Add a formal interpretation section for publication-quality claims and non-claims
