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

## Now

- [ ] Review descriptor frequency table
- [ ] Pick minimum size threshold for confirmatory broad families
- [ ] Extract embeddings from `experiments_2/checkpoint2_real.pt` for a replication representation

## Next

- [ ] Add landmark subsampling analysis
- [ ] Add null model generation
- [ ] Add POM metric comparison: Euclidean vs cosine

## After That

- [ ] Add Morgan fingerprint baseline
- [ ] Add physicochemical descriptor baseline
- [ ] Add family-level size-matched analysis
- [ ] Add utility analysis beyond geometry
