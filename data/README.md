# Data Notes

Place raw source files in `data/raw/` and cleaned intermediates in `data/processed/`.

Expected inputs for the first pass:

- one molecule table, for example CSV or Parquet
- one embedding matrix, for example NPY, NPZ, CSV, or Parquet

The config file should point to the real paths. The code does not assume the files must live inside this repo, but that is the cleanest setup.
