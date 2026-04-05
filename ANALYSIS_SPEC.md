# Analysis Spec

## Core Claim
We will test whether any apparent topological structure in the Principal Odor Map is reproducible, representation-specific, and scientifically useful, rather than assuming that persistent homology on a learned embedding is inherently meaningful.

## Confirmatory Scope

- Primary target: H1 in one fixed POM representation
- Baselines: Morgan fingerprints and physicochemical descriptors
- Metrics:
  - POM: Euclidean and cosine
  - Morgan: Jaccard or Tanimoto-style distance
  - Physicochemical descriptors: Euclidean after z-scoring
- Higher-dimensional structure: exploratory only

## Robustness Criteria For A Real H1 Feature

1. Exceeds the 95th percentile of the matched null persistence distribution
2. Recurs across repeated landmark subsamples
3. Appears under at least two defensible metric or filtration variants
4. Survives mild dimensional compression
5. Can be matched across runs with reasonable birth-death stability

## Null Models

- coordinate permutation
- covariance-matched Gaussian null
- size-matched random subset null for family analyses
- label-shuffle null for family analyses

## Family Analysis Rules

- only broad families above a minimum size threshold
- repeated size matching
- overlap-reduced core-category sensitivity analysis
- family-level claims are secondary to whole-space POM versus baseline comparisons

## Immediate Execution Order

1. Lock the primary representation and file paths
2. Audit table shape, missingness, duplicates, and label frequencies
3. Define candidate broad families
4. Run embedding geometry smoke tests
5. Install topology and chemistry dependencies
6. Add landmark-based H1 analysis
7. Add null models
8. Add baseline representations

## Decision Boundaries

- No claim that loops prove cyclic perceptual dimensions
- No claim that failure to detect H1 proves contractibility
- No family-level interpretation without size-matched controls
- No central claims from UMAP alone
