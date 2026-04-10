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







This repository is released under the MIT License. See [LICENSE](/Users/oboh/bio-research/odor-topology/LICENSE).
