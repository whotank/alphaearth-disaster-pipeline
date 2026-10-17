# Frozen prospective 10-fire MTBS held-out study

## Integrity and sequencing

- Canonical freeze SHA-256: `eb204cfc9d4a16d9c27db4d7641c4ebf3e6fc98ae637ed71a9a7c50a4a32794e`.
- Pre-AlphaEarth population freeze SHA-256: `740b1186e893189606411ada3f0c7c59dc8e97ff5c54ed661ca5d714947a2c7c`.
- Eligibility and H3 populations were frozen before AlphaEarth pixel extraction.
- Per-event feature files were sealed without correlations; analysis opened only after all ten hashes verified.
- Existing configured events, outcome summaries, model labels, and paper files were not imported.

## Reference and feature contract

- Official USFS/USGS MTBS boundary polygons and annual thematic Severity rasters were used.
- H3-r8 centers lie in each official boundary; retained cells have at least 80% MTBS classes 1-5 support.
- Reference is the class 3+4 fraction among classes 1-5; classes 0, 6, and NoData are excluded.
- Sentinel-2 uses full calendar Y-1 and Y+1 SCL-masked median composites; dNBR and dNDVI loss are signed pre-minus-post changes.
- AlphaEarth ECS is the 10 m H3 mean of the 64-axis squared annual-embedding change, in deterministic 40-cell batches.
- Every metric uses the exact same complete-case H3 population; no imputation was performed.

## Attempt and replacement audit

- Attempted rows: 11; failed frozen gates: 1; replacements: 1.
- Replacement was permitted only for P01-P05 and consumed reserves in canonical order.
- P06 land-cover mismatches were reported without changing eligibility.

## Primary event-level inference

- Unweighted mean D: `-0.115799`.
- 95% deterministic event-bootstrap CI: `[-0.393415, 0.078560]`.
- Exact event sign-flip p (descriptive): `0.546875`.
- Event is the replication unit. No cell-level p-values were calculated.

## Secondary sensitivities

- Higher MTBS/Sentinel-2 support and H3-r7 aggregation are labeled secondary and never redefine primary eligibility.

## Paper recommendation

The mean endpoint is negative, but the event-bootstrap interval includes zero. Do not present this study as confirmatory evidence of an ECS advantage; if included, report it symmetrically as an inconclusive prospective replication.
