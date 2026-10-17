# Eight-case development spatial robustness analysis

## Status

This is a frozen, explicitly post hoc spatial sensitivity analysis. It is conditional on these eight development-case maps and does not constitute held-out validation or population-level inference.

- Protocol: `reo2-development-spatial-robustness-v1`
- Protocol SHA-256: `b236d19ef87c73ec1c5107afbc312f35e5e96db9c0c60d3cf960697c79502f15`
- Implementation SHA-256: `96bfe3325bfcad299bc1aed50758002d09461cc0edf51f84f6706bedfa5ceaaf`
- Primary scale: H3 parent levels `2` coarser
- Permutations per eligible event-scale: `9,999`
- Input validation: original frozen SHA-256 for lava and forest-loss joins; Amendment 1 committed-audit validation for regenerated drought joins

## Results

| event | family | fine-cell rho | L1 block rho | L2 block rho | L2 raw p | L2 Holm p | L3 block rho | direction |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kilauea_lava_2018 | lava | 0.774 | 0.848 | 0.875 | 0.0020 | 0.0120 | NA | positive_at_all_assessable_scales_but_incomplete |
| la_palma_lava_2021 | lava | 0.721 | 0.742 | 0.908 | 0.0005 | 0.0035 | NA | positive_at_all_assessable_scales_but_incomplete |
| mauna_loa_lava_2022 | lava | 0.360 | 0.416 | 0.525 | 0.0061 | 0.0244 | NA | positive_at_all_assessable_scales_but_incomplete |
| pacific_nw_forest_loss_2021 | forest_loss | 0.464 | 0.567 | 0.526 | 0.0002 | 0.0016 | -0.150 | scale_dependent_or_nonpositive |
| georgia_forest_loss_2021 | forest_loss | 0.235 | 0.122 | 0.012 | 0.9246 | 0.9246 | -0.462 | scale_dependent_or_nonpositive |
| four_corners_drought_2018 | drought | 0.126 | 0.176 | 0.322 | 0.1212 | 0.2568 | NA | positive_at_all_assessable_scales_but_incomplete |
| north_dakota_drought_2021 | drought | 0.276 | 0.285 | 0.269 | 0.0041 | 0.0205 | 0.386 | directionally_stable |
| iowa_drought_recovery_2024 | drought | 0.152 | 0.179 | 0.176 | 0.0856 | 0.2568 | 0.267 | directionally_stable |

## Fixed-case aggregate

- Primary block correlations assessable: 8/8
- Positive / negative: `8 / 0`
- Equal-event median primary block rho: `0.424`
- Exact two-sided sign-test p: `0.0078`
- Holm-supported positive primary results: `5`
- Spatially robust within case under the frozen classification: `1`

The sign test is a descriptive summary of these fixed cases. It does not make the event collection a probability sample.

## Interpretation limits

- Parent-block aggregation reduces fine-cell pseudoreplication but does not guarantee exchangeability across broad spatial gradients.
- The reported 2.5th and 97.5th permutation quantiles are null envelopes, not confidence intervals for the observed correlation.
- Scale sensitivity is descriptive. No favorable scale replaces the parent-level-2 primary result.
- A missing or ineligible scale is not counted as a robustness pass.
- Every weak, negative, scale-dependent, or ineligible result is retained.
- Drought joins are validated reconstructions, not byte-identical recoveries of the missing original Parquet files.
