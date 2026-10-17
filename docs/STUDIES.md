# Study inventory

## Outcome-blind MTBS fire cohort

Ten U.S. fires were selected from a frozen metadata snapshot across years,
regions, sizes, and dominant land-cover strata. The official MTBS target is the
fraction of valid thematic classes 1–5 that fall in classes 3–4
(moderate/high severity) within each H3 resolution-8 cell.

The retained events are Rhea, Woolsey, Hadweenzic River, Beachie Creek, Sarpy,
Shovel Creek, Walker, Black, Double Creek, and York. Four County failed a
pre-outcome constant-target gate and was replaced by the next prespecified
reserve, Walker. The attempted-event ledger preserves that replacement.

The common population contains 6,864 H3-r8 cells. AlphaEarth associations were
positive in nine fires and ranged from -0.378 to 0.913. The per-event metrics,
population hashes, and inference output are under
`studies/mtbs_heldout_20260817/`.

## TESSERA secondary analysis

The TESSERA comparison uses the same frozen fire cells, event years, and
optical comparator. Nine fires are evaluable. Hadweenzic River is retained as
missing because the official v1.0 manifest lacked a required year-tile object.
Artifacts are under `outputs/tessera_mtbs_secondary/`.

## Development cases

The eight non-fire cases reuse the same AlphaEarth embedding-change score:

- Lava emplacement: Kīlauea 2018 (`rho=0.774`), La Palma 2021
  (`rho=0.721`), and Mauna Loa 2022 (`rho=0.360`).
- Forest loss: Georgia 2021 (`rho=0.235`) and Pacific Northwest 2021
  (`rho=0.464`).
- Drought: Four Corners 2018 (`rho=0.126`), North Dakota 2021
  (`rho=0.276`), and Iowa recovery 2024 (`rho=0.152`).

These cases were not held out and do not support a population-level
cross-hazard claim. Their spatial-sensitivity artifacts are under
`outputs/reo2_development_spatial_robustness/`.

## Temporal support and scale

Fire models and optical comparators use full calendar years `y-1` and `y+1`.
The lava brackets are 2017–2019 for Kīlauea, 2020–2022 for La Palma, and
2021–2023 for Mauna Loa. The primary paper analyses use H3 resolution 8, whose
mean cell area is approximately 0.74 square kilometres. AlphaEarth input
embeddings are sampled at 10 m before cell averaging.

The paper's Spearman coefficients quantify within-event ordering against a
mapped reference. They do not measure event detection, causal attribution, or
reference-map accuracy.
