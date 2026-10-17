# Post hoc MTBS ordinal-severity diagnostic

## Status and guardrail

This analysis was defined after the York result was inspected. It is exploratory
and cannot replace the frozen primary endpoint. The original target definitions
were written before their first correlations were computed. The protocol
SHA-256 is `143a07a339843e37c141a89f034712412093c011128017696e52af96e94106aa`.

An independent audit then found an inherited raster-window edge-rounding defect.
The mandatory correction amendment SHA-256 is `b85f7269c396300c20a253aad65536b98be8163f1460416b8f7cc175377258e1`. Its numerical
direction was approximately known when the amendment was recorded, so the
corrected analysis remains explicitly exploratory.

The event list, event order, H3-resolution-8 populations, AlphaEarth features,
Sentinel-2 features, and TESSERA availability are unchanged.

## Integrity replay

- The immutable sealed primary target reproduces the frozen study.
  Maximum absolute differences are:
  AlphaEarth `1.110e-16`,
  dNBR `1.110e-16`,
  dNDVI loss `1.110e-16`, and
  D `6.592e-17`.
- The corrected window rule adds `7,963` valid MTBS pixels
  (`3,230` of 6,864 cells) relative to the inherited reducer. The
  largest per-cell corrected moderate/high target change is
  `0.003731`.
- Cells with no class 1-4 pixel across all events: `0`.
- Excluding class 5 leaves `24` cells below 80% ordered
  support; the minimum is `45.3%`. The prespecified
  class-5-as-zero sensitivity retains full class 1-5 support.
- Primary events: 10/10. Prespecified TESSERA events: 9/10.

## Cohort-level results

`D` is the embedding correlation minus the stronger same-target dNBR or dNDVI
correlation. Positive means the embedding ranks that target more strongly;
negative means the optical oracle does.

| Target | AE n | Median AlphaEarth rho [range] | Mean AlphaEarth D | 95% event interval | D signs + / - | TESSERA n | Mean TESSERA D | 95% event interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_mod_high | 10 | 0.712 [-0.378, 0.913] | -0.116 | [-0.394, 0.078] | 4/6 | 9 | -0.092 | [-0.320, 0.095] |
| corrected_mod_high_1_5 | 10 | 0.712 [-0.378, 0.914] | -0.116 | [-0.394, 0.079] | 4/6 | 9 | -0.092 | [-0.320, 0.095] |
| ordinal_mean_1_4 | 10 | 0.724 [-0.038, 0.925] | -0.080 | [-0.256, 0.061] | 4/6 | 9 | -0.064 | [-0.181, 0.045] |
| any_severity_2_4 | 10 | 0.502 [0.299, 0.851] | -0.016 | [-0.082, 0.055] | 3/7 | 9 | 0.003 | [-0.083, 0.087] |
| moderate_high_3_4 | 10 | 0.711 [-0.378, 0.913] | -0.115 | [-0.394, 0.079] | 3/7 | 9 | -0.092 | [-0.320, 0.095] |
| high_only_4 | 9 | 0.614 [-0.214, 0.895] | -0.099 | [-0.308, 0.058] | 4/5 | 8 | -0.155 | [-0.310, 0.003] |
| ordinal_mean_green_zero | 10 | 0.724 [-0.038, 0.923] | -0.082 | [-0.257, 0.059] | 4/6 | 9 | -0.065 | [-0.182, 0.045] |

## Ordinal target by event

The ordinal target assigns class 1 = 0, class 2 = 1/3, class 3 = 2/3,
and class 4 = 1. Class 5, increased greenness, is excluded because it is not an
ordered burn-severity class. Its AlphaEarth median correlation is
`0.724`, compared with
`0.712`
for the frozen primary target. Mean AlphaEarth D changes from
`-0.116`
to `-0.080`, but its event interval still
includes zero and the event signs remain 4 positive and 6 negative. Across the
nine common model events, mean AlphaEarth minus TESSERA ordinal rho is
`-0.025`
(`[-0.109,
0.044]`).

| Event | Cells | AlphaEarth rho | dNBR rho | dNDVI rho | AlphaEarth D | TESSERA rho |
| --- | --- | --- | --- | --- | --- | --- |
| RHEA | 1240 | 0.573 | 0.533 | 0.501 | 0.040 | 0.475 |
| WOOLSEY | 434 | 0.781 | 0.725 | 0.744 | 0.037 | 0.826 |
| HADWEENZIC RIVER | 260 | 0.904 | 0.908 | 0.860 | -0.004 | NA |
| BEACHIE CREEK | 1046 | 0.820 | 0.986 | 0.978 | -0.165 | 0.858 |
| SARPY | 253 | 0.433 | 0.678 | 0.416 | -0.245 | 0.506 |
| SHOVEL CREEK | 589 | 0.497 | 0.138 | 0.350 | 0.147 | 0.367 |
| WALKER | 284 | 0.925 | 0.957 | 0.974 | -0.050 | 0.904 |
| BLACK | 1448 | 0.674 | 0.405 | 0.340 | 0.269 | 0.647 |
| DOUBLE CREEK | 900 | 0.775 | 0.884 | 0.726 | -0.109 | 0.816 |
| YORK | 410 | -0.038 | 0.687 | 0.661 | -0.725 | 0.264 |

## York

| Target | AlphaEarth rho | TESSERA rho | dNBR rho | dNDVI rho |
| --- | --- | --- | --- | --- |
| primary_mod_high | -0.378 | -0.008 | 0.837 | 0.633 |
| corrected_mod_high_1_5 | -0.378 | -0.008 | 0.837 | 0.633 |
| ordinal_mean_1_4 | -0.038 | 0.264 | 0.687 | 0.661 |
| any_severity_2_4 | 0.344 | 0.504 | 0.252 | 0.425 |
| moderate_high_3_4 | -0.378 | -0.008 | 0.837 | 0.633 |
| high_only_4 | -0.214 | 0.024 | 0.593 | 0.483 |
| ordinal_mean_green_zero | -0.038 | 0.264 | 0.687 | 0.661 |

York is 84.0% class 2 across the analyzed pixels, with 9.4% class 3, 2.0%
class 4, 4.6% class 1, and no class 5. The gradual ordinal target moves
AlphaEarth from `-0.378` to
`-0.038`, which is approximately zero rather than
evidence of ordered-severity recovery. Only the any-severity target turns
positive (`0.344`); this target mainly distinguishes
class 1 from York's dominant class 2. The matched optical oracle remains
stronger on both the ordinal target (AlphaEarth D
`-0.725`) and any-severity target
(D `-0.081`). York is therefore
target-dependent but not rescued by the ordinal analysis.

## Interpretation boundary

This study asks whether gradual MTBS class weights change the ranking result. It
does not establish that one target is the correct ground truth, and it does not
turn the analysis into fire detection. MTBS classes remain thematic categories
derived mainly from Landsat dNBR and analyst interpretation. The optical oracle
is recomputed for every target, so any embedding improvement is judged against a
matched comparator rather than the old primary comparator.

Rhea has no class-4 pixels, so high-only is unavailable for Rhea and uses nine
AlphaEarth events and eight TESSERA events. All target-specific denominators are
shown above. All targets and all events must be reported together. The ordinal
result is suitable for an exploratory supplement only with this post hoc label
and the reducer-correction disclosure.
