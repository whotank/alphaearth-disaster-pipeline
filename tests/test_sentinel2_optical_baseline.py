from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import sentinel2_optical_baseline as baseline


class OpticalFormulaTests(unittest.TestCase):
    def test_ndvi_nbr_and_absolute_changes(self) -> None:
        pre_ndvi = baseline.normalized_difference(0.60, 0.20)
        post_ndvi = baseline.normalized_difference(0.40, 0.30)
        pre_nbr = baseline.normalized_difference(0.60, 0.30)
        post_nbr = baseline.normalized_difference(0.40, 0.40)

        self.assertAlmostEqual(pre_ndvi, 0.5)
        self.assertAlmostEqual(post_ndvi, 1.0 / 7.0)
        self.assertAlmostEqual(pre_nbr, 1.0 / 3.0)
        self.assertAlmostEqual(post_nbr, 0.0)
        self.assertAlmostEqual(
            baseline.absolute_index_change(pre_ndvi, post_ndvi),
            5.0 / 14.0,
        )
        self.assertAlmostEqual(
            baseline.absolute_index_change(pre_nbr, post_nbr),
            1.0 / 3.0,
        )

    def test_zero_denominator_is_missing_not_imputed(self) -> None:
        result = baseline.normalized_difference(
            np.array([0.4, 0.0]),
            np.array([0.2, 0.0]),
        )

        self.assertAlmostEqual(float(result[0]), 1.0 / 3.0)
        self.assertTrue(np.isnan(result[1]))


class DeterministicEligibilityTests(unittest.TestCase):
    @staticmethod
    def _valid_summary() -> baseline.ReferenceSummary:
        return baseline.ReferenceSummary(
            n_cells=100,
            n_h3_unique=100,
            n_benchmark_finite=100,
            n_ecs_finite=100,
            benchmark_nunique=10,
            benchmark_min=0.0,
            benchmark_max=1.0,
        )

    def test_all_three_bracketed_volcanic_events_are_eligible(self) -> None:
        volcanoes = [
            spec
            for spec in baseline.EVENT_SPECS
            if spec.family == "volcanic_eruption"
        ]

        self.assertEqual(len(volcanoes), 3)
        decisions = [
            baseline.assess_event_eligibility(spec, self._valid_summary())
            for spec in volcanoes
        ]
        self.assertTrue(all(decision.eligible for decision in decisions))

    def test_fire_temporal_rule_is_outcome_independent_and_repeatable(self) -> None:
        specs = {spec.event_key: spec for spec in baseline.EVENT_SPECS}
        summary = self._valid_summary()

        first = baseline.assess_event_eligibility(
            specs["camp_fire_2018"],
            summary,
        )
        second = baseline.assess_event_eligibility(
            specs["camp_fire_2018"],
            summary,
        )
        thomas = baseline.assess_event_eligibility(
            specs["thomas_fire_2017"],
            summary,
        )

        self.assertEqual(first, second)
        self.assertTrue(first.eligible)
        self.assertFalse(thomas.eligible)
        self.assertIn(
            "fire_reference_year_is_not_post_year",
            thomas.reasons,
        )

    def test_nonvarying_reference_is_rejected_before_sentinel2(self) -> None:
        quebec = next(
            spec
            for spec in baseline.EVENT_SPECS
            if spec.event_key == "quebec_fires_2023"
        )
        constant_reference = baseline.ReferenceSummary(
            n_cells=50,
            n_h3_unique=50,
            n_benchmark_finite=50,
            n_ecs_finite=50,
            benchmark_nunique=1,
            benchmark_min=0.0,
            benchmark_max=0.0,
        )

        decision = baseline.assess_event_eligibility(
            quebec,
            constant_reference,
        )

        self.assertFalse(decision.eligible)
        self.assertEqual(
            decision.reasons,
            (
                "reference_has_no_spatial_variation",
                "reference_has_no_positive_support",
            ),
        )

    def test_predefined_selection_has_exact_stable_membership(self) -> None:
        constant_reference = baseline.ReferenceSummary(
            n_cells=50,
            n_h3_unique=50,
            n_benchmark_finite=50,
            n_ecs_finite=50,
            benchmark_nunique=1,
            benchmark_min=0.0,
            benchmark_max=0.0,
        )
        selected = tuple(
            spec.event_key
            for spec in sorted(
                baseline.EVENT_SPECS,
                key=lambda candidate: candidate.event_key,
            )
            if baseline.assess_event_eligibility(
                spec,
                (
                    constant_reference
                    if spec.event_key == "quebec_fires_2023"
                    else self._valid_summary()
                ),
            ).eligible
        )

        self.assertEqual(
            selected,
            (
                "bootleg_fire_2021",
                "caldor_fire_2021",
                "camp_fire_2018",
                "dixie_fire_2021",
                "kilauea_lava_2018",
                "la_palma_lava_2021",
                "mauna_loa_lava_2022",
                "mosquito_fire_2022",
            ),
        )


class SamePopulationJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = pd.DataFrame(
            {
                "h3_index": ["h3-a", "h3-b", "h3-c"],
                "ecs_mean": [0.1, 0.2, 0.3],
                "benchmark_mean": [0.0, 0.5, 1.0],
            }
        )
        self.optical = pd.DataFrame(
            {
                "h3_index": ["h3-c", "h3-a", "h3-b"],
                "abs_dnbr": [0.3, 0.1, 0.2],
                "abs_dndvi": [0.03, 0.01, 0.02],
                "s2_valid_fraction": [0.90, 0.80, 0.49],
            }
        )

    def test_join_restores_reference_order_and_one_common_mask(self) -> None:
        joined = baseline.join_same_population(
            self.reference,
            self.optical,
            min_cell_valid_fraction=0.50,
        )

        self.assertEqual(joined["h3_index"].tolist(), ["h3-a", "h3-b", "h3-c"])
        self.assertEqual(joined["common_valid"].tolist(), [True, False, True])
        valid = joined[joined["common_valid"]]
        self.assertEqual(
            valid[["ecs_mean", "abs_dnbr", "abs_dndvi"]].notna().all().tolist(),
            [True, True, True],
        )

    def test_missing_optical_h3_is_rejected_instead_of_imputed(self) -> None:
        with self.assertRaisesRegex(ValueError, "H3 population mismatch"):
            baseline.join_same_population(
                self.reference,
                self.optical.iloc[:2],
                min_cell_valid_fraction=0.50,
            )

    def test_duplicate_optical_h3_is_rejected(self) -> None:
        duplicated = pd.concat(
            [self.optical, self.optical.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "H3 population mismatch"):
            baseline.join_same_population(
                self.reference,
                duplicated,
                min_cell_valid_fraction=0.50,
            )


if __name__ == "__main__":
    unittest.main()
