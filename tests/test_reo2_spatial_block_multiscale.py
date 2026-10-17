from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h3
import pandas as pd

import reo2_spatial_block_multiscale as multiscale


class ScalePolicyTests(unittest.TestCase):
    def test_one_seventh_support_rule_scales_with_parent_capacity(self) -> None:
        self.assertEqual(multiscale.scale_minimum_cells(7), 1)
        self.assertEqual(multiscale.scale_minimum_cells(41), 6)
        self.assertEqual(multiscale.scale_minimum_cells(49), 7)
        self.assertEqual(multiscale.scale_minimum_cells(343), 49)

    def test_scale_grid_is_fixed_around_immutable_primary(self) -> None:
        self.assertEqual(multiscale.SENSITIVITY_PARENT_LEVELS, (1, 3))
        self.assertEqual(multiscale.PRIMARY_PARENT_LEVELS, 2)
        self.assertEqual(multiscale.ALL_PARENT_LEVELS, (1, 2, 3))

    def test_insufficient_blocks_are_reported_without_permutation_p(self) -> None:
        parent = h3.latlng_to_cell(37.5, -122.2, 7)
        neighboring_parent = next(iter(h3.grid_ring(parent, 2)))
        cells = sorted(h3.cell_to_children(parent, 8)) + sorted(
            h3.cell_to_children(neighboring_parent, 8)
        )
        frame = pd.DataFrame(
            {
                "h3_index": cells,
                "ecs_mean": list(range(len(cells))),
                "benchmark_mean": list(reversed(range(len(cells)))),
            }
        )

        result, _, null_statistics = multiscale.run_scale(
            frame,
            h3_resolution=8,
            parent_levels=1,
            min_blocks=8,
            iterations=999,
            seed=123,
        )

        self.assertEqual(result["status"], "excluded")
        self.assertEqual(
            result["exclusion_reason"],
            "too_few_retained_parent_blocks",
        )
        self.assertEqual(result["n_parent_blocks_retained"], 2)
        self.assertNotIn("monte_carlo_p_two_sided", result)
        self.assertIsNone(null_statistics)

    def test_family_comparison_flags_threshold_and_eligibility_changes(
        self,
    ) -> None:
        summaries = {
            "1": {
                "fire": {
                    "n_events": 8,
                    "n_positive": 8,
                    "n_negative": 0,
                    "interpretation_0_05": (
                        "positive_majority_sign_test_below_0_05"
                    ),
                },
                "forest_loss": {
                    "n_events": 2,
                    "n_positive": 2,
                    "n_negative": 0,
                    "interpretation_0_05": "not_below_0_05",
                },
            },
            "2": {
                "fire": {
                    "n_events": 8,
                    "n_positive": 7,
                    "n_negative": 1,
                    "interpretation_0_05": "not_below_0_05",
                },
                "forest_loss": {
                    "n_events": 2,
                    "n_positive": 2,
                    "n_negative": 0,
                    "interpretation_0_05": "not_below_0_05",
                },
            },
            "3": {
                "fire": {
                    "n_events": 0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "interpretation_0_05": "not_below_0_05",
                },
                "forest_loss": {
                    "n_events": 2,
                    "n_positive": 0,
                    "n_negative": 2,
                    "interpretation_0_05": "not_below_0_05",
                },
            },
        }

        comparison = multiscale.compare_family_interpretations(summaries)

        self.assertEqual(
            comparison["1"]["fire"]["assessment"],
            "sign_test_threshold_interpretation_changed",
        )
        self.assertEqual(
            comparison["3"]["fire"]["assessment"],
            "not_comparable_eligible_event_set_changed",
        )
        self.assertEqual(
            comparison["3"]["forest_loss"]["assessment"],
            "direction_changed",
        )


class MetricAuditTests(unittest.TestCase):
    def test_primary_replay_fails_closed_on_metric_mismatch(self) -> None:
        recorded = {
            "event_key": "synthetic",
            "status": "eligible",
            "exclusion_reason": None,
            "n_parent_blocks_retained": 10,
            "observed_block_spearman": 0.5,
            "monte_carlo_p_two_sided": 0.02,
        }
        replayed = {
            "status": "eligible",
            "exclusion_reason": None,
            "n_parent_blocks_retained": 10,
            "observed_block_spearman": 0.4,
            "monte_carlo_p_two_sided": 0.02,
        }

        with self.assertRaisesRegex(RuntimeError, "observed_block_spearman"):
            multiscale.assert_primary_replay(recorded, replayed)

    def test_cell_metric_matches_qa_average_rank_spearman(self) -> None:
        frame = pd.DataFrame(
            {
                "h3_index": ["a", "b", "c", "d", "e"],
                "ecs_mean": [1.0, 2.0, 2.0, 4.0, 5.0],
                "benchmark_mean": [5.0, 3.0, 3.0, 2.0, 1.0],
            }
        )
        expected = frame["ecs_mean"].rank(method="average").corr(
            frame["benchmark_mean"].rank(method="average")
        )
        event_record = {
            "event_key": "synthetic",
            "disaster_type": "fire",
            "cell_level_spearman_descriptive_only": expected,
            "observed_block_spearman": -0.5,
            "monte_carlo_p_two_sided": 0.1,
        }
        replay = {
            "observed_block_spearman": -0.5,
            "monte_carlo_p_two_sided": 0.1,
        }

        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            (event_dir / "qa_numeric_summary.json").write_text(
                json.dumps(
                    {
                        "n_cells": len(frame),
                        "corr_spearman_benchmark_vs_ecs": expected,
                    }
                ),
                encoding="utf-8",
            )
            audit = multiscale.metric_audit(
                frame=frame,
                event_record=event_record,
                event_dir=event_dir,
                primary_replay=replay,
            )

        self.assertTrue(audit["qa_matches_direct_to_1e_12"])
        self.assertTrue(
            audit["direct_matches_primary_cell_field_to_1e_12"]
        )
        self.assertAlmostEqual(audit["direct_cell_spearman"], expected)
        self.assertIn(
            "sample unit",
            audit["metric_difference_explanation"],
        )

    def test_stale_qa_row_count_is_disclosed_not_treated_as_validation(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "h3_index": ["a", "b", "c"],
                "ecs_mean": [1.0, 2.0, 3.0],
                "benchmark_mean": [1.0, 2.0, 3.0],
            }
        )
        event_record = {
            "event_key": "synthetic",
            "disaster_type": "fire",
            "cell_level_spearman_descriptive_only": 1.0,
            "observed_block_spearman": 1.0,
            "monte_carlo_p_two_sided": 0.1,
        }
        replay = {
            "observed_block_spearman": 1.0,
            "monte_carlo_p_two_sided": 0.1,
        }

        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            (event_dir / "qa_numeric_summary.json").write_text(
                json.dumps(
                    {
                        "n_cells": 2,
                        "corr_spearman_benchmark_vs_ecs": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            audit = multiscale.metric_audit(
                frame=frame,
                event_record=event_record,
                event_dir=event_dir,
                primary_replay=replay,
            )

        self.assertEqual(audit["qa_audit_status"], "stale_n_cells")
        self.assertFalse(audit["qa_comparable_to_current_source"])
        self.assertIsNone(audit["qa_matches_direct_to_1e_12"])


if __name__ == "__main__":
    unittest.main()
