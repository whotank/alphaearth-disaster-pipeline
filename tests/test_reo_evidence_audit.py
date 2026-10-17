from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from reo_evidence_audit import _event_audit


class DroughtEvidenceAuditTests(unittest.TestCase):
    @staticmethod
    def _validation() -> dict:
        return {
            "event": "example_drought",
            "pre_year": 2020,
            "post_year": 2021,
            "weekly_map_count_pre": 30,
            "weekly_map_count_post": 30,
            "h3_cells": 100,
            "h3_coverage": {
                "alphaearth_cells": 100,
                "mirror_cells": 100,
                "official_cells": 100,
                "all_unique": True,
                "sets_equal": True,
                "resolution": 6,
            },
            "pre_archive_sha256": "a" * 64,
            "post_archive_sha256": "b" * 64,
            "aoi_sha256": "c" * 64,
            "official_vs_mirror_pearson": 0.995,
            "official_vs_mirror_spearman": 0.95,
            "official_vs_mirror_mae": 0.10,
            "official_vs_mirror_max_abs_error": 0.20,
        }

    @staticmethod
    def _metric() -> pd.Series:
        return pd.Series(
            {
                "event_key": "example_drought",
                "disaster_type": "drought",
                "benchmark_type": "usdm_season_delta",
                "pre_year": 2020,
                "post_year": 2021,
                "benchmark_metric_year": 2021,
                "benchmark_std": 0.5,
                "benchmark_nunique": 80,
                "pct_loss_vs_alpha": 0.0,
                "n_h3_join": 100,
                "n_joined_rows": 100,
                "pearson_benchmark_ecs": 0.2,
                "spearman_benchmark_ecs": 0.3,
                "publication_reference_source": "official_ndmc_vectors",
            }
        )

    @staticmethod
    def _manifest() -> dict:
        return {
            "pipeline_preset": {
                "disaster_type": "drought",
                "reference_source_state": "official_vector_crosscheck_required",
                "benchmark": {
                    "type": "usdm_season_delta",
                    "metric_year": 2021,
                },
            },
            "integrity_report": {
                "disaster_type": "drought",
                "pre_year": 2020,
                "post_year": 2021,
                "h3_resolution": 6,
                "total_h3_cells_alphaearth": 100,
                "total_h3_cells_inner_join": 100,
                "benchmark_temporal_relation": "reference_year_matches_post",
            },
            "artifacts": {
                "joined_official.parquet": True,
                "qa_official_numeric_summary.json": True,
                "gis_reference_raster_2x2.png": False,
            },
        }

    def test_completed_artifact_overrides_static_crosscheck_requirement(self) -> None:
        with TemporaryDirectory() as directory:
            event_dir = Path(directory)
            (event_dir / "usdm_official_validation.json").write_text(
                json.dumps(self._validation()), encoding="utf-8"
            )
            (event_dir / "qa_official_numeric_summary.json").write_text(
                json.dumps({"n_cells": 100}), encoding="utf-8"
            )

            row = _event_audit(
                self._metric(),
                self._manifest(),
                event_dir,
            )

        self.assertTrue(row["official_reference_validated"])
        self.assertEqual(row["recommendation"], "ready_for_core_results")
        self.assertEqual(
            row["official_validation_status"],
            "validated_official_reference",
        )

    def test_bad_mirror_agreement_is_disclosed_without_replacing_official(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            event_dir = Path(directory)
            validation = self._validation()
            validation["validation_status"] = "validated_official_reference"
            validation["official_vs_mirror_mae"] = 0.13
            (event_dir / "usdm_official_validation.json").write_text(
                json.dumps(validation), encoding="utf-8"
            )
            (event_dir / "qa_official_numeric_summary.json").write_text(
                json.dumps({"n_cells": 100}), encoding="utf-8"
            )

            row = _event_audit(
                self._metric(),
                self._manifest(),
                event_dir,
            )

        self.assertTrue(row["official_reference_validated"])
        self.assertTrue(row["official_reference_complete"])
        self.assertFalse(row["official_mirror_agreement_passed"])
        self.assertEqual(
            row["recommendation"], "ready_with_mirror_disagreement"
        )
        self.assertEqual(
            row["official_validation_status"],
            "official_reference_complete_mirror_disagreement",
        )
        self.assertIn(
            "mae_meets_threshold",
            row["official_validation_failures"],
        )


if __name__ == "__main__":
    unittest.main()
