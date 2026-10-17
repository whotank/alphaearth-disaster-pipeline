from __future__ import annotations

import unittest
from io import BytesIO
from zipfile import ZipFile

from usdm_official_h3 import _season_members
from usdm_validation import (
    evaluate_official_validation,
    expected_weekly_map_count,
)


class UsdmCalendarTests(unittest.TestCase):
    def test_april_october_tuesday_counts(self) -> None:
        self.assertEqual(expected_weekly_map_count(2017), 31)
        self.assertEqual(expected_weekly_map_count(2018), 31)
        self.assertEqual(expected_weekly_map_count(2020), 30)
        self.assertEqual(expected_weekly_map_count(2021), 30)
        self.assertEqual(expected_weekly_map_count(2023), 31)
        self.assertEqual(expected_weekly_map_count(2024), 31)

    def test_archive_members_are_ordered_by_release_date(self) -> None:
        payload = BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("z/USDM_20200630_M.zip", b"")
            archive.writestr("a/usdm_20200623_m.zip", b"")
            archive.writestr("a/USDM_20200616_M.zip", b"")
        payload.seek(0)

        with ZipFile(payload) as archive:
            members = _season_members(archive, 2020)

        self.assertEqual(
            members,
            [
                "a/USDM_20200616_M.zip",
                "a/usdm_20200623_m.zip",
                "z/USDM_20200630_M.zip",
            ],
        )


class OfficialValidationTests(unittest.TestCase):
    @staticmethod
    def _summary() -> dict:
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

    def _evaluate(self, summary: dict) -> dict:
        return evaluate_official_validation(
            summary,
            expected_event="example_drought",
            expected_pre_year=2020,
            expected_post_year=2021,
            expected_h3_cells=100,
        )

    def test_complete_high_agreement_artifact_passes(self) -> None:
        result = self._evaluate(self._summary())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_status_label_cannot_override_failed_evidence(self) -> None:
        summary = self._summary()
        summary["validation_status"] = "validated_official_reference"
        summary["official_vs_mirror_mae"] = 0.13

        result = self._evaluate(summary)

        self.assertFalse(result["passed"])
        self.assertTrue(result["official_reference_complete"])
        self.assertFalse(result["mirror_agreement_passed"])
        self.assertIn("mae_meets_threshold", result["failures"])

    def test_h3_coverage_must_be_exact(self) -> None:
        summary = self._summary()
        summary["h3_coverage"]["official_cells"] = 99
        summary["h3_coverage"]["sets_equal"] = False

        result = self._evaluate(summary)

        self.assertFalse(result["passed"])
        self.assertFalse(result["official_reference_complete"])
        self.assertIn("h3_sets_equal", result["failures"])
        self.assertIn("h3_official_count_matches", result["failures"])


if __name__ == "__main__":
    unittest.main()
