from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fire_analysis import _publication_join


class PublicationJoinSelectionTests(unittest.TestCase):
    @staticmethod
    def _integrity() -> dict:
        return {
            "pre_year": 2020,
            "post_year": 2021,
            "h3_resolution": 6,
            "total_h3_cells_alphaearth": 100,
            "benchmark": {"type": "usdm_season_delta"},
        }

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

    def test_official_join_selected_only_after_recomputed_pass(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            event_dir = output_dir / "example_drought"
            event_dir.mkdir()
            (event_dir / "joined_official.parquet").touch()
            (event_dir / "qa_official_numeric_summary.json").touch()
            (event_dir / "usdm_official_validation.json").write_text(
                json.dumps(self._validation()), encoding="utf-8"
            )

            join, qa, source, _ = _publication_join(
                output_dir,
                "example_drought",
                self._integrity(),
            )

        self.assertEqual(join.name, "joined_official.parquet")
        self.assertEqual(qa.name, "qa_official_numeric_summary.json")
        self.assertEqual(source, "official_ndmc_vectors")

    def test_failed_mirror_agreement_still_uses_authoritative_official(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            event_dir = output_dir / "example_drought"
            event_dir.mkdir()
            (event_dir / "joined_official.parquet").touch()
            (event_dir / "qa_official_numeric_summary.json").touch()
            validation = self._validation()
            validation["official_vs_mirror_pearson"] = 0.98
            (event_dir / "usdm_official_validation.json").write_text(
                json.dumps(validation), encoding="utf-8"
            )

            join, qa, source, _ = _publication_join(
                output_dir,
                "example_drought",
                self._integrity(),
            )

        self.assertEqual(join.name, "joined_official.parquet")
        self.assertEqual(qa.name, "qa_official_numeric_summary.json")
        self.assertEqual(source, "official_ndmc_vectors")


if __name__ == "__main__":
    unittest.main()
