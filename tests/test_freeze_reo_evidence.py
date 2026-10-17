from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from freeze_reo_evidence import _parquet_hashes, verify_release


class EvidenceReleaseConsistencyTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        output_dir = root / "outputs"
        event_dir = output_dir / "event_a"
        event_dir.mkdir(parents=True)
        joined = pd.DataFrame(
            {
                "h3_index": ["a", "b"],
                "ecs_mean": [1.0, 3.0],
                "benchmark_mean": [2.0, 4.0],
            }
        )
        joined_path = event_dir / "joined_inner.parquet"
        joined.to_parquet(joined_path, index=False)
        integrity = {
            "event": "Event A",
            "pre_year": 2020,
            "post_year": 2021,
            "benchmark": {"type": "hansen_gfc_lossyear", "metric_year": 2021},
            "h3_resolution": 7,
            "total_h3_cells_alphaearth": 2,
            "total_h3_cells_benchmark_valid": 2,
            "total_h3_cells_inner_join": 2,
            "pct_data_lost_vs_alpha": 0.0,
        }
        (event_dir / "integrity_report.json").write_text(
            json.dumps(integrity), encoding="utf-8"
        )
        qa = {
            "n_cells": 2,
            "corr_pearson_benchmark_vs_ecs": 1.0,
            "corr_spearman_benchmark_vs_ecs": 1.0,
        }
        qa_path = event_dir / "qa_numeric_summary.json"
        qa_path.write_text(json.dumps(qa), encoding="utf-8")
        manifest = [
            {
                "event_key": "event_a",
                "integrity_report": integrity,
                "artifacts": {
                    "joined_inner.parquet": True,
                    "integrity_report.json": True,
                    "qa_numeric_summary.json": True,
                },
            }
        ]
        (output_dir / "OUTPUTS_EVENTS_MANIFEST.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "event_key": "event_a",
                    "n_h3_join": 2,
                    "n_joined_rows": 2,
                    "publication_join_artifact": "joined_inner.parquet",
                    "publication_reference_source": "pipeline_benchmark",
                    "pearson_benchmark_ecs": 1.0,
                    "spearman_benchmark_ecs": 1.0,
                    "benchmark_std": joined["benchmark_mean"].std(ddof=1),
                    "ecs_mean_std": joined["ecs_mean"].std(ddof=1),
                    "benchmark_nunique": 2,
                    "qa_file": True,
                    "qa_artifact": "qa_numeric_summary.json",
                }
            ]
        ).to_csv(output_dir / "all_events_metrics_summary.csv", index=False)
        pd.DataFrame([{"event_key": "event_a"}]).to_csv(
            output_dir / "reo_evidence_audit.csv", index=False
        )
        (output_dir / "integrity_summary_all_events.json").write_text(
            json.dumps({"event_a": integrity}), encoding="utf-8"
        )
        return output_dir, qa_path

    def test_consistent_release_passes(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir, _ = self._fixture(Path(directory))
            before = _parquet_hashes(output_dir)

            report = verify_release(
                output_dir,
                before_parquet_hashes=before,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["source_parquets_unchanged_during_freeze"])

    def test_stale_qa_count_is_caught(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir, qa_path = self._fixture(Path(directory))
            before = _parquet_hashes(output_dir)
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            qa["n_cells"] = 1
            qa_path.write_text(json.dumps(qa), encoding="utf-8")

            report = verify_release(
                output_dir,
                before_parquet_hashes=before,
            )

        self.assertFalse(report["passed"])
        self.assertIn(
            "event_a: failed checks qa_artifact_matches_join",
            report["issues"],
        )


if __name__ == "__main__":
    unittest.main()
