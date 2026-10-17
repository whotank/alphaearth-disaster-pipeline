from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "mtbs_heldout_20260817"
sys.path.insert(0, str(STUDY))

import analysis_stage
import reference_stage
import study_core
import study_sources


class FreezeIntegrityTests(unittest.TestCase):
    def test_canonical_freeze_has_exact_hash_and_no_newline(self) -> None:
        data = study_core.FREEZE_PATH.read_bytes()
        self.assertFalse(data.endswith(b"\n"))
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            study_core.EXPECTED_FREEZE_SHA256,
        )
        self.assertEqual(study_core.canonical_json_bytes(json.loads(data)), data)

    def test_frozen_candidates_are_unique_and_expected(self) -> None:
        freeze = json.loads(study_core.FREEZE_PATH.read_text(encoding="utf-8"))
        candidates = [*freeze["primaries"], *freeze["reserves"]]
        fire_ids = [item["fire_id"] for item in candidates]
        self.assertEqual(len(fire_ids), 20)
        self.assertEqual(len(set(fire_ids)), 20)
        self.assertEqual(
            [item["name"] for item in freeze["primaries"]],
            [
                "RHEA",
                "WOOLSEY",
                "HADWEENZIC RIVER",
                "BEACHIE CREEK",
                "SARPY",
                "SHOVEL CREEK",
                "FOUR COUNTY",
                "BLACK",
                "DOUBLE CREEK",
                "YORK",
            ],
        )

    def test_protocol_has_frozen_feature_and_inference_rules(self) -> None:
        protocol = json.loads(study_core.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["alphaearth"]["batch_size_h3"], 40)
        self.assertEqual(protocol["alphaearth"]["scale_m"], 10)
        self.assertEqual(protocol["h3_population"]["resolution"], 8)
        self.assertEqual(protocol["h3_population"]["minimum"], 200)
        self.assertEqual(protocol["analysis"]["bootstrap_seed"], 20260817)
        self.assertEqual(protocol["sentinel2"]["dnbr"], "NBRpre-NBRpost")
        self.assertEqual(
            protocol["sentinel2"]["dndvi_loss"], "NDVIpre-NDVIpost"
        )


class ProgressCheckpointTests(unittest.TestCase):
    def test_progress_records_are_durable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            with mock.patch.object(study_core, "PROGRESS_LOG_PATH", path):
                study_core.append_progress(
                    "features", "completed", {"sealed_events": 10}
                )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["stage"], "features")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["details"]["sealed_events"], 10)


class PopulationTests(unittest.TestCase):
    def test_deterministic_cap_uses_frozen_hash_rule(self) -> None:
        cells = [f"cell-{index:04d}" for index in range(30)]
        first = study_core.deterministic_h3_cap("FIRE", cells, cap=7)
        second = study_core.deterministic_h3_cap("FIRE", list(reversed(cells)), cap=7)
        expected = sorted(
            sorted(
                cells,
                key=lambda value: (
                    hashlib.sha256(
                        f"{study_core.H3_CAP_SALT}|FIRE|{value}".encode()
                    ).hexdigest(),
                    value,
                ),
            )[:7]
        )
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_duplicate_h3_is_rejected_before_cap(self) -> None:
        with self.assertRaisesRegex(study_core.StudyError, "duplicate H3"):
            study_core.deterministic_h3_cap("FIRE", ["a", "a"], cap=1)

    def test_h3_centers_are_covered_by_boundary(self) -> None:
        polygon = Polygon(
            [
                (-120.5, 38.5),
                (-120.0, 38.5),
                (-120.0, 39.0),
                (-120.5, 39.0),
                (-120.5, 38.5),
            ]
        )
        cells = study_core.geometry_to_h3(polygon, resolution=8)
        self.assertGreater(len(cells), 0)
        for cell in cells:
            latitude, longitude = study_core.h3.cell_to_latlng(cell)
            self.assertTrue(
                polygon.covers(study_core.Point(longitude, latitude))
            )

    def test_exact_h3_response_is_reordered_and_validated(self) -> None:
        frame = pd.DataFrame(
            {"h3_index": ["c", "a", "b"], "value": [3, 1, 2]}
        )
        output = study_core.validate_h3_frame(
            frame, ["a", "b", "c"], context="test"
        )
        self.assertEqual(output["h3_index"].tolist(), ["a", "b", "c"])
        with self.assertRaisesRegex(study_core.StudyError, "H3 mismatch"):
            study_core.validate_h3_frame(
                frame.iloc[:2], ["a", "b", "c"], context="test"
            )


class FormulaAndInferenceTests(unittest.TestCase):
    def test_spearman_uses_average_ties(self) -> None:
        x = [1, 1, 2, 3]
        y = [4, 3, 2, 1]
        expected = float(
            np.corrcoef(
                pd.Series(x).rank(method="average"),
                pd.Series(y).rank(method="average"),
            )[0, 1]
        )
        self.assertAlmostEqual(analysis_stage.spearman(x, y), expected)

    def test_event_endpoint_uses_best_optical_baseline(self) -> None:
        frame = pd.DataFrame(
            {
                "mtbs_reference": [0, 1, 2, 3, 4],
                "ecs": [0, 2, 1, 3, 4],
                "dnbr": [0, 1, 2, 4, 3],
                "dndvi_loss": [4, 3, 2, 1, 0],
            }
        )
        result = analysis_stage.event_metrics(frame)
        expected = result["rho_ecs_reference"] - max(
            result["rho_dnbr_reference"],
            result["rho_dndvi_loss_reference"],
        )
        self.assertAlmostEqual(result["D_e"], expected)

    def test_exact_randomization_uses_all_event_sign_flips(self) -> None:
        result = analysis_stage.exact_sign_randomization(
            np.asarray([0.1, -0.2, 0.3])
        )
        self.assertEqual(result["event_sign_flips"], 8)
        self.assertFalse(result["primary_inference_changed"])

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.asarray([-0.1, 0.2, 0.4, 0.0])
        first = analysis_stage.bootstrap_mean(values)
        second = analysis_stage.bootstrap_mean(values)
        self.assertEqual(first["ci_95"], second["ci_95"])
        np.testing.assert_array_equal(first["distribution"], second["distribution"])


class SourceAndLeakageTests(unittest.TestCase):
    def test_metadata_date_parser_supports_epoch_and_compact_date(self) -> None:
        self.assertEqual(
            study_sources.value_as_date("20180412").isoformat(), "2018-04-12"
        )
        self.assertEqual(
            study_sources.value_as_date(20180412).isoformat(), "2018-04-12"
        )
        self.assertEqual(
            study_sources.value_as_date(1523491200000).isoformat(), "2018-04-12"
        )

    def test_reference_support_policy_is_fixed(self) -> None:
        self.assertEqual(reference_stage.MIN_MTBS_VALID_FRACTION, 0.80)
        self.assertEqual(reference_stage.MIN_FINAL_H3, 200)
        self.assertEqual(reference_stage.MAX_FINAL_H3, 2000)
        self.assertEqual(reference_stage.MIN_S2_VALID_FRACTION, 0.50)

    def test_study_modules_do_not_import_existing_outcome_modules(self) -> None:
        forbidden = {
            "disaster_embedding_pipeline",
            "fire_analysis",
            "sentinel2_optical_baseline",
            "alphaearth_dimension_study",
            "reo_evidence_audit",
        }
        for path in STUDY.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(
                imported & forbidden,
                f"{path.name} imports {sorted(imported & forbidden)}",
            )

    def test_feature_stage_contains_no_correlation_call(self) -> None:
        tree = ast.parse(
            (STUDY / "features_stage.py").read_text(encoding="utf-8")
        )
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse({"corr", "corrcoef", "spearmanr"} & called)

    def test_secret_scanner_flags_private_key(self) -> None:
        marker = "-----BEGIN " + "PRIVATE KEY-----"
        self.assertTrue(
            analysis_stage.re_search_secret(
                f"{marker}\nnot-a-real-key"
            )
        )
        self.assertFalse(analysis_stage.re_search_secret("ordinary compact report"))


if __name__ == "__main__":
    unittest.main()
