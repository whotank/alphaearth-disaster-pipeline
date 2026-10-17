#!/usr/bin/env python3
"""Validate regenerated drought joins against the committed evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

BASELINE_COMMIT = "9728c04778f13c4dc80a65ff2f20439129374ff9"
AMENDMENT_PATH = Path("paper/DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL_AMENDMENT_1.json")
EVENTS = (
    "four_corners_drought_2018",
    "north_dakota_drought_2021",
    "iowa_drought_recovery_2024",
)
BASELINE_FILES = (
    "integrity_report.json",
    "qa_official_numeric_summary.json",
    "usdm_official_validation.json",
)
EXCLUDED_KEYS = {
    "alphaearth_h3_ecs_sha256",
    "benchmark_h3_mirror_sha256",
    "usdm_official_h3_sha256",
    "joined_official_sha256",
}
REL_TOLERANCE = 1e-10
ABS_TOLERANCE = 1e-10
REQUIRED_JOIN_COLUMNS = {"h3_index", "ecs_mean", "benchmark_mean"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_json(repo_root: Path, commit: str, relative_path: str) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to read the committed validation baseline")
    completed = subprocess.run(
        [git, "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise TypeError(f"Committed JSON is not an object: {relative_path}")
    return payload


def _relative_error(actual: float, expected: float) -> float:
    scale = max(abs(actual), abs(expected))
    return abs(actual - expected) / scale if scale else 0.0


def _compare_recursive(
    expected: Any,
    actual: Any,
    *,
    path: str,
    failures: list[str],
    numeric_differences: list[dict[str, Any]],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            failures.append(f"{path}: expected {expected!r}, observed {actual!r}")
        return

    if isinstance(expected, int) and isinstance(actual, int):
        if expected != actual:
            failures.append(f"{path}: expected {expected}, observed {actual}")
        return

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        expected_float = float(expected)
        actual_float = float(actual)
        absolute = abs(actual_float - expected_float)
        relative = _relative_error(actual_float, expected_float)
        numeric_differences.append(
            {
                "path": path,
                "expected": expected_float,
                "observed": actual_float,
                "absolute_difference": absolute,
                "relative_difference": relative,
            }
        )
        if not math.isclose(
            actual_float,
            expected_float,
            rel_tol=REL_TOLERANCE,
            abs_tol=ABS_TOLERANCE,
        ):
            failures.append(
                f"{path}: expected {expected_float!r}, observed "
                f"{actual_float!r}, abs_diff={absolute:.3g}, "
                f"rel_diff={relative:.3g}"
            )
        return

    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected) - EXCLUDED_KEYS
        actual_keys = set(actual) - EXCLUDED_KEYS
        if expected_keys != actual_keys:
            failures.append(
                f"{path}: key mismatch; missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys & actual_keys):
            _compare_recursive(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                failures=failures,
                numeric_differences=numeric_differences,
            )
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            failures.append(
                f"{path}: expected list length {len(expected)}, observed {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=False)
        ):
            _compare_recursive(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                failures=failures,
                numeric_differences=numeric_differences,
            )
        return

    if expected != actual:
        failures.append(f"{path}: expected {expected!r}, observed {actual!r}")


def _summarize_numeric_differences(
    differences: list[dict[str, Any]],
) -> dict[str, Any]:
    if not differences:
        return {
            "numeric_values_compared": 0,
            "maximum_absolute_difference": None,
            "maximum_relative_difference": None,
            "maximum_absolute_difference_path": None,
            "maximum_relative_difference_path": None,
        }
    maximum_absolute = max(
        differences,
        key=lambda result: result["absolute_difference"],
    )
    maximum_relative = max(
        differences,
        key=lambda result: result["relative_difference"],
    )
    return {
        "numeric_values_compared": len(differences),
        "maximum_absolute_difference": maximum_absolute["absolute_difference"],
        "maximum_relative_difference": maximum_relative["relative_difference"],
        "maximum_absolute_difference_path": maximum_absolute["path"],
        "maximum_relative_difference_path": maximum_relative["path"],
    }


def _validate_join_structure(
    event_dir: Path,
    *,
    expected_cells: int,
    expected_resolution: int,
) -> dict[str, Any]:
    paths = {
        "alphaearth_h3_ecs": event_dir / "alphaearth_h3_ecs.parquet",
        "benchmark_h3_mirror": event_dir / "benchmark_h3.parquet",
        "usdm_official_h3": event_dir / "usdm_official_h3.parquet",
        "joined_official": event_dir / "joined_official.parquet",
    }
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        frames[label] = pd.read_parquet(path)

    if "joined_official" in frames:
        missing_columns = REQUIRED_JOIN_COLUMNS - set(frames["joined_official"].columns)
        if missing_columns:
            failures.append(
                f"joined_official missing columns {sorted(missing_columns)}"
            )
        else:
            joined = frames["joined_official"]
            if not (
                np.isfinite(joined["ecs_mean"].to_numpy(dtype=float)).all()
                and np.isfinite(joined["benchmark_mean"].to_numpy(dtype=float)).all()
            ):
                failures.append("joined_official has nonfinite analysis values")

    h3_sets: dict[str, set[str]] = {}
    for label, frame in frames.items():
        if "h3_index" not in frame:
            failures.append(f"{label} missing h3_index")
            continue
        indices = frame["h3_index"].astype(str)
        if indices.duplicated().any():
            failures.append(f"{label} contains duplicate H3 cells")
        if len(indices) != expected_cells:
            failures.append(
                f"{label} expected {expected_cells} rows, observed {len(indices)}"
            )
        invalid = [index for index in indices if not h3.is_valid_cell(index)]
        if invalid:
            failures.append(f"{label} contains invalid H3 cells")
        wrong_resolution = [
            index
            for index in indices
            if h3.is_valid_cell(index)
            and h3.get_resolution(index) != expected_resolution
        ]
        if wrong_resolution:
            failures.append(
                f"{label} contains H3 cells outside resolution {expected_resolution}"
            )
        h3_sets[label] = set(indices)
    if h3_sets and len({frozenset(indices) for indices in h3_sets.values()}) != 1:
        failures.append("regenerated artifact H3 sets are not identical")

    return {
        "passed": not failures,
        "failures": failures,
        "expected_cells": expected_cells,
        "expected_resolution": expected_resolution,
        "observed_rows": {label: len(frame) for label, frame in frames.items()},
        "all_h3_sets_equal": bool(h3_sets)
        and len({frozenset(indices) for indices in h3_sets.values()}) == 1,
        "artifact_sha256": {
            label: sha256_file(path) for label, path in paths.items() if path.is_file()
        },
    }


def validate_event(
    event_key: str,
    *,
    repo_root: Path,
    evidence_registry: dict[str, Any],
) -> dict[str, Any]:
    event_dir = repo_root / "outputs" / event_key
    failures: list[str] = []
    file_comparisons: dict[str, Any] = {}
    baseline_validation = _git_json(
        repo_root,
        BASELINE_COMMIT,
        f"outputs/{event_key}/usdm_official_validation.json",
    )

    for filename in BASELINE_FILES:
        relative_path = f"outputs/{event_key}/{filename}"
        current_path = repo_root / relative_path
        if not current_path.is_file():
            failures.append(f"missing current audit file {relative_path}")
            continue
        expected = _git_json(repo_root, BASELINE_COMMIT, relative_path)
        actual = json.loads(current_path.read_text(encoding="utf-8"))
        comparison_failures: list[str] = []
        differences: list[dict[str, Any]] = []
        _compare_recursive(
            expected,
            actual,
            path=filename,
            failures=comparison_failures,
            numeric_differences=differences,
        )
        failures.extend(comparison_failures)
        file_comparisons[filename] = {
            "passed": not comparison_failures,
            "failures": comparison_failures,
            **_summarize_numeric_differences(differences),
        }

    source_hash_checks: dict[str, Any] = {}
    for path_key, hash_key in (
        ("pre_archive", "pre_archive_sha256"),
        ("post_archive", "post_archive_sha256"),
        ("aoi", "aoi_sha256"),
    ):
        path = repo_root / str(baseline_validation[path_key])
        observed = sha256_file(path) if path.is_file() else None
        expected = str(baseline_validation[hash_key])
        passed = observed == expected
        source_hash_checks[path_key] = {
            "path": str(path.relative_to(repo_root)),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{path_key} source SHA-256 mismatch")

    structure = _validate_join_structure(
        event_dir,
        expected_cells=int(baseline_validation["h3_cells"]),
        expected_resolution=int(baseline_validation["h3_coverage"]["resolution"]),
    )
    failures.extend(structure["failures"])

    old_hashes = {
        relative_path.split("/", maxsplit=1)[1]: value
        for relative_path, value in evidence_registry["source_parquet_sha256"].items()
        if relative_path.startswith(f"{event_key}/")
        and relative_path.endswith(".parquet")
    }
    return {
        "event_key": event_key,
        "passed": not failures,
        "failures": failures,
        "file_comparisons": file_comparisons,
        "source_hash_checks": source_hash_checks,
        "structure": structure,
        "original_recorded_parquet_sha256": old_hashes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reo2_development_spatial_robustness/"
            "drought_regeneration_validation.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = args.repo_root.resolve()
    amendment_path = repo_root / AMENDMENT_PATH
    amendment_hash = sha256_file(amendment_path)
    evidence_registry = _git_json(
        repo_root,
        BASELINE_COMMIT,
        "outputs/evidence_release_consistency.json",
    )

    event_results = [
        validate_event(
            event_key,
            repo_root=repo_root,
            evidence_registry=evidence_registry,
        )
        for event_key in EVENTS
    ]
    payload = {
        "validation_id": "reo2-drought-regeneration-validation-v1",
        "baseline_commit": BASELINE_COMMIT,
        "amendment_path": str(AMENDMENT_PATH),
        "amendment_sha256": amendment_hash,
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": REL_TOLERANCE,
        "excluded_byte_hash_fields": sorted(EXCLUDED_KEYS),
        "all_events_passed": all(result["passed"] for result in event_results),
        "event_results": event_results,
    }
    output_path = (
        args.output.resolve()
        if args.output.is_absolute()
        else (repo_root / args.output).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", output_path)
    return 0 if payload["all_events_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
