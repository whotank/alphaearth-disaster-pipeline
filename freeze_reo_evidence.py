#!/usr/bin/env python3
"""Regenerate and verify the complete discovered REO evidence release.

This command never accepts an event subset.  It snapshots every Parquet under
the output tree, rebuilds the discovered manifest, aggregate metrics, evidence
audit, and integrity summary, then verifies event sets, counts, correlations,
QA values, manifest integrity, official-USDM provenance, and unchanged source
Parquet hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import fire_analysis
import output_events_bundle
import reo_evidence_audit
from usdm_validation import evaluate_official_validation


LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*.parquet"))
    }


def _discovered_events(output_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_dir()
        and (
            (path / "integrity_report.json").is_file()
            or (path / "joined_inner.parquet").is_file()
        )
    )


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers_match(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    a = _optional_number(left)
    b = _optional_number(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def _resolve_recorded_path(repo_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _official_source_hashes_match(
    validation: dict[str, Any],
    *,
    repo_root: Path,
    event_dir: Path,
) -> tuple[bool, list[str]]:
    pairs = [
        ("pre_archive", "pre_archive_sha256", None),
        ("post_archive", "post_archive_sha256", None),
        ("aoi", "aoi_sha256", None),
        (
            None,
            "alphaearth_h3_ecs_sha256",
            event_dir / "alphaearth_h3_ecs.parquet",
        ),
        (
            None,
            "benchmark_h3_mirror_sha256",
            event_dir / "benchmark_h3.parquet",
        ),
        (
            None,
            "usdm_official_h3_sha256",
            event_dir / "usdm_official_h3.parquet",
        ),
        (
            None,
            "joined_official_sha256",
            event_dir / "joined_official.parquet",
        ),
    ]
    failures: list[str] = []
    for path_field, hash_field, fixed_path in pairs:
        path = fixed_path or _resolve_recorded_path(
            repo_root, validation.get(path_field)
        )
        expected_hash = validation.get(hash_field)
        if (
            path is None
            or not path.is_file()
            or not isinstance(expected_hash, str)
            or _sha256(path) != expected_hash
        ):
            failures.append(hash_field)
    return not failures, failures


def verify_release(
    output_dir: Path,
    *,
    before_parquet_hashes: dict[str, str],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    repo_root = output_dir.parent
    discovered = _discovered_events(output_dir)
    manifest = json.loads(
        (output_dir / "OUTPUTS_EVENTS_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_csv(output_dir / "all_events_metrics_summary.csv")
    audit = pd.read_csv(output_dir / "reo_evidence_audit.csv")
    integrity_summary = json.loads(
        (output_dir / "integrity_summary_all_events.json").read_text(
            encoding="utf-8"
        )
    )

    event_sets = {
        "discovered": discovered,
        "manifest": sorted(str(row["event_key"]) for row in manifest),
        "metrics": sorted(metrics["event_key"].astype(str).tolist()),
        "audit": sorted(audit["event_key"].astype(str).tolist()),
        "integrity_summary": sorted(integrity_summary),
    }
    issues: list[str] = []
    expected_set = set(discovered)
    for name, events in event_sets.items():
        if len(events) != len(set(events)):
            issues.append(f"{name}: duplicate event keys")
        if set(events) != expected_set:
            issues.append(
                f"{name}: event set differs from discovered complete set"
            )

    manifest_by_key = {str(row["event_key"]): row for row in manifest}
    metrics_by_key = {
        str(row["event_key"]): row for _, row in metrics.iterrows()
    }
    audit_by_key = {
        str(row["event_key"]): row for _, row in audit.iterrows()
    }
    event_checks: dict[str, dict[str, Any]] = {}
    event_observations: dict[str, dict[str, Any]] = {}

    for event in discovered:
        event_dir = output_dir / event
        integrity_path = event_dir / "integrity_report.json"
        if not integrity_path.is_file():
            issues.append(f"{event}: missing integrity_report.json")
            continue
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        manifest_entry = manifest_by_key.get(event, {})
        metric = metrics_by_key.get(event)
        audit_row = audit_by_key.get(event)
        if metric is None or audit_row is None:
            issues.append(f"{event}: missing aggregate metric or audit row")
            continue

        checks: dict[str, bool] = {}
        checks["manifest_integrity_matches_file"] = (
            manifest_entry.get("integrity_report") == integrity
        )
        checks["integrity_summary_matches_file"] = (
            integrity_summary.get(event) == integrity
        )

        artifacts = manifest_entry.get("artifacts") or {}
        checks["manifest_artifact_flags_match"] = all(
            bool(recorded) == (event_dir / name).is_file()
            for name, recorded in artifacts.items()
        )

        join_name = str(
            metric.get("publication_join_artifact") or "joined_inner.parquet"
        )
        join_path = event_dir / join_name
        if not join_path.is_file():
            issues.append(f"{event}: missing publication join {join_name}")
            event_checks[event] = checks
            continue
        joined = pd.read_parquet(join_path)
        joined["h3_index"] = joined["h3_index"].astype(str)
        checks["joined_h3_unique"] = (
            joined["h3_index"].nunique() == len(joined)
        )
        checks["aggregate_join_count_matches"] = (
            int(metric["n_joined_rows"]) == len(joined)
            and int(metric["n_h3_join"]) == len(joined)
        )
        checks["integrity_join_count_matches"] = (
            int(integrity["total_h3_cells_inner_join"]) == len(joined)
        )

        x = joined["benchmark_mean"].astype(float)
        y = joined["ecs_mean"].astype(float)
        pearson = x.corr(y, method="pearson")
        spearman = x.corr(y, method="spearman")
        checks["aggregate_pearson_matches_join"] = _numbers_match(
            metric.get("pearson_benchmark_ecs"), pearson
        )
        checks["aggregate_spearman_matches_join"] = _numbers_match(
            metric.get("spearman_benchmark_ecs"), spearman
        )
        checks["aggregate_benchmark_std_matches_join"] = _numbers_match(
            metric.get("benchmark_std"), x.std(ddof=1)
        )
        checks["aggregate_ecs_std_matches_join"] = _numbers_match(
            metric.get("ecs_mean_std"), y.std(ddof=1)
        )
        checks["aggregate_benchmark_nunique_matches_join"] = (
            int(metric["benchmark_nunique"]) == int(x.nunique(dropna=True))
        )

        qa_name = str(metric.get("qa_artifact") or "qa_numeric_summary.json")
        qa_path = event_dir / qa_name
        if bool(metric.get("qa_file")):
            if not qa_path.is_file():
                checks["qa_artifact_matches_join"] = False
            else:
                qa = json.loads(qa_path.read_text(encoding="utf-8"))
                checks["qa_artifact_matches_join"] = (
                    int(qa["n_cells"]) == len(joined)
                    and _numbers_match(
                        qa.get("corr_pearson_benchmark_vs_ecs"), pearson
                    )
                    and _numbers_match(
                        qa.get("corr_spearman_benchmark_vs_ecs"), spearman
                    )
                )
        else:
            checks["qa_artifact_matches_join"] = not qa_path.is_file()

        if integrity.get("benchmark", {}).get("type") == "usdm_season_delta":
            validation_path = event_dir / "usdm_official_validation.json"
            if validation_path.is_file():
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                evaluation = evaluate_official_validation(
                    validation,
                    expected_event=event,
                    expected_pre_year=int(integrity["pre_year"]),
                    expected_post_year=int(integrity["post_year"]),
                    expected_h3_cells=int(
                        integrity["total_h3_cells_alphaearth"]
                    ),
                    expected_h3_resolution=int(integrity["h3_resolution"]),
                )
                hashes_match, hash_failures = _official_source_hashes_match(
                    validation,
                    repo_root=repo_root,
                    event_dir=event_dir,
                )
                checks["official_reference_complete"] = bool(
                    evaluation["official_reference_complete"]
                )
                checks["official_source_hashes_match"] = hashes_match
                checks["official_join_selected"] = (
                    metric.get("publication_reference_source")
                    == "official_ndmc_vectors"
                    and join_name == "joined_official.parquet"
                )
                if not evaluation["official_reference_complete"]:
                    expected_status = "official_reference_incomplete"
                elif evaluation["mirror_agreement_passed"]:
                    expected_status = "validated_official_reference"
                else:
                    expected_status = (
                        "official_reference_complete_mirror_disagreement"
                    )
                checks["official_audit_status_matches"] = (
                    audit_row.get("official_validation_status")
                    == expected_status
                )
                event_observations[event] = {
                    "official_mirror_agreement_passed": bool(
                        evaluation["mirror_agreement_passed"]
                    ),
                    "official_mirror_agreement_failures": list(
                        evaluation["agreement_failures"]
                    ),
                }
                if hash_failures:
                    issues.append(
                        f"{event}: official source hash failures "
                        f"{','.join(hash_failures)}"
                    )
            else:
                checks["official_reference_complete"] = False
                checks["official_source_hashes_match"] = False
                checks["official_join_selected"] = False

        failed_checks = sorted(
            name for name, passed in checks.items() if not passed
        )
        if failed_checks:
            issues.append(f"{event}: failed checks {','.join(failed_checks)}")
        event_checks[event] = checks

    after_parquet_hashes = _parquet_hashes(output_dir)
    parquets_unchanged = before_parquet_hashes == after_parquet_hashes
    if not parquets_unchanged:
        issues.append("Source Parquet hashes changed during aggregate freeze")

    return {
        "passed": not issues,
        "scope": "complete_discovered_event_set",
        "event_count": len(discovered),
        "events": discovered,
        "event_sets": event_sets,
        "source_parquets_unchanged_during_freeze": parquets_unchanged,
        "source_parquet_sha256": after_parquet_hashes,
        "event_checks": event_checks,
        "event_observations": event_observations,
        "issues": issues,
    }


def freeze_evidence(output_dir: Path) -> dict[str, Any]:
    """Regenerate every aggregate from the full discovered event set."""
    output_dir = output_dir.resolve()
    before = _parquet_hashes(output_dir)

    output_events_bundle.cmd_manifest(
        SimpleNamespace(output_dir=output_dir, discover=True, events=[])
    )
    fire_analysis.cmd_metrics(output_dir, discover=True)
    reo_evidence_audit.generate_audit(output_dir)

    integrity_summary = {
        event: json.loads(
            (output_dir / event / "integrity_report.json").read_text(
                encoding="utf-8"
            )
        )
        for event in _discovered_events(output_dir)
    }
    integrity_summary_path = output_dir / "integrity_summary_all_events.json"
    integrity_summary_path.write_text(
        json.dumps(integrity_summary, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", integrity_summary_path)

    report = verify_release(
        output_dir,
        before_parquet_hashes=before,
    )
    report_path = output_dir / "evidence_release_consistency.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    if not report["passed"]:
        raise ValueError(
            "Evidence release consistency failed: "
            + " | ".join(report["issues"])
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        report = freeze_evidence(args.output_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    LOGGER.info(
        "Frozen %d discovered events with unchanged source Parquets",
        report["event_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
