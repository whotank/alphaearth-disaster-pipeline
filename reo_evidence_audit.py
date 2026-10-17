#!/usr/bin/env python3
"""Audit event evidence before building the REO-2 paper tables.

The audit combines the generated metrics summary and event manifest. It does
not call Earth Engine. Its purpose is to separate computationally complete
outputs from scientifically limited reference choices, especially the static
JRC surface-water occurrence placeholder currently used by flood presets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from usdm_validation import evaluate_official_validation

LOGGER = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _event_audit(
    metric: pd.Series,
    manifest_entry: dict[str, Any],
    event_dir: Path,
) -> dict[str, Any]:
    preset = manifest_entry.get("pipeline_preset") or {}
    benchmark_config = preset.get("benchmark") or {}
    integrity = manifest_entry.get("integrity_report") or {}
    artifacts = manifest_entry.get("artifacts") or {}

    event_key = str(metric["event_key"])
    benchmark_type = str(metric.get("benchmark_type") or "")
    disaster_type = metric.get("disaster_type")
    if disaster_type is None or pd.isna(disaster_type) or not str(disaster_type):
        disaster_type = preset.get("disaster_type") or integrity.get("disaster_type") or ""

    post_year = _optional_int(metric.get("post_year"))
    metric_year = _optional_int(metric.get("benchmark_metric_year"))
    benchmark_std = _optional_float(metric.get("benchmark_std"))
    benchmark_nunique = _optional_int(metric.get("benchmark_nunique"))
    pct_loss = _optional_float(metric.get("pct_loss_vs_alpha")) or 0.0
    publication_reference_source = str(
        metric.get("publication_reference_source") or "pipeline_benchmark"
    )

    n_metrics_join = _optional_int(metric.get("n_h3_join"))
    n_parquet = _optional_int(metric.get("n_joined_rows"))
    n_integrity = _optional_int(integrity.get("total_h3_cells_inner_join"))
    counts = [n for n in (n_metrics_join, n_parquet, n_integrity) if n is not None]
    counts_consistent = bool(counts) and len(set(counts)) == 1

    qa_name = (
        "qa_official_numeric_summary.json"
        if publication_reference_source == "official_ndmc_vectors"
        else "qa_numeric_summary.json"
    )
    qa_path = event_dir / qa_name
    qa_n_cells: int | None = None
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa_n_cells = _optional_int(qa.get("n_cells"))
    qa_count_consistent = (
        qa_n_cells is None
        or n_metrics_join is not None
        and qa_n_cells == n_metrics_join
    )

    sidecar_path = event_dir / "gis_temporal_alignment.txt"
    expected_sidecar_header = {
        "modis_mcd64a1": "MODIS MCD64A1 (benchmark_mean on hexes):",
        "hansen_gfc_lossyear": (
            "Hansen GFC annual forest loss (benchmark_mean on hexes):"
        ),
        "jrc_gsw_occurrence": (
            "JRC Global Surface Water occurrence (benchmark_mean on hexes):"
        ),
        "geojson_binary": (
            "Event-specific GeoJSON reference (benchmark_mean on hexes):"
        ),
        "usdm_season_delta": (
            "U.S. Drought Monitor seasonal change (benchmark_mean on hexes):"
        ),
    }.get(benchmark_type)
    if not sidecar_path.is_file():
        sidecar_matches_benchmark = None
    elif expected_sidecar_header is None:
        sidecar_matches_benchmark = False
    else:
        sidecar_text = sidecar_path.read_text(encoding="utf-8")
        sidecar_matches_benchmark = expected_sidecar_header in sidecar_text
    supporting_artifacts_consistent = (
        qa_count_consistent and sidecar_matches_benchmark is not False
    )
    reference_support_documented = (
        benchmark_type != "geojson_binary"
        or bool(preset.get("reference_validity_rule"))
        and (
            bool(benchmark_config.get("valid_geojson_path"))
            or bool(benchmark_config.get("exclude_worldcover_water"))
        )
    )
    official_validation_status = "not_required"
    official_validation_failures: list[str] = []
    official_reference_complete = True
    official_mirror_agreement_passed = True
    official_reference_validated = True
    official_validation: dict[str, Any] = {}
    if benchmark_type == "usdm_season_delta":
        official_validation_status = "missing"
        official_reference_complete = False
        official_mirror_agreement_passed = False
        validation_path = event_dir / "usdm_official_validation.json"
        official_reference_validated = False
        if validation_path.is_file():
            try:
                official_validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                evaluation = evaluate_official_validation(
                    official_validation,
                    expected_event=event_key,
                    expected_pre_year=int(metric["pre_year"]),
                    expected_post_year=int(metric["post_year"]),
                    expected_h3_cells=int(
                        integrity["total_h3_cells_alphaearth"]
                    ),
                    expected_h3_resolution=int(integrity["h3_resolution"]),
                )
                official_validation_failures = list(evaluation["failures"])
                official_reference_complete = bool(
                    evaluation["official_reference_complete"]
                )
                official_mirror_agreement_passed = bool(
                    evaluation["mirror_agreement_passed"]
                )
                official_reference_validated = official_reference_complete
                if not official_reference_complete:
                    official_validation_status = "official_reference_incomplete"
                elif official_mirror_agreement_passed:
                    official_validation_status = "validated_official_reference"
                else:
                    official_validation_status = (
                        "official_reference_complete_mirror_disagreement"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                official_validation_failures = [
                    f"invalid_validation_artifact:{type(exc).__name__}"
                ]
                official_validation_status = "invalid_artifact"
        if (
            official_reference_validated
            and publication_reference_source != "official_ndmc_vectors"
        ):
            official_reference_validated = False
            official_validation_status = "validated_but_aggregate_uses_mirror"
            official_validation_failures.append(
                "publication_metrics_not_using_official_join"
            )

    year_matches_post = (
        post_year is not None and metric_year is not None and post_year == metric_year
    )
    temporal_relation = (
        preset.get("benchmark_temporal_relation")
        or integrity.get("benchmark_temporal_relation")
        or (
            "reference_year_matches_post"
            if preset.get("benchmark_year_matches_post", True)
            else "legacy_calendar_mismatch"
        )
    )
    event_is_bracketed = (
        temporal_relation == "event_between_annual_mosaics"
        and post_year is not None
        and metric_year is not None
        and _optional_int(metric.get("pre_year")) is not None
        and _optional_int(metric.get("pre_year")) < metric_year < post_year
    )
    temporal_alignment_valid = year_matches_post or event_is_bracketed
    static_flood_placeholder = benchmark_type == "jrc_gsw_occurrence"
    degenerate_reference = (
        benchmark_nunique is not None and benchmark_nunique <= 1
    ) or (benchmark_std is not None and benchmark_std == 0.0)

    if static_flood_placeholder:
        reference_state = "static_placeholder"
    elif degenerate_reference:
        reference_state = "degenerate"
    else:
        reference_state = "variable"

    required_visual_artifact_present = bool(
        artifacts.get("gis_reference_raster_2x2.png", False)
    ) or (
        benchmark_type == "usdm_season_delta"
        and bool(artifacts.get("joined_official.parquet", False))
        and bool(artifacts.get("qa_official_numeric_summary.json", False))
    )

    if static_flood_placeholder:
        recommendation = "replace_event_reference"
    elif not counts_consistent:
        recommendation = "repair_output_counts"
    elif not supporting_artifacts_consistent:
        recommendation = "regenerate_supporting_artifacts"
    elif not reference_support_documented:
        recommendation = "validate_reference_support"
    elif not official_reference_validated:
        recommendation = "validate_official_reference"
    elif degenerate_reference:
        recommendation = "retain_as_failure_case"
    elif not temporal_alignment_valid:
        recommendation = "retain_with_calendar_caveat"
    elif not required_visual_artifact_present:
        recommendation = "generate_or_validate_gis_figure"
    elif (
        benchmark_type == "usdm_season_delta"
        and not official_mirror_agreement_passed
    ):
        recommendation = "ready_with_mirror_disagreement"
    else:
        recommendation = "ready_for_core_results"

    notes: list[str] = []
    if static_flood_placeholder:
        notes.append(
            "JRC occurrence is a long-term static layer; metric_year does not create "
            "an event-specific flood extent."
        )
    if pct_loss > 0:
        notes.append(f"Inner join loses {pct_loss:.1f}% of AlphaEarth H3 cells.")
    if degenerate_reference:
        notes.append("Reference has no cross-hex variation; correlation is undefined.")
    if not temporal_alignment_valid:
        notes.append(
            f"Reference year {metric_year} differs from ECS post year {post_year}."
        )
    elif event_is_bracketed:
        notes.append(
            f"Event/reference year {metric_year} is bracketed by full-year "
            f"mosaics {_optional_int(metric.get('pre_year'))} and {post_year}."
        )
    if not counts_consistent:
        notes.append(
            "Join counts disagree across metrics, Parquet row count, and integrity report."
        )
    if not qa_count_consistent:
        notes.append(
            f"{qa_name} has {qa_n_cells} cells but the current join has "
            f"{n_metrics_join}; regenerate QA."
        )
    if sidecar_matches_benchmark is False:
        notes.append(
            f"Temporal sidecar does not describe the {benchmark_type} benchmark; "
            "regenerate or correct it."
        )
    if not reference_support_documented:
        notes.append(
            "Positive-only GeoJSON lacks a documented validity or land-support "
            "mask; outside-polygon space cannot yet be treated as negative."
        )
    if benchmark_type == "usdm_season_delta" and official_reference_validated:
        notes.append(
            "Publication metrics use the official NDMC-vector H3 replay; the "
            "Earth Engine mirror is retained only as an implementation cross-check."
        )
        if not official_mirror_agreement_passed:
            failure_text = ", ".join(official_validation_failures)
            notes.append(
                "The official reference is complete, but the mirror cross-check "
                f"fails the fixed agreement criteria: {failure_text}."
            )
    elif benchmark_type == "usdm_season_delta":
        failure_text = ", ".join(official_validation_failures) or "artifact missing"
        notes.append(
            "Official NDMC-vector validation is incomplete or failed: "
            f"{failure_text}."
        )
    if not required_visual_artifact_present:
        notes.append("The required reference QA visualization is not recorded.")

    return {
        "event_key": event_key,
        "disaster_type": disaster_type,
        "benchmark_type": benchmark_type,
        "pre_year": _optional_int(metric.get("pre_year")),
        "post_year": post_year,
        "benchmark_metric_year": metric_year,
        "benchmark_year_matches_post": year_matches_post,
        "benchmark_temporal_relation": temporal_relation,
        "temporal_alignment_valid": temporal_alignment_valid,
        "n_h3_join": n_metrics_join,
        "counts_consistent": counts_consistent,
        "qa_count_consistent": qa_count_consistent,
        "sidecar_matches_benchmark": sidecar_matches_benchmark,
        "pct_loss_vs_alpha": pct_loss,
        "benchmark_nunique": benchmark_nunique,
        "benchmark_std": benchmark_std,
        "pearson": _optional_float(metric.get("pearson_benchmark_ecs")),
        "spearman": _optional_float(metric.get("spearman_benchmark_ecs")),
        "reference_state": reference_state,
        "publication_reference_source": publication_reference_source,
        "has_gis_figure": bool(artifacts.get("gis_reference_raster_2x2.png", False)),
        "required_visual_artifact_present": required_visual_artifact_present,
        "supporting_artifacts_consistent": supporting_artifacts_consistent,
        "reference_support_documented": reference_support_documented,
        "official_reference_validated": official_reference_validated,
        "official_reference_complete": official_reference_complete,
        "official_mirror_agreement_passed": official_mirror_agreement_passed,
        "official_validation_status": official_validation_status,
        "official_validation_failures": ";".join(
            official_validation_failures
        ),
        "official_vs_mirror_pearson": _optional_float(
            official_validation.get("official_vs_mirror_pearson")
        ),
        "official_vs_mirror_spearman": _optional_float(
            official_validation.get("official_vs_mirror_spearman")
        ),
        "official_vs_mirror_mae": _optional_float(
            official_validation.get("official_vs_mirror_mae")
        ),
        "official_vs_mirror_max_abs_error": _optional_float(
            official_validation.get("official_vs_mirror_max_abs_error")
        ),
        "recommendation": recommendation,
        "notes": " ".join(notes),
    }


def _markdown_report(audit: pd.DataFrame) -> str:
    columns = [
        "event_key",
        "disaster_type",
        "benchmark_type",
        "reference_state",
        "publication_reference_source",
        "benchmark_temporal_relation",
        "temporal_alignment_valid",
        "pct_loss_vs_alpha",
        "required_visual_artifact_present",
        "supporting_artifacts_consistent",
        "reference_support_documented",
        "official_validation_status",
        "recommendation",
    ]
    counts = audit["recommendation"].value_counts().sort_index()
    table_rows: list[list[str]] = []
    for _, row in audit[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column == "pct_loss_vs_alpha":
                values.append(f"{float(value):.1f}")
            else:
                values.append(str(value))
        table_rows.append(values)
    markdown_table = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *("| " + " | ".join(values) + " |" for values in table_rows),
    ]
    lines = [
        "# REO evidence audit",
        "",
        "Generated from `all_events_metrics_summary.csv` and "
        "`OUTPUTS_EVENTS_MANIFEST.json`.",
        "",
        "## Recommendation counts",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in counts.items())
    lines.extend(["", "## Event status", "", *markdown_table, ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `static_placeholder`: the current JRC occurrence layer is not an "
            "event-specific flood extent and must not support event-level flood claims.",
            "- `degenerate`: useful as an evaluation failure case, but correlation is "
            "undefined.",
            "- `retain_with_calendar_caveat`: computationally usable, but reference and "
            "ECS post years differ.",
            "- `ready_for_core_results`: counts, temporal pairing, reference variation, "
            "and the required QA visual pass the automated checks.",
            "- `regenerate_supporting_artifacts`: the joined data may be usable, but a "
            "QA summary or temporal sidecar is stale or describes the wrong benchmark.",
            "- `validate_reference_support`: a positive-only event extent needs an "
            "explicit valid observation or land-support mask before interpretation.",
            "- `validate_official_reference`: a convenient data mirror must be "
            "cross-checked against the authoritative publisher archive under the "
            "fixed agreement and H3-coverage criteria in `usdm_validation.py`.",
            "- `ready_with_mirror_disagreement`: the authoritative vector replay, "
            "calendar, hashes, and H3 coverage are complete, so publication metrics "
            "use it; the convenience mirror failed at least one fixed agreement "
            "threshold and is disclosed rather than silently relabelled.",
            "",
            "Automated checks do not replace GIS/domain validation of AOI, event timing, "
            "or reference semantics.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_audit(output_dir: Path) -> pd.DataFrame:
    """Generate CSV/Markdown audit artifacts and return the audited rows."""
    output_dir = output_dir.resolve()
    metrics_path = output_dir / "all_events_metrics_summary.csv"
    manifest_path = output_dir / "OUTPUTS_EVENTS_MANIFEST.json"

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing {metrics_path}; regenerate aggregate metrics first."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; regenerate the discovered manifest first."
        )

    metrics = pd.read_csv(metrics_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metrics["event_key"].duplicated().any():
        raise ValueError("Duplicate event_key rows in aggregate metrics")
    manifest_keys = [str(entry["event_key"]) for entry in manifest]
    if len(manifest_keys) != len(set(manifest_keys)):
        raise ValueError("Duplicate event_key rows in discovered manifest")
    by_key = {entry["event_key"]: entry for entry in manifest}

    metric_keys = set(metrics["event_key"].astype(str))
    manifest_key_set = set(manifest_keys)
    if metric_keys != manifest_key_set:
        raise ValueError(
            "Metrics/manifest event sets differ: "
            f"metrics_only={sorted(metric_keys - manifest_key_set)}, "
            f"manifest_only={sorted(manifest_key_set - metric_keys)}"
        )

    rows = [
        _event_audit(
            metric,
            by_key[str(metric["event_key"])],
            output_dir / str(metric["event_key"]),
        )
        for _, metric in metrics.iterrows()
    ]
    audit = pd.DataFrame(rows).sort_values(
        ["recommendation", "disaster_type", "event_key"]
    )

    csv_path = output_dir / "reo_evidence_audit.csv"
    md_path = output_dir / "reo_evidence_audit.md"
    audit.to_csv(csv_path, index=False)
    md_path.write_text(_markdown_report(audit), encoding="utf-8")
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", md_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        generate_audit(args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
