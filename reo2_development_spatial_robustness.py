#!/usr/bin/env python3
"""Frozen post hoc spatial robustness analysis for eight development cases.

The protocol is defined in
``paper/DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL.json``. This script validates
every publication join against either its frozen SHA-256 or the uniformly
validated drought-regeneration amendment before computing any block outcome.
It reuses the established coarse-H3 block construction and block-label
permutation machinery without reading any invalidated dimension-correlation
artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reo2_spatial_block_null import (
    EventIneligibleError,
    block_permutation_test,
    build_parent_blocks,
    exact_two_sided_sign_test,
    sha256_file,
    stable_event_seed,
    write_json,
)

LOGGER = logging.getLogger(__name__)

ANALYSIS_VERSION = "1.0.0"
DEFAULT_PROTOCOL = Path("paper/DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL.json")
DEFAULT_AMENDMENT = Path(
    "paper/DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL_AMENDMENT_1.json"
)
DEFAULT_DROUGHT_VALIDATION = Path(
    "outputs/reo2_development_spatial_robustness/drought_regeneration_validation.json"
)
EXPECTED_PROTOCOL_ID = "reo2-development-spatial-robustness-v1"
EXPECTED_EVENT_COUNT = 8
EXPECTED_PARENT_LEVELS = (1, 2, 3)
PRIMARY_PARENT_LEVELS = 2


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""

    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError("Holm p-values must be finite values in [0, 1]")

    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float((total - rank) * values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def scale_min_cells(parent_levels: int) -> int:
    """Preserve the frozen one-seventh complete-parent support threshold."""

    if parent_levels < 1:
        raise ValueError("parent_levels must be positive")
    return math.ceil((7**parent_levels) / 7)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            f"Expected protocol_id {EXPECTED_PROTOCOL_ID!r}, "
            f"found {protocol.get('protocol_id')!r}"
        )
    events = protocol.get("fixed_events")
    if not isinstance(events, list) or len(events) != EXPECTED_EVENT_COUNT:
        raise ValueError(f"Protocol must contain exactly {EXPECTED_EVENT_COUNT} events")
    event_keys = [event.get("event_key") for event in events]
    if any(not isinstance(key, str) or not key for key in event_keys):
        raise ValueError("Every fixed event requires a nonempty event_key")
    if len(set(event_keys)) != len(event_keys):
        raise ValueError("Protocol contains duplicate event keys")

    primary = protocol.get("primary_analysis") or {}
    if int(primary.get("parent_levels", -1)) != PRIMARY_PARENT_LEVELS:
        raise ValueError("Protocol primary parent level differs from implementation")
    if int(primary.get("iterations", -1)) < 999:
        raise ValueError("Protocol requires at least 999 permutations")
    if int(primary.get("minimum_retained_parents", -1)) < 2:
        raise ValueError("Protocol minimum retained parents is invalid")

    sensitivity = protocol.get("scale_sensitivity") or {}
    levels = tuple(int(level) for level in sensitivity.get("fixed_parent_levels", []))
    if levels != EXPECTED_PARENT_LEVELS:
        raise ValueError(
            f"Expected fixed parent levels {EXPECTED_PARENT_LEVELS}, found {levels}"
        )
    return protocol


def _preflight_inputs(
    protocol: dict[str, Any],
    *,
    repo_root: Path,
    drought_validation_by_event: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate all eight artifacts and metadata before any outcome is computed."""

    records: list[dict[str, Any]] = []
    forbidden = [
        (repo_root / relative_path).resolve()
        for relative_path in protocol.get("forbidden_inputs", [])
    ]

    for frozen_event in protocol["fixed_events"]:
        event_key = str(frozen_event["event_key"])
        joined_path = (repo_root / str(frozen_event["join_artifact"])).resolve()
        if any(
            joined_path == path or path in joined_path.parents for path in forbidden
        ):
            raise ValueError(f"Forbidden input path for {event_key}: {joined_path}")
        if not joined_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen join for {event_key}: {joined_path}"
            )

        expected_hash = str(frozen_event["expected_sha256"])
        observed_hash = sha256_file(joined_path)
        input_validation_mode = "original_exact_sha256"
        if observed_hash != expected_hash:
            validation = drought_validation_by_event.get(event_key)
            validated_hash = (
                ((validation or {}).get("structure") or {})
                .get("artifact_sha256", {})
                .get("joined_official")
            )
            if (
                frozen_event.get("family") != "drought"
                or not validation
                or validation.get("passed") is not True
                or validated_hash != observed_hash
            ):
                raise RuntimeError(
                    f"Frozen join SHA-256 mismatch for {event_key}: "
                    f"expected {expected_hash}, observed {observed_hash}; no "
                    "matching passed drought-regeneration validation was found."
                )
            input_validation_mode = "amendment_1_validated_regeneration"

        metadata_path = joined_path.parent / "integrity_report.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing integrity metadata for {event_key}: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        h3_resolution = metadata.get("h3_resolution")
        if not isinstance(h3_resolution, int):
            raise TypeError(
                f"Integrity metadata for {event_key} lacks integer h3_resolution"
            )
        records.append(
            {
                **frozen_event,
                "joined_path": joined_path,
                "observed_sha256": observed_hash,
                "input_validation_mode": input_validation_mode,
                "input_size_bytes": joined_path.stat().st_size,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "h3_resolution": h3_resolution,
            }
        )
    return records


def _load_drought_validation(
    validation_path: Path,
    *,
    amendment_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Missing mandatory drought-regeneration validation: {validation_path}"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("all_events_passed") is not True:
        raise RuntimeError("Drought-regeneration validation did not pass all events")
    expected_amendment_hash = sha256_file(amendment_path)
    if validation.get("amendment_sha256") != expected_amendment_hash:
        raise RuntimeError("Drought-regeneration validation amendment SHA-256 mismatch")
    event_results = validation.get("event_results")
    if not isinstance(event_results, list):
        raise TypeError("Drought validation event_results must be a list")
    by_event = {
        str(result["event_key"]): result
        for result in event_results
        if isinstance(result, dict) and result.get("event_key")
    }
    drought_events = {
        "four_corners_drought_2018",
        "north_dakota_drought_2021",
        "iowa_drought_recovery_2024",
    }
    if set(by_event) != drought_events:
        raise RuntimeError(
            "Drought-regeneration validation does not contain the exact three-event set"
        )
    if any(result.get("passed") is not True for result in by_event.values()):
        raise RuntimeError("At least one drought event failed regeneration validation")
    return validation, by_event


def _event_seed(master_seed: int, event_key: str, parent_levels: int) -> int:
    seed_key = (
        event_key
        if parent_levels == PRIMARY_PARENT_LEVELS
        else f"{event_key}:parent_levels={parent_levels}"
    )
    return stable_event_seed(master_seed, seed_key)


def _analyze_scale(
    frame: pd.DataFrame,
    *,
    event_key: str,
    h3_resolution: int,
    parent_levels: int,
    min_blocks: int,
    iterations: int,
    master_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    min_cells = scale_min_cells(parent_levels)
    try:
        all_blocks, diagnostics = build_parent_blocks(
            frame,
            h3_resolution=h3_resolution,
            parent_levels=parent_levels,
            min_cells_per_block=min_cells,
        )
    except EventIneligibleError as error:
        return (
            {
                "status": "excluded",
                "exclusion_reason": error.code,
                "exclusion_detail": str(error),
                "parent_levels": parent_levels,
                "minimum_finite_cells_per_parent": min_cells,
            },
            pd.DataFrame(),
        )

    result: dict[str, Any] = {
        **diagnostics,
        "status": "excluded",
        "exclusion_reason": None,
        "parent_levels": parent_levels,
        "minimum_finite_cells_per_parent": min_cells,
    }
    retained = all_blocks.loc[all_blocks["retained"]].reset_index(drop=True)
    if len(retained) < min_blocks:
        result["exclusion_reason"] = "too_few_retained_parent_blocks"
        result["exclusion_detail"] = (
            f"{len(retained)} retained blocks; frozen minimum is {min_blocks}"
        )
        return result, all_blocks

    seed = _event_seed(master_seed, event_key, parent_levels)
    test_summary, _ = block_permutation_test(
        retained,
        iterations=iterations,
        seed=seed,
    )
    result.update(test_summary)
    result["event_seed"] = seed
    return result, all_blocks


def _direction_status(scales: dict[str, dict[str, Any]]) -> str:
    eligible = [
        result for result in scales.values() if result.get("status") == "eligible"
    ]
    if len(eligible) != len(EXPECTED_PARENT_LEVELS):
        if eligible and all(
            float(result["observed_block_spearman"]) > 0.0 for result in eligible
        ):
            return "positive_at_all_assessable_scales_but_incomplete"
        return "not_fully_assessable"
    if all(float(result["observed_block_spearman"]) > 0.0 for result in eligible):
        return "directionally_stable"
    return "scale_dependent_or_nonpositive"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _make_report(
    *,
    protocol: dict[str, Any],
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    protocol_hash: str,
    implementation_hash: str,
) -> str:
    lines = [
        "# Eight-case development spatial robustness analysis",
        "",
        "## Status",
        "",
        (
            "This is a frozen, explicitly post hoc spatial sensitivity analysis. "
            "It is conditional on these eight development-case maps and does not "
            "constitute held-out validation or population-level inference."
        ),
        "",
        f"- Protocol: `{protocol['protocol_id']}`",
        f"- Protocol SHA-256: `{protocol_hash}`",
        f"- Implementation SHA-256: `{implementation_hash}`",
        f"- Primary scale: H3 parent levels `{PRIMARY_PARENT_LEVELS}` coarser",
        (
            f"- Permutations per eligible event-scale: "
            f"`{protocol['primary_analysis']['iterations']:,}`"
        ),
        (
            "- Input validation: original frozen SHA-256 for lava and forest-loss "
            "joins; Amendment 1 committed-audit validation for regenerated "
            "drought joins"
        ),
        "",
        "## Results",
        "",
        (
            "| event | family | fine-cell rho | L1 block rho | L2 block rho | "
            "L2 raw p | L2 Holm p | L3 block rho | direction |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        scales = result["scales"]
        primary = scales[str(PRIMARY_PARENT_LEVELS)]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["event_key"],
                    result["family"],
                    _format_float(result["fine_cell_spearman_descriptive_only"]),
                    _format_float(
                        scales["1"].get("observed_block_spearman")
                        if scales["1"].get("status") == "eligible"
                        else None
                    ),
                    _format_float(
                        primary.get("observed_block_spearman")
                        if primary.get("status") == "eligible"
                        else None
                    ),
                    _format_float(primary.get("monte_carlo_p_two_sided"), 4),
                    _format_float(result.get("holm_adjusted_p_primary_eight_tests"), 4),
                    _format_float(
                        scales["3"].get("observed_block_spearman")
                        if scales["3"].get("status") == "eligible"
                        else None
                    ),
                    result["direction_status"],
                ]
            )
            + " |"
        )

    sign = aggregate["primary_block_sign_summary"]
    lines.extend(
        [
            "",
            "## Fixed-case aggregate",
            "",
            (
                f"- Primary block correlations assessable: "
                f"{aggregate['n_primary_assessable']}/{EXPECTED_EVENT_COUNT}"
            ),
            f"- Positive / negative: `{sign['n_positive']} / {sign['n_negative']}`",
            (
                f"- Equal-event median primary block rho: "
                f"`{_format_float(sign['median_block_spearman_equal_event_weight'])}`"
            ),
            (
                f"- Exact two-sided sign-test p: "
                f"`{_format_float(sign['exact_sign_test_p_two_sided'], 4)}`"
            ),
            (
                f"- Holm-supported positive primary results: "
                f"`{aggregate['n_permutation_supported_positive']}`"
            ),
            (
                "- Spatially robust within case under the frozen classification: "
                f"`{aggregate['n_spatially_robust_within_case']}`"
            ),
            "",
            (
                "The sign test is a descriptive summary of these fixed cases. It "
                "does not make the event collection a probability sample."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "- Parent-block aggregation reduces fine-cell pseudoreplication but "
                "does not guarantee exchangeability across broad spatial gradients."
            ),
            (
                "- The reported 2.5th and 97.5th permutation quantiles are null "
                "envelopes, not confidence intervals for the observed correlation."
            ),
            (
                "- Scale sensitivity is descriptive. No favorable scale replaces "
                "the parent-level-2 primary result."
            ),
            "- A missing or ineligible scale is not counted as a robustness pass.",
            "- Every weak, negative, scale-dependent, or ineligible result is retained.",
            (
                "- Drought joins are validated reconstructions, not byte-identical "
                "recoveries of the missing original Parquet files."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument(
        "--drought-validation",
        type=Path,
        default=DEFAULT_DROUGHT_VALIDATION,
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/reo2_development_spatial_robustness"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo_root = args.repo_root.resolve()
    protocol_path = (
        args.protocol.resolve()
        if args.protocol.is_absolute()
        else (repo_root / args.protocol).resolve()
    )
    amendment_path = (
        args.amendment.resolve()
        if args.amendment.is_absolute()
        else (repo_root / args.amendment).resolve()
    )
    drought_validation_path = (
        args.drought_validation.resolve()
        if args.drought_validation.is_absolute()
        else (repo_root / args.drought_validation).resolve()
    )
    analysis_dir = (
        args.analysis_dir.resolve()
        if args.analysis_dir.is_absolute()
        else (repo_root / args.analysis_dir).resolve()
    )
    if analysis_dir.exists():
        existing_paths = {path.resolve() for path in analysis_dir.iterdir()}
        allowed_paths = {drought_validation_path}
        unexpected_paths = existing_paths - allowed_paths
        if unexpected_paths:
            raise FileExistsError(
                f"Analysis directory contains prior outcomes: "
                f"{sorted(str(path) for path in unexpected_paths)}. "
                "Use a new directory; do not overwrite outcomes."
            )

    protocol = _load_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    amendment_hash = sha256_file(amendment_path)
    implementation_hash = sha256_file(Path(__file__).resolve())
    drought_validation, drought_validation_by_event = _load_drought_validation(
        drought_validation_path,
        amendment_path=amendment_path,
    )
    drought_validation_hash = sha256_file(drought_validation_path)

    # Preflight is intentionally complete before any spatial result is computed.
    records = _preflight_inputs(
        protocol,
        repo_root=repo_root,
        drought_validation_by_event=drought_validation_by_event,
    )

    primary = protocol["primary_analysis"]
    iterations = int(primary["iterations"])
    master_seed = int(primary["master_seed"])
    min_blocks = int(primary["minimum_retained_parents"])

    analysis_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = analysis_dir / "blocks"
    per_event_dir = analysis_dir / "per_event"
    blocks_dir.mkdir()
    per_event_dir.mkdir()

    results: list[dict[str, Any]] = []
    for record in records:
        event_key = str(record["event_key"])
        LOGGER.info("Analyzing %s", event_key)
        frame = pd.read_parquet(record["joined_path"])
        scale_results: dict[str, dict[str, Any]] = {}
        for parent_levels in EXPECTED_PARENT_LEVELS:
            scale_result, block_table = _analyze_scale(
                frame,
                event_key=event_key,
                h3_resolution=int(record["h3_resolution"]),
                parent_levels=parent_levels,
                min_blocks=min_blocks,
                iterations=iterations,
                master_seed=master_seed,
            )
            block_path = blocks_dir / f"{event_key}_parent_levels_{parent_levels}.csv"
            block_table.to_csv(block_path, index=False)
            scale_result["derived_block_table"] = str(block_path.relative_to(repo_root))
            scale_results[str(parent_levels)] = scale_result

        fine_rho = scale_results[str(PRIMARY_PARENT_LEVELS)].get(
            "cell_level_spearman_descriptive_only"
        )
        result = {
            "analysis_version": ANALYSIS_VERSION,
            "protocol_id": protocol["protocol_id"],
            "event_key": event_key,
            "event_label": record["metadata"].get("event"),
            "family": record["family"],
            "disaster_type": record["metadata"].get("disaster_type"),
            "join_artifact": record["join_artifact"],
            "input_sha256": record["observed_sha256"],
            "input_validation_mode": record["input_validation_mode"],
            "input_size_bytes": record["input_size_bytes"],
            "h3_resolution": record["h3_resolution"],
            "fine_cell_spearman_descriptive_only": fine_rho,
            "scales": scale_results,
            "direction_status": _direction_status(scale_results),
        }
        results.append(result)

    # Ineligible primary tests count as p=1 so the frozen family remains eight.
    primary_p_for_holm = []
    for result in results:
        primary_result = result["scales"][str(PRIMARY_PARENT_LEVELS)]
        raw_p = (
            float(primary_result["monte_carlo_p_two_sided"])
            if primary_result.get("status") == "eligible"
            else 1.0
        )
        primary_p_for_holm.append(raw_p)
    adjusted = holm_adjust(primary_p_for_holm)

    for result, adjusted_p in zip(results, adjusted, strict=True):
        primary_result = result["scales"][str(PRIMARY_PARENT_LEVELS)]
        if primary_result.get("status") == "eligible":
            result["holm_adjusted_p_primary_eight_tests"] = adjusted_p
            result["permutation_supported_positive"] = bool(
                float(primary_result["observed_block_spearman"]) > 0.0
                and adjusted_p < 0.05
            )
        else:
            result["holm_adjusted_p_primary_eight_tests"] = None
            result["permutation_supported_positive"] = False
        result["spatially_robust_within_case"] = bool(
            result["permutation_supported_positive"]
            and result["direction_status"] == "directionally_stable"
        )
        write_json(per_event_dir / f"{result['event_key']}.json", result)

    primary_effects = [
        float(result["scales"][str(PRIMARY_PARENT_LEVELS)]["observed_block_spearman"])
        for result in results
        if result["scales"][str(PRIMARY_PARENT_LEVELS)].get("status") == "eligible"
    ]
    aggregate = {
        "analysis_version": ANALYSIS_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "amendment_sha256": amendment_hash,
        "drought_regeneration_validation_sha256": drought_validation_hash,
        "implementation_sha256": implementation_hash,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "all_inputs_validated": True,
        "all_inputs_original_exact_sha256_match": all(
            result["input_validation_mode"] == "original_exact_sha256"
            for result in results
        ),
        "drought_regeneration_validation_id": drought_validation.get("validation_id"),
        "n_fixed_events": EXPECTED_EVENT_COUNT,
        "n_primary_assessable": len(primary_effects),
        "primary_block_sign_summary": exact_two_sided_sign_test(primary_effects),
        "n_permutation_supported_positive": int(
            sum(bool(result["permutation_supported_positive"]) for result in results)
        ),
        "n_spatially_robust_within_case": int(
            sum(bool(result["spatially_robust_within_case"]) for result in results)
        ),
        "results": results,
    }
    write_json(analysis_dir / "aggregate_results.json", aggregate)

    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "event_key": result["event_key"],
            "event_label": result["event_label"],
            "family": result["family"],
            "h3_resolution": result["h3_resolution"],
            "fine_cell_spearman_descriptive_only": result[
                "fine_cell_spearman_descriptive_only"
            ],
            "direction_status": result["direction_status"],
            "holm_adjusted_p_primary_eight_tests": result[
                "holm_adjusted_p_primary_eight_tests"
            ],
            "permutation_supported_positive": result["permutation_supported_positive"],
            "spatially_robust_within_case": result["spatially_robust_within_case"],
            "input_sha256": result["input_sha256"],
            "input_validation_mode": result["input_validation_mode"],
        }
        for level in EXPECTED_PARENT_LEVELS:
            scale = result["scales"][str(level)]
            prefix = f"l{level}"
            row[f"{prefix}_status"] = scale.get("status")
            row[f"{prefix}_n_retained_blocks"] = scale.get("n_parent_blocks_retained")
            row[f"{prefix}_block_spearman"] = scale.get("observed_block_spearman")
            row[f"{prefix}_p_two_sided"] = scale.get("monte_carlo_p_two_sided")
            row[f"{prefix}_null_q025"] = scale.get("null_q025")
            row[f"{prefix}_null_q975"] = scale.get("null_q975")
        rows.append(row)
    pd.DataFrame(rows).to_csv(analysis_dir / "aggregate_results.csv", index=False)

    report = _make_report(
        protocol=protocol,
        results=results,
        aggregate=aggregate,
        protocol_hash=protocol_hash,
        implementation_hash=implementation_hash,
    )
    (analysis_dir / "REPORT.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in analysis_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "amendment_sha256": amendment_hash,
        "drought_regeneration_validation_sha256": drought_validation_hash,
        "implementation_sha256": implementation_hash,
        "artifacts": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in artifact_paths
        },
    }
    write_json(analysis_dir / "artifact_manifest.json", manifest)
    LOGGER.info("Wrote %s", analysis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
