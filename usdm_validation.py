#!/usr/bin/env python3
"""Outcome-independent acceptance rules for official USDM vector replays.

The thresholds are fixed in drought-category units before publication metrics
are selected.  They require near-identical spatial ordering and small
rasterization error relative to the full 0--5 USDM score range:

* Pearson >= 0.99 and Spearman >= 0.90;
* mean absolute error <= 0.125 category units (2.5% of the range);
* maximum absolute error <= 0.25 category units (5% of the range);
* exact one-to-one H3 coverage at resolution 6;
* the exact number of Tuesday USDM releases from April 1 through October 31;
* recorded SHA-256 hashes for both official archives and the AOI.

Passing validates the Earth Engine mirror as a close implementation cross-check.
The official vector-derived H3 metric remains the publication reference.
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any


USDM_VALIDATION_CRITERIA: dict[str, Any] = {
    "minimum_official_vs_mirror_pearson": 0.99,
    "minimum_official_vs_mirror_spearman": 0.90,
    "maximum_official_vs_mirror_mae_category_units": 0.125,
    "maximum_official_vs_mirror_max_abs_error_category_units": 0.25,
    "usdm_score_range": [0, 5],
    "required_h3_resolution": 6,
    "season_start": "April 1",
    "season_end_inclusive": "October 31",
    "required_release_weekday": "Tuesday",
    "publication_reference": "official_ndmc_vectors",
    "mirror_role": "implementation_crosscheck_only",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MIRROR_AGREEMENT_CHECKS = {
    "pearson_meets_threshold",
    "spearman_meets_threshold",
    "mae_meets_threshold",
    "max_error_meets_threshold",
}


def expected_weekly_map_count(year: int) -> int:
    """Return the number of Tuesdays from April 1 through October 31."""
    current = date(year, 4, 1)
    end = date(year, 10, 31)
    count = 0
    while current <= end:
        if current.weekday() == 1:
            count += 1
        current += timedelta(days=1)
    return count


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def evaluate_official_validation(
    summary: dict[str, Any],
    *,
    expected_event: str,
    expected_pre_year: int,
    expected_post_year: int,
    expected_h3_cells: int,
    expected_h3_resolution: int = 6,
) -> dict[str, Any]:
    """Re-evaluate a validation artifact instead of trusting a status label."""
    pearson = _finite_number(summary.get("official_vs_mirror_pearson"))
    spearman = _finite_number(summary.get("official_vs_mirror_spearman"))
    mae = _finite_number(summary.get("official_vs_mirror_mae"))
    max_error = _finite_number(summary.get("official_vs_mirror_max_abs_error"))
    coverage = summary.get("h3_coverage")
    if not isinstance(coverage, dict):
        coverage = {}

    checks = {
        "event_identity_matches": summary.get("event") == expected_event,
        "pre_year_matches": summary.get("pre_year") == expected_pre_year,
        "post_year_matches": summary.get("post_year") == expected_post_year,
        "pre_week_count_matches_calendar": summary.get("weekly_map_count_pre")
        == expected_weekly_map_count(expected_pre_year),
        "post_week_count_matches_calendar": summary.get("weekly_map_count_post")
        == expected_weekly_map_count(expected_post_year),
        "h3_cell_count_matches": summary.get("h3_cells") == expected_h3_cells,
        "h3_sets_equal": coverage.get("sets_equal") is True,
        "h3_rows_unique": coverage.get("all_unique") is True,
        "h3_resolution_matches": coverage.get("resolution")
        == expected_h3_resolution,
        "h3_alpha_count_matches": coverage.get("alphaearth_cells")
        == expected_h3_cells,
        "h3_mirror_count_matches": coverage.get("mirror_cells")
        == expected_h3_cells,
        "h3_official_count_matches": coverage.get("official_cells")
        == expected_h3_cells,
        "pre_archive_sha256_recorded": bool(
            _SHA256_RE.fullmatch(str(summary.get("pre_archive_sha256", "")))
        ),
        "post_archive_sha256_recorded": bool(
            _SHA256_RE.fullmatch(str(summary.get("post_archive_sha256", "")))
        ),
        "aoi_sha256_recorded": bool(
            _SHA256_RE.fullmatch(str(summary.get("aoi_sha256", "")))
        ),
        "pearson_meets_threshold": pearson is not None
        and pearson
        >= USDM_VALIDATION_CRITERIA["minimum_official_vs_mirror_pearson"],
        "spearman_meets_threshold": spearman is not None
        and spearman
        >= USDM_VALIDATION_CRITERIA["minimum_official_vs_mirror_spearman"],
        "mae_meets_threshold": mae is not None
        and mae
        <= USDM_VALIDATION_CRITERIA[
            "maximum_official_vs_mirror_mae_category_units"
        ],
        "max_error_meets_threshold": max_error is not None
        and max_error
        <= USDM_VALIDATION_CRITERIA[
            "maximum_official_vs_mirror_max_abs_error_category_units"
        ],
    }
    core_failures = [
        name
        for name, passed in checks.items()
        if name not in _MIRROR_AGREEMENT_CHECKS and not passed
    ]
    agreement_failures = [
        name
        for name in sorted(_MIRROR_AGREEMENT_CHECKS)
        if not checks[name]
    ]
    failures = [*core_failures, *agreement_failures]
    official_reference_complete = not core_failures
    mirror_agreement_passed = not agreement_failures
    return {
        "passed": official_reference_complete and mirror_agreement_passed,
        "official_reference_complete": official_reference_complete,
        "mirror_agreement_passed": mirror_agreement_passed,
        "checks": checks,
        "failures": failures,
        "core_failures": core_failures,
        "agreement_failures": agreement_failures,
        "criteria": dict(USDM_VALIDATION_CRITERIA),
    }
