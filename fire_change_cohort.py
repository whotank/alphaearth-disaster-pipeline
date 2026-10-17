#!/usr/bin/env python3
"""Outcome-blind cohort discovery and freezing for the fire-change studies.

This module intentionally stops before AlphaEarth extraction and before MTBS
severity reduction. It queries only official event metadata, applies frozen
metadata gates, creates a deterministic screening order, and can freeze a final
cohort from a separately produced geometry-only preflight manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger("fire_change_cohort")
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "fire_change_cohort_v1"
MTBS_POINT_QUERY = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/"
    "MapServer/62/query"
)
SELECTION_SALT = (
    "alphaearth-fire-change-fresh-v1|2026-08-25|outcome-blind-metadata"
)
ELIGIBLE_YEARS = (2021, 2022, 2023)
PRIMARY_YEAR_SLOTS = {2021: 6, 2022: 3, 2023: 3}
TARGET_PRIMARY_COUNT = 12
RESERVE_COUNT = 10
SCREENING_POOL_SIZE = 120
MIN_ACRES = 45_000.0
MAX_ACRES = 350_000.0
MAX_PER_STATE = 2
MAX_PER_ECOREGION = 2
HTTP_RETRIES = 3
POINT_OUT_FIELDS = (
    "fire_id,fire_name,asmnt_type,fire_type,ig_date,latitude,longitude,"
    "acres,irwinid,map_id,comments,pre_id,post_id"
)

# Every candidate in the prior held-out freeze is excluded, not only its final
# ten events. This deliberately gives the new study a stricter freshness rule
# than merely checking whether an AlphaEarth outcome was computed.
PRIOR_CANDIDATE_FIRE_IDS = frozenset(
    {
        "OK3585009921020180412",
        "CA3424011870020181108",
        "AK6660614659920190622",
        "OR4482112218820200816",
        "MT4571710705620200902",
        "ID4601211691120210707",
        "KS3910009890020211215",
        "NM3320410805920220514",
        "OR4542811673920220830",
        "CA3528611528620230728",
        "CA4005312066920190904",
        "ID4537511409120220717",
        "NM3571810539920220406",
        "CO4060910587920200813",
        "ID4475611687720200907",
        "TX3233509882920220317",
        "MT4655111135620210710",
        "AZ3237811092720200606",
        "CA3983912034520210702",
        "NV4181211632420180817",
    }
)

# These named events were used in earlier development work. Name-year matching
# catches them even when their MTBS FIRE_ID was never recorded in this repo.
PRIOR_DEVELOPMENT_NAME_YEARS = frozenset(
    {
        ("BOOTLEG", 2021),
        ("CALDOR", 2021),
        ("DIXIE", 2021),
        ("MARSHALL", 2021),
        ("MOSQUITO", 2022),
    }
)

RISK_COMMENT_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bcomplex\b",
        r"\bmerged\b",
        r"\bduplicate\b",
        r"\bincludes?\s+(?:multiple|other)\b",
        r"\bnot\s+discern(?:able|ible)\b",
        r"\bsingle[\s-]*scene\b",
        r"\bpart\s+of\b",
    )
)
SCENE_DATE_PATTERN = re.compile(r"(?=((?:19|20)\d{6}))")
FORBIDDEN_OUTCOME_KEY_PARTS = (
    "ecs",
    "embedding",
    "correlation",
    "spearman",
    "pearson",
    "rho",
    "effect",
    "severity_value",
    "severity_mean",
    "target_value",
    "target_mean",
    "dnbr",
    "dndvi",
    "model_score",
)


class CohortError(RuntimeError):
    """Fatal protocol, source, or cohort-integrity failure."""


@dataclass(frozen=True)
class Candidate:
    fire_id: str
    name: str
    event_year: int
    ignition_date: str
    assessment: str
    fire_type: str
    acres: float
    irwin_id: str
    map_id: int
    pre_id: str
    post_id: str
    comments: str
    latitude: float
    longitude: float
    state_code: str
    selection_hash: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def normalized_irwin_id(value: Any) -> str:
    return str(value or "").strip().strip("{}").upper()


def parse_arcgis_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def scene_dates(value: Any) -> tuple[datetime, ...]:
    dates: list[datetime] = []
    for compact in SCENE_DATE_PATTERN.findall(str(value or "")):
        try:
            dates.append(datetime.strptime(compact, "%Y%m%d").replace(tzinfo=UTC))
        except ValueError:
            continue
    return tuple(dates)


def candidate_selection_hash(fire_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{fire_id}".encode()).hexdigest()


def _field(attributes: dict[str, Any], name: str) -> Any:
    lookup = {str(key).casefold(): value for key, value in attributes.items()}
    return lookup.get(name.casefold())


def _candidate_from_attributes(attributes: dict[str, Any]) -> Candidate:
    ignition = parse_arcgis_date(_field(attributes, "ig_date"))
    if ignition is None:
        raise CohortError("Missing or malformed ignition date")
    fire_id = str(_field(attributes, "fire_id") or "").strip().upper()
    if not fire_id:
        raise CohortError("Missing FIRE_ID")
    try:
        map_id = int(_field(attributes, "map_id"))
        acres = float(_field(attributes, "acres"))
        latitude = float(_field(attributes, "latitude"))
        longitude = float(_field(attributes, "longitude"))
    except (TypeError, ValueError) as exc:
        raise CohortError(f"{fire_id}: malformed numeric metadata") from exc
    return Candidate(
        fire_id=fire_id,
        name=normalized_text(_field(attributes, "fire_name")),
        event_year=ignition.year,
        ignition_date=ignition.date().isoformat(),
        assessment=normalized_text(_field(attributes, "asmnt_type")),
        fire_type=normalized_text(_field(attributes, "fire_type")),
        acres=acres,
        irwin_id=normalized_irwin_id(_field(attributes, "irwinid")),
        map_id=map_id,
        pre_id=str(_field(attributes, "pre_id") or "").strip(),
        post_id=str(_field(attributes, "post_id") or "").strip(),
        comments=str(_field(attributes, "comments") or "").strip(),
        latitude=latitude,
        longitude=longitude,
        state_code=fire_id[:2],
        selection_hash=candidate_selection_hash(fire_id),
    )


def metadata_ineligibility(candidate: Candidate) -> list[str]:
    reasons: list[str] = []
    ignition = datetime.fromisoformat(candidate.ignition_date).replace(tzinfo=UTC)
    pre_dates = scene_dates(candidate.pre_id)
    post_dates = scene_dates(candidate.post_id)

    if candidate.event_year not in ELIGIBLE_YEARS:
        reasons.append("year")
    if candidate.fire_type != "WILDFIRE":
        reasons.append("not_wildfire")
    if candidate.assessment not in {"INITIAL", "EXTENDED"}:
        reasons.append("assessment")
    if not MIN_ACRES <= candidate.acres <= MAX_ACRES:
        reasons.append("area")
    if not candidate.name:
        reasons.append("name")
    if not candidate.irwin_id:
        reasons.append("missing_irwin_id")
    if not candidate.pre_id or not candidate.post_id:
        reasons.append("missing_scene_id")
    if not pre_dates or not any(value < ignition for value in pre_dates):
        reasons.append("pre_scene_not_before_ignition")
    if not post_dates or not any(value > ignition for value in post_dates):
        reasons.append("post_scene_not_after_ignition")
    if candidate.fire_id in PRIOR_CANDIDATE_FIRE_IDS:
        reasons.append("prior_candidate_fire_id")
    if (candidate.name, candidate.event_year) in PRIOR_DEVELOPMENT_NAME_YEARS:
        reasons.append("prior_development_event")
    if any(pattern.search(candidate.comments) for pattern in RISK_COMMENT_PATTERNS):
        reasons.append("risk_comment")
    if not (-125.0 <= candidate.longitude <= -66.0):
        reasons.append("outside_conus_longitude")
    if not (24.0 <= candidate.latitude <= 50.0):
        reasons.append("outside_conus_latitude")
    return sorted(set(reasons))


def _http_json(
    endpoint: str,
    parameters: dict[str, Any],
    *,
    retries: int = HTTP_RETRIES,
) -> tuple[dict[str, Any], str, str]:
    url = f"{endpoint}?{urlencode(parameters)}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "alphaearth-fire-change-study/1.0",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=180) as response:
                data = response.read()
                final_url = response.geturl()
            payload = json.loads(data)
            if "error" in payload:
                raise CohortError(f"ArcGIS error: {payload['error']}")
            return payload, final_url, hashlib.sha256(data).hexdigest()
        except (
            CohortError,
            json.JSONDecodeError,
            OSError,
            TimeoutError,
        ) as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise CohortError(f"HTTP request failed after {retries + 1} attempts: {last_error}")


def query_mtbs_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    where = (
        "ig_date >= DATE '2021-01-01' AND ig_date < DATE '2024-01-01' "
        "AND fire_type='Wildfire'"
    )
    features: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    offset = 0
    page_size = 2000
    while True:
        payload, url, response_sha256 = _http_json(
            MTBS_POINT_QUERY,
            {
                "f": "json",
                "where": where,
                "outFields": POINT_OUT_FIELDS,
                "returnGeometry": "false",
                "orderByFields": "fire_id",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            },
        )
        page = list(payload.get("features", []))
        features.extend(page)
        sources.append(
            {
                "endpoint": MTBS_POINT_QUERY,
                "final_url": url,
                "response_sha256": response_sha256,
                "offset": offset,
                "feature_count": len(page),
            }
        )
        exceeded = bool(payload.get("exceededTransferLimit"))
        if not exceeded and len(page) < page_size:
            break
        if not page:
            raise CohortError("ArcGIS pagination stalled on an empty page")
        offset += len(page)
    return features, sources


def discover_candidates(
    features: Iterable[dict[str, Any]],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    eligible: list[Candidate] = []
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for feature in features:
        attributes = dict(feature.get("attributes", {}))
        try:
            candidate = _candidate_from_attributes(attributes)
            reasons = metadata_ineligibility(candidate)
            if candidate.fire_id in seen:
                reasons = sorted({*reasons, "duplicate_fire_id"})
            seen.add(candidate.fire_id)
            ledger.append(
                {
                    **asdict(candidate),
                    "metadata_eligible": not reasons,
                    "metadata_ineligibility": reasons,
                }
            )
            if not reasons:
                eligible.append(candidate)
        except CohortError as exc:
            ledger.append(
                {
                    "fire_id": normalized_text(_field(attributes, "fire_id")),
                    "metadata_eligible": False,
                    "metadata_ineligibility": ["parse_failure"],
                    "parse_error": str(exc),
                }
            )
    eligible.sort(key=lambda item: (item.selection_hash, item.fire_id))
    return eligible, ledger


def screening_pool(candidates: Sequence[Candidate]) -> list[Candidate]:
    return list(candidates[:SCREENING_POOL_SIZE])


def _validate_preflight_row(row: dict[str, Any]) -> None:
    required = {
        "fire_id",
        "preflight_pass",
        "event_year",
        "state_code",
        "ecoregion_code",
        "selection_hash",
    }
    missing = sorted(required - set(row))
    if missing:
        raise CohortError(f"Preflight row lacks fields: {missing}")
    if not str(row["ecoregion_code"]).strip():
        raise CohortError(f"{row['fire_id']}: missing ecoregion code")


def select_diverse_cohort(
    rows: Sequence[dict[str, Any]],
    *,
    primary_year_slots: dict[int, int] | None = None,
    reserve_count: int = RESERVE_COUNT,
    max_per_state: int = MAX_PER_STATE,
    max_per_ecoregion: int = MAX_PER_ECOREGION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select fixed year slots while enforcing state and ecoregion diversity."""
    slots = dict(PRIMARY_YEAR_SLOTS if primary_year_slots is None else primary_year_slots)
    if sum(slots.values()) != TARGET_PRIMARY_COUNT:
        raise CohortError(
            f"Primary year slots must sum to {TARGET_PRIMARY_COUNT}, got {slots}"
        )
    passed: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("preflight_pass")):
            continue
        _validate_preflight_row(row)
        passed.append(dict(row))
    passed.sort(key=lambda row: (str(row["selection_hash"]), str(row["fire_id"])))

    primaries: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    state_counts: dict[str, int] = {}
    ecoregion_counts: dict[str, int] = {}
    for year in ELIGIBLE_YEARS:
        year_target = int(slots[year])
        year_selected = 0
        for row in passed:
            fire_id = str(row["fire_id"])
            state = str(row["state_code"])
            ecoregion = str(row["ecoregion_code"])
            if fire_id in used_ids or int(row["event_year"]) != year:
                continue
            if state_counts.get(state, 0) >= max_per_state:
                continue
            if ecoregion_counts.get(ecoregion, 0) >= max_per_ecoregion:
                continue
            selected = {
                **row,
                "role": "primary",
                "year_slot": year,
                "slot_rank": len(primaries) + 1,
            }
            primaries.append(selected)
            used_ids.add(fire_id)
            state_counts[state] = state_counts.get(state, 0) + 1
            ecoregion_counts[ecoregion] = ecoregion_counts.get(ecoregion, 0) + 1
            year_selected += 1
            if year_selected == year_target:
                break
        if year_selected != year_target:
            raise CohortError(
                f"Unable to fill {year_target} primary slots for {year}; "
                f"filled {year_selected}"
            )

    reserves: list[dict[str, Any]] = []
    for row in passed:
        fire_id = str(row["fire_id"])
        if fire_id in used_ids:
            continue
        reserves.append(
            {
                **row,
                "role": "reserve",
                "reserve_rank": len(reserves) + 1,
            }
        )
        if len(reserves) == reserve_count:
            break
    if len(reserves) != reserve_count:
        raise CohortError(
            f"Expected {reserve_count} reserves, found only {len(reserves)}"
        )
    return primaries, reserves


def select_causal_cohort(
    rows: Sequence[dict[str, Any]],
    *,
    target_count: int = 8,
    minimum_count: int = 6,
    max_per_state: int = MAX_PER_STATE,
    max_per_ecoregion: int = MAX_PER_ECOREGION,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("causal_preflight_pass")):
            continue
        if int(row.get("event_year", 0)) not in {2022, 2023}:
            continue
        _validate_preflight_row({**row, "preflight_pass": True})
        eligible.append(dict(row))
    eligible.sort(key=lambda row: (str(row["selection_hash"]), str(row["fire_id"])))

    selected: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    ecoregion_counts: dict[str, int] = {}
    for row in eligible:
        state = str(row["state_code"])
        ecoregion = str(row["ecoregion_code"])
        if state_counts.get(state, 0) >= max_per_state:
            continue
        if ecoregion_counts.get(ecoregion, 0) >= max_per_ecoregion:
            continue
        selected.append(
            {
                **row,
                "role": "causal_primary",
                "causal_rank": len(selected) + 1,
            }
        )
        state_counts[state] = state_counts.get(state, 0) + 1
        ecoregion_counts[ecoregion] = ecoregion_counts.get(ecoregion, 0) + 1
        if len(selected) == target_count:
            break
    if len(selected) < minimum_count:
        raise CohortError(
            f"Fresh causal cohort has {len(selected)} events; minimum {minimum_count}"
        )
    return selected


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_outcome_blind_payload(payload: Any) -> None:
    violations = sorted(
        {
            key
            for key in _walk_keys(payload)
            if any(part in key.casefold() for part in FORBIDDEN_OUTCOME_KEY_PARTS)
        }
    )
    if violations:
        raise CohortError(f"Outcome-like fields are forbidden in cohort data: {violations}")


def discovery_manifest(
    eligible: Sequence[Candidate],
    ledger: Sequence[dict[str, Any]],
    sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for row in ledger:
        for reason in row.get("metadata_ineligibility", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    pool = screening_pool(eligible)
    payload = {
        "schema": "alphaearth_fire_change_discovery_v1",
        "created_at_utc": utc_now(),
        "selection_salt": SELECTION_SALT,
        "eligible_years": list(ELIGIBLE_YEARS),
        "metadata_rules": {
            "fire_type": "Wildfire",
            "assessment": ["Initial", "Extended"],
            "minimum_acres": MIN_ACRES,
            "maximum_acres": MAX_ACRES,
            "conus_bounds": {
                "longitude": [-125.0, -66.0],
                "latitude": [24.0, 50.0],
            },
            "require_exact_irwin_id": True,
            "require_two_scene_bracketing": True,
            "prior_candidate_fire_ids_excluded": sorted(PRIOR_CANDIDATE_FIRE_IDS),
            "prior_development_name_years_excluded": [
                {"name": name, "year": year}
                for name, year in sorted(PRIOR_DEVELOPMENT_NAME_YEARS)
            ],
        },
        "queried_feature_count": len(ledger),
        "metadata_eligible_count": len(eligible),
        "planned_selection": {
            "primary_year_slots": {
                str(year): count for year, count in PRIMARY_YEAR_SLOTS.items()
            },
            "target_primary_count": TARGET_PRIMARY_COUNT,
            "reserve_count": RESERVE_COUNT,
            "maximum_per_state": MAX_PER_STATE,
            "maximum_per_ecoregion": MAX_PER_ECOREGION,
        },
        "ineligibility_counts": dict(sorted(reason_counts.items())),
        "screening_pool_size_rule": SCREENING_POOL_SIZE,
        "screening_pool_count": len(pool),
        "screening_pool": [asdict(candidate) for candidate in pool],
        "source_pages": list(sources),
        "outcome_fields_used": [],
    }
    validate_outcome_blind_payload(payload)
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def freeze_from_preflight(
    discovery_path: Path,
    preflight_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("discovery_payload_sha256") != discovery.get("payload_sha256"):
        raise CohortError("Preflight does not bind to the supplied discovery manifest")
    rows = list(preflight.get("events", []))
    validate_outcome_blind_payload(rows)
    primaries, reserves = select_diverse_cohort(rows)
    payload = {
        "schema": "alphaearth_fire_change_fresh_cohort_freeze_v1",
        "frozen_at_utc": utc_now(),
        "status": "FROZEN_INPUTS_ONLY_NO_ALPHAEARTH_OR_TARGET_OUTCOMES_INSPECTED",
        "selection_salt": SELECTION_SALT,
        "discovery_manifest": discovery_path.as_posix(),
        "discovery_payload_sha256": discovery["payload_sha256"],
        "preflight_manifest": preflight_path.as_posix(),
        "preflight_payload_sha256": preflight["payload_sha256"],
        "primaries": primaries,
        "reserves": reserves,
        "replacement": {
            "allowed": "Only geometry, official identity, overlap, support, or extraction-completeness gates declared before outcomes.",
            "order": "Ascending reserve_rank.",
            "outcome_dependent_replacement": False,
        },
        "outcome_fields_used": [],
    }
    validate_outcome_blind_payload(payload)
    payload["payload_sha256"] = canonical_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


def freeze_causal_from_preflight(
    discovery_path: Path,
    preflight_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("discovery_payload_sha256") != discovery.get("payload_sha256"):
        raise CohortError("Preflight does not bind to the supplied discovery manifest")
    rows = list(preflight.get("events", []))
    validate_outcome_blind_payload(rows)
    events = select_causal_cohort(rows)
    payload = {
        "schema": "alphaearth_fire_ecs_causal_fresh_cohort_freeze_v1",
        "frozen_at_utc": utc_now(),
        "status": "FROZEN_INPUTS_ONLY_NO_ALPHAEARTH_OUTCOMES_INSPECTED",
        "selection_salt": SELECTION_SALT,
        "discovery_manifest": discovery_path.as_posix(),
        "discovery_payload_sha256": discovery["payload_sha256"],
        "preflight_manifest": preflight_path.as_posix(),
        "preflight_payload_sha256": preflight["payload_sha256"],
        "target_event_count": 8,
        "minimum_event_count": 6,
        "selected_event_count": len(events),
        "target_shortfall": 8 - len(events),
        "events": events,
        "selection_note": (
            "Every preflight-passing 2022-2023 event was eligible in frozen hash "
            "order, subject to the unchanged state and ecoregion diversity caps."
        ),
        "outcome_fields_used": [],
    }
    validate_outcome_blind_payload(payload)
    payload["payload_sha256"] = canonical_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output namespace for discovery artifacts.",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover", help="Query official metadata and create a screening pool."
    )
    discover.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and validate metadata without writing artifacts.",
    )

    freeze = subparsers.add_parser(
        "freeze", help="Freeze final primaries and reserves from geometry preflight."
    )
    freeze.add_argument("--discovery", type=Path, required=True)
    freeze.add_argument("--preflight", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and select without writing the cohort freeze.",
    )

    freeze_causal = subparsers.add_parser(
        "freeze-causal",
        help="Freeze the fresh 2022-2023 matched-BACI cohort.",
    )
    freeze_causal.add_argument("--discovery", type=Path, required=True)
    freeze_causal.add_argument("--preflight", type=Path, required=True)
    freeze_causal.add_argument("--output", type=Path, required=True)
    freeze_causal.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.command == "discover":
        features, sources = query_mtbs_metadata()
        eligible, ledger = discover_candidates(features)
        manifest = discovery_manifest(eligible, ledger, sources)
        LOGGER.info(
            "Discovered %d metadata-eligible events from %d records; screening pool %d",
            len(eligible),
            len(ledger),
            len(manifest["screening_pool"]),
        )
        if not args.dry_run:
            atomic_write_json(args.output_dir / "discovery.json", manifest)
            atomic_write_json(
                args.output_dir / "metadata_ledger.json",
                {
                    "schema": "alphaearth_fire_change_metadata_ledger_v1",
                    "created_at_utc": utc_now(),
                    "rows": ledger,
                    "outcome_fields_used": [],
                },
            )
        return 0

    if args.command == "freeze":
        if args.dry_run:
            discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            if preflight.get("discovery_payload_sha256") != discovery.get(
                "payload_sha256"
            ):
                raise CohortError("Preflight does not match discovery manifest")
            validate_outcome_blind_payload(preflight.get("events", []))
            primaries, reserves = select_diverse_cohort(preflight["events"])
            LOGGER.info(
                "Dry run selected %d primaries and %d reserves",
                len(primaries),
                len(reserves),
            )
        else:
            payload = freeze_from_preflight(
                args.discovery, args.preflight, args.output
            )
            LOGGER.info(
                "Froze %d primaries and %d reserves",
                len(payload["primaries"]),
                len(payload["reserves"]),
            )
        return 0
    if args.command == "freeze-causal":
        if args.dry_run:
            discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            if preflight.get("discovery_payload_sha256") != discovery.get(
                "payload_sha256"
            ):
                raise CohortError("Preflight does not match discovery manifest")
            events = select_causal_cohort(preflight.get("events", []))
            LOGGER.info("Dry run selected %d causal events", len(events))
        else:
            payload = freeze_causal_from_preflight(
                args.discovery, args.preflight, args.output
            )
            LOGGER.info("Froze %d causal events", payload["selected_event_count"])
        return 0
    raise CohortError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
