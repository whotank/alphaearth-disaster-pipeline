"""Shared integrity, checkpoint, and deterministic helpers for the MTBS study."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Sequence

import ee
import h3
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon, mapping


LOGGER = logging.getLogger("mtbs_heldout")

STUDY_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
FREEZE_PATH = STUDY_ROOT / "freeze" / "canonical.json"
FREEZE_AUDIT_PATH = STUDY_ROOT / "freeze" / "freeze_audit.json"
PROTOCOL_PATH = STUDY_ROOT / "protocol.json"
RAW_ROOT = STUDY_ROOT / "raw"
CHECKPOINT_ROOT = STUDY_ROOT / "checkpoints"
SEALED_ROOT = STUDY_ROOT / "sealed"
OUTPUT_ROOT = STUDY_ROOT / "outputs"
ATTEMPT_OUTPUT_ROOT = OUTPUT_ROOT / "attempts"
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
FIGURE_ROOT = ANALYSIS_ROOT / "figures"
PROGRESS_LOG_PATH = CHECKPOINT_ROOT / "progress.jsonl"

EXPECTED_FREEZE_SHA256 = (
    "eb204cfc9d4a16d9c27db4d7641c4ebf3e6fc98ae637ed71a9a7c50a4a32794e"
)
EXPECTED_PROTOCOL_SHA256 = (
    "14ea4b1ae082f599ac0aa6b5984acb9672a5e1d54621af627333ceffd8ac20b5"
)
BASE_COMMIT = "2a6fb2933f535d6372d7bd1d7eb5158be7d2345f"
EXPECTED_BRANCH = "study/mtbs-heldout-20260817"

MTBS_SEVERITY_COLLECTION = "USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1"
ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
ALPHAEARTH_BANDS = tuple(f"A{number:02d}" for number in range(64))
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ("B4", "B8", "B12")
S2_SCL_EXCLUDED = (0, 1, 3, 7, 8, 9, 10, 11)
WORLD_COVER_COLLECTION = "ESA/WorldCover/v100"
WORLD_COVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
WORLD_COVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
H3_CAP_SALT = "alphaearth-mtbs-common-h3-v1"
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPLICATES = 100_000


class StudyError(RuntimeError):
    """Fatal protocol or artifact-integrity failure."""


class GateFailure(StudyError):
    """Frozen P01-P05 eligibility failure."""

    def __init__(self, gate: str, message: str) -> None:
        if gate not in {"P01", "P02", "P03", "P04", "P05"}:
            raise ValueError(f"Invalid replacement gate {gate}")
        self.gate = gate
        super().__init__(message)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(STUDY_ROOT.resolve()).as_posix()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any, *, readonly: bool = False) -> None:
    data = (
        json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    if readonly and path.exists():
        if path.read_bytes() != data:
            raise StudyError(f"Refusing to replace immutable artifact {relative(path)}")
        return
    if path.exists() and not os.access(path, os.W_OK):
        os.chmod(path, 0o644)
    atomic_write_bytes(path, data)
    if readonly:
        os.chmod(path, 0o444)


def append_progress(
    stage: str, status: str, details: dict[str, Any] | None = None
) -> None:
    """Durably append a compact, path-safe resume record."""
    payload = {
        "timestamp_utc": utc_now(),
        "stage": stage,
        "status": status,
        "details": jsonable(details or {}),
    }
    data = canonical_json_bytes(payload) + b"\n"
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        PROGRESS_LOG_PATH,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def h3_population_hash(h3_ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(str(value) for value in h3_ids).encode("utf-8"))


def deterministic_h3_cap(
    fire_id: str, h3_ids: Sequence[str], cap: int = 2000
) -> list[str]:
    unique = sorted(set(str(value) for value in h3_ids))
    if len(unique) != len(h3_ids):
        raise StudyError(f"{fire_id}: duplicate H3 IDs before cap")
    if len(unique) <= cap:
        return unique
    selected = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(
                f"{H3_CAP_SALT}|{fire_id}|{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )[:cap]
    return sorted(selected)


def verify_environment() -> dict[str, Any]:
    freeze_bytes = FREEZE_PATH.read_bytes()
    if freeze_bytes.endswith(b"\n"):
        raise StudyError("Canonical freeze unexpectedly has a trailing newline")
    actual_freeze = sha256_bytes(freeze_bytes)
    if actual_freeze != EXPECTED_FREEZE_SHA256:
        raise StudyError(
            f"Canonical freeze hash mismatch: {actual_freeze} != "
            f"{EXPECTED_FREEZE_SHA256}"
        )
    freeze = json.loads(freeze_bytes)
    if canonical_json_bytes(freeze) != freeze_bytes:
        raise StudyError("Canonical freeze does not match documented encoding")
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise StudyError(
            f"Protocol hash mismatch: {protocol_hash} != "
            f"{EXPECTED_PROTOCOL_SHA256}"
        )
    audit = json.loads(FREEZE_AUDIT_PATH.read_text(encoding="utf-8"))
    if not audit.get("verified") or audit.get("canonical_sha256") != actual_freeze:
        raise StudyError("Freeze audit does not verify canonical bytes")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != BASE_COMMIT:
        raise StudyError(f"Wrong base commit: {head} != {BASE_COMMIT}")
    if branch != EXPECTED_BRANCH:
        raise StudyError(f"Wrong isolated branch: {branch!r}")
    return {
        "freeze": freeze,
        "freeze_sha256": actual_freeze,
        "protocol_sha256": protocol_hash,
        "head": head,
        "branch": branch,
    }


def leakage_audit(freeze: dict[str, Any]) -> dict[str, Any]:
    candidates = [*freeze["primaries"], *freeze["reserves"]]
    fire_ids = [str(candidate["fire_id"]) for candidate in candidates]
    matches: dict[str, list[str]] = {fire_id: [] for fire_id in fire_ids}
    suffixes = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        candidate_path = path.relative_to(REPOSITORY_ROOT)
        if (
            candidate_path.parts[0] in {".git", "outputs", "studies"}
            or ".venv" in candidate_path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fire_id in fire_ids:
            if fire_id in text:
                matches[fire_id].append(candidate_path.as_posix())
    nonempty = {key: value for key, value in matches.items() if value}
    if nonempty:
        raise StudyError(f"Frozen FIRE_ID leakage in existing files: {nonempty}")
    return {
        "audit_schema": "alphaearth_mtbs_leakage_audit_v1",
        "checked_candidate_count": len(candidates),
        "exact_fire_id_matches_outside_study": nonempty,
        "forbidden_module_imports": [],
        "frozen_repository_check": freeze["repository_check"],
        "passed": not nonempty and bool(freeze["repository_check"]["verified"]),
    }


def geometry_to_h3(
    geometry: Polygon | MultiPolygon, resolution: int = 8
) -> list[str]:
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    cells: set[str] = set()
    for polygon in polygons:
        exterior = [(lat, lon) for lon, lat in polygon.exterior.coords[:-1]]
        holes = [
            [(lat, lon) for lon, lat in ring.coords[:-1]]
            for ring in polygon.interiors
        ]
        cells.update(
            str(cell)
            for cell in h3.polygon_to_cells(
                h3.LatLngPoly(exterior, *holes), resolution
            )
        )
    retained: list[str] = []
    for cell in sorted(cells):
        latitude, longitude = h3.cell_to_latlng(cell)
        if geometry.covers(Point(float(longitude), float(latitude))):
            retained.append(cell)
    if not retained:
        raise GateFailure("P03", "Boundary produced no center-contained H3 cells")
    return retained


def ee_geometry(geometry: Polygon | MultiPolygon) -> ee.Geometry:
    return ee.Geometry(mapping(geometry), proj="EPSG:4326", geodesic=False)


def h3_to_ee_polygon(h3_index: str) -> ee.Geometry:
    ring = [
        [float(longitude), float(latitude)]
        for latitude, longitude in h3.cell_to_boundary(str(h3_index))
    ]
    ring.append(ring[0])
    return ee.Geometry.Polygon([ring], proj="EPSG:4326", geodesic=False)


def h3_feature_collection(h3_ids: Sequence[str]) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(h3_to_ee_polygon(value), {"h3_index": str(value)})
            for value in h3_ids
        ]
    )


def validate_h3_frame(
    frame: pd.DataFrame, h3_ids: Sequence[str], *, context: str
) -> pd.DataFrame:
    expected = [str(value) for value in h3_ids]
    if len(expected) != len(set(expected)):
        raise StudyError(f"{context}: duplicate requested H3 IDs")
    if "h3_index" not in frame.columns:
        raise StudyError(f"{context}: response lacks h3_index")
    output = frame.copy()
    output["h3_index"] = output["h3_index"].astype(str)
    counts = Counter(output["h3_index"])
    duplicate = sorted(key for key, value in counts.items() if value > 1)
    missing = sorted(set(expected) - set(output["h3_index"]))
    unexpected = sorted(set(output["h3_index"]) - set(expected))
    if duplicate or missing or unexpected or len(output) != len(expected):
        raise StudyError(
            f"{context}: H3 mismatch duplicate={duplicate[:5]} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    order = {value: index for index, value in enumerate(expected)}
    return (
        output.assign(_order=output["h3_index"].map(order))
        .sort_values("_order", kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def feature_collection_frame(collection: ee.FeatureCollection) -> pd.DataFrame:
    payload = collection.getInfo()
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties", {}))
        if properties.get("h3_index") is not None:
            properties["h3_index"] = str(properties["h3_index"])
        rows.append(properties)
    return pd.DataFrame(rows)


def reduce_h3_checkpointed(
    image: ee.Image,
    h3_ids: Sequence[str],
    *,
    scale_m: int,
    batch_size: int,
    tile_scale: int,
    max_retries: int,
    checkpoint_dir: Path,
    checkpoint_config: dict[str, Any],
    context: str,
) -> pd.DataFrame:
    cells = [str(value) for value in h3_ids]
    validate_h3_frame(
        pd.DataFrame({"h3_index": cells}), cells, context=f"{context} request"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_bytes(canonical_json_bytes(checkpoint_config))
    config_path = checkpoint_dir / "config.json"
    config_payload = {"config": checkpoint_config, "config_sha256": config_hash}
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
            raise StudyError(f"{context}: checkpoint configuration changed")
    else:
        write_json(config_path, config_payload)

    frames: list[pd.DataFrame] = []
    total_batches = math.ceil(len(cells) / batch_size)
    for batch_number, start in enumerate(range(0, len(cells), batch_size), start=1):
        batch_ids = cells[start : start + batch_size]
        path = checkpoint_dir / f"batch_{batch_number:04d}.parquet"
        if path.exists():
            frame = validate_h3_frame(
                pd.read_parquet(path),
                batch_ids,
                context=f"{context} resumed batch {batch_number}",
            )
            frames.append(frame)
            continue
        reduced = image.reduceRegions(
            collection=h3_feature_collection(batch_ids),
            reducer=ee.Reducer.mean(),
            scale=int(scale_m),
            tileScale=int(tile_scale),
        )
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                frame = validate_h3_frame(
                    feature_collection_frame(reduced),
                    batch_ids,
                    context=f"{context} batch {batch_number}",
                )
                write_parquet(path, frame)
                frames.append(frame)
                break
            except Exception as exc:  # pragma: no cover - EE dependent
                last_error = exc
                if attempt >= max_retries:
                    raise StudyError(
                        f"{context}: batch {batch_number}/{total_batches} failed"
                    ) from exc
                delay = 2**attempt
                LOGGER.warning(
                    "%s batch %d/%d failed; retrying in %d s",
                    context,
                    batch_number,
                    total_batches,
                    delay,
                )
                time.sleep(delay)
        else:  # pragma: no cover
            raise StudyError(f"{context}: unreachable batch failure {last_error}")
        LOGGER.info(
            "%s batch %d/%d complete (%d/%d cells)",
            context,
            batch_number,
            total_batches,
            min(start + batch_size, len(cells)),
            len(cells),
        )
    return validate_h3_frame(
        pd.concat(frames, ignore_index=True, sort=False),
        cells,
        context=f"{context} combined",
    )


def normalize_numeric_columns(
    frame: pd.DataFrame, expected: Sequence[str]
) -> pd.DataFrame:
    output = pd.DataFrame({"h3_index": frame["h3_index"].astype(str)})
    for name in expected:
        candidates = (name, f"{name}_mean", name.upper(), f"{name}_mean".upper())
        found = next((candidate for candidate in candidates if candidate in frame), None)
        output[name] = (
            pd.to_numeric(frame[found], errors="coerce") if found else np.nan
        )
    return output


def update_runtime(stage: str, started: float, extra: dict[str, Any]) -> None:
    path = OUTPUT_ROOT / "runtime_manifest.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema": "alphaearth_mtbs_runtime_v1", "stages": {}}
    )
    payload["stages"][stage] = {
        "completed_at_utc": utc_now(),
        "duration_seconds": time.monotonic() - started,
        **extra,
    }
    payload["storage_bytes"] = {
        "raw": directory_size(RAW_ROOT),
        "checkpoints": directory_size(CHECKPOINT_ROOT),
        "sealed": directory_size(SEALED_ROOT),
        "compact_outputs": directory_size(OUTPUT_ROOT),
    }
    write_json(path, payload)
