#!/usr/bin/env python3
"""Execute the frozen TESSERA v1 comparator on sealed MTBS H3 populations.

The command has four durable phases: metadata-only preflight, resumable object
download, outcome-blind TESSERA aggregation for every accessible event, and
analysis only after all event aggregation checkpoints are complete.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.stats import spearmanr
from shapely import geometry as sg
from shapely import ops as so


LOGGER = logging.getLogger("tessera_mtbs_secondary")
ROOT = Path(__file__).resolve().parent
DEFAULT_MTBS_ROOT = Path(
    os.environ.get(
        "MTBS_HELDOUT_ROOT",
        ROOT / "studies" / "mtbs_heldout_20260817",
    )
)
DEFAULT_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "alphaearth-tessera-mtbs-v1"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "tessera_mtbs_secondary"
DEFAULT_PROTOCOL = ROOT / "tessera_mtbs_secondary_protocol_v1.canonical.json"
DEFAULT_MANIFEST = (
    Path.home()
    / ".cache"
    / "alphaearth-tessera-la-palma-pilot"
    / "manifest"
    / "manifest.parquet"
)

PROTOCOL_SHA256 = (
    "8571e8ac505add4657645c08eee3b141b4bcc2d68298d0463c6063bfff3b2380"
)
MTBS_CONTRACT_SHA256 = (
    "eb204cfc9d4a16d9c27db4d7641c4ebf3e6fc98ae637ed71a9a7c50a4a32794e"
)
POPULATION_SHA256 = (
    "740b1186e893189606411ada3f0c7c59dc8e97ff5c54ed661ca5d714947a2c7c"
)
SEALED_MANIFEST_SHA256 = (
    "9a7451582831e87d812268db04791718f21d9366a041c205d4a5b132d905b19a"
)
GEOTESSERA_VERSION = "0.9.0"
DATASET_VERSION = "1.0"
DATASET_VARIANT = "vultr"
VERSION_PATH = "v1"
BASE_URL = "https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1"
CELL_MIN_VALID_FRACTION = 0.90
EVENT_MIN_VALID_FRACTION = 0.90
MIN_EVENTS_FOR_INFERENCE = 8
MAX_TRANSFER_BYTES = 150 * 1024**3
MAX_CACHE_BYTES = 175 * 1024**3
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_817
EMBEDDING_DIMENSION = 128

EXPECTED_EVENTS = [
    ("OK3585009921020180412", "RHEA", 2018),
    ("CA3424011870020181108", "WOOLSEY", 2018),
    ("AK6660614659920190622", "HADWEENZIC RIVER", 2019),
    ("OR4482112218820200816", "BEACHIE CREEK", 2020),
    ("MT4571710705620200902", "SARPY", 2020),
    ("ID4601211691120210707", "SHOVEL CREEK", 2021),
    ("CA4005312066920190904", "WALKER", 2019),
    ("NM3320410805920220514", "BLACK", 2022),
    ("OR4542811673920220830", "DOUBLE CREEK", 2022),
    ("CA3528611528620230728", "YORK", 2023),
]
FORBIDDEN_EVENT_ID = "KS3910009890020211215"


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    role: str
    url: str
    local_path: str
    content_length: int
    etag: str
    last_modified: str
    checksum_crc64nvme: str
    checksum_type: str
    tile_id: str
    lon: float
    lat: float
    year: int | None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crc64nvme_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    from awscrt import checksums

    value = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            value = checksums.crc64nvme(chunk, value)
    return base64.b64encode(value.to_bytes(8, "big")).decode("ascii")


def ordered_h3_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def normalize_etag(value: str | None) -> str | None:
    return None if value is None else value.strip().strip('"')


def http_head(url: str) -> dict[str, Any]:
    request = Request(
        url,
        method="HEAD",
        headers={"x-amz-checksum-mode": "ENABLED"},
    )
    with urlopen(request, timeout=120) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return {
            "http_status": int(response.status),
            "content_length": int(headers.get("content-length", "0")),
            "etag": normalize_etag(headers.get("etag")),
            "last_modified": headers.get("last-modified"),
            "checksum_crc64nvme": headers.get("x-amz-checksum-crc64nvme"),
            "checksum_type": headers.get("x-amz-checksum-type"),
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def decode_embeddings(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    quantized_float = quantized.astype(np.float32, copy=False)
    scale_float = scales.astype(np.float32, copy=False)
    if scale_float.ndim == quantized_float.ndim - 1:
        scale_float = scale_float[..., np.newaxis]
    if scale_float.ndim != quantized_float.ndim:
        raise ValueError("Producer scales are incompatible with embedding shape")
    return quantized_float * scale_float


def normalized_squared_displacement(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    before_norm = np.linalg.norm(before, axis=-1)
    after_norm = np.linalg.norm(after, axis=-1)
    valid = (
        np.isfinite(before).all(axis=-1)
        & np.isfinite(after).all(axis=-1)
        & np.isfinite(before_norm)
        & np.isfinite(after_norm)
        & (before_norm > 0.0)
        & (after_norm > 0.0)
    )
    values = np.full(before_norm.shape, np.nan, dtype=np.float32)
    if valid.any():
        before_unit = before[valid] / before_norm[valid, np.newaxis]
        after_unit = after[valid] / after_norm[valid, np.newaxis]
        values[valid] = np.sum(
            np.square(after_unit - before_unit),
            axis=-1,
            dtype=np.float32,
        )
    return values, valid


def decoded_vector_valid(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1)
    return (
        np.isfinite(values).all(axis=-1)
        & np.isfinite(norms)
        & (norms > 0.0)
    )


def event_tile_coordinates(h3_ids: list[str]) -> list[tuple[float, float]]:
    tiles: set[tuple[float, float]] = set()
    for h3_id in h3_ids:
        polygon = sg.Polygon(
            [(longitude, latitude) for latitude, longitude in h3.cell_to_boundary(h3_id)]
        )
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        for lon_index in range(
            math.floor(min_lon * 10),
            math.floor(max_lon * 10) + 1,
        ):
            for lat_index in range(
                math.floor(min_lat * 10),
                math.floor(max_lat * 10) + 1,
            ):
                lon = round(lon_index / 10 + 0.05, 2)
                lat = round(lat_index / 10 + 0.05, 2)
                footprint = sg.box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
                if polygon.intersection(footprint).area > 0.0:
                    tiles.add((lon, lat))
    return sorted(tiles, key=lambda item: (item[1], item[0]))


def verify_decoding() -> dict[str, Any]:
    from geotessera import dequantize_embedding

    quantized = (np.arange(4 * 128).reshape(2, 2, 128) % 127 - 63).astype(
        np.int8
    )
    scales = np.array([[0.125, 0.25], [0.5, 1.0]], dtype=np.float32)
    ours = decode_embeddings(quantized, scales)
    official = dequantize_embedding(quantized, scales)
    if not np.array_equal(ours, official):
        raise RuntimeError("Local decoding differs from official GeoTessera")
    return {
        "maximum_absolute_difference": float(
            np.max(np.abs(ours.astype(float) - official.astype(float)))
        ),
        "official_function": "geotessera.dequantize_embedding",
    }


def verify_contracts(
    protocol_path: Path,
    mtbs_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    if hashlib.sha256(protocol_raw).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("TESSERA protocol hash mismatch")
    if protocol_raw != canonical_json_bytes(protocol):
        raise RuntimeError("TESSERA protocol is not exact canonical JSON")

    freeze_path = mtbs_root / "freeze" / "canonical.json"
    population_path = mtbs_root / "outputs" / "population_freeze.json"
    sealed_manifest_path = mtbs_root / "outputs" / "sealed_manifest.json"
    expected_hashes = {
        freeze_path: MTBS_CONTRACT_SHA256,
        population_path: POPULATION_SHA256,
        sealed_manifest_path: SEALED_MANIFEST_SHA256,
    }
    verified_hashes: dict[str, str] = {}
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {path}")
        verified_hashes[str(path.resolve())] = actual

    population = load_json(population_path)
    sealed_manifest = load_json(sealed_manifest_path)
    population_events = population["events"]
    manifest_events = sealed_manifest["events"]
    actual_events = [
        (event["fire_id"], event["name"], int(event["event_year"]))
        for event in population_events
    ]
    if actual_events != EXPECTED_EVENTS:
        raise RuntimeError(f"Frozen event cohort mismatch: {actual_events}")
    if FORBIDDEN_EVENT_ID in {item[0] for item in actual_events}:
        raise RuntimeError("FOUR COUNTY re-entered the frozen cohort")
    if sealed_manifest.get("population_freeze_sha256") != POPULATION_SHA256:
        raise RuntimeError("Sealed manifest points to another population")
    if [event["fire_id"] for event in manifest_events] != [
        item[0] for item in EXPECTED_EVENTS
    ]:
        raise RuntimeError("Sealed manifest event order differs from population")

    manifest_by_id = {event["fire_id"]: event for event in manifest_events}
    for event in population_events:
        ids = [str(value) for value in event["h3_ids"]]
        if len(ids) != int(event["h3_count"]):
            raise RuntimeError(f"{event['name']}: H3 count mismatch")
        if ordered_h3_hash(ids) != event["h3_sha256"]:
            raise RuntimeError(f"{event['name']}: H3 order hash mismatch")
        if any(h3.get_resolution(value) != 8 for value in ids):
            raise RuntimeError(f"{event['name']}: non-H3-r8 cell")
        sealed = manifest_by_id[event["fire_id"]]
        if sealed["h3_sha256"] != event["h3_sha256"]:
            raise RuntimeError(f"{event['name']}: sealed H3 hash mismatch")
        sealed_path = mtbs_root / sealed["sealed_file"]
        if sha256_file(sealed_path) != sealed["sealed_sha256"]:
            raise RuntimeError(f"{event['name']}: sealed feature hash mismatch")

    installed = importlib.metadata.version("geotessera")
    if installed != GEOTESSERA_VERSION:
        raise RuntimeError(
            f"GeoTessera version mismatch: {installed} != {GEOTESSERA_VERSION}"
        )
    decode_audit = verify_decoding()
    verification = {
        "protocol_sha256": PROTOCOL_SHA256,
        "mtbs_contract_sha256": MTBS_CONTRACT_SHA256,
        "population_freeze_sha256": POPULATION_SHA256,
        "sealed_manifest_sha256": SEALED_MANIFEST_SHA256,
        "verified_file_hashes": verified_hashes,
        "events": [
            {
                "slot_rank": index,
                "fire_id": event["fire_id"],
                "name": event["name"],
                "event_year": event["event_year"],
                "h3_count": event["h3_count"],
                "h3_sha256": event["h3_sha256"],
            }
            for index, event in enumerate(population_events, start=1)
        ],
        "forbidden_event_absent": True,
        "geotessera_version": installed,
        "decoding": decode_audit,
        "python": sys.version.split()[0],
    }
    return population, sealed_manifest, verification


def local_object_path(cache_dir: Path, url: str) -> Path:
    marker = "/tessera-embeddings/"
    if marker not in url:
        raise ValueError(f"Unexpected TESSERA URL: {url}")
    return cache_dir / "raw" / url.split(marker, maxsplit=1)[1]


def object_id(
    role: str,
    lon: float,
    lat: float,
    year: int | None,
) -> str:
    return f"{role}|{year if year is not None else 'static'}|{lon:.2f}|{lat:.2f}"


def manifest_event_plan(
    event: dict[str, Any],
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    tile_coordinates = event_tile_coordinates(event["h3_ids"])
    years = [int(event["event_year"]) - 1, int(event["event_year"]) + 1]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for year in years:
        for lon, lat in tile_coordinates:
            matches = manifest[
                (manifest["version"].astype(str) == DATASET_VERSION)
                & (manifest["variant"].astype(str) == DATASET_VARIANT)
                & (manifest["year"].astype(int) == year)
                & np.isclose(manifest["lon"].astype(float), lon)
                & np.isclose(manifest["lat"].astype(float), lat)
            ]
            if len(matches) != 1:
                missing.append(
                    {
                        "year": year,
                        "lon": lon,
                        "lat": lat,
                        "matching_manifest_rows": int(len(matches)),
                    }
                )
                continue
            row = matches.iloc[0]
            rows.append(
                {
                    "year": year,
                    "lon": lon,
                    "lat": lat,
                    "tile_id": f"grid_{lon:.2f}_{lat:.2f}",
                    "grid_size": int(row["grid_size"]),
                    "scales_size": int(row["scales_size"]),
                    "grid_path": str(row["grid_path"]),
                    "scales_path": str(row["scales_path"]),
                }
            )
    return {
        "fire_id": event["fire_id"],
        "name": event["name"],
        "event_year": int(event["event_year"]),
        "pre_year": years[0],
        "post_year": years[1],
        "h3_count": int(event["h3_count"]),
        "h3_sha256": event["h3_sha256"],
        "tiles": [
            {"lon": lon, "lat": lat, "tile_id": f"grid_{lon:.2f}_{lat:.2f}"}
            for lon, lat in tile_coordinates
        ],
        "manifest_rows": rows,
        "missing_manifest_rows": missing,
        "manifest_compatible": not missing,
    }


def remote_object_candidates(
    event_plans: list[dict[str, Any]],
    cache_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    candidates: dict[str, dict[str, Any]] = {}
    owners: dict[str, set[str]] = {}
    for event in event_plans:
        if not event["manifest_compatible"]:
            continue
        fire_id = event["fire_id"]
        for row in event["manifest_rows"]:
            lon = float(row["lon"])
            lat = float(row["lat"])
            year = int(row["year"])
            tile_id = row["tile_id"]
            prefix = (
                f"{BASE_URL}/global_0.1_degree_representation/"
                f"{year}/{tile_id}"
            )
            for role, suffix, size in (
                ("embedding", ".npy", int(row["grid_size"])),
                ("scales", "_scales.npy", int(row["scales_size"])),
            ):
                identifier = object_id(role, lon, lat, year)
                url = f"{prefix}/{tile_id}{suffix}"
                candidates.setdefault(
                    identifier,
                    {
                        "object_id": identifier,
                        "role": role,
                        "url": url,
                        "local_path": str(local_object_path(cache_dir, url)),
                        "manifest_content_length": size,
                        "tile_id": tile_id,
                        "lon": lon,
                        "lat": lat,
                        "year": year,
                    },
                )
                owners.setdefault(identifier, set()).add(fire_id)
        for tile in event["tiles"]:
            lon = float(tile["lon"])
            lat = float(tile["lat"])
            tile_id = tile["tile_id"]
            identifier = object_id("landmask", lon, lat, None)
            url = f"{BASE_URL}/global_0.1_degree_tiff_all/{tile_id}.tiff"
            candidates.setdefault(
                identifier,
                {
                    "object_id": identifier,
                    "role": "landmask",
                    "url": url,
                    "local_path": str(local_object_path(cache_dir, url)),
                    "manifest_content_length": None,
                    "tile_id": tile_id,
                    "lon": lon,
                    "lat": lat,
                    "year": None,
                },
            )
            owners.setdefault(identifier, set()).add(fire_id)
    return candidates, owners


def lock_remote_objects(
    candidates: dict[str, dict[str, Any]],
    *,
    workers: int,
) -> tuple[dict[str, ObjectSpec], dict[str, str]]:
    locked: dict[str, ObjectSpec] = {}
    failures: dict[str, str] = {}

    def inspect(item: tuple[str, dict[str, Any]]) -> tuple[str, ObjectSpec]:
        identifier, candidate = item
        head = http_head(candidate["url"])
        if head["http_status"] != 200:
            raise RuntimeError(f"HTTP {head['http_status']}")
        expected_size = candidate["manifest_content_length"]
        if expected_size is not None and head["content_length"] != expected_size:
            raise RuntimeError(
                f"manifest/HEAD size mismatch {expected_size} != "
                f"{head['content_length']}"
            )
        if (
            head["checksum_type"] != "FULL_OBJECT"
            or not head["checksum_crc64nvme"]
        ):
            raise RuntimeError("missing full-object CRC64NVMe")
        return identifier, ObjectSpec(
            object_id=identifier,
            role=candidate["role"],
            url=candidate["url"],
            local_path=candidate["local_path"],
            content_length=int(head["content_length"]),
            etag=str(head["etag"]),
            last_modified=str(head["last_modified"]),
            checksum_crc64nvme=str(head["checksum_crc64nvme"]),
            checksum_type=str(head["checksum_type"]),
            tile_id=candidate["tile_id"],
            lon=float(candidate["lon"]),
            lat=float(candidate["lat"]),
            year=candidate["year"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(inspect, item): item[0]
            for item in candidates.items()
        }
        for future in concurrent.futures.as_completed(future_to_id):
            identifier = future_to_id[future]
            try:
                locked_id, spec = future.result()
                locked[locked_id] = spec
            except Exception as error:
                failures[identifier] = f"{type(error).__name__}: {error}"
    return locked, failures


def cache_logical_bytes(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in cache_dir.rglob("*")
        if path.is_file()
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    population, sealed_manifest, verification = verify_contracts(
        args.protocol,
        args.mtbs_root,
    )
    protocol = load_json(args.protocol)
    manifest_lock = protocol["data_lock"]["pilot_manifest"]
    if sha256_file(args.manifest) != manifest_lock["sha256"]:
        raise RuntimeError("Official TESSERA manifest local hash mismatch")
    manifest_head = http_head(manifest_lock["url"])
    expected_manifest_head = {
        "content_length": int(manifest_lock["content_length"]),
        "etag": manifest_lock["etag"],
        "last_modified": manifest_lock["last_modified"],
        "checksum_crc64nvme": manifest_lock["checksum_crc64nvme"],
        "checksum_type": "FULL_OBJECT",
    }
    for key, expected in expected_manifest_head.items():
        if manifest_head.get(key) != expected:
            raise RuntimeError(
                f"Official manifest metadata changed for {key}: "
                f"{manifest_head.get(key)} != {expected}"
            )
    manifest = pd.read_parquet(args.manifest)
    event_plans = [
        manifest_event_plan(event, manifest) for event in population["events"]
    ]
    candidates, owners = remote_object_candidates(event_plans, args.cache_dir)
    locked, head_failures = lock_remote_objects(
        candidates,
        workers=args.head_workers,
    )
    event_failure_reasons: dict[str, list[str]] = {}
    for event in event_plans:
        if event["missing_manifest_rows"]:
            event_failure_reasons.setdefault(event["fire_id"], []).append(
                "official manifest lacks exactly one 1.0/vultr row for one or "
                "more required (year,tile) pairs"
            )
    for identifier, failure in head_failures.items():
        for fire_id in owners[identifier]:
            event_failure_reasons.setdefault(fire_id, []).append(
                f"object metadata/access failure for {identifier}: {failure}"
            )

    evaluable_ids = {
        event["fire_id"]
        for event in event_plans
        if event["fire_id"] not in event_failure_reasons
    }
    required_object_ids = {
        identifier
        for identifier, object_owners in owners.items()
        if object_owners & evaluable_ids
        and identifier in locked
    }
    specs = [locked[identifier] for identifier in sorted(required_object_ids)]
    planned_transfer = sum(spec.content_length for spec in specs)
    if planned_transfer > MAX_TRANSFER_BYTES:
        raise RuntimeError(
            f"Frozen transfer cap exceeded: {planned_transfer} > "
            f"{MAX_TRANSFER_BYTES}"
        )
    existing_cache = cache_logical_bytes(args.cache_dir)
    missing_bytes = sum(
        spec.content_length
        for spec in specs
        if not Path(spec.local_path).is_file()
    )
    planned_cache = existing_cache + missing_bytes
    if planned_cache > MAX_CACHE_BYTES:
        raise RuntimeError(
            f"Frozen cache cap exceeded: {planned_cache} > {MAX_CACHE_BYTES}"
        )
    free_bytes = shutil.disk_usage(args.cache_dir.parent).free
    if free_bytes < missing_bytes + 5 * 1024**3:
        raise RuntimeError(
            f"Insufficient free disk before download: free={free_bytes}, "
            f"required_with_margin={missing_bytes + 5 * 1024**3}"
        )

    for event in event_plans:
        reasons = event_failure_reasons.get(event["fire_id"], [])
        event["preflight_status"] = (
            "evaluable_pending_download"
            if not reasons
            else "secondary_missing_access"
        )
        event["preflight_failure_reasons"] = sorted(set(reasons))
        event["required_object_ids"] = sorted(
            identifier
            for identifier, object_owners in owners.items()
            if event["fire_id"] in object_owners
            and identifier in required_object_ids
        )
        event["planned_object_bytes"] = sum(
            locked[identifier].content_length
            for identifier in event["required_object_ids"]
        )

    preflight = {
        "schema": "tessera_mtbs_secondary_preflight_v1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "verification": verification,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_lock["sha256"],
            "head": manifest_head,
            "dataset_version": DATASET_VERSION,
            "dataset_variant": DATASET_VARIANT,
        },
        "events": event_plans,
        "evaluable_event_count": len(evaluable_ids),
        "secondary_missing_preflight_count": len(event_plans) - len(evaluable_ids),
        "resource_gate": {
            "passed": True,
            "planned_unique_object_count": len(specs),
            "planned_transfer_bytes": planned_transfer,
            "maximum_transfer_bytes": MAX_TRANSFER_BYTES,
            "existing_cache_bytes": existing_cache,
            "planned_cache_bytes": planned_cache,
            "maximum_cache_bytes": MAX_CACHE_BYTES,
            "free_disk_bytes": free_bytes,
            "missing_download_bytes": missing_bytes,
        },
        "outcomes_accessed": False,
        "sealed_manifest_event_count": len(sealed_manifest["events"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "preflight.json", preflight)
    write_json(
        output_dir / "object_plan.json",
        {
            "schema": "tessera_mtbs_object_plan_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "population_freeze_sha256": POPULATION_SHA256,
            "objects": [asdict(spec) for spec in specs],
            "head_failures": head_failures,
        },
    )
    LOGGER.info(
        "Preflight complete: %d/%d events; %d objects; %.3f GiB",
        len(evaluable_ids),
        len(event_plans),
        len(specs),
        planned_transfer / 1024**3,
    )
    return preflight


def verify_local_object(spec: ObjectSpec) -> dict[str, Any]:
    path = Path(spec.local_path)
    if path.stat().st_size != spec.content_length:
        raise RuntimeError(f"Wrong local size for {spec.object_id}")
    crc = crc64nvme_file(path)
    if crc != spec.checksum_crc64nvme:
        raise RuntimeError(f"Wrong local CRC64NVMe for {spec.object_id}")
    return {
        "size_bytes": path.stat().st_size,
        "checksum_crc64nvme": crc,
        "sha256": sha256_file(path),
    }


def download_one(spec: ObjectSpec, *, retries: int = 5) -> dict[str, Any]:
    destination = Path(spec.local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return {
            "object_id": spec.object_id,
            "transferred_bytes": 0,
            "resumed_from_bytes": spec.content_length,
            "integrity": verify_local_object(spec),
            "cache_hit": True,
        }
    partial = destination.with_name(f"{destination.name}.part")
    transferred = 0
    initial_partial = partial.stat().st_size if partial.is_file() else 0
    if initial_partial > spec.content_length:
        partial.unlink()
        initial_partial = 0

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"x-amz-checksum-mode": "ENABLED"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(spec.url, method="GET", headers=headers)
        try:
            with urlopen(request, timeout=300) as response:
                status = int(response.status)
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        transferred += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size == spec.content_length:
                break
            raise RuntimeError(
                f"incomplete object {partial.stat().st_size}/"
                f"{spec.content_length}"
            )
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {spec.object_id}"
                ) from error
            time.sleep(min(2**attempt, 30))
    if partial.stat().st_size != spec.content_length:
        raise RuntimeError(f"Download size mismatch for {spec.object_id}")
    if crc64nvme_file(partial) != spec.checksum_crc64nvme:
        raise RuntimeError(f"Downloaded CRC64NVMe mismatch for {spec.object_id}")
    partial.replace(destination)
    return {
        "object_id": spec.object_id,
        "transferred_bytes": transferred,
        "resumed_from_bytes": initial_partial,
        "integrity": verify_local_object(spec),
        "cache_hit": False,
    }


def load_specs(output_dir: Path) -> list[ObjectSpec]:
    plan = load_json(output_dir / "object_plan.json")
    if plan["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("Object plan protocol mismatch")
    return [ObjectSpec(**item) for item in plan["objects"]]


def run_download(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = load_json(args.output_dir / "preflight.json")
    if not preflight["resource_gate"]["passed"]:
        raise RuntimeError("Resource preflight did not pass")
    specs = load_specs(args.output_dir)
    checkpoint_path = args.output_dir / "download_checkpoint.json"
    checkpoint = (
        load_json(checkpoint_path)
        if checkpoint_path.is_file()
        else {
            "schema": "tessera_mtbs_download_checkpoint_v1",
            "objects": {},
            "actual_transfer_bytes": 0,
        }
    )

    def pending(spec: ObjectSpec) -> bool:
        record = checkpoint["objects"].get(spec.object_id)
        return not (
            record
            and record.get("status") == "complete"
            and Path(spec.local_path).is_file()
            and Path(spec.local_path).stat().st_size == spec.content_length
        )

    pending_specs = [spec for spec in specs if pending(spec)]
    LOGGER.info("Downloading/verifying %d pending objects", len(pending_specs))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as executor:
        future_to_spec = {
            executor.submit(download_one, spec): spec for spec in pending_specs
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            result = future.result()
            checkpoint["objects"][spec.object_id] = {
                "status": "complete",
                **result,
            }
            checkpoint["actual_transfer_bytes"] += int(
                result["transferred_bytes"]
            )
            checkpoint["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            write_json(checkpoint_path, checkpoint)
            completed += 1
            LOGGER.info(
                "Object %d/%d complete: %s",
                completed,
                len(pending_specs),
                spec.object_id,
            )

    incomplete = [
        spec.object_id
        for spec in specs
        if checkpoint["objects"].get(spec.object_id, {}).get("status")
        != "complete"
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete object downloads: {incomplete}")
    summary = {
        "schema": "tessera_mtbs_download_summary_v1",
        "object_count": len(specs),
        "actual_transfer_bytes": int(checkpoint["actual_transfer_bytes"]),
        "logical_cache_bytes": cache_logical_bytes(args.cache_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "objects": [
            {
                **asdict(spec),
                **checkpoint["objects"][spec.object_id],
            }
            for spec in specs
        ],
    }
    write_json(args.output_dir / "object_inventory.json", summary)
    return summary


def project_geometry(geometry: Any, target_crs: Any) -> Any:
    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )
    return so.transform(transformer.transform, geometry)


def selected_scales(
    scales: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    if scales.ndim == 2:
        return np.asarray(scales[rows, columns])
    return np.asarray(scales[rows, columns, :])


def scale_valid(scales: np.ndarray) -> np.ndarray:
    valid = np.isfinite(scales) & (scales > 0.0)
    return valid if valid.ndim == 1 else valid.all(axis=-1)


def spec_lookup(specs: list[ObjectSpec]) -> dict[tuple[str, str, int | None], ObjectSpec]:
    return {
        (spec.tile_id, spec.role, spec.year): spec
        for spec in specs
    }


def verify_mtbs_source(
    mtbs_root: Path,
    fire_id: str,
) -> tuple[Path, dict[str, Any]]:
    source_manifest_path = mtbs_root / "outputs" / "source_manifest.json"
    source_manifest = load_json(source_manifest_path)
    relative = f"raw/mtbs/{fire_id}/severity.tif"
    matches = [
        item
        for item in source_manifest["entries"]
        if item.get("relative_path") == relative
        and item.get("kind") == "mtbs_severity_geotiff"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{fire_id}: sealed severity source not unique")
    path = mtbs_root / relative
    actual = sha256_file(path)
    if actual != matches[0]["sha256"]:
        raise RuntimeError(f"{fire_id}: sealed severity hash mismatch")
    return path, {
        "relative_path": relative,
        "sha256": actual,
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }


def aggregate_event(
    event: dict[str, Any],
    specs: list[ObjectSpec],
    *,
    mtbs_root: Path,
    chunk_pixels: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from geotessera import dequantize_embedding

    started = time.perf_counter()
    h3_ids = [str(value) for value in event["h3_ids"]]
    polygons = [
        sg.Polygon(
            [(longitude, latitude) for latitude, longitude in h3.cell_to_boundary(value)]
        )
        for value in h3_ids
    ]
    severity_path, severity_provenance = verify_mtbs_source(
        mtbs_root,
        event["fire_id"],
    )
    lookup = spec_lookup(specs)
    n_cells = len(h3_ids)
    candidate_counts = np.zeros(n_cells + 1, dtype=np.int64)
    valid_counts = np.zeros(n_cells + 1, dtype=np.int64)
    displacement_sums = np.zeros(n_cells + 1, dtype=np.float64)
    tile_audits: list[dict[str, Any]] = []
    pre_year = int(event["event_year"]) - 1
    post_year = int(event["event_year"]) + 1

    with rasterio.open(severity_path) as severity:
        for tile in event["tiles"]:
            tile_id = tile["tile_id"]
            lon = float(tile["lon"])
            lat = float(tile["lat"])
            pre_grid = lookup[(tile_id, "embedding", pre_year)]
            pre_scale = lookup[(tile_id, "scales", pre_year)]
            post_grid = lookup[(tile_id, "embedding", post_year)]
            post_scale = lookup[(tile_id, "scales", post_year)]
            landmask_spec = lookup[(tile_id, "landmask", None)]
            pre_q = np.load(pre_grid.local_path, mmap_mode="r")
            pre_scales = np.load(pre_scale.local_path, mmap_mode="r")
            post_q = np.load(post_grid.local_path, mmap_mode="r")
            post_scales = np.load(post_scale.local_path, mmap_mode="r")
            if (
                pre_q.dtype != np.int8
                or post_q.dtype != np.int8
                or pre_q.shape != post_q.shape
                or pre_q.ndim != 3
                or pre_q.shape[-1] != EMBEDDING_DIMENSION
            ):
                raise RuntimeError(f"{event['name']} {tile_id}: embedding mismatch")
            if (
                pre_scales.dtype != np.float32
                or post_scales.dtype != np.float32
            ):
                raise RuntimeError(f"{event['name']} {tile_id}: scale dtype mismatch")
            with rasterio.open(landmask_spec.local_path) as landmask:
                shape = (landmask.height, landmask.width)
                transform = landmask.transform
                crs = landmask.crs
            if crs is None or pre_q.shape[:2] != shape:
                raise RuntimeError(f"{event['name']} {tile_id}: grid mismatch")

            footprint = sg.box(
                lon - 0.05,
                lat - 0.05,
                lon + 0.05,
                lat + 0.05,
            )
            projected_cells: list[tuple[Any, int]] = []
            for index, polygon in enumerate(polygons, start=1):
                clipped = polygon.intersection(footprint)
                if not clipped.is_empty and clipped.area > 0.0:
                    projected_cells.append(
                        (project_geometry(clipped, crs), index)
                    )
            labels = rasterize(
                projected_cells,
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.int32,
                all_touched=False,
            )
            tile_mask = rasterize(
                [(project_geometry(footprint, crs), 1)],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            ).astype(bool)
            severity_values = np.zeros(shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(severity, 1),
                destination=severity_values,
                src_transform=severity.transform,
                src_crs=severity.crs,
                src_nodata=severity.nodata,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                num_threads=2,
            )
            support = (
                tile_mask
                & (labels > 0)
                & (severity_values >= 1)
                & (severity_values <= 5)
            )
            tile_candidate = np.bincount(
                labels[support],
                minlength=n_cells + 1,
            )
            candidate_counts += tile_candidate
            support_rows, support_columns = np.nonzero(support)
            tile_valid = 0
            reason_counts = {
                "pre_invalid_scale": 0,
                "post_invalid_scale": 0,
                "pre_zero_quantized_vector": 0,
                "post_zero_quantized_vector": 0,
                "pre_invalid_decoded_vector": 0,
                "post_invalid_decoded_vector": 0,
            }
            decode_difference = 0.0
            for start in range(0, len(support_rows), chunk_pixels):
                stop = min(start + chunk_pixels, len(support_rows))
                rows = support_rows[start:stop]
                columns = support_columns[start:stop]
                pre_q_chunk = np.asarray(pre_q[rows, columns, :])
                post_q_chunk = np.asarray(post_q[rows, columns, :])
                pre_s_chunk = selected_scales(pre_scales, rows, columns)
                post_s_chunk = selected_scales(post_scales, rows, columns)
                pre_s_valid = scale_valid(pre_s_chunk)
                post_s_valid = scale_valid(post_s_chunk)
                pre_zero = (pre_q_chunk == 0).all(axis=-1)
                post_zero = (post_q_chunk == 0).all(axis=-1)
                pre_decoded = decode_embeddings(pre_q_chunk, pre_s_chunk)
                post_decoded = decode_embeddings(post_q_chunk, post_s_chunk)
                displacement, vector_valid = normalized_squared_displacement(
                    pre_decoded,
                    post_decoded,
                )
                pair_valid = (
                    pre_s_valid
                    & post_s_valid
                    & ~pre_zero
                    & ~post_zero
                    & vector_valid
                )
                if start == 0 and len(rows):
                    if pre_s_chunk.ndim == 1:
                        official = dequantize_embedding(
                            pre_q_chunk[:4, np.newaxis, :],
                            pre_s_chunk[:4, np.newaxis],
                        )[:, 0, :]
                    else:
                        official = dequantize_embedding(
                            pre_q_chunk[:4],
                            pre_s_chunk[:4],
                        )
                    ours = pre_decoded[:4]
                    decode_difference = float(
                        np.max(np.abs(ours.astype(float) - official.astype(float)))
                    )
                    if not np.array_equal(ours, official):
                        raise RuntimeError(
                            f"{event['name']} {tile_id}: official decode mismatch"
                        )
                reason_counts["pre_invalid_scale"] += int((~pre_s_valid).sum())
                reason_counts["post_invalid_scale"] += int((~post_s_valid).sum())
                reason_counts["pre_zero_quantized_vector"] += int(pre_zero.sum())
                reason_counts["post_zero_quantized_vector"] += int(post_zero.sum())
                reason_counts["pre_invalid_decoded_vector"] += int(
                    (~decoded_vector_valid(pre_decoded)).sum()
                )
                reason_counts["post_invalid_decoded_vector"] += int(
                    (~decoded_vector_valid(post_decoded)).sum()
                )
                if pair_valid.any():
                    valid_rows = rows[pair_valid]
                    valid_columns = columns[pair_valid]
                    valid_labels = labels[valid_rows, valid_columns]
                    valid_values = displacement[pair_valid].astype(
                        np.float64,
                        copy=False,
                    )
                    valid_counts += np.bincount(
                        valid_labels,
                        minlength=n_cells + 1,
                    )
                    displacement_sums += np.bincount(
                        valid_labels,
                        weights=valid_values,
                        minlength=n_cells + 1,
                    )
                    tile_valid += int(pair_valid.sum())
            tile_audits.append(
                {
                    "tile_id": tile_id,
                    "lon": lon,
                    "lat": lat,
                    "pre_year": pre_year,
                    "post_year": post_year,
                    "shape": list(pre_q.shape),
                    "crs": str(crs),
                    "transform": list(transform)[:6],
                    "candidate_pixel_count": int(tile_candidate.sum()),
                    "paired_valid_pixel_count": tile_valid,
                    "missing_pair_pixel_count": (
                        int(tile_candidate.sum()) - tile_valid
                    ),
                    "invalid_reason_counts_nonexclusive": reason_counts,
                    "decode_official_max_abs_difference": decode_difference,
                }
            )

    candidate = candidate_counts[1:]
    valid = valid_counts[1:]
    if (candidate <= 0).any():
        missing_ids = [
            h3_ids[index]
            for index in np.flatnonzero(candidate <= 0)
        ]
        raise RuntimeError(
            f"{event['name']}: sealed H3 cells lack candidate pixels: "
            f"{missing_ids[:10]}"
        )
    valid_fraction = valid / candidate
    cell_pass = valid_fraction >= CELL_MIN_VALID_FRACTION
    pooled_fraction = float(valid.sum() / candidate.sum())
    coverage_pass = bool(
        cell_pass.all() and pooled_fraction >= EVENT_MIN_VALID_FRACTION
    )
    means = np.divide(
        displacement_sums[1:],
        valid,
        out=np.full(n_cells, np.nan, dtype=float),
        where=valid > 0,
    )
    frame = pd.DataFrame(
        {
            "h3_index": h3_ids,
            "tessera_candidate_pixel_count": candidate,
            "tessera_paired_valid_pixel_count": valid,
            "tessera_paired_valid_fraction": valid_fraction,
            "tessera_missing_pair_fraction": 1.0 - valid_fraction,
            "tessera_cell_coverage_pass": cell_pass,
            "tessera_mean_normalized_squared_displacement": means,
        }
    )
    audit = {
        "schema": "tessera_mtbs_event_aggregation_v1",
        "fire_id": event["fire_id"],
        "name": event["name"],
        "event_year": event["event_year"],
        "pre_year": pre_year,
        "post_year": post_year,
        "h3_count": n_cells,
        "h3_sha256": ordered_h3_hash(h3_ids),
        "candidate_pixel_count": int(candidate.sum()),
        "paired_valid_pixel_count": int(valid.sum()),
        "pooled_paired_valid_fraction": pooled_fraction,
        "cells_passing_coverage": int(cell_pass.sum()),
        "cells_failing_coverage": int((~cell_pass).sum()),
        "minimum_cell_paired_valid_fraction": float(valid_fraction.min()),
        "coverage_pass": coverage_pass,
        "coverage_rule": {
            "minimum_cell_paired_valid_fraction": CELL_MIN_VALID_FRACTION,
            "minimum_pooled_event_paired_valid_fraction": (
                EVENT_MIN_VALID_FRACTION
            ),
            "every_sealed_h3_cell_must_pass": True,
            "no_imputation": True,
        },
        "severity_provenance": severity_provenance,
        "tile_audits": tile_audits,
        "runtime_seconds": time.perf_counter() - started,
        "association_computed": False,
    }
    return frame, audit


def run_process(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = load_json(args.output_dir / "preflight.json")
    specs = load_specs(args.output_dir)
    inventory = load_json(args.output_dir / "object_inventory.json")
    if len(inventory["objects"]) != len(specs):
        raise RuntimeError("Object inventory is incomplete")
    population = load_json(args.mtbs_root / "outputs" / "population_freeze.json")
    population_by_id = {
        event["fire_id"]: event for event in population["events"]
    }
    spec_by_id = {spec.object_id: spec for spec in specs}
    process_dir = args.output_dir / "events"
    process_dir.mkdir(parents=True, exist_ok=True)
    event_summaries: list[dict[str, Any]] = []
    for event_plan in preflight["events"]:
        fire_id = event_plan["fire_id"]
        if event_plan["preflight_status"] != "evaluable_pending_download":
            event_summaries.append(
                {
                    "fire_id": fire_id,
                    "name": event_plan["name"],
                    "status": "secondary_missing_access",
                    "reasons": event_plan["preflight_failure_reasons"],
                }
            )
            continue
        checkpoint_path = process_dir / f"{fire_id}.json"
        parquet_path = process_dir / f"{fire_id}.parquet"
        if checkpoint_path.is_file() and parquet_path.is_file():
            existing = load_json(checkpoint_path)
            if (
                existing.get("h3_sha256") == event_plan["h3_sha256"]
                and existing.get("parquet_sha256") == sha256_file(parquet_path)
                and existing.get("status")
                in {"complete", "secondary_missing_coverage"}
            ):
                event_summaries.append(existing)
                LOGGER.info("Reusing complete event checkpoint: %s", event_plan["name"])
                continue
        event_specs = [
            spec_by_id[identifier]
            for identifier in event_plan["required_object_ids"]
        ]
        event = {
            **population_by_id[fire_id],
            "tiles": event_plan["tiles"],
        }
        frame, audit = aggregate_event(
            event,
            event_specs,
            mtbs_root=args.mtbs_root,
            chunk_pixels=args.chunk_pixels,
        )
        frame.to_parquet(parquet_path, index=False)
        status = (
            "complete"
            if audit["coverage_pass"]
            else "secondary_missing_coverage"
        )
        checkpoint = {
            **audit,
            "status": status,
            "parquet_path": str(parquet_path.resolve()),
            "parquet_sha256": sha256_file(parquet_path),
            "protocol_sha256": PROTOCOL_SHA256,
            "population_freeze_sha256": POPULATION_SHA256,
        }
        write_json(checkpoint_path, checkpoint)
        event_summaries.append(checkpoint)
        LOGGER.info(
            "Processed %s: %s, coverage=%.6f",
            event_plan["name"],
            status,
            audit["pooled_paired_valid_fraction"],
        )
    expected_ids = [event["fire_id"] for event in preflight["events"]]
    if [event["fire_id"] for event in event_summaries] != expected_ids:
        raise RuntimeError("Process summary event order changed")
    event_compute_seconds = sum(
        float(event.get("runtime_seconds", 0.0))
        for event in event_summaries
    )
    summary_refresh_seconds = time.perf_counter() - started
    summary = {
        "schema": "tessera_mtbs_processing_summary_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "population_freeze_sha256": POPULATION_SHA256,
        "all_events_processed_before_analysis": True,
        "events": event_summaries,
        "complete_event_count": sum(
            event["status"] == "complete" for event in event_summaries
        ),
        "secondary_missing_access_count": sum(
            event["status"] == "secondary_missing_access"
            for event in event_summaries
        ),
        "secondary_missing_coverage_count": sum(
            event["status"] == "secondary_missing_coverage"
            for event in event_summaries
        ),
        "candidate_pixel_count": sum(
            int(event.get("candidate_pixel_count", 0))
            for event in event_summaries
        ),
        "paired_valid_pixel_count": sum(
            int(event.get("paired_valid_pixel_count", 0))
            for event in event_summaries
        ),
        "event_compute_seconds": event_compute_seconds,
        "summary_invocation_seconds": summary_refresh_seconds,
        "runtime_seconds": event_compute_seconds,
        "analysis_started": False,
    }
    summary["pooled_paired_valid_fraction"] = (
        summary["paired_valid_pixel_count"] / summary["candidate_pixel_count"]
        if summary["candidate_pixel_count"]
        else None
    )
    write_json(args.output_dir / "processing_summary.json", summary)
    return summary


def validate_analysis_ready(
    preflight: dict[str, Any],
    processing: dict[str, Any],
) -> None:
    if not processing.get("all_events_processed_before_analysis"):
        raise RuntimeError("All-event processing barrier is not complete")
    expected = [event["fire_id"] for event in preflight["events"]]
    observed = [event["fire_id"] for event in processing["events"]]
    if observed != expected:
        raise RuntimeError("Processing barrier event order mismatch")
    allowed = {
        "complete",
        "secondary_missing_access",
        "secondary_missing_coverage",
    }
    if any(event["status"] not in allowed for event in processing["events"]):
        raise RuntimeError("An event lacks a terminal processing status")


def correlation(values: np.ndarray, reference: np.ndarray) -> float:
    if not (
        np.isfinite(values).all()
        and np.isfinite(reference).all()
        and np.ptp(values) > 0.0
        and np.ptp(reference) > 0.0
    ):
        raise RuntimeError("Correlation input is nonfinite or constant")
    return float(spearmanr(values, reference).statistic)


def bootstrap_mean(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "method": "nonparametric percentile event bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def paper_recommendation(inference: dict[str, Any]) -> str:
    secondary = inference.get("secondary_endpoint")
    if secondary is None:
        return (
            "Do not add a cohort-level TESSERA claim to the paper: fewer than "
            "eight events were evaluable. If disclosed, report only the "
            "prespecified event values and access/coverage missingness."
        )
    mean = float(secondary["unweighted_mean_s_e"])
    lower = float(secondary["bootstrap"]["lower"])
    upper = float(secondary["bootstrap"]["upper"])
    if lower > 0.0:
        result = (
            "The prespecified secondary interval is wholly positive. A compact "
            "secondary-result paragraph is supportable"
        )
    elif upper < 0.0:
        result = (
            "The prespecified secondary interval is wholly negative. Report "
            "the negative secondary result rather than a favorable comparator claim"
        )
    else:
        result = (
            "The prespecified secondary interval includes zero. Report the "
            "secondary comparison as inconclusive, not as evidence of superiority"
        )
    return (
        f"{result} (mean S_e={mean:.6f}, 95% interval "
        f"[{lower:.6f}, {upper:.6f}]). Keep the frozen AlphaEarth primary "
        "endpoint and conclusion unchanged; do not compare raw model magnitudes."
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = load_json(args.output_dir / "preflight.json")
    processing = load_json(args.output_dir / "processing_summary.json")
    validate_analysis_ready(preflight, processing)
    sealed_manifest = load_json(args.mtbs_root / "outputs" / "sealed_manifest.json")
    sealed_by_id = {
        event["fire_id"]: event for event in sealed_manifest["events"]
    }
    process_by_id = {
        event["fire_id"]: event for event in processing["events"]
    }
    rows: list[dict[str, Any]] = []
    per_h3_frames: list[pd.DataFrame] = []
    for event_plan in preflight["events"]:
        fire_id = event_plan["fire_id"]
        process = process_by_id[fire_id]
        base = {
            "slot_rank": EXPECTED_EVENTS.index(
                (
                    fire_id,
                    event_plan["name"],
                    int(event_plan["event_year"]),
                )
            )
            + 1,
            "fire_id": fire_id,
            "name": event_plan["name"],
            "event_year": event_plan["event_year"],
            "h3_count": event_plan["h3_count"],
            "status": process["status"],
            "missing_reasons": "; ".join(process.get("reasons", [])),
            "rho_tessera": np.nan,
            "rho_dnbr": np.nan,
            "rho_dndvi_loss": np.nan,
            "rho_alphaearth": np.nan,
            "s_e": np.nan,
            "a_e": np.nan,
            "pooled_paired_valid_fraction": process.get(
                "pooled_paired_valid_fraction",
                np.nan,
            ),
            "minimum_cell_paired_valid_fraction": process.get(
                "minimum_cell_paired_valid_fraction",
                np.nan,
            ),
        }
        sealed_entry = sealed_by_id[fire_id]
        sealed_path = args.mtbs_root / sealed_entry["sealed_file"]
        if sha256_file(sealed_path) != sealed_entry["sealed_sha256"]:
            raise RuntimeError(f"{event_plan['name']}: sealed hash changed")
        primary = pd.read_parquet(sealed_path)
        primary_ids = primary["h3_index"].astype(str).tolist()
        if ordered_h3_hash(primary_ids) != event_plan["h3_sha256"]:
            raise RuntimeError(
                f"{event_plan['name']}: sealed analysis population mismatch"
            )
        reference = primary["mtbs_reference"].to_numpy(dtype=float)
        rho_dnbr = correlation(
            primary["dnbr"].to_numpy(dtype=float),
            reference,
        )
        rho_dndvi = correlation(
            primary["dndvi_loss"].to_numpy(dtype=float),
            reference,
        )
        rho_alphaearth = correlation(
            primary["ecs"].to_numpy(dtype=float),
            reference,
        )
        base.update(
            {
                "rho_dnbr": rho_dnbr,
                "rho_dndvi_loss": rho_dndvi,
                "rho_alphaearth": rho_alphaearth,
            }
        )
        if process["status"] != "complete":
            if process["status"] == "secondary_missing_coverage":
                base["missing_reasons"] = "fixed paired-valid coverage gate failed"
            rows.append(base)
            continue
        tessera_path = Path(process["parquet_path"])
        if sha256_file(tessera_path) != process["parquet_sha256"]:
            raise RuntimeError(f"{event_plan['name']}: TESSERA output changed")
        tessera = pd.read_parquet(tessera_path)
        tessera_ids = tessera["h3_index"].astype(str).tolist()
        if (
            primary_ids != tessera_ids
            or ordered_h3_hash(primary_ids) != event_plan["h3_sha256"]
        ):
            raise RuntimeError(f"{event_plan['name']}: analysis population mismatch")
        if not tessera["tessera_cell_coverage_pass"].all():
            raise RuntimeError(f"{event_plan['name']}: failed cell entered analysis")
        reference = primary["mtbs_reference"].to_numpy(dtype=float)
        tessera_values = tessera[
            "tessera_mean_normalized_squared_displacement"
        ].to_numpy(dtype=float)
        rho_tessera = correlation(tessera_values, reference)
        s_e = rho_tessera - max(rho_dnbr, rho_dndvi)
        a_e = rho_alphaearth - rho_tessera
        base.update(
            {
                "rho_tessera": rho_tessera,
                "rho_dnbr": rho_dnbr,
                "rho_dndvi_loss": rho_dndvi,
                "rho_alphaearth": rho_alphaearth,
                "s_e": s_e,
                "a_e": a_e,
            }
        )
        rows.append(base)
        joined = primary[
            [
                "h3_index",
                "mtbs_reference",
                "dnbr",
                "dndvi_loss",
                "ecs",
            ]
        ].copy()
        joined.insert(0, "event_year", event_plan["event_year"])
        joined.insert(0, "name", event_plan["name"])
        joined.insert(0, "fire_id", fire_id)
        for column in tessera.columns:
            if column != "h3_index":
                joined[column] = tessera[column].to_numpy()
        per_h3_frames.append(joined)

    frame = pd.DataFrame(rows)
    primary_metrics_path = (
        args.mtbs_root / "outputs" / "analysis" / "per_event_metrics.json"
    )
    primary_metrics = load_json(primary_metrics_path)
    frozen_primary_by_id = {
        event["fire_id"]: event for event in primary_metrics["events"]
    }
    primary_differences: list[float] = []
    for row in frame.itertuples(index=False):
        frozen = frozen_primary_by_id[row.fire_id]
        comparisons = (
            (row.rho_dnbr, frozen["rho_dnbr_reference"]),
            (row.rho_dndvi_loss, frozen["rho_dndvi_loss_reference"]),
            (row.rho_alphaearth, frozen["rho_ecs_reference"]),
        )
        for observed, expected in comparisons:
            difference = abs(float(observed) - float(expected))
            primary_differences.append(difference)
            if difference > 1e-12:
                raise RuntimeError(
                    f"{row.name}: recomputed primary metric differs from "
                    "the frozen analysis"
                )
    primary_metrics_verification = {
        "path": str(primary_metrics_path.resolve()),
        "sha256": sha256_file(primary_metrics_path),
        "event_count": len(primary_metrics["events"]),
        "maximum_absolute_correlation_difference": max(primary_differences),
        "verified": True,
    }
    evaluable = frame[frame["status"] == "complete"].copy()
    m = len(evaluable)
    inference: dict[str, Any] = {
        "evaluable_event_count": m,
        "planned_event_count": len(frame),
        "minimum_events_required": MIN_EVENTS_FOR_INFERENCE,
        "cohort_inference_permitted": m >= MIN_EVENTS_FOR_INFERENCE,
        "unit": "event",
    }
    if m >= MIN_EVENTS_FOR_INFERENCE:
        s_values = evaluable["s_e"].to_numpy(dtype=float)
        a_values = evaluable["a_e"].to_numpy(dtype=float)
        inference["secondary_endpoint"] = {
            "event_statistic": (
                "S_e=rho_TESSERA-max(rho_dNBR,rho_dNDVI_loss)"
            ),
            "unweighted_mean_s_e": float(s_values.mean()),
            "bootstrap": bootstrap_mean(s_values),
            "positive_event_count": int((s_values > 0).sum()),
            "zero_event_count": int((s_values == 0).sum()),
            "negative_event_count": int((s_values < 0).sum()),
            "multiplicity_label": "one prespecified secondary endpoint",
        }
        inference["descriptive_alphaearth_tessera"] = {
            "event_statistic": "A_e=rho_AlphaEarth-rho_TESSERA",
            "unweighted_mean_a_e": float(a_values.mean()),
            "bootstrap": bootstrap_mean(a_values),
            "label": "descriptive only; no model-ranking claim",
        }
    else:
        inference["secondary_endpoint"] = None
        inference["descriptive_alphaearth_tessera"] = None
    recommendation = paper_recommendation(inference)
    write_json(args.output_dir / "inference.json", inference)
    frame.to_csv(args.output_dir / "per_event_metrics.csv", index=False)
    write_json(
        args.output_dir / "per_event_metrics.json",
        {
            "schema": "tessera_mtbs_per_event_metrics_v1",
            "events": frame.replace({np.nan: None}).to_dict(orient="records"),
        },
    )
    if per_h3_frames:
        pd.concat(per_h3_frames, ignore_index=True).to_parquet(
            args.output_dir / "per_h3_compact.parquet",
            index=False,
        )
    make_figure(frame, args.output_dir / "tessera_mtbs_secondary.png")
    analysis_runtime = time.perf_counter() - started
    inventory_summary = load_json(args.output_dir / "object_inventory.json")
    phase_runtime = {
        "preflight_seconds": float(preflight["runtime_seconds"]),
        "download_seconds": float(inventory_summary["elapsed_seconds"]),
        "processing_compute_seconds": float(
            processing["event_compute_seconds"]
        ),
        "analysis_seconds": analysis_runtime,
    }
    analysis = {
        "schema": "tessera_mtbs_secondary_analysis_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "mtbs_contract_sha256": MTBS_CONTRACT_SHA256,
        "population_freeze_sha256": POPULATION_SHA256,
        "sealed_manifest_sha256": SEALED_MANIFEST_SHA256,
        "all_events_processed_before_analysis": True,
        "primary_metrics_verification": primary_metrics_verification,
        "execution_source": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "test_path": str(
                (ROOT / "tests" / "test_tessera_mtbs_secondary.py").resolve()
            ),
            "test_sha256": sha256_file(
                ROOT / "tests" / "test_tessera_mtbs_secondary.py"
            ),
        },
        "inference": inference,
        "paper_recommendation": recommendation,
        "runtime_seconds": analysis_runtime,
        "phase_runtime": {
            **phase_runtime,
            "end_to_end_phase_seconds": sum(phase_runtime.values()),
        },
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_cross_model_magnitudes_compared": False,
    }
    write_json(args.output_dir / "analysis_summary.json", analysis)
    write_report(args, preflight, processing, frame, inference, analysis)
    write_artifact_manifest(args.output_dir)
    return analysis


def make_figure(frame: pd.DataFrame, path: Path) -> None:
    evaluable = frame[frame["status"] == "complete"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(evaluable))
    width = 0.2
    axes[0].bar(x - 1.5 * width, evaluable["rho_tessera"], width, label="TESSERA")
    axes[0].bar(x - 0.5 * width, evaluable["rho_alphaearth"], width, label="AlphaEarth")
    axes[0].bar(x + 0.5 * width, evaluable["rho_dnbr"], width, label="dNBR")
    axes[0].bar(
        x + 1.5 * width,
        evaluable["rho_dndvi_loss"],
        width,
        label="dNDVI loss",
    )
    axes[0].set_xticks(x, evaluable["name"], rotation=45, ha="right")
    axes[0].set_ylabel("Spearman rho with sealed MTBS reference")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=8)
    colors = np.where(evaluable["s_e"] >= 0, "#3b8c6e", "#b14a4a")
    axes[1].bar(x, evaluable["s_e"], color=colors)
    axes[1].set_xticks(x, evaluable["name"], rotation=45, ha="right")
    axes[1].set_ylabel("S_e (TESSERA - stronger optical rho)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    figure.suptitle(
        "Frozen TESSERA v1 secondary comparator (event-level, not model ranking)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    args: argparse.Namespace,
    preflight: dict[str, Any],
    processing: dict[str, Any],
    frame: pd.DataFrame,
    inference: dict[str, Any],
    analysis: dict[str, Any],
) -> None:
    inventory = load_json(args.output_dir / "object_inventory.json")
    lines = [
        "# Frozen TESSERA v1 secondary comparator on MTBS",
        "",
        "This analysis executed the prospectively frozen secondary amendment. "
        "It does not alter the MTBS cohort, primary endpoint, replacements, "
        "H3 populations, manuscript, or primary artifact.",
        "",
        "## Integrity",
        "",
        f"- TESSERA protocol SHA-256: `{PROTOCOL_SHA256}`.",
        f"- MTBS contract SHA-256: `{MTBS_CONTRACT_SHA256}`.",
        f"- Population freeze SHA-256: `{POPULATION_SHA256}`.",
        f"- Sealed manifest SHA-256: `{SEALED_MANIFEST_SHA256}`.",
        f"- Evaluable secondary events: {inference['evaluable_event_count']}/10.",
        "- Recomputed sealed optical/AlphaEarth correlations matched the "
        "frozen primary metrics with maximum absolute difference "
        f"{analysis['primary_metrics_verification']['maximum_absolute_correlation_difference']:.3g}.",
        "",
        "## Per-event results",
        "",
    ]
    for row in frame.itertuples(index=False):
        if row.status == "complete":
            lines.append(
                f"- **{row.name}**: rho_TESSERA={row.rho_tessera:.6f}, "
                f"rho_dNBR={row.rho_dnbr:.6f}, "
                f"rho_dNDVI_loss={row.rho_dndvi_loss:.6f}, "
                f"rho_AlphaEarth={row.rho_alphaearth:.6f}, "
                f"S_e={row.s_e:.6f}, A_e={row.a_e:.6f}; "
                f"pooled coverage={row.pooled_paired_valid_fraction:.6f}."
            )
        else:
            lines.append(
                f"- **{row.name}**: `{row.status}` — {row.missing_reasons}; "
                f"rho_dNBR={row.rho_dnbr:.6f}, "
                f"rho_dNDVI_loss={row.rho_dndvi_loss:.6f}, "
                f"rho_AlphaEarth={row.rho_alphaearth:.6f}; "
                "rho_TESSERA and S_e are missing by protocol."
            )
    lines.extend(["", "## Frozen event-level inference", ""])
    secondary = inference.get("secondary_endpoint")
    if secondary:
        bootstrap = secondary["bootstrap"]
        lines.extend(
            [
                f"- Unweighted mean S_e: "
                f"{secondary['unweighted_mean_s_e']:.6f}.",
                f"- 95% event-bootstrap interval: "
                f"[{bootstrap['lower']:.6f}, {bootstrap['upper']:.6f}].",
                f"- Event signs (+/0/-): "
                f"{secondary['positive_event_count']}/"
                f"{secondary['zero_event_count']}/"
                f"{secondary['negative_event_count']}.",
            ]
        )
        descriptive = inference["descriptive_alphaearth_tessera"]
        desc_bootstrap = descriptive["bootstrap"]
        lines.extend(
            [
                f"- Descriptive mean A_e (AlphaEarth minus TESSERA rho): "
                f"{descriptive['unweighted_mean_a_e']:.6f}; 95% descriptive "
                f"event-bootstrap interval "
                f"[{desc_bootstrap['lower']:.6f}, "
                f"{desc_bootstrap['upper']:.6f}].",
            ]
        )
    else:
        lines.append(
            "- Fewer than eight events were evaluable; cohort-level inference "
            "was not computed."
        )
    lines.extend(
        [
            "",
            "## Resource accounting",
            "",
            f"- Actual network transfer: "
            f"{inventory['actual_transfer_bytes']:,} bytes.",
            f"- External cache logical size: "
            f"{inventory['logical_cache_bytes']:,} bytes.",
            f"- Preflight runtime: {preflight['runtime_seconds']:.3f} seconds.",
            f"- Download runtime: {inventory['elapsed_seconds']:.3f} seconds.",
            f"- Processing compute runtime: "
            f"{processing['event_compute_seconds']:.3f} seconds.",
            f"- Analysis runtime: {analysis['runtime_seconds']:.3f} seconds.",
            f"- End-to-end phase runtime: "
            f"{analysis['phase_runtime']['end_to_end_phase_seconds']:.3f} seconds.",
            "",
            "## Paper recommendation",
            "",
            analysis["paper_recommendation"],
            "",
            "## Interpretation",
            "",
            "S_e is the TESSERA correlation minus the stronger optical "
            "correlation for each event. The AlphaEarth-versus-TESSERA paired "
            "difference is descriptive. Raw ECS magnitudes were never compared "
            "across models. Annual products can include non-fire change, and "
            "TESSERA fuses SAR and optical inputs; results are a secondary "
            "held-out comparison, not a model-ranking license.",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_artifact_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.json"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not path.name.endswith(".tmp")
    )
    entries = [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    write_json(
        manifest_path,
        {
            "schema": "tessera_mtbs_secondary_artifact_manifest_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "population_freeze_sha256": POPULATION_SHA256,
            "entries": entries,
            "file_count": len(entries),
            "total_bytes": sum(entry["bytes"] for entry in entries),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen TESSERA v1 comparator for sealed MTBS events"
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "download", "process", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mtbs-root", type=Path, default=DEFAULT_MTBS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--head-workers", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--chunk-pixels", type=int, default=25_000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    phases = {
        "preflight": [run_preflight],
        "download": [run_download],
        "process": [run_process],
        "analyze": [run_analysis],
        "all": [run_preflight, run_download, run_process, run_analysis],
    }
    for function in phases[args.phase]:
        function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
