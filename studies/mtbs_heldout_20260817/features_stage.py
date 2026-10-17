"""Stage 2: AlphaEarth ECS extraction into sealed, correlation-free files."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import ee
import numpy as np
import pandas as pd

from study_core import (
    ALPHAEARTH_BANDS,
    ALPHAEARTH_COLLECTION,
    CHECKPOINT_ROOT,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    OUTPUT_ROOT,
    SEALED_ROOT,
    StudyError,
    canonical_json_bytes,
    ee_geometry,
    h3_population_hash,
    normalize_numeric_columns,
    reduce_h3_checkpointed,
    relative,
    sha256_bytes,
    sha256_file,
    update_runtime,
    validate_h3_frame,
    verify_environment,
    write_json,
    write_parquet,
)
from study_sources import load_official_boundary


ECS_SCALE_M = 10
ECS_BATCH_SIZE = 40
TILE_SCALE = 4
MAX_RETRIES = 3


def build_ecs_image(
    geometry: Any, pre_year: int, post_year: int
) -> tuple[ee.Image, dict[str, Any]]:
    region = ee_geometry(geometry)

    def collection_for(year: int) -> ee.ImageCollection:
        return (
            ee.ImageCollection(ALPHAEARTH_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(region)
            .select(list(ALPHAEARTH_BANDS))
        )

    pre_collection = collection_for(pre_year)
    post_collection = collection_for(post_year)
    pre_count = int(pre_collection.size().getInfo())
    post_count = int(post_collection.size().getInfo())
    pre_indexes = [
        str(value)
        for value in pre_collection.aggregate_array("system:index").getInfo()
    ]
    post_indexes = [
        str(value)
        for value in post_collection.aggregate_array("system:index").getInfo()
    ]
    if pre_count < 1 or post_count < 1:
        raise StudyError(
            f"AlphaEarth annual mosaic missing for {pre_year} or {post_year}"
        )
    pre = pre_collection.mosaic().select(list(ALPHAEARTH_BANDS)).clip(region)
    post = post_collection.mosaic().select(list(ALPHAEARTH_BANDS)).clip(region)
    common_valid = (
        pre.mask().reduce(ee.Reducer.min()).And(
            post.mask().reduce(ee.Reducer.min())
        )
    )
    ecs = (
        post.subtract(pre)
        .pow(2)
        .reduce(ee.Reducer.sum())
        .rename("ecs")
        .float()
        .updateMask(common_valid)
        .clip(region)
    )
    support = common_valid.unmask(0).float().rename("ecs_valid_fraction").clip(region)
    metadata = {
        "collection": ALPHAEARTH_COLLECTION,
        "bands": list(ALPHAEARTH_BANDS),
        "pre_year": pre_year,
        "post_year": post_year,
        "pre_image_count": pre_count,
        "post_image_count": post_count,
        "pre_image_indexes": pre_indexes,
        "post_image_indexes": post_indexes,
        "full_calendar_windows": True,
        "pixel_formula": "sum((post_axis-pre_axis)^2 for A00-A63)",
        "scale_m": ECS_SCALE_M,
        "batch_size_h3": ECS_BATCH_SIZE,
    }
    return ecs.addBands(support), metadata


def _population() -> tuple[dict[str, Any], str]:
    path = OUTPUT_ROOT / "population_freeze.json"
    audit_path = OUTPUT_ROOT / "population_freeze_audit.json"
    if not path.exists() or not audit_path.exists():
        raise StudyError("Run the reference stage before AlphaEarth extraction")
    population_sha = sha256_file(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not audit.get("verified")
        or audit.get("population_freeze_sha256") != population_sha
    ):
        raise StudyError("Population freeze audit mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("freeze_sha256") != EXPECTED_FREEZE_SHA256:
        raise StudyError("Population freeze canonical hash mismatch")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise StudyError("Population freeze protocol hash mismatch")
    if len(payload.get("events", [])) != 10:
        raise StudyError("Population freeze does not contain ten events")
    return payload, population_sha


def run_features(*, ee_project: str, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    verify_environment()
    population, population_sha = _population()
    ee.Initialize(project=ee_project)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_root = OUTPUT_ROOT / "feature_extraction"
    manifest_events: list[dict[str, Any]] = []
    alphaearth_sources: list[dict[str, Any]] = []

    for event in population["events"]:
        fire_id = str(event["fire_id"])
        prealpha_path = STUDY_PATH(event["prealpha_parquet"])
        if sha256_file(prealpha_path) != event["prealpha_parquet_sha256"]:
            raise StudyError(f"{fire_id}: pre-AlphaEarth feature hash mismatch")
        frame = pd.read_parquet(prealpha_path)
        h3_ids = [str(value) for value in event["h3_ids"]]
        frame = validate_h3_frame(
            frame, h3_ids, context=f"{fire_id} frozen population"
        )
        if h3_population_hash(h3_ids) != event["h3_sha256"]:
            raise StudyError(f"{fire_id}: frozen H3 hash mismatch")

        sealed_path = SEALED_ROOT / f"{fire_id}.parquet"
        event_metadata_path = metadata_root / f"{fire_id}.json"
        if sealed_path.exists() and event_metadata_path.exists() and not force:
            metadata = json.loads(event_metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("population_freeze_sha256") != population_sha
                or metadata.get("sealed_sha256") != sha256_file(sealed_path)
            ):
                raise StudyError(f"{fire_id}: sealed resume metadata mismatch")
            sealed = validate_h3_frame(
                pd.read_parquet(sealed_path),
                h3_ids,
                context=f"{fire_id} resumed sealed features",
            )
            if len(sealed) != len(frame):
                raise StudyError(f"{fire_id}: resumed sealed row count changed")
            manifest_events.append(
                {
                    "slot_rank": event["slot_rank"],
                    "fire_id": fire_id,
                    "name": event["name"],
                    "rows": len(sealed),
                    "h3_sha256": event["h3_sha256"],
                    "sealed_file": relative(sealed_path),
                    "sealed_sha256": metadata["sealed_sha256"],
                    "metadata_file": relative(event_metadata_path),
                }
            )
            alphaearth_sources.append(metadata["source_entry"])
            continue

        geometry = load_official_boundary(fire_id)
        image, collection_metadata = build_ecs_image(
            geometry, int(event["event_year"]) - 1, int(event["event_year"]) + 1
        )
        collection_metadata_hash = sha256_bytes(
            canonical_json_bytes(collection_metadata)
        )
        reduced = reduce_h3_checkpointed(
            image,
            h3_ids,
            scale_m=ECS_SCALE_M,
            batch_size=ECS_BATCH_SIZE,
            tile_scale=TILE_SCALE,
            max_retries=MAX_RETRIES,
            checkpoint_dir=CHECKPOINT_ROOT / "alphaearth" / fire_id,
            checkpoint_config={
                "fire_id": fire_id,
                "population_freeze_sha256": population_sha,
                "h3_sha256": event["h3_sha256"],
                "collection_metadata_sha256": collection_metadata_hash,
                "scale_m": ECS_SCALE_M,
                "batch_size": ECS_BATCH_SIZE,
            },
            context=f"{fire_id} AlphaEarth ECS",
        )
        ecs = normalize_numeric_columns(reduced, ["ecs", "ecs_valid_fraction"])
        ecs = validate_h3_frame(
            ecs, h3_ids, context=f"{fire_id} normalized AlphaEarth"
        )
        if (
            not np.isfinite(ecs["ecs"]).all()
            or not np.isfinite(ecs["ecs_valid_fraction"]).all()
            or (ecs["ecs_valid_fraction"] <= 0).any()
        ):
            raise StudyError(
                f"{fire_id}: AlphaEarth has missing fixed-population cells; "
                "no imputation or post-freeze cell deletion is permitted"
            )
        sealed = frame.merge(
            ecs, on="h3_index", how="left", validate="one_to_one", sort=False
        )
        sealed = validate_h3_frame(
            sealed, h3_ids, context=f"{fire_id} sealed complete case"
        )
        required = ["mtbs_reference", "dnbr", "dndvi_loss", "ecs"]
        if not np.isfinite(sealed[required].to_numpy(dtype=float)).all():
            raise StudyError(f"{fire_id}: sealed file is not exact complete case")
        write_parquet(sealed_path, sealed)
        sealed_sha = sha256_file(sealed_path)
        source_entry = {
            "kind": "alphaearth_annual_mosaics",
            "fire_id": fire_id,
            "collection": ALPHAEARTH_COLLECTION,
            "metadata_sha256": collection_metadata_hash,
            "metadata": collection_metadata,
        }
        metadata = {
            "schema": "alphaearth_mtbs_sealed_event_metadata_v1",
            "fire_id": fire_id,
            "population_freeze_sha256": population_sha,
            "h3_sha256": event["h3_sha256"],
            "rows": len(sealed),
            "sealed_file": relative(sealed_path),
            "sealed_sha256": sealed_sha,
            "correlations_computed": False,
            "endpoint_computed": False,
            "source_entry": source_entry,
        }
        write_json(event_metadata_path, metadata)
        manifest_events.append(
            {
                "slot_rank": event["slot_rank"],
                "fire_id": fire_id,
                "name": event["name"],
                "rows": len(sealed),
                "h3_sha256": event["h3_sha256"],
                "sealed_file": relative(sealed_path),
                "sealed_sha256": sealed_sha,
                "metadata_file": relative(event_metadata_path),
            }
        )
        alphaearth_sources.append(source_entry)

    manifest_events.sort(key=lambda item: int(item["slot_rank"]))
    if len(manifest_events) != 10:
        raise StudyError("Not all ten sealed event files completed")
    manifest = {
        "schema": "alphaearth_mtbs_sealed_manifest_v1",
        "population_freeze_sha256": population_sha,
        "all_events_complete": True,
        "correlations_computed": False,
        "events": manifest_events,
    }
    manifest_path = OUTPUT_ROOT / "sealed_manifest.json"
    write_json(manifest_path, manifest, readonly=True)

    source_path = OUTPUT_ROOT / "source_manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    existing = [
        item
        for item in source_manifest["entries"]
        if item.get("kind") != "alphaearth_annual_mosaics"
    ]
    source_manifest["entries"] = [*existing, *alphaearth_sources]
    write_json(source_path, source_manifest)
    summary = {
        "events": 10,
        "sealed_manifest_sha256": sha256_file(manifest_path),
        "population_freeze_sha256": population_sha,
        "total_h3": sum(int(item["rows"]) for item in manifest_events),
    }
    update_runtime("features", started, summary)
    return summary


def STUDY_PATH(relative_path: str) -> Path:
    from study_core import STUDY_ROOT

    path = (STUDY_ROOT / relative_path).resolve()
    path.relative_to(STUDY_ROOT.resolve())
    return path
