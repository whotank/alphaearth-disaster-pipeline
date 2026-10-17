"""Stage 1: frozen MTBS QA, optical support, replacements, and H3 freeze."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import ee
import numpy as np
import pandas as pd
import rasterio

from study_core import (
    ATTEMPT_OUTPUT_ROOT,
    CHECKPOINT_ROOT,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    GateFailure,
    OUTPUT_ROOT,
    RAW_ROOT,
    S2_BANDS,
    S2_COLLECTION,
    S2_SCL_EXCLUDED,
    StudyError,
    WORLD_COVER_CODES,
    WORLD_COVER_COLLECTION,
    WORLD_COVER_LABELS,
    deterministic_h3_cap,
    ee_geometry,
    h3_population_hash,
    leakage_audit,
    normalize_numeric_columns,
    reduce_h3_checkpointed,
    relative,
    sha256_file,
    update_runtime,
    utc_now,
    validate_h3_frame,
    verify_environment,
    write_csv,
    write_json,
    write_parquet,
)
from study_sources import (
    download_mtbs_severity,
    h3_reference_frame,
    metadata_mismatches,
    overlap_audit,
    query_candidate_sources,
    query_intersecting_boundaries,
)


MAX_SCENE_CLOUD_PERCENT = 70.0
MIN_CLEAR_OBSERVATIONS = 2
MIN_S2_VALID_FRACTION = 0.50
MIN_MTBS_VALID_FRACTION = 0.80
MIN_FINAL_H3 = 200
MAX_FINAL_H3 = 2000
S2_SCALE_M = 10
S2_BATCH_SIZE = 200
TILE_SCALE = 4
MAX_RETRIES = 3


def _mask_s2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    clear = ee.Image(1)
    for class_value in S2_SCL_EXCLUDED:
        clear = clear.And(scl.neq(class_value))
    return (
        image.select(list(S2_BANDS))
        .updateMask(clear)
        .multiply(0.0001)
        .copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])
    )


def _s2_collection(region: ee.Geometry, year: int) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PERCENT))
        .map(_mask_s2)
    )


def _normalized_difference(
    positive: ee.Image, negative: ee.Image, name: str
) -> tuple[ee.Image, ee.Image]:
    denominator = positive.add(negative)
    valid = denominator.abs().gt(1e-6)
    return (
        positive.subtract(negative).divide(denominator).rename(name).updateMask(valid),
        valid,
    )


def build_s2_and_landcover_image(
    geometry: Any, event_year: int
) -> tuple[ee.Image, dict[str, Any]]:
    region = ee_geometry(geometry)
    pre_year = event_year - 1
    post_year = event_year + 1
    pre_collection = _s2_collection(region, pre_year)
    post_collection = _s2_collection(region, post_year)
    pre_count_scenes = int(pre_collection.size().getInfo())
    post_count_scenes = int(post_collection.size().getInfo())
    if pre_count_scenes == 0 or post_count_scenes == 0:
        raise GateFailure(
            "P01",
            f"Sentinel-2 collection empty for {pre_year} or {post_year}",
        )
    pre_indexes = [
        str(value)
        for value in pre_collection.aggregate_array("system:index").getInfo()
    ]
    post_indexes = [
        str(value)
        for value in post_collection.aggregate_array("system:index").getInfo()
    ]
    pre = pre_collection.median().select(list(S2_BANDS))
    post = post_collection.median().select(list(S2_BANDS))
    pre_count = pre_collection.select("B8").count().rename("pre_clear_observations")
    post_count = (
        post_collection.select("B8").count().rename("post_clear_observations")
    )
    pre_ndvi, pre_ndvi_valid = _normalized_difference(
        pre.select("B8"), pre.select("B4"), "pre_ndvi"
    )
    post_ndvi, post_ndvi_valid = _normalized_difference(
        post.select("B8"), post.select("B4"), "post_ndvi"
    )
    pre_nbr, pre_nbr_valid = _normalized_difference(
        pre.select("B8"), pre.select("B12"), "pre_nbr"
    )
    post_nbr, post_nbr_valid = _normalized_difference(
        post.select("B8"), post.select("B12"), "post_nbr"
    )
    common_valid = (
        pre_count.gte(MIN_CLEAR_OBSERVATIONS)
        .And(post_count.gte(MIN_CLEAR_OBSERVATIONS))
        .And(pre_ndvi_valid)
        .And(post_ndvi_valid)
        .And(pre_nbr_valid)
        .And(post_nbr_valid)
    )
    dnbr = (
        pre_nbr.subtract(post_nbr)
        .rename("dnbr")
        .updateMask(common_valid)
        .clip(region)
    )
    dndvi_loss = (
        pre_ndvi.subtract(post_ndvi)
        .rename("dndvi_loss")
        .updateMask(common_valid)
        .clip(region)
    )
    image = (
        dnbr.addBands(dndvi_loss)
        .addBands(common_valid.unmask(0).float().rename("s2_valid_fraction"))
        .addBands(pre_count.float().clip(region))
        .addBands(post_count.float().clip(region))
    )
    world_cover = (
        ee.ImageCollection(WORLD_COVER_COLLECTION).first().select("Map")
    )
    for code in WORLD_COVER_CODES:
        image = image.addBands(
            world_cover.eq(code).float().rename(f"wc_{code}").clip(region)
        )
    metadata = {
        "collection": S2_COLLECTION,
        "pre_year": pre_year,
        "post_year": post_year,
        "pre_scene_count": pre_count_scenes,
        "post_scene_count": post_count_scenes,
        "pre_scene_indexes": pre_indexes,
        "post_scene_indexes": post_indexes,
        "maximum_scene_cloud_percent": MAX_SCENE_CLOUD_PERCENT,
        "minimum_clear_observations_each_year": MIN_CLEAR_OBSERVATIONS,
        "scl_excluded_classes": list(S2_SCL_EXCLUDED),
        "annual_composite": "median",
        "world_cover_collection": WORLD_COVER_COLLECTION,
    }
    return image, metadata


def _class_histogram(raster_path: Path) -> dict[str, int]:
    with rasterio.open(raster_path) as dataset:
        values = dataset.read(1)
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def _land_cover_summary(
    frame: pd.DataFrame, expected_proxy: str
) -> dict[str, Any]:
    fractions = {
        int(code): float(pd.to_numeric(frame[f"wc_{code}"], errors="coerce").mean())
        for code in WORLD_COVER_CODES
    }
    finite = {code: value for code, value in fractions.items() if math.isfinite(value)}
    dominant_code = max(finite, key=finite.get) if finite else None
    grass_shrub_sparse = sum(fractions[code] for code in (20, 30, 60))
    forest = fractions[10]
    observed_group = (
        "forest_woodland" if forest >= grass_shrub_sparse else "grass_shrub_sparse"
    )
    expected_group = (
        "forest_woodland"
        if str(expected_proxy).startswith("forest")
        else "grass_shrub_sparse"
    )
    return {
        "source": WORLD_COVER_COLLECTION,
        "class_fractions_unweighted_h3_mean": {
            str(code): fractions[code] for code in WORLD_COVER_CODES
        },
        "dominant_class_code": dominant_code,
        "dominant_class_label": WORLD_COVER_LABELS.get(dominant_code),
        "observed_group": observed_group,
        "frozen_proxy": expected_proxy,
        "frozen_proxy_group": expected_group,
        "mismatch": observed_group != expected_group,
        "eligibility_effect": "none",
    }


def _attempt_paths(fire_id: str) -> tuple[Path, Path]:
    return (
        ATTEMPT_OUTPUT_ROOT / f"{fire_id}.json",
        CHECKPOINT_ROOT / "eligible" / f"{fire_id}.parquet",
    )


def attempt_candidate(
    candidate: dict[str, Any],
    *,
    attempt_sequence: int,
    slot_primary: dict[str, Any],
    replacement_for: str | None,
    force: bool,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    fire_id = str(candidate["fire_id"])
    summary_path, eligible_path = _attempt_paths(fire_id)
    if summary_path.exists() and not force:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("freeze_sha256") != EXPECTED_FREEZE_SHA256
            or cached.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        ):
            raise StudyError(f"{fire_id}: cached attempt belongs to another protocol")
        frame = pd.read_parquet(eligible_path) if cached["eligible"] else None
        return cached, frame

    started = time.monotonic()
    result: dict[str, Any] = {
        "attempt_sequence": attempt_sequence,
        "fire_id": fire_id,
        "name": candidate["name"],
        "candidate_role": candidate["role"],
        "candidate_rank": candidate["rank"],
        "slot_primary_fire_id": slot_primary["fire_id"],
        "slot_primary_name": slot_primary["name"],
        "replacement_for": replacement_for,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "eligible": False,
        "status": "ATTEMPTING",
        "failed_gate": None,
        "failure": None,
        "source_entries": [],
        "gate_results": {},
    }
    try:
        properties, geometry, source_entries = query_candidate_sources(candidate)
        result["source_entries"].extend(source_entries)
        result["gate_results"]["P01_boundary_metadata"] = "pass"

        mismatches = metadata_mismatches(candidate, properties)
        result["metadata_mismatches"] = mismatches
        if mismatches:
            raise GateFailure("P04", "; ".join(mismatches))
        result["gate_results"]["P04_source_identity"] = "pass"

        intersections, intersection_sources = query_intersecting_boundaries(
            fire_id, geometry
        )
        result["source_entries"].extend(intersection_sources)
        overlap = overlap_audit(candidate, geometry, intersections)
        result["overlap_audit"] = overlap
        if float(overlap["temporal_union_fraction"]) > 0.05:
            raise GateFailure(
                "P02",
                "Y-1 through Y+1 exact overlap "
                f"{float(overlap['temporal_union_fraction']):.6f} exceeds 0.05",
            )
        result["gate_results"]["P02_exact_overlap"] = "pass"

        raster_path, raster_metadata, raster_sources = download_mtbs_severity(
            candidate, geometry
        )
        result["source_entries"].extend(raster_sources)
        result["raster_metadata"] = raster_metadata
        result["raster_class_histogram"] = _class_histogram(raster_path)
        download_projection = raster_metadata["download_projection"]
        resolution = download_projection["resolution_m_at_center"]
        if download_projection["crs"] == "EPSG:4326":
            expected_degrees = 30.0 / 111_319.49079327357
            angular_resolution = download_projection["resolution"]
            angular_ok = all(
                math.isclose(
                    float(value), expected_degrees, rel_tol=0.01, abs_tol=1e-9
                )
                for value in angular_resolution
            )
            scale_ok = (
                angular_ok
                and 20.0 <= max(map(float, resolution)) <= 40.0
                and min(map(float, resolution)) > 0.0
            )
        else:
            scale_ok = all(20.0 <= float(value) <= 40.0 for value in resolution)
        if (
            int(download_projection["requested_scale_m"]) != 30
            or not scale_ok
        ):
            raise GateFailure(
                "P04",
                "Unexpected MTBS raster requested/angular/ground resolution "
                f"{download_projection}",
            )
        result["gate_results"]["P01_raster_parse"] = "pass"
        result["gate_results"]["P04_crs_scale_product"] = "pass"

        reference = h3_reference_frame(fire_id, geometry, raster_path)
        result["h3_center_cells"] = int(len(reference))
        supported = reference[
            reference["mtbs_valid_fraction"] >= MIN_MTBS_VALID_FRACTION
        ].copy()
        result["h3_mtbs_supported_cells"] = int(len(supported))
        if len(supported) < MIN_FINAL_H3:
            raise GateFailure(
                "P03",
                f"Only {len(supported)} cells have >=80% MTBS support before optical QA",
            )
        if (
            not np.isfinite(supported["mtbs_reference"]).all()
            or supported["mtbs_reference"].nunique(dropna=True) < 2
        ):
            raise GateFailure("P05", "MTBS reference is nonfinite or constant")

        event_year = int(str(candidate["date"])[:4])
        s2_image, s2_metadata = build_s2_and_landcover_image(geometry, event_year)
        result["sentinel2_metadata"] = s2_metadata
        expected_columns = [
            "dnbr",
            "dndvi_loss",
            "s2_valid_fraction",
            "pre_clear_observations",
            "post_clear_observations",
            *[f"wc_{code}" for code in WORLD_COVER_CODES],
        ]
        reduced = reduce_h3_checkpointed(
            s2_image,
            supported["h3_index"].astype(str).tolist(),
            scale_m=S2_SCALE_M,
            batch_size=S2_BATCH_SIZE,
            tile_scale=TILE_SCALE,
            max_retries=MAX_RETRIES,
            checkpoint_dir=CHECKPOINT_ROOT / "s2" / fire_id,
            checkpoint_config={
                "fire_id": fire_id,
                "freeze_sha256": EXPECTED_FREEZE_SHA256,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "h3_sha256": h3_population_hash(
                    supported["h3_index"].astype(str).tolist()
                ),
                "scale_m": S2_SCALE_M,
                "batch_size": S2_BATCH_SIZE,
                "metadata": s2_metadata,
            },
            context=f"{fire_id} Sentinel-2/WorldCover",
        )
        optical = normalize_numeric_columns(reduced, expected_columns)
        optical = validate_h3_frame(
            optical,
            supported["h3_index"].astype(str).tolist(),
            context=f"{fire_id} normalized optical",
        )
        joined = supported.merge(
            optical, on="h3_index", how="left", validate="one_to_one", sort=False
        )
        common_mask = (
            np.isfinite(joined["mtbs_reference"])
            & np.isfinite(joined["dnbr"])
            & np.isfinite(joined["dndvi_loss"])
            & (joined["s2_valid_fraction"] >= MIN_S2_VALID_FRACTION)
        )
        common = joined.loc[common_mask].copy()
        result["h3_pre_cap_common_cells"] = int(len(common))
        if len(common) < MIN_FINAL_H3:
            raise GateFailure(
                "P03", f"Only {len(common)} complete reference/optical cells"
            )
        retained_ids = deterministic_h3_cap(
            fire_id, common["h3_index"].astype(str).tolist(), MAX_FINAL_H3
        )
        final = (
            common.set_index("h3_index")
            .loc[retained_ids]
            .reset_index()
        )
        final = validate_h3_frame(
            final, retained_ids, context=f"{fire_id} final pre-AlphaEarth population"
        )
        for column in ("mtbs_reference", "dnbr", "dndvi_loss"):
            if (
                not np.isfinite(final[column]).all()
                or final[column].nunique(dropna=True) < 2
            ):
                raise GateFailure("P05", f"{column} is nonfinite or constant")
        result["gate_results"]["P03_common_population"] = "pass"
        result["gate_results"]["P05_finite_nonconstant"] = "pass"
        result["h3_final_cells"] = int(len(final))
        result["h3_final_sha256"] = h3_population_hash(retained_ids)
        result["h3_cap_applied"] = len(common) > MAX_FINAL_H3
        result["land_cover_audit"] = _land_cover_summary(
            final, str(candidate["land_cover"])
        )
        result["gate_results"]["P06_land_cover"] = (
            "mismatch_report_only"
            if result["land_cover_audit"]["mismatch"]
            else "pass"
        )

        final.insert(0, "fire_id", fire_id)
        final.insert(1, "fire_name", candidate["name"])
        final.insert(2, "event_year", event_year)
        final.insert(3, "pre_year", event_year - 1)
        final.insert(4, "post_year", event_year + 1)
        write_parquet(eligible_path, final)
        result["prealpha_parquet_sha256"] = sha256_file(eligible_path)
        result["prealpha_parquet"] = relative(eligible_path)
        result["eligible"] = True
        result["status"] = "ELIGIBLE_PREALPHA_FROZEN"
        result["gate_results"]["P01"] = "pass"
        result["gate_results"]["P02"] = "pass"
        result["gate_results"]["P03"] = "pass"
        result["gate_results"]["P04"] = "pass"
        result["gate_results"]["P05"] = "pass"
        result["duration_seconds"] = time.monotonic() - started
        write_json(summary_path, result)
        return result, final
    except GateFailure as exc:
        result["status"] = "INELIGIBLE_FROZEN_GATE"
        result["failed_gate"] = exc.gate
        result["failure"] = str(exc)
        result["duration_seconds"] = time.monotonic() - started
        result["gate_results"][exc.gate] = "fail"
        write_json(summary_path, result)
        return result, None


def _ledger_rows(attempts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in attempts:
        rows.append(
            {
                "attempt_sequence": item["attempt_sequence"],
                "slot_primary_fire_id": item["slot_primary_fire_id"],
                "slot_primary_name": item["slot_primary_name"],
                "fire_id": item["fire_id"],
                "name": item["name"],
                "candidate_role": item["candidate_role"],
                "candidate_rank": item["candidate_rank"],
                "replacement_for": item["replacement_for"],
                "status": item["status"],
                "eligible": item["eligible"],
                "failed_gate": item["failed_gate"],
                "failure": item["failure"],
                "h3_center_cells": item.get("h3_center_cells"),
                "h3_mtbs_supported_cells": item.get("h3_mtbs_supported_cells"),
                "h3_pre_cap_common_cells": item.get("h3_pre_cap_common_cells"),
                "h3_final_cells": item.get("h3_final_cells"),
                "h3_final_sha256": item.get("h3_final_sha256"),
                "p06_land_cover_mismatch": item.get(
                    "land_cover_audit", {}
                ).get("mismatch"),
                "duration_seconds": item.get("duration_seconds"),
            }
        )
    return pd.DataFrame(rows)


def run_reference(*, ee_project: str, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    environment = verify_environment()
    freeze = environment["freeze"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = leakage_audit(freeze)
    write_json(OUTPUT_ROOT / "leakage_audit.json", audit)
    ee.Initialize(project=ee_project)

    primaries = sorted(freeze["primaries"], key=lambda item: int(item["rank"]))
    reserves = sorted(freeze["reserves"], key=lambda item: int(item["rank"]))
    attempts: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    attempt_sequence = 0
    reserve_index = 0

    for primary in primaries:
        candidate = primary
        replacement_for: str | None = None
        while True:
            attempt_sequence += 1
            result, frame = attempt_candidate(
                candidate,
                attempt_sequence=attempt_sequence,
                slot_primary=primary,
                replacement_for=replacement_for,
                force=force,
            )
            attempts.append(result)
            if result["eligible"]:
                if frame is None:
                    raise StudyError(f"{candidate['fire_id']}: eligible frame missing")
                final_rows.append(
                    {
                        "slot_rank": int(primary["rank"]),
                        "slot": primary["slot"],
                        "primary_fire_id": primary["fire_id"],
                        "primary_name": primary["name"],
                        "fire_id": candidate["fire_id"],
                        "name": candidate["name"],
                        "candidate_role": candidate["role"],
                        "candidate_rank": candidate["rank"],
                        "replacement": candidate["fire_id"] != primary["fire_id"],
                        "replacement_for": replacement_for,
                        "event_year": int(str(candidate["date"])[:4]),
                        "h3_count": int(result["h3_final_cells"]),
                        "h3_sha256": result["h3_final_sha256"],
                        "h3_ids": frame["h3_index"].astype(str).tolist(),
                        "prealpha_parquet": result["prealpha_parquet"],
                        "prealpha_parquet_sha256": result[
                            "prealpha_parquet_sha256"
                        ],
                        "p06_land_cover_audit": result["land_cover_audit"],
                        "source_hashes": {
                            entry["kind"]: entry["sha256"]
                            for entry in result["source_entries"]
                            if "sha256" in entry
                        },
                    }
                )
                break
            if result["failed_gate"] not in {"P01", "P02", "P03", "P04", "P05"}:
                raise StudyError(
                    f"{candidate['fire_id']}: invalid replacement reason "
                    f"{result['failed_gate']}"
                )
            if reserve_index >= len(reserves):
                raise StudyError(
                    f"Reserves exhausted while replacing {primary['fire_id']}"
                )
            candidate = reserves[reserve_index]
            reserve_index += 1
            replacement_for = str(primary["fire_id"])

    if len(final_rows) != 10:
        raise StudyError(f"Expected ten final events, got {len(final_rows)}")
    final_rows.sort(key=lambda item: item["slot_rank"])
    ledger = _ledger_rows(attempts)
    write_csv(OUTPUT_ROOT / "attempted_ledger.csv", ledger)
    write_json(
        OUTPUT_ROOT / "attempted_ledger.json",
        {
            "schema": "alphaearth_mtbs_attempted_ledger_v1",
            "attempts": attempts,
        },
    )
    source_entries = [
        entry
        for attempt in attempts
        for entry in attempt.get("source_entries", [])
    ]
    write_json(
        OUTPUT_ROOT / "source_manifest.json",
        {
            "schema": "alphaearth_mtbs_source_manifest_v1",
            "freeze_sha256": EXPECTED_FREEZE_SHA256,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "entries": source_entries,
        },
    )
    population = {
        "schema": "alphaearth_mtbs_prealpha_population_freeze_v1",
        "base_commit": environment["head"],
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "alphaearth_accessed": False,
        "selection_outcomes_used": [],
        "replacement_rule": freeze["replacement"],
        "events": final_rows,
    }
    population_path = OUTPUT_ROOT / "population_freeze.json"
    write_json(population_path, population, readonly=True)
    population_sha = sha256_file(population_path)
    audit_path = OUTPUT_ROOT / "population_freeze_audit.json"
    if not audit_path.exists():
        write_json(
            audit_path,
            {
                "schema": "alphaearth_mtbs_population_freeze_audit_v1",
                "population_freeze_sha256": population_sha,
                "verified": True,
                "verified_at_utc": utc_now(),
            },
            readonly=True,
        )
    final_compact = [
        {key: value for key, value in row.items() if key != "h3_ids"}
        for row in final_rows
    ]
    write_json(
        OUTPUT_ROOT / "final_cohort.json",
        {
            "schema": "alphaearth_mtbs_final_cohort_v1",
            "population_freeze_sha256": population_sha,
            "events": final_compact,
        },
    )
    summary = {
        "attempted": len(attempts),
        "eligible": 10,
        "failed": int((~ledger["eligible"].astype(bool)).sum()),
        "replacements": sum(bool(row["replacement"]) for row in final_rows),
        "population_freeze_sha256": population_sha,
    }
    update_runtime("reference", started, summary)
    return summary
