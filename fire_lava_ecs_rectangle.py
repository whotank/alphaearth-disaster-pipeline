"""Extract and render expanded ECS rectangles for selected visual examples.

The original Thomas and La Palma visual masks use circular processing AOIs.
Held-out MTBS fires use event-shaped H3 support, so each reference circle is
the centroid-centred circle enclosing that support. In all cases the expanded
rectangle is three diameters wide and two diameters tall:

* west/east bounds are three radii from the circle centre;
* south/north bounds are two radii from the circle centre.

Only AlphaEarth ECS is extended.  Existing RGB and mapped-reference panels
remain bound to their original sources.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import contextily as cx
import ee
import geopandas as gpd
import h3
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from shapely import geometry as sg
from shapely import ops as so

import disaster_embedding_pipeline as pipeline
import export_development_visual_panels as panels
import fire_change_cohort as cohort

LOGGER = logging.getLogger("fire_lava_ecs_rectangle")
CONTRACT_VERSION = "fire_lava_ecs_rectangle_v1"
WEST_EAST_RADIUS_MULTIPLIER = 3.0
SOUTH_NORTH_RADIUS_MULTIPLIER = 2.0
DEFAULT_BATCH_SIZE = 500
DEFAULT_TILE_SCALE = 4
DEFAULT_EE_WORKERS = 1
MAX_EE_WORKERS = 8
MAX_TARGET_H3 = 50_000
ORIGINAL_MASK_NAME = "alphaearth_ecs_original_support_mask.png"
RECTANGLE_DATA_NAME = "alphaearth_ecs_rectangle.parquet"
RECTANGLE_METADATA_NAME = "ecs_rectangle_metadata.json"
RECTANGLE_RENDER_NAME = "alphaearth_ecs_mask.png"
BASEMAP_SOURCE = cx.providers.Esri.WorldGrayCanvas


class EcsRectangleError(RuntimeError):
    """Fatal geometry, extraction, cache, or rendering error."""


@dataclass(frozen=True)
class EcsRectangleSpec:
    """Frozen geometry and temporal definition for one expanded ECS panel."""

    event_key: str
    center_lon: float
    center_lat: float
    radius_m: float
    pre_year: int
    post_year: int
    h3_resolution: int
    original_h3_ids: tuple[str, ...]
    source_kind: str
    source_sha256: str


def _local_crs(center_lon: float, center_lat: float) -> pyproj.CRS:
    return pyproj.CRS.from_proj4(
        "+proj=aeqd "
        f"+lat_0={center_lat:.12f} +lon_0={center_lon:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )


def _transformer(
    source: pyproj.CRS | str,
    destination: pyproj.CRS | str,
) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(
        source,
        destination,
        always_xy=True,
    )


def _h3_polygon(cell: str) -> sg.Polygon:
    return sg.Polygon(
        [
            (float(longitude), float(latitude))
            for latitude, longitude in h3.cell_to_boundary(str(cell))
        ]
    )


def _read_original_h3(join_path: Path) -> tuple[str, ...]:
    frame = pd.read_parquet(join_path, columns=["h3_index"])
    values = tuple(sorted(frame["h3_index"].astype(str)))
    if not values:
        raise EcsRectangleError(f"Original H3 support is empty: {join_path}")
    if len(values) != len(set(values)):
        raise EcsRectangleError(f"Original H3 support has duplicates: {join_path}")
    resolutions = {h3.get_resolution(value) for value in values}
    if len(resolutions) != 1:
        raise EcsRectangleError(
            f"Original H3 support mixes resolutions {sorted(resolutions)}"
        )
    return values


def _support_reference_circle(
    h3_ids: Sequence[str],
) -> tuple[float, float, float]:
    """Return a centroid-centred circle enclosing complete H3 support."""

    centers = [h3.cell_to_latlng(str(cell)) for cell in h3_ids]
    initial_lat = float(np.mean([latitude for latitude, _ in centers]))
    initial_lon = float(np.mean([longitude for _, longitude in centers]))
    initial_crs = _local_crs(initial_lon, initial_lat)
    to_initial = _transformer("EPSG:4326", initial_crs)
    from_initial = _transformer(initial_crs, "EPSG:4326")
    projected_support = so.unary_union(
        [so.transform(to_initial.transform, _h3_polygon(cell)) for cell in h3_ids]
    )
    projected_centroid = projected_support.centroid
    center_lon, center_lat = from_initial.transform(
        projected_centroid.x,
        projected_centroid.y,
    )

    final_crs = _local_crs(center_lon, center_lat)
    to_final = _transformer("EPSG:4326", final_crs)
    radius_m = 0.0
    for cell in h3_ids:
        polygon = so.transform(to_final.transform, _h3_polygon(cell))
        for longitude_m, latitude_m in polygon.exterior.coords:
            radius_m = max(radius_m, math.hypot(longitude_m, latitude_m))
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise EcsRectangleError("Support reference-circle radius is invalid")
    return float(center_lon), float(center_lat), float(radius_m)


def build_spec(
    *,
    event_key: str,
    config: Mapping[str, object],
    join_path: Path,
    source_kind: str,
) -> EcsRectangleSpec:
    """Build the event-specific circle definition without contacting Earth Engine."""

    original_h3 = _read_original_h3(join_path)
    h3_resolution = h3.get_resolution(original_h3[0])
    if source_kind == "pipeline_join":
        required = {
            "center_lon",
            "center_lat",
            "processing_buffer_km",
            "h3_resolution",
        }
        missing = sorted(required - config.keys())
        if missing:
            raise EcsRectangleError(f"{event_key}: missing geometry fields {missing}")
        configured_resolution = int(config["h3_resolution"])
        if configured_resolution != h3_resolution:
            raise EcsRectangleError(
                f"{event_key}: configured H3 resolution {configured_resolution} "
                f"does not match source resolution {h3_resolution}"
            )
        center_lon = float(config["center_lon"])
        center_lat = float(config["center_lat"])
        radius_m = float(config["processing_buffer_km"]) * 1000.0
    elif source_kind == "heldout_mtbs_sealed":
        center_lon, center_lat, radius_m = _support_reference_circle(original_h3)
    else:
        raise EcsRectangleError(f"{event_key}: unsupported source kind {source_kind!r}")
    return EcsRectangleSpec(
        event_key=event_key,
        center_lon=center_lon,
        center_lat=center_lat,
        radius_m=radius_m,
        pre_year=int(config["pre_year"]),
        post_year=int(config["post_year"]),
        h3_resolution=h3_resolution,
        original_h3_ids=original_h3,
        source_kind=source_kind,
        source_sha256=panels.sha256_file(join_path),
    )


def rectangle_geometry(spec: EcsRectangleSpec) -> sg.Polygon:
    """Return the requested 6R-by-4R local rectangle in EPSG:4326."""

    local = _local_crs(spec.center_lon, spec.center_lat)
    to_wgs84 = _transformer(local, "EPSG:4326")
    rectangle_local = sg.box(
        -WEST_EAST_RADIUS_MULTIPLIER * spec.radius_m,
        -SOUTH_NORTH_RADIUS_MULTIPLIER * spec.radius_m,
        WEST_EAST_RADIUS_MULTIPLIER * spec.radius_m,
        SOUTH_NORTH_RADIUS_MULTIPLIER * spec.radius_m,
    )
    transformed = so.transform(to_wgs84.transform, rectangle_local)
    if not isinstance(transformed, sg.Polygon):
        raise EcsRectangleError(f"{spec.event_key}: invalid rectangle geometry")
    return transformed


def circle_geometry(spec: EcsRectangleSpec) -> sg.Polygon:
    """Return the original/reference circle in EPSG:4326."""

    local = _local_crs(spec.center_lon, spec.center_lat)
    to_wgs84 = _transformer(local, "EPSG:4326")
    transformed = so.transform(
        to_wgs84.transform,
        sg.Point(0.0, 0.0).buffer(spec.radius_m, quad_segs=180),
    )
    if not isinstance(transformed, sg.Polygon):
        raise EcsRectangleError(f"{spec.event_key}: invalid circle geometry")
    return transformed


def target_h3_ids(spec: EcsRectangleSpec) -> tuple[str, ...]:
    """Polyfill the rectangle and verify it contains the old support."""

    values = tuple(
        pipeline._polygon_to_h3_cells(
            rectangle_geometry(spec),
            spec.h3_resolution,
        )
    )
    if len(values) > MAX_TARGET_H3:
        raise EcsRectangleError(
            f"{spec.event_key}: rectangle requires {len(values)} H3 cells, "
            f"above safety limit {MAX_TARGET_H3}"
        )
    missing_original = sorted(set(spec.original_h3_ids) - set(values))
    if missing_original:
        raise EcsRectangleError(
            f"{spec.event_key}: rectangle excludes {len(missing_original)} original "
            f"H3 cells; first={missing_original[:5]}"
        )
    return values


def _fingerprint(
    spec: EcsRectangleSpec,
    h3_ids: Sequence[str],
) -> str:
    return cohort.canonical_sha256(
        {
            "contract": CONTRACT_VERSION,
            "event_key": spec.event_key,
            "source_kind": spec.source_kind,
            "source_sha256": spec.source_sha256,
            "center_lon": spec.center_lon,
            "center_lat": spec.center_lat,
            "radius_m": spec.radius_m,
            "pre_year": spec.pre_year,
            "post_year": spec.post_year,
            "h3_resolution": spec.h3_resolution,
            "west_east_radius_multiplier": WEST_EAST_RADIUS_MULTIPLIER,
            "south_north_radius_multiplier": SOUTH_NORTH_RADIUS_MULTIPLIER,
            "h3_ids": list(h3_ids),
            "ecs_definition": "mean per-H3 sum of squared 64-D annual differences",
            "scale_m": 10,
        }
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.stem}.pending{path.suffix}")
    frame.to_parquet(pending, index=False)
    pending.replace(path)


def _validate_extraction(
    frame: pd.DataFrame,
    expected_h3: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    required = {"h3_index", "ecs_mean"}
    missing_columns = sorted(required - set(frame))
    if missing_columns:
        raise EcsRectangleError(f"{context}: missing columns {missing_columns}")
    output = frame[["h3_index", "ecs_mean"]].copy()
    output["h3_index"] = output["h3_index"].astype(str)
    output["ecs_mean"] = pd.to_numeric(output["ecs_mean"], errors="coerce")
    expected = [str(value) for value in expected_h3]
    if output["h3_index"].duplicated().any():
        raise EcsRectangleError(f"{context}: duplicate H3 rows")
    if set(output["h3_index"]) != set(expected) or len(output) != len(expected):
        raise EcsRectangleError(f"{context}: incomplete H3 coverage")
    order = {value: position for position, value in enumerate(expected)}
    output["_order"] = output["h3_index"].map(order)
    return (
        output.sort_values("_order", kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _extract_rectangle(
    *,
    spec: EcsRectangleSpec,
    h3_ids: Sequence[str],
    output_dir: Path,
    batch_size: int,
    tile_scale: int,
    ee_workers: int,
    max_retries: int,
    force: bool,
    data_name: str = RECTANGLE_DATA_NAME,
    metadata_name: str = RECTANGLE_METADATA_NAME,
    checkpoint_directory: str = "ecs_rectangle_checkpoints",
) -> pd.DataFrame:
    if (
        batch_size <= 0
        or tile_scale <= 0
        or not 1 <= ee_workers <= MAX_EE_WORKERS
        or max_retries < 0
    ):
        raise EcsRectangleError("Invalid Earth Engine batch settings")
    for value in (data_name, metadata_name, checkpoint_directory):
        if not value or Path(value).name != value:
            raise EcsRectangleError(f"Invalid rectangle output name: {value!r}")
    fingerprint = _fingerprint(spec, h3_ids)
    data_path = output_dir / data_name
    metadata_path = output_dir / metadata_name
    if data_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint and metadata.get(
            "data_sha256"
        ) == panels.sha256_file(data_path):
            return _validate_extraction(
                pd.read_parquet(data_path),
                h3_ids,
                context=f"{spec.event_key} cached rectangle",
            )

    rectangle = rectangle_geometry(spec)
    coordinates = [
        [float(longitude), float(latitude)]
        for longitude, latitude in rectangle.exterior.coords
    ]
    region = ee.Geometry.Polygon([coordinates], proj="EPSG:4326", geodesic=False)
    pre = pipeline._annual_embedding_mosaic(spec.pre_year, region)
    post = pipeline._annual_embedding_mosaic(spec.post_year, region)
    ecs_image = pipeline._ecs_image(pre, post).clip(region)

    checkpoint_root = (
        output_dir
        / checkpoint_directory
        / f"{fingerprint[:24]}_b{batch_size}_t{tile_scale}"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    frames_by_number: dict[int, pd.DataFrame] = {}
    pending_batches: list[tuple[int, list[str], Path]] = []
    batch_count = math.ceil(len(h3_ids) / batch_size)
    for batch_number, start in enumerate(
        range(0, len(h3_ids), batch_size),
        start=1,
    ):
        batch_ids = list(h3_ids[start : start + batch_size])
        checkpoint_path = checkpoint_root / f"batch_{batch_number:05d}.parquet"
        if checkpoint_path.is_file() and not force:
            batch = _validate_extraction(
                pd.read_parquet(checkpoint_path),
                batch_ids,
                context=f"{spec.event_key} cached batch {batch_number}",
            )
            frames_by_number[batch_number] = batch
            LOGGER.info(
                "%s expanded ECS batch %d/%d loaded from checkpoint",
                spec.event_key,
                batch_number,
                batch_count,
            )
        else:
            pending_batches.append((batch_number, batch_ids, checkpoint_path))

    def extract_batch(
        task: tuple[int, list[str], Path],
    ) -> tuple[int, pd.DataFrame]:
        batch_number, batch_ids, checkpoint_path = task
        reduced = pipeline._reduce_regions_h3_batched(
            ecs_image,
            batch_ids,
            reducer=ee.Reducer.mean(),
            scale=10,
            batch_size=len(batch_ids),
            tile_scale=tile_scale,
            context=f"{spec.event_key} expanded ECS batch {batch_number}/{batch_count}",
            max_retries=max_retries,
        )
        if "mean" in reduced:
            ecs_values = pd.to_numeric(reduced["mean"], errors="coerce")
        elif "ECS" in reduced:
            ecs_values = pd.to_numeric(reduced["ECS"], errors="coerce")
        else:
            ecs_values = pd.Series(np.nan, index=reduced.index, dtype=float)
        batch = _validate_extraction(
            pd.DataFrame(
                {
                    "h3_index": reduced["h3_index"].astype(str),
                    "ecs_mean": ecs_values,
                }
            ),
            batch_ids,
            context=f"{spec.event_key} batch {batch_number}",
        )
        _atomic_write_parquet(batch, checkpoint_path)
        return batch_number, batch

    worker_count = min(ee_workers, len(pending_batches))
    if pending_batches:
        LOGGER.info(
            "%s extracting %d missing ECS batches with %d worker(s)",
            spec.event_key,
            len(pending_batches),
            worker_count,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(extract_batch, task) for task in pending_batches]
            for future in as_completed(futures):
                batch_number, batch = future.result()
                frames_by_number[batch_number] = batch
                LOGGER.info(
                    "%s expanded ECS batch %d/%d sealed",
                    spec.event_key,
                    batch_number,
                    batch_count,
                )

    frames = [
        frames_by_number[batch_number] for batch_number in range(1, batch_count + 1)
    ]

    combined = _validate_extraction(
        pd.concat(frames, ignore_index=True),
        h3_ids,
        context=f"{spec.event_key} complete rectangle",
    )
    finite_mask = np.isfinite(combined["ecs_mean"].to_numpy(dtype=float))
    finite_h3 = set(combined.loc[finite_mask, "h3_index"].astype(str))
    valid_count = int(finite_mask.sum())
    missing_original = sorted(set(spec.original_h3_ids) - finite_h3)
    if missing_original:
        raise EcsRectangleError(
            f"{spec.event_key}: {len(missing_original)} original cells lack "
            f"finite ECS after rectangle extraction; first={missing_original[:5]}"
        )
    _atomic_write_parquet(combined, data_path)
    metadata: dict[str, Any] = {
        "schema": CONTRACT_VERSION,
        "created_at_utc": cohort.utc_now(),
        "fingerprint": fingerprint,
        "event_key": spec.event_key,
        "source_kind": spec.source_kind,
        "source_sha256": spec.source_sha256,
        "center_lon": spec.center_lon,
        "center_lat": spec.center_lat,
        "reference_circle_radius_m": spec.radius_m,
        "rectangle": {
            "west_east_radius_multiplier": WEST_EAST_RADIUS_MULTIPLIER,
            "south_north_radius_multiplier": SOUTH_NORTH_RADIUS_MULTIPLIER,
            "width_m": 2.0 * WEST_EAST_RADIUS_MULTIPLIER * spec.radius_m,
            "height_m": 2.0 * SOUTH_NORTH_RADIUS_MULTIPLIER * spec.radius_m,
            "geojson": sg.mapping(rectangle),
        },
        "pre_year": spec.pre_year,
        "post_year": spec.post_year,
        "h3_resolution": spec.h3_resolution,
        "original_h3_count": len(spec.original_h3_ids),
        "rectangle_h3_count": len(h3_ids),
        "finite_ecs_h3_count": valid_count,
        "h3_sha256": cohort.canonical_sha256(list(h3_ids)),
        "data_path": panels._display_path(data_path),
        "data_sha256": panels.sha256_file(data_path),
        "earth_engine": {
            "project_supplied_by_caller": True,
            "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "scale_m": 10,
            "batch_size": batch_size,
            "tile_scale": tile_scale,
            "concurrent_workers": ee_workers,
        },
    }
    metadata["payload_sha256"] = cohort.canonical_sha256(metadata)
    cohort.atomic_write_json(metadata_path, metadata)
    return combined


def _render_rectangle(
    *,
    frame: pd.DataFrame,
    spec: EcsRectangleSpec,
    output_path: Path,
) -> None:
    finite = frame.loc[np.isfinite(frame["ecs_mean"].to_numpy(dtype=float))].copy()
    if finite.empty or finite["ecs_mean"].nunique() < 2:
        raise EcsRectangleError(f"{spec.event_key}: ECS rectangle is not plottable")
    finite["geometry"] = finite["h3_index"].map(_h3_polygon)
    gdf = gpd.GeoDataFrame(finite, geometry="geometry", crs="EPSG:4326")
    robust = pipeline._robust_iqr_scores_1d(
        gdf["ecs_mean"].to_numpy(dtype=float, copy=False)
    )
    gdf["_ecs_plot_z"] = pipeline._linear_01_within(robust)
    projected = gdf.to_crs(epsg=3857)
    circle = gpd.GeoSeries([circle_geometry(spec)], crs="EPSG:4326").to_crs(epsg=3857)
    rectangle = gpd.GeoSeries(
        [rectangle_geometry(spec)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    minimum_x, minimum_y, maximum_x, maximum_y = (
        float(value) for value in rectangle.total_bounds
    )
    pad = max(maximum_x - minimum_x, maximum_y - minimum_y) * 0.015

    figure, axis = plt.subplots(figsize=(9.6, 6.4), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#F4F5F3")
    projected.plot(
        column="_ecs_plot_z",
        ax=axis,
        cmap=pipeline.JOIN_HEATMAP_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        linewidth=0.2,
        edgecolor="white",
        zorder=2,
    )
    prior_collection_count = len(axis.collections)
    circle.boundary.plot(
        ax=axis,
        color="#00E5FF",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    for collection in axis.collections[prior_collection_count:]:
        collection.set_path_effects(
            [
                path_effects.Stroke(linewidth=3.2, foreground="#18242A"),
                path_effects.Normal(),
            ]
        )
    axis.set_xlim(minimum_x - pad, maximum_x + pad)
    axis.set_ylim(minimum_y - pad, maximum_y + pad)
    try:
        cx.add_basemap(
            axis,
            crs=projected.crs,
            source=BASEMAP_SOURCE,
            attribution=False,
            zorder=0,
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - tile fallback
        LOGGER.warning("Basemap unavailable for %s: %s", output_path, exc)
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def export_rectangle(
    *,
    event_key: str,
    config: Mapping[str, object],
    join_path: Path,
    source_kind: str,
    output_dir: Path,
    ee_project: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tile_scale: int = DEFAULT_TILE_SCALE,
    ee_workers: int = DEFAULT_EE_WORKERS,
    max_retries: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Extract/cache the expanded ECS and replace the ECS panel used in the 2x2."""

    spec = build_spec(
        event_key=event_key,
        config=config,
        join_path=join_path,
        source_kind=source_kind,
    )
    expanded_h3 = target_h3_ids(spec)
    original_render = output_dir / RECTANGLE_RENDER_NAME
    if not original_render.is_file():
        raise EcsRectangleError(
            f"{event_key}: render the original ECS mask before expanding it"
        )
    shutil.copyfile(original_render, output_dir / ORIGINAL_MASK_NAME)
    ee.Initialize(project=ee_project)
    frame = _extract_rectangle(
        spec=spec,
        h3_ids=expanded_h3,
        output_dir=output_dir,
        batch_size=batch_size,
        tile_scale=tile_scale,
        ee_workers=ee_workers,
        max_retries=max_retries,
        force=force,
    )
    _render_rectangle(
        frame=frame,
        spec=spec,
        output_path=original_render,
    )
    metadata_path = output_dir / RECTANGLE_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["render"] = {
        "path": panels._display_path(original_render),
        "sha256": panels.sha256_file(original_render),
        "normalization": "rectangle-wide IQR z-score followed by linear min-max to [0,1]",
        "original_circle_outline": {
            "color": "#00E5FF",
            "style": "dashed",
        },
    }
    metadata["original_support_render"] = {
        "path": panels._display_path(output_dir / ORIGINAL_MASK_NAME),
        "sha256": panels.sha256_file(output_dir / ORIGINAL_MASK_NAME),
    }
    metadata.pop("payload_sha256", None)
    metadata["payload_sha256"] = cohort.canonical_sha256(metadata)
    cohort.atomic_write_json(metadata_path, metadata)
    return metadata


def export_temporal_variant(
    *,
    event_key: str,
    variant_key: str,
    pre_year: int,
    post_year: int,
    config: Mapping[str, object],
    join_path: Path,
    source_kind: str,
    output_dir: Path,
    ee_project: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tile_scale: int = DEFAULT_TILE_SCALE,
    ee_workers: int = DEFAULT_EE_WORKERS,
    max_retries: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Export the same expanded support for a separately named temporal window."""

    if not re.fullmatch(r"[a-z0-9_]+", variant_key):
        raise EcsRectangleError(f"Invalid temporal variant key: {variant_key!r}")
    if pre_year < 2017 or post_year <= pre_year:
        raise EcsRectangleError(
            f"Invalid AlphaEarth temporal variant: {pre_year}->{post_year}"
        )
    base_spec = build_spec(
        event_key=event_key,
        config=config,
        join_path=join_path,
        source_kind=source_kind,
    )
    spec = replace(
        base_spec,
        event_key=f"{event_key}__{variant_key}",
        pre_year=int(pre_year),
        post_year=int(post_year),
    )
    expanded_h3 = target_h3_ids(spec)
    data_name = f"alphaearth_ecs_rectangle_{variant_key}.parquet"
    metadata_name = f"ecs_rectangle_metadata_{variant_key}.json"
    render_name = f"alphaearth_ecs_mask_{variant_key}.png"
    ee.Initialize(project=ee_project)
    frame = _extract_rectangle(
        spec=spec,
        h3_ids=expanded_h3,
        output_dir=output_dir,
        batch_size=batch_size,
        tile_scale=tile_scale,
        ee_workers=ee_workers,
        max_retries=max_retries,
        force=force,
        data_name=data_name,
        metadata_name=metadata_name,
        checkpoint_directory=f"ecs_rectangle_checkpoints_{variant_key}",
    )
    render_path = output_dir / render_name
    _render_rectangle(frame=frame, spec=spec, output_path=render_path)
    metadata_path = output_dir / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["temporal_variant"] = {
        "base_event_key": event_key,
        "variant_key": variant_key,
        "pre_year": int(pre_year),
        "post_year": int(post_year),
        "interpretation": (
            "Availability-supported shifted window on the identical spatial "
            "support; it is not an event-free placebo when an endpoint is the "
            "incident year."
        ),
    }
    metadata["render"] = {
        "path": panels._display_path(render_path),
        "sha256": panels.sha256_file(render_path),
        "normalization": (
            "variant-wide IQR z-score followed by linear min-max to [0,1]"
        ),
        "original_circle_outline": {
            "color": "#00E5FF",
            "style": "dashed",
        },
    }
    metadata.pop("payload_sha256", None)
    metadata["payload_sha256"] = cohort.canonical_sha256(metadata)
    cohort.atomic_write_json(metadata_path, metadata)
    return metadata
