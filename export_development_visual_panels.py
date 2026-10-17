#!/usr/bin/env python3
"""Export the four visual panels behind the development-case 2x2 example.

For each event frozen in ``DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL.json`` this
script writes:

* ``before.png`` — pre-year Sentinel-2 RGB context;
* ``after.png`` — post-year Sentinel-2 RGB context;
* ``alphaearth_ecs_mask.png`` — normalized AlphaEarth ECS on the H3 grid; and
* ``mapped_reference_mask.png`` — normalized publication reference on the same
  H3 grid.

Existing GIS 2x2 figures are reused for RGB context where available. Missing
context is downloaded from Earth Engine. Masks are regenerated uniformly from
the protocol-bound publication joins, including ``joined_official.parquet`` for
the three drought cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import contextily as cx
import ee
import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon

import disaster_embedding_pipeline as pipeline
import fire_change_cohort as cohort
import gis_raster_reference_figure as gis_figure
from reference_imagery import joined_extent_polygon_wgs84

LOGGER = logging.getLogger("export_development_visual_panels")
ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "paper" / "DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL.json"
DEFAULT_AMENDMENT = (
    ROOT / "paper" / "DEVELOPMENT_SPATIAL_ROBUSTNESS_PROTOCOL_AMENDMENT_1.json"
)
DEFAULT_DROUGHT_VALIDATION = (
    ROOT
    / "outputs"
    / "reo2_development_spatial_robustness"
    / "drought_regeneration_validation.json"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "reo2_development_visual_panels"
PANEL_FILES = (
    "before.png",
    "after.png",
    "alphaearth_ecs_mask.png",
    "mapped_reference_mask.png",
)
SCL_EXCLUDED_CLASSES = (0, 1, 3, 7, 8, 9, 10, 11)
BASEMAP_SOURCE = cx.providers.Esri.WorldGrayCanvas


class VisualPanelError(RuntimeError):
    """Protocol, source-artifact, rendering, or output-integrity failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return resolved.as_posix()


def load_event_specs(protocol_path: Path) -> list[dict[str, Any]]:
    """Load and verify the eight protocol-bound development events."""

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "reo2-development-spatial-robustness-v1":
        raise VisualPanelError("Unexpected development protocol")
    values = payload.get("fixed_events")
    if not isinstance(values, list) or len(values) != 8:
        raise VisualPanelError("Development protocol must contain eight events")
    drought_validation = json.loads(
        DEFAULT_DROUGHT_VALIDATION.read_text(encoding="utf-8")
    )
    if not bool(drought_validation.get("all_events_passed")):
        raise VisualPanelError("Drought regeneration validation did not pass")
    if drought_validation.get("amendment_sha256") != sha256_file(DEFAULT_AMENDMENT):
        raise VisualPanelError("Drought validation amendment binding changed")
    validated_drought = {
        str(result["event_key"]): dict(result)
        for result in drought_validation.get("event_results", [])
    }
    output: list[dict[str, Any]] = []
    for value in values:
        event = dict(value)
        join_path = (ROOT / str(event["join_artifact"])).resolve()
        if not join_path.is_file():
            raise VisualPanelError(
                f"{event['event_key']}: missing publication join {join_path}"
            )
        observed = sha256_file(join_path)
        if observed != str(event["expected_sha256"]):
            validation = validated_drought.get(str(event["event_key"]))
            validated_hash = (
                ((validation or {}).get("structure") or {})
                .get("artifact_sha256", {})
                .get("joined_official")
            )
            if (
                event.get("family") != "drought"
                or not validation
                or not bool(validation.get("passed"))
                or validated_hash != observed
            ):
                raise VisualPanelError(
                    f"{event['event_key']}: publication join SHA-256 changed"
                )
            event["join_binding"] = "amendment_validated_drought_regeneration"
        else:
            event["join_binding"] = "frozen_exact_sha256"
        event["join_path"] = join_path
        output.append(event)
    event_ids = [str(event["event_key"]) for event in output]
    if len(event_ids) != len(set(event_ids)):
        raise VisualPanelError("Development protocol repeats an event")
    return output


def context_tiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop pre/post RGB panels from the existing GIS 2x2 layout."""

    height, width = image.shape[:2]

    def crop(
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
    ) -> np.ndarray:
        return image[
            round(height * y_start) : round(height * y_end),
            round(width * x_start) : round(width * x_end),
        ]

    return (
        crop(0.014, 0.456, 0.045, 0.505),
        crop(0.520, 0.962, 0.045, 0.505),
    )


def _save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array)
    if values.ndim != 3 or values.shape[2] not in {3, 4}:
        raise VisualPanelError(f"Invalid RGB array for {path}: {values.shape}")
    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values, 0.0, 1.0)
        values = np.rint(values * 255.0).astype(np.uint8)
    else:
        values = np.clip(values, 0, 255).astype(np.uint8)
    Image.fromarray(values).convert("RGB").save(path, optimize=True)


def _h3_polygon(cell: str) -> Polygon:
    return Polygon(
        [(longitude, latitude) for latitude, longitude in h3.cell_to_boundary(cell)]
    )


def _load_normalized_masks(
    join_path: Path,
    *,
    ecs_column: str = "ecs_mean",
    reference_column: str = "benchmark_mean",
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    frame = pd.read_parquet(
        join_path,
        columns=["h3_index", ecs_column, reference_column],
    ).copy()
    frame = frame.rename(
        columns={
            ecs_column: "ecs_mean",
            reference_column: "benchmark_mean",
        }
    )
    if frame.empty:
        raise VisualPanelError(f"Empty publication join: {join_path}")
    frame["h3_index"] = frame["h3_index"].astype(str)
    if frame["h3_index"].duplicated().any():
        raise VisualPanelError(f"Duplicate H3 cells in {join_path}")
    values = frame[["ecs_mean", "benchmark_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise VisualPanelError(f"Nonfinite mask values in {join_path}")
    frame["geometry"] = frame["h3_index"].map(_h3_polygon)
    gdf = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
    normalized, _, _ = pipeline._join_heatmap_plot_columns(gdf)
    projected = normalized.to_crs(epsg=3857)
    bounds = tuple(float(value) for value in projected.total_bounds)
    return projected, bounds


def _render_mask(
    gdf: gpd.GeoDataFrame,
    *,
    column: str,
    bounds: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    pad = max(maximum_x - minimum_x, maximum_y - minimum_y) * 0.06
    figure, axis = plt.subplots(figsize=(6.4, 6.4), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#F4F5F3")
    gdf.plot(
        column=column,
        ax=axis,
        cmap=pipeline.JOIN_HEATMAP_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.68,
        linewidth=0.35,
        edgecolor="white",
        zorder=2,
    )
    axis.set_xlim(minimum_x - pad, maximum_x + pad)
    axis.set_ylim(minimum_y - pad, maximum_y + pad)
    try:
        cx.add_basemap(
            axis,
            crs=gdf.crs,
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


def render_masks(
    join_path: Path,
    output_dir: Path,
    *,
    ecs_column: str = "ecs_mean",
    reference_column: str = "benchmark_mean",
) -> None:
    gdf, bounds = _load_normalized_masks(
        join_path,
        ecs_column=ecs_column,
        reference_column=reference_column,
    )
    _render_mask(
        gdf,
        column="_ecs_plot_z",
        bounds=bounds,
        output_path=output_dir / "alphaearth_ecs_mask.png",
    )
    _render_mask(
        gdf,
        column="_bench_plot_z",
        bounds=bounds,
        output_path=output_dir / "mapped_reference_mask.png",
    )


def _read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        values = source.read()
    if values.shape[0] < 3:
        raise VisualPanelError(f"Sentinel download has fewer than three bands: {path}")
    return values[:3]


def _s2_calendar_rgb(
    region: ee.Geometry,
    year: int,
    *,
    max_scenes: int,
    composite: str = "mosaic",
) -> ee.Image:
    """Build a full-calendar SCL-masked RGB visualization."""

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 70))
    )
    if max_scenes > 0:
        collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(max_scenes)

    def mask_scene(image: ee.Image) -> ee.Image:
        scl = image.select("SCL")
        clear = ee.Image(1)
        for class_value in SCL_EXCLUDED_CLASSES:
            clear = clear.And(scl.neq(class_value))
        return image.select(["B4", "B3", "B2"]).updateMask(clear)

    masked = collection.map(mask_scene)
    if composite == "median":
        image = masked.median()
    elif composite == "mosaic":
        image = (
            collection.sort("CLOUDY_PIXEL_PERCENTAGE", False).map(mask_scene).mosaic()
        )
    else:
        raise VisualPanelError(f"Unknown Sentinel-2 context composite: {composite}")
    image = image.clip(region).float()
    return image.visualize(
        bands=["B4", "B3", "B2"],
        min=[100, 100, 100],
        max=[4000, 4000, 4000],
        gamma=[1.05, 1.05, 1.0],
    )


def download_context(
    *,
    config: Mapping[str, Any],
    join_path: Path,
    output_dir: Path,
    scale_m: int,
    max_scenes: int,
    composite: str = "mosaic",
) -> int:
    """Download missing calendar-year Sentinel context and return used scale."""

    frame_polygon = joined_extent_polygon_wgs84(join_path)
    region = ee.Geometry(frame_polygon.__geo_interface__)
    pre_year = int(config["pre_year"])
    post_year = int(config["post_year"])
    pre_image = _s2_calendar_rgb(
        region,
        pre_year,
        max_scenes=max_scenes,
        composite=composite,
    )
    post_image = _s2_calendar_rgb(
        region,
        post_year,
        max_scenes=max_scenes,
        composite=composite,
    )
    temporary_pre = output_dir / ".before.pending.tif"
    temporary_post = output_dir / ".after.pending.tif"
    used_scale = int(scale_m)
    for attempt in range(5):
        try:
            gis_figure._download_s2_geotiff(
                pre_image,
                region,
                temporary_pre,
                used_scale,
            )
            gis_figure._download_s2_geotiff(
                post_image,
                region,
                temporary_post,
                used_scale,
            )
            break
        except Exception:
            if attempt >= 4 or used_scale >= 540:
                raise
            used_scale = round(used_scale * 1.5)
            LOGGER.warning(
                "Retrying Sentinel context at %d m for %s",
                used_scale,
                config["label"],
            )
    pre_rgb = _read_rgb(temporary_pre)
    with rasterio.open(temporary_pre) as pre_source:
        pre_profile = pre_source.profile.copy()
        pre_transform = pre_source.transform
        pre_crs = pre_source.crs
    with rasterio.open(temporary_post) as post_source:
        post_rgb_source = post_source.read()[:3]
        post_transform = post_source.transform
        post_crs = post_source.crs
    post_rgb = np.zeros_like(pre_rgb)
    for band in range(3):
        reproject(
            source=post_rgb_source[band],
            destination=post_rgb[band],
            src_transform=post_transform,
            src_crs=post_crs,
            dst_transform=pre_transform,
            dst_crs=pre_crs,
            resampling=Resampling.bilinear,
        )
    if tuple(pre_rgb.shape[1:]) != (
        int(pre_profile["height"]),
        int(pre_profile["width"]),
    ):
        raise VisualPanelError("Sentinel pre-grid dimensions changed while reading")
    _save_rgb(np.moveaxis(pre_rgb, 0, -1), output_dir / "before.png")
    _save_rgb(np.moveaxis(post_rgb, 0, -1), output_dir / "after.png")
    temporary_pre.unlink(missing_ok=True)
    temporary_post.unlink(missing_ok=True)
    return used_scale


def export_context(
    *,
    event_key: str,
    config: Mapping[str, Any],
    join_path: Path,
    output_dir: Path,
    ee_project: str,
    scale_m: int,
    max_scenes: int,
) -> dict[str, Any]:
    source_figure = ROOT / "outputs" / event_key / "gis_reference_raster_2x2.png"
    if source_figure.is_file():
        before, after = context_tiles(mpimg.imread(source_figure))
        _save_rgb(before, output_dir / "before.png")
        _save_rgb(after, output_dir / "after.png")
        return {
            "method": "cropped_existing_calendar_year_gis_2x2",
            "source_path": _display_path(source_figure),
            "source_sha256": sha256_file(source_figure),
        }
    ee.Initialize(project=ee_project)
    used_scale = download_context(
        config=config,
        join_path=join_path,
        output_dir=output_dir,
        scale_m=scale_m,
        max_scenes=max_scenes,
    )
    return {
        "method": "earth_engine_calendar_year_scl_masked_sentinel2_download",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "composite": "SCL-masked clearest-scene-priority mosaic",
        "scale_m": used_scale,
        "scene_cloud_filter_lte_pct": 70,
        "scene_limit": max_scenes or None,
        "scl_excluded_classes": list(SCL_EXCLUDED_CLASSES),
    }


def build_contact_sheet(
    output_dir: Path,
    *,
    label: str,
    pre_year: int,
    post_year: int,
    ecs_title: str = "AlphaEarth ECS mask",
) -> None:
    paths = [output_dir / name for name in PANEL_FILES]
    titles = (
        f"Before RGB ({pre_year})",
        f"After RGB ({post_year})",
        ecs_title,
        "Mapped-reference mask",
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=160)
    for axis, path, title in zip(axes.flat, paths, titles, strict=True):
        axis.imshow(mpimg.imread(path))
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_axis_off()
    figure.suptitle(label, fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "contact_sheet.png", facecolor="white")
    plt.close(figure)


def export_all(
    *,
    protocol_path: Path,
    output_root: Path,
    ee_project: str,
    scale_m: int,
    max_scenes: int,
) -> dict[str, Any]:
    specs = load_event_specs(protocol_path)
    configurations = pipeline._event_configurations()
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        event_key = str(spec["event_key"])
        LOGGER.info("Visual panels %d/%d: %s", index, len(specs), event_key)
        config = configurations.get(event_key)
        if config is None:
            raise VisualPanelError(f"Missing pipeline configuration for {event_key}")
        event_output = output_root / event_key
        event_output.mkdir(parents=True, exist_ok=True)
        context = export_context(
            event_key=event_key,
            config=config,
            join_path=Path(spec["join_path"]),
            output_dir=event_output,
            ee_project=ee_project,
            scale_m=scale_m,
            max_scenes=max_scenes,
        )
        render_masks(Path(spec["join_path"]), event_output)
        build_contact_sheet(
            event_output,
            label=str(config["label"]),
            pre_year=int(config["pre_year"]),
            post_year=int(config["post_year"]),
        )
        files = {}
        for name in (*PANEL_FILES, "contact_sheet.png"):
            path = event_output / name
            if not path.is_file() or path.stat().st_size == 0:
                raise VisualPanelError(f"{event_key}: missing rendered panel {name}")
            with Image.open(path) as image:
                dimensions = [int(image.width), int(image.height)]
            files[name] = {
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "dimensions_px": dimensions,
            }
        records.append(
            {
                "event_key": event_key,
                "family": spec["family"],
                "label": config["label"],
                "pre_year": int(config["pre_year"]),
                "post_year": int(config["post_year"]),
                "publication_join": {
                    "path": _display_path(Path(spec["join_path"])),
                    "sha256": sha256_file(Path(spec["join_path"])),
                    "binding": spec["join_binding"],
                },
                "context": context,
                "mask_normalization": (
                    "per-layer IQR z-score followed by linear min-max to [0,1]"
                ),
                "files": files,
            }
        )
    manifest = {
        "schema": "reo2_development_visual_panels_v1",
        "created_at_utc": cohort.utc_now(),
        "protocol": {
            "path": _display_path(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "drought_regeneration_amendment": {
            "path": _display_path(DEFAULT_AMENDMENT),
            "sha256": sha256_file(DEFAULT_AMENDMENT),
            "validation_path": _display_path(DEFAULT_DROUGHT_VALIDATION),
            "validation_sha256": sha256_file(DEFAULT_DROUGHT_VALIDATION),
        },
        "event_count": len(records),
        "events": records,
    }
    manifest["payload_sha256"] = cohort.canonical_sha256(manifest)
    cohort.atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ee-project", default="project0-256023")
    parser.add_argument("--scale-m", type=int, default=120)
    parser.add_argument(
        "--s2-max-scenes",
        type=int,
        default=0,
        help="Optional clearest-scene limit; 0 (default) uses all retained scenes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    protocol_path = args.protocol.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not output_root.is_relative_to((ROOT / "outputs").resolve()):
        raise VisualPanelError("Output directory must remain under outputs/")
    if args.scale_m <= 0 or args.s2_max_scenes < 0:
        raise VisualPanelError("Scale must be positive and scene limit nonnegative")
    specs = load_event_specs(protocol_path)
    if args.dry_run:
        LOGGER.info("Verified %d development visual-panel inputs", len(specs))
        return 0
    manifest = export_all(
        protocol_path=protocol_path,
        output_root=output_root,
        ee_project=args.ee_project,
        scale_m=args.scale_m,
        max_scenes=args.s2_max_scenes,
    )
    LOGGER.info(
        "Wrote four panels plus contact sheets for %d events under %s",
        manifest["event_count"],
        output_root,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        VisualPanelError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        sys.exit(2)
