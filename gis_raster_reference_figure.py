#!/usr/bin/env python3
"""
GIS-style reference figure (not geopandas draw-order only):

1) Download **georeferenced** Sentinel-2 RGB (GeoTIFF) for ``pre_year`` and ``post_year`` from Earth Engine
   (same padded join extent as ``reference_imagery.joined_extent_polygon_wgs84``).
2) **Rasterize** joined H3 polygons to the **exact** pixel grid of each GeoTIFF, filling ``ecs_mean`` and
   ``benchmark_mean`` (float32, NaN outside cells).
3) Build **bottom row** by compositing true RGB with a colormap layer driven only by those raster values
   (same grid as top row — no misaligned PNG extent tricks). ECS and benchmark share the **same**
   colormap name. Default ``--overlay-stretch per_layer_01``: IQR-robust z per layer, then **linear**
   min--max to ``[0,1]`` **separately** so each panel hits black (layer min) and full span. Alternatives:
   ``shared_robust`` (pooled q02--q98 on z), ``rank``, ``quantile``.

Output per event: ``gis_reference_raster_2x2.png`` plus optional GeoTIFFs ``s2_pre_grid.tif``,
``s2_post_grid.tif``, ``raster_ecs_mean.tif``, ``raster_benchmark_mean.tif`` for QA in QGIS.
Configured **fire**, **flood**, and **forest_loss** events are supported (not only wildfire).

Requires: rasterio, Earth Engine auth, ``EE_PROJECT_ID`` or ``GOOGLE_CLOUD_PROJECT`` (see below).

**Temporal alignment (important):**
AlphaEarth ECS uses **full calendar-year** mosaics for ``pre_year`` and ``post_year`` (Jan 1→next Jan 1).
MODIS MCD64A1 uses **full calendar-year** ``metric_year`` (often equal to ``post_year``, but see
``benchmark_year_matches_post`` on some presets). Sentinel-2 defaults to **June–September** only
(``--s2-window jun_sep``) for clearer skies — **not** the same temporal aggregation as AlphaEarth.
Use ``--s2-window calendar_year`` to approximate annual framing (often cloudier / patchier).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any


def _ensure_writable_mplconfigdir() -> None:
    """If MPLCONFIGDIR is missing or not writable (e.g. tutorial placeholder), use repo .mplconfig/."""
    fallback = Path(__file__).resolve().parent / ".mplconfig"
    cand = os.environ.get("MPLCONFIGDIR", "").strip()
    ok = False
    if cand:
        try:
            Path(cand).mkdir(parents=True, exist_ok=True)
            probe = Path(cand) / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            ok = True
        except OSError:
            ok = False
    if not ok:
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback)


_ensure_writable_mplconfigdir()

import ee
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from rasterio.enums import MergeAlg

from fire_analysis import FIRES_GOOD
from reference_imagery import _h3_to_polygon, joined_extent_polygon_wgs84
from verification_export import _ee_init

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _write_geotiff_bytes(content: bytes, out_tif: Path) -> None:
    if len(content) >= 2 and content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not names:
                raise ValueError("ZIP from Earth Engine contained no .tif")
            out_tif.write_bytes(zf.read(names[0]))
    else:
        out_tif.write_bytes(content)


def _download_s2_geotiff(vis_image: Any, region_ee: Any, out_tif: Path, scale_m: int) -> None:
    """Small enough for getDownloadURL limits (~32 MB); raise scale if needed."""
    params: dict[str, Any] = {
        "region": region_ee,
        "scale": int(scale_m),
        "format": "GEO_TIFF",
        "filePerBand": False,
        "maxPixels": 1e8,
    }
    import requests

    url = vis_image.getDownloadURL(params)
    r = requests.get(url, timeout=600)
    if not r.ok:
        logger.error("GEE download failed (%s): %s", r.status_code, (r.text or "")[:500])
        r.raise_for_status()
    _write_geotiff_bytes(r.content, out_tif)


def _s2_mosaic_vis(region: ee.Geometry, year: int, max_scenes: int, window: str) -> Any:
    """Sentinel-2 SR median RGB (visualized) — ``window`` = ``jun_sep`` | ``calendar_year``."""
    if window == "calendar_year":
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
    elif window == "jun_sep":
        start = ee.Date.fromYMD(year, 6, 1)
        end = ee.Date.fromYMD(year, 9, 30)
    else:
        raise ValueError(f"Unknown s2 window: {window!r}")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 70))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .limit(int(max_scenes))
    )
    img = col.median().select(["B4", "B3", "B2"]).clip(region).float()
    return img.visualize(bands=["B4", "B3", "B2"], min=[100, 100, 100], max=[4000, 4000, 4000], gamma=[1.05, 1.05, 1.0])


def _alignment_note_text(
    label: str,
    pre_y: int,
    post_y: int,
    benchmark_type: str,
    metric_year: int,
    bench_matches_post: bool,
    temporal_relation: str,
    reference_validity_rule: str | None,
    s2_window: str,
) -> str:
    s2_desc = (
        "June 1 – September 30 of each year (northern clear-season composite; default)."
        if s2_window == "jun_sep"
        else "January 1 – December 31 of each year (closer to AlphaEarth’s calendar-year mosaic idea; often cloudier)."
    )
    if benchmark_type == "modis_mcd64a1":
        benchmark_desc = (
            "MODIS MCD64A1 (benchmark_mean on hexes):\n"
            f"  • metric_year={metric_year}: calendar-year stack; per-pixel "
            "benchmark is annual max BurnDate (0 = no mapped burn).\n"
        )
    elif benchmark_type == "hansen_gfc_lossyear":
        benchmark_desc = (
            "Hansen GFC annual forest loss (benchmark_mean on hexes):\n"
            f"  • metric_year={metric_year}: pixels whose lossyear encodes "
            f"{metric_year}, weighted by treecover2000/100; all other pixels are 0.\n"
            "  • Forest loss is cause-agnostic and may include harvest, fire, "
            "mortality, or other disturbance.\n"
        )
    elif benchmark_type == "jrc_gsw_occurrence":
        benchmark_desc = (
            "JRC Global Surface Water occurrence (benchmark_mean on hexes):\n"
            "  • Static long-term water occurrence (0–100), used only as a "
            "catalog placeholder for pipeline testing.\n"
            f"  • metric_year={metric_year} is metadata only and does not make "
            "this an event-specific flood extent.\n"
        )
    elif benchmark_type == "geojson_binary":
        benchmark_desc = (
            "Event-specific GeoJSON reference (benchmark_mean on hexes):\n"
            "  • Binary raster painted as 1 inside the supplied geometry and 0 "
            "outside before H3 reduction.\n"
            f"  • metric_year={metric_year} records the intended event year; "
            "source dates and class semantics must be documented separately.\n"
            f"  • Valid support: {reference_validity_rule or 'not documented'}.\n"
        )
    elif benchmark_type == "usdm_season_delta":
        benchmark_desc = (
            "U.S. Drought Monitor seasonal change (benchmark_mean on hexes):\n"
            "  • Absolute change between April-October mean severity scores; "
            "None=0 and D0-D4=1-5.\n"
            f"  • metric_year={metric_year} is the comparison/post year. "
            "Spearman association is primary because USDM classes are ordinal.\n"
        )
    else:
        benchmark_desc = (
            f"Benchmark {benchmark_type!r} (benchmark_mean on hexes):\n"
            f"  • metric_year={metric_year}; consult the pipeline configuration "
            "for product semantics.\n"
        )
    if temporal_relation == "event_between_annual_mosaics":
        temporal_desc = (
            f"  • The event/reference year {metric_year} is bracketed by full "
            f"calendar mosaics {pre_y} and {post_y}.\n\n"
        )
    else:
        temporal_desc = (
            "  • If years differ, consult the event configuration before "
            "interpreting association.\n\n"
        )
    return (
        f"Event: {label}\n\n"
        "AlphaEarth embeddings (ECS input):\n"
        f"  • pre_year={pre_y}, post_year={post_y}: annual GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL mosaics, "
        "each built from filterDate Jan 1 → Jan 1 next calendar year (see _annual_embedding_mosaic in pipeline).\n"
        "  • Pixel ECS is the sum of squared per-band deltas (post − pre), then mean-reduced to H3.\n\n"
        f"{benchmark_desc}"
        f"  • temporal relation: {temporal_relation}; "
        f"benchmark_year_matches_post = {bench_matches_post}.\n"
        f"{temporal_desc}"
        "Sentinel-2 RGB in this figure:\n"
        f"  • Window mode: {s2_window}\n"
        f"  • {s2_desc}\n"
        f"  • Left/top “pre” composite uses calendar year {pre_y}; right/top “post” uses {post_y}.\n\n"
        "Interpreting the Sentinel-2 context:\n"
        f"  • With jun_sep, the pre panel covers only summer {pre_y}; color may reflect "
        "agriculture, senescent vegetation, exposed soil, smoke, or prior disturbance.\n"
        "  • S2 RGB is not the benchmark layer; it is optical reflectance for the chosen date window only.\n"
        "  • ECS still uses full-year AlphaEarth mosaics unless you change the pipeline.\n"
    )


def _joined_gdf(joined_path: Path) -> gpd.GeoDataFrame:
    joined_df = pd.read_parquet(joined_path).copy()
    joined_df["h3_index"] = joined_df["h3_index"].astype(str)
    joined_df["geometry"] = joined_df["h3_index"].map(_h3_to_polygon)
    return gpd.GeoDataFrame(joined_df, geometry="geometry", crs="EPSG:4326")


def _rasterize_hex_column(
    gdf_wgs84: gpd.GeoDataFrame,
    column: str,
    ref_profile: dict,
) -> np.ndarray:
    """Burn hex values onto the reference raster grid (same CRS, transform, shape)."""
    h, w = int(ref_profile["height"]), int(ref_profile["width"])
    transform = ref_profile["transform"]
    crs = ref_profile["crs"]
    gdf = gdf_wgs84.to_crs(crs)
    out = np.full((h, w), np.nan, dtype=np.float32)
    shapes: list[tuple[Any, float]] = []
    for geom, val in zip(gdf.geometry, gdf[column].astype(float)):
        if geom is None or geom.is_empty:
            continue
        if not np.isfinite(val):
            continue
        shapes.append((geom, float(val)))
    if not shapes:
        return out
    rio_features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
        merge_alg=MergeAlg.replace,
        out=out,
    )
    return out


def _rank01_within_aoi(arr: np.ndarray) -> np.ndarray:
    """Average rank then linear to ``[0,1]`` so min/max finite values hit colormap ends (NaN elsewhere)."""
    flat = arr.ravel()
    m = np.isfinite(flat)
    out_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
    if not np.any(m):
        return out_flat.reshape(arr.shape)
    s = pd.Series(flat[m].astype(np.float64))
    r = s.rank(method="average", ascending=True).astype(np.float64)
    n = int(r.notna().sum())
    if n <= 1:
        out_flat[np.flatnonzero(m)] = 0.5
    else:
        u = ((r - 1.0) / (n - 1.0)).astype(np.float32)
        out_flat[np.flatnonzero(m)] = u.to_numpy()
    return out_flat.reshape(arr.shape)


def _robust_iqr_scores(arr: np.ndarray) -> np.ndarray:
    """Per-layer (x - median) / IQR with NaN preserved; IQR from finite cells only."""
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    m = np.isfinite(arr)
    if not np.any(m):
        return out
    x = arr[m].astype(np.float64)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr < 1e-12:
        sd = float(np.std(x))
        iqr = sd if sd > 1e-12 else 1.0
    out[m] = (arr[m].astype(np.float64) - med) / iqr
    return out


def _linear01_raster(z: np.ndarray) -> np.ndarray:
    """Map finite raster values to ``[0,1]`` linearly (min→0, max→1); NaN preserved."""
    out = np.full(z.shape, np.nan, dtype=np.float64)
    m = np.isfinite(z)
    if not np.any(m):
        return out
    lo = float(np.nanmin(z[m]))
    hi = float(np.nanmax(z[m]))
    if hi <= lo:
        out[m] = 0.5
    else:
        out[m] = (z[m].astype(np.float64) - lo) / (hi - lo)
    return out


def _pooled_quantile_bounds(a: np.ndarray, b: np.ndarray, lo_q: float, hi_q: float) -> tuple[float, float]:
    """Single vmin/vmax from concatenated finite robust scores (or any comparable arrays)."""
    fa = a[np.isfinite(a)]
    fb = b[np.isfinite(b)]
    pool = np.concatenate([fa.ravel(), fb.ravel()]) if fa.size or fb.size else np.array([0.0, 1.0])
    if pool.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.quantile(pool, lo_q)), float(np.quantile(pool, hi_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(pool)), float(np.nanmax(pool))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax


def _norm_vmin_vmax(arr: np.ndarray, lo_q: float, hi_q: float) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.quantile(finite, [lo_q, hi_q])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax


def _composite_rgb_with_raster(
    rgb_u8: np.ndarray,
    value_raster: np.ndarray,
    cmap_name: str,
    blend: float,
    lo_q: float,
    hi_q: float,
    stretch: str,
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
) -> np.ndarray:
    """
    rgb_u8: (3,H,W) uint8
    value_raster: (H,W) float, NaN = no overlay
    stretch: ``rank`` | ``quantile`` | ``fixed`` (use ``fixed_vmin``/``fixed_vmax`` on ``value_raster``).
    Returns float RGB (H,W,3) in [0,1]
    """
    rgb = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    rgb = np.moveaxis(rgb, 0, -1)
    valid = np.isfinite(value_raster)
    cmap = plt.get_cmap(cmap_name)
    if stretch == "rank":
        ranked = _rank01_within_aoi(value_raster)
        valid = np.isfinite(ranked)
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
        for_cmap = np.where(valid, ranked, 0.0)
    elif stretch == "quantile":
        vmin, vmax = _norm_vmin_vmax(value_raster, lo_q, hi_q)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        for_cmap = np.where(valid, value_raster, vmin)
    elif stretch == "fixed":
        if fixed_vmin is None or fixed_vmax is None:
            raise ValueError("stretch='fixed' requires fixed_vmin and fixed_vmax")
        norm = mcolors.Normalize(vmin=fixed_vmin, vmax=fixed_vmax, clip=True)
        for_cmap = np.where(valid, value_raster, fixed_vmin)
    else:
        raise ValueError(f"Unknown overlay stretch: {stretch!r}")
    rgba = cmap(norm(for_cmap))
    overlay_rgb = rgba[..., :3].astype(np.float32)
    w = blend * valid[..., None]
    return (1.0 - w) * rgb + w * overlay_rgb


def _imshow_rasterio_rgb(ax, rgb_chw: np.ndarray, transform: rasterio.Affine, title: str) -> None:
    h, w = rgb_chw.shape[1], rgb_chw.shape[2]
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, transform)
    rgb_hwc = np.moveaxis(rgb_chw, 0, -1)
    ax.imshow(
        np.clip(rgb_hwc, 0, 255).astype(np.uint8),
        extent=(left, right, bottom, top),
        origin="upper",
        interpolation="bilinear",
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")


def export_gis_raster_figure(
    out_root: Path,
    events: list[str],
    s2_max_scenes: int,
    scale_m: int,
    blend: float,
    write_geotiffs: bool,
    s2_window: str,
    overlay_cmap: str,
    overlay_stretch: str,
) -> None:
    from disaster_embedding_pipeline import _event_configurations
    from rasterio.warp import Resampling, reproject

    _ee_init()
    cfgs = _event_configurations()

    for event in events:
        if event not in cfgs:
            logger.warning("Skip %s (unknown event key)", event)
            continue
        cfg = cfgs[event]
        label = str(cfg.get("label", event))
        jp = out_root / event / "joined_inner.parquet"
        if not jp.is_file():
            logger.warning("Missing %s", jp)
            continue
        out_dir = out_root / event
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            frame_poly = joined_extent_polygon_wgs84(jp)
        except ValueError as e:
            logger.warning("%s: %s", event, e)
            continue

        region_ee = ee.Geometry(frame_poly.__geo_interface__)
        pre_y = int(cfg["pre_year"])
        post_y = int(cfg["post_year"])
        spec = cfg["benchmark"]
        metric_year = int(spec.metric_year)
        bench_matches = bool(cfg.get("benchmark_year_matches_post", True))
        temporal_relation = cfg.get(
            "benchmark_temporal_relation",
            (
                "reference_year_matches_post"
                if bench_matches
                else "legacy_calendar_mismatch"
            ),
        )
        note_path = out_dir / "gis_temporal_alignment.txt"
        note_path.write_text(
            _alignment_note_text(
                label,
                pre_y,
                post_y,
                spec.type,
                metric_year,
                bench_matches,
                temporal_relation,
                cfg.get("reference_validity_rule"),
                s2_window,
            ),
            encoding="utf-8",
        )
        vis_pre = _s2_mosaic_vis(region_ee, pre_y, max_scenes=s2_max_scenes, window=s2_window)
        vis_post = _s2_mosaic_vis(region_ee, post_y, max_scenes=s2_max_scenes, window=s2_window)

        pre_tif = out_dir / "_tmp_s2_pre.tif"
        post_tif = out_dir / "_tmp_s2_post.tif"
        scale_try = int(scale_m)
        for attempt in range(5):
            try:
                _download_s2_geotiff(vis_pre, region_ee, pre_tif, scale_try)
                _download_s2_geotiff(vis_post, region_ee, post_tif, scale_try)
                break
            except Exception as e:
                if attempt >= 4 or scale_try >= 360:
                    logger.exception("%s: GeoTIFF download failed", event)
                    raise
                scale_try = int(scale_try * 1.5)
                logger.warning("%s: retrying S2 download with coarser scale=%s m (%s)", event, scale_try, e)

        with rasterio.open(pre_tif) as src_pre:
            pre_prof = src_pre.profile.copy()
            pre_rgb = src_pre.read()
            pre_transform = src_pre.transform
            pre_crs = src_pre.crs

        with rasterio.open(post_tif) as src_post:
            post_rgb_raw = src_post.read()
            post_transform = src_post.transform
            post_crs = src_post.crs

        post_rgb = np.zeros_like(pre_rgb)
        n_b = min(3, post_rgb_raw.shape[0])
        for b in range(n_b):
            reproject(
                source=post_rgb_raw[b],
                destination=post_rgb[b],
                src_transform=post_transform,
                src_crs=post_crs,
                dst_transform=pre_transform,
                dst_crs=pre_crs,
                resampling=Resampling.bilinear,
            )
        post_prof = pre_prof.copy()

        gdf = _joined_gdf(jp)
        ecs_r = _rasterize_hex_column(gdf, "ecs_mean", pre_prof)
        bench_r = _rasterize_hex_column(gdf, "benchmark_mean", pre_prof)

        if write_geotiffs:
            nd = -9999.0
            meta1 = pre_prof.copy()
            meta1.update(count=1, dtype="float32", nodata=nd)
            with rasterio.open(out_dir / "s2_pre_grid.tif", "w", **pre_prof) as dst:
                dst.write(pre_rgb)
            with rasterio.open(out_dir / "s2_post_grid.tif", "w", **post_prof) as dst:
                dst.write(post_rgb)
            ecs_w = np.where(np.isfinite(ecs_r), ecs_r, nd).astype(np.float32)
            bench_w = np.where(np.isfinite(bench_r), bench_r, nd).astype(np.float32)
            with rasterio.open(out_dir / "raster_ecs_mean.tif", "w", **meta1) as dst:
                dst.write(ecs_w, 1)
            with rasterio.open(out_dir / "raster_benchmark_mean.tif", "w", **meta1) as dst:
                dst.write(bench_w, 1)

        if overlay_stretch == "per_layer_01":
            ecs_u = _linear01_raster(_robust_iqr_scores(ecs_r))
            bench_u = _linear01_raster(_robust_iqr_scores(bench_r))
            comp_pre_ecs = _composite_rgb_with_raster(
                pre_rgb, ecs_u, overlay_cmap, blend, 0.02, 0.98, "fixed", 0.0, 1.0
            )
            comp_post_bench = _composite_rgb_with_raster(
                post_rgb, bench_u, overlay_cmap, blend, 0.02, 0.98, "fixed", 0.0, 1.0
            )
            stretch_short = "per-layer IQR-z → linear [0,1]"
        elif overlay_stretch == "shared_robust":
            ecs_z = _robust_iqr_scores(ecs_r)
            bench_z = _robust_iqr_scores(bench_r)
            vmin, vmax = _pooled_quantile_bounds(ecs_z, bench_z, 0.02, 0.98)
            comp_pre_ecs = _composite_rgb_with_raster(
                pre_rgb, ecs_z, overlay_cmap, blend, 0.02, 0.98, "fixed", vmin, vmax
            )
            comp_post_bench = _composite_rgb_with_raster(
                post_rgb, bench_z, overlay_cmap, blend, 0.02, 0.98, "fixed", vmin, vmax
            )
            stretch_short = f"shared IQR-z; pooled q02–q98 [{vmin:.2f}, {vmax:.2f}]"
        else:
            comp_pre_ecs = _composite_rgb_with_raster(
                pre_rgb, ecs_r, overlay_cmap, blend, 0.02, 0.98, overlay_stretch
            )
            comp_post_bench = _composite_rgb_with_raster(
                post_rgb, bench_r, overlay_cmap, blend, 0.02, 0.98, overlay_stretch
            )
            stretch_short = (
                "rank 0–1 (within AOI)" if overlay_stretch == "rank" else "q02–q98 (per layer)"
            )

        out_png = out_dir / "gis_reference_raster_2x2.png"
        win_short = "Jun–Sep" if s2_window == "jun_sep" else "calendar yr"
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=140)
        _imshow_rasterio_rgb(
            axes[0, 0],
            pre_rgb,
            pre_transform,
            f"Sentinel-2 RGB ({pre_y}, {win_short})",
        )
        _imshow_rasterio_rgb(
            axes[0, 1],
            post_rgb,
            pre_transform,
            f"Sentinel-2 RGB ({post_y}, {win_short})",
        )
        h, w = pre_rgb.shape[1], pre_rgb.shape[2]
        left, bottom, right, top = rasterio.transform.array_bounds(h, w, pre_transform)
        axes[1, 0].imshow(
            np.clip(comp_pre_ecs, 0, 1),
            extent=(left, right, bottom, top),
            origin="upper",
            interpolation="bilinear",
        )
        axes[1, 0].set_title(
            f"RGB + ecs_mean (cmap={overlay_cmap}, {stretch_short}; α={blend:.2f})"
        )
        axes[1, 0].set_axis_off()
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 1].imshow(
            np.clip(comp_post_bench, 0, 1),
            extent=(left, right, bottom, top),
            origin="upper",
            interpolation="bilinear",
        )
        axes[1, 1].set_title(
            f"RGB + benchmark_mean (cmap={overlay_cmap}, {stretch_short}; α={blend:.2f})"
        )
        axes[1, 1].set_axis_off()
        axes[1, 1].set_aspect("equal", adjustable="box")
        fig.suptitle(
            label + f" — GIS grid (S2 window={s2_window}; see {note_path.name})"
        )
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s and %s (S2 scale=%s m)", out_png, note_path.name, scale_try)

        for p in (pre_tif, post_tif):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--events", type=str, default=",".join(FIRES_GOOD))
    p.add_argument("--s2-max-scenes", type=int, default=20)
    p.add_argument(
        "--scale-m",
        type=int,
        default=90,
        help="Sentinel-2 export pixel size in meters (coarser = smaller download, EE ~32 MB cap).",
    )
    p.add_argument(
        "--blend",
        type=float,
        default=0.62,
        help="Weight of colormap layer where hex raster is valid (0=pure RGB, 1=pure colormap).",
    )
    p.add_argument(
        "--write-geotiffs",
        action="store_true",
        help="Also write s2_*_grid.tif and raster_*_mean.tif next to the figure.",
    )
    p.add_argument(
        "--s2-window",
        choices=("jun_sep", "calendar_year"),
        default="jun_sep",
        help="Sentinel-2 date filter: jun_sep (default, clearer) vs calendar_year (closer to AlphaEarth annual span).",
    )
    p.add_argument(
        "--overlay-cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap for BOTH bottom-row ECS and benchmark blends.",
    )
    p.add_argument(
        "--overlay-stretch",
        choices=("per_layer_01", "shared_robust", "rank", "quantile"),
        default="per_layer_01",
        help="per_layer_01 (default): IQR-z per layer, then linear min--max to [0,1] each (full inferno incl. black). "
        "shared_robust: pooled q02--q98 on z (same vmin/vmax both). rank / quantile: as before.",
    )
    p.add_argument(
        "--ee-project",
        type=str,
        default=None,
        help="Google Cloud **Project ID** from console.cloud.google.com (IAM & Admin → Settings), "
        "not a display name or doc placeholder. Earth Engine must be enabled for that project. "
        "Sets EE_PROJECT_ID for this run.",
    )
    args = p.parse_args()
    if args.ee_project:
        os.environ["EE_PROJECT_ID"] = args.ee_project.strip()
    ev = [x.strip() for x in args.events.split(",") if x.strip()]
    export_gis_raster_figure(
        args.output_dir.resolve(),
        ev,
        s2_max_scenes=args.s2_max_scenes,
        scale_m=args.scale_m,
        blend=args.blend,
        write_geotiffs=args.write_geotiffs,
        s2_window=args.s2_window,
        overlay_cmap=args.overlay_cmap,
        overlay_stretch=args.overlay_stretch,
    )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    raise SystemExit(main())
