#!/usr/bin/env python3
"""
Export (1) residual fire-candidate H3 hexes and (2) Sentinel-2 RGB reference mosaics (open data).

candidates — Score hexes from consensus embedding axes (weighted z-scored mean Δ²); flag
  low MODIS benchmark & high score → GeoJSON for human / imagery review.

imagery  — Download a cloud-screened Sentinel-2 SR median composite (Copernicus programme)
  as PNG via Earth Engine getThumbURL (true-color RGB).

all      — Run candidates then imagery for the same event list.

Attribution: Contains modified Copernicus Sentinel data [year] processed by Google Earth Engine.
See https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import requests
import shapely.geometry as sg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GOOD = [
    "bootleg_fire_2021",
    "caldor_fire_2021",
    "camp_fire_2018",
    "dixie_fire_2021",
    "mosquito_fire_2022",
    "thomas_fire_2017",
    "tubbs_fire_2017",
]
DEFAULT_POOR = ["fort_mcmurray_fire_2016", "marshall_fire_2021"]


def _resolve_ee_project_id() -> str | None:
    """Earth Engine High-Volume / Cloud API requires an explicit GCP project id."""
    return os.environ.get("EE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")


# Documentation placeholders that must never be used as GCP project IDs.
_DISALLOWED_EE_PROJECT_IDS = frozenset(
    {
        "your_project_id",
        "your-project-id",
        "your_gcp_project_id",
        "your-gcp-project-id",  # common copy-paste from docs / bundle examples
        "changeme",
        "example",
        "my-real-ee-project-123",
        "my-gcp-project-id",
    }
)


def _ee_init() -> None:
    pid = _resolve_ee_project_id()
    if not pid:
        raise SystemExit(
            "Earth Engine needs a Google Cloud project id. Set EE_PROJECT_ID or GOOGLE_CLOUD_PROJECT "
            "to your **actual** project id from Cloud Console (IAM & Admin → Settings → Project ID). "
            "See https://developers.google.com/earth-engine/guides/access"
        )
    pid = pid.strip()
    if pid.lower() in _DISALLOWED_EE_PROJECT_IDS:
        raise SystemExit(
            f"EE project id {pid!r} is a documentation example, not a real GCP project. "
            "In https://console.cloud.google.com/ open your lab project → copy **Project ID** "
            "(lowercase, often has digits) and pass that, e.g. "
            "`--ee-project $(gcloud config get-value project)` if gcloud points at the right project."
        )
    try:
        ee.Initialize(project=pid)
    except ee.EEException as e:
        raise SystemExit(
            f"Earth Engine could not use project {pid!r}.\n"
            f"Reason: {e}\n\n"
            "Confirm the id in Cloud Console (it must exist). Register Earth Engine / enable the API for "
            "that project, and sign in with an account that can use it (`earthengine authenticate` or ADC). "
            "https://developers.google.com/earth-engine/guides/access"
        )


def _h3_to_polygon(h: str) -> sg.Polygon:
    boundary = h3.cell_to_boundary(h)
    ring = [(float(lon), float(lat)) for lat, lon in boundary]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return sg.Polygon(ring)


def _load_consensus_bands(consensus_path: Path, top_n: int) -> tuple[list[str], dict[str, float]]:
    df = pd.read_csv(consensus_path).sort_values("mean_abs_r", ascending=False).head(top_n)
    bands = df["band"].astype(str).tolist()
    weights = {str(r["band"]): max(0.0, float(r["mean_r"])) for _, r in df.iterrows()}
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    return bands, weights


def _normalize_axis_df(df: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"h3_index": df["h3_index"].astype(str)})
    for b in bands:
        col = None
        for cand in (b, f"{b}_mean"):
            if cand in df.columns:
                col = cand
                break
        if col is None:
            for c in df.columns:
                if c == "h3_index":
                    continue
                if str(c).startswith(b) and "mean" in str(c).lower():
                    col = c
                    break
        out[b] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    return out


def export_candidates(
    out_root: Path,
    events: list[str],
    consensus_path: Path,
    top_bands: int,
    bench_low_q: float,
    score_high_q: float,
) -> None:
    from disaster_embedding_pipeline import (
        EMB_BANDS,
        _annual_embedding_mosaic,
        _buffer_point_meters,
        _event_configurations,
        _fc_to_dataframe,
        _h3_feature_collection,
        _polygon_to_h3_cells,
    )

    bands, wmap = _load_consensus_bands(consensus_path, top_bands)
    if not bands:
        raise SystemExit("No bands from consensus file")

    _ee_init()
    cfgs = _event_configurations()

    for event in events:
        if event not in cfgs or cfgs[event].get("disaster_type") != "fire":
            logger.warning("Skip %s (not a configured fire)", event)
            continue
        cfg = cfgs[event]
        jp = out_root / event / "joined_inner.parquet"
        if not jp.is_file():
            logger.warning("Missing %s — run pipeline first", jp)
            continue

        lon, lat = cfg["center_lon"], cfg["center_lat"]
        buf_m = float(cfg["processing_buffer_km"]) * 1000.0
        aoi_shapely = _buffer_point_meters(lon, lat, buf_m)
        aoi_ee = ee.Geometry(aoi_shapely.__geo_interface__)
        res = int(cfg["h3_resolution"])
        cells = _polygon_to_h3_cells(aoi_shapely, res)
        hex_fc = _h3_feature_collection(cells)

        pre = _annual_embedding_mosaic(int(cfg["pre_year"]), aoi_ee)
        post = _annual_embedding_mosaic(int(cfg["post_year"]), aoi_ee)
        # Only consensus bands (faster, smaller payload)
        missing = [b for b in bands if b not in EMB_BANDS]
        if missing:
            raise ValueError(f"Unknown bands vs EMB_BANDS: {missing}")
        axis_sq = post.subtract(pre).pow(2).select(bands)
        reduced = axis_sq.reduceRegions(collection=hex_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
        axis_df = _normalize_axis_df(_fc_to_dataframe(reduced), bands)

        joined = pd.read_parquet(jp)
        joined["h3_index"] = joined["h3_index"].astype(str)
        m = axis_df.merge(joined, on="h3_index", how="inner", suffixes=("", "_j"))
        if m.empty:
            logger.warning("%s: empty merge", event)
            continue

        bm = m["benchmark_mean"].astype(float)
        zcols = {}
        for b in bands:
            x = m[b].astype(float)
            std = float(x.std(ddof=0))
            mu = float(x.mean())
            zcols[b] = (x - mu) / std if std > 1e-12 else x * 0.0
        score = np.zeros(len(m), dtype=float)
        for b in bands:
            score += wmap.get(b, 0.0) * zcols[b].to_numpy()
        m["_axis_score"] = score
        m["_benchmark_mean"] = bm

        b_lo = float(bm.quantile(bench_low_q))
        s_hi = float(m["_axis_score"].quantile(score_high_q))
        cand = m[(bm <= b_lo) & (m["_axis_score"] >= s_hi)].copy()
        cand["geometry"] = cand["h3_index"].map(_h3_to_polygon)
        attr = cand[
            [
                "h3_index",
                "_axis_score",
                "_benchmark_mean",
                "ecs_mean",
                "geometry",
            ]
        ].rename(
            columns={
                "_axis_score": "axis_score_z",
                "_benchmark_mean": "benchmark_mean",
            }
        )
        gdf = gpd.GeoDataFrame(attr, geometry="geometry", crs="EPSG:4326")

        out_dir = out_root / event
        out_dir.mkdir(parents=True, exist_ok=True)
        gj = out_dir / "fire_candidates_residual.geojson"
        gdf.to_file(gj, driver="GeoJSON")
        meta = {
            "event": event,
            "n_hex_aoi": len(cells),
            "n_joined": int(len(m)),
            "n_candidates": int(len(gdf)),
            "consensus_file": str(consensus_path),
            "top_bands": bands,
            "weights_sum1": wmap,
            "benchmark_low_quantile": bench_low_q,
            "score_high_quantile": score_high_q,
            "benchmark_threshold_used": b_lo,
            "score_threshold_used": s_hi,
        }
        (out_dir / "fire_candidates_residual.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote %s (%s candidates)", gj, len(gdf))


def _s2_mosaic_rgb_image(region: ee.Geometry, year: int, max_scenes: int) -> ee.Image:
    """RGB composite: median of the clearest N Sentinel-2 scenes (June–Sept, north-temperate window)."""
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 9, 30)
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


def _download_thumb(
    image_vis: ee.Image,
    region: ee.Geometry,
    out_png: Path,
    dimensions: int = 1024,
    dimension_fallbacks: tuple[int, ...] = (768, 512, 384),
) -> None:
    # EE REST thumbnails often return 400 if the region polygon has many vertices (buffer circles).
    thumb_region = region.bounds()
    dim_chain = (int(dimensions),) + tuple(int(d) for d in dimension_fallbacks if int(d) < int(dimensions))
    last_err: str | None = None
    for dim in dim_chain:
        url = image_vis.getThumbURL(
            {
                "region": thumb_region,
                "dimensions": str(dim),
                "format": "png",
            }
        )
        r = requests.get(url, timeout=300)
        if r.ok:
            out_png.write_bytes(r.content)
            if dim != int(dimensions):
                logger.warning("Thumbnail used reduced size %s px (memory/limits)", dim)
            logger.info("Wrote %s", out_png)
            return
        snippet = (r.text or "")[:800]
        last_err = snippet
        if r.status_code == 400 and "memory" in snippet.lower() and dim > 256:
            logger.warning("Earth Engine thumbnail 400 (retry smaller): %s", snippet[:200])
            continue
        logger.error("Earth Engine thumbnail failed (%s): %s", r.status_code, snippet)
        r.raise_for_status()
    raise RuntimeError(f"Earth Engine thumbnail failed after retries: {last_err}")


def export_imagery(
    out_root: Path,
    events: list[str],
    include_candidates: bool,
    imagery_year: int | None,
    s2_max_scenes: int = 20,
    thumb_dimensions: int = 1024,
    imagery_buffer_km: float | None = None,
) -> None:
    from disaster_embedding_pipeline import _buffer_point_meters, _event_configurations

    _ee_init()
    cfgs = _event_configurations()

    attrib = (
        "Imagery: Contains modified Copernicus Sentinel data processed in Google Earth Engine.\n"
        "Programme: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi\n"
        "Collection: COPERNICUS/S2_SR_HARMONIZED — median of the N lowest-cloud scenes (June–Sept).\n"
    )

    for event in events:
        if event not in cfgs or cfgs[event].get("disaster_type") != "fire":
            continue
        cfg = cfgs[event]
        year = int(imagery_year if imagery_year is not None else cfg["post_year"])
        lon, lat = cfg["center_lon"], cfg["center_lat"]
        buf_km = float(cfg["processing_buffer_km"])
        if imagery_buffer_km is not None:
            buf_km = min(buf_km, float(imagery_buffer_km))
            if buf_km < float(cfg["processing_buffer_km"]):
                logger.info(
                    "%s: imagery AOI buffer %.1f km (capped from %.1f km)",
                    event,
                    buf_km,
                    float(cfg["processing_buffer_km"]),
                )
        buf_m = buf_km * 1000.0
        aoi_shapely = _buffer_point_meters(lon, lat, buf_m)
        aoi_ee = ee.Geometry(aoi_shapely.__geo_interface__)

        out_dir = out_root / event
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "IMAGERY_ATTRIBUTION.txt").write_text(
            attrib
            + f"Event: {event}\nImagery_year: {year}\nMax_scenes: {s2_max_scenes}\nThumb_px: {thumb_dimensions}\n",
            encoding="utf-8",
        )

        def _try_mosaic(geom: ee.Geometry, scenes: int) -> ee.Image:
            return _s2_mosaic_rgb_image(geom, year, max_scenes=scenes)

        scenes_try = int(s2_max_scenes)
        vis = _try_mosaic(aoi_ee, scenes_try)
        try:
            _download_thumb(vis, aoi_ee, out_dir / "sentinel2_mosaic_postfire_rgb.png", dimensions=thumb_dimensions)
        except RuntimeError:
            if scenes_try <= 8:
                raise
            scenes_try = max(8, scenes_try // 2)
            logger.warning("%s: retrying AOI mosaic with max_scenes=%s (EE memory limit)", event, scenes_try)
            vis = _try_mosaic(aoi_ee, scenes_try)
            _download_thumb(vis, aoi_ee, out_dir / "sentinel2_mosaic_postfire_rgb.png", dimensions=thumb_dimensions)

        if include_candidates:
            cg = out_dir / "fire_candidates_residual.geojson"
            if cg.is_file():
                gdf = gpd.read_file(cg)
                if len(gdf) == 0:
                    logger.warning("%s: no candidates for candidate imagery", event)
                else:
                    u = gdf.geometry.union_all()
                    buf = gpd.GeoSeries([u], crs="EPSG:4326").to_crs(3857).buffer(800).to_crs(4326).iloc[0]
                    reg = ee.Geometry(buf.__geo_interface__)
                    vis2 = _try_mosaic(reg, int(s2_max_scenes))
                    _download_thumb(
                        vis2,
                        reg,
                        out_dir / "sentinel2_candidates_extent_rgb.png",
                        dimensions=min(thumb_dimensions, 900),
                    )
            else:
                logger.info("%s: no fire_candidates_residual.geojson — run candidates first", event)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("candidates", help="Export residual candidate hex GeoJSON per event")
    pc.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pc.add_argument(
        "--events",
        type=str,
        default=",".join(DEFAULT_GOOD + DEFAULT_POOR),
        help="Comma-separated fire event keys.",
    )
    pc.add_argument(
        "--consensus",
        type=Path,
        default=Path("outputs/fire_axis_consensus_good_fires.csv"),
        help="Consensus CSV from fire_analysis.py axes/consensus.",
    )
    pc.add_argument("--top-bands", type=int, default=12, help="How many top consensus bands to use in the score.")
    pc.add_argument("--bench-low-q", type=float, default=0.35, help="Hexes at/below this benchmark quantile are 'low MODIS'.")
    pc.add_argument("--score-high-q", type=float, default=0.82, help="Hexes at/above this score quantile are 'high embedding'.")

    pi = sub.add_parser("imagery", help="Export Sentinel-2 RGB PNG for AOI and optional candidate extent")
    pi.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pi.add_argument("--events", type=str, default=",".join(DEFAULT_GOOD + DEFAULT_POOR))
    pi.add_argument(
        "--imagery-year",
        type=int,
        default=None,
        help="Override calendar year for June–Sept mosaic (default: each event's post_year).",
    )
    pi.add_argument(
        "--no-candidates",
        action="store_true",
        help="Do not try to export sentinel2_candidates_extent_rgb.png.",
    )
    pi.add_argument(
        "--s2-max-scenes",
        type=int,
        default=20,
        help="Median composite uses only the N lowest-cloud Sentinel-2 scenes (saves EE memory vs full-season median).",
    )
    pi.add_argument(
        "--thumb-dimensions",
        type=int,
        default=1024,
        help="Max thumbnail width/height in pixels; smaller values reduce EE memory.",
    )
    pi.add_argument(
        "--imagery-buffer-km",
        type=float,
        default=None,
        help="If set, caps the fire AOI radius (km) for Sentinel-2 export (smaller = less EE memory).",
    )

    pa = sub.add_parser("all", help="candidates then imagery (with candidate mosaics when present)")
    pa.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pa.add_argument("--events", type=str, default=",".join(DEFAULT_GOOD + DEFAULT_POOR))
    pa.add_argument("--consensus", type=Path, default=Path("outputs/fire_axis_consensus_good_fires.csv"))
    pa.add_argument("--top-bands", type=int, default=12)
    pa.add_argument("--bench-low-q", type=float, default=0.35)
    pa.add_argument("--score-high-q", type=float, default=0.82)
    pa.add_argument("--imagery-year", type=int, default=None)
    pa.add_argument("--s2-max-scenes", type=int, default=20)
    pa.add_argument("--thumb-dimensions", type=int, default=1024)
    pa.add_argument("--imagery-buffer-km", type=float, default=None)

    args = p.parse_args()
    out = args.output_dir.resolve()
    if args.cmd == "candidates":
        ev = [x.strip() for x in args.events.split(",") if x.strip()]
        export_candidates(out, ev, args.consensus.resolve(), args.top_bands, args.bench_low_q, args.score_high_q)
    elif args.cmd == "imagery":
        ev = [x.strip() for x in args.events.split(",") if x.strip()]
        export_imagery(
            out,
            ev,
            include_candidates=not args.no_candidates,
            imagery_year=args.imagery_year,
            s2_max_scenes=args.s2_max_scenes,
            thumb_dimensions=args.thumb_dimensions,
            imagery_buffer_km=args.imagery_buffer_km,
        )
    elif args.cmd == "all":
        ev = [x.strip() for x in args.events.split(",") if x.strip()]
        export_candidates(out, ev, args.consensus.resolve(), args.top_bands, args.bench_low_q, args.score_high_q)
        export_imagery(
            out,
            ev,
            include_candidates=True,
            imagery_year=args.imagery_year,
            s2_max_scenes=args.s2_max_scenes,
            thumb_dimensions=args.thumb_dimensions,
            imagery_buffer_km=args.imagery_buffer_km,
        )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    raise SystemExit(main())
