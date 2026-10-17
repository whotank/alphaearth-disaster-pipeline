#!/usr/bin/env python3
"""
Download true-color reference imagery for the *same map extent* as ``heatmap_join_side_by_side_basemap.png``:
bounds of joined H3 hexes in Web Mercator plus 6% padding (matches ``_plot_join_heatmaps_basemap``).

Writes per event (``--events`` comma list; default in ``main()`` is FIRES_GOOD):
  - ``sentinel2_reference_pre_rgb.png``  — Sentinel-2 SR composite, June–September of ``pre_year``
  - ``sentinel2_reference_post_rgb.png`` — same window for ``post_year``
  - ``sentinel2_reference_pre_post_side_by_side.png`` — two-panel figure (same layout idea as ECS/benchmark heatmaps)
  - ``REFERENCE_IMAGERY.txt`` — source and interpretation notes

Imagery is **Sentinel-2** (10 m, Copernicus programme), not piloted-aircraft orthophotography. For US-only
1 m aerials, NAIP is a possible future add-on (state/year coverage varies).
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import ee
import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry as sg

from fire_analysis import FIRES_GOOD
from verification_export import _download_thumb, _ee_init, _s2_mosaic_rgb_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Match disaster_embedding_pipeline._plot_join_heatmaps_basemap padding.
_EXTENT_PAD_FRAC = 0.06

_ATTRIB = """Reference imagery (Sentinel-2)
------------------------------
Contains modified Copernicus Sentinel data processed in Google Earth Engine.
Product: COPERNICUS/S2_SR_HARMONIZED — median of the N lowest-cloud scenes per date window.
Windows: June 1 – September 30 (northern growing season) for pipeline pre_year and post_year.
Extent: same as join basemap heatmaps — union of joined_inner H3 cells, EPSG:3857 bounds + 6% pad.

This is 10 m multispectral satellite RGB, not crewed-aircraft “aerial” orthophotos.
"""


def _h3_to_polygon(h: str) -> sg.Polygon:
    boundary = h3.cell_to_boundary(h)
    ring = [(float(lon), float(lat)) for lat, lon in boundary]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return sg.Polygon(ring)


def joined_extent_polygon_wgs84(joined_path: Path) -> sg.Polygon:
    """Padded map frame (EPSG:3857 bounds + 6% pad), reprojected to WGS84 — same as join basemap heatmaps."""
    df = pd.read_parquet(joined_path)
    if df.empty or "h3_index" not in df.columns:
        raise ValueError(f"No usable h3_index in {joined_path}")
    df = df.copy()
    df["h3_index"] = df["h3_index"].astype(str)
    df["geometry"] = df["h3_index"].map(_h3_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf_m = gdf.to_crs(epsg=3857)
    b = gdf_m.total_bounds
    span = max(b[2] - b[0], b[3] - b[1])
    pad = span * _EXTENT_PAD_FRAC
    box_m = sg.box(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    g_region = gpd.GeoSeries([box_m], crs="EPSG:3857").to_crs(4326).iloc[0]
    if isinstance(g_region, sg.MultiPolygon):
        g_region = max(g_region.geoms, key=lambda p: p.area)
    if not isinstance(g_region, sg.Polygon):
        g_region = g_region.convex_hull
    return g_region


def _extent_from_joined_parquet(joined_path: Path) -> ee.Geometry:
    poly = joined_extent_polygon_wgs84(joined_path)
    return ee.Geometry(poly.__geo_interface__)


def export_pre_post_reference(
    out_root: Path,
    events: list[str],
    s2_max_scenes: int,
    thumb_dimensions: int,
) -> None:
    from disaster_embedding_pipeline import _event_configurations

    _ee_init()
    cfgs = _event_configurations()

    for event in events:
        if event not in cfgs:
            logger.warning("Skip %s (unknown event key)", event)
            continue
        cfg = cfgs[event]
        jp = out_root / event / "joined_inner.parquet"
        if not jp.is_file():
            logger.warning("Missing %s — run pipeline for this event first", jp)
            continue

        try:
            region = _extent_from_joined_parquet(jp)
        except ValueError as e:
            logger.warning("%s: %s", event, e)
            continue

        out_dir = out_root / event
        out_dir.mkdir(parents=True, exist_ok=True)
        pre_y = int(cfg["pre_year"])
        post_y = int(cfg["post_year"])
        label = str(cfg.get("label", event))

        (out_dir / "REFERENCE_IMAGERY.txt").write_text(
            _ATTRIB + f"\nEvent: {label}\npre_year: {pre_y}\npost_year: {post_y}\nmax_scenes: {s2_max_scenes}\n",
            encoding="utf-8",
        )

        pre_png = out_dir / "sentinel2_reference_pre_rgb.png"
        post_png = out_dir / "sentinel2_reference_post_rgb.png"
        vis_pre = _s2_mosaic_rgb_image(region, pre_y, max_scenes=s2_max_scenes)
        vis_post = _s2_mosaic_rgb_image(region, post_y, max_scenes=s2_max_scenes)
        _download_thumb(vis_pre, region, pre_png, dimensions=thumb_dimensions)
        _download_thumb(vis_post, region, post_png, dimensions=thumb_dimensions)

        combo = out_dir / "sentinel2_reference_pre_post_side_by_side.png"
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=140)
            axes[0].imshow(plt.imread(pre_png))
            axes[0].set_axis_off()
            axes[0].set_title(f"Sentinel-2 reference — {pre_y} (Jun–Sep)")
            axes[1].imshow(plt.imread(post_png))
            axes[1].set_axis_off()
            axes[1].set_title(f"Sentinel-2 reference — {post_y} (Jun–Sep)")
            fig.suptitle(label + " — same extent as join basemap heatmap")
            fig.tight_layout()
            fig.savefig(combo, bbox_inches="tight")
            plt.close(fig)
            logger.info("Wrote %s", combo)
        except Exception:
            logger.exception("Failed to write combined figure %s", combo)
            plt.close("all")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--events",
        type=str,
        default=",".join(FIRES_GOOD),
        help="Comma-separated fire keys (default: FIRES_GOOD from fire_analysis).",
    )
    p.add_argument("--s2-max-scenes", type=int, default=20)
    p.add_argument("--thumb-dimensions", type=int, default=1024)
    p.add_argument(
        "--ee-project",
        type=str,
        default=None,
        help="Google Cloud project id for Earth Engine (sets EE_PROJECT_ID for this run).",
    )
    args = p.parse_args()
    if args.ee_project:
        os.environ["EE_PROJECT_ID"] = args.ee_project.strip()
    ev = [x.strip() for x in args.events.split(",") if x.strip()]
    export_pre_post_reference(
        args.output_dir.resolve(),
        ev,
        s2_max_scenes=args.s2_max_scenes,
        thumb_dimensions=args.thumb_dimensions,
    )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    raise SystemExit(main())
