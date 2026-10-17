#!/usr/bin/env python3
"""
Phase 1 — AlphaEarth annual embeddings + disaster benchmarks on a common H3 grid.

Tasks: GEE extraction & safety probe (A), H3 aggregation of ECS (B), inner join to
benchmarks with temporal + string H3 integrity (C), integrity report + join heatmaps (D;
same colormap; per-layer IQR-z then **linear** rescale to ``[0,1]`` (min→0, max→1) so each panel
uses the full ``inferno`` range including black at the layer minimum).

Fire benchmark (US/Canada presets like Camp Fire): **NASA MODIS MCD64A1** on Earth Engine
(`MODIS/061/MCD64A1`), **calendar year** = ``metric_year`` / ``post_year``. Per-pixel value is
the **annual maximum of the ``BurnDate`` band** (ordinal burn day; **0** = no mapped burn that
year); H3 cells use the **mean** of that raster for ``benchmark_mean`` (spread for QA vs a
binary mask). Citation: Giglio et al., MCD64A1 burned area product, NASA LP DAAC.

**Forest-loss benchmark (non-fire):** ``hansen_gfc_lossyear`` uses **UMD Hansen GFC**
(``UMD/hansen/global_forest_change_2024_v1_12``). ``metric_year`` is a **calendar year**; the
``lossyear`` raster stores **codes 1–24 for loss in 2001–2024**, so we match ``lossyear ==
metric_year - 2000``. Per-pixel ``benchmark`` is **treecover2000 (0–1)** on matching loss
pixels, else **0**. H3 ``benchmark_mean`` is the spatial mean. Loss includes harvest, fire, and
other disturbance — not wildfire-only.

**AlphaEarth annual embeddings** in EE start **2017**; do not set ``pre_year`` to 2015/2016 or
the mosaic is empty (no bands). Legacy fires (e.g. 2016 burn) use ``benchmark_year_matches_post:
False`` so MODIS can stay on the **true burn year** while ECS uses the first valid pair (e.g.
2017→2018).

Run (after `earthengine authenticate`):
  python disaster_embedding_pipeline.py --event camp_fire_2018

Optional: set ``EE_PROJECT_ID`` or ``GOOGLE_CLOUD_PROJECT`` to your Google Cloud project id for
Earth Engine (see https://developers.google.com/earth-engine/guides/auth ).

Run all four configured events and merge integrity JSON:
  python disaster_embedding_pipeline.py --all-events

Run only **good** fires (``FIRES_GOOD`` in ``fire_analysis.py``) and merge integrity JSON:
  python disaster_embedding_pipeline.py --good-fires

Multi-event manifest + batch figures (see ``output_events_bundle.py``)::
  python output_events_bundle.py manifest --output-dir outputs
  python output_events_bundle.py figures --output-dir outputs --ee-project YOUR_GCP_PROJECT_ID
  python output_events_bundle.py all --output-dir outputs --ee-project YOUR_GCP_PROJECT_ID

Attribution: AlphaEarth Foundations Satellite Embedding dataset (CC-BY 4.0), Google / Google DeepMind.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import ee
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so
from pyproj import Geod

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WGS84 = Geod(ellps="WGS84")
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMB_BANDS = [f"A{i:02d}" for i in range(64)]
DEFAULT_EE_H3_BATCH_SIZE = 40
DEFAULT_EE_BATCH_RETRIES = 2

# Side-by-side join heatmaps: same colormap; each panel normalized to [0,1] for identical colorbar span.
JOIN_HEATMAP_CMAP = "inferno"


def _robust_iqr_scores_1d(x: np.ndarray) -> np.ndarray:
    """Per-column (x - median) / IQR; NaN preserved (same idea as GIS reference figure)."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    m = np.isfinite(x)
    if not np.any(m):
        return out
    arr = x[m].astype(np.float64)
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr < 1e-12:
        sd = float(np.std(arr))
        iqr = sd if sd > 1e-12 else 1.0
    out[m] = (x[m].astype(np.float64) - med) / iqr
    return out


def _linear_01_within(arr: np.ndarray) -> np.ndarray:
    """Linear map finite values to ``[0,1]``: min→0, max→1 (ties at extrema share 0 or 1)."""
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    m = np.isfinite(arr)
    if not np.any(m):
        return out
    lo = float(np.nanmin(arr[m]))
    hi = float(np.nanmax(arr[m]))
    if hi <= lo:
        out[m] = 0.5
    else:
        out[m] = (arr[m].astype(np.float64) - lo) / (hi - lo)
    return out


def _join_heatmap_plot_columns(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float, float]:
    """Per-layer IQR-z, then **linear** ``[0,1]`` rescale per panel (``vmin=0``, ``vmax=1``)."""
    ecs_z = _robust_iqr_scores_1d(gdf["ecs_mean"].to_numpy(dtype=float, copy=False))
    bench_z = _robust_iqr_scores_1d(gdf["benchmark_mean"].to_numpy(dtype=float, copy=False))
    ecs_u = _linear_01_within(ecs_z)
    bench_u = _linear_01_within(bench_z)
    out = gdf.copy()
    out["_ecs_plot_z"] = ecs_u
    out["_bench_plot_z"] = bench_u
    return out, 0.0, 1.0


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark layer definition. For MODIS/Hansen, ``metric_year`` is the calendar year of the product."""

    type: str  # MODIS, Hansen, JRC, GeoJSON, or USDM seasonal change
    metric_year: int
    reference_pre_year: int | None = None
    geojson_path: str | None = None
    valid_geojson_path: str | None = None
    exclude_worldcover_water: bool = False
    nominal_scale_m: int | None = None


def _event_configurations() -> dict[str, dict[str, Any]]:
    """
    Swap `active_event` in main or pass `--event` to run other sites.
    Coordinates are approximate event centers; tune buffers for your study extent.
    """
    return {
        "camp_fire_2018": {
            "label": "Camp Fire (California, 2018)",
            "center_lon": -121.42,
            "center_lat": 39.76,
            "pre_year": 2017,
            "post_year": 2018,
            "processing_buffer_km": 22.0,
            "probe_center_radius_m": 2500.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 90.0,
            "h3_resolution": 8,
            "max_h3_cells": 6000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(
                type="modis_mcd64a1",
                metric_year=2018,
            ),
        },
        # --- More CONUS / Canada wildfires (same official benchmark as Camp Fire; see module docstring.) ---
        "dixie_fire_2021": {
            "label": "Dixie Fire (California, 2021)",
            "center_lon": -121.38,
            "center_lat": 40.02,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 38.0,
            "probe_center_radius_m": 4000.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2021),
        },
        "marshall_fire_2021": {
            "label": "Marshall Fire (Colorado, 2021) — ECS 2021→2022; MODIS burn 2021 (Dec 30 fire)",
            "center_lon": -105.17,
            "center_lat": 39.95,
            "pre_year": 2021,
            "post_year": 2022,
            "benchmark_year_matches_post": False,
            "processing_buffer_km": 14.0,
            "probe_center_radius_m": 2500.0,
            "control_offset_km": 35.0,
            "control_bearing_deg": 90.0,
            "h3_resolution": 8,
            "max_h3_cells": 6000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2021),
        },
        "thomas_fire_2017": {
            "label": "Thomas Fire (Southern California, 2017) — ECS 2017→2018; MODIS burn 2017",
            "center_lon": -119.28,
            "center_lat": 34.42,
            "pre_year": 2017,
            "post_year": 2018,
            "benchmark_year_matches_post": False,
            "processing_buffer_km": 32.0,
            "probe_center_radius_m": 3500.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 270.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2017),
        },
        "tubbs_fire_2017": {
            "label": "Tubbs Fire (Sonoma / Napa, California, 2017) — ECS 2017→2018; MODIS burn 2017",
            "center_lon": -122.62,
            "center_lat": 38.50,
            "pre_year": 2017,
            "post_year": 2018,
            "benchmark_year_matches_post": False,
            "processing_buffer_km": 22.0,
            "probe_center_radius_m": 3000.0,
            "control_offset_km": 45.0,
            "control_bearing_deg": 315.0,
            "h3_resolution": 8,
            "max_h3_cells": 6000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2017),
        },
        "fort_mcmurray_fire_2016": {
            "label": "Fort McMurray wildfire (Alberta, 2016) — ECS 2017→2018; MODIS burn 2016",
            "center_lon": -111.38,
            "center_lat": 56.73,
            "pre_year": 2017,
            "post_year": 2018,
            "benchmark_year_matches_post": False,
            "processing_buffer_km": 45.0,
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 55.0,
            "control_bearing_deg": 45.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2016),
        },
        "caldor_fire_2021": {
            "label": "Caldor Fire (California, 2021)",
            "center_lon": -120.31,
            "center_lat": 38.78,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 32.0,
            "probe_center_radius_m": 3500.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 225.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2021),
        },
        "bootleg_fire_2021": {
            "label": "Bootleg Fire (Oregon, 2021)",
            "center_lon": -121.25,
            "center_lat": 42.42,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 38.0,
            "probe_center_radius_m": 4000.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 90.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2021),
        },
        "mosquito_fire_2022": {
            "label": "Mosquito Fire (California, 2022)",
            "center_lon": -120.55,
            "center_lat": 39.05,
            "pre_year": 2021,
            "post_year": 2022,
            "processing_buffer_km": 28.0,
            "probe_center_radius_m": 3500.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(type="modis_mcd64a1", metric_year=2022),
        },
        "quebec_fires_2023": {
            "label": "Quebec wildfires (2023, Chibougamau / Lac Saint-Jean corridor)",
            "center_lon": -73.85,
            "center_lat": 49.35,
            "pre_year": 2022,
            "post_year": 2023,
            "processing_buffer_km": 55.0,
            "probe_center_radius_m": 4000.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 270.0,
            "h3_resolution": 7,
            "max_h3_cells": 8000,
            "disaster_type": "fire",
            "benchmark": BenchmarkSpec(
                type="modis_mcd64a1",
                metric_year=2023,
            ),
        },
        "pakistan_floods_2022": {
            "label": "Pakistan floods (2022)",
            "center_lon": 68.35,
            "center_lat": 26.85,
            "pre_year": 2021,
            "post_year": 2022,
            "processing_buffer_km": 55.0,
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 45.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "flood",
            "benchmark": BenchmarkSpec(
                type="jrc_gsw_occurrence",
                metric_year=2022,
            ),
        },
        "brazil_floods_2024": {
            "label": "Rio Grande do Sul floods (Brazil, 2024)",
            "center_lon": -51.2,
            "center_lat": -29.95,
            "pre_year": 2023,
            "post_year": 2024,
            "processing_buffer_km": 60.0,
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 50.0,
            "control_bearing_deg": 135.0,
            "h3_resolution": 7,
            "max_h3_cells": 9000,
            "disaster_type": "flood",
            "benchmark": BenchmarkSpec(
                type="jrc_gsw_occurrence",
                metric_year=2024,
            ),
        },
        # --- Persistent volcanic emplacement bracketed by full pre/post years ---
        "kilauea_lava_2018": {
            "label": "Kilauea lower East Rift Zone lava emplacement (USGS, 2018)",
            "center_lon": -154.8668,
            "center_lat": 19.4770,
            "pre_year": 2017,
            "post_year": 2019,
            "benchmark_year_matches_post": False,
            "benchmark_temporal_relation": "event_between_annual_mosaics",
            "processing_buffer_km": 12.0,
            "probe_center_radius_m": 2500.0,
            "control_offset_km": 25.0,
            "control_bearing_deg": 270.0,
            "h3_resolution": 8,
            "max_h3_cells": 4000,
            "disaster_type": "volcanic_eruption",
            "reference_validity_rule": (
                "ESA WorldCover 2020 non-water support, union the official "
                "USGS lava footprint where new land formed"
            ),
            "reference_metadata_path": (
                "data/references/kilauea_2018_usgs/reference_metadata.json"
            ),
            "benchmark": BenchmarkSpec(
                type="geojson_binary",
                metric_year=2018,
                geojson_path=(
                    "data/references/kilauea_2018_usgs/"
                    "lava_flow_footprint.geojson"
                ),
                exclude_worldcover_water=True,
                nominal_scale_m=10,
            ),
        },
        "mauna_loa_lava_2022": {
            "label": "Mauna Loa summit and Northeast Rift Zone lava emplacement (USGS, 2022)",
            "center_lon": -155.5717,
            "center_lat": 19.5439,
            "pre_year": 2021,
            "post_year": 2023,
            "benchmark_year_matches_post": False,
            "benchmark_temporal_relation": "event_between_annual_mosaics",
            "processing_buffer_km": 18.0,
            "probe_center_radius_m": 3000.0,
            "control_offset_km": 35.0,
            "control_bearing_deg": 90.0,
            "h3_resolution": 8,
            "max_h3_cells": 6000,
            "disaster_type": "volcanic_eruption",
            "reference_validity_rule": (
                "ESA WorldCover 2020 non-water support, union the official "
                "USGS lava footprint"
            ),
            "reference_metadata_path": (
                "data/references/mauna_loa_2022_usgs/reference_metadata.json"
            ),
            "benchmark": BenchmarkSpec(
                type="geojson_binary",
                metric_year=2022,
                geojson_path=(
                    "data/references/mauna_loa_2022_usgs/"
                    "lava_flow_footprint.geojson"
                ),
                exclude_worldcover_water=True,
                nominal_scale_m=10,
            ),
        },
        "la_palma_lava_2021": {
            "label": "La Palma Cumbre Vieja lava emplacement (Copernicus EMS, 2021)",
            "center_lon": -17.8990,
            "center_lat": 28.6157,
            "pre_year": 2020,
            "post_year": 2022,
            "benchmark_year_matches_post": False,
            "benchmark_temporal_relation": "event_between_annual_mosaics",
            "processing_buffer_km": 4.5,
            "probe_center_radius_m": 1500.0,
            "control_offset_km": 12.0,
            "control_bearing_deg": 0.0,
            "h3_resolution": 9,
            "max_h3_cells": 3000,
            "disaster_type": "volcanic_eruption",
            "reference_validity_rule": (
                "CEMS post-event image footprint minus Not Analysed polygons, "
                "intersected with ESA WorldCover 2020 non-water support and "
                "unioned with the official lava footprint"
            ),
            "reference_metadata_path": (
                "data/references/la_palma_2021_cems/reference_metadata.json"
            ),
            "benchmark": BenchmarkSpec(
                type="geojson_binary",
                metric_year=2021,
                geojson_path=(
                    "data/references/la_palma_2021_cems/"
                    "lava_flow_footprint.geojson"
                ),
                valid_geojson_path=(
                    "data/references/la_palma_2021_cems/"
                    "valid_observation_footprint.geojson"
                ),
                exclude_worldcover_water=True,
                nominal_scale_m=10,
            ),
        },
        # --- Season-scale drought transitions from fixed April-October USDM means ---
        "four_corners_drought_2018": {
            "label": "Four Corners drought intensification (Upper San Juan, 2018)",
            "center_lon": -107.9260,
            "center_lat": 36.6931,
            "pre_year": 2017,
            "post_year": 2018,
            "processing_buffer_km": 0.0,
            "aoi_geojson_path": (
                "data/references/usdm/aoi_upper_san_juan_huc6.geojson"
            ),
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 100.0,
            "control_bearing_deg": 90.0,
            "h3_resolution": 6,
            "max_h3_cells": 2000,
            "disaster_type": "drought",
            "reference_validity_rule": (
                "Official USGS HUC6 boundary; every weekly USDM map from "
                "April 1 through October 31; None=0 and D0-D4=1-5"
            ),
            "reference_source_state": (
                "official_vector_crosscheck_required"
            ),
            "benchmark": BenchmarkSpec(
                type="usdm_season_delta",
                reference_pre_year=2017,
                metric_year=2018,
                nominal_scale_m=500,
            ),
        },
        "north_dakota_drought_2021": {
            "label": "North Dakota drought intensification (2021)",
            "center_lon": -100.4694,
            "center_lat": 47.4463,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 0.0,
            "aoi_geojson_path": (
                "data/references/usdm/aoi_north_dakota.geojson"
            ),
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 150.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 6,
            "max_h3_cells": 6000,
            "disaster_type": "drought",
            "reference_validity_rule": (
                "Official Census state boundary; every weekly USDM map from "
                "April 1 through October 31; None=0 and D0-D4=1-5"
            ),
            "reference_source_state": (
                "official_vector_crosscheck_required"
            ),
            "benchmark": BenchmarkSpec(
                type="usdm_season_delta",
                reference_pre_year=2020,
                metric_year=2021,
                nominal_scale_m=500,
            ),
        },
        "iowa_drought_recovery_2024": {
            "label": "Iowa drought recovery transition (2024)",
            "center_lon": -93.5001,
            "center_lat": 42.0746,
            "pre_year": 2023,
            "post_year": 2024,
            "processing_buffer_km": 0.0,
            "aoi_geojson_path": "data/references/usdm/aoi_iowa.geojson",
            "probe_center_radius_m": 5000.0,
            "control_offset_km": 150.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 6,
            "max_h3_cells": 5000,
            "disaster_type": "drought",
            "reference_validity_rule": (
                "Official Census state boundary; every weekly USDM map from "
                "April 1 through October 31; None=0 and D0-D4=1-5"
            ),
            "reference_source_state": (
                "official_vector_crosscheck_required"
            ),
            "benchmark": BenchmarkSpec(
                type="usdm_season_delta",
                reference_pre_year=2023,
                metric_year=2024,
                nominal_scale_m=500,
            ),
        },
        # --- Forest loss (Hansen GFC annual lossyear); pick a **forested** AOI or benchmark is flat ---
        "pacific_nw_forest_loss_2021": {
            "label": "Pacific NW forest loss (Hansen GFC 2021; treecover2000 × annual loss)",
            "center_lon": -123.35,
            "center_lat": 45.52,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 28.0,
            "probe_center_radius_m": 3500.0,
            "control_offset_km": 45.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 8,
            "max_h3_cells": 12000,
            "disaster_type": "forest_loss",
            "benchmark": BenchmarkSpec(type="hansen_gfc_lossyear", metric_year=2021),
        },
        "georgia_forest_loss_2021": {
            "label": "Southeast US forest loss (Hansen GFC 2021; treecover2000 × annual loss)",
            "center_lon": -81.97,
            "center_lat": 32.754,
            "pre_year": 2020,
            "post_year": 2021,
            "processing_buffer_km": 28.0,
            "probe_center_radius_m": 3500.0,
            "control_offset_km": 45.0,
            "control_bearing_deg": 180.0,
            "h3_resolution": 8,
            "max_h3_cells": 12000,
            "disaster_type": "forest_loss",
            "benchmark": BenchmarkSpec(type="hansen_gfc_lossyear", metric_year=2021),
        },
    }


def _offset_point_m(lon: float, lat: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Return (lon, lat) of a point forward along a geodesic."""
    lon2, lat2, _ = WGS84.fwd(lon, lat, bearing_deg, distance_m)
    return float(lon2), float(lat2)


def _ee_point(lon: float, lat: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat])


def _annual_embedding_mosaic(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date(f"{year}-01-01")
    end = start.advance(1, "year")
    ic = (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .filterBounds(region)
        .select(EMB_BANDS)
    )
    return ic.mosaic().clip(region).set("emb_year", year)


def _ecs_image(pre: ee.Image, post: ee.Image) -> ee.Image:
    diff = post.subtract(pre)
    return diff.pow(2).reduce(ee.Reducer.sum()).rename("ECS").float()


# Hansen GFC v1.12 on Earth Engine. ``lossyear`` band is **not** a 4-digit calendar year: it is
# an integer code 1–24 meaning loss primarily in **2001–2024** (see EE dataset catalog).
_HANSEN_GFC_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
_HANSEN_CALENDAR_YEAR_MIN = 2001
_HANSEN_CALENDAR_YEAR_MAX = 2024


def _hansen_lossyear_code(calendar_year: int) -> int:
    """Map calendar year (e.g. 2021) to Hansen ``lossyear`` pixel code (e.g. 21)."""
    code = int(calendar_year) - 2000
    if code < 1 or code > (_HANSEN_CALENDAR_YEAR_MAX - 2000):
        raise ValueError(
            f"Hansen lossyear codes exist for calendar years "
            f"{_HANSEN_CALENDAR_YEAR_MIN}–{_HANSEN_CALENDAR_YEAR_MAX}; got {calendar_year}"
        )
    return code


def _hansen_annual_forest_loss_image(metric_year: int, region: ee.Geometry) -> ee.Image:
    """
    Tree-cover-weighted annual loss from Hansen ``lossyear`` × ``treecover2000``.

    ``metric_year`` is a **calendar year** (e.g. 2021); we match Hansen's encoded ``lossyear``
    value ``metric_year - 2000`` (e.g. 21 for 2021).

    Per-pixel ``benchmark`` is ``treecover2000/100`` (0–1) on those loss pixels, else 0.
    H3 ``benchmark_mean`` is the spatial mean over the hex.

    Hansen loss aggregates harvest, wildfire, mortality, etc.; it is not a wildfire-only mask.
    """
    if metric_year < _HANSEN_CALENDAR_YEAR_MIN or metric_year > _HANSEN_CALENDAR_YEAR_MAX:
        raise ValueError(
            f"Hansen forest benchmark only for {_HANSEN_CALENDAR_YEAR_MIN}–{_HANSEN_CALENDAR_YEAR_MAX}; "
            f"got {metric_year}"
        )
    lyr = _hansen_lossyear_code(metric_year)
    gfc = ee.Image(_HANSEN_GFC_ASSET)
    lossyear = gfc.select("lossyear").unmask(0)
    tree2000 = gfc.select("treecover2000").float().divide(100.0).clamp(0.0, 1.0).unmask(0)
    mask = lossyear.eq(lyr).float()
    return tree2000.multiply(mask).rename("benchmark").clip(region)


def _modis_burn_probability_image(metric_year: int, region: ee.Geometry) -> ee.Image:
    """
    MODIS/061/MCD64A1: annual per-pixel max of ``BurnDate`` (ordinal burn day in the product;
    0 means no burn that year). H3 ``mean`` is then a continuous 0–366-ish field.

    A strict 0/1 burned mask often collapses to 1.0 for every hex inside large fires at 500 m,
    which makes QA correlations undefined; this preserves official MODIS while retaining spread.
    """
    start = ee.Date(f"{metric_year}-01-01")
    end = start.advance(1, "year")
    coll = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start, end).select("BurnDate")
    burn_day = coll.max().rename("benchmark").float().unmask(0)
    return burn_day.clip(region)


def _jrc_occurrence_image(region: ee.Geometry) -> ee.Image:
    """
    Static JRC GSW occurrence (0–100). For floods, prefer UNOSAT GeoJSON (`geojson_binary`)
    for true event extents; this is a catalog placeholder for end-to-end runs.
    """
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").float()
    return jrc.clip(region).rename("benchmark")


_USDM_ASSET = "projects/sat-io/open-datasets/us-drought-monitor"


def _usdm_season_score(year: int, region: ee.Geometry) -> ee.Image:
    """April through October mean USDM score: None=0, D0=1, ..., D4=5."""
    start = ee.Date(f"{year}-04-01")
    end = ee.Date(f"{year}-11-01")
    collection = ee.ImageCollection(_USDM_ASSET).filterDate(start, end)

    def _score(image: ee.Image) -> ee.Image:
        return image.select("DM").add(1).unmask(0).rename("score").float()

    return collection.map(_score).mean().clip(region)


def _usdm_season_image_count(year: int) -> int:
    return int(
        ee.ImageCollection(_USDM_ASSET)
        .filterDate(f"{year}-04-01", f"{year}-11-01")
        .size()
        .getInfo()
    )


def _usdm_season_delta_image(
    spec: BenchmarkSpec,
    region: ee.Geometry,
) -> ee.Image:
    if spec.reference_pre_year is None:
        raise ValueError("usdm_season_delta requires reference_pre_year")
    before = _usdm_season_score(int(spec.reference_pre_year), region)
    after = _usdm_season_score(int(spec.metric_year), region)
    return after.subtract(before).abs().rename("benchmark").float()


def _geojson_feature_collection(geojson_path: str) -> ee.FeatureCollection:
    with Path(geojson_path).expanduser().resolve().open(encoding="utf-8") as f:
        gj = json.load(f)
    return ee.FeatureCollection(gj.get("features", gj))


def _geojson_benchmark_image(
    spec: BenchmarkSpec,
    region: ee.Geometry,
) -> ee.Image:
    if not spec.geojson_path:
        raise ValueError("geojson_binary benchmark requires geojson_path")
    positive_fc = _geojson_feature_collection(spec.geojson_path)
    positive_mask = ee.Image(0).byte().paint(positive_fc, 1).clip(region)

    valid_mask = ee.Image(1).byte().clip(region)
    if spec.valid_geojson_path:
        valid_fc = _geojson_feature_collection(spec.valid_geojson_path)
        valid_mask = ee.Image(0).byte().paint(valid_fc, 1).clip(region)
    if spec.exclude_worldcover_water:
        worldcover_land = (
            ee.ImageCollection("ESA/WorldCover/v100")
            .first()
            .select("Map")
            .neq(80)
            .unmask(0)
            .clip(region)
        )
        valid_mask = valid_mask.And(worldcover_land)

    # Event polygons remain valid where the event created new land over water.
    valid_mask = valid_mask.Or(positive_mask).selfMask()
    base = ee.Image(0).float().updateMask(valid_mask)
    return base.paint(positive_fc, 1).rename("benchmark").clip(region)


def _benchmark_image(spec: BenchmarkSpec, region: ee.Geometry) -> ee.Image:
    if spec.type == "modis_mcd64a1":
        return _modis_burn_probability_image(spec.metric_year, region)
    if spec.type == "hansen_gfc_lossyear":
        return _hansen_annual_forest_loss_image(spec.metric_year, region)
    if spec.type == "jrc_gsw_occurrence":
        return _jrc_occurrence_image(region)
    if spec.type == "usdm_season_delta":
        return _usdm_season_delta_image(spec, region)
    if spec.type == "geojson_binary":
        return _geojson_benchmark_image(spec, region)
    raise ValueError(f"Unknown benchmark type: {spec.type}")


def _buffer_point_meters(lon: float, lat: float, buffer_m: float) -> sg.Polygon:
    """Disk AOI in WGS84 from geodesic buffer via projected coordinates (geopandas)."""
    g = gpd.GeoSeries([sg.Point(lon, lat)], crs="EPSG:4326").to_crs(3857)
    b = g.buffer(float(buffer_m)).to_crs(4326)
    geom = b.iloc[0]
    if isinstance(geom, sg.MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    if not isinstance(geom, sg.Polygon):
        raise TypeError("Expected Polygon buffer geometry")
    return geom


def _polygon_to_h3_cells(
    geometry: sg.Polygon | sg.MultiPolygon,
    resolution: int,
) -> list[str]:
    polygons = (
        list(geometry.geoms)
        if isinstance(geometry, sg.MultiPolygon)
        else [geometry]
    )
    cells: set[str] = set()
    for polygon in polygons:
        exterior = [
            (lat, lon)
            for lon, lat in polygon.exterior.coords[:-1]
        ]
        if len(exterior) < 3:
            continue
        holes = [
            [(lat, lon) for lon, lat in ring.coords[:-1]]
            for ring in polygon.interiors
        ]
        h3_polygon = h3.LatLngPoly(exterior, *holes)
        cells.update(str(cell) for cell in h3.polygon_to_cells(h3_polygon, resolution))
    if not cells:
        raise ValueError("AOI produced no H3 cells")
    return sorted(cells)


def _configured_aoi(
    cfg: dict[str, Any],
) -> sg.Polygon | sg.MultiPolygon:
    aoi_path = cfg.get("aoi_geojson_path")
    if not aoi_path:
        return _buffer_point_meters(
            float(cfg["center_lon"]),
            float(cfg["center_lat"]),
            float(cfg["processing_buffer_km"]) * 1000.0,
        )
    frame = gpd.read_file(Path(aoi_path).expanduser().resolve())
    if frame.empty or frame.crs is None:
        raise ValueError(f"AOI file is empty or lacks CRS: {aoi_path}")
    geometry = frame.to_crs(4326).geometry.make_valid().union_all()
    if isinstance(geometry, sg.GeometryCollection):
        polygon_parts: list[sg.Polygon] = []
        for part in geometry.geoms:
            if isinstance(part, sg.Polygon):
                polygon_parts.append(part)
            elif isinstance(part, sg.MultiPolygon):
                polygon_parts.extend(part.geoms)
        geometry = so.unary_union(polygon_parts)
    if not isinstance(geometry, (sg.Polygon, sg.MultiPolygon)):
        raise TypeError(f"AOI must be polygonal: {aoi_path}")
    return geometry


def _h3_to_ee_polygon(h: str) -> ee.Geometry:
    boundary = h3.cell_to_boundary(h)
    ring = [[float(lon), float(lat)] for lat, lon in boundary]
    ring.append(ring[0])
    return ee.Geometry.Polygon([ring], proj="EPSG:4326", geodesic=False)


def _h3_feature_collection(hex_ids: Iterable[str]) -> ee.FeatureCollection:
    feats = []
    for h in hex_ids:
        feats.append(ee.Feature(_h3_to_ee_polygon(h), {"h3_index": str(h)}))
    return ee.FeatureCollection(feats)


def _fc_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    feats = info.get("features", [])
    rows: list[dict[str, Any]] = []
    for f in feats:
        props = dict(f.get("properties", {}))
        if "h3_index" in props and props["h3_index"] is not None:
            props["h3_index"] = str(props["h3_index"])
        rows.append(props)
    return pd.DataFrame(rows)


def _validate_h3_coverage(
    frame: pd.DataFrame,
    expected_hex_ids: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Require exactly one result row per expected H3 ID and restore expected order."""
    expected = [str(hex_id) for hex_id in expected_hex_ids]
    expected_counts = Counter(expected)
    expected_duplicates = sorted(
        hex_id for hex_id, count in expected_counts.items() if count > 1
    )
    if expected_duplicates:
        raise ValueError(
            f"{context}: duplicate requested H3 IDs: {expected_duplicates[:10]}"
        )
    if "h3_index" not in frame.columns:
        raise ValueError(
            f"{context}: Earth Engine response lacks h3_index; "
            f"columns={list(frame.columns)}"
        )
    if frame["h3_index"].isna().any():
        raise ValueError(f"{context}: Earth Engine response contains null H3 IDs")

    out = frame.copy()
    out["h3_index"] = out["h3_index"].astype(str)
    actual = out["h3_index"].tolist()
    actual_counts = Counter(actual)
    duplicates = sorted(
        hex_id for hex_id, count in actual_counts.items() if count > 1
    )
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if duplicates or missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            f"{context}: invalid H3 coverage "
            f"(expected_rows={len(expected)}, actual_rows={len(actual)}, "
            f"duplicates={duplicates[:10]}, missing={missing[:10]}, "
            f"unexpected={unexpected[:10]})"
        )

    order = {hex_id: position for position, hex_id in enumerate(expected)}
    out["_expected_h3_order"] = out["h3_index"].map(order)
    out = (
        out.sort_values("_expected_h3_order", kind="stable")
        .drop(columns="_expected_h3_order")
        .reset_index(drop=True)
    )
    return out


def _reduce_regions_h3_batched(
    image: ee.Image,
    hex_ids: Iterable[str],
    *,
    reducer: ee.Reducer,
    scale: int,
    batch_size: int = DEFAULT_EE_H3_BATCH_SIZE,
    tile_scale: int = 4,
    context: str,
    max_retries: int = DEFAULT_EE_BATCH_RETRIES,
) -> pd.DataFrame:
    """
    Reduce H3 cells in deterministic client-side batches and validate exact coverage.

    Each cell still receives an independent ``reduceRegions`` reduction at the requested
    native scale. Batching only bounds the size of each server response / ``getInfo`` call.
    """
    cells = [str(hex_id) for hex_id in hex_ids]
    if batch_size <= 0:
        raise ValueError(f"{context}: batch_size must be positive; got {batch_size}")
    if max_retries < 0:
        raise ValueError(f"{context}: max_retries must be non-negative; got {max_retries}")
    if not cells:
        raise ValueError(f"{context}: no H3 cells to reduce")

    # Fail before any EE request if the caller supplied ambiguous coverage.
    requested = pd.DataFrame({"h3_index": cells})
    _validate_h3_coverage(requested, cells, context=f"{context} request")

    batch_count = (len(cells) + batch_size - 1) // batch_size
    logger.info(
        "%s: reducing %d H3 cells in %d batch(es) "
        "(batch_size=%d, scale=%d m, tileScale=%d)",
        context,
        len(cells),
        batch_count,
        batch_size,
        scale,
        tile_scale,
    )
    frames: list[pd.DataFrame] = []
    for batch_number, start in enumerate(range(0, len(cells), batch_size), start=1):
        batch_ids = cells[start : start + batch_size]
        reduced = image.reduceRegions(
            collection=_h3_feature_collection(batch_ids),
            reducer=reducer,
            scale=scale,
            tileScale=tile_scale,
        )
        batch_frame: pd.DataFrame | None = None
        for attempt in range(max_retries + 1):
            try:
                batch_frame = _fc_to_dataframe(reduced)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"{context}: batch {batch_number}/{batch_count} getInfo failed "
                        f"after {attempt + 1} attempt(s)"
                    ) from exc
                delay_seconds = 2**attempt
                logger.warning(
                    "%s: batch %d/%d getInfo attempt %d failed (%s); "
                    "retrying in %d s",
                    context,
                    batch_number,
                    batch_count,
                    attempt + 1,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
        if batch_frame is None:  # pragma: no cover - defensive; loop returns or raises
            raise RuntimeError(f"{context}: batch {batch_number}/{batch_count} returned no frame")
        batch_frame = _validate_h3_coverage(
            batch_frame,
            batch_ids,
            context=f"{context} batch {batch_number}/{batch_count}",
        )
        frames.append(batch_frame)
        logger.info(
            "%s: completed batch %d/%d (%d/%d cells)",
            context,
            batch_number,
            batch_count,
            min(start + len(batch_ids), len(cells)),
            len(cells),
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return _validate_h3_coverage(combined, cells, context=f"{context} combined")


def _combined_ecs_reducer() -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.percentile([95]), sharedInputs=True)
    )


def _finalize_reduce_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Normalize Earth Engine reduceRegions column names for a single-band image."""
    out = df.copy()
    if "h3_index" in out.columns:
        out["h3_index"] = out["h3_index"].astype(str)

    rename: dict[str, str] = {}
    for col in out.columns:
        if col == "h3_index":
            continue
        if col == "mean":
            rename[col] = f"{prefix}_mean"
        elif col == "max":
            rename[col] = f"{prefix}_max"
        elif col == "stdDev":
            rename[col] = f"{prefix}_stddev"
    out = out.rename(columns=rename)

    if f"{prefix}_p95" not in out.columns:
        for col in list(out.columns):
            if col == "h3_index":
                continue
            cl = col.lower()
            if "p95" in cl or "percentile" in cl:
                out = out.rename(columns={col: f"{prefix}_p95"})
                break
    return out


def _sample_embedding_table(
    image: ee.Image,
    region: ee.Geometry,
    n: int,
    seed: int,
) -> pd.DataFrame:
    sample_fc = (
        image.select(EMB_BANDS)
        .sample(region=region, scale=10, numPixels=n, geometries=False, seed=seed, tileScale=2)
        .limit(n)
    )
    return _fc_to_dataframe(sample_fc)


def _vector_magnitude_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"count": 0, "mean_l2": float("nan"), "nan_vectors": 0}
    mat = df[EMB_BANDS].to_numpy(dtype=float)
    norms = np.linalg.norm(mat, axis=1)
    return {
        "count": int(len(norms)),
        "mean_l2": float(np.nanmean(norms)),
        "nan_vectors": int(np.isnan(norms).sum()),
    }


def task_a_safety_probe(cfg: dict[str, Any], aoi: ee.Geometry) -> dict[str, Any]:
    """100 random 10 m pixels in event core + 100 in a geodesic control disk ~50 km away."""
    lon, lat = cfg["center_lon"], cfg["center_lat"]
    clon, clat = _offset_point_m(lon, lat, cfg["control_bearing_deg"], cfg["control_offset_km"] * 1000.0)
    r_m = float(cfg["probe_center_radius_m"])
    center_region = _ee_point(lon, lat).buffer(r_m).intersection(aoi)
    control_geom = _ee_point(clon, clat).buffer(r_m)
    # Mosaics are clipped to the region passed in; the control disk lies outside the processing AOI,
    # so we must clip embeddings to AOI ∪ control or control sampling returns zero pixels.
    probe_clip = aoi.union(control_geom)

    pre_img = _annual_embedding_mosaic(int(cfg["pre_year"]), probe_clip)
    post_img = _annual_embedding_mosaic(int(cfg["post_year"]), probe_clip)

    n = 100
    pre_center = _sample_embedding_table(pre_img, center_region, n=n, seed=1)
    post_center = _sample_embedding_table(post_img, center_region, n=n, seed=1)
    pre_control = _sample_embedding_table(pre_img, control_geom, n=n, seed=2)
    post_control = _sample_embedding_table(post_img, control_geom, n=n, seed=2)

    stats = {
        "center_pre": _vector_magnitude_stats(pre_center),
        "center_post": _vector_magnitude_stats(post_center),
        "control_pre": _vector_magnitude_stats(pre_control),
        "control_post": _vector_magnitude_stats(post_control),
        "control_center_lonlat": (clon, clat),
    }

    print("\n=== Task A — Safety probe (unit-sphere embeddings: |v| ≈ 1) ===", flush=True)
    print(
        f"Event core disk radius: {r_m:.0f} m | Control disk center offset: {cfg['control_offset_km']:.1f} km",
        flush=True,
    )
    for name, s in [
        ("Center / pre-year", stats["center_pre"]),
        ("Center / post-year", stats["center_post"]),
        ("Control / pre-year", stats["control_pre"]),
        ("Control / post-year", stats["control_post"]),
    ]:
        print(
            f"{name}: n={s['count']}, mean L2 magnitude={s['mean_l2']:.6f}, "
            f"nan_vectors={s['nan_vectors']}",
            flush=True,
        )
    print("=== End Task A ===\n", flush=True)
    return stats


def task_b_h3_ecs(
    cfg: dict[str, Any],
    aoi_ee: ee.Geometry,
    aoi_poly: sg.Polygon | sg.MultiPolygon,
    *,
    ee_batch_size: int = DEFAULT_EE_H3_BATCH_SIZE,
) -> pd.DataFrame:
    pre = _annual_embedding_mosaic(int(cfg["pre_year"]), aoi_ee)
    post = _annual_embedding_mosaic(int(cfg["post_year"]), aoi_ee)
    ecs = _ecs_image(pre, post)

    res = int(cfg["h3_resolution"])
    cells = _polygon_to_h3_cells(aoi_poly, res)
    if len(cells) > int(cfg["max_h3_cells"]):
        raise ValueError(
            f"H3 cell count {len(cells)} exceeds max_h3_cells={cfg['max_h3_cells']}. "
            "Lower h3_resolution, shrink processing_buffer_km, or raise max_h3_cells."
        )

    reducer = _combined_ecs_reducer()
    df = _reduce_regions_h3_batched(
        ecs,
        cells,
        reducer=reducer,
        scale=10,
        tile_scale=4,
        batch_size=ee_batch_size,
        context="AlphaEarth ECS",
    )
    df = _finalize_reduce_columns(df, "ecs")
    expected = {"h3_index", "ecs_mean", "ecs_max", "ecs_stddev", "ecs_p95"}
    missing = expected - set(df.columns)
    if missing:
        logger.warning("Unexpected ECS columns; missing %s. Available: %s", missing, list(df.columns))
    return df


def task_b_h3_benchmark(
    cfg: dict[str, Any],
    aoi_ee: ee.Geometry,
    hex_ids: list[str],
    *,
    ee_batch_size: int = DEFAULT_EE_H3_BATCH_SIZE,
) -> pd.DataFrame:
    spec: BenchmarkSpec = cfg["benchmark"]
    bench = _benchmark_image(spec, aoi_ee)
    reducer = _combined_ecs_reducer()  # same stats: mean/max/stdDev/p95 of benchmark raster
    scale_m = (
        int(spec.nominal_scale_m)
        if spec.nominal_scale_m is not None
        else 500
        if spec.type == "modis_mcd64a1"
        else 30
    )
    df = _reduce_regions_h3_batched(
        bench,
        hex_ids,
        reducer=reducer,
        scale=scale_m,
        tile_scale=4,
        batch_size=ee_batch_size,
        context=f"{spec.type} benchmark",
    )
    df = _finalize_reduce_columns(df, "benchmark")
    return df


def task_c_inner_join(alpha_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    alpha_df = alpha_df.copy()
    bench_df = bench_df.copy()
    alpha_df["h3_index"] = alpha_df["h3_index"].astype(str)
    bench_df["h3_index"] = bench_df["h3_index"].astype(str)

    key_cols = ["h3_index"]
    merged = alpha_df.merge(bench_df, on=key_cols, how="inner", validate="one_to_one")
    # Valid official metric: finite benchmark_mean (reduceRegions nulls -> NaN in client)
    merged = merged[np.isfinite(merged["benchmark_mean"])].copy()
    merged = merged[np.isfinite(merged["ecs_mean"])].copy()
    return merged


def task_d_report_and_plots(
    cfg: dict[str, Any],
    alpha_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_alpha = int(alpha_df["h3_index"].nunique())
    if "benchmark_mean" in bench_df.columns:
        n_bench_valid = int(bench_df.loc[np.isfinite(bench_df["benchmark_mean"]), "h3_index"].nunique())
    else:
        n_bench_valid = 0
    n_join = int(joined_df["h3_index"].nunique())
    loss_pct = float((1.0 - (n_join / n_alpha)) * 100.0) if n_alpha else float("nan")

    if not joined_df.empty and "benchmark_mean" in joined_df.columns:
        bm = joined_df["benchmark_mean"].astype(float)
        nu = int(bm.nunique(dropna=True))
        std_b = float(bm.std(ddof=1)) if len(bm) > 1 else 0.0
        if cfg.get("disaster_type") == "forest_loss" and (nu <= 3 or (np.isfinite(std_b) and std_b < 1e-10)):
            logger.warning(
                "Forest-loss benchmark is nearly uniform (benchmark_mean nunique=%s, std=%s). "
                "Hansen may show no loss in this AOI/year at 30 m — move center or widen buffer.",
                nu,
                std_b,
            )

    report = {
        "event": cfg["label"],
        "disaster_type": str(cfg.get("disaster_type", "unknown")),
        "pre_year": int(cfg["pre_year"]),
        "post_year": int(cfg["post_year"]),
        "benchmark": {
            "type": cfg["benchmark"].type,
            "metric_year": int(cfg["benchmark"].metric_year),
            "reference_pre_year": cfg["benchmark"].reference_pre_year,
            "nominal_scale_m": cfg["benchmark"].nominal_scale_m,
        },
        "benchmark_temporal_relation": cfg.get(
            "benchmark_temporal_relation",
            (
                "reference_year_matches_post"
                if cfg.get("benchmark_year_matches_post", True)
                else "legacy_calendar_mismatch"
            ),
        ),
        "h3_resolution": int(cfg["h3_resolution"]),
        "total_h3_cells_alphaearth": n_alpha,
        "total_h3_cells_benchmark_valid": n_bench_valid,
        "total_h3_cells_inner_join": n_join,
        "pct_data_lost_vs_alpha": loss_pct,
    }
    if cfg["benchmark"].type == "usdm_season_delta":
        report["benchmark"]["weekly_image_count_pre"] = (
            _usdm_season_image_count(
                int(cfg["benchmark"].reference_pre_year)
            )
        )
        report["benchmark"]["weekly_image_count_post"] = (
            _usdm_season_image_count(int(cfg["benchmark"].metric_year))
        )
    (out_dir / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "integrity_report.json")

    def _h3_to_shapely(h: str) -> sg.Polygon:
        boundary = h3.cell_to_boundary(h)
        ring = [(float(lon), float(lat)) for lat, lon in boundary]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        return sg.Polygon(ring)

    joined_df = joined_df.copy()
    joined_df["geometry"] = joined_df["h3_index"].map(_h3_to_shapely)
    gdf = gpd.GeoDataFrame(joined_df, geometry="geometry", crs="EPSG:4326")
    gdf_hm, vmin_hm, vmax_hm = _join_heatmap_plot_columns(gdf)

    heatmap_path = out_dir / "heatmap_join_side_by_side.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
        gdf_hm.plot(
            column="_ecs_plot_z",
            ax=axes[0],
            legend=True,
            cmap=JOIN_HEATMAP_CMAP,
            vmin=vmin_hm,
            vmax=vmax_hm,
            linewidth=0.2,
            edgecolor="k",
        )
        axes[0].set_title(f"AlphaEarth ECS (IQR-z linear [0–1]; {JOIN_HEATMAP_CMAP})")
        gdf_hm.plot(
            column="_bench_plot_z",
            ax=axes[1],
            legend=True,
            cmap=JOIN_HEATMAP_CMAP,
            vmin=vmin_hm,
            vmax=vmax_hm,
            linewidth=0.2,
            edgecolor="k",
        )
        axes[1].set_title(f"Benchmark (IQR-z linear [0–1]; {JOIN_HEATMAP_CMAP})")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(
            cfg["label"]
            + " — join alignment (each panel: IQR-z min–max to [0,1]; same cmap & limits 0–1)"
        )
        fig.tight_layout()
        fig.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", heatmap_path)
    except Exception:
        logger.exception("Failed to write %s", heatmap_path)
        plt.close("all")

    basemap_path = out_dir / "heatmap_join_side_by_side_basemap.png"
    try:
        _plot_join_heatmaps_basemap(gdf_hm, cfg["label"], basemap_path, vmin_hm, vmax_hm, JOIN_HEATMAP_CMAP)
    except Exception:
        logger.exception("Failed to write %s", basemap_path)
        plt.close("all")

    return report


def _plot_join_heatmaps_basemap(
    gdf_wgs84: gpd.GeoDataFrame,
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    """Side-by-side ECS / benchmark choropleth over web tiles; expects linear-[0,1] columns ``_ecs_plot_z`` / ``_bench_plot_z``."""
    try:
        import contextily as cx
    except ImportError:
        logger.warning("contextily not installed; skipping basemap plot. pip install contextily")
        return

    gdf_m = gdf_wgs84.to_crs(epsg=3857)
    pad = max(gdf_m.total_bounds[2] - gdf_m.total_bounds[0], gdf_m.total_bounds[3] - gdf_m.total_bounds[1]) * 0.06

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=140)
    panels: list[tuple[Any, str, str]] = [
        (axes[0], "_ecs_plot_z", f"AlphaEarth ECS (IQR-z linear [0–1]; {cmap})"),
        (axes[1], "_bench_plot_z", f"Benchmark (IQR-z linear [0–1]; {cmap})"),
    ]
    for ax, column, panel_title in panels:
        gdf_m.plot(
            column=column,
            ax=ax,
            legend=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.62,
            linewidth=0.35,
            edgecolor="white",
            legend_kwds={"shrink": 0.55},
        )
        try:
            cx.add_basemap(ax, crs=gdf_m.crs, source=cx.providers.CartoDB.Positron)
        except Exception as exc:
            logger.warning("Could not fetch basemap tiles (%s). Saving choropleth without tiles.", exc)
        ax.set_title(panel_title)
        ax.set_axis_off()
        b = gdf_m.total_bounds
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)

    fig.suptitle(title + f" — join basemap (IQR-z linear [0–1]; cmap limits [{vmin:.2f}, {vmax:.2f}])")
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Wrote %s", out_path)


def run_pipeline(
    event_key: str,
    output_root: Path,
    *,
    reuse_alphaearth: bool = False,
    ee_batch_size: int = DEFAULT_EE_H3_BATCH_SIZE,
) -> None:
    configs = _event_configurations()
    if event_key not in configs:
        raise KeyError(f"Unknown event {event_key!r}. Choose one of: {sorted(configs)}")
    cfg = configs[event_key]
    spec: BenchmarkSpec = cfg["benchmark"]
    temporal_relation = cfg.get(
        "benchmark_temporal_relation",
        (
            "reference_year_matches_post"
            if cfg.get("benchmark_year_matches_post", True)
            else "legacy_calendar_mismatch"
        ),
    )
    if temporal_relation == "event_between_annual_mosaics":
        pre_y, event_y, post_y = (
            int(cfg["pre_year"]),
            int(spec.metric_year),
            int(cfg["post_year"]),
        )
        if not pre_y < event_y < post_y:
            raise ValueError(
                "event_between_annual_mosaics requires "
                f"pre_year < metric_year < post_year; got "
                f"{pre_y}, {event_y}, {post_y}"
            )
        logger.info(
            "Using full-year mosaics that bracket the event: %s < %s < %s.",
            pre_y,
            event_y,
            post_y,
        )
    elif cfg.get("benchmark_year_matches_post", True):
        if int(cfg["post_year"]) != int(spec.metric_year):
            raise ValueError(
                f"Strict temporal mismatch: post_year={cfg['post_year']} vs benchmark.metric_year={spec.metric_year}"
            )
    else:
        my = int(spec.metric_year)
        pre_y, post_y = int(cfg["pre_year"]), int(cfg["post_year"])
        lo, hi = min(pre_y, post_y) - 2, max(pre_y, post_y) + 1
        if not (lo <= my <= hi):
            raise ValueError(
                f"benchmark.metric_year={my} outside plausible window [{lo}, {hi}] "
                f"for embedding years {pre_y}/{post_y}"
            )
        logger.warning(
            "Using benchmark calendar year %s with AlphaEarth pre/post %s/%s (legacy fire; see event label).",
            my,
            pre_y,
            post_y,
        )

    aoi_shapely = _configured_aoi(cfg)

    # Any ee.Geometry / Image / Collection construction requires ee.Initialize() first.
    project_id = os.environ.get("EE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise SystemExit(
            "Earth Engine needs EE_PROJECT_ID or GOOGLE_CLOUD_PROJECT set to your GCP project id "
            "(see https://developers.google.com/earth-engine/guides/access )."
        )
    ee.Initialize(project=project_id)
    aoi_ee = ee.Geometry(aoi_shapely.__geo_interface__)
    logger.info("Running %s", cfg["label"])

    out_dir = output_root / event_key
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = out_dir / "alphaearth_h3_ecs.parquet"
    probe_path = out_dir / "task_a_probe.json"
    if reuse_alphaearth:
        if not alpha_path.is_file():
            raise FileNotFoundError(
                f"--reuse-alphaearth requires existing {alpha_path}"
            )
        alpha_df = pd.read_parquet(alpha_path)
        alpha_df["h3_index"] = alpha_df["h3_index"].astype(str)
        logger.info("Reusing %s (%d rows)", alpha_path, len(alpha_df))
    else:
        probe_stats = task_a_safety_probe(cfg, aoi_ee)
        alpha_df = task_b_h3_ecs(
            cfg,
            aoi_ee,
            aoi_shapely,
            ee_batch_size=ee_batch_size,
        )
        alpha_df.to_parquet(alpha_path, index=False)
        probe_path.write_text(
            json.dumps(probe_stats, indent=2),
            encoding="utf-8",
        )
    hex_ids = sorted(alpha_df["h3_index"].astype(str).unique().tolist())
    bench_df = task_b_h3_benchmark(
        cfg,
        aoi_ee,
        hex_ids,
        ee_batch_size=ee_batch_size,
    )
    joined = task_c_inner_join(alpha_df, bench_df)

    bench_df.to_parquet(out_dir / "benchmark_h3.parquet", index=False)
    joined.to_parquet(out_dir / "joined_inner.parquet", index=False)

    report = task_d_report_and_plots(cfg, alpha_df, bench_df, joined, out_dir)
    logger.info("Integrity summary: %s", json.dumps(report, indent=2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event",
        default="camp_fire_2018",
        choices=sorted(_event_configurations().keys()),
        help="Event key from the configuration dictionary.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for parquet/json/png artifacts.",
    )
    p.add_argument(
        "--all-events",
        action="store_true",
        help="Run every event in the configuration dictionary and write integrity_summary_all_events.json.",
    )
    p.add_argument(
        "--good-fires",
        action="store_true",
        help="Run each key in fire_analysis.FIRES_GOOD and write integrity_summary_good_fires.json.",
    )
    p.add_argument(
        "--reuse-alphaearth",
        action="store_true",
        help=(
            "Reuse each event's existing alphaearth_h3_ecs.parquet and rerun "
            "only the benchmark, join, integrity report, and heatmaps."
        ),
    )
    p.add_argument(
        "--ee-batch-size",
        type=int,
        default=DEFAULT_EE_H3_BATCH_SIZE,
        metavar="CELLS",
        help=(
            "H3 cells per Earth Engine reduceRegions/getInfo request "
            f"(default: {DEFAULT_EE_H3_BATCH_SIZE})."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.ee_batch_size <= 0:
        raise SystemExit("--ee-batch-size must be a positive integer.")
    out = args.output_dir.resolve()
    if args.all_events and args.good_fires:
        raise SystemExit("Use only one of --all-events or --good-fires.")
    if args.all_events:
        keys = sorted(_event_configurations().keys())
        for key in keys:
            run_pipeline(
                key,
                out,
                reuse_alphaearth=args.reuse_alphaearth,
                ee_batch_size=args.ee_batch_size,
            )
        combined: dict[str, Any] = {}
        for key in keys:
            report_path = out / key / "integrity_report.json"
            if report_path.is_file():
                combined[key] = json.loads(report_path.read_text(encoding="utf-8"))
        (out / "integrity_summary_all_events.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out / "integrity_summary_all_events.json")
        return 0
    if args.good_fires:
        from fire_analysis import FIRES_GOOD

        cfgs = _event_configurations()
        keys = [k for k in FIRES_GOOD if k in cfgs]
        missing = [k for k in FIRES_GOOD if k not in cfgs]
        for k in missing:
            logger.warning("FIRES_GOOD key %s not in event configurations; skipped", k)
        for key in keys:
            run_pipeline(
                key,
                out,
                reuse_alphaearth=args.reuse_alphaearth,
                ee_batch_size=args.ee_batch_size,
            )
        combined: dict[str, Any] = {}
        for key in keys:
            report_path = out / key / "integrity_report.json"
            if report_path.is_file():
                combined[key] = json.loads(report_path.read_text(encoding="utf-8"))
        summary_path = out / "integrity_summary_good_fires.json"
        summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        logger.info("Wrote %s", summary_path)
        return 0
    run_pipeline(
        args.event,
        out,
        reuse_alphaearth=args.reuse_alphaearth,
        ee_batch_size=args.ee_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
