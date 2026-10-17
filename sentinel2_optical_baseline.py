#!/usr/bin/env python3
"""Conventional annual Sentinel-2 change baselines for the REO-2 event study.

This study is intentionally isolated from the main disaster pipeline. It reads
each event's frozen ``joined_inner.parquet`` as the target H3/reference
population and writes only beneath ``outputs/optical_change_baseline``.

Predefined optical measures
---------------------------
For each configured pre/post calendar-year pair:

* NDVI = (B8 - B4) / (B8 + B4)
* NBR = (B8 - B12) / (B8 + B12)
* absolute dNDVI = |NDVI_post - NDVI_pre|
* absolute dNBR = |NBR_pre - NBR_post|

Each annual index is calculated from the median cloud-masked surface
reflectance composite for that full calendar year. The collection, scene-level
cloud filter, and median-mosaic approach follow the repository's existing GIS
reference code. Unlike that visualization code, this analytical baseline adds
an explicit Sentinel-2 SCL pixel mask and never applies RGB stretches, ranks,
quantiles, or outcome-dependent metric selection.

Eligibility is fixed before any Sentinel-2 values are requested. All three
valid volcanic-reference events are candidates. A fire is eligible only when
its MODIS reference year equals its configured post year, so the full pre year
does not overlap the reference event year, and when its reference population
has finite, nonconstant positive support. No FIRES_GOOD or visual/correlation
labels are imported or consulted.

Correlations are descriptive Pearson/Spearman coefficients with no cell-level
p-values. Within an event, ECS and both optical baselines are always evaluated
on the identical complete-case H3 set. Aggregate summaries treat events,
rather than cells, as the descriptive units and do not pool cells.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable, Sequence
import warnings

import ee
import h3
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

ANALYSIS_VERSION = "1.0.0"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ("B4", "B8", "B12")
SCL_EXCLUDED_CLASSES = (0, 1, 3, 7, 8, 9, 10, 11)
PREDICTORS = ("ecs_mean", "abs_dnbr", "abs_dndvi")
PREDICTOR_LABELS = {
    "ecs_mean": "AlphaEarth ECS",
    "abs_dnbr": "Sentinel-2 |dNBR|",
    "abs_dndvi": "Sentinel-2 |dNDVI|",
}


@dataclass(frozen=True)
class EventSpec:
    """Frozen candidate metadata; deliberately independent of analysis results."""

    event_key: str
    label: str
    family: str
    pre_year: int
    post_year: int
    reference_type: str
    reference_year: int


# This is the full configured fire population plus all three valid volcanic
# reference events at study definition time. It intentionally does not import
# FIRES_GOOD/FIRES_POOR or discover candidates from result summaries.
EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec(
        "bootleg_fire_2021",
        "Bootleg Fire (Oregon, 2021)",
        "fire",
        2020,
        2021,
        "modis_mcd64a1",
        2021,
    ),
    EventSpec(
        "caldor_fire_2021",
        "Caldor Fire (California, 2021)",
        "fire",
        2020,
        2021,
        "modis_mcd64a1",
        2021,
    ),
    EventSpec(
        "camp_fire_2018",
        "Camp Fire (California, 2018)",
        "fire",
        2017,
        2018,
        "modis_mcd64a1",
        2018,
    ),
    EventSpec(
        "dixie_fire_2021",
        "Dixie Fire (California, 2021)",
        "fire",
        2020,
        2021,
        "modis_mcd64a1",
        2021,
    ),
    EventSpec(
        "fort_mcmurray_fire_2016",
        "Fort McMurray wildfire (Alberta, 2016)",
        "fire",
        2017,
        2018,
        "modis_mcd64a1",
        2016,
    ),
    EventSpec(
        "marshall_fire_2021",
        "Marshall Fire (Colorado, 2021)",
        "fire",
        2021,
        2022,
        "modis_mcd64a1",
        2021,
    ),
    EventSpec(
        "mosquito_fire_2022",
        "Mosquito Fire (California, 2022)",
        "fire",
        2021,
        2022,
        "modis_mcd64a1",
        2022,
    ),
    EventSpec(
        "quebec_fires_2023",
        "Quebec wildfires (2023, Chibougamau / Lac Saint-Jean corridor)",
        "fire",
        2022,
        2023,
        "modis_mcd64a1",
        2023,
    ),
    EventSpec(
        "thomas_fire_2017",
        "Thomas Fire (Southern California, 2017)",
        "fire",
        2017,
        2018,
        "modis_mcd64a1",
        2017,
    ),
    EventSpec(
        "tubbs_fire_2017",
        "Tubbs Fire (Sonoma / Napa, California, 2017)",
        "fire",
        2017,
        2018,
        "modis_mcd64a1",
        2017,
    ),
    EventSpec(
        "kilauea_lava_2018",
        "Kilauea lower East Rift Zone lava emplacement (USGS, 2018)",
        "volcanic_eruption",
        2017,
        2019,
        "geojson_binary",
        2018,
    ),
    EventSpec(
        "la_palma_lava_2021",
        "La Palma Cumbre Vieja lava emplacement (Copernicus EMS, 2021)",
        "volcanic_eruption",
        2020,
        2022,
        "geojson_binary",
        2021,
    ),
    EventSpec(
        "mauna_loa_lava_2022",
        "Mauna Loa summit and Northeast Rift Zone lava emplacement (USGS, 2022)",
        "volcanic_eruption",
        2021,
        2023,
        "geojson_binary",
        2022,
    ),
)


@dataclass(frozen=True)
class ReferenceSummary:
    n_cells: int
    n_h3_unique: int
    n_benchmark_finite: int
    n_ecs_finite: int
    benchmark_nunique: int
    benchmark_min: float | None
    benchmark_max: float | None
    source_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EligibilityDecision:
    event_key: str
    eligible: bool
    reasons: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalized_difference(numerator_positive: Any, numerator_negative: Any) -> Any:
    """Return (positive - negative)/(positive + negative), NaN at zero denominator."""

    positive = np.asarray(numerator_positive, dtype=float)
    negative = np.asarray(numerator_negative, dtype=float)
    denominator = positive + negative
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            positive - negative,
            denominator,
            out=np.full(np.broadcast_shapes(positive.shape, negative.shape), np.nan),
            where=np.abs(denominator) > 1e-12,
        )
    return float(result) if result.ndim == 0 else result


def absolute_index_change(pre_index: Any, post_index: Any) -> Any:
    """Absolute post-minus-pre index change; sign convention is immaterial after abs."""

    result = np.abs(np.asarray(post_index, dtype=float) - np.asarray(pre_index, dtype=float))
    return float(result) if result.ndim == 0 else result


def assess_event_eligibility(
    spec: EventSpec,
    summary: ReferenceSummary | None,
) -> EligibilityDecision:
    """Apply only frozen temporal and source/reference-validity rules."""

    reasons: list[str] = []
    if summary is None:
        reasons.append("missing_joined_reference_population")
    else:
        reasons.extend(summary.source_errors)
        if summary.n_cells == 0:
            reasons.append("empty_reference_population")
        if summary.n_h3_unique != summary.n_cells:
            reasons.append("duplicate_h3_ids")
        if summary.n_benchmark_finite != summary.n_cells:
            reasons.append("nonfinite_reference_values")
        if summary.n_ecs_finite != summary.n_cells:
            reasons.append("nonfinite_ecs_values")
        if summary.benchmark_nunique < 2:
            reasons.append("reference_has_no_spatial_variation")
        if summary.benchmark_max is None or summary.benchmark_max <= 0:
            reasons.append("reference_has_no_positive_support")

    if spec.family == "fire":
        if spec.reference_type != "modis_mcd64a1":
            reasons.append("fire_reference_is_not_modis_mcd64a1")
        if spec.post_year != spec.pre_year + 1:
            reasons.append("fire_years_are_not_consecutive")
        if spec.reference_year != spec.post_year:
            reasons.append("fire_reference_year_is_not_post_year")
    elif spec.family == "volcanic_eruption":
        if spec.reference_type != "geojson_binary":
            reasons.append("volcanic_reference_is_not_event_geometry")
        if not spec.pre_year < spec.reference_year < spec.post_year:
            reasons.append("volcanic_event_not_bracketed_by_annual_mosaics")
    else:
        reasons.append("unsupported_event_family")

    # Stable de-duplication makes output deterministic even if multiple checks
    # produce the same reason in future revisions.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return EligibilityDecision(
        event_key=spec.event_key,
        eligible=not unique_reasons,
        reasons=unique_reasons,
    )


def _validate_integrity_metadata(
    spec: EventSpec,
    integrity_path: Path,
) -> tuple[str, ...]:
    if not integrity_path.is_file():
        return ("missing_integrity_report",)
    try:
        metadata = json.loads(integrity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("unreadable_integrity_report",)

    errors: list[str] = []
    if int(metadata.get("pre_year", -1)) != spec.pre_year:
        errors.append("integrity_pre_year_mismatch")
    if int(metadata.get("post_year", -1)) != spec.post_year:
        errors.append("integrity_post_year_mismatch")
    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("integrity_benchmark_missing")
    else:
        if benchmark.get("type") != spec.reference_type:
            errors.append("integrity_reference_type_mismatch")
        if int(benchmark.get("metric_year", -1)) != spec.reference_year:
            errors.append("integrity_reference_year_mismatch")
    return tuple(errors)


def load_reference_population(
    input_root: Path,
    spec: EventSpec,
) -> tuple[pd.DataFrame | None, ReferenceSummary | None]:
    """Load but never mutate the event's existing joined source Parquet."""

    event_dir = input_root / spec.event_key
    joined_path = event_dir / "joined_inner.parquet"
    if not joined_path.is_file():
        return None, None

    source_errors = list(
        _validate_integrity_metadata(spec, event_dir / "integrity_report.json")
    )
    try:
        frame = pd.read_parquet(joined_path)
    except Exception:
        LOGGER.exception("Could not read %s", joined_path)
        return None, ReferenceSummary(
            n_cells=0,
            n_h3_unique=0,
            n_benchmark_finite=0,
            n_ecs_finite=0,
            benchmark_nunique=0,
            benchmark_min=None,
            benchmark_max=None,
            source_errors=tuple(source_errors + ["unreadable_joined_reference_population"]),
        )

    required = {"h3_index", "ecs_mean", "benchmark_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        source_errors.append("missing_required_columns:" + ",".join(missing))
        summary = ReferenceSummary(
            n_cells=int(len(frame)),
            n_h3_unique=0,
            n_benchmark_finite=0,
            n_ecs_finite=0,
            benchmark_nunique=0,
            benchmark_min=None,
            benchmark_max=None,
            source_errors=tuple(source_errors),
        )
        return frame, summary

    population = frame[["h3_index", "ecs_mean", "benchmark_mean"]].copy()
    population["h3_index"] = population["h3_index"].astype(str)
    population["ecs_mean"] = pd.to_numeric(population["ecs_mean"], errors="coerce")
    population["benchmark_mean"] = pd.to_numeric(
        population["benchmark_mean"], errors="coerce"
    )
    benchmark = population["benchmark_mean"]
    summary = ReferenceSummary(
        n_cells=int(len(population)),
        n_h3_unique=int(population["h3_index"].nunique(dropna=False)),
        n_benchmark_finite=int(np.isfinite(benchmark.to_numpy(dtype=float)).sum()),
        n_ecs_finite=int(
            np.isfinite(population["ecs_mean"].to_numpy(dtype=float)).sum()
        ),
        benchmark_nunique=int(benchmark.nunique(dropna=True)),
        benchmark_min=_optional_float(benchmark.min()),
        benchmark_max=_optional_float(benchmark.max()),
        source_errors=tuple(source_errors),
    )
    return population, summary


def build_eligibility_table(
    input_root: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate all frozen candidates in event-key order before any EE request."""

    rows: list[dict[str, Any]] = []
    populations: dict[str, pd.DataFrame] = {}
    for spec in sorted(EVENT_SPECS, key=lambda item: item.event_key):
        population, summary = load_reference_population(input_root, spec)
        decision = assess_event_eligibility(spec, summary)
        if population is not None:
            populations[spec.event_key] = population
        row: dict[str, Any] = {
            **asdict(spec),
            "eligible": decision.eligible,
            "exclusion_reasons": ";".join(decision.reasons),
        }
        if summary is not None:
            row.update(asdict(summary))
            row["source_errors"] = ";".join(summary.source_errors)
        rows.append(row)
    return pd.DataFrame(rows), populations


def _validate_exact_h3_population(
    frame: pd.DataFrame,
    expected_h3: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    expected = [str(value) for value in expected_h3]
    expected_duplicates = sorted(
        key for key, count in Counter(expected).items() if count > 1
    )
    if expected_duplicates:
        raise ValueError(
            f"{context}: duplicate expected H3 IDs: {expected_duplicates[:10]}"
        )
    if "h3_index" not in frame.columns:
        raise ValueError(f"{context}: missing h3_index column")
    if frame["h3_index"].isna().any():
        raise ValueError(f"{context}: null H3 ID")

    result = frame.copy()
    result["h3_index"] = result["h3_index"].astype(str)
    actual = result["h3_index"].tolist()
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicates or missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            f"{context}: H3 population mismatch "
            f"(expected={len(expected)}, actual={len(actual)}, "
            f"duplicates={duplicates[:10]}, missing={missing[:10]}, "
            f"unexpected={unexpected[:10]})"
        )

    order = {h3_index: index for index, h3_index in enumerate(expected)}
    result["_population_order"] = result["h3_index"].map(order)
    return (
        result.sort_values("_population_order", kind="stable")
        .drop(columns="_population_order")
        .reset_index(drop=True)
    )


def join_same_population(
    reference: pd.DataFrame,
    optical: pd.DataFrame,
    *,
    min_cell_valid_fraction: float,
) -> pd.DataFrame:
    """Strict one-to-one join and one common complete-case mask for all predictors."""

    reference_ids = reference["h3_index"].astype(str).tolist()
    reference_checked = _validate_exact_h3_population(
        reference,
        reference_ids,
        context="reference population",
    )
    optical_checked = _validate_exact_h3_population(
        optical,
        reference_ids,
        context="Sentinel-2 population",
    )
    merged = reference_checked.merge(
        optical_checked,
        on="h3_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    for column in (
        "ecs_mean",
        "benchmark_mean",
        "abs_dnbr",
        "abs_dndvi",
        "s2_valid_fraction",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    finite = np.ones(len(merged), dtype=bool)
    for column in (
        "ecs_mean",
        "benchmark_mean",
        "abs_dnbr",
        "abs_dndvi",
        "s2_valid_fraction",
    ):
        finite &= np.isfinite(merged[column].to_numpy(dtype=float))
    merged["common_valid"] = finite & (
        merged["s2_valid_fraction"].to_numpy(dtype=float)
        >= float(min_cell_valid_fraction)
    )
    return merged


def _h3_to_ee_polygon(h3_index: str) -> ee.Geometry:
    boundary = h3.cell_to_boundary(h3_index)
    ring = [[float(lon), float(lat)] for lat, lon in boundary]
    ring.append(ring[0])
    return ee.Geometry.Polygon([ring], proj="EPSG:4326", geodesic=False)


def _h3_feature_collection(h3_ids: Iterable[str]) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(_h3_to_ee_polygon(str(h3_index)), {"h3_index": str(h3_index)})
            for h3_index in h3_ids
        ]
    )


def _h3_bounds(h3_ids: Sequence[str]) -> ee.Geometry:
    longitudes: list[float] = []
    latitudes: list[float] = []
    for h3_index in h3_ids:
        for latitude, longitude in h3.cell_to_boundary(str(h3_index)):
            longitudes.append(float(longitude))
            latitudes.append(float(latitude))
    if not longitudes:
        raise ValueError("Cannot construct an Earth Engine region from no H3 cells")
    pad_degrees = 0.002
    return ee.Geometry.Rectangle(
        [
            min(longitudes) - pad_degrees,
            min(latitudes) - pad_degrees,
            max(longitudes) + pad_degrees,
            max(latitudes) + pad_degrees,
        ],
        proj="EPSG:4326",
        geodesic=False,
    )


def _mask_s2_sr(image: ee.Image) -> ee.Image:
    """Mask SCL no-data/defect/shadow/cloud/cirrus/snow classes."""

    scl = image.select("SCL")
    clear = ee.Image(1)
    for class_value in SCL_EXCLUDED_CLASSES:
        clear = clear.And(scl.neq(class_value))
    return (
        image.select(list(S2_BANDS))
        .updateMask(clear)
        .multiply(0.0001)
        .copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])
    )


def _s2_annual_collection(
    region: ee.Geometry,
    year: int,
    *,
    max_scene_cloud_pct: float,
) -> ee.ImageCollection:
    start = ee.Date.fromYMD(int(year), 1, 1)
    end = start.advance(1, "year")
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud_pct))
        .map(_mask_s2_sr)
    )


def _safe_ee_normalized_difference(
    positive: ee.Image,
    negative: ee.Image,
    name: str,
) -> tuple[ee.Image, ee.Image]:
    denominator = positive.add(negative)
    denominator_valid = denominator.abs().gt(1e-6)
    index = positive.subtract(negative).divide(denominator).rename(name)
    return index.updateMask(denominator_valid), denominator_valid


def build_optical_change_image(
    region: ee.Geometry,
    pre_year: int,
    post_year: int,
    *,
    max_scene_cloud_pct: float,
    min_clear_observations: int,
) -> tuple[ee.Image, dict[str, int]]:
    """Build fixed annual |dNBR|/|dNDVI| image plus explicit support bands."""

    pre_collection = _s2_annual_collection(
        region, pre_year, max_scene_cloud_pct=max_scene_cloud_pct
    )
    post_collection = _s2_annual_collection(
        region, post_year, max_scene_cloud_pct=max_scene_cloud_pct
    )
    scene_counts = {
        "pre_scene_count": int(pre_collection.size().getInfo()),
        "post_scene_count": int(post_collection.size().getInfo()),
    }
    if scene_counts["pre_scene_count"] == 0 or scene_counts["post_scene_count"] == 0:
        return ee.Image(), scene_counts

    pre = pre_collection.median().select(list(S2_BANDS))
    post = post_collection.median().select(list(S2_BANDS))
    pre_count = pre_collection.select("B8").count().rename("pre_clear_observations")
    post_count = (
        post_collection.select("B8").count().rename("post_clear_observations")
    )

    pre_ndvi, pre_ndvi_valid = _safe_ee_normalized_difference(
        pre.select("B8"), pre.select("B4"), "pre_ndvi"
    )
    post_ndvi, post_ndvi_valid = _safe_ee_normalized_difference(
        post.select("B8"), post.select("B4"), "post_ndvi"
    )
    pre_nbr, pre_nbr_valid = _safe_ee_normalized_difference(
        pre.select("B8"), pre.select("B12"), "pre_nbr"
    )
    post_nbr, post_nbr_valid = _safe_ee_normalized_difference(
        post.select("B8"), post.select("B12"), "post_nbr"
    )

    common_valid = (
        pre_count.gte(int(min_clear_observations))
        .And(post_count.gte(int(min_clear_observations)))
        .And(pre_ndvi_valid)
        .And(post_ndvi_valid)
        .And(pre_nbr_valid)
        .And(post_nbr_valid)
    )
    abs_dnbr = (
        pre_nbr.subtract(post_nbr)
        .abs()
        .rename("abs_dnbr")
        .updateMask(common_valid)
    )
    abs_dndvi = (
        post_ndvi.subtract(pre_ndvi)
        .abs()
        .rename("abs_dndvi")
        .updateMask(common_valid)
    )
    support = common_valid.unmask(0).float().rename("s2_valid_fraction")
    image = (
        abs_dnbr.addBands(abs_dndvi)
        .addBands(support)
        .addBands(pre_count.unmask(0).float())
        .addBands(post_count.unmask(0).float())
        .clip(region)
    )
    return image, scene_counts


def _feature_collection_to_frame(collection: ee.FeatureCollection) -> pd.DataFrame:
    info = collection.getInfo()
    rows: list[dict[str, Any]] = []
    for feature in info.get("features", []):
        properties = dict(feature.get("properties", {}))
        if properties.get("h3_index") is not None:
            properties["h3_index"] = str(properties["h3_index"])
        rows.append(properties)
    return pd.DataFrame(rows)


def reduce_optical_h3(
    image: ee.Image,
    h3_ids: Sequence[str],
    *,
    scale_m: int,
    batch_size: int,
    tile_scale: int,
    max_retries: int,
    context: str,
) -> pd.DataFrame:
    """Deterministic H3 batches with exact response-population validation."""

    cells = [str(value) for value in h3_ids]
    _validate_exact_h3_population(
        pd.DataFrame({"h3_index": cells}),
        cells,
        context=f"{context} request",
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.count(),
        sharedInputs=True,
    )
    batch_count = math.ceil(len(cells) / batch_size)
    frames: list[pd.DataFrame] = []
    for batch_number, start in enumerate(range(0, len(cells), batch_size), start=1):
        batch_ids = cells[start : start + batch_size]
        reduced = image.reduceRegions(
            collection=_h3_feature_collection(batch_ids),
            reducer=reducer,
            scale=int(scale_m),
            tileScale=int(tile_scale),
        )
        for attempt in range(max_retries + 1):
            try:
                frame = _feature_collection_to_frame(reduced)
                frame = _validate_exact_h3_population(
                    frame,
                    batch_ids,
                    context=f"{context} batch {batch_number}/{batch_count}",
                )
                frames.append(frame)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"{context}: batch {batch_number}/{batch_count} failed "
                        f"after {attempt + 1} attempts"
                    ) from exc
                delay = 2**attempt
                LOGGER.warning(
                    "%s batch %d/%d failed (%s); retrying in %d s",
                    context,
                    batch_number,
                    batch_count,
                    exc,
                    delay,
                )
                time.sleep(delay)
        LOGGER.info(
            "%s: completed batch %d/%d (%d/%d cells)",
            context,
            batch_number,
            batch_count,
            min(start + batch_size, len(cells)),
            len(cells),
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _validate_exact_h3_population(
        combined,
        cells,
        context=f"{context} combined",
    )
    expected_columns = {
        "abs_dnbr_mean": "abs_dnbr",
        "abs_dndvi_mean": "abs_dndvi",
        "s2_valid_fraction_mean": "s2_valid_fraction",
        "pre_clear_observations_mean": "pre_clear_observations_mean",
        "post_clear_observations_mean": "post_clear_observations_mean",
        "abs_dnbr_count": "abs_dnbr_pixel_count",
        "abs_dndvi_count": "abs_dndvi_pixel_count",
        "s2_valid_fraction_count": "h3_sample_pixel_count",
    }
    for column in expected_columns:
        if column not in combined.columns:
            combined[column] = np.nan
    result = combined[
        ["h3_index", *expected_columns.keys()]
    ].rename(columns=expected_columns)
    for column in result.columns:
        if column != "h3_index":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def descriptive_correlation(
    frame: pd.DataFrame,
    predictor: str,
) -> tuple[float | None, float | None]:
    x = pd.to_numeric(frame[predictor], errors="coerce")
    y = pd.to_numeric(frame["benchmark_mean"], errors="coerce")
    finite = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pearson = _optional_float(x.corr(y, method="pearson"))
        spearman = _optional_float(x.corr(y, method="spearman"))
    return pearson, spearman


def _event_metric_rows(
    spec: EventSpec,
    joined: pd.DataFrame,
    *,
    min_event_valid_cells: int,
    min_event_valid_coverage: float,
    scene_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = joined[joined["common_valid"]].copy()
    n_target = int(len(joined))
    n_common = int(len(valid))
    coverage = n_common / n_target if n_target else 0.0
    support_valid = (
        n_common >= int(min_event_valid_cells)
        and coverage >= float(min_event_valid_coverage)
    )

    rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for predictor in PREDICTORS:
        if support_valid:
            pearson, spearman = descriptive_correlation(valid, predictor)
        else:
            pearson, spearman = None, None
        correlations[predictor] = {
            "pearson_r": pearson,
            "spearman_r": spearman,
        }
        rows.append(
            {
                "event_key": spec.event_key,
                "label": spec.label,
                "family": spec.family,
                "pre_year": spec.pre_year,
                "post_year": spec.post_year,
                "reference_type": spec.reference_type,
                "reference_year": spec.reference_year,
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS[predictor],
                "n_reference_h3": n_target,
                "n_common_valid_h3": n_common,
                "valid_h3_coverage": coverage,
                "s2_support_valid": support_valid,
                "pre_scene_count": scene_counts["pre_scene_count"],
                "post_scene_count": scene_counts["post_scene_count"],
                "pearson_r": pearson,
                "spearman_r": spearman,
            }
        )

    direct: dict[str, Any] = {}
    if support_valid:
        for optical_predictor in ("abs_dnbr", "abs_dndvi"):
            proxy = pd.DataFrame(
                {
                    "ecs_mean": valid["ecs_mean"],
                    "benchmark_mean": valid[optical_predictor],
                }
            )
            pearson, spearman = descriptive_correlation(proxy, "ecs_mean")
            direct[f"ecs_vs_{optical_predictor}"] = {
                "pearson_r": pearson,
                "spearman_r": spearman,
            }
    summary = {
        "event": asdict(spec),
        "status": "analyzed" if support_valid else "invalid_s2_support",
        "support": {
            "n_reference_h3": n_target,
            "n_common_valid_h3": n_common,
            "valid_h3_coverage": coverage,
            "minimum_cell_valid_fraction": float(
                joined.attrs.get("min_cell_valid_fraction", np.nan)
            ),
            "minimum_event_valid_cells": int(min_event_valid_cells),
            "minimum_event_valid_coverage": float(min_event_valid_coverage),
            **scene_counts,
        },
        "correlations_vs_reference": correlations,
        "correlations_ecs_vs_optical": direct,
        "inference": (
            "Descriptive coefficients only; no cell-level p-values or independent "
            "cell assumptions."
        ),
    }
    return rows, summary


def aggregate_event_results(per_event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = per_event[
        per_event["s2_support_valid"].astype(bool)
    ].copy()
    aggregate_rows: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        subset = valid[valid["predictor"] == predictor]
        row: dict[str, Any] = {
            "predictor": predictor,
            "predictor_label": PREDICTOR_LABELS[predictor],
        }
        for method in ("pearson_r", "spearman_r"):
            values = pd.to_numeric(subset[method], errors="coerce").dropna()
            row[f"n_events_{method}"] = int(len(values))
            row[f"mean_{method}"] = _optional_float(values.mean())
            row[f"median_{method}"] = _optional_float(values.median())
            row[f"min_{method}"] = _optional_float(values.min())
            row[f"max_{method}"] = _optional_float(values.max())
        aggregate_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    for optical in ("abs_dnbr", "abs_dndvi"):
        for method in ("pearson_r", "spearman_r"):
            wide = valid.pivot(
                index="event_key",
                columns="predictor",
                values=method,
            )
            if "ecs_mean" not in wide or optical not in wide:
                differences = pd.Series(dtype=float)
            else:
                differences = (
                    pd.to_numeric(wide["ecs_mean"], errors="coerce")
                    - pd.to_numeric(wide[optical], errors="coerce")
                ).dropna()
            paired_rows.append(
                {
                    "comparison": f"ecs_mean_minus_{optical}",
                    "correlation_method": method.removesuffix("_r"),
                    "n_events": int(len(differences)),
                    "mean_difference": _optional_float(differences.mean()),
                    "median_difference": _optional_float(differences.median()),
                    "ecs_higher_count": int((differences > 0).sum()),
                    "equal_count": int((differences == 0).sum()),
                    "optical_higher_count": int((differences < 0).sum()),
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows)


def write_head_to_head_figure(per_event: pd.DataFrame, output_path: Path) -> None:
    plot_data = per_event[
        per_event["s2_support_valid"].astype(bool)
        & per_event["pearson_r"].notna()
        & per_event["spearman_r"].notna()
    ].copy()
    if plot_data.empty:
        LOGGER.warning("No valid event coefficients; skipping %s", output_path)
        return

    mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_order = sorted(plot_data["event_key"].unique())
    y_positions = {event: index for index, event in enumerate(event_order)}
    offsets = {"ecs_mean": -0.18, "abs_dnbr": 0.0, "abs_dndvi": 0.18}
    styles = {
        "ecs_mean": ("o", "#3b4cc0"),
        "abs_dnbr": ("s", "#b40426"),
        "abs_dndvi": ("^", "#2a9d55"),
    }
    fig_height = max(4.8, 0.48 * len(event_order) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, fig_height), dpi=160, sharey=True)
    for axis, method, title in zip(
        axes,
        ("pearson_r", "spearman_r"),
        ("Pearson r", "Spearman r"),
    ):
        for predictor in PREDICTORS:
            subset = plot_data[plot_data["predictor"] == predictor]
            marker, color = styles[predictor]
            ys = [
                y_positions[event] + offsets[predictor]
                for event in subset["event_key"]
            ]
            axis.scatter(
                subset[method].astype(float),
                ys,
                label=PREDICTOR_LABELS[predictor],
                marker=marker,
                color=color,
                s=42,
                zorder=3,
            )
        axis.axvline(0.0, color="0.45", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel(f"{title} with event reference")
        axis.set_title(title)
    axes[0].set_yticks(range(len(event_order)))
    axes[0].set_yticklabels(event_order, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle(
        "REO-2 conventional optical baseline\n"
        "Same valid H3 cells within each event; descriptive coefficients, no cell p-values",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        result["head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        result["working_tree_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        result["head"] = None
        result["working_tree_dirty"] = None
    return result


def study_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "analysis_name": "REO-2 conventional optical-change baseline",
        "analysis_version": ANALYSIS_VERSION,
        "selection_frozen_before_sentinel2_queries": True,
        "candidate_events": [asdict(spec) for spec in EVENT_SPECS],
        "eligibility_rules": {
            "all_families": [
                "existing joined_inner.parquet and integrity_report.json",
                "exactly one H3 row per cell",
                "finite ECS and benchmark on the reference population",
                "reference has at least two distinct values and positive support",
            ],
            "fire": [
                "MODIS MCD64A1 reference",
                "post_year = pre_year + 1",
                "reference_year = post_year (leakage guard: event/reference year is not the full pre year)",
            ],
            "volcanic_eruption": [
                "event-specific binary geometry reference",
                "pre_year < reference_year < post_year",
            ],
            "explicitly_not_used": [
                "FIRES_GOOD or FIRES_POOR",
                "visual alignment labels",
                "ECS/reference correlations",
                "Sentinel-2 values or correlations",
            ],
        },
        "sentinel2": {
            "collection": S2_COLLECTION,
            "date_window": "[January 1, January 1 of next year)",
            "annual_composite": "per-band median of every retained scene",
            "scene_filter": {
                "CLOUDY_PIXEL_PERCENTAGE_lte": args.max_scene_cloud_pct,
                "scene_limit": None,
            },
            "pixel_mask": {
                "source": "SCL",
                "excluded_classes": list(SCL_EXCLUDED_CLASSES),
                "excluded_meanings": [
                    "no data",
                    "saturated/defective",
                    "cloud shadow",
                    "unclassified",
                    "medium-probability cloud",
                    "high-probability cloud",
                    "thin cirrus",
                    "snow/ice",
                ],
                "minimum_clear_observations_per_year": args.min_clear_observations,
                "qa60_used": False,
                "qa60_reason": (
                    "SCL is temporally consistent for the study years; QA60 has a "
                    "documented production gap/change and is not combined with SCL."
                ),
            },
            "reflectance_scale_factor": 0.0001,
            "bands": {
                "red": "B4",
                "nir": "B8",
                "swir2": "B12",
            },
            "analysis_scale_m": args.scale_m,
            "h3_reducer": "mean over jointly valid optical pixels",
            "formulas": {
                "NDVI": "(B8 - B4) / (B8 + B4)",
                "NBR": "(B8 - B12) / (B8 + B12)",
                "abs_dNDVI": "abs(NDVI_post - NDVI_pre)",
                "abs_dNBR": "abs(NBR_pre - NBR_post)",
            },
            "common_pixel_support": (
                "Both measures require all B4/B8/B12 denominators and at least "
                "the minimum clear observations in both years."
            ),
            "existing_gis_logic_review": {
                "reused": [
                    "COPERNICUS/S2_SR_HARMONIZED",
                    "CLOUDY_PIXEL_PERCENTAGE scene screening",
                    "median mosaicking",
                    "full-calendar-year mode",
                ],
                "not_reused": [
                    "June-September visualization window",
                    "clearest-N scene truncation",
                    "RGB min/max/gamma",
                    "rank/quantile/IQR visualization stretches",
                ],
                "analysis_addition": "explicit SCL cloud/shadow/cirrus/snow pixel mask",
            },
        },
        "support_thresholds": {
            "minimum_valid_pixel_fraction_per_h3": args.min_cell_valid_fraction,
            "minimum_valid_h3_fraction_per_event": args.min_event_valid_coverage,
            "minimum_valid_h3_count_per_event": args.min_event_valid_cells,
            "insufficient_support_policy": "report invalid; never impute",
        },
        "comparison": {
            "target_population": (
                "exact H3/reference rows in each source joined_inner.parquet"
            ),
            "same_population_rule": (
                "ECS, abs_dNBR, and abs_dNDVI correlations use one identical "
                "event-level complete-case H3 mask"
            ),
            "coefficients": ["Pearson r", "Spearman r"],
            "cell_level_significance_tests": False,
            "aggregate_unit": "event",
            "pooled_cell_correlation": False,
            "metric_selection": "all predefined predictors are always reported",
        },
        "earth_engine": {
            "project": args.ee_project,
            "batch_size": args.ee_batch_size,
            "tile_scale": args.tile_scale,
            "max_retries": args.max_retries,
        },
    }


def _selection_markdown(eligibility: pd.DataFrame) -> str:
    lines = [
        "# REO-2 optical baseline eligibility",
        "",
        "Selection was evaluated before any Sentinel-2 query. No FIRES_GOOD, visual, "
        "or correlation-based labels were used.",
        "",
        "| event | family | pre/post | reference year | eligible | reason |",
        "|---|---|---:|---:|:---:|---|",
    ]
    for row in eligibility.itertuples(index=False):
        reason = row.exclusion_reasons or "passes predefined criteria"
        lines.append(
            f"| {row.event_key} | {row.family} | {row.pre_year}/{row.post_year} | "
            f"{row.reference_year} | {'yes' if row.eligible else 'no'} | {reason} |"
        )
    lines.extend(
        [
            "",
            "Fire temporal rule: the MODIS/reference year must equal `post_year`; this "
            "prevents a full-calendar pre composite from overlapping the mapped fire year. "
            "Volcanic events instead require the mapped eruption year to be strictly "
            "bracketed by pre/post years.",
            "",
        ]
    )
    return "\n".join(lines)


def _cautious_conclusion(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any]:
    predictor_medians: dict[str, dict[str, float | None]] = {}
    for row in aggregate.to_dict(orient="records"):
        predictor_medians[str(row["predictor"])] = {
            "median_pearson_r": _optional_float(row.get("median_pearson_r")),
            "median_spearman_r": _optional_float(row.get("median_spearman_r")),
        }
    comparisons = paired.to_dict(orient="records")
    finite_comparisons = [
        row for row in comparisons if int(row.get("n_events", 0)) > 0
    ]
    if not finite_comparisons:
        text = (
            "Sentinel-2 support was insufficient for a head-to-head conclusion. "
            "No values were imputed."
        )
    else:
        pearson_wins = sum(
            int(row["ecs_higher_count"])
            for row in finite_comparisons
            if row["correlation_method"] == "pearson"
        )
        pearson_total = sum(
            int(row["n_events"])
            for row in finite_comparisons
            if row["correlation_method"] == "pearson"
        )
        spearman_wins = sum(
            int(row["ecs_higher_count"])
            for row in finite_comparisons
            if row["correlation_method"] == "spearman"
        )
        spearman_total = sum(
            int(row["n_events"])
            for row in finite_comparisons
            if row["correlation_method"] == "spearman"
        )
        text = (
            f"Across paired event-level comparisons, ECS had the higher descriptive "
            f"coefficient in {pearson_wins}/{pearson_total} Pearson and "
            f"{spearman_wins}/{spearman_total} Spearman comparisons against the two "
            "predefined optical baselines. This indicates relative alignment with the "
            "available references, not statistically independent or uniquely incremental "
            "information: H3 cells are spatially dependent, references are imperfect, and "
            "no held-out predictive model was fit."
        )
    return {
        "text": text,
        "event_level_predictor_medians": predictor_medians,
        "paired_comparisons": comparisons,
        "scope_limit": (
            "Descriptive same-population comparison only; no cell-level significance "
            "and no claim of causal or unique information."
        ),
    }


def run(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    eligibility, populations = build_eligibility_table(input_root)
    requested = (
        {value.strip() for value in args.events.split(",") if value.strip()}
        if args.events
        else set(eligibility.loc[eligibility["eligible"], "event_key"].astype(str))
    )
    unknown = sorted(requested - set(eligibility["event_key"].astype(str)))
    if unknown:
        raise SystemExit(f"Unknown candidate event(s): {unknown}")
    eligibility["requested_for_run"] = eligibility["event_key"].isin(requested)

    config = study_config(args)
    if args.dry_run:
        print(
            eligibility[
                [
                    "event_key",
                    "family",
                    "eligible",
                    "exclusion_reasons",
                    "requested_for_run",
                ]
            ].to_string(index=False)
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(output_root / "eligibility.csv", index=False)
    (output_root / "eligibility.md").write_text(
        _selection_markdown(eligibility),
        encoding="utf-8",
    )
    _write_json(output_root / "study_config.json", config)
    if args.selection_only:
        LOGGER.info("Wrote deterministic selection artifacts to %s", output_root)
        return 0

    eligible_keys = set(
        eligibility.loc[eligibility["eligible"], "event_key"].astype(str)
    )
    run_keys = sorted(requested & eligible_keys)
    if not run_keys:
        raise SystemExit("No requested event passes the predefined eligibility rules")

    ee.Initialize(project=args.ee_project)
    spec_by_key = {spec.event_key: spec for spec in EVENT_SPECS}
    metric_rows: list[dict[str, Any]] = []
    run_status: list[dict[str, Any]] = []
    scene_counts_by_event: dict[str, dict[str, int]] = {}

    for event_key in run_keys:
        spec = spec_by_key[event_key]
        population = populations[event_key].copy()
        event_output = output_root / event_key
        event_output.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "%s: full-year Sentinel-2 %d -> %d on %d reference H3 cells",
            event_key,
            spec.pre_year,
            spec.post_year,
            len(population),
        )
        started = time.monotonic()
        try:
            h3_ids = population["h3_index"].astype(str).tolist()
            region = _h3_bounds(h3_ids)
            optical_image, scene_counts = build_optical_change_image(
                region,
                spec.pre_year,
                spec.post_year,
                max_scene_cloud_pct=args.max_scene_cloud_pct,
                min_clear_observations=args.min_clear_observations,
            )
            scene_counts_by_event[event_key] = scene_counts
            if (
                scene_counts["pre_scene_count"] == 0
                or scene_counts["post_scene_count"] == 0
            ):
                summary = {
                    "event": asdict(spec),
                    "status": "invalid_s2_support",
                    "reason": "zero_retained_sentinel2_scenes",
                    **scene_counts,
                }
                _write_json(event_output / "event_summary.json", summary)
                run_status.append(
                    {
                        "event_key": event_key,
                        "status": "invalid_s2_support",
                        "detail": "zero_retained_sentinel2_scenes",
                    }
                )
                continue

            optical = reduce_optical_h3(
                optical_image,
                h3_ids,
                scale_m=args.scale_m,
                batch_size=args.ee_batch_size,
                tile_scale=args.tile_scale,
                max_retries=args.max_retries,
                context=event_key,
            )
            joined = join_same_population(
                population,
                optical,
                min_cell_valid_fraction=args.min_cell_valid_fraction,
            )
            joined.attrs["min_cell_valid_fraction"] = args.min_cell_valid_fraction
            joined.to_parquet(event_output / "same_population_h3.parquet", index=False)

            rows, summary = _event_metric_rows(
                spec,
                joined,
                min_event_valid_cells=args.min_event_valid_cells,
                min_event_valid_coverage=args.min_event_valid_coverage,
                scene_counts=scene_counts,
            )
            metric_rows.extend(rows)
            summary["elapsed_seconds"] = time.monotonic() - started
            summary["source_joined_parquet"] = str(
                input_root / event_key / "joined_inner.parquet"
            )
            _write_json(event_output / "event_summary.json", summary)
            pd.DataFrame(rows).to_csv(
                event_output / "correlations.csv",
                index=False,
            )
            run_status.append(
                {
                    "event_key": event_key,
                    "status": summary["status"],
                    "detail": "",
                }
            )
        except Exception as exc:
            LOGGER.exception("%s failed; retaining an explicit invalid/error record", event_key)
            _write_json(
                event_output / "event_summary.json",
                {
                    "event": asdict(spec),
                    "status": "processing_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "policy": "No imputation; event omitted from coefficient summaries.",
                },
            )
            run_status.append(
                {
                    "event_key": event_key,
                    "status": "processing_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    status_frame = pd.DataFrame(run_status)
    status_frame.to_csv(output_root / "run_status.csv", index=False)
    per_event = pd.DataFrame(metric_rows)
    if per_event.empty:
        LOGGER.error("No event produced metric rows; see run_status.csv")
        return 2

    per_event.to_csv(output_root / "per_event_results.csv", index=False)
    aggregate, paired = aggregate_event_results(per_event)
    aggregate.to_csv(output_root / "aggregate_results.csv", index=False)
    paired.to_csv(output_root / "paired_event_differences.csv", index=False)
    head_to_head = per_event[
        [
            "event_key",
            "family",
            "n_reference_h3",
            "n_common_valid_h3",
            "valid_h3_coverage",
            "predictor",
            "pearson_r",
            "spearman_r",
        ]
    ].copy()
    head_to_head.to_csv(output_root / "head_to_head_table.csv", index=False)
    write_head_to_head_figure(per_event, output_root / "head_to_head.png")

    conclusion = _cautious_conclusion(aggregate, paired)
    _write_json(output_root / "conclusion.json", conclusion)
    (output_root / "conclusion.txt").write_text(
        conclusion["text"] + "\n",
        encoding="utf-8",
    )

    source_provenance: list[dict[str, Any]] = []
    for event_key in sorted(set(eligibility["event_key"].astype(str))):
        joined_path = input_root / event_key / "joined_inner.parquet"
        integrity_path = input_root / event_key / "integrity_report.json"
        source_provenance.append(
            {
                "event_key": event_key,
                "joined_inner_path": str(joined_path),
                "joined_inner_sha256": (
                    _sha256(joined_path) if joined_path.is_file() else None
                ),
                "integrity_path": str(integrity_path),
                "integrity_sha256": (
                    _sha256(integrity_path) if integrity_path.is_file() else None
                ),
            }
        )
    provenance = {
        "generated_utc": _utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "command_configuration": config,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_files": source_provenance,
        "scene_counts": scene_counts_by_event,
        "software": {
            "python": platform.python_version(),
            "earthengine-api": _package_version("earthengine-api"),
            "h3": _package_version("h3"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "pyarrow": _package_version("pyarrow"),
            "matplotlib": _package_version("matplotlib"),
        },
        "git": _git_metadata(repo_root),
        "output_note": (
            "No source Parquet, main pipeline, paper, manifest, or audit file was modified."
        ),
    }
    _write_json(output_root / "provenance.json", provenance)
    LOGGER.info("Completed optical baseline study in %s", output_root)
    print(conclusion["text"])
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Existing per-event pipeline outputs (read-only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optical_change_baseline"),
        help="Dedicated study output directory.",
    )
    parser.add_argument(
        "--ee-project",
        default=os.environ.get("EE_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or "project0-256023",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Optional comma-separated subset of predefined candidates; default is every eligible event.",
    )
    parser.add_argument("--scale-m", type=int, default=20)
    parser.add_argument("--max-scene-cloud-pct", type=float, default=70.0)
    parser.add_argument("--min-clear-observations", type=int, default=2)
    parser.add_argument("--min-cell-valid-fraction", type=float, default=0.50)
    parser.add_argument("--min-event-valid-coverage", type=float, default=0.50)
    parser.add_argument("--min-event-valid-cells", type=int, default=30)
    parser.add_argument("--ee-batch-size", type=int, default=200)
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write eligibility/config artifacts without querying Earth Engine.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deterministic eligibility without writing or querying Earth Engine.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.max_scene_cloud_pct <= 100.0:
        parser.error("--max-scene-cloud-pct must be in [0, 100]")
    for name in ("min_cell_valid_fraction", "min_event_valid_coverage"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.min_clear_observations < 1:
        parser.error("--min-clear-observations must be positive")
    if args.min_event_valid_cells < 2:
        parser.error("--min-event-valid-cells must be at least 2")
    if args.scale_m <= 0 or args.ee_batch_size <= 0 or args.tile_scale <= 0:
        parser.error("scale, batch size, and tile scale must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
