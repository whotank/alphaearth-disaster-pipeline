"""Official MTBS source acquisition and geometry/raster QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import ee
import h3
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from study_core import (
    GateFailure,
    MTBS_SEVERITY_COLLECTION,
    RAW_ROOT,
    StudyError,
    atomic_write_bytes,
    ee_geometry,
    geometry_to_h3,
    relative,
    sha256_bytes,
)


ARCGIS_ROOT = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer"
)
MTBS_POINT_LAYER = 62
MTBS_BOUNDARY_LAYER = 63
HTTP_RETRIES = 3
AREA_TRANSFORMER = Transformer.from_crs(
    "EPSG:4326", "EPSG:6933", always_xy=True
)
WGS84 = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Download:
    data: bytes
    final_url: str
    content_type: str | None


def http_get(url: str, *, retries: int = HTTP_RETRIES) -> Download:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "alphaearth-mtbs-heldout-study/1.0",
                    "Accept": "*/*",
                },
            )
            with urlopen(request, timeout=180) as response:
                return Download(
                    data=response.read(),
                    final_url=response.geturl(),
                    content_type=response.headers.get_content_type(),
                )
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise GateFailure(
        "P01", f"HTTP request failed after {retries + 1} attempts: {last_error}"
    )


def arcgis_query(
    layer: int,
    parameters: dict[str, Any],
    *,
    raw_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = {
        "f": "geojson",
        "outSR": 4326,
        "returnGeometry": "true",
        **parameters,
    }
    url = f"{ARCGIS_ROOT}/{layer}/query?{urlencode(query)}"
    download = http_get(url)
    atomic_write_bytes(raw_path, download.data)
    try:
        payload = json.loads(download.data)
    except json.JSONDecodeError as exc:
        raise GateFailure("P01", f"ArcGIS layer {layer} returned invalid JSON") from exc
    if "error" in payload:
        raise GateFailure("P01", f"ArcGIS layer {layer} error: {payload['error']}")
    return payload, {
        "kind": "arcgis_geojson",
        "layer": layer,
        "relative_path": relative(raw_path),
        "sha256": sha256_bytes(download.data),
        "bytes": len(download.data),
        "url": url,
    }


def field(properties: dict[str, Any], name: str) -> Any:
    lookup = {str(key).casefold(): value for key, value in properties.items()}
    return lookup.get(name.casefold())


def normalized_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def value_as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        compact = str(int(value))
        if re.fullmatch(r"(?:19|20)\d{6}", compact):
            return datetime.strptime(compact, "%Y%m%d").date()
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).date()
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").date()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def polygonal_geometry(geojson_geometry: dict[str, Any]) -> Polygon | MultiPolygon:
    geometry = make_valid(shape(geojson_geometry))
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, MultiPolygon):
                polygons.extend(part.geoms)
        merged = make_valid(unary_union(polygons))
        if isinstance(merged, (Polygon, MultiPolygon)):
            return merged
    raise GateFailure("P01", f"Official boundary is not polygonal: {geometry.geom_type}")


def query_candidate_sources(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], Polygon | MultiPolygon, list[dict[str, Any]]]:
    fire_id = str(candidate["fire_id"])
    event_raw = RAW_ROOT / "mtbs" / fire_id
    where = f"fire_id='{fire_id}'"
    point_payload, point_source = arcgis_query(
        MTBS_POINT_LAYER,
        {"where": where, "outFields": "*"},
        raw_path=event_raw / "official_point.geojson",
    )
    boundary_payload, boundary_source = arcgis_query(
        MTBS_BOUNDARY_LAYER,
        {"where": where, "outFields": "*"},
        raw_path=event_raw / "official_boundary.geojson",
    )
    point_features = point_payload.get("features", [])
    boundary_features = boundary_payload.get("features", [])
    if len(point_features) != 1:
        raise GateFailure(
            "P01", f"{fire_id}: expected one official point, got {len(point_features)}"
        )
    if len(boundary_features) != 1:
        raise GateFailure(
            "P01",
            f"{fire_id}: expected one official boundary, got {len(boundary_features)}",
        )
    point_properties = dict(point_features[0].get("properties", {}))
    boundary_properties = dict(boundary_features[0].get("properties", {}))
    geometry = polygonal_geometry(boundary_features[0]["geometry"])
    return (
        {**boundary_properties, **point_properties},
        geometry,
        [point_source, boundary_source],
    )


def metadata_mismatches(
    candidate: dict[str, Any], properties: dict[str, Any]
) -> list[str]:
    comparisons = {
        "fire_id": (field(properties, "fire_id"), candidate["fire_id"]),
        "fire_name": (
            normalized_name(field(properties, "fire_name")),
            normalized_name(candidate["name"]),
        ),
        "asmnt_type": (
            normalized_name(field(properties, "asmnt_type")),
            normalized_name(candidate["assessment"]),
        ),
        "map_id": (field(properties, "map_id"), candidate["map_id"]),
        "pre_id": (
            normalized_name(field(properties, "pre_id")),
            normalized_name(candidate["pre_id"]),
        ),
        "post_id": (
            normalized_name(field(properties, "post_id")),
            normalized_name(candidate["post_id"]),
        ),
    }
    mismatches: list[str] = []
    for name, (observed, expected) in comparisons.items():
        if name == "map_id":
            try:
                matches = int(observed) == int(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(f"{name}: observed={observed!r}, expected={expected!r}")
    observed_date = value_as_date(field(properties, "ig_date"))
    expected_date = date.fromisoformat(str(candidate["date"]))
    if observed_date != expected_date:
        mismatches.append(
            f"ig_date: observed={observed_date}, expected={expected_date}"
        )
    return mismatches


def query_intersecting_boundaries(
    fire_id: str, geometry: Polygon | MultiPolygon
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    envelope = {
        "xmin": minimum_x,
        "ymin": minimum_y,
        "xmax": maximum_x,
        "ymax": maximum_y,
        "spatialReference": {"wkid": 4326},
    }
    all_features: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload, source = arcgis_query(
            MTBS_BOUNDARY_LAYER,
            {
                "where": "1=1",
                "outFields": "fire_id,fire_name,ig_date,fire_type,asmnt_type",
                "geometry": json.dumps(envelope, separators=(",", ":")),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": "fire_id",
            },
            raw_path=(
                RAW_ROOT / "mtbs" / fire_id / f"intersections_{offset:04d}.geojson"
            ),
        )
        features = payload.get("features", [])
        all_features.extend(features)
        sources.append(source)
        if len(features) < 2000:
            return all_features, sources
        offset += len(features)


def overlap_audit(
    candidate: dict[str, Any],
    geometry: Polygon | MultiPolygon,
    features: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fire_id = str(candidate["fire_id"])
    year = int(str(candidate["date"])[:4])
    projected_candidate = transform(AREA_TRANSFORMER.transform, geometry)
    candidate_area = float(projected_candidate.area)
    if candidate_area <= 0:
        raise GateFailure("P01", f"{fire_id}: nonpositive projected boundary area")

    temporal_intersections: list[Any] = []
    historical_intersections: list[Any] = []
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = dict(feature.get("properties", {}))
        other_id = str(field(properties, "fire_id") or "")
        if not other_id or other_id == fire_id or not feature.get("geometry"):
            continue
        other_geometry = polygonal_geometry(feature["geometry"])
        intersection = projected_candidate.intersection(
            transform(AREA_TRANSFORMER.transform, other_geometry)
        )
        area = float(intersection.area)
        if area <= 0:
            continue
        other_date = value_as_date(field(properties, "ig_date"))
        in_window = (
            other_date is not None and year - 1 <= other_date.year <= year + 1
        )
        historical_intersections.append(intersection)
        if in_window:
            temporal_intersections.append(intersection)
        rows.append(
            {
                "fire_id": other_id,
                "fire_name": field(properties, "fire_name"),
                "ignition_date": other_date.isoformat() if other_date else None,
                "intersection_area_m2": area,
                "candidate_fraction": area / candidate_area,
                "in_y_minus_1_to_y_plus_1": in_window,
            }
        )
    temporal_area = (
        float(unary_union(temporal_intersections).area)
        if temporal_intersections
        else 0.0
    )
    historical_area = (
        float(unary_union(historical_intersections).area)
        if historical_intersections
        else 0.0
    )
    return {
        "area_crs": "EPSG:6933",
        "candidate_area_m2": candidate_area,
        "temporal_window": [year - 1, year + 1],
        "temporal_union_intersection_m2": temporal_area,
        "temporal_union_fraction": temporal_area / candidate_area,
        "all_time_union_intersection_m2": historical_area,
        "all_time_union_fraction": historical_area / candidate_area,
        "intersections": sorted(
            rows,
            key=lambda row: (
                not bool(row["in_y_minus_1_to_y_plus_1"]),
                -float(row["candidate_fraction"]),
                str(row["fire_id"]),
            ),
        ),
    }


def download_mtbs_severity(
    candidate: dict[str, Any], geometry: Polygon | MultiPolygon
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    fire_id = str(candidate["fire_id"])
    year = int(str(candidate["date"])[:4])
    region = ee_geometry(geometry)
    collection = (
        ee.ImageCollection(MTBS_SEVERITY_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select("Severity")
    )
    count = int(collection.size().getInfo())
    indexes = [str(value) for value in collection.aggregate_array("system:index").getInfo()]
    if count < 1:
        raise GateFailure("P01", f"{fire_id}: no official MTBS severity image for {year}")
    image = collection.mosaic().select("Severity").clip(region)
    projection = image.projection().getInfo()
    url = image.getDownloadURL(
        {
            "name": f"{fire_id}_{year}_mtbs_severity",
            "region": region,
            "scale": 30,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    download = http_get(url)
    event_raw = RAW_ROOT / "mtbs" / fire_id
    if download.data.startswith(b"PK\x03\x04"):
        archive_path = event_raw / "severity_download.zip"
        atomic_write_bytes(archive_path, download.data)
        with zipfile.ZipFile(BytesIO(download.data)) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
                and ".." not in Path(name).parts
            )
            if len(names) != 1:
                raise GateFailure(
                    "P01", f"{fire_id}: expected one severity GeoTIFF, got {names}"
                )
            tiff_bytes = archive.read(names[0])
    else:
        archive_path = event_raw / "severity_download.tif"
        atomic_write_bytes(archive_path, download.data)
        tiff_bytes = download.data
    raster_path = event_raw / "severity.tif"
    atomic_write_bytes(raster_path, tiff_bytes)
    try:
        with rasterio.open(raster_path) as dataset:
            if dataset.count != 1:
                raise GateFailure("P01", f"{fire_id}: severity raster is not single-band")
            center_x = (dataset.bounds.left + dataset.bounds.right) / 2.0
            center_y = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
            if dataset.crs.is_geographic:
                x_resolution_m = abs(
                    WGS84.inv(
                        center_x,
                        center_y,
                        center_x + dataset.res[0],
                        center_y,
                    )[2]
                )
                y_resolution_m = abs(
                    WGS84.inv(
                        center_x,
                        center_y,
                        center_x,
                        center_y + dataset.res[1],
                    )[2]
                )
            else:
                x_resolution_m, y_resolution_m = map(abs, dataset.res)
            raster_metadata = {
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "requested_scale_m": 30,
                "width": dataset.width,
                "height": dataset.height,
                "count": dataset.count,
                "dtype": dataset.dtypes[0],
                "nodata": dataset.nodata,
                "resolution": [abs(dataset.res[0]), abs(dataset.res[1])],
                "resolution_m_at_center": [x_resolution_m, y_resolution_m],
                "bounds": list(dataset.bounds),
            }
            if dataset.crs is None or dataset.width <= 0 or dataset.height <= 0:
                raise GateFailure("P01", f"{fire_id}: invalid raster CRS/dimensions")
    except rasterio.errors.RasterioError as exc:
        raise GateFailure("P01", f"{fire_id}: severity GeoTIFF failed to parse") from exc

    metadata = {
        "collection": MTBS_SEVERITY_COLLECTION,
        "year": year,
        "image_count": count,
        "image_indexes": indexes,
        "image_index_sha256": sha256_bytes(
            json.dumps(indexes, separators=(",", ":")).encode("utf-8")
        ),
        "native_projection": projection,
        "download_projection": raster_metadata,
        "class_semantics": {
            "valid_reference_classes": [1, 2, 3, 4, 5],
            "primary_positive_classes": [3, 4],
            "excluded": [0, 6, "NoData"],
        },
    }
    sources = [
        {
            "kind": "mtbs_severity_download",
            "collection": MTBS_SEVERITY_COLLECTION,
            "relative_path": relative(archive_path),
            "sha256": sha256_bytes(download.data),
            "bytes": len(download.data),
            "url": "Earth Engine generated download URL (ephemeral; not retained)",
        },
        {
            "kind": "mtbs_severity_geotiff",
            "relative_path": relative(raster_path),
            "sha256": sha256_bytes(tiff_bytes),
            "bytes": len(tiff_bytes),
        },
    ]
    return raster_path, metadata, sources


def h3_reference_frame(
    fire_id: str,
    geometry: Polygon | MultiPolygon,
    raster_path: Path,
) -> pd.DataFrame:
    h3_ids = geometry_to_h3(geometry, resolution=8)
    rows: list[dict[str, Any]] = []
    with rasterio.open(raster_path) as dataset:
        transformer = Transformer.from_crs(
            "EPSG:4326", dataset.crs, always_xy=True
        )
        for h3_index in h3_ids:
            ring = [
                (float(longitude), float(latitude))
                for latitude, longitude in h3.cell_to_boundary(h3_index)
            ]
            ring.append(ring[0])
            polygon = Polygon(ring)
            projected = transform(transformer.transform, polygon)
            window = from_bounds(*projected.bounds, transform=dataset.transform)
            window = window.round_offsets().round_lengths()
            data = dataset.read(1, window=window, boundless=True, fill_value=0)
            inside = geometry_mask(
                [mapping(projected)],
                out_shape=data.shape,
                transform=dataset.window_transform(window),
                invert=True,
                all_touched=False,
            )
            values = data[inside]
            valid = np.isin(values, (1, 2, 3, 4, 5))
            valid_count = int(valid.sum())
            sample_count = int(len(values))
            positive_count = int(np.isin(values, (3, 4)).sum())
            support = valid_count / sample_count if sample_count else 0.0
            reference = (
                positive_count / valid_count if valid_count else float("nan")
            )
            rows.append(
                {
                    "h3_index": h3_index,
                    "mtbs_reference": reference,
                    "mtbs_valid_fraction": support,
                    "mtbs_valid_pixels": valid_count,
                    "mtbs_positive_pixels": positive_count,
                    "mtbs_h3_sample_pixels": sample_count,
                }
            )
    frame = pd.DataFrame(rows).sort_values("h3_index").reset_index(drop=True)
    if frame["h3_index"].duplicated().any():
        raise GateFailure("P01", f"{fire_id}: duplicate raster H3 reductions")
    return frame


def load_official_boundary(fire_id: str) -> Polygon | MultiPolygon:
    path = RAW_ROOT / "mtbs" / fire_id / "official_boundary.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if len(features) != 1:
        raise StudyError(f"{fire_id}: cached official boundary is invalid")
    return polygonal_geometry(features[0]["geometry"])
