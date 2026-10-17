#!/usr/bin/env python3
"""Normalize an authoritative polygon extent into a binary GeoJSON reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(input_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    source = gpd.read_file(input_path)
    if source.crs is None:
        raise ValueError(f"Input has no declared CRS: {input_path}")
    source_crs = source.crs.to_string()
    polygonal = source[source.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polygonal.empty:
        raise ValueError("Input contains no Polygon or MultiPolygon features.")
    polygonal["geometry"] = polygonal.geometry.make_valid()
    polygonal = polygonal[~polygonal.geometry.is_empty & polygonal.geometry.notna()]
    normalized = polygonal.to_crs(4326)
    geometry = normalized.geometry.union_all()
    output = gpd.GeoDataFrame(
        [{"ref_class": "affected", "geometry": geometry}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    projected = output.to_crs(output.estimate_utm_crs())
    summary = {
        "source_feature_count": int(len(source)),
        "polygon_feature_count": int(len(polygonal)),
        "source_crs": source_crs,
        "output_crs": "EPSG:4326",
        "output_bounds_wgs84": [float(value) for value in output.total_bounds],
        "affected_area_km2": float(projected.geometry.area.sum() / 1_000_000.0),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-geojson", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--observation-start", required=True)
    parser.add_argument("--observation-end", required=True)
    parser.add_argument("--license", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_geojson = args.output_geojson.resolve()
    output_metadata = args.output_metadata.resolve()
    normalized, summary = _normalize(input_path)

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = json.loads(normalized.to_json(drop_id=True))
    output_geojson.write_text(
        json.dumps(feature_collection, indent=2),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "source_title": args.source_title,
        "source_id": args.source_id,
        "source_url": args.source_url,
        "observation_start": args.observation_start,
        "observation_end": args.observation_end,
        "license": args.license,
        "reference_semantics": (
            "Binary affected-area footprint. Outside the polygon is a valid "
            "negative only within the configured analysis AOI."
        ),
        "prepared_utc": datetime.now(UTC).isoformat(),
        "input_path": str(args.input),
        "input_sha256": _sha256(input_path),
        "output_geojson": str(args.output_geojson),
        "output_sha256": _sha256(output_geojson),
        **summary,
    }
    if args.source_archive:
        archive = args.source_archive.resolve()
        metadata["source_archive"] = str(args.source_archive)
        metadata["source_archive_sha256"] = _sha256(archive)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_geojson}")
    print(f"Wrote {output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
