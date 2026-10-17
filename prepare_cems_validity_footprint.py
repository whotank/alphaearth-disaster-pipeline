#!/usr/bin/env python3
"""Build valid CEMS analysis support from AOI and image-footprint vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--image-footprint", type=Path, required=True)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--output-geojson", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aoi = gpd.read_file(args.aoi).to_crs(4326)
    footprints = gpd.read_file(args.image_footprint).to_crs(4326)
    source_rows = footprints[footprints["or_src_id"] == args.source_id]
    image_rows = source_rows[source_rows["obj_type"] == "Image Footprint"]
    nodata_rows = source_rows[
        source_rows["obj_type"].str.contains(
            "Not Analysed",
            case=False,
            na=False,
        )
    ]
    if image_rows.empty:
        raise ValueError(f"No Image Footprint for source ID {args.source_id}")

    valid = aoi.geometry.union_all().intersection(image_rows.geometry.union_all())
    if not nodata_rows.empty:
        valid = valid.difference(nodata_rows.geometry.union_all())
    output = gpd.GeoDataFrame(
        [{"ref_class": "valid_support", "geometry": valid}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.output_geojson.write_text(
        json.dumps(json.loads(output.to_json(drop_id=True)), indent=2),
        encoding="utf-8",
    )

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "valid_support_geojson": str(args.output_geojson),
            "valid_support_sha256": _sha256(args.output_geojson),
            "valid_support_rule": (
                f"areaOfInterestA intersect Image Footprint source "
                f"{args.source_id}, minus matching Not Analysed features"
            ),
            "valid_support_area_km2": float(
                output.to_crs(output.estimate_utm_crs()).geometry.area.sum()
                / 1_000_000.0
            ),
        }
    )
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_geojson}")
    print(f"Updated {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
