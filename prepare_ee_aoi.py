#!/usr/bin/env python3
"""Export one authoritative Earth Engine boundary feature to local GeoJSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--filter-field", required=True)
    parser.add_argument("--filter-value", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ee-project")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = (
        args.ee_project
        or os.environ.get("EE_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not project:
        raise SystemExit(
            "Pass --ee-project or set EE_PROJECT_ID/GOOGLE_CLOUD_PROJECT."
        )
    ee.Initialize(project=project)
    matches = ee.FeatureCollection(args.asset).filter(
        ee.Filter.eq(args.filter_field, args.filter_value)
    )
    count = int(matches.size().getInfo())
    if count != 1:
        raise ValueError(f"Expected one matching feature, found {count}")
    feature = ee.Feature(matches.first())
    properties = feature.toDictionary().getInfo()
    properties.update(
        {
            "source_asset": args.asset,
            "source_filter_field": args.filter_field,
            "source_filter_value": args.filter_value,
        }
    )
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": properties,
                "geometry": feature.geometry().getInfo(),
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
