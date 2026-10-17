#!/usr/bin/env python3
"""Cross-check USDM mirror benchmarks against official weekly NDMC vectors."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import shapely.geometry as sg

from joined_qa import run_qa
from usdm_validation import (
    USDM_VALIDATION_CRITERIA,
    evaluate_official_validation,
    expected_weekly_map_count,
)


# The 2020 annual archive lowercases one otherwise standard weekly member.
_WEEK_RE = re.compile(r"USDM_(\d{8})_M\.zip$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h3_polygon(index: str) -> sg.Polygon:
    ring = [(lon, lat) for lat, lon in h3.cell_to_boundary(index)]
    return sg.Polygon(ring)


def _season_members(archive: ZipFile, year: int) -> list[str]:
    selected: list[str] = []
    for name in archive.namelist():
        match = _WEEK_RE.search(name)
        if not match:
            continue
        date = pd.Timestamp(match.group(1))
        if pd.Timestamp(f"{year}-04-01") <= date < pd.Timestamp(
            f"{year}-11-01"
        ):
            selected.append(name)
    return sorted(
        selected,
        key=lambda name: _WEEK_RE.search(name).group(1),
    )


def _weekly_frame(archive: ZipFile, member: str) -> gpd.GeoDataFrame:
    with TemporaryDirectory() as directory:
        nested = ZipFile(BytesIO(archive.read(member)))
        nested.extractall(directory)
        shapefile = next(Path(directory).glob("*.shp"))
        frame = gpd.read_file(shapefile)
    if "DM" not in frame or frame.crs is None:
        raise ValueError(f"Invalid USDM shapefile in {member}")
    return frame[["DM", "geometry"]]


def _season_scores(
    archive_path: Path,
    year: int,
    support: gpd.GeoDataFrame,
) -> tuple[np.ndarray, list[str]]:
    weekly_scores: list[np.ndarray] = []
    with ZipFile(archive_path) as archive:
        members = _season_members(archive, year)
        expected_count = expected_weekly_map_count(year)
        if len(members) != expected_count:
            raise ValueError(
                f"Expected {expected_count} Tuesday April-October maps for {year}, "
                f"found {len(members)}"
            )
        dates = [pd.Timestamp(_WEEK_RE.search(member).group(1)) for member in members]
        if len(set(dates)) != len(dates) or any(day.weekday() != 1 for day in dates):
            raise ValueError(
                f"USDM archive {archive_path} has duplicate or non-Tuesday season maps"
            )
        support_area = support.geometry.area.to_numpy()
        if (
            len(support_area) != len(support)
            or not np.isfinite(support_area).all()
            or np.any(support_area <= 0)
        ):
            raise ValueError("Official-vector H3 support has empty or invalid geometry")
        for member in members:
            frame = _weekly_frame(archive, member).to_crs(support.crs)
            score = np.zeros(len(support), dtype=float)
            for row in frame.itertuples(index=False):
                overlap_area = support.geometry.intersection(row.geometry).area
                score += (int(row.DM) + 1) * (
                    overlap_area.to_numpy() / support_area
                )
            weekly_scores.append(score)
    return (
        np.mean(np.vstack(weekly_scores), axis=0),
        [str(day.date()) for day in dates],
    )


def _validate_h3_frame(
    frame: pd.DataFrame,
    expected_indices: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if "h3_index" not in frame:
        raise ValueError(f"{label} is missing h3_index")
    out = frame.copy()
    out["h3_index"] = out["h3_index"].astype(str)
    actual = out["h3_index"].tolist()
    duplicates = sorted(out.loc[out["h3_index"].duplicated(), "h3_index"].unique())
    missing = sorted(set(expected_indices) - set(actual))
    unexpected = sorted(set(actual) - set(expected_indices))
    if duplicates or missing or unexpected or len(actual) != len(expected_indices):
        raise ValueError(
            f"{label} H3 coverage mismatch: rows={len(actual)}, "
            f"expected={len(expected_indices)}, duplicates={duplicates[:10]}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    order = {index: position for position, index in enumerate(expected_indices)}
    return (
        out.assign(_h3_order=out["h3_index"].map(order))
        .sort_values("_h3_order", kind="stable")
        .drop(columns="_h3_order")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--pre-year", type=int, required=True)
    parser.add_argument("--post-year", type=int, required=True)
    parser.add_argument("--pre-archive", type=Path, required=True)
    parser.add_argument("--post-archive", type=Path, required=True)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_dir = args.output_dir.resolve() / args.event
    recorded_event_dir = args.output_dir / args.event
    alpha_path = event_dir / "alphaearth_h3_ecs.parquet"
    mirror_path = event_dir / "benchmark_h3.parquet"
    alpha_raw = pd.read_parquet(alpha_path)
    if "h3_index" not in alpha_raw:
        raise ValueError(f"{alpha_path} is missing h3_index")
    indices = alpha_raw["h3_index"].astype(str).tolist()
    alpha = _validate_h3_frame(alpha_raw, indices, label="AlphaEarth")
    mirror = _validate_h3_frame(
        pd.read_parquet(mirror_path),
        indices,
        label="Earth Engine USDM mirror",
    )

    resolutions = sorted({h3.get_resolution(index) for index in indices})
    if len(resolutions) != 1:
        raise ValueError(f"Mixed H3 resolutions in {alpha_path}: {resolutions}")

    hexagons = gpd.GeoDataFrame(
        {"h3_index": indices},
        geometry=[_h3_polygon(index) for index in indices],
        crs="EPSG:4326",
    )
    aoi = gpd.read_file(args.aoi).to_crs(4326).geometry.make_valid().union_all()
    support = hexagons.copy()
    support.geometry = support.geometry.intersection(aoi)
    support = support.to_crs(5070)

    pre_score, pre_dates = _season_scores(
        args.pre_archive,
        args.pre_year,
        support,
    )
    post_score, post_dates = _season_scores(
        args.post_archive,
        args.post_year,
        support,
    )
    official = pd.DataFrame(
        {
            "h3_index": indices,
            "usdm_pre_season_mean": pre_score,
            "usdm_post_season_mean": post_score,
            "benchmark_mean_official": np.abs(post_score - pre_score),
        }
    )
    official = _validate_h3_frame(official, indices, label="Official NDMC vectors")
    output_path = event_dir / "usdm_official_h3.parquet"
    official.to_parquet(output_path, index=False)

    comparison = official.merge(mirror, on="h3_index", validate="one_to_one")
    x = comparison["benchmark_mean_official"].to_numpy(dtype=float)
    y = comparison["benchmark_mean"].to_numpy(dtype=float)
    absolute_error = np.abs(x - y)

    official_for_join = official.copy()
    official_for_join["benchmark_mean"] = official_for_join[
        "benchmark_mean_official"
    ]
    official_for_join["reference_metric_source"] = "official_ndmc_vectors"
    joined_official = alpha.merge(
        official_for_join,
        on="h3_index",
        validate="one_to_one",
    )
    if not (
        np.isfinite(joined_official["ecs_mean"]).all()
        and np.isfinite(joined_official["benchmark_mean"]).all()
    ):
        raise ValueError("Official joined metric contains non-finite ECS or USDM values")
    joined_official_path = event_dir / "joined_official.parquet"
    joined_official.to_parquet(joined_official_path, index=False)
    official_qa = run_qa(
        joined_official_path,
        event_dir,
        artifact_prefix="official",
        reference_label="official NDMC vectors",
    )

    mirror_joined = alpha.merge(mirror, on="h3_index", validate="one_to_one")
    official_vs_ecs_pearson = float(
        pearsonr(
            joined_official["benchmark_mean"],
            joined_official["ecs_mean"],
        ).statistic
    )
    official_vs_ecs_spearman = float(
        spearmanr(
            joined_official["benchmark_mean"],
            joined_official["ecs_mean"],
        ).statistic
    )
    mirror_vs_ecs_pearson = float(
        pearsonr(
            mirror_joined["benchmark_mean"],
            mirror_joined["ecs_mean"],
        ).statistic
    )
    mirror_vs_ecs_spearman = float(
        spearmanr(
            mirror_joined["benchmark_mean"],
            mirror_joined["ecs_mean"],
        ).statistic
    )

    summary = {
        "event": args.event,
        "pre_year": args.pre_year,
        "post_year": args.post_year,
        "season": "April 1 through October 31",
        "score_mapping": "None=0, D0=1, D1=2, D2=3, D3=4, D4=5",
        "weekly_map_count_pre": len(pre_dates),
        "weekly_map_count_post": len(post_dates),
        "expected_weekly_map_count_pre": expected_weekly_map_count(args.pre_year),
        "expected_weekly_map_count_post": expected_weekly_map_count(args.post_year),
        "weekly_release_dates_pre": pre_dates,
        "weekly_release_dates_post": post_dates,
        "h3_cells": len(comparison),
        "h3_coverage": {
            "alphaearth_cells": len(alpha),
            "mirror_cells": len(mirror),
            "official_cells": len(official),
            "official_joined_cells": len(joined_official),
            "all_unique": all(
                frame["h3_index"].nunique() == len(frame)
                for frame in (alpha, mirror, official, joined_official)
            ),
            "sets_equal": (
                set(alpha["h3_index"])
                == set(mirror["h3_index"])
                == set(official["h3_index"])
                == set(joined_official["h3_index"])
            ),
            "resolution": resolutions[0],
        },
        "official_vs_mirror_pearson": float(pearsonr(x, y).statistic),
        "official_vs_mirror_spearman": float(spearmanr(x, y).statistic),
        "official_vs_mirror_mae": float(absolute_error.mean()),
        "official_vs_mirror_max_abs_error": float(absolute_error.max()),
        "official_vs_ecs_pearson": official_vs_ecs_pearson,
        "official_vs_ecs_spearman": official_vs_ecs_spearman,
        "mirror_vs_ecs_pearson": mirror_vs_ecs_pearson,
        "mirror_vs_ecs_spearman": mirror_vs_ecs_spearman,
        "official_qa_pearson": official_qa["corr_pearson_benchmark_vs_ecs"],
        "official_qa_spearman": official_qa["corr_spearman_benchmark_vs_ecs"],
        "publication_reference": "official_ndmc_vectors",
        "mirror_role": "implementation_crosscheck_only",
        "official_h3_artifact": str(
            recorded_event_dir / "usdm_official_h3.parquet"
        ),
        "official_join_artifact": str(
            recorded_event_dir / "joined_official.parquet"
        ),
        "mirror_h3_artifact": str(recorded_event_dir / "benchmark_h3.parquet"),
        "pre_archive": str(args.pre_archive),
        "pre_archive_sha256": _sha256(args.pre_archive),
        "post_archive": str(args.post_archive),
        "post_archive_sha256": _sha256(args.post_archive),
        "aoi": str(args.aoi),
        "aoi_sha256": _sha256(args.aoi),
        "alphaearth_h3_ecs_sha256": _sha256(alpha_path),
        "benchmark_h3_mirror_sha256": _sha256(mirror_path),
        "usdm_official_h3_sha256": _sha256(output_path),
        "joined_official_sha256": _sha256(joined_official_path),
        "official_archive_url_pattern": (
            "https://droughtmonitor.unl.edu/data/shapefiles_m/"
            "{YEAR}_USDM_M.zip"
        ),
        "citation": (
            "National Drought Mitigation Center, U.S. Department of "
            "Agriculture, and National Oceanic and Atmospheric Administration"
        ),
    }
    evaluation = evaluate_official_validation(
        summary,
        expected_event=args.event,
        expected_pre_year=args.pre_year,
        expected_post_year=args.post_year,
        expected_h3_cells=len(alpha),
        expected_h3_resolution=USDM_VALIDATION_CRITERIA[
            "required_h3_resolution"
        ],
    )
    summary["validation_criteria"] = evaluation["criteria"]
    summary["validation_checks"] = evaluation["checks"]
    summary["validation_failures"] = evaluation["failures"]
    summary["official_reference_complete"] = evaluation[
        "official_reference_complete"
    ]
    summary["mirror_agreement_passed"] = evaluation[
        "mirror_agreement_passed"
    ]
    summary["core_validation_failures"] = evaluation["core_failures"]
    summary["mirror_agreement_failures"] = evaluation[
        "agreement_failures"
    ]
    if not evaluation["official_reference_complete"]:
        summary["validation_status"] = "official_reference_incomplete"
    elif evaluation["mirror_agreement_passed"]:
        summary["validation_status"] = "validated_official_reference"
    else:
        summary[
            "validation_status"
        ] = "official_reference_complete_mirror_disagreement"

    summary_path = event_dir / "usdm_official_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {joined_official_path}")
    print(f"Wrote {summary_path}")
    return 0 if evaluation["official_reference_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
