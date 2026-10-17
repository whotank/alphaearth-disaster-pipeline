#!/usr/bin/env python3
"""Export standalone before/after/ECS/reference panels for fire/lava cases."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import ee
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import disaster_embedding_pipeline as pipeline
import export_development_visual_panels as panels
import fire_change_cohort as cohort
import fire_lava_ecs_rectangle as rectangles

LOGGER = logging.getLogger("export_fire_lava_visual_pair")
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "reo2_fire_lava_visual_pair"
HELDOUT_ROOT = Path(
    os.environ.get(
        "MTBS_HELDOUT_ROOT",
        ROOT / "studies" / "mtbs_heldout_20260817",
    )
)
WALKER_FIRE_ID = "CA4005312066920190904"
SHOVEL_CREEK_FIRE_ID = "ID4601211691120210707"
RHEA_FIRE_ID = "OK3585009921020180412"
WOOLSEY_FIRE_ID = "CA3424011870020181108"
CASES = (
    {
        "event_key": "thomas_fire_2017",
        "family": "fire",
        "source_kind": "pipeline_join",
        "selection_note": (
            "Quintessential fire example: iconic event, clean pre/post source "
            "coverage, and strong AlphaEarth-to-mapped-reference association."
        ),
        "context_window": "June-September",
        "reference": "MODIS MCD64A1 annual BurnDate",
    },
    {
        "event_key": "la_palma_lava_2021",
        "family": "lava",
        "source_kind": "pipeline_join",
        "selection_note": "Existing manuscript lava example.",
        "context_window": "calendar year",
        "reference": "Copernicus EMS lava-emplacement extent",
    },
    {
        "event_key": "walker_fire_2019",
        "fire_id": WALKER_FIRE_ID,
        "expected_name": "WALKER",
        "event_year": 2019,
        "label": "Walker Fire (California, 2019; held-out MTBS reserve)",
        "family": "fire",
        "source_kind": "heldout_mtbs_sealed",
        "selection_note": (
            "Prespecified reserve replacing Four County before AlphaEarth access."
        ),
        "context_window": "full calendar years; SCL-masked annual median",
        "reference": (
            "Official MTBS mapped moderate/high fraction: classes 3 or 4 among "
            "valid classes 1-5"
        ),
    },
    {
        "event_key": "shovel_creek_fire_2021",
        "fire_id": SHOVEL_CREEK_FIRE_ID,
        "expected_name": "SHOVEL CREEK",
        "event_year": 2021,
        "label": "Shovel Creek Fire (Idaho, 2021; held-out MTBS primary)",
        "family": "fire",
        "source_kind": "heldout_mtbs_sealed",
        "selection_note": "Prespecified primary fire in the frozen held-out cohort.",
        "context_window": "full calendar years; SCL-masked annual median",
        "reference": (
            "Official MTBS mapped moderate/high fraction: classes 3 or 4 among "
            "valid classes 1-5"
        ),
    },
    {
        "event_key": "kilauea_lava_2018",
        "output_key": "kil",
        "label": "Kīlauea lower East Rift Zone lava emplacement (USGS, 2018)",
        "family": "lava",
        "source_kind": "pipeline_join",
        "selection_note": "Predefined 2018 lava development case.",
        "context_window": "full calendar years",
        "reference": "USGS mapped lava-emplacement footprint",
        "post_window": {
            "variant_key": "post_window_2018_2020",
            "pre_year": 2018,
            "post_year": 2020,
        },
    },
    {
        "event_key": "rhea_fire_2018",
        "output_key": "rh",
        "fire_id": RHEA_FIRE_ID,
        "expected_name": "RHEA",
        "event_year": 2018,
        "label": "Rhea Fire (Oklahoma, United States, 2018; held-out MTBS)",
        "family": "fire",
        "source_kind": "heldout_mtbs_sealed",
        "selection_note": "Prespecified primary fire in the frozen held-out cohort.",
        "context_window": "full calendar years; SCL-masked annual median",
        "reference": (
            "Official MTBS mapped moderate/high fraction: classes 3 or 4 among "
            "valid classes 1-5"
        ),
        "post_window": {
            "variant_key": "post_window_2018_2020",
            "pre_year": 2018,
            "post_year": 2020,
        },
    },
    {
        "event_key": "woolsey_fire_2018",
        "output_key": "woo",
        "fire_id": WOOLSEY_FIRE_ID,
        "expected_name": "WOOLSEY",
        "event_year": 2018,
        "label": "Woolsey Fire (California, United States, 2018; held-out MTBS)",
        "family": "fire",
        "source_kind": "heldout_mtbs_sealed",
        "selection_note": "Prespecified primary fire in the frozen held-out cohort.",
        "context_window": "full calendar years; SCL-masked annual median",
        "reference": (
            "Official MTBS mapped moderate/high fraction: classes 3 or 4 among "
            "valid classes 1-5"
        ),
        "post_window": {
            "variant_key": "post_window_2018_2020",
            "pre_year": 2018,
            "post_year": 2020,
        },
    },
)


def _heldout_paths(fire_id: str) -> dict[str, Path]:
    event_root = HELDOUT_ROOT / "raw" / "mtbs" / fire_id
    return {
        "sealed": HELDOUT_ROOT / "sealed" / f"{fire_id}.parquet",
        "feature_metadata": (
            HELDOUT_ROOT / "outputs" / "feature_extraction" / f"{fire_id}.json"
        ),
        "attempt_metadata": (HELDOUT_ROOT / "outputs" / "attempts" / f"{fire_id}.json"),
        "sealed_manifest": HELDOUT_ROOT / "outputs" / "sealed_manifest.json",
        "analysis_metrics": (
            HELDOUT_ROOT / "outputs" / "analysis" / "per_event_metrics.csv"
        ),
        "official_boundary": event_root / "official_boundary.geojson",
        "severity_raster": event_root / "severity.tif",
    }


def _walker_paths() -> dict[str, Path]:
    """Return Walker source paths for backward-compatible callers."""

    return _heldout_paths(WALKER_FIRE_ID)


def _shovel_creek_paths() -> dict[str, Path]:
    return _heldout_paths(SHOVEL_CREEK_FIRE_ID)


def heldout_source(
    *,
    fire_id: str,
    expected_name: str,
    expected_event_year: int,
    label: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify and describe one sealed held-out ECS/MTBS source."""

    paths = _heldout_paths(fire_id)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise panels.VisualPanelError(
            f"{expected_name}: missing held-out sources: {missing}"
        )
    feature = json.loads(paths["feature_metadata"].read_text(encoding="utf-8"))
    attempt = json.loads(paths["attempt_metadata"].read_text(encoding="utf-8"))
    sealed_manifest = json.loads(paths["sealed_manifest"].read_text(encoding="utf-8"))
    manifest_record = next(
        (event for event in sealed_manifest["events"] if event["fire_id"] == fire_id),
        None,
    )
    if manifest_record is None:
        raise panels.VisualPanelError(
            f"{expected_name} is absent from the sealed manifest"
        )
    sealed_sha = panels.sha256_file(paths["sealed"])
    if (
        sealed_sha != feature["sealed_sha256"]
        or sealed_sha != manifest_record["sealed_sha256"]
    ):
        raise panels.VisualPanelError(f"{expected_name} sealed artifact hash changed")
    frame = pd.read_parquet(
        paths["sealed"],
        columns=[
            "fire_id",
            "fire_name",
            "event_year",
            "pre_year",
            "post_year",
            "h3_index",
            "ecs",
            "mtbs_reference",
        ],
    )
    if (
        len(frame) != int(feature["rows"])
        or frame["h3_index"].duplicated().any()
        or set(frame["fire_id"].astype(str)) != {fire_id}
        or set(frame["fire_name"].astype(str)) != {expected_name}
        or set(frame["event_year"].astype(int)) != {expected_event_year}
    ):
        raise panels.VisualPanelError(f"{expected_name} sealed frame identity changed")
    if not (
        frame["ecs"].notna().all()
        and frame["mtbs_reference"].notna().all()
        and frame["ecs"].nunique() > 1
        and frame["mtbs_reference"].nunique() > 1
    ):
        raise panels.VisualPanelError(
            f"{expected_name} ECS/reference masks are invalid"
        )
    metrics = pd.read_csv(paths["analysis_metrics"])
    metric = metrics.loc[metrics["fire_id"].astype(str) == fire_id]
    if len(metric) != 1:
        raise panels.VisualPanelError(
            f"{expected_name} held-out metric row is not unique"
        )
    replacement_for = metric["replacement_for"].iloc[0]
    config = {
        "label": label,
        "pre_year": int(frame["pre_year"].iloc[0]),
        "post_year": int(frame["post_year"].iloc[0]),
    }
    source = {
        "study_root": str(HELDOUT_ROOT),
        "sealed": {
            "path": panels._display_path(paths["sealed"]),
            "sha256": sealed_sha,
            "rows": len(frame),
            "h3_sha256": feature["h3_sha256"],
            "ecs_column": "ecs",
            "reference_column": "mtbs_reference",
        },
        "reference_definition": (
            "Per-H3 fraction of official MTBS pixels in classes 3 or 4 among "
            "valid classes 1-5; center-in-polygon raster reduction."
        ),
        "source_artifacts": {
            name: {
                "path": panels._display_path(path),
                "sha256": panels.sha256_file(path),
            }
            for name, path in paths.items()
            if name != "sealed"
        },
        "heldout_result": {
            "rho_ecs_reference": float(metric["rho_ecs_reference"].iloc[0]),
            "replacement": bool(metric["replacement"].iloc[0]),
            "replacement_for": (
                None if pd.isna(replacement_for) else str(replacement_for)
            ),
        },
        "sentinel_context_binding": {
            "attempt_metadata_sha256": panels.sha256_file(paths["attempt_metadata"]),
            "collection": attempt["sentinel2_metadata"]["collection"],
            "scene_cloud_filter_lte_pct": attempt["sentinel2_metadata"][
                "maximum_scene_cloud_percent"
            ],
            "scl_excluded_classes": attempt["sentinel2_metadata"][
                "scl_excluded_classes"
            ],
            "pre_scene_count": attempt["sentinel2_metadata"]["pre_scene_count"],
            "post_scene_count": attempt["sentinel2_metadata"]["post_scene_count"],
        },
    }
    return config, source


def walker_source() -> tuple[dict[str, object], dict[str, object]]:
    """Verify and describe Walker's sealed held-out ECS/MTBS source."""

    return heldout_source(
        fire_id=WALKER_FIRE_ID,
        expected_name="WALKER",
        expected_event_year=2019,
        label="Walker Fire (California, 2019; held-out MTBS reserve)",
    )


def shovel_creek_source() -> tuple[dict[str, object], dict[str, object]]:
    """Verify and describe Shovel Creek's sealed held-out ECS/MTBS source."""

    return heldout_source(
        fire_id=SHOVEL_CREEK_FIRE_ID,
        expected_name="SHOVEL CREEK",
        expected_event_year=2021,
        label="Shovel Creek Fire (Idaho, 2021; held-out MTBS primary)",
    )


def _heldout_context(
    output_dir: Path,
    *,
    fire_id: str,
    config: dict[str, object],
    source: dict[str, object],
    ee_project: str,
    refresh: bool,
) -> dict[str, object]:
    metadata_path = output_dir / "context_metadata.json"
    before_path = output_dir / "before.png"
    after_path = output_dir / "after.png"
    fingerprint = cohort.canonical_sha256(
        {
            "source": source["sentinel_context_binding"],
            "pre_year": config["pre_year"],
            "post_year": config["post_year"],
            "scale_m": 30,
            "composite": "SCL-masked full-calendar median",
            "display": {
                "bands": ["B4", "B3", "B2"],
                "min": [100, 100, 100],
                "max": [4000, 4000, 4000],
                "gamma": [1.05, 1.05, 1.0],
            },
        }
    )
    if metadata_path.is_file() and before_path.is_file() and after_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hashes_match = (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("before_sha256") == panels.sha256_file(before_path)
            and metadata.get("after_sha256") == panels.sha256_file(after_path)
        )
        if hashes_match and not refresh:
            return metadata
    ee.Initialize(project=ee_project)
    used_scale = panels.download_context(
        config=config,
        join_path=_heldout_paths(fire_id)["sealed"],
        output_dir=output_dir,
        scale_m=30,
        max_scenes=0,
        composite="median",
    )
    metadata = {
        "fingerprint": fingerprint,
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "composite": "SCL-masked full-calendar median",
        "pre_year": config["pre_year"],
        "post_year": config["post_year"],
        "requested_scale_m": 30,
        "used_scale_m": used_scale,
        "before_sha256": panels.sha256_file(before_path),
        "after_sha256": panels.sha256_file(after_path),
        "source_attempt_metadata": source["sentinel_context_binding"],
    }
    cohort.atomic_write_json(metadata_path, metadata)
    return metadata


def export_cases(
    output_root: Path,
    *,
    ee_project: str,
    case_keys: set[str] | None = None,
    refresh_walker_context: bool = False,
    refresh_heldout_context: bool = False,
    refresh_ecs_rectangles: bool = False,
    ecs_batch_size: int = rectangles.DEFAULT_BATCH_SIZE,
    ecs_tile_scale: int = rectangles.DEFAULT_TILE_SCALE,
    ecs_workers: int = rectangles.DEFAULT_EE_WORKERS,
) -> dict[str, object]:
    configurations = pipeline._event_configurations()
    known_case_keys = {str(item["event_key"]) for item in CASES}
    unknown_case_keys = sorted((case_keys or set()) - known_case_keys)
    if unknown_case_keys:
        raise panels.VisualPanelError(f"Unknown visual cases: {unknown_case_keys}")
    selected_cases = tuple(
        item
        for item in CASES
        if case_keys is None or str(item["event_key"]) in case_keys
    )
    records: list[dict[str, object]] = []
    for item in selected_cases:
        event_key = str(item["event_key"])
        output_key = str(item.get("output_key", event_key))
        event_output = output_root / output_key
        event_output.mkdir(parents=True, exist_ok=True)
        if item["source_kind"] == "heldout_mtbs_sealed":
            fire_id = str(item["fire_id"])
            config, source = heldout_source(
                fire_id=fire_id,
                expected_name=str(item["expected_name"]),
                expected_event_year=int(item["event_year"]),
                label=str(item["label"]),
            )
            context = _heldout_context(
                event_output,
                fire_id=fire_id,
                config=config,
                source=source,
                ee_project=ee_project,
                refresh=(
                    refresh_heldout_context
                    or (refresh_walker_context and fire_id == WALKER_FIRE_ID)
                ),
            )
            join_source = _heldout_paths(fire_id)["sealed"]
            panels.render_masks(
                join_source,
                event_output,
                ecs_column="ecs",
                reference_column="mtbs_reference",
            )
            source_record = source
            context_record = context
        else:
            config = dict(configurations[event_key])
            if "label" in item:
                config["label"] = str(item["label"])
            event_source = ROOT / "outputs" / event_key
            context_source = event_source / "gis_reference_raster_2x2.png"
            join_source = event_source / "joined_inner.parquet"
            if not context_source.is_file() or not join_source.is_file():
                raise panels.VisualPanelError(
                    f"{event_key}: missing 2x2 context or publication join"
                )
            before, after = panels.context_tiles(mpimg.imread(context_source))
            panels._save_rgb(before, event_output / "before.png")
            panels._save_rgb(after, event_output / "after.png")
            panels.render_masks(join_source, event_output)
            source_record = {
                "publication_join": {
                    "path": panels._display_path(join_source),
                    "sha256": panels.sha256_file(join_source),
                    "ecs_column": "ecs_mean",
                    "reference_column": "benchmark_mean",
                }
            }
            context_record = {
                "method": "cropped_existing_gis_2x2",
                "path": panels._display_path(context_source),
                "sha256": panels.sha256_file(context_source),
            }
        rectangle_record = rectangles.export_rectangle(
            event_key=event_key,
            config=config,
            join_path=join_source,
            source_kind=str(item["source_kind"]),
            output_dir=event_output,
            ee_project=ee_project,
            batch_size=ecs_batch_size,
            tile_scale=ecs_tile_scale,
            ee_workers=ecs_workers,
            force=refresh_ecs_rectangles,
        )
        temporal_variants: list[dict[str, object]] = []
        post_window = item.get("post_window")
        if isinstance(post_window, dict):
            temporal_variants.append(
                rectangles.export_temporal_variant(
                    event_key=event_key,
                    variant_key=str(post_window["variant_key"]),
                    pre_year=int(post_window["pre_year"]),
                    post_year=int(post_window["post_year"]),
                    config=config,
                    join_path=join_source,
                    source_kind=str(item["source_kind"]),
                    output_dir=event_output,
                    ee_project=ee_project,
                    batch_size=ecs_batch_size,
                    tile_scale=ecs_tile_scale,
                    ee_workers=ecs_workers,
                    force=refresh_ecs_rectangles,
                )
            )
        panels.build_contact_sheet(
            event_output,
            label=str(config["label"]),
            pre_year=int(config["pre_year"]),
            post_year=int(config["post_year"]),
            ecs_title="AlphaEarth ECS (expanded; cyan = reference circle)",
        )

        files: dict[str, object] = {}
        for name in (
            *panels.PANEL_FILES,
            rectangles.ORIGINAL_MASK_NAME,
            "contact_sheet.png",
            *(
                str(variant["render"]["path"]).rsplit("/", maxsplit=1)[-1]
                for variant in temporal_variants
            ),
        ):
            path = event_output / name
            with Image.open(path) as image:
                dimensions = [int(image.width), int(image.height)]
            files[name] = {
                "path": panels._display_path(path),
                "sha256": panels.sha256_file(path),
                "dimensions_px": dimensions,
            }
        records.append(
            {
                **item,
                "output_key": output_key,
                "label": config["label"],
                "pre_year": int(config["pre_year"]),
                "post_year": int(config["post_year"]),
                "context_source": context_record,
                "data_source": source_record,
                "ecs_rectangle": rectangle_record,
                "temporal_variants": temporal_variants,
                "mask_normalization": {
                    "ecs": (
                        "expanded-rectangle IQR z-score followed by linear min-max "
                        "to [0,1]"
                    ),
                    "reference": (
                        "original-support IQR z-score followed by linear min-max "
                        "to [0,1]"
                    ),
                },
                "files": files,
            }
        )

    manifest_path = output_root / "manifest.json"
    merged_records = records
    if case_keys is not None and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_records = {
            str(record["event_key"]): record for record in existing.get("events", [])
        }
        existing_records.update(
            {str(record["event_key"]): record for record in records}
        )
        merged_records = [
            existing_records[str(item["event_key"])]
            for item in CASES
            if str(item["event_key"]) in existing_records
        ]
    manifest: dict[str, object] = {
        "schema": "reo2_fire_lava_visual_cases_v4",
        "created_at_utc": cohort.utc_now(),
        "event_count": len(merged_records),
        "events": merged_records,
    }
    manifest["payload_sha256"] = cohort.canonical_sha256(manifest)
    cohort.atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ee-project", default="project0-256023")
    parser.add_argument("--refresh-walker-context", action="store_true")
    parser.add_argument("--refresh-heldout-context", action="store_true")
    parser.add_argument("--refresh-ecs-rectangles", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=[str(item["event_key"]) for item in CASES],
        help="Export only selected case(s) and merge them into the manifest.",
    )
    parser.add_argument(
        "--ecs-batch-size",
        type=int,
        default=rectangles.DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--ecs-tile-scale",
        type=int,
        default=rectangles.DEFAULT_TILE_SCALE,
    )
    parser.add_argument(
        "--ecs-workers",
        type=int,
        choices=range(1, rectangles.MAX_EE_WORKERS + 1),
        default=rectangles.DEFAULT_EE_WORKERS,
        help="Concurrent Earth Engine requests (1-8).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    output_root = args.output_dir.expanduser().resolve()
    if not output_root.is_relative_to((ROOT / "outputs").resolve()):
        raise panels.VisualPanelError("Output directory must remain under outputs/")
    manifest = export_cases(
        output_root,
        ee_project=args.ee_project,
        case_keys=set(args.case) if args.case else None,
        refresh_walker_context=args.refresh_walker_context,
        refresh_heldout_context=args.refresh_heldout_context,
        refresh_ecs_rectangles=args.refresh_ecs_rectangles,
        ecs_batch_size=args.ecs_batch_size,
        ecs_tile_scale=args.ecs_tile_scale,
        ecs_workers=args.ecs_workers,
    )
    LOGGER.info(
        "Wrote %d fire/lava visual sets under %s",
        manifest["event_count"],
        output_root,
    )
    print(
        json.dumps(
            {"output_dir": str(output_root), "event_count": manifest["event_count"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
