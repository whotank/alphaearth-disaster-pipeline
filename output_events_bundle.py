#!/usr/bin/env python3
"""
Bundle helpers for multi-event outputs:

1) **manifest** — Write ``outputs/OUTPUTS_EVENTS_MANIFEST.json`` and ``.md`` with
   AOI center (lon/lat), buffer, pre/post embedding years, benchmark calendar year,
   how years relate (``benchmark_year_matches_post``), and merged ``integrity_report.json``.

2) **pipeline** — Run ``disaster_embedding_pipeline.py`` once per event (Earth Engine).

3) **figures** — Run ``reference_imagery``, ``heatmap_reference_overlay``, and
   ``gis_raster_reference_figure`` for the same event list (requires EE + joined parquets).

4) **metrics** — Run ``fire_analysis.py metrics --discover`` (no EE) to write
   ``all_events_metrics_summary.csv`` under ``--output-dir``, or pass ``--events`` for a comma list.

Default event list matches the paper bundle (camp, dixie, fort, georgia, marshall,
mosquito, pacific NW forest loss, pakistan flood, quebec, thomas, tubbs).

Examples (activate your project venv first; set ``EE_PROJECT_ID`` or pass ``--ee-project``).
Use your **real** Cloud **Project ID** (Console → IAM & Admin → Settings), not a placeholder string::

  export EE_PROJECT_ID=my-lab-project-123456   # or: export GOOGLE_CLOUD_PROJECT=...

  # Table-style metadata: centers, pre/post years, benchmark year logic, integrity JSON merge
  python output_events_bundle.py manifest --output-dir outputs

  # Same, but every folder under outputs/ that has integrity_report and/or joined_inner
  python output_events_bundle.py manifest --output-dir outputs --discover

  # Recompute embeddings + benchmarks + heatmaps only (no S2 GIS figure scripts)
  python output_events_bundle.py pipeline --output-dir outputs

  # S2 reference PNGs, heatmap-on-reference, and GIS 2x2 for the default bundle (needs EE)
  python output_events_bundle.py figures --output-dir outputs --ee-project my-lab-project-123456

  # Pipeline then figures in one go
  python output_events_bundle.py all --output-dir outputs --ee-project my-lab-project-123456

  # Join/correlation metrics CSV for every output folder (no Earth Engine)
  python output_events_bundle.py metrics --output-dir outputs

  # Custom list (e.g. add Brazil floods)
  python output_events_bundle.py figures --output-dir outputs --ee-project my-lab-project-123456 \\
      brazil_floods_2024 camp_fire_2018

Temporal / year logic (see also module docstrings in ``disaster_embedding_pipeline.py``):

- **AlphaEarth ECS:** annual mosaics for ``pre_year`` and ``post_year`` (full calendar years in EE).
- **MODIS MCD64A1:** ``benchmark.metric_year`` is the calendar year of the BurnDate annual max layer;
  usually ``metric_year == post_year`` unless ``benchmark_year_matches_post: false`` (legacy fires
  where embeddings use the first valid pair after AlphaEarth’s 2017 start while MODIS stays on the
  true burn year).
- **Hansen GFC:** ``metric_year`` matches calendar loss in ``lossyear`` (see pipeline docstring).
- **JRC GSW:** ``occurrence`` is a static long-term surface-water layer. ``metric_year`` is metadata
  only and does not make this placeholder an event-specific flood extent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _ensure_writable_mplconfigdir() -> None:
    """If MPLCONFIGDIR is missing or not writable (e.g. tutorial placeholder), use repo ``.mplconfig/``."""
    fallback = Path(__file__).resolve().parent / ".mplconfig"
    cand = os.environ.get("MPLCONFIGDIR", "").strip()
    ok = False
    if cand:
        try:
            Path(cand).mkdir(parents=True, exist_ok=True)
            probe = Path(cand) / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            ok = True
        except OSError:
            ok = False
    if not ok:
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback)


_ensure_writable_mplconfigdir()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# User-requested bundle (canonical event keys under ``outputs/<key>/``).
DEFAULT_BUNDLE_EVENTS: tuple[str, ...] = (
    "camp_fire_2018",
    "dixie_fire_2021",
    "fort_mcmurray_fire_2016",
    "georgia_forest_loss_2021",
    "marshall_fire_2021",
    "mosquito_fire_2022",
    "pacific_nw_forest_loss_2021",
    "pakistan_floods_2022",
    "quebec_fires_2023",
    "thomas_fire_2017",
    "tubbs_fire_2017",
)

TEMPORAL_LOGIC_MD = """## How pre/post and benchmark years are chosen

| Layer | Rule |
|-------|------|
| **AlphaEarth (ECS)** | ``pre_year`` and ``post_year`` are **full calendar-year** annual embedding mosaics (Jan 1 → next Jan 1 in Earth Engine). AlphaEarth V1 annual starts **2017**; do not set ``pre_year`` to 2015/2016. |
| **MODIS MCD64A1** (fires) | ``benchmark.metric_year`` is the **calendar year** of the BurnDate annual-max raster. Normally **``metric_year = post_year``**. If **``benchmark_year_matches_post: false``**, MODIS stays on the **true burn year** while ECS uses ``pre_year``→``post_year`` (first valid pair after 2017). |
| **Hansen GFC** (forest loss) | ``metric_year`` is the calendar year; ``lossyear`` codes map to calendar year (see pipeline docstring). |
| **USDM** (drought) | The publication metric is rebuilt from official NDMC weekly vectors for every Tuesday from April 1 through October 31. The Earth Engine mirror is retained only as an implementation cross-check. |
| **JRC GSW** (floods) | ``occurrence`` is a static long-term water layer used as a pipeline placeholder; ``metric_year`` is metadata only, not an event filter. |

**AOI:** geodesic disk centered at ``(center_lon, center_lat)`` with radius ``processing_buffer_km``.
**H3:** resolution ``h3_resolution``, capped at ``max_h3_cells`` cells from the AlphaEarth sample.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _cfg_public_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if k == "benchmark":
            b = v
            out["benchmark"] = {
                "type": getattr(b, "type", str(b)),
                "metric_year": int(getattr(b, "metric_year", b)),
                "reference_pre_year": getattr(b, "reference_pre_year", None),
                "geojson_path": getattr(b, "geojson_path", None),
                "valid_geojson_path": getattr(b, "valid_geojson_path", None),
                "exclude_worldcover_water": bool(
                    getattr(b, "exclude_worldcover_water", False)
                ),
                "nominal_scale_m": getattr(b, "nominal_scale_m", None),
            }
        else:
            out[k] = v
    return out


def _joined_centroid(joined_path: Path) -> dict[str, float] | None:
    try:
        import h3
        import pandas as pd
    except ImportError:
        return None
    if not joined_path.is_file():
        return None
    df = pd.read_parquet(joined_path, columns=["h3_index"])
    if df.empty:
        return None
    lat_acc, lon_acc, n = 0.0, 0.0, 0
    for h in df["h3_index"].astype(str).unique():
        try:
            lat, lon = h3.cell_to_latlng(h)
        except Exception:
            continue
        lat_acc += lat
        lon_acc += lon
        n += 1
    if not n:
        return None
    return {"centroid_lat_joined_hexes": lat_acc / n, "centroid_lon_joined_hexes": lon_acc / n}


def _discover_event_dirs(out_root: Path, *, require_joined: bool) -> list[str]:
    names: list[str] = []
    for p in out_root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        has_join = (p / "joined_inner.parquet").is_file()
        has_ir = (p / "integrity_report.json").is_file()
        if require_joined and not has_join:
            continue
        if not require_joined and not (has_join or has_ir):
            continue
        names.append(p.name)
    return sorted(names)


def cmd_manifest(ns: argparse.Namespace) -> int:
    from disaster_embedding_pipeline import _event_configurations

    out_root: Path = ns.output_dir.resolve()
    cfgs = _event_configurations()
    if ns.discover:
        events = _discover_event_dirs(out_root, require_joined=False)
    else:
        events = list(ns.events) if ns.events else list(DEFAULT_BUNDLE_EVENTS)

    rows: list[dict[str, Any]] = []
    for key in events:
        row: dict[str, Any] = {"event_key": key}
        ir_path = out_root / key / "integrity_report.json"
        jp = out_root / key / "joined_inner.parquet"
        if ir_path.is_file():
            row["integrity_report"] = json.loads(ir_path.read_text(encoding="utf-8"))
        official_validation_path = (
            out_root / key / "usdm_official_validation.json"
        )
        if official_validation_path.is_file():
            row["official_reference_validation"] = json.loads(
                official_validation_path.read_text(encoding="utf-8")
            )
        if key in cfgs:
            row["pipeline_preset"] = _cfg_public_dict(cfgs[key])
        else:
            row["pipeline_preset"] = None
            row["preset_note"] = "Event directory exists but key is not in disaster_embedding_pipeline._event_configurations()"
        cen = _joined_centroid(jp)
        if cen:
            row["from_joined_inner_parquet"] = cen
        row["artifacts"] = {
            "joined_inner.parquet": jp.is_file(),
            "integrity_report.json": ir_path.is_file(),
            "heatmap_join_side_by_side.png": (out_root / key / "heatmap_join_side_by_side.png").is_file(),
            "heatmap_join_side_by_side_basemap.png": (
                out_root / key / "heatmap_join_side_by_side_basemap.png"
            ).is_file(),
            "gis_reference_raster_2x2.png": (out_root / key / "gis_reference_raster_2x2.png").is_file(),
            "gis_temporal_alignment.txt": (out_root / key / "gis_temporal_alignment.txt").is_file(),
            "sentinel2_reference_pre_rgb.png": (out_root / key / "sentinel2_reference_pre_rgb.png").is_file(),
            "heatmap_join_on_reference_pre_post.png": (
                out_root / key / "heatmap_join_on_reference_pre_post.png"
            ).is_file(),
            "usdm_official_h3.parquet": (
                out_root / key / "usdm_official_h3.parquet"
            ).is_file(),
            "joined_official.parquet": (
                out_root / key / "joined_official.parquet"
            ).is_file(),
            "usdm_official_validation.json": official_validation_path.is_file(),
            "qa_official_numeric_summary.json": (
                out_root / key / "qa_official_numeric_summary.json"
            ).is_file(),
        }
        rows.append(row)

    json_path = out_root / "OUTPUTS_EVENTS_MANIFEST.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        "# Outputs event manifest",
        "",
        "Auto-generated by `python output_events_bundle.py manifest`.",
        "",
        TEMPORAL_LOGIC_MD,
        "",
        "## Per-event summary",
        "",
        "| Event | Center (preset) lon, lat | Buffer km | pre→post | Benchmark (type, year) | matches post | Official validation |",
        "|-------|---------------------------|-----------|----------|--------------------------|--------------|---------------------|",
    ]
    for r in rows:
        key = r["event_key"]
        pr = r.get("pipeline_preset") or {}
        lon = pr.get("center_lon", "")
        lat = pr.get("center_lat", "")
        buf = pr.get("processing_buffer_km", "")
        pre, post = pr.get("pre_year", ""), pr.get("post_year", "")
        b = pr.get("benchmark") or {}
        bt, my = b.get("type", ""), b.get("metric_year", "")
        mpost = pr.get("benchmark_year_matches_post", True)
        validation = r.get("official_reference_validation") or {}
        validation_status = validation.get("validation_status", "")
        md_lines.append(
            f"| `{key}` | {lon}, {lat} | {buf} | {pre}→{post} | "
            f"{bt}, {my} | {mpost} | {validation_status} |"
        )
    md_lines += ["", f"Full JSON: `{json_path.name}`", ""]
    md_path = out_root / "OUTPUTS_EVENTS_MANIFEST.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    return 0


def _events_list(ns: argparse.Namespace, out_root: Path, *, figures: bool) -> list[str]:
    if ns.discover:
        return _discover_event_dirs(out_root, require_joined=figures)
    return list(ns.events) if ns.events else list(DEFAULT_BUNDLE_EVENTS)


def _run_py(script_name: str, args: list[str], cwd: Path) -> None:
    script = cwd / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script), *args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def cmd_pipeline(ns: argparse.Namespace) -> int:
    root = _repo_root()
    out_root = ns.output_dir.resolve()
    events = _events_list(ns, out_root, figures=False)
    for ev in events:
        args = ["--event", ev, "--output-dir", str(out_root)]
        _run_py("disaster_embedding_pipeline.py", args, root)
    cmd_manifest(argparse.Namespace(output_dir=out_root, discover=True, events=[]))
    return 0


def cmd_figures(ns: argparse.Namespace) -> int:
    root = _repo_root()
    out_root = ns.output_dir.resolve()
    events = _events_list(ns, out_root, figures=True)
    if not events:
        logger.warning("No events with joined_inner.parquet; nothing to do.")
        return 1
    ev_csv = ",".join(events)
    ee_args: list[str] = []
    if ns.ee_project:
        ee_args = ["--ee-project", ns.ee_project.strip()]

    _run_py(
        "reference_imagery.py",
        ["--output-dir", str(out_root), "--events", ev_csv, *ee_args],
        root,
    )
    _run_py(
        "heatmap_reference_overlay.py",
        [
            "--output-dir",
            str(out_root),
            "--events",
            ev_csv,
            "--fetch-missing",
            *ee_args,
        ],
        root,
    )
    _run_py(
        "gis_raster_reference_figure.py",
        [
            "--output-dir",
            str(out_root),
            "--events",
            ev_csv,
            *ee_args,
        ],
        root,
    )
    cmd_manifest(argparse.Namespace(output_dir=out_root, discover=True, events=[]))
    return 0


def cmd_all(ns: argparse.Namespace) -> int:
    r = cmd_pipeline(ns)
    if r != 0:
        return r
    return cmd_figures(ns)


def cmd_metrics(ns: argparse.Namespace) -> int:
    root = _repo_root()
    out_root = ns.output_dir.resolve()
    parts = ["metrics", "--output-dir", str(out_root)]
    csv_arg = (getattr(ns, "metrics_events_csv", None) or "").strip()
    if csv_arg:
        parts.extend(["--events", csv_arg])
    else:
        parts.append("--discover")
    _run_py("fire_analysis.py", parts, root)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("manifest", help="Write OUTPUTS_EVENTS_MANIFEST.{json,md}")
    pm.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pm.add_argument(
        "--discover",
        action="store_true",
        help="Use all output subdirs that have integrity_report or joined_inner",
    )
    pm.add_argument(
        "events",
        nargs="*",
        metavar="EVENT",
        help="Event keys (default: built-in bundle)",
    )
    pm.set_defaults(func=cmd_manifest)

    pp = sub.add_parser("pipeline", help="Run disaster_embedding_pipeline for each event")
    pp.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pp.add_argument("--discover", action="store_true", help="Events = dirs with joined_inner (re-run)")
    pp.add_argument("events", nargs="*", metavar="EVENT")
    pp.set_defaults(func=cmd_pipeline)

    pf = sub.add_parser("figures", help="Reference PNGs, heatmap on reference, GIS 2x2")
    pf.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pf.add_argument(
        "--discover",
        action="store_true",
        help="Events = dirs that already have joined_inner.parquet",
    )
    pf.add_argument("events", nargs="*", metavar="EVENT")
    pf.add_argument(
        "--ee-project",
        type=str,
        default=None,
        help="GCP project id for Earth Engine (passed to child scripts)",
    )
    pf.set_defaults(func=cmd_figures)

    pa = sub.add_parser(
        "all",
        help="Run disaster_embedding_pipeline for each event, then figure exports (EE)",
    )
    pa.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pa.add_argument(
        "--discover",
        action="store_true",
        help="Discover events: pipeline uses dirs with integrity or joined; figures need joined",
    )
    pa.add_argument("events", nargs="*", metavar="EVENT")
    pa.add_argument(
        "--ee-project",
        type=str,
        default=None,
        help="GCP project id for Earth Engine (figures step)",
    )
    pa.set_defaults(func=cmd_all)

    pmet = sub.add_parser(
        "metrics",
        help="Write all_events_metrics_summary.csv (fire_analysis metrics --discover)",
    )
    pmet.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pmet.add_argument(
        "--events",
        type=str,
        default="",
        dest="metrics_events_csv",
        metavar="E1,E2,...",
        help="Comma-separated keys; if empty, discover all event folders with outputs",
    )
    pmet.set_defaults(func=cmd_metrics)

    ns = p.parse_args()
    if ns.cmd == "manifest":
        if not ns.discover and not ns.events:
            ns.events = list(DEFAULT_BUNDLE_EVENTS)
    elif ns.cmd in ("pipeline", "figures", "all"):
        if not ns.discover and not ns.events:
            ns.events = list(DEFAULT_BUNDLE_EVENTS)

    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
