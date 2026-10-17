#!/usr/bin/env python3
"""
Fire benchmark vs AlphaEarth: summary metrics tables and per-embedding-axis (A00–A63) analysis.

Modes:
  metrics   — CSV + Markdown from existing pipeline outputs (no Earth Engine).
              Default: ``FIRES_GOOD`` + ``FIRES_POOR`` → ``fire_metrics_summary.csv``.
              ``--discover``: every output subdir with integrity/join → ``all_events_metrics_summary.csv``.
              ``--events a,b,c``: that list → ``all_events_metrics_summary.csv``.
  axes      — Per H3 hex, mean squared delta (post−pre)² per axis; correlate with benchmark_mean (EE).
  consensus — Combine axis Pearson r from ``axes`` outputs across “good” fires only.

See ``outputs/fire_detection_notes.txt`` (written by default metrics only) for using axes to flag MODIS gaps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
import warnings
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Visual alignment judged good vs weak (user); all use MODIS MCD64A1 benchmark in pipeline.
FIRES_GOOD = [
    "bootleg_fire_2021",
    "caldor_fire_2021",
    "camp_fire_2018",
    "dixie_fire_2021",
    "mosquito_fire_2022",
    "thomas_fire_2017",
    "tubbs_fire_2017",
]
FIRES_POOR = ["fort_mcmurray_fire_2016", "marshall_fire_2021"]


def _corr_pearson(a: pd.Series, b: pd.Series) -> float | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = a.corr(b, method="pearson")
    return float(c) if c is not None and np.isfinite(c) else None


def _corr_spearman_no_scipy(a: pd.Series, b: pd.Series) -> float | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = a.rank().corr(b.rank(), method="pearson")
    return float(c) if c is not None and np.isfinite(c) else None


def _load_integrity(out_root: Path, event: str) -> dict[str, Any] | None:
    p = out_root / event / "integrity_report.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _publication_join(
    out_root: Path,
    event: str,
    integrity: dict[str, Any] | None,
) -> tuple[Path, Path, str, dict[str, Any] | None]:
    """Select official USDM joins only when the evidence passes fixed criteria."""
    event_dir = out_root / event
    mirror_join = event_dir / "joined_inner.parquet"
    mirror_qa = event_dir / "qa_numeric_summary.json"
    benchmark = (integrity or {}).get("benchmark")
    benchmark_type = benchmark.get("type") if isinstance(benchmark, dict) else None
    if benchmark_type != "usdm_season_delta":
        return mirror_join, mirror_qa, "pipeline_benchmark", None

    validation_path = event_dir / "usdm_official_validation.json"
    official_join = event_dir / "joined_official.parquet"
    official_qa = event_dir / "qa_official_numeric_summary.json"
    if not validation_path.is_file():
        return mirror_join, mirror_qa, "ee_mirror_unvalidated", None

    from usdm_validation import evaluate_official_validation

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    evaluation = evaluate_official_validation(
        validation,
        expected_event=event,
        expected_pre_year=int((integrity or {}).get("pre_year")),
        expected_post_year=int((integrity or {}).get("post_year")),
        expected_h3_cells=int(
            (integrity or {}).get("total_h3_cells_alphaearth")
        ),
        expected_h3_resolution=int((integrity or {}).get("h3_resolution")),
    )
    validation["_recomputed_evaluation"] = evaluation
    if (
        evaluation["official_reference_complete"]
        and official_join.is_file()
        and official_qa.is_file()
    ):
        return official_join, official_qa, "official_ndmc_vectors", validation
    return mirror_join, mirror_qa, "ee_mirror_unvalidated", validation


def _group_for_event(event: str) -> str:
    if event in FIRES_GOOD:
        return "good"
    if event in FIRES_POOR:
        return "poor"
    return "other"


def _discover_metric_event_dirs(out_root: Path) -> list[str]:
    """Subdirs under out_root that look like pipeline outputs."""
    names: list[str] = []
    for p in out_root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "integrity_report.json").is_file() or (p / "joined_inner.parquet").is_file():
            names.append(p.name)
    return sorted(names)


def _metrics_one_event(out_root: Path, event: str, group: str) -> dict[str, Any]:
    row: dict[str, Any] = {"event_key": event, "group": group}
    ir = _load_integrity(out_root, event)
    if ir:
        row["label"] = ir.get("event", "")
        row["pre_year"] = ir.get("pre_year")
        row["post_year"] = ir.get("post_year")
        row["n_h3_alpha"] = ir.get("total_h3_cells_alphaearth")
        row["n_h3_join"] = ir.get("total_h3_cells_inner_join")
        row["pct_loss_vs_alpha"] = ir.get("pct_data_lost_vs_alpha")
        row["disaster_type"] = ir.get("disaster_type")
        bench = ir.get("benchmark") if isinstance(ir.get("benchmark"), dict) else None
        row["benchmark_type"] = bench.get("type") if bench else None
        row["benchmark_metric_year"] = bench.get("metric_year") if bench else None
    else:
        for k in (
            "label",
            "pre_year",
            "post_year",
            "n_h3_alpha",
            "n_h3_join",
            "pct_loss_vs_alpha",
            "disaster_type",
            "benchmark_type",
            "benchmark_metric_year",
        ):
            row[k] = None

    jp, qa, reference_source, validation = _publication_join(
        out_root,
        event,
        ir,
    )
    row["publication_reference_source"] = reference_source
    row["publication_join_artifact"] = jp.name
    if validation is not None:
        evaluation = validation["_recomputed_evaluation"]
        row["official_validation_passed"] = bool(evaluation["passed"])
        row["official_reference_complete"] = bool(
            evaluation["official_reference_complete"]
        )
        row["official_mirror_agreement_passed"] = bool(
            evaluation["mirror_agreement_passed"]
        )
        row["official_validation_failures"] = ";".join(evaluation["failures"])
        for field in (
            "official_vs_mirror_pearson",
            "official_vs_mirror_spearman",
            "official_vs_mirror_mae",
            "official_vs_mirror_max_abs_error",
            "mirror_vs_ecs_pearson",
            "mirror_vs_ecs_spearman",
        ):
            row[field] = validation.get(field)
    else:
        row["official_validation_passed"] = None
        row["official_reference_complete"] = None
        row["official_mirror_agreement_passed"] = None
        row["official_validation_failures"] = None

    if jp.is_file():
        j = pd.read_parquet(jp)
        row["n_joined_rows"] = int(len(j))
        if "ecs_mean" in j.columns and "benchmark_mean" in j.columns:
            x, y = j["benchmark_mean"].astype(float), j["ecs_mean"].astype(float)
            row["pearson_benchmark_ecs"] = _corr_pearson(x, y)
            row["spearman_benchmark_ecs"] = _corr_spearman_no_scipy(x, y)
            row["ecs_mean_std"] = float(y.std(ddof=1)) if len(y) > 1 else None
            row["benchmark_std"] = float(x.std(ddof=1)) if len(x) > 1 else None
            row["benchmark_nunique"] = int(x.nunique(dropna=True))
        else:
            row["pearson_benchmark_ecs"] = None
            row["spearman_benchmark_ecs"] = None
    else:
        row["n_joined_rows"] = 0
        row["pearson_benchmark_ecs"] = None
        row["spearman_benchmark_ecs"] = None

    if qa.is_file():
        row["qa_file"] = True
        row["qa_artifact"] = qa.name
    else:
        row["qa_file"] = False
        row["qa_artifact"] = qa.name

    return row


def cmd_metrics(
    out_root: Path,
    *,
    events_csv: str | None = None,
    discover: bool = False,
) -> None:
    """
    Default: FIRES_GOOD + FIRES_POOR → ``fire_metrics_summary.csv``.
    With ``--events`` or ``--discover``: all matching folders → ``all_events_metrics_summary.csv``.
    """
    legacy = not discover and not (events_csv and events_csv.strip())
    rows: list[dict[str, Any]] = []
    if legacy:
        for e in FIRES_GOOD:
            rows.append(_metrics_one_event(out_root, e, "good"))
        for e in FIRES_POOR:
            rows.append(_metrics_one_event(out_root, e, "poor"))
        out_csv = out_root / "fire_metrics_summary.csv"
    elif events_csv and events_csv.strip():
        keys = [x.strip() for x in events_csv.split(",") if x.strip()]
        for e in keys:
            rows.append(_metrics_one_event(out_root, e, "specified"))
        out_csv = out_root / "all_events_metrics_summary.csv"
    else:
        keys = _discover_metric_event_dirs(out_root)
        for e in keys:
            rows.append(_metrics_one_event(out_root, e, "discovered"))
        out_csv = out_root / "all_events_metrics_summary.csv"

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    if legacy:
        notes = out_root / "fire_detection_notes.txt"
        notes.write_text(
            "Using embedding axes to find fires missing from MODIS (MCD64A1):\n"
            "1) Run: python fire_analysis.py axes --output-dir outputs\n"
            "2) Read fire_axis_consensus_good_fires.csv for dimensions that align with burn across good fires.\n"
            "3) Build a composite score from those axes (e.g. weighted sum of per-hex mean squared deltas).\n"
            "4) Flag hexes where score is high but benchmark_mean is low — candidates for omission / commission errors\n"
            "   or fires outside the MODIS burn calendar (timing, cloud, product gaps).\n"
            "5) Validate with independent perimeters (MTBS, agency shapefiles) before operational use.\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s", notes)

    # Markdown table (stdout-friendly)
    cols = [
        "event_key",
        "group",
        "disaster_type",
        "benchmark_type",
        "benchmark_metric_year",
        "n_h3_join",
        "pct_loss_vs_alpha",
        "pearson_benchmark_ecs",
        "spearman_benchmark_ecs",
        "benchmark_std",
        "ecs_mean_std",
    ]
    show = df[[c for c in cols if c in df.columns]]
    title = "### Fire metrics (joined H3 cells)\n" if legacy else "### All events metrics\n"
    print(f"\n{title}")
    try:
        print(show.to_markdown(index=False))
    except ImportError:
        print(show.to_string(index=False))
    print()


def _normalize_axis_hex_df(df: pd.DataFrame, emb_bands: list[str]) -> pd.DataFrame:
    """Map EE reduceRegions column names to A00..A63."""
    out = pd.DataFrame({"h3_index": df["h3_index"].astype(str)})
    for b in emb_bands:
        found = None
        for cand in (b, f"{b}_mean", f"{b}_mean".upper()):
            if cand in df.columns:
                found = cand
                break
        if found is None:
            for col in df.columns:
                if col == "h3_index":
                    continue
                if col == b or (col.startswith(b) and "mean" in col.lower()):
                    found = col
                    break
        if found is not None:
            out[b] = pd.to_numeric(df[found], errors="coerce")
        else:
            out[b] = np.nan
    return out


def cmd_axes(out_root: Path, events: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from disaster_embedding_pipeline import (
        EMB_BANDS,
        _annual_embedding_mosaic,
        _buffer_point_meters,
        _event_configurations,
        _fc_to_dataframe,
        _h3_feature_collection,
        _polygon_to_h3_cells,
    )

    project_id = os.environ.get("EE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise SystemExit(
            "Earth Engine needs EE_PROJECT_ID or GOOGLE_CLOUD_PROJECT (see disaster_embedding_pipeline doc / EE access guide)."
        )
    ee.Initialize(project=project_id)

    configs = _event_configurations()

    all_corr_rows: list[dict[str, Any]] = []

    for event in events:
        if event not in configs:
            logger.warning("Unknown event %s, skip", event)
            continue
        cfg = configs[event]
        if cfg.get("disaster_type") != "fire":
            logger.warning("%s is not disaster_type=fire, skip", event)
            continue

        lon, lat = cfg["center_lon"], cfg["center_lat"]
        buf_m = float(cfg["processing_buffer_km"]) * 1000.0
        aoi_shapely = _buffer_point_meters(lon, lat, buf_m)
        aoi_ee = ee.Geometry(aoi_shapely.__geo_interface__)
        res = int(cfg["h3_resolution"])
        cells = _polygon_to_h3_cells(aoi_shapely, res)
        if len(cells) > int(cfg.get("max_h3_cells", 10_000)):
            raise ValueError(f"{event}: too many H3 cells ({len(cells)}), raise max_h3_cells or shrink buffer")

        logger.info("%s: EE reduceRegions on %s H3 cells (axis means of squared deltas)", event, len(cells))
        pre = _annual_embedding_mosaic(int(cfg["pre_year"]), aoi_ee)
        post = _annual_embedding_mosaic(int(cfg["post_year"]), aoi_ee)
        axis_sq = post.subtract(pre).pow(2).select(EMB_BANDS)
        hex_fc = _h3_feature_collection(cells)
        reduced = axis_sq.reduceRegions(
            collection=hex_fc,
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=4,
        )
        axis_df = _fc_to_dataframe(reduced)
        axis_df = _normalize_axis_hex_df(axis_df, EMB_BANDS)

        jp = out_root / event / "joined_inner.parquet"
        if not jp.is_file():
            logger.warning("Missing %s — run pipeline for %s first", jp, event)
            continue
        joined = pd.read_parquet(jp)
        joined["h3_index"] = joined["h3_index"].astype(str)
        m = axis_df.merge(joined[["h3_index", "benchmark_mean", "ecs_mean"]], on="h3_index", how="inner")
        if m.empty:
            logger.warning("%s: inner merge axis↔joined empty", event)
            continue

        bench = m["benchmark_mean"].astype(float)
        corrs: dict[str, float | None] = {}
        for b in EMB_BANDS:
            corrs[b] = _corr_pearson(m[b], bench)

        corr_df = pd.DataFrame(
            [{"band": b, "pearson_r_benchmark": corrs[b], "abs_r": abs(corrs[b]) if corrs[b] is not None else np.nan} for b in EMB_BANDS]
        )
        corr_df = corr_df.sort_values("abs_r", ascending=False, na_position="last")
        ev_dir = out_root / event
        ev_dir.mkdir(parents=True, exist_ok=True)
        corr_path = ev_dir / "axis_benchmark_pearson.csv"
        corr_df.to_csv(corr_path, index=False)
        logger.info("Wrote %s", corr_path)

        for _, r in corr_df.iterrows():
            all_corr_rows.append(
                {
                    "event_key": event,
                    "band": r["band"],
                    "pearson_r_benchmark": r["pearson_r_benchmark"],
                    "abs_r": r["abs_r"],
                }
            )

        # Bar chart: Matplotlib cannot draw bars for None/NaN widths (e.g. constant benchmark).
        plot_df = corr_df.dropna(subset=["pearson_r_benchmark"]).sort_values("abs_r", ascending=False).head(20)
        if plot_df.empty:
            logger.warning("%s: no finite Pearson r for bar chart; skip axis_top20_pearson_bars.png", event)
        else:
            top = plot_df.iloc[::-1]
            vals = top["pearson_r_benchmark"].astype(float)
            fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
            colors = ["tab:green" if x > 0 else "tab:red" if x < 0 else "gray" for x in vals]
            ax.barh(top["band"].astype(str), vals, color=colors)
            ax.axvline(0, color="k", linewidth=0.5)
            ax.set_xlabel("Pearson r (axis mean Δ² vs benchmark_mean)")
            ax.set_title(f"{event} — top |r| embedding axes")
            fig.tight_layout()
            fig.savefig(ev_dir / "axis_top20_pearson_bars.png", bbox_inches="tight")
            plt.close(fig)

    if not all_corr_rows:
        logger.warning("No axis correlations computed.")
        return

    long_df = pd.DataFrame(all_corr_rows)
    wide = long_df.pivot(index="band", columns="event_key", values="pearson_r_benchmark")
    wide_path = out_root / "fire_axis_pearson_matrix.csv"
    wide.to_csv(wide_path)
    logger.info("Wrote %s", wide_path)


def cmd_consensus(out_root: Path) -> None:
    """Mean |r| and #good fires with r>0.15 per band, good fires only."""
    frames = []
    for e in FIRES_GOOD:
        p = out_root / e / "axis_benchmark_pearson.csv"
        if not p.is_file():
            logger.warning("Missing %s — run axes for %s", p, e)
            continue
        d = pd.read_csv(p)
        d["event_key"] = e
        frames.append(d[["band", "pearson_r_benchmark", "event_key"]])
    if not frames:
        raise SystemExit("No axis_benchmark_pearson.csv files found for good fires. Run: python fire_analysis.py axes")

    long = pd.concat(frames, ignore_index=True)
    g = long.groupby("band")["pearson_r_benchmark"].agg(
        mean_abs_r=lambda s: np.nanmean(np.abs(s.to_numpy(dtype=float))),
        n_events="count",
        n_positive_gt_0_15=lambda s: int((s.astype(float) > 0.15).sum()),
        mean_r="mean",
    )
    g = g.sort_values("mean_abs_r", ascending=False)
    out_path = out_root / "fire_axis_consensus_good_fires.csv"
    g.to_csv(out_path)
    logger.info("Wrote %s", out_path)
    print("\n### Top 15 embedding axes by mean |Pearson r| vs benchmark (good fires only)\n")
    try:
        print(g.head(15).to_markdown())
    except ImportError:
        print(g.head(15).to_string())
    print()

    # Heatmap: Pearson r for top consensus bands × good fires
    mat_path = out_root / "fire_axis_pearson_matrix.csv"
    if mat_path.is_file():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wide = pd.read_csv(mat_path, index_col=0)
        top_bands = list(g.head(24).index)
        cols = [c for c in FIRES_GOOD if c in wide.columns]
        sub = wide.reindex(top_bands)[cols].astype(float)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index, fontsize=7)
        ax.set_title("Pearson r: mean axis Δ² vs benchmark (good fires)")
        fig.colorbar(im, ax=ax, shrink=0.55, label="Pearson r")
        fig.tight_layout()
        hp = out_root / "fire_axis_pearson_heatmap_good_fires.png"
        fig.savefig(hp, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", hp)

    # How often each band appears in the top-10 by |r| per good fire (overlap signal)
    votes: Counter[str] = Counter()
    for e in FIRES_GOOD:
        p = out_root / e / "axis_benchmark_pearson.csv"
        if not p.is_file():
            continue
        d = pd.read_csv(p).sort_values("abs_r", ascending=False).head(10)
        for b in d["band"].astype(str):
            votes[b] += 1
    vote_path = out_root / "fire_axis_top10_vote_count.json"
    vote_path.write_text(json.dumps(dict(votes.most_common()), indent=2), encoding="utf-8")
    logger.info("Wrote %s", vote_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    m0 = sub.add_parser(
        "metrics",
        help="Build CSV summary from pipeline outputs (default: good/poor fires only; use --discover for all)",
    )
    m0.add_argument("--output-dir", type=Path, default=Path("outputs"))
    m0.add_argument(
        "--discover",
        action="store_true",
        help="Include every event folder under output-dir with integrity_report.json or joined_inner.parquet",
    )
    m0.add_argument(
        "--events",
        type=str,
        default=None,
        metavar="E1,E2,...",
        help="Comma-separated event keys (overrides --discover if both set)",
    )

    m1 = sub.add_parser("axes", help="Per-axis Pearson vs benchmark (requires EE + prior pipeline run)")
    m1.add_argument("--output-dir", type=Path, default=Path("outputs"))
    m1.add_argument(
        "--events",
        type=str,
        default=",".join(FIRES_GOOD + FIRES_POOR),
        help="Comma-separated event keys (default: all nine user-listed fires).",
    )

    m2 = sub.add_parser("consensus", help="Summarize axis correlations across good fires")
    m2.add_argument("--output-dir", type=Path, default=Path("outputs"))

    args = p.parse_args()
    out = args.output_dir.resolve()

    if args.cmd == "metrics":
        cmd_metrics(out, events_csv=args.events, discover=args.discover)
    elif args.cmd == "axes":
        events = [x.strip() for x in args.events.split(",") if x.strip()]
        cmd_axes(out, events)
        cmd_consensus(out)
    elif args.cmd == "consensus":
        cmd_consensus(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
