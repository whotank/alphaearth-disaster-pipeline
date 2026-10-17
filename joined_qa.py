#!/usr/bin/env python3
"""
Post-run QA on inner-joined H3 cells (AlphaEarth ECS × official benchmark only).

Reads outputs/<event>/joined_inner.parquet — no Earth Engine calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _event_choices() -> list[str]:
    from disaster_embedding_pipeline import _event_configurations

    return sorted(_event_configurations().keys())


def _corr_pearson(a: pd.Series, b: pd.Series) -> float | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = a.corr(b, method="pearson")
    return float(c) if c is not None and np.isfinite(c) else None


def _corr_spearman_no_scipy(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman = Pearson on ranks (SciPy not required; matches pandas for many cases)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = a.rank().corr(b.rank(), method="pearson")
    return float(c) if c is not None and np.isfinite(c) else None


def summarize_joined(df: pd.DataFrame) -> dict:
    """Compute joined-cell diagnostics without writing or mutating source data."""
    if df.empty:
        raise ValueError("No joined rows")

    need = {"ecs_mean", "benchmark_mean"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            f"Joined frame missing columns {missing}; have {list(df.columns)}"
        )

    work = df.copy()
    x = work["benchmark_mean"].astype(float)
    y = work["ecs_mean"].astype(float)

    std_x = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    std_y = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    corr_reason: str | None = None
    if std_x == 0.0 or not np.isfinite(std_x):
        corr_reason = "benchmark_mean has zero or invalid variance (cannot correlate)."
    elif std_y == 0.0 or not np.isfinite(std_y):
        corr_reason = "ecs_mean has zero or invalid variance (cannot correlate)."
    elif int(x.nunique(dropna=True)) < 2 or int(y.nunique(dropna=True)) < 2:
        corr_reason = "at least one variable is constant (unique values < 2)."

    pearson = _corr_pearson(x, y)
    spearman = _corr_spearman_no_scipy(x, y)
    if pearson is None and spearman is None and corr_reason is None:
        corr_reason = "correlation returned non-finite value (check inputs)."

    # Low vs high benchmark (MODIS: ordinal burn day; Hansen forest: mean treecover×loss in hex, often ≪1).
    eps = 1e-9
    unburned = x <= eps
    burned = x > eps
    med_b = float(x.median())
    low_b = x <= med_b
    high_b = x > med_b

    summary = {
        "n_cells": int(len(df)),
        "benchmark_mean_describe": x.describe().to_dict(),
        "ecs_mean_describe": y.describe().to_dict(),
        "diagnostics": {
            "benchmark_std_sample": std_x,
            "ecs_std_sample": std_y,
            "benchmark_nunique": int(x.nunique(dropna=True)),
            "ecs_nunique": int(y.nunique(dropna=True)),
            "corr_unavailable_reason": corr_reason,
        },
        "corr_pearson_benchmark_vs_ecs": pearson,
        "corr_spearman_benchmark_vs_ecs": spearman,
        "mean_ecs_when_benchmark_le_eps": float(y.loc[unburned].mean()) if unburned.any() else None,
        "mean_ecs_when_benchmark_gt_eps": float(y.loc[burned].mean()) if burned.any() else None,
        "n_cells_benchmark_le_eps": int(unburned.sum()),
        "n_cells_benchmark_gt_eps": int(burned.sum()),
        "median_benchmark_mean": med_b,
        "mean_ecs_benchmark_at_or_below_median": float(y.loc[low_b].mean()) if low_b.any() else None,
        "mean_ecs_benchmark_above_median": float(y.loc[high_b].mean()) if high_b.any() else None,
        "n_cells_benchmark_at_or_below_median": int(low_b.sum()),
        "n_cells_benchmark_above_median": int(high_b.sum()),
    }

    # Stratify by benchmark quartiles (alignment: higher benchmark tier → higher ECS on average)
    try:
        work["_bm_q"] = pd.qcut(x, q=4, duplicates="drop")
        rows: list[dict] = []
        for name, sub in work.groupby("_bm_q", observed=True)["ecs_mean"]:
            rows.append(
                {
                    "benchmark_quartile": str(name),
                    "n": int(len(sub)),
                    "ecs_mean_mean": float(sub.mean()),
                    "ecs_mean_median": float(sub.median()),
                }
            )
        summary["ecs_by_benchmark_quartile"] = rows
    except ValueError as e:
        logger.warning("Quartile stratification skipped: %s", e)
        summary["ecs_by_benchmark_quartile"] = []
    return summary


def run_qa(
    parquet_path: Path,
    out_dir: Path,
    *,
    artifact_prefix: str = "",
    reference_label: str = "benchmark",
) -> dict:
    """Write numeric and visual QA for a joined Parquet."""
    df = pd.read_parquet(parquet_path)
    try:
        summary = summarize_joined(df)
    except ValueError as exc:
        raise SystemExit(f"{parquet_path}: {exc}") from exc

    prefix = artifact_prefix.strip("_")
    stem = f"qa_{prefix}_" if prefix else "qa_"
    x = df["benchmark_mean"].astype(float)
    y = df["ecs_mean"].astype(float)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{stem}numeric_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    hb = ax.hexbin(x, y, gridsize=35, cmap="viridis", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel(f"benchmark_mean ({reference_label}, H3 mean)")
    ax.set_ylabel("ecs_mean (H3 mean squared L2 Δ embedding)")
    ax.set_title("Joined cells — density")
    fig.tight_layout()
    scatter_path = out_dir / f"{stem}hexbin_ecs_vs_benchmark.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", scatter_path)

    # Plain scatter (lighter cells visible)
    fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=140)
    ax2.scatter(x, y, s=8, alpha=0.35, c="tab:blue", edgecolors="none")
    ax2.set_xlabel("benchmark_mean")
    ax2.set_ylabel("ecs_mean")
    ax2.set_title("Joined cells — scatter")
    fig2.tight_layout()
    p2 = out_dir / f"{stem}scatter_ecs_vs_benchmark.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Wrote %s", p2)

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default="camp_fire_2018", choices=_event_choices())
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Override path to joined_inner.parquet",
    )
    p.add_argument(
        "--artifact-prefix",
        default="",
        help="Optional QA artifact prefix, for example 'official'.",
    )
    p.add_argument(
        "--reference-label",
        default="benchmark",
        help="Reference label used in plot axes.",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    root = args.output_dir.resolve()
    pq = args.parquet or (root / args.event / "joined_inner.parquet")
    if not pq.is_file():
        raise SystemExit(f"Missing {pq} — run disaster_embedding_pipeline.py for this event first.")

    summary = run_qa(
        pq,
        root / args.event,
        artifact_prefix=args.artifact_prefix,
        reference_label=args.reference_label,
    )

    logging.info("--- QA summary (joined cells only) ---")
    logging.info("n_cells=%s", summary["n_cells"])
    diag = summary.get("diagnostics", {})
    if diag.get("corr_unavailable_reason"):
        logging.info("Correlation skipped: %s", diag["corr_unavailable_reason"])
    logging.info(
        "Diagnostics: benchmark std=%s (nunique=%s), ecs std=%s (nunique=%s)",
        diag.get("benchmark_std_sample"),
        diag.get("benchmark_nunique"),
        diag.get("ecs_std_sample"),
        diag.get("ecs_nunique"),
    )
    logging.info("Pearson(benchmark_mean, ecs_mean)=%s", summary["corr_pearson_benchmark_vs_ecs"])
    logging.info("Spearman(benchmark_mean, ecs_mean)=%s", summary["corr_spearman_benchmark_vs_ecs"])
    logging.info(
        "Mean ecs | benchmark<=1e-9: %s (n=%s)",
        summary["mean_ecs_when_benchmark_le_eps"],
        summary["n_cells_benchmark_le_eps"],
    )
    logging.info(
        "Mean ecs | benchmark>1e-9: %s (n=%s)",
        summary["mean_ecs_when_benchmark_gt_eps"],
        summary["n_cells_benchmark_gt_eps"],
    )
    logging.info(
        "Mean ecs | benchmark<=median (%s): %s (n=%s)",
        summary["median_benchmark_mean"],
        summary["mean_ecs_benchmark_at_or_below_median"],
        summary["n_cells_benchmark_at_or_below_median"],
    )
    logging.info(
        "Mean ecs | benchmark>median: %s (n=%s)",
        summary["mean_ecs_benchmark_above_median"],
        summary["n_cells_benchmark_above_median"],
    )
    if summary.get("ecs_by_benchmark_quartile"):
        logging.info(
            "ECS by benchmark quartile:\n%s",
            pd.DataFrame(summary["ecs_by_benchmark_quartile"]).to_string(index=False),
        )
    logging.info("Artifacts under %s", root / args.event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
