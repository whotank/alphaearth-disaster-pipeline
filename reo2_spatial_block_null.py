#!/usr/bin/env python3
"""Spatially constrained block-permutation sensitivity analysis for REO-2.

The inferential unit inside an event is a coarse H3 parent block, not an
individual joined H3 cell. Fine-cell ECS and benchmark values are averaged
within H3 parents two resolutions coarser, and benchmark block means are
permuted against fixed ECS block means. The preregistered statistic is the
two-sided Spearman correlation across retained blocks.

This is intentionally a separate, read-only analysis of
``outputs/<event>/joined_inner.parquet``. It never writes into event source
directories or changes pipeline, paper, manifest, or audit files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import h3
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

ANALYSIS_VERSION = "1.0.0"
DEFAULT_SEED = 20260815
DEFAULT_ITERATIONS = 9_999
DEFAULT_PARENT_LEVELS = 2
DEFAULT_MIN_CELLS_PER_BLOCK = 7
DEFAULT_MIN_BLOCKS = 8
DEFAULT_EVENT_TYPES = ("fire", "forest_loss", "volcanic_eruption")
DEFAULT_STATISTIC = "spearman_block_means"
DEFAULT_ALTERNATIVE = "two-sided"

REQUIRED_COLUMNS = ("h3_index", "ecs_mean", "benchmark_mean")


class EventIneligibleError(ValueError):
    """An event fails a prospectively defined, outcome-independent rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def stable_event_seed(master_seed: int, event_key: str) -> int:
    """Derive an order-independent event seed from a master seed and key."""

    digest = hashlib.sha256(event_key.encode("utf-8")).digest()
    event_component = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((int(master_seed) + event_component) % (2**63 - 1))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _optional_float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spearman_rho(x: Iterable[float], y: Iterable[float]) -> float | None:
    """Spearman rho with average ranks and explicit degenerate handling."""

    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.shape != y_values.shape or x_values.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional and equal length")
    if len(x_values) < 2:
        return None
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return None

    x_rank = pd.Series(x_values).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y_values).rank(method="average").to_numpy(dtype=float)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def draw_block_permutation_indices(
    n_blocks: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw whole-block permutations; no fine-cell labels are shuffled."""

    if n_blocks < 2:
        raise ValueError("At least two blocks are required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    return np.vstack([rng.permutation(n_blocks) for _ in range(iterations)])


def block_permutation_test(
    blocks: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run the preregistered two-sided block-level Spearman randomization."""

    required = {"ecs_block_mean", "benchmark_block_mean"}
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Block table missing columns: {sorted(missing)}")

    ecs = blocks["ecs_block_mean"].to_numpy(dtype=float)
    benchmark = blocks["benchmark_block_mean"].to_numpy(dtype=float)
    if len(ecs) < 2:
        return (
            {
                "status": "excluded",
                "exclusion_reason": "fewer_than_two_blocks",
            },
            None,
        )
    if np.unique(ecs).size < 2:
        return (
            {
                "status": "excluded",
                "exclusion_reason": "ecs_block_mean_constant",
            },
            None,
        )
    if np.unique(benchmark).size < 2:
        return (
            {
                "status": "excluded",
                "exclusion_reason": "benchmark_block_mean_constant",
            },
            None,
        )

    ecs_rank = pd.Series(ecs).rank(method="average").to_numpy(dtype=float)
    benchmark_rank = (
        pd.Series(benchmark).rank(method="average").to_numpy(dtype=float)
    )
    ecs_centered = ecs_rank - ecs_rank.mean()
    benchmark_centered = benchmark_rank - benchmark_rank.mean()
    denominator = float(
        np.sqrt(
            np.dot(ecs_centered, ecs_centered)
            * np.dot(benchmark_centered, benchmark_centered)
        )
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return (
            {
                "status": "excluded",
                "exclusion_reason": "nonfinite_block_rank_variance",
            },
            None,
        )

    observed = float(np.dot(ecs_centered, benchmark_centered) / denominator)
    rng = np.random.default_rng(seed)
    permutations = draw_block_permutation_indices(len(blocks), iterations, rng)
    null_statistics = (
        benchmark_centered[permutations] @ ecs_centered / denominator
    ).astype(np.float64)

    exceedances = int(
        np.count_nonzero(np.abs(null_statistics) >= abs(observed) - 1e-15)
    )
    p_value = (exceedances + 1.0) / (iterations + 1.0)
    mc_standard_error = math.sqrt(
        p_value * (1.0 - p_value) / (iterations + 1.0)
    )
    quantiles = np.quantile(
        null_statistics,
        [0.005, 0.025, 0.5, 0.975, 0.995],
    )

    return (
        {
            "status": "eligible",
            "exclusion_reason": None,
            "statistic": DEFAULT_STATISTIC,
            "alternative": DEFAULT_ALTERNATIVE,
            "observed_block_spearman": observed,
            "iterations": int(iterations),
            "exceedances_abs_ge_observed": exceedances,
            "monte_carlo_p_two_sided": p_value,
            "monte_carlo_standard_error": mc_standard_error,
            "p_value_formula": "(1 + exceedances) / (1 + iterations)",
            "null_mean": float(np.mean(null_statistics)),
            "null_std_sample": float(np.std(null_statistics, ddof=1)),
            "null_q005": float(quantiles[0]),
            "null_q025": float(quantiles[1]),
            "null_median": float(quantiles[2]),
            "null_q975": float(quantiles[3]),
            "null_q995": float(quantiles[4]),
        },
        null_statistics,
    )


def _child_capacity(parent_h3: str, fine_resolution: int) -> int:
    try:
        return int(h3.cell_to_children_size(parent_h3, fine_resolution))
    except AttributeError:
        return len(h3.cell_to_children(parent_h3, fine_resolution))


def build_parent_blocks(
    frame: pd.DataFrame,
    *,
    h3_resolution: int,
    parent_levels: int,
    min_cells_per_block: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate finite joined cells into deterministic coarse H3 parents."""

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise EventIneligibleError(
            "missing_required_columns",
            f"joined data missing columns {sorted(missing)}",
        )
    if parent_levels < 1 or h3_resolution - parent_levels < 0:
        raise ValueError("parent_levels must yield a non-negative H3 resolution")
    if min_cells_per_block < 1:
        raise ValueError("min_cells_per_block must be positive")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    data["h3_index"] = data["h3_index"].astype(str)
    if data["h3_index"].duplicated().any():
        duplicates = (
            data.loc[data["h3_index"].duplicated(keep=False), "h3_index"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise EventIneligibleError(
            "duplicate_h3_index",
            f"joined data contain duplicate H3 IDs (examples: {duplicates})",
        )

    data["ecs_mean"] = pd.to_numeric(data["ecs_mean"], errors="coerce")
    data["benchmark_mean"] = pd.to_numeric(
        data["benchmark_mean"], errors="coerce"
    )
    finite = np.isfinite(data["ecs_mean"].to_numpy(dtype=float)) & np.isfinite(
        data["benchmark_mean"].to_numpy(dtype=float)
    )
    finite_data = data.loc[finite].copy()
    if finite_data.empty:
        raise EventIneligibleError(
            "no_finite_joined_cells",
            "joined data have no cells finite in both analysis variables",
        )

    invalid_h3 = [
        cell for cell in finite_data["h3_index"] if not h3.is_valid_cell(cell)
    ]
    if invalid_h3:
        raise EventIneligibleError(
            "invalid_h3_index",
            f"invalid H3 IDs (examples: {invalid_h3[:10]})",
        )
    wrong_resolution = [
        cell
        for cell in finite_data["h3_index"]
        if int(h3.get_resolution(cell)) != int(h3_resolution)
    ]
    if wrong_resolution:
        raise EventIneligibleError(
            "h3_resolution_mismatch",
            (
                f"H3 IDs do not all match metadata resolution {h3_resolution} "
                f"(examples: {wrong_resolution[:10]})"
            ),
        )

    parent_resolution = int(h3_resolution - parent_levels)
    finite_data["parent_h3"] = finite_data["h3_index"].map(
        lambda cell: h3.cell_to_parent(cell, parent_resolution)
    )
    blocks = (
        finite_data.groupby("parent_h3", sort=True)
        .agg(
            n_cells=("h3_index", "size"),
            ecs_block_mean=("ecs_mean", "mean"),
            benchmark_block_mean=("benchmark_mean", "mean"),
        )
        .reset_index()
    )
    blocks["parent_resolution"] = parent_resolution
    blocks["fine_resolution"] = int(h3_resolution)
    blocks["capacity_cells"] = blocks["parent_h3"].map(
        lambda parent: _child_capacity(parent, int(h3_resolution))
    )
    blocks["coverage_fraction"] = (
        blocks["n_cells"] / blocks["capacity_cells"]
    )
    blocks["parent_area_km2"] = blocks["parent_h3"].map(
        lambda parent: float(h3.cell_area(parent, unit="km^2"))
    )
    blocks["retained"] = blocks["n_cells"] >= int(min_cells_per_block)

    retained = blocks.loc[blocks["retained"]].reset_index(drop=True)
    diagnostics = {
        "n_source_rows": int(len(data)),
        "n_finite_cells": int(len(finite_data)),
        "n_nonfinite_cells_removed": int(len(data) - len(finite_data)),
        "fine_h3_resolution": int(h3_resolution),
        "parent_h3_resolution": parent_resolution,
        "parent_levels": int(parent_levels),
        "full_nonpentagon_block_capacity_cells": int(7**parent_levels),
        "min_cells_per_block": int(min_cells_per_block),
        "n_parent_blocks_total": int(len(blocks)),
        "n_parent_blocks_retained": int(len(retained)),
        "n_parent_blocks_low_support_removed": int((~blocks["retained"]).sum()),
        "n_finite_cells_in_retained_blocks": int(retained["n_cells"].sum()),
        "retained_block_n_cells_min": (
            int(retained["n_cells"].min()) if not retained.empty else None
        ),
        "retained_block_n_cells_median": (
            float(retained["n_cells"].median()) if not retained.empty else None
        ),
        "retained_block_n_cells_max": (
            int(retained["n_cells"].max()) if not retained.empty else None
        ),
        "retained_parent_area_km2_mean": (
            float(retained["parent_area_km2"].mean())
            if not retained.empty
            else None
        ),
        "cell_level_spearman_descriptive_only": spearman_rho(
            finite_data["ecs_mean"],
            finite_data["benchmark_mean"],
        ),
    }
    return blocks, diagnostics


def exact_two_sided_sign_test(effects: Iterable[float]) -> dict[str, Any]:
    """Exact event-level sign test; zero effects are omitted."""

    values = np.asarray(list(effects), dtype=float)
    values = values[np.isfinite(values)]
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    zero = int(np.count_nonzero(values == 0.0))
    nonzero = positive + negative
    if nonzero == 0:
        p_value = None
    else:
        lower_tail_k = min(positive, negative)
        lower_tail = sum(
            math.comb(nonzero, k) for k in range(lower_tail_k + 1)
        ) / (2**nonzero)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "n_events": int(len(values)),
        "n_positive": positive,
        "n_negative": negative,
        "n_zero_omitted_from_sign_test": zero,
        "n_nonzero_for_sign_test": nonzero,
        "mean_block_spearman_equal_event_weight": (
            float(np.mean(values)) if len(values) else None
        ),
        "median_block_spearman_equal_event_weight": (
            float(np.median(values)) if len(values) else None
        ),
        "exact_sign_test_p_two_sided": p_value,
        "replication_unit": "event",
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def discover_candidates(
    output_root: Path,
    event_types: set[str],
    selected_events: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    for event_dir in sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        if event_dir.name.startswith(".") or event_dir.name == "reo2_spatial_block_null":
            continue
        if selected_events is not None and event_dir.name not in selected_events:
            continue
        integrity_path = event_dir / "integrity_report.json"
        if not integrity_path.is_file():
            continue
        try:
            metadata = json.loads(integrity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            candidates.append(
                {
                    "event_key": event_dir.name,
                    "event_dir": event_dir,
                    "metadata": {},
                    "metadata_error": str(error),
                }
            )
            continue

        disaster_type = str(metadata.get("disaster_type", "unknown"))
        record = {
            "event_key": event_dir.name,
            "event_dir": event_dir,
            "metadata": metadata,
            "metadata_error": None,
        }
        if disaster_type in event_types:
            candidates.append(record)
        else:
            out_of_scope.append(
                {
                    "event_key": event_dir.name,
                    "disaster_type": disaster_type,
                    "reason": "disaster_type_out_of_scope",
                }
            )

    if selected_events is not None:
        discovered = {record["event_key"] for record in candidates + out_of_scope}
        for missing_event in sorted(selected_events - discovered):
            candidates.append(
                {
                    "event_key": missing_event,
                    "event_dir": output_root / missing_event,
                    "metadata": {},
                    "metadata_error": "event directory or integrity_report.json missing",
                }
            )
    return candidates, out_of_scope


def _base_event_result(
    record: dict[str, Any],
    output_root: Path,
    event_seed: int,
) -> dict[str, Any]:
    metadata = record["metadata"]
    benchmark = metadata.get("benchmark") or {}
    event_dir = record["event_dir"]
    joined_path = event_dir / "joined_inner.parquet"
    try:
        source_path = str(joined_path.relative_to(output_root.parent))
    except ValueError:
        source_path = str(joined_path)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "event_key": record["event_key"],
        "event_label": metadata.get("event"),
        "disaster_type": metadata.get("disaster_type"),
        "pre_year": metadata.get("pre_year"),
        "post_year": metadata.get("post_year"),
        "benchmark_type": benchmark.get("type"),
        "benchmark_metric_year": benchmark.get("metric_year"),
        "benchmark_temporal_relation": metadata.get(
            "benchmark_temporal_relation"
        ),
        "metadata_total_h3_cells_alphaearth": metadata.get(
            "total_h3_cells_alphaearth"
        ),
        "metadata_total_h3_cells_benchmark_valid": metadata.get(
            "total_h3_cells_benchmark_valid"
        ),
        "metadata_total_h3_cells_inner_join": metadata.get(
            "total_h3_cells_inner_join"
        ),
        "metadata_pct_data_lost_vs_alpha": metadata.get(
            "pct_data_lost_vs_alpha"
        ),
        "source_joined_parquet": source_path,
        "event_seed": int(event_seed),
        "status": "excluded",
        "exclusion_reason": None,
    }


def analyze_candidate(
    record: dict[str, Any],
    *,
    output_root: Path,
    blocks_dir: Path,
    master_seed: int,
    iterations: int,
    parent_levels: int,
    min_cells_per_block: int,
    min_blocks: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    event_key = record["event_key"]
    event_seed = stable_event_seed(master_seed, event_key)
    result = _base_event_result(record, output_root, event_seed)

    if record.get("metadata_error"):
        result["exclusion_reason"] = "missing_or_invalid_event_metadata"
        result["exclusion_detail"] = record["metadata_error"]
        return result, None

    metadata = record["metadata"]
    h3_resolution = metadata.get("h3_resolution")
    if not isinstance(h3_resolution, int):
        result["exclusion_reason"] = "missing_h3_resolution_metadata"
        return result, None

    joined_path = record["event_dir"] / "joined_inner.parquet"
    if not joined_path.is_file():
        result["exclusion_reason"] = "missing_joined_inner_parquet"
        return result, None

    result["input_size_bytes"] = joined_path.stat().st_size
    result["input_sha256"] = sha256_file(joined_path)
    result["h3_resolution"] = int(h3_resolution)
    try:
        frame = pd.read_parquet(joined_path)
        all_blocks, diagnostics = build_parent_blocks(
            frame,
            h3_resolution=int(h3_resolution),
            parent_levels=parent_levels,
            min_cells_per_block=min_cells_per_block,
        )
    except EventIneligibleError as error:
        result["exclusion_reason"] = error.code
        result["exclusion_detail"] = str(error)
        return result, None
    except Exception as error:
        result["exclusion_reason"] = "source_read_or_validation_error"
        result["exclusion_detail"] = f"{type(error).__name__}: {error}"
        return result, None

    blocks_path = blocks_dir / f"{event_key}_blocks.csv"
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    all_blocks.to_csv(blocks_path, index=False)
    result["derived_block_table"] = str(blocks_path)
    result.update(diagnostics)

    retained = all_blocks.loc[all_blocks["retained"]].reset_index(drop=True)
    if len(retained) < min_blocks:
        result["exclusion_reason"] = "too_few_retained_parent_blocks"
        result["exclusion_detail"] = (
            f"{len(retained)} retained blocks; preregistered minimum is "
            f"{min_blocks}"
        )
        return result, None

    test_summary, null_statistics = block_permutation_test(
        retained,
        iterations=iterations,
        seed=event_seed,
    )
    result.update(test_summary)
    return result, null_statistics


def _aggregate_row(result: dict[str, Any]) -> dict[str, Any]:
    columns = (
        "event_key",
        "event_label",
        "disaster_type",
        "pre_year",
        "post_year",
        "benchmark_type",
        "benchmark_metric_year",
        "benchmark_temporal_relation",
        "status",
        "exclusion_reason",
        "exclusion_detail",
        "h3_resolution",
        "parent_h3_resolution",
        "n_source_rows",
        "n_finite_cells",
        "n_nonfinite_cells_removed",
        "n_parent_blocks_total",
        "n_parent_blocks_retained",
        "n_parent_blocks_low_support_removed",
        "n_finite_cells_in_retained_blocks",
        "cell_level_spearman_descriptive_only",
        "observed_block_spearman",
        "iterations",
        "monte_carlo_p_two_sided",
        "monte_carlo_standard_error",
        "bh_fdr_adjusted_p_all_eligible_events",
        "null_q025",
        "null_median",
        "null_q975",
        "event_seed",
        "input_size_bytes",
        "input_sha256",
        "source_joined_parquet",
        "derived_block_table",
    )
    return {column: result.get(column) for column in columns}


def aggregate_event_groups(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [result for result in results if result["status"] == "eligible"]

    def effects_for(types: set[str] | None) -> list[float]:
        return [
            float(result["observed_block_spearman"])
            for result in eligible
            if types is None or result.get("disaster_type") in types
        ]

    return {
        "all_eligible": exact_two_sided_sign_test(effects_for(None)),
        "volcanic_eruption": exact_two_sided_sign_test(
            effects_for({"volcanic_eruption"})
        ),
        "fire": exact_two_sided_sign_test(effects_for({"fire"})),
        "forest_loss": exact_two_sided_sign_test(
            effects_for({"forest_loss"})
        ),
        "fire_or_forest_loss": exact_two_sided_sign_test(
            effects_for({"fire", "forest_loss"})
        ),
    }


def make_figure(results: list[dict[str, Any]], output_path: Path) -> bool:
    eligible = sorted(
        (result for result in results if result["status"] == "eligible"),
        key=lambda result: (str(result.get("disaster_type")), result["event_key"]),
    )
    if not eligible:
        return False

    colors = {
        "fire": "#d95f02",
        "forest_loss": "#1b9e77",
        "volcanic_eruption": "#7570b3",
    }
    figure_height = max(4.5, 0.42 * len(eligible) + 1.8)
    fig, ax = plt.subplots(figsize=(8.6, figure_height), dpi=160)
    for position, result in enumerate(eligible):
        color = colors.get(str(result.get("disaster_type")), "#444444")
        ax.hlines(
            position,
            float(result["null_q025"]),
            float(result["null_q975"]),
            color=color,
            linewidth=2.3,
            alpha=0.75,
        )
        ax.scatter(
            float(result["observed_block_spearman"]),
            position,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            s=38,
            zorder=3,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_yticks(np.arange(len(eligible)))
    ax.set_yticklabels(
        [
            (
                f"{result['event_key']}  "
                f"(p={result['monte_carlo_p_two_sided']:.3g})"
            )
            for result in eligible
        ],
        fontsize=8,
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel("Spearman correlation across coarse H3 block means")
    ax.set_title(
        "REO-2 spatial block null\n"
        "points: observed; bars: central 95% permutation-null interval"
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _format_number(value: Any, digits: int = 3) -> str:
    number = _optional_float(value)
    return "NA" if number is None else f"{number:.{digits}g}"


def methodology_report(
    *,
    results: list[dict[str, Any]],
    aggregates: dict[str, Any],
    config: dict[str, Any],
    out_of_scope: list[dict[str, Any]],
) -> str:
    eligible = [result for result in results if result["status"] == "eligible"]
    excluded = [result for result in results if result["status"] != "eligible"]
    lines = [
        "# REO-2 spatially constrained null/control",
        "",
        "## Preregistered analysis used",
        "",
        (
            f"Before computing event correlations in this run, the analysis "
            f"source fixed seed **{config['seed']}**, "
            f"**{config['iterations']:,}** Monte Carlo permutations, H3 "
            f"parents **{config['parent_levels']} resolutions coarser** than "
            f"the joined cells, at least "
            f"**{config['min_cells_per_block']} finite cells per retained "
            f"block**, at least **{config['min_blocks']} retained blocks per "
            f"event**, and a **two-sided Spearman correlation of unweighted "
            f"block means**. Event eligibility uses metadata type, file/schema "
            f"validity, H3 validity/resolution, block count, and nondegenerate "
            f"block-level variables; it never uses the observed correlation."
        ),
        "",
        (
            "Fine H3 cells were aggregated to their deterministic coarse H3 "
            "parent. The benchmark block means were then permuted as whole "
            "units against fixed ECS block means. The randomization tail area "
            "uses `(1 + exceedances) / (1 + iterations)`. Thus individual H3 "
            "cells are not treated as replicates."
        ),
        "",
        "## Event results",
        "",
        "| event | type | status | retained blocks | block rho | two-sided MC p | BH p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item["event_key"]):
        reason = (
            "eligible"
            if result["status"] == "eligible"
            else f"excluded: {result.get('exclusion_reason')}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["event_key"]),
                    str(result.get("disaster_type") or "unknown"),
                    reason,
                    str(result.get("n_parent_blocks_retained") or "NA"),
                    _format_number(result.get("observed_block_spearman")),
                    _format_number(result.get("monte_carlo_p_two_sided")),
                    _format_number(
                        result.get("bh_fdr_adjusted_p_all_eligible_events")
                    ),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Event-level aggregation",
            "",
            (
                "Aggregate summaries give every eligible event equal weight. "
                "The exact two-sided sign test uses events—not H3 cells or "
                "coarse blocks—as replication units."
            ),
            "",
            "| group | events | positive / negative | median rho | sign-test p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_name, summary in aggregates.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    group_name,
                    str(summary["n_events"]),
                    f"{summary['n_positive']} / {summary['n_negative']}",
                    _format_number(
                        summary["median_block_spearman_equal_event_weight"]
                    ),
                    _format_number(summary["exact_sign_test_p_two_sided"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Eligibility and exclusions",
            "",
            (
                f"{len(results)} fire/forest-loss/volcanic candidates were "
                f"found from per-event integrity metadata; {len(eligible)} "
                f"were analyzable and {len(excluded)} were excluded by the "
                "rules above."
            ),
        ]
    )
    for result in excluded:
        lines.append(
            f"- `{result['event_key']}`: {result.get('exclusion_reason')}"
            + (
                f" ({result['exclusion_detail']})"
                if result.get("exclusion_detail")
                else ""
            )
        )
    if out_of_scope:
        scope_text = ", ".join(
            f"`{item['event_key']}` ({item['disaster_type']})"
            for item in out_of_scope
        )
        lines.append(f"- Metadata-discovered but out of scope: {scope_text}.")

    lines.extend(
        [
            "",
            "## Limitations and interpretation",
            "",
            (
                "- The method protects against fine-cell pseudoreplication and "
                "retains local support within parent blocks, but permutation "
                "does **not** preserve autocorrelation or broad gradients "
                "between parent blocks."
            ),
            (
                "- Exchangeability of coarse block labels may fail under "
                "terrain, land-cover, radial, coastal, or directional gradients; "
                "AOI-edge blocks also have unequal observed support."
            ),
            (
                "- The two-resolution coarsening and seven-cell support cutoff "
                "are defensible fixed defaults, not an empirically established "
                "correlation range. A multiscale analysis would be needed to "
                "show robustness to block size."
            ),
            (
                "- The event collection is small and not a probability sample. "
                "Per-event Monte Carlo tail areas are conditional spatial "
                "randomization diagnostics; only the sign summary uses events "
                "as replicates, and its power is limited."
            ),
            (
                "- Benchmark quality, missing benchmark coverage, and annual "
                "temporal alignment remain unchanged. Correlation is not a "
                "causal effect or a calibrated disaster detector."
            ),
            (
                "- Defaults were frozen in the analysis source before outcomes "
                "were computed in this execution, but this is not a third-party "
                "time-stamped preregistration because the source data already "
                "existed."
            ),
            "",
            (
                "**Assessment:** this null is suitable as a compact sensitivity "
                "analysis against independent-cell inference. On its own it is "
                "**not strong enough to serve as the main paper's definitive "
                "spatial null**; that would require multiscale or covariate-"
                "restricted spatial randomization and more independent events."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Default: <output-root>/reo2_spatial_block_null",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--parent-levels",
        type=int,
        default=DEFAULT_PARENT_LEVELS,
    )
    parser.add_argument(
        "--min-cells-per-block",
        type=int,
        default=DEFAULT_MIN_CELLS_PER_BLOCK,
    )
    parser.add_argument("--min-blocks", type=int, default=DEFAULT_MIN_BLOCKS)
    parser.add_argument(
        "--event-types",
        default=",".join(DEFAULT_EVENT_TYPES),
        help="Comma-separated metadata disaster types.",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Optional comma-separated event keys; default discovers all in-scope events.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.iterations < 999:
        raise SystemExit("--iterations must be at least 999 for this analysis")

    output_root = args.output_root.resolve()
    analysis_dir = (
        args.analysis_dir.resolve()
        if args.analysis_dir is not None
        else output_root / "reo2_spatial_block_null"
    )
    event_types = {
        value.strip() for value in args.event_types.split(",") if value.strip()
    }
    selected_events = (
        {value.strip() for value in args.events.split(",") if value.strip()}
        if args.events
        else None
    )
    candidates, out_of_scope = discover_candidates(
        output_root,
        event_types,
        selected_events,
    )
    if not candidates:
        raise SystemExit("No in-scope events with integrity metadata were found")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = analysis_dir / "blocks"
    per_event_dir = analysis_dir / "per_event"
    null_arrays: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for record in candidates:
        event_key = record["event_key"]
        LOGGER.info("Analyzing %s", event_key)
        result, null_statistics = analyze_candidate(
            record,
            output_root=output_root,
            blocks_dir=blocks_dir,
            master_seed=args.seed,
            iterations=args.iterations,
            parent_levels=args.parent_levels,
            min_cells_per_block=args.min_cells_per_block,
            min_blocks=args.min_blocks,
        )
        results.append(result)
        if null_statistics is not None:
            null_arrays[event_key] = null_statistics

    eligible = [result for result in results if result["status"] == "eligible"]
    adjusted = benjamini_hochberg(
        [float(result["monte_carlo_p_two_sided"]) for result in eligible]
    )
    for result, adjusted_p in zip(eligible, adjusted, strict=True):
        result["bh_fdr_adjusted_p_all_eligible_events"] = adjusted_p

    for result in results:
        write_json(per_event_dir / f"{result['event_key']}.json", result)

    aggregates = aggregate_event_groups(results)
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": int(args.seed),
        "iterations": int(args.iterations),
        "parent_levels": int(args.parent_levels),
        "min_cells_per_block": int(args.min_cells_per_block),
        "min_blocks": int(args.min_blocks),
        "event_types": sorted(event_types),
        "statistic": DEFAULT_STATISTIC,
        "alternative": DEFAULT_ALTERNATIVE,
        "run_matches_source_defaults": bool(
            args.seed == DEFAULT_SEED
            and args.iterations == DEFAULT_ITERATIONS
            and args.parent_levels == DEFAULT_PARENT_LEVELS
            and args.min_cells_per_block == DEFAULT_MIN_CELLS_PER_BLOCK
            and args.min_blocks == DEFAULT_MIN_BLOCKS
            and event_types == set(DEFAULT_EVENT_TYPES)
            and selected_events is None
        ),
        "eligibility_rule": (
            "per-event integrity metadata type in scope; joined_inner.parquet "
            "present with required finite, unique, valid H3 cells at metadata "
            "resolution; >= min_blocks parent blocks each with >= "
            "min_cells_per_block finite cells; nonconstant ECS and benchmark "
            "block means"
        ),
        "p_value_interpretation": (
            "conditional Monte Carlo block-randomization tail area, not an "
            "independent-cell or population-of-events p-value"
        ),
    }
    aggregate_payload = {
        "analysis": config,
        "candidate_count": len(results),
        "eligible_count": len(eligible),
        "excluded_count": len(results) - len(eligible),
        "event_results": results,
        "event_level_aggregates": aggregates,
        "out_of_scope_metadata_events": out_of_scope,
    }
    write_json(analysis_dir / "aggregate_results.json", aggregate_payload)
    pd.DataFrame([_aggregate_row(result) for result in results]).to_csv(
        analysis_dir / "aggregate_results.csv",
        index=False,
    )
    np.savez_compressed(
        analysis_dir / "null_statistics.npz",
        **null_arrays,
    )
    figure_written = make_figure(
        results,
        analysis_dir / "event_block_null_summary.png",
    )
    report = methodology_report(
        results=results,
        aggregates=aggregates,
        config=config,
        out_of_scope=out_of_scope,
    )
    (analysis_dir / "methodology_report.md").write_text(
        report,
        encoding="utf-8",
    )

    LOGGER.info(
        "Wrote %d eligible and %d excluded candidate results to %s",
        len(eligible),
        len(results) - len(eligible),
        analysis_dir,
    )
    if figure_written:
        LOGGER.info("Wrote compact figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
