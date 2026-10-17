"""Stage 3: sealed unblinding, event-level inference, and compact reporting."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Sequence

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_core import (
    ANALYSIS_ROOT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FIGURE_ROOT,
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REPOSITORY_ROOT,
    STUDY_ROOT,
    StudyError,
    relative,
    sha256_file,
    update_runtime,
    verify_environment,
    write_csv,
    write_json,
)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left = pd.Series(np.asarray(x, dtype=float))
    right = pd.Series(np.asarray(y, dtype=float))
    if (
        len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or left.nunique() < 2
        or right.nunique() < 2
    ):
        raise StudyError("Spearman inputs must be finite and nonconstant")
    left_rank = left.rank(method="average").to_numpy(dtype=float)
    right_rank = right.rank(method="average").to_numpy(dtype=float)
    correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not math.isfinite(correlation):
        raise StudyError("Spearman correlation is nonfinite")
    return correlation


def event_metrics(frame: pd.DataFrame) -> dict[str, float]:
    reference = frame["mtbs_reference"].to_numpy(dtype=float)
    rho_ecs = spearman(frame["ecs"], reference)
    rho_dnbr = spearman(frame["dnbr"], reference)
    rho_dndvi = spearman(frame["dndvi_loss"], reference)
    return {
        "rho_ecs_reference": rho_ecs,
        "rho_dnbr_reference": rho_dnbr,
        "rho_dndvi_loss_reference": rho_dndvi,
        "D_e": rho_ecs - max(rho_dnbr, rho_dndvi),
    }


def bootstrap_mean(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "method": "event resampling with replacement; percentile interval",
        "ci_95": [float(low), float(high)],
        "distribution": distribution,
    }


def exact_sign_randomization(values: np.ndarray) -> dict[str, Any]:
    observed = float(values.mean())
    null = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=float,
    )
    randomization_p = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    nonzero = values[values != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())
    n = len(nonzero)
    if n:
        lower_tail = sum(math.comb(n, index) for index in range(0, min(positive, negative) + 1))
        sign_p = min(1.0, 2.0 * lower_tail / (2**n))
    else:
        sign_p = 1.0
    return {
        "label": "descriptive exact paired sign/randomization summaries",
        "event_sign_flips": int(len(null)),
        "two_sided_randomization_p": randomization_p,
        "positive_D_events": positive,
        "negative_D_events": negative,
        "zero_D_events": int((values == 0).sum()),
        "two_sided_exact_sign_p": float(sign_p),
        "primary_inference_changed": False,
    }


def _study_path(path_text: str) -> Path:
    path = (STUDY_ROOT / path_text).resolve()
    path.relative_to(STUDY_ROOT.resolve())
    return path


def load_sealed() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], str]:
    population_path = OUTPUT_ROOT / "population_freeze.json"
    manifest_path = OUTPUT_ROOT / "sealed_manifest.json"
    if not population_path.exists() or not manifest_path.exists():
        raise StudyError("Reference and feature stages must complete before analysis")
    population_sha = sha256_file(population_path)
    population = json.loads(population_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("population_freeze_sha256") != population_sha
        or not manifest.get("all_events_complete")
        or len(manifest.get("events", [])) != 10
    ):
        raise StudyError("Sealed manifest is incomplete or not tied to population freeze")
    manifest_by_id = {str(item["fire_id"]): item for item in manifest["events"]}
    frames: dict[str, pd.DataFrame] = {}
    for event in population["events"]:
        fire_id = str(event["fire_id"])
        sealed = manifest_by_id.get(fire_id)
        if sealed is None:
            raise StudyError(f"{fire_id}: absent from sealed manifest")
        path = _study_path(sealed["sealed_file"])
        if sha256_file(path) != sealed["sealed_sha256"]:
            raise StudyError(f"{fire_id}: sealed file hash mismatch")
        frame = pd.read_parquet(path)
        if (
            len(frame) != int(event["h3_count"])
            or frame["h3_index"].astype(str).tolist() != event["h3_ids"]
            or frame["h3_index"].duplicated().any()
        ):
            raise StudyError(f"{fire_id}: sealed H3 population changed")
        required = ["mtbs_reference", "dnbr", "dndvi_loss", "ecs"]
        if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
            raise StudyError(f"{fire_id}: sealed file is not complete case")
        frames[fire_id] = frame
    return population["events"], frames, population_sha


def _sensitivity_rows(
    events: list[dict[str, Any]], frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        fire_id = str(event["fire_id"])
        source = frames[fire_id]
        support = source[
            (source["mtbs_valid_fraction"] >= 0.90)
            & (source["s2_valid_fraction"] >= 0.75)
        ].copy()
        if len(support) >= 50:
            try:
                values = event_metrics(support)
            except StudyError:
                values = {}
        else:
            values = {}
        rows.append(
            {
                "sensitivity": "higher_support_thresholds",
                "fire_id": fire_id,
                "name": event["name"],
                "n_cells": len(support),
                "available": bool(values),
                **values,
            }
        )

        coarse = source.copy()
        coarse["h3_parent_r7"] = [
            h3.cell_to_parent(str(value), 7) for value in coarse["h3_index"]
        ]
        numeric = [
            "mtbs_reference",
            "dnbr",
            "dndvi_loss",
            "ecs",
        ]
        coarse = coarse.groupby("h3_parent_r7", sort=True)[numeric].mean().reset_index()
        if len(coarse) >= 30:
            try:
                values = event_metrics(coarse)
            except StudyError:
                values = {}
        else:
            values = {}
        rows.append(
            {
                "sensitivity": "aggregate_to_h3_r7",
                "fire_id": fire_id,
                "name": event["name"],
                "n_cells": len(coarse),
                "available": bool(values),
                **values,
            }
        )
    return pd.DataFrame(rows)


def _plot_event_metrics(frame: pd.DataFrame) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / "event_endpoints.png"
    ordered = frame.sort_values("slot_rank", ascending=False)
    labels = [
        f"{row.name} ({row.fire_id})"
        for row in ordered.itertuples(index=False)
    ]
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    colors = ["#1f77b4" if value >= 0 else "#b22222" for value in ordered["D_e"]]
    axis.barh(np.arange(len(ordered)), ordered["D_e"], color=colors, alpha=0.85)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(ordered)), labels=labels, fontsize=8)
    axis.set_xlabel(
        r"$D_e=\rho(ECS,ref)-\max(\rho(dNBR,ref),\rho(dNDVI_{loss},ref))$"
    )
    axis.set_title("Frozen prospective MTBS held-out event endpoints")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _plot_bootstrap(distribution: np.ndarray, observed: float) -> Path:
    path = FIGURE_ROOT / "event_bootstrap.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.hist(distribution, bins=60, color="#4c78a8", alpha=0.85)
    axis.axvline(observed, color="black", linewidth=1.5, label="observed mean")
    axis.axvline(0, color="#b22222", linewidth=1, linestyle="--", label="zero")
    axis.set_xlabel("Bootstrap mean D")
    axis.set_ylabel("Event-bootstrap replicates")
    axis.set_title("Deterministic event bootstrap (seed 20260817)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _recommendation(mean_d: float, interval: Sequence[float]) -> str:
    low, high = float(interval[0]), float(interval[1])
    if low > 0:
        return (
            "The prospective event-level result favors ECS and its 95% event-bootstrap "
            "interval excludes zero. Include it as the principal held-out replication, "
            "while retaining the finite ten-event and product-alignment limitations."
        )
    if high < 0:
        return (
            "The prospective event-level result favors the optical baselines. Do not "
            "use this study as support for an ECS advantage; include it only as a "
            "clearly labeled negative held-out result if scientifically relevant."
        )
    direction = "positive" if mean_d > 0 else "negative"
    return (
        f"The mean endpoint is {direction}, but the event-bootstrap interval includes "
        "zero. Do not present this study as confirmatory evidence of an ECS advantage; "
        "if included, report it symmetrically as an inconclusive prospective replication."
    )


def _methodology_report(
    metrics: pd.DataFrame,
    aggregate: dict[str, Any],
    recommendation: str,
    population_sha: str,
) -> str:
    ledger = pd.read_csv(OUTPUT_ROOT / "attempted_ledger.csv")
    lines = [
        "# Frozen prospective 10-fire MTBS held-out study",
        "",
        "## Integrity and sequencing",
        "",
        f"- Canonical freeze SHA-256: `{verify_environment()['freeze_sha256']}`.",
        f"- Pre-AlphaEarth population freeze SHA-256: `{population_sha}`.",
        "- Eligibility and H3 populations were frozen before AlphaEarth pixel extraction.",
        "- Per-event feature files were sealed without correlations; analysis opened only after all ten hashes verified.",
        "- Existing configured events, outcome summaries, model labels, and paper files were not imported.",
        "",
        "## Reference and feature contract",
        "",
        "- Official USFS/USGS MTBS boundary polygons and annual thematic Severity rasters were used.",
        "- H3-r8 centers lie in each official boundary; retained cells have at least 80% MTBS classes 1-5 support.",
        "- Reference is the class 3+4 fraction among classes 1-5; classes 0, 6, and NoData are excluded.",
        "- Sentinel-2 uses full calendar Y-1 and Y+1 SCL-masked median composites; dNBR and dNDVI loss are signed pre-minus-post changes.",
        "- AlphaEarth ECS is the 10 m H3 mean of the 64-axis squared annual-embedding change, in deterministic 40-cell batches.",
        "- Every metric uses the exact same complete-case H3 population; no imputation was performed.",
        "",
        "## Attempt and replacement audit",
        "",
        f"- Attempted rows: {len(ledger)}; failed frozen gates: {int((~ledger['eligible'].astype(bool)).sum())}; replacements: {int(metrics['replacement'].sum())}.",
        "- Replacement was permitted only for P01-P05 and consumed reserves in canonical order.",
        "- P06 land-cover mismatches were reported without changing eligibility.",
        "",
        "## Primary event-level inference",
        "",
        f"- Unweighted mean D: `{aggregate['mean_D']:.6f}`.",
        f"- 95% deterministic event-bootstrap CI: `[{aggregate['bootstrap']['ci_95'][0]:.6f}, {aggregate['bootstrap']['ci_95'][1]:.6f}]`.",
        f"- Exact event sign-flip p (descriptive): `{aggregate['descriptive_exact']['two_sided_randomization_p']:.6f}`.",
        "- Event is the replication unit. No cell-level p-values were calculated.",
        "",
        "## Secondary sensitivities",
        "",
        "- Higher MTBS/Sentinel-2 support and H3-r7 aggregation are labeled secondary and never redefine primary eligibility.",
        "",
        "## Paper recommendation",
        "",
        recommendation,
        "",
    ]
    return "\n".join(lines)


def build_artifact_manifest() -> dict[str, Any]:
    excluded_parts = {"raw", "checkpoints", "sealed", "scratch"}
    excluded_suffixes = {".tif", ".tiff", ".parquet", ".zip"}
    roots = [
        STUDY_ROOT / "freeze",
        PROTOCOL_PATH,
        STUDY_ROOT / "README.md",
        STUDY_ROOT / "study_core.py",
        STUDY_ROOT / "study_sources.py",
        STUDY_ROOT / "reference_stage.py",
        STUDY_ROOT / "features_stage.py",
        STUDY_ROOT / "analysis_stage.py",
        STUDY_ROOT / "run_study.py",
        OUTPUT_ROOT,
        REPOSITORY_ROOT / "tests" / "test_mtbs_heldout_study.py",
    ]
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths.add(root)
        else:
            paths.update(path for path in root.rglob("*") if path.is_file())
    entries: list[dict[str, Any]] = []
    forbidden: list[str] = []
    for path in sorted(paths):
        if path == OUTPUT_ROOT / "artifact_manifest.json":
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        data = path.read_bytes()
        display = path.relative_to(REPOSITORY_ROOT).as_posix()
        if path.suffix.lower() in {".py", ".json", ".md", ".csv", ".txt"}:
            text = data.decode("utf-8", errors="ignore")
            mac_home_prefix = "/Users" + "/"
            if re.search(re.escape(mac_home_prefix) + r"[^/\\\s\"']+", text):
                forbidden.append(f"{display}:absolute_user_path")
            if re_search_secret(text):
                forbidden.append(f"{display}:possible_secret")
        entries.append(
            {"path": display, "bytes": len(data), "sha256": sha256_file(path)}
        )
    if forbidden:
        raise StudyError(f"Compact artifact allowlist safety failure: {forbidden}")
    return {
        "schema": "alphaearth_mtbs_compact_artifact_manifest_v1",
        "raw_or_large_data_included": False,
        "absolute_identifying_paths_included": False,
        "secrets_included": False,
        "entries": entries,
    }


def re_search_secret(text: str) -> bool:
    patterns = (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*[\"'][^\"']{12,}",
        r"AKIA[0-9A-Z]{16}",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def run_analysis() -> dict[str, Any]:
    started = time.monotonic()
    verify_environment()
    events, frames, population_sha = load_sealed()
    rows: list[dict[str, Any]] = []
    for event in events:
        fire_id = str(event["fire_id"])
        rows.append(
            {
                "slot_rank": event["slot_rank"],
                "slot": event["slot"],
                "primary_fire_id": event["primary_fire_id"],
                "primary_name": event["primary_name"],
                "fire_id": fire_id,
                "name": event["name"],
                "replacement": event["replacement"],
                "replacement_for": event["replacement_for"],
                "event_year": event["event_year"],
                "n_h3": len(frames[fire_id]),
                "h3_sha256": event["h3_sha256"],
                **event_metrics(frames[fire_id]),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("slot_rank").reset_index(drop=True)
    values = metrics["D_e"].to_numpy(dtype=float)
    bootstrap = bootstrap_mean(values)
    exact = exact_sign_randomization(values)
    aggregate = {
        "schema": "alphaearth_mtbs_primary_inference_v1",
        "independent_unit": "event",
        "n_events": 10,
        "mean_D": float(values.mean()),
        "median_D_descriptive": float(np.median(values)),
        "bootstrap": {
            key: value for key, value in bootstrap.items() if key != "distribution"
        },
        "descriptive_exact": exact,
        "cell_level_p_values": False,
    }
    recommendation = _recommendation(
        aggregate["mean_D"], aggregate["bootstrap"]["ci_95"]
    )
    aggregate["paper_recommendation"] = recommendation

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS_ROOT / "per_event_metrics.csv", metrics)
    write_json(
        ANALYSIS_ROOT / "per_event_metrics.json",
        {
            "schema": "alphaearth_mtbs_per_event_metrics_v1",
            "events": metrics.to_dict(orient="records"),
        },
    )
    write_json(ANALYSIS_ROOT / "primary_inference.json", aggregate)
    sensitivity = _sensitivity_rows(events, frames)
    write_csv(ANALYSIS_ROOT / "secondary_sensitivities.csv", sensitivity)
    sensitivity_summary: list[dict[str, Any]] = []
    for name, group in sensitivity.groupby("sensitivity", sort=True):
        available = group[group["available"].astype(bool)]
        sensitivity_summary.append(
            {
                "sensitivity": name,
                "events_available": len(available),
                "mean_D": (
                    float(available["D_e"].mean()) if len(available) == 10 else None
                ),
                "primary_eligibility_changed": False,
            }
        )
    write_json(
        ANALYSIS_ROOT / "secondary_sensitivities.json",
        {
            "schema": "alphaearth_mtbs_secondary_sensitivity_v1",
            "labeled_secondary": True,
            "summaries": sensitivity_summary,
        },
    )
    figures = [
        _plot_event_metrics(metrics),
        _plot_bootstrap(bootstrap["distribution"], aggregate["mean_D"]),
    ]
    report = _methodology_report(metrics, aggregate, recommendation, population_sha)
    (ANALYSIS_ROOT / "methodology_report.md").write_text(report, encoding="utf-8")
    write_json(
        OUTPUT_ROOT / "analysis_artifacts.json",
        {
            "schema": "alphaearth_mtbs_analysis_artifacts_v1",
            "population_freeze_sha256": population_sha,
            "files": [
                relative(ANALYSIS_ROOT / "per_event_metrics.csv"),
                relative(ANALYSIS_ROOT / "per_event_metrics.json"),
                relative(ANALYSIS_ROOT / "primary_inference.json"),
                relative(ANALYSIS_ROOT / "secondary_sensitivities.csv"),
                relative(ANALYSIS_ROOT / "secondary_sensitivities.json"),
                relative(ANALYSIS_ROOT / "methodology_report.md"),
                *[relative(path) for path in figures],
            ],
        },
    )
    update_runtime(
        "analysis",
        started,
        {
            "events": 10,
            "mean_D": aggregate["mean_D"],
            "bootstrap_ci_95": aggregate["bootstrap"]["ci_95"],
        },
    )
    artifact_manifest = build_artifact_manifest()
    write_json(OUTPUT_ROOT / "artifact_manifest.json", artifact_manifest)
    return aggregate
