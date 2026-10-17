#!/usr/bin/env python3
"""Verify the compact camera-ready evidence bundle without network access."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def main() -> int:
    summary = load_json(ROOT / "artifacts" / "release_summary.json")
    heldout_summary = summary["alphaearth_heldout"]
    tessera_summary = summary["tessera_secondary"]
    assert isinstance(heldout_summary, dict)
    assert isinstance(tessera_summary, dict)

    figure_rows = load_csv(ROOT / "paper" / "reo2_primary_figure_data.csv")
    assert [int(row["slot"]) for row in figure_rows] == list(range(1, 11))
    alphaearth = [float(row["rho_alphaearth"]) for row in figure_rows]
    close(statistics.median(alphaearth), float(heldout_summary["median_rho"]))
    assert sum(value > 0 for value in alphaearth) == 9
    assert sum(bool(row["rho_tessera"].strip()) for row in figure_rows) == 9

    heldout_rows = load_csv(
        ROOT
        / "studies"
        / "mtbs_heldout_20260817"
        / "outputs"
        / "analysis"
        / "per_event_metrics.csv"
    )
    assert len(heldout_rows) == int(heldout_summary["events"])
    assert sum(int(row["n_h3"]) for row in heldout_rows) == int(
        heldout_summary["cells"]
    )
    primary = load_json(
        ROOT
        / "studies"
        / "mtbs_heldout_20260817"
        / "outputs"
        / "analysis"
        / "primary_inference.json"
    )
    close(float(primary["mean_D"]), float(heldout_summary["mean_incremental_contrast"]))
    observed_ci = [float(value) for value in primary["bootstrap"]["ci_95"]]
    expected_ci = [
        float(value) for value in heldout_summary["incremental_contrast_ci95"]
    ]
    for observed, expected in zip(observed_ci, expected_ci, strict=True):
        close(observed, expected)
    assert observed_ci[0] < 0.0 < observed_ci[1]

    tessera_rows = load_csv(
        ROOT
        / "outputs"
        / "tessera_mtbs_secondary"
        / "per_event_metrics.csv"
    )
    assert len(tessera_rows) == int(tessera_summary["events_total"])
    assert sum(row["status"] == "complete" for row in tessera_rows) == int(
        tessera_summary["events_evaluable"]
    )
    tessera = load_json(
        ROOT / "outputs" / "tessera_mtbs_secondary" / "inference.json"
    )
    endpoint = tessera["secondary_endpoint"]
    close(
        float(endpoint["unweighted_mean_s_e"]),
        float(tessera_summary["mean_incremental_contrast"]),
    )
    tessera_ci = [
        float(endpoint["bootstrap"]["lower"]),
        float(endpoint["bootstrap"]["upper"]),
    ]
    for observed, expected in zip(
        tessera_ci,
        tessera_summary["incremental_contrast_ci95"],
        strict=True,
    ):
        close(observed, float(expected))
    assert tessera_ci[0] < 0.0 < tessera_ci[1]

    print("Release verification passed.")
    print("10 held-out fires; 6,864 H3-r8 cells; 9 TESSERA events evaluable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
