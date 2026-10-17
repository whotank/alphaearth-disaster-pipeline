#!/usr/bin/env python3
"""CLI for the frozen prospective MTBS held-out study."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Callable, Sequence

from analysis_stage import run_analysis
from features_stage import run_features
from reference_stage import run_reference
from study_core import (
    OUTPUT_ROOT,
    STUDY_ROOT,
    append_progress,
    sha256_file,
    verify_environment,
    write_json,
)


def status() -> dict[str, Any]:
    environment = verify_environment()
    paths = {
        "attempted_ledger": OUTPUT_ROOT / "attempted_ledger.json",
        "population_freeze": OUTPUT_ROOT / "population_freeze.json",
        "sealed_manifest": OUTPUT_ROOT / "sealed_manifest.json",
        "primary_inference": OUTPUT_ROOT / "analysis" / "primary_inference.json",
        "runtime_manifest": OUTPUT_ROOT / "runtime_manifest.json",
    }
    artifacts = {
        name: {
            "exists": path.exists(),
            "relative_path": path.relative_to(STUDY_ROOT).as_posix(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
        for name, path in paths.items()
    }
    sealed_events = 0
    all_events_complete = False
    if paths["sealed_manifest"].exists():
        sealed = json.loads(paths["sealed_manifest"].read_text(encoding="utf-8"))
        sealed_events = len(sealed.get("events", []))
        all_events_complete = bool(sealed.get("all_events_complete"))
    attempted = 0
    failed_attempts = 0
    if paths["attempted_ledger"].exists():
        ledger = json.loads(paths["attempted_ledger"].read_text(encoding="utf-8"))
        attempts = ledger.get("attempts", [])
        attempted = len(attempts)
        failed_attempts = sum(not bool(item.get("eligible")) for item in attempts)
    if not paths["population_freeze"].exists():
        next_command = (
            "python studies/mtbs_heldout_20260817/run_study.py "
            "reference --ee-project project0-256023"
        )
    elif not all_events_complete:
        next_command = (
            "python studies/mtbs_heldout_20260817/run_study.py "
            "features --ee-project project0-256023"
        )
    elif not paths["primary_inference"].exists():
        next_command = (
            "python studies/mtbs_heldout_20260817/run_study.py analyze"
        )
    else:
        next_command = None
    result = {
        "schema": "alphaearth_mtbs_progress_state_v1",
        "freeze_sha256": environment["freeze_sha256"],
        "base_commit": environment["head"],
        "branch": environment["branch"],
        "attempted": attempted,
        "failed_attempts": failed_attempts,
        "sealed_events": sealed_events,
        "all_events_complete": all_events_complete,
        "analysis_complete": paths["primary_inference"].exists(),
        "artifacts": artifacts,
        "next_resumable_command": next_command,
    }
    write_json(OUTPUT_ROOT / "progress_state.json", result)
    append_progress(
        "study",
        "snapshot",
        {
            "attempted": attempted,
            "failed_attempts": failed_attempts,
            "sealed_events": sealed_events,
            "analysis_complete": result["analysis_complete"],
            "next_resumable_command": next_command,
        },
    )
    return result


def run_logged(stage: str, operation: Callable[[], Any]) -> Any:
    append_progress(stage, "started")
    try:
        result = operation()
    except Exception as error:
        append_progress(
            stage,
            "failed",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    append_progress(stage, "completed", result if isinstance(result, dict) else {})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the frozen ten-fire MTBS held-out study"
    )
    parser.add_argument(
        "command",
        choices=("verify-freeze", "reference", "features", "analyze", "all", "status"),
    )
    parser.add_argument("--ee-project", default="project0-256023")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run mutable per-event checkpoints; immutable freezes are never replaced.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result: Any
    if args.command == "verify-freeze":
        environment = verify_environment()
        result = {
            "freeze_sha256": environment["freeze_sha256"],
            "protocol_sha256": environment["protocol_sha256"],
            "base_commit": environment["head"],
            "branch": environment["branch"],
            "verified": True,
        }
    elif args.command == "reference":
        result = run_logged(
            "reference",
            lambda: run_reference(ee_project=args.ee_project, force=args.force),
        )
    elif args.command == "features":
        result = run_logged(
            "features",
            lambda: run_features(ee_project=args.ee_project, force=args.force),
        )
    elif args.command == "analyze":
        result = run_logged("analysis", run_analysis)
    elif args.command == "all":
        result = {
            "reference": run_logged(
                "reference",
                lambda: run_reference(
                    ee_project=args.ee_project, force=args.force
                ),
            ),
            "features": run_logged(
                "features",
                lambda: run_features(
                    ee_project=args.ee_project, force=args.force
                ),
            ),
            "analysis": run_logged("analysis", run_analysis),
        }
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
