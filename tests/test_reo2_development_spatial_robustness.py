from __future__ import annotations

from pathlib import Path

import pytest

import reo2_development_spatial_robustness as robustness


def test_holm_adjust_uses_all_frozen_tests() -> None:
    adjusted = robustness.holm_adjust([0.01, 0.04, 0.03, 1.0])

    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 1.0])


def test_scale_support_preserves_one_seventh_parent_capacity() -> None:
    assert [robustness.scale_min_cells(level) for level in (1, 2, 3)] == [
        1,
        7,
        49,
    ]


def test_direction_requires_every_frozen_scale_to_be_positive() -> None:
    stable = {
        str(level): {
            "status": "eligible",
            "observed_block_spearman": value,
        }
        for level, value in zip((1, 2, 3), (0.4, 0.3, 0.2), strict=True)
    }
    unstable = {
        **stable,
        "3": {
            "status": "eligible",
            "observed_block_spearman": -0.1,
        },
    }
    incomplete = {
        **stable,
        "3": {
            "status": "excluded",
            "exclusion_reason": "too_few_retained_parent_blocks",
        },
    }

    assert robustness._direction_status(stable) == "directionally_stable"
    assert robustness._direction_status(unstable) == "scale_dependent_or_nonpositive"
    assert (
        robustness._direction_status(incomplete)
        == "positive_at_all_assessable_scales_but_incomplete"
    )


def test_preflight_stops_on_frozen_hash_mismatch(tmp_path: Path) -> None:
    event_dir = tmp_path / "outputs" / "event"
    event_dir.mkdir(parents=True)
    joined = event_dir / "joined_inner.parquet"
    joined.write_bytes(b"not-the-frozen-input")
    (event_dir / "integrity_report.json").write_text(
        '{"h3_resolution": 8}',
        encoding="utf-8",
    )
    protocol = {
        "fixed_events": [
            {
                "event_key": "event",
                "family": "test",
                "join_artifact": "outputs/event/joined_inner.parquet",
                "expected_sha256": "0" * 64,
            }
        ],
        "forbidden_inputs": [],
    }

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        robustness._preflight_inputs(
            protocol,
            repo_root=tmp_path,
            drought_validation_by_event={},
        )


def test_preflight_accepts_only_matching_validated_drought_hash(
    tmp_path: Path,
) -> None:
    event_dir = tmp_path / "outputs" / "drought"
    event_dir.mkdir(parents=True)
    joined = event_dir / "joined_official.parquet"
    joined.write_bytes(b"validated-regeneration")
    observed_hash = robustness.sha256_file(joined)
    (event_dir / "integrity_report.json").write_text(
        '{"h3_resolution": 6}',
        encoding="utf-8",
    )
    protocol = {
        "fixed_events": [
            {
                "event_key": "drought",
                "family": "drought",
                "join_artifact": "outputs/drought/joined_official.parquet",
                "expected_sha256": "0" * 64,
            }
        ],
        "forbidden_inputs": [],
    }
    validation = {
        "drought": {
            "event_key": "drought",
            "passed": True,
            "structure": {
                "artifact_sha256": {
                    "joined_official": observed_hash,
                }
            },
        }
    }

    records = robustness._preflight_inputs(
        protocol,
        repo_root=tmp_path,
        drought_validation_by_event=validation,
    )

    assert records[0]["input_validation_mode"] == ("amendment_1_validated_regeneration")
