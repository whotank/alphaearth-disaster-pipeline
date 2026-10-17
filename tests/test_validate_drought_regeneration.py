from __future__ import annotations

import validate_drought_regeneration as validation


def _compare(expected: object, actual: object) -> list[str]:
    failures: list[str] = []
    validation._compare_recursive(
        expected,
        actual,
        path="root",
        failures=failures,
        numeric_differences=[],
    )
    return failures


def test_recursive_comparison_accepts_machine_precision_drift() -> None:
    expected = {"value": 0.17521445018392845}
    actual = {"value": 0.17521445018392845 + 1e-15}

    assert _compare(expected, actual) == []


def test_recursive_comparison_rejects_material_numeric_drift() -> None:
    expected = {"value": 0.17521445018392845}
    actual = {"value": 0.17521445018392845 + 1e-5}

    failures = _compare(expected, actual)

    assert len(failures) == 1
    assert "root.value" in failures[0]


def test_recursive_comparison_ignores_only_declared_binary_hashes() -> None:
    expected = {
        "joined_official_sha256": "old",
        "publication_reference": "official_ndmc_vectors",
    }
    actual = {
        "joined_official_sha256": "new",
        "publication_reference": "official_ndmc_vectors",
    }

    assert _compare(expected, actual) == []
