from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import disaster_embedding_pipeline as pipeline


class _FakeImage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reduceRegions(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return kwargs["collection"]


class H3CoverageValidationTests(unittest.TestCase):
    def test_missing_h3_id_is_rejected(self) -> None:
        frame = pd.DataFrame({"h3_index": ["a", "c"], "mean": [1.0, 3.0]})

        with self.assertRaisesRegex(ValueError, r"missing=\['b'\]"):
            pipeline._validate_h3_coverage(
                frame,
                ["a", "b", "c"],
                context="test",
            )

    def test_duplicate_h3_id_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {"h3_index": ["a", "b", "b"], "mean": [1.0, 2.0, 2.0]}
        )

        with self.assertRaisesRegex(ValueError, r"duplicates=\['b'\]"):
            pipeline._validate_h3_coverage(
                frame,
                ["a", "b", "c"],
                context="test",
            )

    def test_duplicate_requested_h3_id_is_rejected_before_ee(self) -> None:
        image = _FakeImage()

        with self.assertRaisesRegex(ValueError, "duplicate requested H3 IDs"):
            pipeline._reduce_regions_h3_batched(
                image,
                ["a", "a"],
                reducer=object(),
                scale=10,
                batch_size=40,
                context="test",
            )

        self.assertEqual(image.calls, [])


class H3BatchingTests(unittest.TestCase):
    @staticmethod
    def _frame_for_batch(batch_ids: list[str]) -> pd.DataFrame:
        # EE feature ordering is not relied upon; final rows follow requested H3 order.
        return pd.DataFrame(
            {
                "h3_index": list(reversed(batch_ids)),
                "mean": list(range(len(batch_ids))),
            }
        )

    def test_small_event_uses_one_request_with_original_reduction_parameters(
        self,
    ) -> None:
        image = _FakeImage()
        cells = ["h3-a", "h3-b", "h3-c"]

        with (
            patch.object(
                pipeline,
                "_h3_feature_collection",
                side_effect=lambda batch: list(batch),
            ),
            patch.object(
                pipeline,
                "_fc_to_dataframe",
                side_effect=self._frame_for_batch,
            ),
        ):
            result = pipeline._reduce_regions_h3_batched(
                image,
                cells,
                reducer="same-reducer",
                scale=10,
                tile_scale=4,
                batch_size=40,
                context="AlphaEarth ECS",
            )

        self.assertEqual(result["h3_index"].tolist(), cells)
        self.assertEqual(len(image.calls), 1)
        self.assertEqual(image.calls[0]["collection"], cells)
        self.assertEqual(image.calls[0]["reducer"], "same-reducer")
        self.assertEqual(image.calls[0]["scale"], 10)
        self.assertEqual(image.calls[0]["tileScale"], 4)

    def test_batches_are_deterministic_and_complete(self) -> None:
        image = _FakeImage()
        cells = [f"h3-{number}" for number in range(5)]

        with (
            patch.object(
                pipeline,
                "_h3_feature_collection",
                side_effect=lambda batch: list(batch),
            ),
            patch.object(
                pipeline,
                "_fc_to_dataframe",
                side_effect=self._frame_for_batch,
            ),
        ):
            result = pipeline._reduce_regions_h3_batched(
                image,
                cells,
                reducer="reducer",
                scale=10,
                batch_size=2,
                context="AlphaEarth ECS",
            )

        self.assertEqual(
            [call["collection"] for call in image.calls],
            [["h3-0", "h3-1"], ["h3-2", "h3-3"], ["h3-4"]],
        )
        self.assertEqual(result["h3_index"].tolist(), cells)
        self.assertEqual(len(result), len(cells))


if __name__ == "__main__":
    unittest.main()
