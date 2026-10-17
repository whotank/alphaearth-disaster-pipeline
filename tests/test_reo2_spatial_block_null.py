from __future__ import annotations

import unittest

import h3
import numpy as np
import pandas as pd

import reo2_spatial_block_null as spatial_null


class BlockPermutationTests(unittest.TestCase):
    @staticmethod
    def _block_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "parent_h3": [f"block-{index}" for index in range(10)],
                "ecs_block_mean": [
                    0.03,
                    0.12,
                    0.19,
                    0.31,
                    0.38,
                    0.51,
                    0.57,
                    0.68,
                    0.79,
                    0.91,
                ],
                "benchmark_block_mean": [
                    0.10,
                    0.08,
                    0.24,
                    0.35,
                    0.29,
                    0.55,
                    0.61,
                    0.73,
                    0.70,
                    0.96,
                ],
            }
        )

    def test_same_seed_is_bitwise_deterministic(self) -> None:
        blocks = self._block_table()

        first_summary, first_null = spatial_null.block_permutation_test(
            blocks,
            iterations=999,
            seed=20260815,
        )
        second_summary, second_null = spatial_null.block_permutation_test(
            blocks,
            iterations=999,
            seed=20260815,
        )

        self.assertEqual(first_summary, second_summary)
        self.assertIsNotNone(first_null)
        self.assertIsNotNone(second_null)
        np.testing.assert_array_equal(first_null, second_null)

    def test_permutations_preserve_whole_block_multiset(self) -> None:
        blocks = self._block_table()
        benchmark = blocks["benchmark_block_mean"].to_numpy()
        permutations = spatial_null.draw_block_permutation_indices(
            len(blocks),
            iterations=50,
            rng=np.random.default_rng(1234),
        )

        expected = np.sort(benchmark)
        for permutation in permutations:
            np.testing.assert_array_equal(
                np.sort(benchmark[permutation]),
                expected,
            )
            np.testing.assert_array_equal(
                np.sort(permutation),
                np.arange(len(blocks)),
            )

    def test_constant_reference_is_excluded_without_p_value(self) -> None:
        blocks = self._block_table()
        blocks["benchmark_block_mean"] = 1.0

        summary, null_statistics = spatial_null.block_permutation_test(
            blocks,
            iterations=999,
            seed=99,
        )

        self.assertEqual(summary["status"], "excluded")
        self.assertEqual(
            summary["exclusion_reason"],
            "benchmark_block_mean_constant",
        )
        self.assertNotIn("monte_carlo_p_two_sided", summary)
        self.assertIsNone(null_statistics)


class ParentBlockConstructionTests(unittest.TestCase):
    def test_h3_siblings_are_aggregated_without_cellwise_shuffling(self) -> None:
        first_parent = h3.latlng_to_cell(37.5, -122.2, 6)
        second_parent = next(iter(h3.grid_ring(first_parent, 2)))
        first_children = sorted(h3.cell_to_children(first_parent, 8))[:7]
        second_children = sorted(h3.cell_to_children(second_parent, 8))[:7]
        frame = pd.DataFrame(
            {
                "h3_index": first_children + second_children,
                "ecs_mean": list(range(7)) + list(range(10, 17)),
                "benchmark_mean": list(range(20, 27)) + list(range(40, 47)),
            }
        )

        blocks, diagnostics = spatial_null.build_parent_blocks(
            frame,
            h3_resolution=8,
            parent_levels=2,
            min_cells_per_block=7,
        )

        self.assertEqual(set(blocks["parent_h3"]), {first_parent, second_parent})
        self.assertEqual(blocks["n_cells"].tolist(), [7, 7])
        self.assertTrue(blocks["retained"].all())
        means = blocks.set_index("parent_h3")
        self.assertEqual(means.loc[first_parent, "ecs_block_mean"], 3.0)
        self.assertEqual(means.loc[second_parent, "ecs_block_mean"], 13.0)
        self.assertEqual(diagnostics["n_parent_blocks_retained"], 2)
        self.assertEqual(diagnostics["n_finite_cells_in_retained_blocks"], 14)


if __name__ == "__main__":
    unittest.main()
