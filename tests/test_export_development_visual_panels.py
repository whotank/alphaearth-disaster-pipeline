from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import export_development_visual_panels as panels


def test_frozen_development_protocol_resolves_eight_publication_joins() -> None:
    specs = panels.load_event_specs(panels.DEFAULT_PROTOCOL)

    assert len(specs) == 8
    assert {spec["family"] for spec in specs} == {
        "lava",
        "forest_loss",
        "drought",
    }
    assert all(Path(spec["join_path"]).is_file() for spec in specs)
    drought_paths = {
        Path(spec["join_path"]).name for spec in specs if spec["family"] == "drought"
    }
    assert drought_paths == {"joined_official.parquet"}


def test_context_tile_crop_matches_prior_combined_figure_contract() -> None:
    image = np.zeros((1_000, 2_000, 3), dtype=np.float32)
    image[:, :, 0] = np.arange(2_000, dtype=np.float32)[None, :]

    before, after = panels.context_tiles(image)

    assert before.shape == (460, 884, 3)
    assert after.shape == (460, 884, 3)
    assert float(before[0, 0, 0]) == 28.0
    assert float(after[0, 0, 0]) == 1_040.0


def test_save_rgb_removes_alpha_and_writes_nonempty_png(tmp_path: Path) -> None:
    source = np.ones((12, 18, 4), dtype=np.float32)
    source[:, :, 3] = 0.5
    output = tmp_path / "panel.png"

    panels._save_rgb(source, output)

    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (18, 12)


def test_mask_loader_accepts_heldout_column_names(tmp_path: Path) -> None:
    source = tmp_path / "heldout.parquet"
    pd.DataFrame(
        {
            "h3_index": ["8828144003fffff", "8828144007fffff"],
            "ecs": [0.2, 0.8],
            "mtbs_reference": [0.1, 0.9],
        }
    ).to_parquet(source, index=False)

    frame, bounds = panels._load_normalized_masks(
        source,
        ecs_column="ecs",
        reference_column="mtbs_reference",
    )

    assert frame["_ecs_plot_z"].tolist() == [0.0, 1.0]
    assert frame["_bench_plot_z"].tolist() == [0.0, 1.0]
    assert len(bounds) == 4
