# REO-2 paper preview video

This directory builds a short, caption-led paper preview from the project's
real Sentinel-2 context images, official hazard-reference maps, AlphaEarth ECS
maps, and frozen numerical results. It does not use generated imagery, stock
footage, or a synthetic video service.

## Build

From the repository root:

```bash
uv run \
  --with-requirements paper_preview_video/requirements.txt \
  python paper_preview_video/build_preview_video.py
```

The build writes:

- `output/paper_preview_v1.mp4`: 1920×1080 H.264 video
- `output/paper_preview_v1_storyboard.jpg`: all scenes at a glance
- `output/paper_preview_v1_poster.jpg`: title/poster frame
- `output/paper_preview_v1.srt`: editable narration/caption timing
- `output/paper_preview_v1_manifest.json`: source hashes and render metadata

Use `--quick` for a 1280×720, 15 fps review render. Use `--validate-only` to
check all inputs and frozen numerical invariants without rendering.

## Version 1 structure

1. Paper title and scope
2. The annual before/after question
3. How embedding change squared (ECS) is computed
4. Woolsey Fire example against MTBS
5. Kīlauea example against the USGS lava map
6. Frozen ten-fire event results
7. Incremental-value result and uncertainty
8. Eight-case development audit
9. Main takeaway and repository link

The MP4 is intentionally silent. The burned-in text and SRT file provide the
first-pass explanation. A recorded voice track can be added in a later
iteration without changing the visual build.

## Provenance

The editable values and asset paths live in `preview_data.json`.

- Per-fire coefficients:
  `paper/reo2_primary_figure_data.csv`
- Frozen aggregate values:
  `artifacts/release_summary.json`
- Kīlauea and Woolsey imagery/maps:
  `outputs/reo2_fire_lava_visual_pair/{kil,woo}/`

Map colors are normalized independently for display. The reported Spearman
coefficients use the untransformed values on the matched original H3-r8 cells,
not the expanded visualization rectangles.
