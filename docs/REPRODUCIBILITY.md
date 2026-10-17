# Reproducibility guide

The release supports three levels of replay. The first two are offline; the
third requires the upstream services and reference products described in
`DATA.md`.

## 1. Verify the reported evidence

Install Python 3.12 or newer and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python verify_release.py
python -m pytest -q
```

`verify_release.py` checks the frozen event counts, 6,864-cell held-out
population, median AlphaEarth coefficient, AlphaEarth event-bootstrap
endpoint, TESSERA missingness, and TESSERA endpoint. It performs no network
calls.

## 2. Rebuild the manuscript and preview

The manuscript requires a recent TeX Live installation:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The expected output is a seven-page PDF with four main-text pages, followed by
references and appendices.

Validate the preview inputs without encoding video:

```bash
python paper_preview_video/build_preview_video.py --validate-only
```

To render the MP4:

```bash
python -m pip install -r paper_preview_video/requirements.txt
python paper_preview_video/build_preview_video.py
```

The video manifest records source hashes, frame count, codec, dimensions, and
duration. The build uses existing project imagery and mapped-reference
artifacts.

## 3. Re-extract upstream data

Set up Google Earth Engine and provide a project:

```bash
earthengine authenticate
export EE_PROJECT_ID=your-project-id
```

The principal extraction and analysis entry points are:

- `disaster_embedding_pipeline.py` for AlphaEarth and event-reference H3
  products.
- `sentinel2_optical_baseline.py` for the dNBR/dNDVI comparator.
- `studies/mtbs_heldout_20260817/run_study.py` for the frozen held-out fire
  workflow.
- `tessera_mtbs_secondary.py` for the prespecified TESSERA analysis.
- `reo2_development_spatial_robustness.py` for the development-case block
  sensitivity.

Some visual exporters additionally require the raw MTBS perimeter and severity
files. Set `MTBS_HELDOUT_ROOT` to a full held-out study directory when those
files are outside this compact release.

## Frozen design

The held-out cohort, reserves, H3 populations, estimand, and bootstrap seed
were fixed before AlphaEarth extraction. Each fire—not each spatial cell—is an
inferential unit. Weak, negative, replacement, and missing outcomes remain in
the artifact.

The compact release intentionally omits credentials, caches, Earth Engine
temporary exports, TESSERA tile archives, and third-party raw archives.
