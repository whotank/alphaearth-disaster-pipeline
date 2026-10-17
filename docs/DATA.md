# Data sources and redistribution

This repository contains compact derived evidence, not complete upstream data
archives. The frozen artifacts are sufficient to verify the reported event
statistics without re-downloading imagery. Re-extraction requires access to
the sources below.

## Embeddings and imagery

- **AlphaEarth Foundations Satellite Embedding V1**:
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` in Google Earth Engine. The product
  provides annual 64-dimensional embeddings at 10 m. Consult the Earth Engine
  catalog entry for current terms and attribution.
- **TESSERA v1.0**: annual 128-dimensional Sentinel-1/2 embeddings retrieved
  from the official v1.0 `vultr` manifest. The protocol records one unavailable
  required year-tile object; that event remains explicitly missing.
- **Sentinel-2 surface reflectance**:
  `COPERNICUS/S2_SR_HARMONIZED` in Earth Engine. The paper uses cloud-masked,
  full-calendar annual medians of B4, B8, and B12.

## Event references

- **MTBS** official fire perimeters and 30 m thematic burn-severity rasters.
  The release includes frozen per-cell derived values and population hashes,
  not the complete MTBS download archive.
- **USGS** 2018 Kīlauea and 2022 Mauna Loa lava-emplacement products.
- **Copernicus Emergency Management Service EMSR546** for the 2021 La Palma
  lava extent and analyzed footprint.
- **Hansen Global Forest Change** annual loss year and year-2000 canopy cover.
- **U.S. Drought Monitor** official weekly vectors from the National Drought
  Mitigation Center.
- **ESA WorldCover** for non-water support in lava analyses.

The bibliography in `paper/references.bib` gives the citations and persistent
identifiers used in the manuscript.

## Included derived artifacts

- CSV and JSON event metrics and event-bootstrap summaries.
- Immutable held-out H3 populations and compact Parquet cell tables.
- Protocols, freeze audits, attempted-event ledgers, and content hashes.
- Selected PNG/JPEG visual products for Kīlauea, Rhea, and Woolsey.
- A caption-led MP4 assembled from those visual products.

Map colors are normalized for display. Reported coefficients are calculated
from the untransformed matched H3 values.

## Licensing

The root MIT license applies to repository software. It does not replace the
licenses or terms of upstream datasets. Before redistributing imagery or
derived geospatial products, review the applicable Google, ESA, USGS,
Copernicus, MTBS, NDMC, Hansen/UMD, and TESSERA terms and retain required
attribution.
