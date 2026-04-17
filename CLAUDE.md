# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A machine learning competition focused on **pixel-level deforestation detection** from multimodal satellite imagery. The goal is to build models that identify deforestation events using Sentinel-1 radar, Sentinel-2 optical, and AEF foundation model embeddings — supporting EU Deforestation Regulation (EUDR) compliance.

## Commands

```bash
# Setup environment
make install                  # Creates .venv, installs uv, installs requirements.txt

# Or manually:
python3.10 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m uv pip install -r requirements.txt

# Download dataset from public S3
make download_data_from_s3    # Downloads to ./data/makeathon-challenge/
# Or with custom args:
python3 download_data.py --bucket_name osapiens-terra-challenge --folder_name makeathon-challenge --local_dir ./data

# Launch notebook
source .venv/bin/activate && jupyter notebook challenge.ipynb
```

## Architecture

### Data Pipeline

```
S3 (osapiens-terra-challenge) → download_data.py → ./data/makeathon-challenge/
    ├── tiles/              # GeoJSON metadata (train/test tile boundaries, IDs)
    ├── sentinel1/          # Radar RTC time series (all-weather)
    ├── sentinel2/          # Optical L2A time series (clear-sky only)
    ├── aef_embeddings/     # Foundation model embeddings per pixel
    └── labels/             # Weak labels: GLADS, GLADS2, RADD (may conflict)
```

### Submission Pipeline

Model output → `submission_utils.py` → GeoJSON submission:
- `raster_to_geojson(raster, transform, crs)` converts a binary raster (0=no deforestation, 1=deforestation) to a GeoJSON FeatureCollection
- Applies minimum area filter (default 0.5 hectares) and reprojects to EPSG:4326
- Output is a GeoJSON FeatureCollection of Polygon/MultiPolygon deforestation regions

### Key Challenge Properties

- **Generalization**: Train and test tiles are in **different geographic regions** — overfitting to a region will fail
- **Weak supervision**: Labels from GLADS, GLADS2, and RADD may conflict; handling disagreement is part of the problem
- **Multimodal**: Combining radar (cloud-penetrating), optical, and embeddings is expected to help
- **Time series**: Each tile has observations across multiple dates; temporal patterns matter
- **Pixel-level output**: Final predictions are per-pixel binary classifications

### Core Files

- [challenge.ipynb](challenge.ipynb) — Reference notebook: data structure, label encodings, visualizations, submission format
- [submission_utils.py](submission_utils.py) — Only utility to call: `raster_to_geojson()`
- [download_data.py](download_data.py) — S3 downloader (uses unsigned requests; no AWS credentials needed)
- [osapiens-challenge-full-description.md](osapiens-challenge-full-description.md) — Full problem statement and scoring criteria

### Dependencies

Key packages: `rasterio`, `geopandas`, `shapely` (geospatial), `numpy`, `pandas`, `boto3` (S3), `tqdm`. See [requirements.txt](requirements.txt) for pinned versions.
