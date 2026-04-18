"""Utilities for converting deforestation prediction rasters into submittable GeoJSON."""

import json
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

VALID_GEOMETRY_TYPES = {"Polygon", "MultiPolygon"}
YYMM_RE = re.compile(r"^\d{4}$")


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
) -> dict:
    """Convert a binary deforestation prediction raster to a GeoJSON FeatureCollection.

    Reads a single-band GeoTIFF where 1 indicates deforestation and 0 indicates
    no deforestation, vectorises the foreground pixels into polygons, removes
    polygons smaller than ``min_area_ha``, reprojects the result to EPSG:4326,
    and returns (and optionally writes) a GeoJSON FeatureCollection.

    The caller is responsible for binarising their model output before passing
    it to this function. This function is designed to be the final step in the
    submission pipeline: take your binarised prediction raster, call this
    function, and upload the resulting ``.geojson`` file to the leaderboard.

    Args:
        raster_path: Path to the input GeoTIFF. Must be a single-band raster
            with binary values (0 = no deforestation, 1 = deforestation).
        output_path: Optional path at which to write the GeoJSON file. Parent
            directories are created automatically. If ``None``, the result is
            returned but not written to disk.
        min_area_ha: Minimum polygon area in hectares. Polygons smaller than
            this threshold are removed before the output is written. Area is
            computed in the appropriate UTM projection so the filter is
            metric-accurate regardless of the raster's native CRS. Defaults
            to ``0.5``.

    Returns:
        A GeoJSON-compatible ``dict`` representing a FeatureCollection. Each
        Feature corresponds to one contiguous deforestation polygon in
        EPSG:4326 (longitude/latitude, WGS-84).

    Raises:
        FileNotFoundError: If ``raster_path`` does not point to an existing file.
        ValueError: If the raster contains no deforestation pixels (all zeros),
            or if all polygons are smaller than ``min_area_ha``.

    Example:
        >>> geojson = raster_to_geojson(
        ...     raster_path="predictions/tile_18NVJ.tif",
        ...     output_path="submission/tile_18NVJ.geojson",
        ...     min_area_ha=0.5,
        ... )
        >>> print(len(geojson["features"]), "deforestation polygons")
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        raise ValueError(
            f"No deforestation pixels (value=1) found in {raster_path}. "
            "Ensure the raster has been binarised before calling this function."
        )

    # Vectorise connected foreground regions into polygons
    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    # Filter by area: project to UTM for metric-accurate ha calculation
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    gdf["time_step"] = None

    geojson = json.loads(gdf.to_json())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)

    return geojson


def _normalise_time_step(value):
    """Return a validated YYMM time step, or ``None`` when omitted/null."""
    if value is None:
        return None

    if isinstance(value, int):
        text = f"{value:04d}"
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError(
            f"time_step must be a 4-digit YYMM string/int, null, or omitted; got {type(value).__name__}"
        )

    if not YYMM_RE.fullmatch(text):
        raise ValueError(f"time_step must be in YYMM format, got {value!r}")

    month = int(text[2:])
    if not 1 <= month <= 12:
        raise ValueError(f"time_step month must be between 01 and 12, got {value!r}")

    return int(text)


def _normalise_feature(feature: dict, source: str = "<memory>") -> dict:
    """Strip a feature down to the fields accepted by the submission validator."""
    if feature.get("type") != "Feature":
        raise ValueError(f"{source}: every entry must be a GeoJSON Feature")

    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"{source}: feature geometry must be an object")

    geometry_type = geometry.get("type")
    if geometry_type not in VALID_GEOMETRY_TYPES:
        raise ValueError(
            f"{source}: geometry type must be one of {sorted(VALID_GEOMETRY_TYPES)}, got {geometry_type!r}"
        )

    properties = feature.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        raise ValueError(f"{source}: feature properties must be an object when present")

    norm_properties = {}
    if "time_step" in properties:
        time_step = _normalise_time_step(properties.get("time_step"))
        if time_step is not None:
            norm_properties["time_step"] = time_step

    return {"type": "Feature", "properties": norm_properties, "geometry": geometry}


def validate_submission_geojson(geojson: dict, output_path: str | Path | None = None) -> dict:
    """Validate a submission against the upload requirements.

    Args:
        geojson: Parsed GeoJSON object.
        output_path: Optional destination path. When provided, the suffix must
            be ``.geojson``.

    Returns:
        The input ``geojson`` when validation succeeds.

    Raises:
        ValueError: If the GeoJSON does not match the expected submission schema.
    """
    if output_path is not None and Path(output_path).suffix.lower() != ".geojson":
        raise ValueError(f"Submission file must use the .geojson extension: {output_path}")

    if not isinstance(geojson, dict):
        raise ValueError("GeoJSON must parse to an object")

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON top-level type must be FeatureCollection")

    features = geojson.get("features")
    if not isinstance(features, list):
        raise ValueError("FeatureCollection must contain a features array")

    for idx, feature in enumerate(features):
        _normalise_feature(feature, source=f"feature[{idx}]")

    return geojson


def merge_geojson_featurecollections(
    geojson_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
) -> dict:
    """Merge per-tile GeoJSON FeatureCollections into one uploadable file.

    The merged output keeps only the fields accepted by the submission format:
    ``geometry`` and the optional ``properties.time_step``.
    """
    features = []
    for path_like in geojson_paths:
        path = Path(path_like)
        with open(path) as f:
            payload = json.load(f)
        validate_submission_geojson(payload, output_path=path)
        for feature in payload["features"]:
            features.append(_normalise_feature(feature, source=str(path)))

    merged = {"type": "FeatureCollection", "features": features}
    validate_submission_geojson(merged, output_path=output_path)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(merged, f)

    return merged
