"""Local browser-based raster preview tool for the osapiens challenge dataset.

This module serves a small static frontend plus lightweight JSON and PNG
endpoints so we can inspect challenge tiles in a regular web browser without
opening the notebook or desktop GIS software.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-osapiens-viewer")

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .inventory import AEF_DIR, LABELS_DIR, S1_DIR, S2_DIR, Tile, list_tiles
from .labels import fused_labels, load_reference_grid

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = REPO_ROOT / "viewer"

S2_RE = re.compile(
    r"^(?P<tile>.+)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$"
)
S1_RE = re.compile(
    r"^(?P<tile>.+)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_"
    r"(?P<orbit>ascending|descending)\.tif$"
)
GLADL_RE = re.compile(r"^gladl_(?P<tile>.+)_alert(?P<year>\d{2})\.tif$")

AEF_RGB_BANDS = (6, 42, 49)
DEFAULT_PREVIEW_SIZE = 768

ALERT_CMAP = mcolors.ListedColormap(
    ["#d3d3d3", "#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"]
)
ALERT_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ALERT_CMAP.N)

GLADL_CMAP = mcolors.ListedColormap(["#d3d3d3", "#fecc5c", "#e31a1c"])
GLADL_NORM = mcolors.BoundaryNorm([-0.5, 1.0, 2.5, 3.5], GLADL_CMAP.N)

RADD_CMAP = mcolors.ListedColormap(["#d3d3d3", "#fd8d3c", "#e31a1c"])
RADD_NORM = mcolors.BoundaryNorm([-0.5, 1.0, 2.5, 3.5], RADD_CMAP.N)

FUSED_HARD_CMAP = mcolors.ListedColormap(["#d3d3d3", "#d7301f"])
FUSED_HARD_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], FUSED_HARD_CMAP.N)

FUSED_SOFT_CMAP = mcolors.ListedColormap(["#d3d3d3", "#ffffb2", "#fd8d3c", "#d7301f"])
FUSED_SOFT_NORM = mcolors.BoundaryNorm(
    [-0.1, 1 / 6, 0.5, 5 / 6, 1.1], FUSED_SOFT_CMAP.N
)


def _month_key(year: int, month: int) -> int:
    return year * 12 + month


def _require_tile(split: str, tile_id: str) -> Tile:
    for tile in list_tiles(split):
        if tile.tile_id == tile_id:
            return tile
    raise FileNotFoundError(f"unknown tile '{tile_id}' for split '{split}'")


def _safe_percentile_normalize(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = arr[valid_mask]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _preview_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        raise ValueError("preview size must be positive")
    longest = max(height, width)
    if longest <= max_size:
        return height, width
    scale = max_size / float(longest)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def _downsample_label(arr: np.ndarray, max_size: int) -> np.ndarray:
    step = max(1, int(np.ceil(max(arr.shape) / max_size)))
    return arr[::step, ::step]


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    plt.imsave(buf, np.clip(rgb, 0.0, 1.0), format="png")
    return buf.getvalue()


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    buf = BytesIO()
    plt.imsave(buf, np.clip(rgba, 0.0, 1.0), format="png")
    return buf.getvalue()


def _encode_colormapped_png(arr: np.ndarray, cmap, norm) -> bytes:
    rgba = cmap(norm(arr))
    rgb = rgba[..., :3]
    return _encode_rgb_png(rgb.astype(np.float32))


def _read_resized(
    path: Path,
    indexes: list[int] | tuple[int, ...],
    *,
    max_size: int,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.open(path) as src:
        out_h, out_w = _preview_shape(src.height, src.width, max_size)
        data = src.read(
            indexes,
            out_shape=(len(indexes), out_h, out_w),
            resampling=resampling,
        ).astype(np.float32)
    return data


def _discover_s2_entries(tile: Tile) -> list[dict[str, object]]:
    tile_dir = S2_DIR / tile.split / f"{tile.tile_id}__s2_l2a"
    entries: list[dict[str, object]] = []
    if not tile_dir.exists():
        return entries

    parsed: list[tuple[int, int, dict[str, object]]] = []
    for path in tile_dir.glob("*.tif"):
        match = S2_RE.match(path.name)
        if match is None:
            continue
        year = int(match["year"])
        month = int(match["month"])
        parsed.append(
            (
                year,
                month,
                {
                    "id": f"{year:04d}-{month:02d}",
                    "label": f"{year:04d}-{month:02d}",
                    "year": year,
                    "month": month,
                },
            )
        )
    return [entry for _, _, entry in sorted(parsed)]


def _discover_s1_entries(tile: Tile) -> list[dict[str, object]]:
    tile_dir = S1_DIR / tile.split / f"{tile.tile_id}__s1_rtc"
    entries: list[dict[str, object]] = []
    if not tile_dir.exists():
        return entries

    parsed: list[tuple[int, int, str, dict[str, object]]] = []
    for path in tile_dir.glob("*.tif"):
        match = S1_RE.match(path.name)
        if match is None:
            continue
        year = int(match["year"])
        month = int(match["month"])
        orbit = match["orbit"]
        parsed.append(
            (
                year,
                month,
                orbit,
                {
                    "id": f"{year:04d}-{month:02d}-{orbit}",
                    "label": f"{year:04d}-{month:02d} {orbit}",
                    "year": year,
                    "month": month,
                    "orbit": orbit,
                },
            )
        )
    return [entry for _, _, _, entry in sorted(parsed)]


def _discover_label_entries(tile: Tile) -> list[dict[str, str]]:
    if tile.split != "train":
        return []

    entries: list[dict[str, str]] = []
    glads2_alert = LABELS_DIR / "glads2" / f"glads2_{tile.tile_id}_alert.tif"
    if glads2_alert.exists():
        entries.append({"id": "glads2", "label": "GLAD-S2 confidence"})

    gladl_dir = LABELS_DIR / "gladl"
    for path in sorted(gladl_dir.glob(f"gladl_{tile.tile_id}_alert*.tif")):
        match = GLADL_RE.match(path.name)
        if match is None:
            continue
        yy = match["year"]
        entries.append({"id": f"gladl-{yy}", "label": f"GLAD-L alerts 20{yy}"})

    radd_path = LABELS_DIR / "radd" / f"radd_{tile.tile_id}_labels.tif"
    if radd_path.exists():
        entries.append({"id": "radd", "label": "RADD confidence"})

    if tile.years:
        entries.append({"id": "fused-hard", "label": "Fused majority vote"})
        entries.append({"id": "fused-soft", "label": "Fused agreement"})

    return entries


def _build_timeline_entries(
    s2_entries: list[dict[str, object]],
    s1_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_month: dict[tuple[int, int], dict[str, object]] = {}
    for entry in [*s2_entries, *s1_entries]:
        year = int(entry["year"])
        month = int(entry["month"])
        by_month[(year, month)] = {
            "id": f"{year:04d}-{month:02d}",
            "label": f"{year:04d}-{month:02d}",
            "year": year,
            "month": month,
        }
    return [
        by_month[key]
        for key in sorted(by_month, key=lambda item: _month_key(item[0], item[1]))
    ]


@lru_cache(maxsize=16)
def viewer_tiles(split: str) -> dict[str, object]:
    tiles = list_tiles(split)
    return {
        "split": split,
        "tiles": [
            {
                "tile_id": tile.tile_id,
                "years": list(tile.years),
            }
            for tile in tiles
        ],
    }


@lru_cache(maxsize=64)
def tile_manifest(split: str, tile_id: str) -> dict[str, object]:
    tile = _require_tile(split, tile_id)
    s2_entries = _discover_s2_entries(tile)
    s1_entries = _discover_s1_entries(tile)
    aef_entries = [{"id": str(year), "label": str(year), "year": year} for year in tile.years]
    label_entries = _discover_label_entries(tile)
    timeline_entries = _build_timeline_entries(s2_entries, s1_entries)

    defaults = {
        "s2": s2_entries[0]["id"] if s2_entries else None,
        "s1": s1_entries[0]["id"] if s1_entries else None,
        "aef": aef_entries[0]["id"] if aef_entries else None,
        "label": next(
            (entry["id"] for entry in label_entries if entry["id"] == "fused-soft"),
            label_entries[0]["id"] if label_entries else None,
        ),
        "overlay": next(
            (entry["id"] for entry in label_entries if entry["id"] == "fused-soft"),
            label_entries[0]["id"] if label_entries else None,
        ),
        "timeline": timeline_entries[0]["id"] if timeline_entries else None,
    }

    return {
        "split": split,
        "tile_id": tile.tile_id,
        "years": list(tile.years),
        "aef_rgb_bands": list(AEF_RGB_BANDS),
        "s2": s2_entries,
        "s1": s1_entries,
        "aef": aef_entries,
        "labels": label_entries,
        "timeline": timeline_entries,
        "defaults": defaults,
    }


def _s2_path(split: str, tile_id: str, item: str) -> Path:
    year_s, month_s = item.split("-")
    return (
        S2_DIR
        / split
        / f"{tile_id}__s2_l2a"
        / f"{tile_id}__s2_l2a_{int(year_s)}_{int(month_s)}.tif"
    )


def _s1_path(split: str, tile_id: str, item: str) -> Path:
    year_s, month_s, orbit = item.split("-")
    return (
        S1_DIR
        / split
        / f"{tile_id}__s1_rtc"
        / f"{tile_id}__s1_rtc_{int(year_s)}_{int(month_s)}_{orbit}.tif"
    )


def _aef_path(split: str, tile_id: str, item: str) -> Path:
    return AEF_DIR / split / f"{tile_id}_{int(item)}.tiff"


def _render_s2_preview(split: str, tile_id: str, item: str, max_size: int) -> bytes:
    path = _s2_path(split, tile_id, item)
    if not path.exists():
        raise FileNotFoundError(path)

    red, green, blue = _read_resized(
        path, [4, 3, 2], max_size=max_size, resampling=Resampling.bilinear
    )
    rgb = np.stack(
        [
            _safe_percentile_normalize(red, red > 0),
            _safe_percentile_normalize(green, green > 0),
            _safe_percentile_normalize(blue, blue > 0),
        ],
        axis=-1,
    )
    return _encode_rgb_png(rgb)


def _render_s1_preview(split: str, tile_id: str, item: str, max_size: int) -> bytes:
    path = _s1_path(split, tile_id, item)
    if not path.exists():
        raise FileNotFoundError(path)

    backscatter = _read_resized(
        path, [1], max_size=max_size, resampling=Resampling.bilinear
    )[0]
    db = np.full_like(backscatter, np.nan, dtype=np.float32)
    positive = backscatter > 0
    db[positive] = 10.0 * np.log10(backscatter[positive])
    valid = np.isfinite(db)
    gray = _safe_percentile_normalize(np.nan_to_num(db, nan=0.0), valid)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return _encode_rgb_png(rgb)


def _render_aef_preview(split: str, tile_id: str, item: str, max_size: int) -> bytes:
    path = _aef_path(split, tile_id, item)
    if not path.exists():
        raise FileNotFoundError(path)

    bands = _read_resized(
        path,
        list(AEF_RGB_BANDS),
        max_size=max_size,
        resampling=Resampling.bilinear,
    )
    rgb = np.stack(
        [
            _safe_percentile_normalize(bands[0], np.isfinite(bands[0])),
            _safe_percentile_normalize(bands[1], np.isfinite(bands[1])),
            _safe_percentile_normalize(bands[2], np.isfinite(bands[2])),
        ],
        axis=-1,
    )
    return _encode_rgb_png(rgb)


def _render_label_preview(split: str, tile_id: str, item: str, max_size: int) -> bytes:
    if split != "train":
        raise FileNotFoundError("labels are only available for the train split")

    if item == "glads2":
        path = LABELS_DIR / "glads2" / f"glads2_{tile_id}_alert.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = _read_resized(
            path, [1], max_size=max_size, resampling=Resampling.nearest
        )[0].astype(np.uint8)
        return _encode_colormapped_png(arr, ALERT_CMAP, ALERT_NORM)

    if item.startswith("gladl-"):
        yy = item.split("-", 1)[1]
        path = LABELS_DIR / "gladl" / f"gladl_{tile_id}_alert{yy}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = _read_resized(
            path, [1], max_size=max_size, resampling=Resampling.nearest
        )[0].astype(np.uint8)
        return _encode_colormapped_png(arr, GLADL_CMAP, GLADL_NORM)

    if item == "radd":
        path = LABELS_DIR / "radd" / f"radd_{tile_id}_labels.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = _read_resized(
            path, [1], max_size=max_size, resampling=Resampling.nearest
        )[0]
        conf = (raw // 10000).astype(np.uint8)
        return _encode_colormapped_png(conf, RADD_CMAP, RADD_NORM)

    tile = _require_tile(split, tile_id)
    if not tile.years:
        raise FileNotFoundError(f"no AEF years available for tile '{tile_id}'")
    ref_shape, ref_transform, ref_crs = load_reference_grid(tile.aef_path(tile.years[0]))
    hard, soft = fused_labels(tile_id, ref_shape, ref_transform, ref_crs)

    if item == "fused-hard":
        arr = _downsample_label(hard.astype(np.uint8), max_size)
        return _encode_colormapped_png(arr, FUSED_HARD_CMAP, FUSED_HARD_NORM)

    if item == "fused-soft":
        arr = _downsample_label(soft.astype(np.float32), max_size)
        return _encode_colormapped_png(arr, FUSED_SOFT_CMAP, FUSED_SOFT_NORM)

    raise FileNotFoundError(f"unknown label item '{item}'")


def _resolve_label_source(
    split: str,
    tile_id: str,
    item: str,
) -> tuple[np.ndarray, object, object, object, object]:
    if split != "train":
        raise FileNotFoundError("labels are only available for the train split")

    if item == "glads2":
        path = LABELS_DIR / "glads2" / f"glads2_{tile_id}_alert.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            return (
                src.read(1).astype(np.float32),
                src.transform,
                src.crs,
                ALERT_CMAP,
                ALERT_NORM,
            )

    if item.startswith("gladl-"):
        yy = item.split("-", 1)[1]
        path = LABELS_DIR / "gladl" / f"gladl_{tile_id}_alert{yy}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            return (
                src.read(1).astype(np.float32),
                src.transform,
                src.crs,
                GLADL_CMAP,
                GLADL_NORM,
            )

    if item == "radd":
        path = LABELS_DIR / "radd" / f"radd_{tile_id}_labels.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            raw = src.read(1).astype(np.float32)
            conf = np.floor_divide(raw, 10000).astype(np.float32)
            return conf, src.transform, src.crs, RADD_CMAP, RADD_NORM

    tile = _require_tile(split, tile_id)
    if not tile.years:
        raise FileNotFoundError(f"no AEF years available for tile '{tile_id}'")
    ref_shape, ref_transform, ref_crs = load_reference_grid(tile.aef_path(tile.years[0]))
    hard, soft = fused_labels(tile_id, ref_shape, ref_transform, ref_crs)

    if item == "fused-hard":
        return hard.astype(np.float32), ref_transform, ref_crs, FUSED_HARD_CMAP, FUSED_HARD_NORM

    if item == "fused-soft":
        return soft.astype(np.float32), ref_transform, ref_crs, FUSED_SOFT_CMAP, FUSED_SOFT_NORM

    raise FileNotFoundError(f"unknown label item '{item}'")


def _resolve_base_grid(
    kind: str,
    split: str,
    tile_id: str,
    item: str,
    max_size: int,
) -> tuple[tuple[int, int], object, object]:
    if kind == "s2":
        path = _s2_path(split, tile_id, item)
    elif kind == "s1":
        path = _s1_path(split, tile_id, item)
    else:
        raise FileNotFoundError(f"unsupported overlay base kind '{kind}'")

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        out_h, out_w = _preview_shape(src.height, src.width, max_size)
        dst_transform = src.transform * src.transform.scale(
            src.width / out_w,
            src.height / out_h,
        )
        return (out_h, out_w), dst_transform, src.crs


def _render_overlay_preview(
    base_kind: str,
    split: str,
    tile_id: str,
    base_item: str,
    label_item: str,
    max_size: int,
) -> bytes:
    (out_h, out_w), dst_transform, dst_crs = _resolve_base_grid(
        base_kind, split, tile_id, base_item, max_size
    )
    src_arr, src_transform, src_crs, cmap, norm = _resolve_label_source(
        split, tile_id, label_item
    )

    dst_arr = np.zeros((out_h, out_w), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    rgba = cmap(norm(dst_arr)).astype(np.float32)
    rgba[..., 3] = np.where(dst_arr > 0, 1.0, 0.0)
    return _encode_rgba_png(rgba)


@lru_cache(maxsize=128)
def render_preview(kind: str, split: str, tile_id: str, item: str, max_size: int) -> bytes:
    if kind == "s2":
        return _render_s2_preview(split, tile_id, item, max_size)
    if kind == "s1":
        return _render_s1_preview(split, tile_id, item, max_size)
    if kind == "aef":
        return _render_aef_preview(split, tile_id, item, max_size)
    if kind == "label":
        return _render_label_preview(split, tile_id, item, max_size)
    raise FileNotFoundError(f"unknown preview kind '{kind}'")


@lru_cache(maxsize=128)
def render_overlay(
    base_kind: str,
    split: str,
    tile_id: str,
    base_item: str,
    label_item: str,
    max_size: int,
) -> bytes:
    return _render_overlay_preview(base_kind, split, tile_id, base_item, label_item, max_size)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "osapiens-viewer/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/":
                self._serve_file(VIEWER_DIR / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/static/app.js":
                self._serve_file(
                    VIEWER_DIR / "app.js",
                    "application/javascript; charset=utf-8",
                )
                return
            if parsed.path == "/static/styles.css":
                self._serve_file(VIEWER_DIR / "styles.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/healthz":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
                return
            if parsed.path == "/api/tiles":
                params = parse_qs(parsed.query)
                split = params.get("split", ["train"])[0]
                self._send_json(viewer_tiles(split))
                return
            if parsed.path == "/api/tile":
                params = parse_qs(parsed.query)
                split = params.get("split", ["train"])[0]
                tile_id = self._required_param(params, "tile_id")
                self._send_json(tile_manifest(split, tile_id))
                return
            if parsed.path == "/api/preview":
                params = parse_qs(parsed.query)
                kind = self._required_param(params, "kind")
                split = params.get("split", ["train"])[0]
                tile_id = self._required_param(params, "tile_id")
                item = self._required_param(params, "item")
                size = int(params.get("size", [str(DEFAULT_PREVIEW_SIZE)])[0])
                png = render_preview(kind, split, tile_id, item, size)
                self._send_bytes(png, "image/png")
                return
            if parsed.path == "/api/overlay":
                params = parse_qs(parsed.query)
                base_kind = self._required_param(params, "base_kind")
                split = params.get("split", ["train"])[0]
                tile_id = self._required_param(params, "tile_id")
                base_item = self._required_param(params, "base_item")
                label_item = self._required_param(params, "label_item")
                size = int(params.get("size", [str(DEFAULT_PREVIEW_SIZE)])[0])
                png = render_overlay(base_kind, split, tile_id, base_item, label_item, size)
                self._send_bytes(png, "image/png")
                return
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:  # pragma: no cover - last-resort HTTP error
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"unknown route '{parsed.path}'")

    def _required_param(self, params: dict[str, list[str]], key: str) -> str:
        values = params.get(key)
        if not values or not values[0]:
            raise ValueError(f"missing query parameter '{key}'")
        return values[0]

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self._send_bytes(path.read_bytes(), content_type)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the osapiens raster visualizer in a local web browser."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on.")
    return parser


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), ViewerHandler)
    print(
        "Serving osapiens raster visualizer at "
        f"http://{host}:{port} (Ctrl+C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
