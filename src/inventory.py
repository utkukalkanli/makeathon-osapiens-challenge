"""Dataset inventory: discover tiles and per-tile AEF years."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("data/makeathon-challenge")
AEF_DIR = DATA_ROOT / "aef-embeddings"
S2_DIR = DATA_ROOT / "sentinel-2"
S1_DIR = DATA_ROOT / "sentinel-1"
LABELS_DIR = DATA_ROOT / "labels" / "train"

AEF_RE = re.compile(r"^(?P<tile>.+)_(?P<year>\d{4})\.tiff?$")


@dataclass(frozen=True)
class Tile:
    tile_id: str
    split: str  # "train" or "test"
    years: tuple[int, ...]

    def aef_path(self, year: int) -> Path:
        return AEF_DIR / self.split / f"{self.tile_id}_{year}.tiff"


def list_tiles(split: str) -> list[Tile]:
    aef_split = AEF_DIR / split
    by_tile: dict[str, list[int]] = {}
    for p in aef_split.glob("*.tif*"):
        m = AEF_RE.match(p.name)
        if not m:
            continue
        by_tile.setdefault(m["tile"], []).append(int(m["year"]))
    tiles = [
        Tile(tile_id=t, split=split, years=tuple(sorted(ys)))
        for t, ys in sorted(by_tile.items())
    ]
    return tiles


def train_tiles() -> list[Tile]:
    return list_tiles("train")


def test_tiles() -> list[Tile]:
    return list_tiles("test")


if __name__ == "__main__":
    for t in train_tiles():
        print("train", t.tile_id, t.years)
    for t in test_tiles():
        print("test ", t.tile_id, t.years)
