"""Build static Orbitree data: label PNGs, tiles.json, models.json, predictions.

Run from the repo root:
    python -m tools.web_data.build --submissions ../MAKEATHON --out web/public/data
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from shapely.geometry import shape

from .countries import TILE_COUNTRY
from .pipeline import (assign_tile_ids, colorize_agreement, colorize_mask,
                       hectares_by_tile, label_stats, simplify_fc, tile_grid)

BASE = "https://osapiens-terra-challenge.s3.amazonaws.com/makeathon-challenge/"
DATA_ROOT = Path("data/makeathon-challenge")
FEATURED = "temporal_s2s1_norm_pseudo_v1_thr040"
COLORS = {"glads2": "#facc15", "gladl": "#fb923c", "radd": "#f43f5e"}
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix: str):
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get("https://osapiens-terra-challenge.s3.amazonaws.com/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s:Contents", NS):
            yield c.find("s:Key", NS).text
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return
        token = root.findtext("s:NextContinuationToken", namespaces=NS)


def download(rel: str):
    dest = DATA_ROOT / rel
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(BASE + rel, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)


def fetch_inputs():
    for name in ("train", "test"):
        download(f"metadata/{name}_tiles.geojson")
    keys = [k for k in list_keys("makeathon-challenge/labels/train/") if not k.endswith("/")]
    for i, k in enumerate(keys, 1):
        download(k[len("makeathon-challenge/"):])
    print(f"label files present: {len(keys)}", file=sys.stderr)


def load_tiles(split: str):
    fc = json.load(open(DATA_ROOT / "metadata" / f"{split}_tiles.geojson"))
    out = []
    for f in fc["features"]:
        geom = shape(f["geometry"])
        out.append((f["properties"]["name"], geom))
    return out


def model_entry(mid: str) -> dict:
    m = re.search(r"thr(\d{3})", mid)
    thr = int(m.group(1)) / 100 if m else None
    sensors = "S2+S1" if "s2s1" in mid else "S2"
    parts = [sensors]
    if "norm" in mid:
        parts.append("norm")
    pm = re.search(r"pseudo_(v\d+)", mid)
    if pm:
        parts.append(f"pseudo {pm.group(1)}")
    else:
        vm = re.search(r"_(v\d+)", mid)
        if vm:
            parts.append(vm.group(1))
    if thr is not None:
        parts.append(f"thr {thr:.2f}")
    return {"id": mid, "label": " · ".join(parts), "threshold": thr, "featured": mid == FEATURED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=0.00003)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fetch_inputs()
    from src.labels import glads2_mask, gladl_mask, radd_mask

    train = load_tiles("train")
    test = load_tiles("test")

    subs = sorted(Path(args.submissions).glob("*_submission.geojson"))
    models = [model_entry(p.name[: -len("_submission.geojson")]) for p in subs]
    models.sort(key=lambda m: (not m["featured"], m["id"]))
    model_ids = [m["id"] for m in models]

    tiles = []
    ranking = []
    for tid, geom in train:
        bounds = list(geom.bounds)
        shp, tf = tile_grid(bounds)
        m1 = glads2_mask(tid, shp, tf, "EPSG:4326")
        m2 = gladl_mask(tid, shp, tf, "EPSG:4326")
        m3 = radd_mask(tid, shp, tf, "EPSG:4326")
        d = out / "labels" / tid
        d.mkdir(parents=True, exist_ok=True)
        for name, rgba in (("agreement", colorize_agreement(m1, m2, m3)),
                           ("glads2", colorize_mask(m1, COLORS["glads2"])),
                           ("gladl", colorize_mask(m2, COLORS["gladl"])),
                           ("radd", colorize_mask(m3, COLORS["radd"]))):
            Image.fromarray(rgba, "RGBA").save(d / f"{name}.png", optimize=True)
        stats = label_stats(m1, m2, m3)
        ranking.append((tid, stats))
        c = geom.centroid
        tiles.append({"id": tid, "split": "train", "country": TILE_COUNTRY.get(tid, "Unknown"),
                      "bounds": bounds, "center": [c.x, c.y], "labelStats": stats})
    test_entries = {}
    for tid, geom in test:
        c = geom.centroid
        test_entries[tid] = {"id": tid, "split": "test", "country": TILE_COUNTRY.get(tid, "Unknown"),
                             "bounds": list(geom.bounds), "center": [c.x, c.y],
                             "hectares": {mid: 0.0 for mid in model_ids}}
    tile_polys = dict(test)
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for p, m in zip(subs, [model_entry(p.name[: -len("_submission.geojson")]) for p in subs]):
        fc = json.load(open(p))
        fc = assign_tile_ids(fc, tile_polys)
        for tid, ha in hectares_by_tile(fc).items():
            if tid in test_entries:
                test_entries[tid]["hectares"][m["id"]] = ha
        fc = simplify_fc(fc, tolerance=args.tolerance)
        fc["features"] = [f for f in fc["features"]
                          if f["geometry"] and not shape(f["geometry"]).is_empty]
        with open(pred_dir / f"{m['id']}.geojson", "w") as f:
            json.dump(fc, f, separators=(",", ":"))
    tiles.extend(test_entries.values())
    json.dump(tiles, open(out / "tiles.json", "w"), indent=1)
    json.dump(models, open(out / "models.json", "w"), indent=1)

    print("Disagreement ranking (singleSourcePct, alertedPx >= 2000):")
    rows = sorted((r for r in ranking if r[1]["alertedPx"] >= 2000),
                  key=lambda r: -r[1]["singleSourcePct"])
    for tid, s in rows:
        print(f"{tid}\tsingleSourcePct={s['singleSourcePct']}\tallAgreePct={s['allAgreePct']}\talertedPx={s['alertedPx']}")


if __name__ == "__main__":
    main()
