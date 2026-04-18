"""Merge per-tile prediction GeoJSONs into one uploadable FeatureCollection."""
from __future__ import annotations

import argparse
from pathlib import Path

from submission_utils import merge_geojson_featurecollections


def main(
    input_dir: str = "submission",
    output_path: str = "submission/submission.geojson",
    pattern: str = "*.geojson",
):
    in_dir = Path(input_dir)
    out_path = Path(output_path)
    candidates = sorted(
        p for p in in_dir.glob(pattern)
        if p.is_file() and p.resolve() != out_path.resolve()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No input GeoJSON files found in {in_dir} matching {pattern!r}"
        )

    merged = merge_geojson_featurecollections(candidates, output_path=out_path)
    print(f"merged {len(candidates)} files -> {out_path}")
    print(f"total features={len(merged['features'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="submission")
    ap.add_argument("--output_path", default="submission/submission.geojson")
    ap.add_argument("--pattern", default="*.geojson")
    args = ap.parse_args()
    main(**vars(args))
