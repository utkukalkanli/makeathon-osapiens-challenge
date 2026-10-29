# Orbitree — Design

**Date:** 2026-10-29
**Status:** Draft, awaiting approval

## Name

The app is called **Orbitree** (user's choice): from orbit down to every tree, which is the app's signature zoom. Tagline: "from orbit to every tree" (builder may refine). In the logo or title, visually separate the parts (e.g. Orbi·tree or OrbiTree) so the pronunciation is clear.

**Credits:** "osapiens Terra Makeathon 2026" is shown visibly, in the opening chapter and in a small about/credits area in explore mode. This was a solo project. Credit "Built by Utku Kalkanlı" (my call, using the git author name; user may change), plus a link to the repo.

## Intent

This repo holds our osapiens Terra Makeathon 2026 solution (pixel-level deforestation detection from Sentinel-1, Sentinel-2 and AEF embeddings). We want a public web app that tells the project's story to a jury and to a general audience.

The first second should land a "wow": a high-resolution, realistic 3D Earth the presenter can spin freely. From there a scroll-driven story flies the camera to our tiles. It shows that the weak labels disagree, explains how we approached the problem, and shows our model's predictions in regions it never saw during training. At the end the globe is handed to the viewer for free exploration.

The user presents it live. It must look polished on a laptop and a projector, and it must be easy for anyone to open from a link.

## Goals

- A realistic 3D globe with atmosphere and a space backdrop, with seamless zoom from orbit down to field-level (~10 m) satellite imagery.
- A scroll-driven story (NYT/Pudding style): narrative text on one side, and the globe flies to and reveals layers per chapter.
- Show the challenge's weak labels (GLAD-S2, GLAD-L, RADD) on the 16 train tiles, including where they agree and where they conflict.
- Show our model's predictions (all 9 submission versions) on the 5 test tiles, side by side with public GFW alert layers.
- After the story, a free-explore mode: layer toggles, model-version picker, tile list, per-tile hectares.
- Deployable as a static site on Vercel with no backend.

## Non-goals

- No backend, database, auth or Supabase. *(Builder's note: the user originally mentioned Supabase. We agreed Vercel alone is enough because the data is ~25 MB and static. Supabase can be added later if uploads or comments are ever wanted.)*
- No running models in the browser or on a server.
- No raw challenge Sentinel/AEF imagery. The data is hundreds of GB and the user has no disk space for it. Public basemap imagery stands in for it.
- No retraining or new predictions. We only use the existing submissions.

## Anti-goals (failure even if it "works")

- Visible jank, blank globe or long white loading screens during a live demo.
- **Overclaiming.** The test tiles have no ground truth. Copy must never say our model is "right" or "the truth" where it differs from GLAD/RADD. Say "compare", "where our model sees loss that alerts miss", and so on. Only validation metrics from `APPROACH.md` are stated as performance numbers.
- Something that looks like a generic GIS dashboard. It should feel like a story first.

## Data available (verified)

| What | Where | Size | Notes |
|---|---|---|---|
| Tile boundaries | `s3://osapiens-terra-challenge/makeathon-challenge/metadata/{train,test}_tiles.geojson` (public HTTPS) | KB | 16 train, 5 test. Property `name`, e.g. `18NYH_2_1` |
| Weak labels (train only) | `.../makeathon-challenge/labels/train/{glads2,radd,gladl}/` | 3.9 MB total | 17 glads2, 17 radd, 161 gladl (per-year) GeoTIFFs. Encodings and thresholds in `APPROACH.md` §1.2 and `src/labels.py` |
| Our predictions (test only) | `../MAKEATHON/*.geojson` (sibling folder of this repo) | 1–3 MB each, 9 files | EPSG:4326 polygons, **no properties**. Spatial join with test tile bounds gives tile id (verified: all polygons fall in the 5 test tiles) |
| Presentation | `../MAKEATHON/presentation.html`, `slides/*.html` | — | Source for chapter 3 copy and diagram |
| Metrics | `APPROACH.md` §6 | — | Final model `temporal_s2s1_norm_pseudo_v1`: val F1 0.763, val IoU 0.622. Earlier `s2s1_norm_v1`: leaderboard Union IoU 40.29% |
| Pseudo-labels | `pseudo_v2/*.npz` (in repo, test tiles) | 17 MB | Optional, not required by this design |

Labels exist **only** on train tiles and predictions **only** on test tiles. This is why test tiles are compared against public GFW layers instead.

### External tile services (verified reachable without keys, 2026-10-29)

- **Esri World Imagery:** `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`. Main basemap, attribution required.
- **EOX Sentinel-2 cloudless:** `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg`. Alternative or fallback.
- **GFW integrated alerts:** `https://tiles.globalforestwatch.org/gfw_integrated_alerts/latest/default/{z}/{x}/{y}.png`. Returns rendered PNGs and combines GLAD-L, GLAD-S2 and RADD.
- **GFW per-source alerts:** `wur_radd_alerts`, `umd_glad_landsat_alerts`, `umd_glad_sentinel2_alerts` (same URL shape). All three returned *identical-size* PNGs for the same tile, which suggests encoded data tiles that need client-side decoding. **Builder must verify.** If decoding is impractical, use integrated alerts plus `umd_tree_cover_loss/latest/dynamic` (rendered, verified) and drop per-source GFW layers.

## Decisions made

**User decided:**
- Story layout (option C): scroll narrative left, globe right, ending in free exploration.
- Engine: **MapLibre GL JS** (v5+, globe projection) rather than raw three.js or Cesium.
- Hosting: Vercel only.
- Chapter outline as below.

**My calls (user can override):**
- UI copy in **English**.
- Stack: **Vite + React + TypeScript**, app in `web/` inside this repo; Vercel project root = `web/`.
- Featured/default model: **`temporal_s2s1_norm_pseudo_v1_thr040`** (the final retrain). The other 8 appear in the free-explore picker.
- Primary target: desktop/laptop and projector at 1080p+. Mobile must not break (story stacks vertically, globe on top) but isn't polished.
- Add `.superpowers/` to `.gitignore`.

## Experience

### Chapters

1. **Opening.** The whole Earth slowly auto-rotates in space. 21 tiles glow as pins (green = train/labelled, red = test/predicted). Title, one-line hook on EUDR (the EU Deforestation Regulation), and a "scroll to begin" cue. The user can still grab and spin the globe.
2. **The labels disagree.** Fly to one train tile (builder picks the tile with the most striking disagreement from computed stats). Show a per-pixel agreement overlay: all 3 sources agree / 2 / only 1. Each source can be toggled. A stat states the % of alerted pixels where only one source fires, and explains why (clouds blind optical sensors, radar misses flat terrain, Landsat is annual).
3. **Our approach.** The globe holds on the tile (or pulls back slightly). The left column shows a simplified pipeline graphic based on the presentation: S1 radar + S2 optical + AEF embeddings → temporal U-Net → pseudo-labels → polygons. Headline metric: val F1 0.763 / IoU 0.622.
4. **Unseen regions.** Fly sequentially Colombia → Cameroon/Congo region (33NTE) → Thailand/Laos (47QMA) → Cambodia/Vietnam (48PWA), with one scroll step per stop or a sub-step sequence. Show our polygons over the satellite imagery with GFW alerts toggle-able beside them, plus hectares detected per tile.
5. **Explore.** The narrative collapses into a compact control panel: layer toggles (imagery, labels, our model, GFW), model-version picker (9 versions), tile list (click = fly-to), and per-tile stats. The globe is fully interactive.

Scrolling back up must replay camera states correctly (chapters are declarative camera + layer states, not one-shot animations).

### Visual direction

Dark, space-first. Atmosphere glow on the globe, starfield behind. Overlay colours must read on top of satellite imagery: our model in a bright cool tone (e.g. cyan/blue), labels and alerts in warm tones, and the agreement ramp from yellow to red. Attribution for Esri, GFW and the challenge data is visible but discreet.

## Build-time data pipeline

A one-off script (Python, its own small requirements: `rasterio`, `numpy`, `shapely`, `requests`) produces static files under `web/public/data/`, which are committed:

- `tiles.json`: per tile: id, split, bounds, centroid, country/region name, stats (label agreement % for train; hectares per model version for test).
- Train label overlays: per tile, a transparent PNG of the agreement map, reprojected to EPSG:4326 so it can be placed as a MapLibre image source by its bounds. Per-source binary masks as well, if needed for toggling (a single RGBA PNG with channels per source is fine; builder decides).
- Test predictions: the 9 GeoJSONs with a `tile_id` property added, optionally simplified, plus a `models.json` listing id, display name and threshold.

Label decoding and thresholds must match `APPROACH.md` §1.2 (GLAD-S2 ≥ 3, GLAD-L ≥ 2 for years 21–25, RADD confidence ≥ 2, alerts from 2021 onward). Reuse the decoding logic in `src/labels.py` where possible. The script reads remote data over HTTPS from the public bucket, or downloads only the label folder (3.9 MB). It must not depend on the full dataset being present.

The total committed data should stay small enough for fast first load (target: < 30 MB total, and the opening chapter loads only `tiles.json`).

## Left to the builder

- Story scroll mechanics (IntersectionObserver or a small library).
- Starfield and atmosphere implementation details, and the pin glow effect.
- Exact tile picked for chapter 2, and exact wording of all copy (within the anti-goals).
- Whether to simplify the prediction GeoJSONs, and how.
- Country names for tiles (hard-coded lookup is fine for 21 tiles).
- Handling the per-source GFW layers (see verification note above).

## Deployment

`web/` builds with `vite build` to static files. The user creates the Vercel project and deploys it; nothing is deployed on the user's behalf without confirmation. No environment variables or secrets are needed.
