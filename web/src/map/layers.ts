import type { ImageSourceSpecification, Map as MapLibreMap } from "maplibre-gl";
import { labelUrl, predictionUrl } from "../data/load";
import type { LabelKind, LayerState, ModelMeta, Tile } from "../data/types";
import { ATTRIBUTION, COLORS, GFW_ALERTS_URL, TREE_COVER_LOSS_URL } from "./sources";

export const LABEL_KINDS: LabelKind[] = ["agreement", "glads2", "gladl", "radd"];

export const GFW_ALERTS_ID = "gfw-alerts";
export const TREE_COVER_LOSS_ID = "tree-cover-loss";
/** Everything data-related is inserted below this layer so pins stay on top. */
const PINS_GLOW_ID = "pins-glow";

export const labelId = (tileId: string, kind: LabelKind) => `label-${tileId}-${kind}`;
export const predSourceId = (modelId: string) => `pred-${modelId}`;
export const predFillId = (modelId: string) => `pred-${modelId}-fill`;
export const predLineId = (modelId: string) => `pred-${modelId}-line`;

const trainTiles = (tiles: Tile[]) => tiles.filter((t) => t.split === "train");

/** Visibility for every layer id this module manages (plus pins). Pure. */
export function layerVisibility(
  state: LayerState,
  tiles: Tile[],
  models: ModelMeta[],
): Record<string, boolean> {
  const v: Record<string, boolean> = {
    [TREE_COVER_LOSS_ID]: state.treeCoverLoss,
    [GFW_ALERTS_ID]: state.gfwAlerts,
    pins: state.pins,
    "pins-glow": state.pins,
  };
  for (const t of trainTiles(tiles)) {
    const tileOn = state.labelTiles === "all" || state.labelTiles.includes(t.id);
    for (const k of LABEL_KINDS) v[labelId(t.id, k)] = tileOn && state.labels[k];
  }
  for (const m of models) {
    const on = state.prediction && state.modelId === m.id;
    v[predFillId(m.id)] = on;
    v[predLineId(m.id)] = on;
  }
  return v;
}

function imageCoordinates([w, s, e, n]: Tile["bounds"]): ImageSourceSpecification["coordinates"] {
  return [
    [w, n],
    [e, n],
    [e, s],
    [w, s],
  ];
}

const beforeId = (map: MapLibreMap) => (map.getLayer(PINS_GLOW_ID) ? PINS_GLOW_ID : undefined);

/** Adds GFW raster sources and train-tile label image sources, all hidden. */
export function addDataLayers(map: MapLibreMap, tiles: Tile[], _models: ModelMeta[]): void {
  const before = beforeId(map);
  const hidden = { visibility: "none" } as const;

  if (!map.getSource(TREE_COVER_LOSS_ID)) {
    map.addSource(TREE_COVER_LOSS_ID, {
      type: "raster",
      tiles: [TREE_COVER_LOSS_URL],
      tileSize: 256,
      maxzoom: 12,
      attribution: ATTRIBUTION.treeCoverLoss,
    });
    map.addLayer(
      { id: TREE_COVER_LOSS_ID, type: "raster", source: TREE_COVER_LOSS_ID, layout: hidden, paint: { "raster-opacity": 0.7 } },
      before,
    );
  }
  if (!map.getSource(GFW_ALERTS_ID)) {
    map.addSource(GFW_ALERTS_ID, {
      type: "raster",
      tiles: [GFW_ALERTS_URL],
      tileSize: 256,
      maxzoom: 12,
      attribution: ATTRIBUTION.gfw,
    });
    map.addLayer(
      { id: GFW_ALERTS_ID, type: "raster", source: GFW_ALERTS_ID, layout: hidden, paint: { "raster-opacity": 0.85 } },
      before,
    );
  }

  for (const t of trainTiles(tiles)) {
    for (const k of LABEL_KINDS) {
      const id = labelId(t.id, k);
      if (map.getSource(id)) continue;
      map.addSource(id, {
        type: "image",
        url: labelUrl(t.id, k),
        coordinates: imageCoordinates(t.bounds),
      });
      map.addLayer(
        {
          id,
          type: "raster",
          source: id,
          layout: hidden,
          paint: { "raster-opacity": 0.9, "raster-resampling": "nearest", "raster-fade-duration": 0 },
        },
        before,
      );
    }
  }
}

/** Add one model's prediction source and layers on first need. */
function ensurePrediction(map: MapLibreMap, modelId: string): void {
  const src = predSourceId(modelId);
  if (map.getSource(src)) return;
  const before = beforeId(map);
  map.addSource(src, { type: "geojson", data: predictionUrl(modelId) });
  map.addLayer(
    {
      id: predFillId(modelId),
      type: "fill",
      source: src,
      layout: { visibility: "none" },
      paint: { "fill-color": COLORS.model, "fill-opacity": 0.35 },
    },
    before,
  );
  map.addLayer(
    {
      id: predLineId(modelId),
      type: "line",
      source: src,
      layout: { visibility: "none" },
      paint: { "line-color": COLORS.model, "line-width": 1.5 },
    },
    before,
  );
}

/** Idempotent: sync every managed layer's visibility with the state. */
export function applyLayerState(
  map: MapLibreMap,
  state: LayerState,
  tiles: Tile[],
  models: ModelMeta[],
): void {
  if (state.prediction && models.some((m) => m.id === state.modelId)) ensurePrediction(map, state.modelId);
  const vis = layerVisibility(state, tiles, models);
  for (const [id, on] of Object.entries(vis)) {
    if (!map.getLayer(id)) continue;
    const want = on ? "visible" : "none";
    if (map.getLayoutProperty(id, "visibility") !== want) map.setLayoutProperty(id, "visibility", want);
  }
}
