import { describe, expect, test } from "vitest";
import { DEFAULT_LAYERS } from "../data/load";
import type { ModelMeta, Tile } from "../data/types";
import type { Map as MapLibreMap } from "maplibre-gl";
import { applyLayerState, layerVisibility } from "./layers";

const mk = (id: string, split: "train" | "test"): Tile => ({
  id,
  split,
  country: "X",
  bounds: [0, 0, 1, 1],
  center: [0.5, 0.5],
});
const tiles = [mk("T1", "train"), mk("T2", "train"), mk("TEST1", "test")];
const models: ModelMeta[] = [
  { id: "a", label: "A", threshold: 0.4, featured: true },
  { id: "b", label: "B", threshold: null, featured: false },
];

describe("layerVisibility", () => {
  test("only the selected model's prediction layers are visible", () => {
    const v = layerVisibility({ ...DEFAULT_LAYERS, prediction: true, modelId: "b" }, tiles, models);
    expect(v["pred-b-fill"]).toBe(true);
    expect(v["pred-b-line"]).toBe(true);
    expect(v["pred-a-fill"]).toBe(false);
    expect(v["pred-a-line"]).toBe(false);
  });
  test("predictions hidden when prediction is off", () => {
    const v = layerVisibility(DEFAULT_LAYERS, tiles, models);
    expect(v["pred-a-fill"]).toBe(false);
  });
  test("labels respect labelTiles subset", () => {
    const v = layerVisibility(
      { ...DEFAULT_LAYERS, labels: { agreement: true, glads2: false, gladl: false, radd: false }, labelTiles: ["T1"] },
      tiles,
      models,
    );
    expect(v["label-T1-agreement"]).toBe(true);
    expect(v["label-T2-agreement"]).toBe(false);
    expect(v["label-T1-radd"]).toBe(false);
  });
  test("test tiles never get label layers", () => {
    const keys = Object.keys(layerVisibility(DEFAULT_LAYERS, tiles, models));
    expect(keys.some((k) => k.startsWith("label-TEST1"))).toBe(false);
  });
  test("gfw, tree cover and pins follow state", () => {
    const v = layerVisibility({ ...DEFAULT_LAYERS, gfwAlerts: true, pins: false }, tiles, models);
    expect(v["gfw-alerts"]).toBe(true);
    expect(v["tree-cover-loss"]).toBe(false);
    expect(v["pins"]).toBe(false);
    expect(v["pins-glow"]).toBe(false);
  });
});

function stubMap() {
  const sources = new Map<string, unknown>();
  const layers = new Map<string, Record<string, unknown>>([["pins-glow", {}]]);
  const map = {
    getLayer: (id: string) => layers.get(id),
    getSource: (id: string) => sources.get(id),
    addSource: (id: string, spec: unknown) => void sources.set(id, spec),
    addLayer: (l: { id: string }) => void layers.set(l.id, l as unknown as Record<string, unknown>),
    getLayoutProperty: (id: string) => (layers.get(id)?.layout as { visibility?: string } | undefined)?.visibility,
    setLayoutProperty: (id: string, _p: string, v: string) => {
      layers.get(id)!.layout = { visibility: v };
    },
  };
  return { map: map as unknown as MapLibreMap, sources, layers };
}

describe("applyLayerState laziness", () => {
  test("hidden labels add no label source", () => {
    const { map, sources } = stubMap();
    applyLayerState(map, DEFAULT_LAYERS, tiles, models);
    expect([...sources.keys()].filter((k) => k.startsWith("label-"))).toEqual([]);
  });
  test("visible label adds exactly one source and layer, even when applied twice", () => {
    const { map, sources, layers } = stubMap();
    const st = {
      ...DEFAULT_LAYERS,
      labels: { agreement: true, glads2: false, gladl: false, radd: false },
      labelTiles: ["T1"],
    };
    applyLayerState(map, st, tiles, models);
    applyLayerState(map, st, tiles, models);
    expect([...sources.keys()].filter((k) => k.startsWith("label-"))).toEqual(["label-T1-agreement"]);
    expect([...layers.keys()].filter((k) => k.startsWith("label-"))).toEqual(["label-T1-agreement"]);
    expect(layers.get("label-T1-agreement")!.layout).toEqual({ visibility: "visible" });
  });
  test("does nothing before the style is ready (no pins layer)", () => {
    const { map, sources, layers } = stubMap();
    layers.delete("pins-glow");
    applyLayerState(map, { ...DEFAULT_LAYERS, prediction: true, modelId: "a" }, tiles, models);
    expect(sources.size).toBe(0);
  });
});
