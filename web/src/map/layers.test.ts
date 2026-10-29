import { describe, expect, test } from "vitest";
import { DEFAULT_LAYERS } from "../data/load";
import type { ModelMeta, Tile } from "../data/types";
import { layerVisibility } from "./layers";

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
