import { afterEach, expect, test, vi } from "vitest";
import { labelUrl, loadTiles, predictionUrl } from "./load";

afterEach(() => vi.unstubAllGlobals());

test("loadTiles parses tiles.json", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, json: async () => [{ id: "A", split: "train", country: "X", bounds: [0, 0, 1, 1], center: [0.5, 0.5] }] })));
  expect((await loadTiles())[0].id).toBe("A");
});

test("loadTiles throws a readable error on HTTP failure", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })));
  await expect(loadTiles()).rejects.toThrow(/tiles\.json/);
});

test("url helpers", () => {
  expect(labelUrl("X", "radd")).toBe("/data/labels/X/radd.png");
  expect(predictionUrl("m")).toBe("/data/predictions/m.geojson");
});
