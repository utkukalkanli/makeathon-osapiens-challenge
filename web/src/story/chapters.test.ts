import { expect, test } from "vitest";
import type { Tile } from "../data/types";
import { cameraFor } from "../map/camera";
import tilesJson from "../../public/data/tiles.json";
import { CHAPTERS, DISAGREE_TILE } from "./chapters";

const tiles = tilesJson as unknown as Tile[];
const steps = CHAPTERS.flatMap((c) => c.steps);

test("every chapter tile id exists in tiles.json", () => {
  expect(tiles.length).toBeGreaterThan(0);
  for (const s of steps) expect(() => cameraFor(s.camera, tiles)).not.toThrow();
  for (const s of steps) {
    const lt = s.layers.labelTiles;
    if (Array.isArray(lt)) for (const id of lt) expect(tiles.some((t) => t.id === id)).toBe(true);
  }
});

test("chapter order is fixed", () => {
  expect(CHAPTERS.map((c) => c.id)).toEqual(["opening", "disagree", "approach", "unseen", "explore"]);
});

test("step ids are unique and match the brief", () => {
  expect(steps.map((s) => s.id)).toEqual([
    "opening",
    "disagree-agreement",
    "disagree-sources",
    "approach",
    "unseen-colombia",
    "unseen-cameroon",
    "unseen-cambodia",
    "explore",
  ]);
});

test("disagree tile is the controller-ruled Laos train tile", () => {
  expect(DISAGREE_TILE).toBe("48QWD_2_2");
  expect(tiles.find((t) => t.id === DISAGREE_TILE)?.split).toBe("train");
});

test("only the opening step auto-rotates; unseen steps show prediction + GFW", () => {
  expect(steps.filter((s) => s.autoRotate).map((s) => s.id)).toEqual(["opening"]);
  for (const s of steps.filter((s) => s.id.startsWith("unseen-"))) {
    expect(s.layers.prediction).toBe(true);
    expect(s.layers.gfwAlerts).toBe(true);
  }
});
