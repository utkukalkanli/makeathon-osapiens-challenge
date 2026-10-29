import { expect, test } from "vitest";
import type { Tile } from "../data/types";
import { cameraFor } from "./camera";

const tiles: Tile[] = [
  { id: "T1", split: "train", country: "X", bounds: [0, 0, 1, 1], center: [0.5, 0.5] },
  { id: "T2", split: "test", country: "Y", bounds: [2, 2, 3, 3], center: [2.5, 2.5] },
];

test("cameraFor resolves a tile target deterministically", () => {
  expect(cameraFor({ tileId: "T1" }, tiles)).toEqual({ center: tiles[0].center, zoom: 12.2, pitch: 0, bearing: 0 });
});

test("cameraFor honours tile-target overrides", () => {
  expect(cameraFor({ tileId: "T2", zoom: 9, pitch: 30, bearing: 10 }, tiles)).toEqual({
    center: [2.5, 2.5],
    zoom: 9,
    pitch: 30,
    bearing: 10,
  });
});

test("cameraFor passes an explicit camera through", () => {
  const cam = { center: [10, 12] as [number, number], zoom: 1.6, pitch: 0, bearing: 0 };
  expect(cameraFor(cam, tiles)).toEqual(cam);
});

test("cameraFor throws on unknown tile", () => {
  expect(() => cameraFor({ tileId: "nope" }, tiles)).toThrow(/nope/);
});
