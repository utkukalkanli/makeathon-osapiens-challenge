import { expect, test } from "vitest";
import type { Tile } from "../data/types";
import { formatHectares, tileHectares } from "./format";

const testTile: Tile = { id: "T", split: "test", country: "X", bounds: [0, 0, 1, 1], center: [0, 0], hectares: { m: 12.6 } };
const trainTile: Tile = { ...testTile, id: "R", split: "train", hectares: { x: 5 } };

test("formatHectares", () => {
  expect(formatHectares(0)).toBe("0 ha");
  expect(formatHectares(412.3)).toBe("412 ha");
  expect(formatHectares(12345)).toBe("12,345 ha");
});

test("tileHectares missing key is 0", () => {
  expect(tileHectares({ ...testTile, hectares: {} }, "x")).toBe(0);
  expect(tileHectares({ ...testTile, hectares: undefined }, "x")).toBe(0);
  expect(tileHectares(trainTile, "x")).toBe(0);
  expect(tileHectares(testTile, "m")).toBe(12.6);
});
