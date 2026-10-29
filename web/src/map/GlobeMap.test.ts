import { expect, test } from "vitest";
import { createAutoRotate, shouldFallback } from "./GlobeMap";

test("basemap falls back to EOX after repeated Esri errors", () => {
  expect(shouldFallback(7)).toBe(false);
  expect(shouldFallback(8)).toBe(true);
});

test("autoRotate stops after user interaction", () => {
  const r = createAutoRotate();
  expect(r.tick(1000)).toBeGreaterThan(0);
  r.interact();
  expect(r.active).toBe(false);
  expect(r.tick(1000)).toBe(0);
});
