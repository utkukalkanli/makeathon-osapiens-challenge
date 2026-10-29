import type { Tile } from "../data/types";

const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function formatHectares(ha: number): string {
  return `${nf.format(ha)} ha`;
}

/** Predicted hectares for a model on a tile; 0 for train tiles or a missing key. */
export function tileHectares(tile: Tile, modelId: string): number {
  if (tile.split !== "test") return 0;
  return tile.hectares?.[modelId] ?? 0;
}
