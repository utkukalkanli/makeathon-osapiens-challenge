import type { Camera, CameraTarget, Tile } from "../data/types";

export const TILE_ZOOM = 12.2;

/** Resolve a step's camera target to a concrete camera. Throws on an unknown tile id. */
export function cameraFor(target: CameraTarget, tiles: Tile[]): Camera {
  if (!("tileId" in target)) {
    return { center: target.center, zoom: target.zoom, pitch: target.pitch, bearing: target.bearing };
  }
  const tile = tiles.find((t) => t.id === target.tileId);
  if (!tile) throw new Error(`Unknown tile id in camera target: ${target.tileId}`);
  return {
    center: tile.center,
    zoom: target.zoom ?? TILE_ZOOM,
    pitch: target.pitch ?? 0,
    bearing: target.bearing ?? 0,
  };
}
