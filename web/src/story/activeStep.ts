export interface StepRect {
  id: string;
  top: number;
  bottom: number;
}

/**
 * The step whose box contains `anchorY`; otherwise the one whose nearest edge is closest
 * (earlier step wins a tie). `null` when there are no steps.
 */
export function pickActiveStep(rects: StepRect[], anchorY: number): string | null {
  let best: string | null = null;
  let bestDist = Infinity;
  for (const r of rects) {
    if (anchorY >= r.top && anchorY <= r.bottom) return r.id;
    const d = anchorY < r.top ? r.top - anchorY : anchorY - r.bottom;
    if (d < bestDist) {
      bestDist = d;
      best = r.id;
    }
  }
  return best;
}

/** Below this width the map is pinned to the top of the viewport and the story runs beneath. */
export const NARROW_MAX_WIDTH = 767;
/** Fraction of the viewport height the pinned map covers on narrow screens (matches the CSS). */
export const NARROW_MAP_FRACTION = 0.55;

/**
 * Viewport y where a step becomes active: `fraction` of the height on wide screens; on narrow
 * screens the middle of the visible text area below the pinned map.
 */
export function anchorY(width: number, height: number, fraction = 0.5): number {
  if (width > NARROW_MAX_WIDTH) return height * fraction;
  return height * (NARROW_MAP_FRACTION + (1 - NARROW_MAP_FRACTION) / 2);
}
