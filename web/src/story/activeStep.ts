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
