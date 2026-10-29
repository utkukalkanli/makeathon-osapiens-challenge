import { useEffect, useRef } from "react";

/** Deterministic PRNG so the sky is identical on every load (mulberry32). */
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TINTS = ["255,255,255", "202,220,255", "255,236,210", "180,205,255"];

function paint(canvas: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const rand = rng(20261029);
  // A faint diagonal Milky Way band: stars are denser near it.
  const bandAt = (x: number) => h * 0.85 - x * (h / w) * 0.7;
  const count = Math.round((w * h) / 1400);
  for (let i = 0; i < count; i++) {
    const x = rand() * w;
    let y = rand() * h;
    if (rand() < 0.35) y = bandAt(x) + (rand() + rand() + rand() - 1.5) * h * 0.18;
    const m = rand();
    const r = m > 0.995 ? 1.5 : m > 0.97 ? 1.0 : 0.55;
    const alpha = 0.25 + rand() * (r > 1 ? 0.75 : 0.5);
    const tint = TINTS[Math.floor(rand() * TINTS.length)];
    ctx.fillStyle = `rgba(${tint},${alpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (r > 1.2) {
      const g = ctx.createRadialGradient(x, y, 0, x, y, 6);
      g.addColorStop(0, `rgba(${tint},0.35)`);
      g.addColorStop(1, `rgba(${tint},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(x - 6, y - 6, 12, 12);
    }
  }
}

/** Static, seeded starfield drawn once per resize behind the transparent map canvas. */
export function Starfield() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    paint(canvas);
    let t = 0;
    const onResize = () => {
      clearTimeout(t);
      t = window.setTimeout(() => paint(canvas), 120);
    };
    window.addEventListener("resize", onResize);
    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
    };
  }, []);
  return <canvas ref={ref} className="starfield" aria-hidden="true" />;
}
