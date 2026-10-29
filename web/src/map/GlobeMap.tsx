import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource, RasterLayerSpecification } from "maplibre-gl";
import type { FeatureCollection, Point } from "geojson";
import type { Tile } from "../data/types";
import { ATTRIBUTION, COLORS, EOX_URL, SATELLITE_SOURCE, baseStyle } from "./sources";
import { Starfield } from "./Starfield";

/* ---------- pure helpers (unit-tested) ---------- */

/** Esri tile errors after which the basemap switches to EOX Sentinel-2 cloudless. */
export const FALLBACK_AFTER_ERRORS = 8;

export function shouldFallback(errorCount: number): boolean {
  return errorCount >= FALLBACK_AFTER_ERRORS;
}

export interface FallbackCounter {
  /** Record a failed basemap tile; true once the streak warrants a fallback. */
  error(): boolean;
  /** Record a loaded basemap tile: the failure streak starts over. */
  success(): void;
}

/** Counts *consecutive* basemap failures, so scattered 404s never trigger the fallback. */
export function createFallbackCounter(): FallbackCounter {
  let streak = 0;
  return {
    error: () => shouldFallback(++streak),
    success: () => {
      streak = 0;
    },
  };
}

/** Degrees of spin per second while idling in orbit. */
export const AUTO_ROTATE_DEG_PER_SEC = 3;

export interface AutoRotate {
  /** Rotation delta in degrees for an elapsed frame time; 0 once stopped. */
  tick(dtMs: number): number;
  /** The user touched the globe: stop for good. */
  interact(): void;
  readonly active: boolean;
}

export function createAutoRotate(degPerSec = AUTO_ROTATE_DEG_PER_SEC): AutoRotate {
  let active = true;
  return {
    tick(dtMs) {
      if (!active || dtMs <= 0) return 0;
      return (Math.min(dtMs, 100) / 1000) * degPerSec; // clamp: no jump after a background tab
    },
    interact() {
      active = false;
    },
    get active() {
      return active;
    },
  };
}

/**
 * Whether a direct input event on the globe should stop the idle spin. The wheel only counts
 * when it zooms the map; with scroll-zoom off it scrolls the page (the story) instead.
 */
export function countsAsInteraction(type: string, scrollZoom: boolean): boolean {
  if (type === "wheel") return scrollZoom;
  return type === "mousedown" || type === "touchstart";
}

export function hasWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

export function pinsGeoJSON(tiles: Tile[]): FeatureCollection<Point> {
  return {
    type: "FeatureCollection",
    features: tiles.map((t) => ({
      type: "Feature",
      properties: { id: t.id, split: t.split, country: t.country },
      geometry: { type: "Point", coordinates: t.center },
    })),
  };
}

/* ---------- component ---------- */

const PINS_SOURCE = "pins";
/** Only spin while the whole planet is in view. */
const AUTO_ROTATE_MAX_ZOOM = 4;
const splitColor: maplibregl.ExpressionSpecification = [
  "match",
  ["get", "split"],
  "train",
  COLORS.trainPin,
  COLORS.testPin,
];

function addPins(map: maplibregl.Map, tiles: Tile[]) {
  map.addSource(PINS_SOURCE, { type: "geojson", data: pinsGeoJSON(tiles) });
  map.addLayer({
    id: "pins-glow",
    type: "circle",
    source: PINS_SOURCE,
    paint: {
      "circle-color": splitColor,
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 12, 5, 18, 10, 26],
      "circle-blur": 1,
      "circle-opacity": ["interpolate", ["linear"], ["zoom"], 0, 0.7, 10, 0.55, 13, 0.15],
    },
  });
  map.addLayer({
    id: "pins",
    type: "circle",
    source: PINS_SOURCE,
    paint: {
      "circle-color": splitColor,
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 4, 5, 5, 10, 6.5],
      "circle-stroke-color": "#f8fafc",
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 0, 0.8, 10, 1.5],
      "circle-stroke-opacity": 0.85,
      "circle-opacity": ["interpolate", ["linear"], ["zoom"], 10, 1, 13, 0.35],
    },
  });
}

/**
 * Replace the Esri basemap with EOX Sentinel-2 cloudless, keeping the layer's place in the
 * stack. Re-adding the source (rather than setTiles) drops Esri's failing tile requests
 * cleanly and gives the attribution control the EOX credit.
 */
function switchToEox(map: maplibregl.Map) {
  const layers = map.getStyle().layers;
  const i = layers.findIndex((l) => l.id === SATELLITE_SOURCE);
  if (i < 0) return;
  const layer = layers[i] as RasterLayerSpecification;
  const beforeId = layers[i + 1]?.id;
  map.removeLayer(SATELLITE_SOURCE);
  map.removeSource(SATELLITE_SOURCE);
  map.addSource(SATELLITE_SOURCE, {
    type: "raster",
    tiles: [EOX_URL],
    tileSize: 256,
    maxzoom: 16,
    attribution: ATTRIBUTION.eox,
  });
  map.addLayer(layer, beforeId);
}

export interface GlobeMapProps {
  tiles: Tile[];
  onReady?: (map: maplibregl.Map) => void;
  autoRotate?: boolean;
  /** Wheel zooms the map when true; when false the wheel scrolls the page. Drag stays on. */
  scrollZoom?: boolean;
}

export function GlobeMap({ tiles, onReady, autoRotate = true, scrollZoom = true }: GlobeMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const tilesRef = useRef(tiles);
  const onReadyRef = useRef(onReady);
  const autoRotateRef = useRef(autoRotate);
  const scrollZoomRef = useRef(scrollZoom);
  const rotatorRef = useRef<AutoRotate>(createAutoRotate());
  const [failure, setFailure] = useState<string | null>(null);

  tilesRef.current = tiles;
  onReadyRef.current = onReady;

  // Re-arm the spin whenever the caller switches auto-rotate back on.
  useEffect(() => {
    if (autoRotate && !autoRotateRef.current) rotatorRef.current = createAutoRotate();
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    scrollZoomRef.current = scrollZoom;
    const map = mapRef.current;
    if (!map) return;
    if (scrollZoom) map.scrollZoom.enable();
    else map.scrollZoom.disable();
  }, [scrollZoom]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    if (!hasWebGL()) {
      setFailure("Your browser has WebGL disabled or unavailable, so the 3D globe can't be drawn.");
      return;
    }

    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container,
        style: baseStyle(),
        center: [-20, 8],
        zoom: window.innerWidth < 768 ? 1.1 : 2.1,
        minZoom: 0.5,
        maxZoom: 18.5,
        maxPitch: 75,
        attributionControl: { compact: true, customAttribution: ATTRIBUTION.labels },
        canvasContextAttributes: { antialias: true },
        fadeDuration: 200,
        scrollZoom: scrollZoomRef.current,
      });
    } catch (e) {
      console.error(e);
      setFailure("The 3D globe couldn't start (WebGL context creation failed).");
      return;
    }
    mapRef.current = map;
    if (import.meta.env.DEV) (window as unknown as { __orbitreeMap?: maplibregl.Map }).__orbitreeMap = map;

    // Basemap resilience: after a streak of Esri failures swap to EOX Sentinel-2 cloudless.
    const fallback = createFallbackCounter();
    let fellBack = false;
    let switching = false;
    let disposed = false;
    map.on("sourcedata", (e) => {
      if (e.sourceId === SATELLITE_SOURCE && e.tile && !fellBack) fallback.success();
    });
    map.on("error", (e: maplibregl.ErrorEvent & { sourceId?: string }) => {
      if (e.sourceId === SATELLITE_SOURCE) {
        if (fellBack) {
          console.warn("EOX basemap tile failed:", e.error?.message);
          return;
        }
        if (switching || !fallback.error()) return;
        switching = true;
        // Defer out of the error callback (MapLibre is mid-dispatch here).
        setTimeout(() => {
          if (disposed) return;
          try {
            switchToEox(map);
            fellBack = true;
            console.warn("Esri imagery failed repeatedly; basemap switched to EOX Sentinel-2 cloudless.");
          } catch (err) {
            console.error("Basemap fallback failed", err);
          } finally {
            switching = false;
          }
        }, 0);
        return;
      }
      console.error(e.error);
    });

    // Any direct touch on the globe stops the idle spin.
    const stop = (e: Event) => {
      if (countsAsInteraction(e.type, scrollZoomRef.current)) rotatorRef.current.interact();
    };
    const canvasContainer = map.getCanvasContainer();
    const opts: AddEventListenerOptions = { passive: true };
    canvasContainer.addEventListener("mousedown", stop, opts);
    canvasContainer.addEventListener("touchstart", stop, opts);
    canvasContainer.addEventListener("wheel", stop, opts);

    // Init on style load, not "load": "load" waits for every in-view tile, so one hanging
    // basemap request would otherwise hold back the pins and onReady indefinitely.
    let initialized = false;
    const init = () => {
      if (initialized || disposed) return;
      initialized = true;
      addPins(map, tilesRef.current);
      onReadyRef.current?.(map);
    };
    if (map.isStyleLoaded()) init();
    else map.once("style.load", init);

    let raf = 0;
    let last = performance.now();
    const frame = (now: number) => {
      const dt = now - last;
      last = now;
      if (autoRotateRef.current && !map.isMoving() && map.getZoom() < AUTO_ROTATE_MAX_ZOOM) {
        const delta = rotatorRef.current.tick(dt);
        if (delta > 0) {
          const c = map.getCenter();
          // Earth turns eastward, so the fixed camera drifts west.
          map.jumpTo({ center: [c.lng - delta, c.lat] });
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      canvasContainer.removeEventListener("mousedown", stop);
      canvasContainer.removeEventListener("touchstart", stop);
      canvasContainer.removeEventListener("wheel", stop);
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Keep pins in sync if tiles arrive after the map loaded.
  useEffect(() => {
    mapRef.current?.getSource<GeoJSONSource>(PINS_SOURCE)?.setData(pinsGeoJSON(tiles));
  }, [tiles]);

  return (
    <div className="globe">
      <Starfield />
      {failure ? (
        <div className="globe-fallback" role="alert">
          <p className="globe-fallback__title">The globe needs WebGL</p>
          <p>{failure}</p>
          <p className="globe-fallback__hint">
            Try a recent Chrome, Edge, Firefox or Safari with hardware acceleration enabled.
          </p>
        </div>
      ) : (
        <div ref={containerRef} className="globe-map" />
      )}
    </div>
  );
}

export default GlobeMap;
