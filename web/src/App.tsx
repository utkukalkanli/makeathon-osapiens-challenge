import { useCallback, useEffect, useRef, useState } from "react";
import type maplibregl from "maplibre-gl";
import type { LayerState, ModelMeta, Step, Tile } from "./data/types";
import { DEFAULT_LAYERS, loadModels, loadTiles } from "./data/load";
import { GlobeMap } from "./map/GlobeMap";
import { addDataLayers, applyLayerState } from "./map/layers";
import { cameraFor } from "./map/camera";
import { CHAPTERS } from "./story/chapters";
import { Story } from "./story/Story";
import { ExplorePanel } from "./explore/ExplorePanel";

const FLY_DURATION_MS = 2500;

export default function App() {
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [step, setStep] = useState<Step>(CHAPTERS[0].steps[0]);

  const [exploreState, setExploreState] = useState<LayerState>(() => ({
    ...DEFAULT_LAYERS,
    ...CHAPTERS.flatMap((c) => c.steps).find((x) => x.id === "explore")?.layers,
    prediction: true,
    labelTiles: "all",
  }));

  const mapRef = useRef<maplibregl.Map | null>(null);
  const tilesRef = useRef(tiles);
  const modelsRef = useRef(models);
  const stepRef = useRef(step);
  /** Step id the camera last flew to, so data arriving late doesn't re-fly. */
  const flownRef = useRef<string | null>(null);
  tilesRef.current = tiles;
  modelsRef.current = models;
  stepRef.current = step;
  const exploreStateRef = useRef(exploreState);
  exploreStateRef.current = exploreState;

  useEffect(() => {
    let cancelled = false;
    loadTiles()
      .then((t) => !cancelled && setTiles(t))
      .catch((e) => console.error(e));
    loadModels()
      .then((m) => !cancelled && setModels(m))
      .catch((e) => console.error(e));
    return () => {
      cancelled = true;
    };
  }, []);

  /** Bring the map in line with the current step; safe to call any time. */
  const sync = useCallback(() => {
    const map = mapRef.current;
    const t = tilesRef.current;
    const s = stepRef.current;
    if (!map || t.length === 0) return;
    if (flownRef.current !== s.id) {
      flownRef.current = s.id; // set first: a bad target is logged once, not on every sync
      try {
        // A new flyTo interrupts the one in flight, so fast scrolling settles on the last step.
        map.flyTo({ ...cameraFor(s.camera, t), duration: FLY_DURATION_MS, essential: true });
      } catch (e) {
        console.error(e);
      }
    }
    const state =
      s.id === "explore" ? exploreStateRef.current : { ...DEFAULT_LAYERS, ...s.layers };
    applyLayerState(map, state, t, modelsRef.current);
  }, []);

  const onFlyTo = useCallback((tileId: string) => {
    const map = mapRef.current;
    if (!map) return;
    try {
      map.flyTo({ ...cameraFor({ tileId }, tilesRef.current), duration: FLY_DURATION_MS, essential: true });
    } catch (e) {
      console.error(e);
    }
  }, []);

  const onReady = useCallback(
    (map: maplibregl.Map) => {
      mapRef.current = map;
      flownRef.current = null; // a fresh map (e.g. StrictMode remount) has not flown anywhere
      addDataLayers(map, tilesRef.current, modelsRef.current);
      sync();
    },
    [sync],
  );

  useEffect(sync, [step, tiles, models, exploreState, sync]);

  const explore = step.id === "explore";

  return (
    <main className={`app${explore ? " app--explore" : ""}`}>
      <div className="app-map">
        <GlobeMap
          tiles={tiles}
          onReady={onReady}
          autoRotate={!!step.autoRotate}
          scrollZoom={explore}
        />
      </div>
      {explore && (
        <ExplorePanel
          tiles={tiles}
          models={models}
          state={exploreState}
          onChange={setExploreState}
          onFlyTo={onFlyTo}
        />
      )}
      <Story chapters={CHAPTERS} tiles={tiles} models={models} onStepChange={setStep} />
    </main>
  );
}
