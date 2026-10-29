import type { LabelKind, LayerState, ModelMeta, Tile } from "./types";

export const FEATURED_MODEL_ID = "temporal_s2s1_norm_pseudo_v1_thr040";

export const DEFAULT_LAYERS: LayerState = {
  pins: true,
  labels: { agreement: false, glads2: false, gladl: false, radd: false },
  labelTiles: "all",
  prediction: false,
  modelId: FEATURED_MODEL_ID,
  gfwAlerts: false,
  treeCoverLoss: false,
};

async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new Error(`Failed to fetch ${url}: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!res.ok) throw new Error(`Failed to load ${url} (HTTP ${res.status})`);
  return (await res.json()) as T;
}

export const loadTiles = (): Promise<Tile[]> => getJson<Tile[]>("/data/tiles.json");
export const loadModels = (): Promise<ModelMeta[]> => getJson<ModelMeta[]>("/data/models.json");

export const labelUrl = (tileId: string, kind: LabelKind): string =>
  `/data/labels/${tileId}/${kind}.png`;
export const predictionUrl = (modelId: string): string =>
  `/data/predictions/${modelId}.geojson`;
