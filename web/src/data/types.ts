export type Bounds = [number, number, number, number];

export interface Tile {
  id: string;
  split: "train" | "test";
  country: string;
  bounds: Bounds;
  center: [number, number];
  labelStats?: {
    alertedPx: number;
    singleSourcePct: number;
    allAgreePct: number;
    glads2Pct: number;
    gladlPct: number;
    raddPct: number;
  };
  hectares?: Record<string, number>;
}

export interface ModelMeta {
  id: string;
  label: string;
  threshold: number | null;
  featured: boolean;
}

export type LabelKind = "agreement" | "glads2" | "gladl" | "radd";

export interface LayerState {
  pins: boolean;
  labels: Record<LabelKind, boolean>;
  labelTiles: string[] | "all";
  prediction: boolean;
  modelId: string;
  gfwAlerts: boolean;
  treeCoverLoss: boolean;
}

export interface Camera {
  center: [number, number];
  zoom: number;
  pitch: number;
  bearing: number;
}

export type CameraTarget =
  | { tileId: string; zoom?: number; pitch?: number; bearing?: number }
  | Camera;

export interface Step {
  id: string;
  camera: CameraTarget;
  layers: Partial<LayerState>;
  autoRotate?: boolean;
}

export interface Chapter {
  id: string;
  steps: Step[];
}
