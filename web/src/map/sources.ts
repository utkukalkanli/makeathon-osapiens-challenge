import type { StyleSpecification } from "maplibre-gl";

// Esri and EOX use {z}/{y}/{x}; GFW uses {z}/{x}/{y}.
export const ESRI_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
export const EOX_URL =
  "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg";
export const GFW_ALERTS_URL =
  "https://tiles.globalforestwatch.org/gfw_integrated_alerts/latest/default/{z}/{x}/{y}.png";
export const TREE_COVER_LOSS_URL =
  "https://tiles.globalforestwatch.org/umd_tree_cover_loss/latest/dynamic/{z}/{x}/{y}.png";

export const ATTRIBUTION = {
  esri: "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  gfw: "Alerts © Global Forest Watch (UMD GLAD, WUR RADD)",
  treeCoverLoss: "Hansen/UMD/Google/USGS/NASA",
  labels: "Weak labels: GLAD-S2, GLAD-L, RADD via osapiens Terra Makeathon 2026",
  // Shown only if the basemap falls back to EOX.
  eox: "Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023)",
} as const;

export const COLORS = {
  model: "#22d3ee",
  glads2: "#facc15",
  gladl: "#fb923c",
  radd: "#f43f5e",
  agreement1: "#fde68a",
  agreement2: "#f97316",
  agreement3: "#dc2626",
  trainPin: "#4ade80",
  testPin: "#f87171",
} as const;

export const BACKGROUND = "#02040a";
export const SATELLITE_SOURCE = "satellite";

export function baseStyle(): StyleSpecification {
  return {
    version: 8,
    projection: { type: "globe" },
    sources: {
      [SATELLITE_SOURCE]: {
        type: "raster",
        tiles: [ESRI_URL],
        tileSize: 256,
        maxzoom: 19,
        attribution: ATTRIBUTION.esri,
      },
    },
    // Viewport-anchored sun from the left: a lit limb that stays put while the globe spins.
    light: { anchor: "viewport", position: [1.5, 270, 20] },
    sky: {
      "sky-color": BACKGROUND,
      "horizon-color": "#5aa9ff",
      "fog-color": "#0b1a33",
      "sky-horizon-blend": 0.6,
      "horizon-fog-blend": 0.7,
      "fog-ground-blend": 0.9,
      // Full atmosphere glow from orbit, fades out as we descend to the trees.
      "atmosphere-blend": ["interpolate", ["linear"], ["zoom"], 0, 1, 5, 1, 7, 0],
    },
    layers: [
      // Dark ocean/land fallback painted on the globe until imagery arrives.
      { id: "background", type: "background", paint: { "background-color": "#050b16" } },
      {
        id: SATELLITE_SOURCE,
        type: "raster",
        source: SATELLITE_SOURCE,
        paint: { "raster-fade-duration": 250, "raster-saturation": 0.05, "raster-contrast": 0.05 },
      },
    ],
  };
}
