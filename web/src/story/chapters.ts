import type { Chapter } from "../data/types";

/** Laos train tile with the most visible label disagreement (controller ruling, Task 2). */
export const DISAGREE_TILE = "18NWJ_8_9";

const NO_LABELS = { agreement: false, glads2: false, gladl: false, radd: false };

export const CHAPTERS: Chapter[] = [
  {
    id: "opening",
    steps: [
      {
        id: "opening",
        camera: { center: [10, 12], zoom: 1.6, pitch: 0, bearing: 0 },
        autoRotate: true,
        layers: { pins: true },
      },
    ],
  },
  {
    id: "disagree",
    steps: [
      {
        id: "disagree-agreement",
        camera: { tileId: DISAGREE_TILE },
        layers: { labels: { ...NO_LABELS, agreement: true }, labelTiles: [DISAGREE_TILE] },
      },
      {
        id: "disagree-sources",
        camera: { tileId: DISAGREE_TILE },
        layers: {
          labels: { agreement: false, glads2: true, gladl: true, radd: true },
          labelTiles: [DISAGREE_TILE],
        },
      },
    ],
  },
  {
    id: "approach",
    steps: [
      {
        id: "approach",
        camera: { tileId: DISAGREE_TILE, zoom: 9 },
        layers: { labels: { ...NO_LABELS, agreement: true } },
      },
    ],
  },
  {
    id: "unseen",
    // 47QMA_6_2 is skipped: the featured model has 0 polygons there.
    steps: [
      { id: "unseen-colombia", camera: { tileId: "18NYH_2_1" }, layers: { prediction: true, gfwAlerts: true } },
      { id: "unseen-cameroon", camera: { tileId: "33NTE_5_1" }, layers: { prediction: true, gfwAlerts: true } },
      { id: "unseen-cambodia", camera: { tileId: "48PWA_0_6" }, layers: { prediction: true, gfwAlerts: true } },
    ],
  },
  {
    id: "explore",
    steps: [
      {
        id: "explore",
        camera: { center: [40, 8], zoom: 1.8, pitch: 0, bearing: 0 },
        layers: { pins: true },
      },
    ],
  },
];
