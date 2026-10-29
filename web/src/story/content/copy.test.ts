import { describe, expect, test } from "vitest";
import { createElement } from "react";
import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { ModelMeta, Tile } from "../../data/types";
import { CHAPTERS, DISAGREE_TILE } from "../chapters";
import { Story } from "../Story";
import { Opening } from "./Opening";
import { Disagree } from "./Disagree";
import { Approach } from "./Approach";
import { Unseen } from "./Unseen";
import { Credits } from "../../components/Credits";

const FEATURED = "temporal_s2s1_norm_pseudo_v1_thr040";

const MODELS: ModelMeta[] = [
  { id: FEATURED, label: "S2+S1 · norm · pseudo v1 · thr 0.40", threshold: 0.4, featured: true },
  { id: "other", label: "Other", threshold: null, featured: false },
];

const tile = (id: string, country: string, extra: Partial<Tile> = {}): Tile => ({
  id,
  split: "test",
  country,
  bounds: [0, 0, 1, 1],
  center: [0.5, 0.5],
  ...extra,
});

const TILES: Tile[] = [
  tile(DISAGREE_TILE, "Laos", {
    split: "train",
    labelStats: {
      alertedPx: 182445,
      singleSourcePct: 66.3,
      allAgreePct: 0,
      glads2Pct: 0,
      gladlPct: 9.7,
      raddPct: 13.7,
    },
  }),
  tile("18NYH_2_1", "Colombia", { hectares: { [FEATURED]: 725.7, other: 1 } }),
  tile("33NTE_5_1", "Cameroon", { hectares: { [FEATURED]: 309.6 } }),
  tile("48PWA_0_6", "Cambodia", { hectares: { [FEATURED]: 534.8 } }),
];

/** Rendered visible text: markup with tags stripped and entities decoded. */
function text(el: ReactElement): string {
  return renderToStaticMarkup(el)
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

const FORBIDDEN = /\b(correct|truth|accurate|right)\b|never seen/i;

const laos = TILES[0];
const CASES: [string, ReactElement][] = [
  ["Opening", createElement(Opening)],
  ["Disagree agreement", createElement(Disagree, { tile: laos, view: "agreement" })],
  ["Disagree sources", createElement(Disagree, { tile: laos, view: "sources" })],
  ["Disagree loading", createElement(Disagree, { tile: undefined, view: "agreement" })],
  ["Approach", createElement(Approach)],
  ...TILES.slice(1).map(
    (t): [string, ReactElement] => [`Unseen ${t.country}`, createElement(Unseen, { tile: t, model: MODELS[0] })],
  ),
  ["Unseen loading", createElement(Unseen, { tile: undefined, model: undefined })],
  ["Credits", createElement(Credits)],
  ["Story", createElement(Story, { chapters: CHAPTERS, tiles: TILES, models: MODELS, onStepChange: () => {} })],
];

describe("honest copy", () => {
  test.each(CASES)("%s makes no correctness claims", (_name, el) => {
    const t = text(el);
    expect(t.length).toBeGreaterThan(0);
    expect(t).not.toMatch(FORBIDDEN);
  });
});

test("Approach shows the validation metrics", () => {
  const t = text(createElement(Approach));
  expect(t).toContain("0.763");
  expect(t).toContain("0.622");
  expect(t).toMatch(/71-channel/);
  expect(t).toMatch(/2 teachers/);
  expect(t).toMatch(/retrain/i);
});

test("Opening has the wordmark, tagline, EUDR line and pin legend", () => {
  const t = text(createElement(Opening));
  expect(t).toContain("from orbit to every tree");
  expect(t).toMatch(/EU Deforestation Regulation/);
  expect(t).toMatch(/2020/);
  expect(t).toContain("osapiens Terra Makeathon 2026");
  expect(t).toMatch(/labelled/i);
  expect(t).toMatch(/no labels from these regions were used in training/i);
});

test("Disagree reads the single-source share and blind spots from the tile", () => {
  const t = text(createElement(Disagree, { tile: laos, view: "sources" }));
  expect(t).toContain("66.3");
  expect(t).toMatch(/clouds/i);
  expect(t).toMatch(/annual/i);
  expect(t).toMatch(/flat|low-backscatter/i);
});

test("Unseen reads hectares for the featured model and says no labels from here were used", () => {
  const t = text(createElement(Unseen, { tile: TILES[1], model: MODELS[0] }));
  expect(t).toContain("Colombia");
  expect(t).toContain("18NYH_2_1");
  expect(t).toContain("725.7");
  expect(t).toMatch(/No labels from here in training/);
  expect(t).toMatch(/RADD alerts \(Global Forest Watch\)/);
});

test("Story maps every step to real content", () => {
  const t = text(createElement(Story, { chapters: CHAPTERS, tiles: TILES, models: MODELS, onStepChange: () => {} }));
  for (const s of ["66.3", "725.7", "309.6", "534.8", "0.763", "Cameroon", "Cambodia"]) expect(t).toContain(s);
});

test("Credits links the GitHub repo in a new tab", () => {
  const html = renderToStaticMarkup(createElement(Credits));
  expect(html).toContain('href="https://github.com/utkukalkanli/makeathon-osapiens-challenge"');
  expect(html).toContain('target="_blank"');
  expect(html).toMatch(/rel="noopener[^"]*"/);
  expect(text(createElement(Credits))).toMatch(/Built by Utku Kalkanlı/);
});
