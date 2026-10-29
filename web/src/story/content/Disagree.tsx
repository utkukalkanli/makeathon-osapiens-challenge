import type { Tile } from "../../data/types";
import { COLORS } from "../../map/sources";
import { Legend } from "../../components/Legend";

const fmtPct = (v: number) => v.toFixed(1);
const fmtInt = (v: number) => v.toLocaleString("en-US");
const share = (pct: number) => (pct === 0 ? "nothing" : `${fmtPct(pct)}% of pixels`);

export interface DisagreeProps {
  tile: Tile | undefined;
  view: "agreement" | "sources";
}

export function Disagree({ tile, view }: DisagreeProps) {
  const s = tile?.labelStats;
  const where = tile ? `this tile in ${tile.country}` : "this tile";

  if (view === "agreement") {
    return (
      <div className="chapter chapter--disagree">
        <p className="eyebrow">Chapter 1 · The labels</p>
        <h2>Three alert systems watch the same forest</h2>
        {s && (
          <>
            <div className="stat">
              <span className="stat__num">
                {fmtPct(s.singleSourcePct)}
                <span className="stat__unit">%</span>
              </span>
              <span className="stat__label">of alerted pixels are flagged by only one source</span>
            </div>
            <p>
              On {where}, {fmtInt(s.alertedPx)} pixels are flagged as forest loss by at least one
              system. All three agree on {fmtPct(s.allAgreePct)}% of them.
            </p>
          </>
        )}
        <Legend
          title="Sources flagging a pixel"
          items={[
            { color: COLORS.agreement1, label: "1 source" },
            { color: COLORS.agreement2, label: "2 sources" },
            { color: COLORS.agreement3, label: "All 3 sources" },
          ]}
        />
      </div>
    );
  }

  return (
    <div className="chapter chapter--disagree">
      <p className="eyebrow">Chapter 1 · The labels</p>
      <h2>Each source has a blind spot</h2>
      <p>
        Every system sees the forest through a different sensor, so each misses something
        {s ? <>, which helps explain why {fmtPct(s.singleSourcePct)}% of alerted pixels have a single witness</> : null}
        . Train on any one source and a model learns its blind spot.
      </p>
      {s && (
        <p>
          On {where}, GLAD-L flags {share(s.gladlPct)}, RADD flags {share(s.raddPct)} and the
          optical Sentinel-2 system flags {share(s.glads2Pct)}.
        </p>
      )}
      <Legend
        title="In general"
        items={[
          { color: COLORS.glads2, label: "GLAD-S2 · Sentinel-2 optical", note: "Clouds blind optical sensors." },
          { color: COLORS.gladl, label: "GLAD-L · Landsat optical", note: "Annual cadence: loss shows up late." },
          {
            color: COLORS.radd,
            label: "RADD · Sentinel-1 radar",
            note: "Sees through clouds, but struggles on flat or low-backscatter terrain.",
          },
        ]}
      />
    </div>
  );
}

export default Disagree;
