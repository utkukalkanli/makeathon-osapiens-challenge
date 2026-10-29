import type { ModelMeta, Tile } from "../../data/types";
import { COLORS } from "../../map/sources";
import { Legend } from "../../components/Legend";

/** Approximate tint of the GFW integrated-alerts tiles; the raster is styled by GFW, not us. */
export const GFW_SWATCH = "#e879a6";

export interface UnseenProps {
  tile: Tile | undefined;
  model: ModelMeta | undefined;
}

export function Unseen({ tile, model }: UnseenProps) {
  const ha = tile && model ? tile.hectares?.[model.id] : undefined;
  return (
    <div className="chapter chapter--unseen">
      <p className="eyebrow">Chapter 3 · New ground</p>
      <h2 className="place">{tile?.country ?? "—"}</h2>
      <p className="tile-meta">
        <code>{tile?.id ?? "—"}</code>
        <span className="badge">never seen in training</span>
      </p>
      <div className="stat stat--model">
        <span className="stat__num">
          {ha === undefined ? "—" : ha.toFixed(1)}
          <span className="stat__unit"> ha</span>
        </span>
        <span className="stat__label">of forest loss flagged by our model on this tile</span>
      </div>
      <p>
        Compare the cyan areas with Global Forest Watch alerts: where both appear, independent
        systems see the same change; where only cyan appears, our model sees loss that alerts
        miss, or flags change no alert has confirmed.
      </p>
      <Legend
        items={[
          { color: COLORS.model, label: "Orbitree model", note: model?.label },
          { color: GFW_SWATCH, label: "GFW integrated alerts", note: "GLAD-L, GLAD-S2 and RADD combined" },
        ]}
      />
    </div>
  );
}

export default Unseen;
