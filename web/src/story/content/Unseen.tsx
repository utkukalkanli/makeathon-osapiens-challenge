import type { ModelMeta, Tile } from "../../data/types";
import { COLORS } from "../../map/sources";
import { Legend } from "../../components/Legend";

export interface UnseenProps {
  tile: Tile | undefined;
  model: ModelMeta | undefined;
}

export function Unseen({ tile, model }: UnseenProps) {
  const ha = tile && model ? tile.hectares?.[model.id] : undefined;
  return (
    <div className="chapter chapter--unseen">
      <p className="eyebrow">Chapter 3 · Unlabelled regions</p>
      <h2 className="place">{tile?.country ?? "—"}</h2>
      <p className="tile-meta">
        <code>{tile?.id ?? "—"}</code>
        <span className="badge">No labels from here in training</span>
      </p>
      <div className="stat stat--model">
        <span className="stat__num">
          {ha === undefined ? "—" : ha.toFixed(1)}
          <span className="stat__unit"> ha</span>
        </span>
        <span className="stat__label">of forest loss flagged by our model on this tile</span>
      </div>
      <p>
        Compare the cyan areas with RADD radar alerts from Global Forest Watch, a system
        independent of our model: where both appear, they see the same change; where only cyan
        appears, the model flags change that no alert has confirmed.
      </p>
      <Legend
        items={[
          { color: COLORS.model, label: "Orbitree model", note: model?.label },
          { color: COLORS.gfwAlerts, label: "RADD alerts (Global Forest Watch)", note: "radar-based, independent of our model" },
        ]}
      />
    </div>
  );
}

export default Unseen;
