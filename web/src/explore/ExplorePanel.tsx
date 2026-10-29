import type { LabelKind, LayerState, ModelMeta, Tile } from "../data/types";
import { Credits } from "../components/Credits";
import { formatHectares, tileHectares } from "./format";

export interface ExplorePanelProps {
  tiles: Tile[];
  models: ModelMeta[];
  state: LayerState;
  onChange: (s: LayerState) => void;
  onFlyTo: (tileId: string) => void;
}

const LABEL_TOGGLES: { kind: LabelKind; label: string }[] = [
  { kind: "agreement", label: "Label agreement" },
  { kind: "glads2", label: "GLAD-S2" },
  { kind: "gladl", label: "GLAD-L" },
  { kind: "radd", label: "RADD (labels)" },
];

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="explore__toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function ExplorePanel({ tiles, models, state, onChange, onFlyTo }: ExplorePanelProps) {
  const set = (patch: Partial<LayerState>) => onChange({ ...state, ...patch });
  const ordered = [...models].sort((a, b) => Number(b.featured) - Number(a.featured));

  const byCountry = new Map<string, Tile[]>();
  for (const t of tiles) byCountry.set(t.country, [...(byCountry.get(t.country) ?? []), t]);
  const countries = [...byCountry.keys()].sort();

  return (
    <aside className="explore" aria-label="Explore">
      <h2 className="explore__title">Explore</h2>

      <section className="explore__group">
        <h3>Layers</h3>
        {LABEL_TOGGLES.map(({ kind, label }) => (
          <Toggle
            key={kind}
            label={label}
            checked={state.labels[kind]}
            onChange={(v) => set({ labels: { ...state.labels, [kind]: v } })}
          />
        ))}
        <Toggle label="Orbitree model" checked={state.prediction} onChange={(v) => set({ prediction: v })} />
        <Toggle label="RADD alerts (Global Forest Watch)" checked={state.gfwAlerts} onChange={(v) => set({ gfwAlerts: v })} />
        <Toggle label="Tree cover loss (Hansen/UMD)" checked={state.treeCoverLoss} onChange={(v) => set({ treeCoverLoss: v })} />
      </section>

      <section className="explore__group">
        <h3>Model version</h3>
        <select
          className="explore__select"
          aria-label="Model version"
          value={state.modelId}
          onChange={(e) => set({ modelId: e.target.value })}
        >
          {ordered.map((m) => (
            <option key={m.id} value={m.id}>
              {m.featured ? "★ " : ""}
              {m.label}
            </option>
          ))}
        </select>
      </section>

      <section className="explore__group">
        <h3>Tiles</h3>
        {countries.map((c) => (
          <div key={c} className="explore__country">
            <h4>{c}</h4>
            <ul>
              {byCountry.get(c)!.map((t) => (
                <li key={t.id}>
                  <button type="button" className="explore__tile" onClick={() => onFlyTo(t.id)}>
                    <code>{t.id}</code>
                    <span className={`badge badge--${t.split}`}>{t.split}</span>
                    {t.split === "test" && (
                      <span className="explore__ha">{formatHectares(tileHectares(t, state.modelId))}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      <button type="button" className="explore__back" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
        ↑ Back to story
      </button>
      <Credits />
    </aside>
  );
}

export default ExplorePanel;
