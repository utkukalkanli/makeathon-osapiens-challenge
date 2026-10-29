const INPUTS = [
  { name: "S1 radar", note: "sees through clouds" },
  { name: "S2 optical", note: "colour and vegetation" },
  { name: "AEF embeddings", note: "foundation-model features" },
];

const OUTPUTS = [
  { name: "Predictions", note: "per-pixel deforestation map" },
  { name: "Polygons", note: "deforestation areas ≥ 0.5 ha" },
];

export function PipelineGraphic() {
  return (
    <figure className="pipeline" aria-label="Orbitree pipeline">
      <div className="pipeline__inputs">
        {INPUTS.map((i) => (
          <div key={i.name} className="pipeline__node pipeline__node--input">
            <span className="pipeline__name">{i.name}</span>
            <span className="pipeline__note">{i.note}</span>
          </div>
        ))}
      </div>
      <span className="pipeline__arrow" aria-hidden="true">↓</span>
      <div className="pipeline__model">
        <div className="pipeline__node pipeline__node--model">
          <span className="pipeline__name">71-channel temporal U-Net</span>
          <span className="pipeline__note">six years, 2020–2025 · modality masks + dropout</span>
        </div>
        <div className="pipeline__loop">
          <span className="pipeline__loop-icon" aria-hidden="true">↻</span>
          <span className="pipeline__note">
            <strong>2 teachers</strong> (S2-only, S2+S1) pseudo-label unlabelled test tiles,
            uncertain pixels ignored → retrain
          </span>
        </div>
      </div>
      {OUTPUTS.map((s) => (
        <div key={s.name} className="pipeline__stage">
          <span className="pipeline__arrow" aria-hidden="true">↓</span>
          <div className="pipeline__node">
            <span className="pipeline__name">{s.name}</span>
            <span className="pipeline__note">{s.note}</span>
          </div>
        </div>
      ))}
    </figure>
  );
}

export function Approach() {
  return (
    <div className="chapter chapter--approach">
      <p className="eyebrow">Chapter 2 · Our approach</p>
      <h2>Learn from every signal, not one alert</h2>
      <p>
        One model reads radar, optical and embeddings together, and is trained to cope with
        missing sensors.
      </p>
      <PipelineGraphic />
      <dl className="metrics">
        <div className="metric">
          <dt>val F1</dt>
          <dd>0.763</dd>
        </div>
        <div className="metric">
          <dt>val IoU</dt>
          <dd>0.622</dd>
        </div>
      </dl>
      <p className="footnote">
        Validation scores of the final model on held-out tiles. An earlier S2+S1 model
        (s2s1_norm_v1) scored 40.29% Union IoU on the public leaderboard.
      </p>
    </div>
  );
}

export default Approach;
