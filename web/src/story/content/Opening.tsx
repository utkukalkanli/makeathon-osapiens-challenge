import { COLORS } from "../../map/sources";
import { Legend } from "../../components/Legend";
import { Wordmark } from "./Wordmark";

export function Opening() {
  return (
    <div className="chapter chapter--opening">
      <p className="eyebrow">osapiens Terra Makeathon 2026</p>
      <Wordmark />
      <p className="tagline">from orbit to every tree</p>
      <p className="lede">
        The EU Deforestation Regulation requires proof that products were not grown on land
        deforested after 2020.
      </p>
      <p>
        Orbitree reads radar, optical and foundation-model signals from satellites to map forest
        loss, pixel by pixel.
      </p>
      <Legend
        items={[
          { color: COLORS.trainPin, shape: "dot", label: "Training tiles", note: "labelled by alert systems" },
          { color: COLORS.testPin, shape: "dot", label: "Test tiles", note: "predicted by our model, never seen in training" },
        ]}
      />
      <p className="scroll-cue" aria-hidden="true">
        <span>Scroll to begin</span>
        <span className="scroll-cue__arrow">↓</span>
      </p>
    </div>
  );
}

export default Opening;
