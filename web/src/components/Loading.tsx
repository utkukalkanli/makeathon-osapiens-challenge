import { Starfield } from "../map/Starfield";
import { Wordmark } from "../story/content/Wordmark";

/** Wordmark over the starfield while the globe warms up; fades out once `done`. */
export function Loading({ done }: { done: boolean }) {
  return (
    <div className={`loading${done ? " loading--done" : ""}`} aria-hidden={done} role="status">
      <Starfield />
      <div className="loading__mark">
        <Wordmark as="span" />
        <p className="loading__tagline">from orbit to every tree</p>
        <span className="loading__pulse" aria-hidden="true" />
      </div>
    </div>
  );
}
