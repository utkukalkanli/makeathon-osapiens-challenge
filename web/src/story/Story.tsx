import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import type { Chapter, Step } from "../data/types";
import { pickActiveStep } from "./activeStep";

/** Fraction of the viewport height where a step becomes active. */
export const ANCHOR = 0.5;

/** Placeholder copy; Task 7 replaces this with a step id -> content component map. */
const PLACEHOLDER: Record<string, ReactNode> = {
  opening: <><h1>Orbitree</h1><p>Watching the world's forests from orbit.</p></>,
  "disagree-agreement": <><h2>Three alert systems</h2><p>Where do GLAD-S2, GLAD-L and RADD agree?</p></>,
  "disagree-sources": <><h2>They often disagree</h2><p>Each source on its own tells a different story.</p></>,
  approach: <><h2>Our approach</h2><p>Learning from noisy, conflicting labels.</p></>,
  "unseen-colombia": <><h2>Unseen: Colombia</h2><p>Predictions on a region the model never saw.</p></>,
  "unseen-cameroon": <><h2>Unseen: Cameroon</h2><p>Predictions against GFW integrated alerts.</p></>,
  "unseen-cambodia": <><h2>Unseen: Cambodia</h2><p>Predictions against GFW integrated alerts.</p></>,
  explore: <><h2>Explore</h2><p>Spin the globe and pick any tile.</p></>,
};

export interface StoryProps {
  chapters: Chapter[];
  onStepChange: (step: Step) => void;
}

export function Story({ chapters, onStepChange }: StoryProps) {
  const sectionRefs = useRef(new Map<string, HTMLElement>());
  const activeRef = useRef<string | null>(null);
  const onStepChangeRef = useRef(onStepChange);
  onStepChangeRef.current = onStepChange;

  const steps = chapters.flatMap((c) => c.steps.map((s) => ({ chapter: c.id, step: s })));

  useEffect(() => {
    const byId = new Map(chapters.flatMap((c) => c.steps).map((s) => [s.id, s]));
    let raf = 0;
    const measure = () => {
      raf = 0;
      const rects = [...sectionRefs.current].map(([id, el]) => {
        const r = el.getBoundingClientRect();
        return { id, top: r.top, bottom: r.bottom };
      });
      const id = pickActiveStep(rects, window.innerHeight * ANCHOR);
      if (id === null || id === activeRef.current) return;
      activeRef.current = id;
      const step = byId.get(id);
      if (step) onStepChangeRef.current(step);
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };
    measure(); // fire for the initial position
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (raf) cancelAnimationFrame(raf);
      activeRef.current = null; // StrictMode remount re-fires the current step
    };
  }, [chapters]);

  return (
    <div className="story">
      {steps.map(({ chapter, step }) => (
        <section
          key={step.id}
          data-step={step.id}
          data-chapter={chapter}
          className={`story-step story-step--${step.id}`}
          ref={(el) => {
            if (el) sectionRefs.current.set(step.id, el);
            else sectionRefs.current.delete(step.id);
          }}
        >
          <div className="story-card">{PLACEHOLDER[step.id] ?? <p>{step.id}</p>}</div>
        </section>
      ))}
    </div>
  );
}

export default Story;
