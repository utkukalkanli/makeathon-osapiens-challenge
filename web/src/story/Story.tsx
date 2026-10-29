import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import type { Chapter, ModelMeta, Step, Tile } from "../data/types";
import { anchorY, pickActiveStep } from "./activeStep";
import { DISAGREE_TILE } from "./chapters";
import { Opening } from "./content/Opening";
import { Disagree } from "./content/Disagree";
import { Approach } from "./content/Approach";
import { Unseen } from "./content/Unseen";
import { Credits } from "../components/Credits";

/** Fraction of the viewport height where a step becomes active. */
export const ANCHOR = 0.5;

interface ContentContext {
  tiles: Tile[];
  models: ModelMeta[];
}

const tileById = (ctx: ContentContext, id: string) => ctx.tiles.find((t) => t.id === id);
const featured = (ctx: ContentContext) => ctx.models.find((m) => m.featured);

/** Step id -> chapter content. Unseen steps resolve their tile from the step camera. */
export function stepContent(step: Step, ctx: ContentContext): ReactNode {
  switch (step.id) {
    case "opening":
      return <Opening />;
    case "disagree-agreement":
      return <Disagree tile={tileById(ctx, DISAGREE_TILE)} view="agreement" />;
    case "disagree-sources":
      return <Disagree tile={tileById(ctx, DISAGREE_TILE)} view="sources" />;
    case "approach":
      return <Approach />;
    case "explore":
      // Task 8 replaces this with the explore panel.
      return (
        <>
          <h2>Explore</h2>
          <p>Spin the globe and pick any tile.</p>
          <Credits />
        </>
      );
  }
  if (step.id.startsWith("unseen-") && "tileId" in step.camera) {
    return <Unseen tile={tileById(ctx, step.camera.tileId)} model={featured(ctx)} />;
  }
  return <p>{step.id}</p>;
}

export interface StoryProps {
  chapters: Chapter[];
  tiles: Tile[];
  models: ModelMeta[];
  onStepChange: (step: Step) => void;
}

export function Story({ chapters, tiles, models, onStepChange }: StoryProps) {
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
      const id = pickActiveStep(rects, anchorY(window.innerWidth, window.innerHeight, ANCHOR));
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
          <div className="story-card">{stepContent(step, { tiles, models })}</div>
        </section>
      ))}
    </div>
  );
}

export default Story;
