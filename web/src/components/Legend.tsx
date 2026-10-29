import type { ReactNode } from "react";

export interface LegendItem {
  color: string;
  label: ReactNode;
  note?: ReactNode;
  /** "dot" for map pins, "fill" for raster/polygon areas. */
  shape?: "dot" | "fill";
}

export function Legend({ items, title }: { items: LegendItem[]; title?: string }) {
  return (
    <div className="legend">
      {title && <p className="legend__title">{title}</p>}
      <ul className="legend__list">
        {items.map((it, i) => (
          <li key={i} className="legend__item">
            <span
              className={`legend__swatch legend__swatch--${it.shape ?? "fill"}`}
              style={{ background: it.color }}
              aria-hidden="true"
            />
            <span className="legend__text">
              <span className="legend__label">{it.label}</span>
              {it.note && <span className="legend__note">{it.note}</span>}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Legend;
