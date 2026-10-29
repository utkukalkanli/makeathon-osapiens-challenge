import { useEffect, useState } from "react";
import type { Tile } from "./data/types";
import { loadTiles } from "./data/load";
import { GlobeMap } from "./map/GlobeMap";

export default function App() {
  const [tiles, setTiles] = useState<Tile[]>([]);

  useEffect(() => {
    let cancelled = false;
    loadTiles()
      .then((t) => !cancelled && setTiles(t))
      .catch((e) => console.error(e));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="app">
      <GlobeMap tiles={tiles} autoRotate />
    </main>
  );
}
