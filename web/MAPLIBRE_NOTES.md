# MapLibre GL JS v5 notes (installed 5.24; docs checked via context7 v5.19)

- Globe: in the style spec `projection: { type: "globe" }`, or at runtime
  `map.on("style.load", () => map.setProjection({ type: "globe" }))`.
  `map.getProjection().type` returns the current type. Flat: `{ type: "mercator" }`.
- Sky/atmosphere: style-level `sky: { "atmosphere-blend": [...] }` or `map.setSky({...})`;
  `map.setSky(undefined)` disables. Props: `sky-color`, `sky-horizon-blend`, `horizon-color`,
  `horizon-fog-blend`, `fog-color`, `fog-ground-blend`, `atmosphere-blend` (0..1, may be a
  zoom expression, e.g. interpolate zoom 0->1, 5->1, 7->0 fades atmosphere out when zoomed in).
- Globe lighting: style `light: { anchor: "map", position: [1.5, 90, 80] }` (radial, azimuth, polar).
- Image source coordinates: `{ type: "image", url, coordinates: [TL, TR, BR, BL] }`, each `[lng, lat]`,
  clockwise from top-left: `[[w,n],[e,n],[e,s],[w,s]]` from bounds `[w,s,e,n]`.
  Update later with `(map.getSource(id) as ImageSource).updateImage({ url, coordinates })`.
- Errors: `map.on("error", (e) => ...)`; `e.error` is an Error (ErrorEvent.error). Failed tile/image
  fetches also carry `e.sourceId` (and for HTTP errors `e.error.status`, `e.error.url` when present),
  so filter on `e.sourceId` to attribute a failure to a layer, e.g. basemap tiles vs label images.
  Treat missing `sourceId` as a generic/style error.
- `maplibregl.supported` does NOT exist in v5 (removed; absent from the typings). Do WebGL detection
  yourself: `!!document.createElement("canvas").getContext("webgl2" | "webgl")`, and also guard
  `new Map()` in try/catch (it throws if WebGL context creation fails).
- Import CSS once: `import "maplibre-gl/dist/maplibre-gl.css"` (done in src/main.tsx).
- Camera: `map.easeTo/flyTo({ center, zoom, pitch, bearing, duration })`. Globe renders at low zoom
  (< ~6 most visible); pitch/bearing work on globe.
