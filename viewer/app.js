const state = {
  split: "train",
  tileId: null,
  manifest: null,
  timelineIndex: 0,
};

const previewConfigs = [
  {
    kind: "s2",
    selectId: "s2-select",
    imageId: "s2-image",
    overlayImageId: "s2-overlay",
    statusId: "s2-status",
    emptyText: "No Sentinel-2 observation found for this tile.",
  },
  {
    kind: "s1",
    selectId: "s1-select",
    imageId: "s1-image",
    overlayImageId: "s1-overlay",
    statusId: "s1-status",
    emptyText: "No Sentinel-1 observation found for this tile.",
  },
  {
    kind: "aef",
    selectId: "aef-select",
    imageId: "aef-image",
    statusId: "aef-status",
    emptyText: "No AEF embedding year found for this tile.",
  },
  {
    kind: "label",
    selectId: "label-select",
    imageId: "label-image",
    statusId: "label-status",
    emptyText: "No label raster is available for this tile.",
  },
];

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function fillSelect(select, entries, selectedId, placeholder = null) {
  select.innerHTML = "";

  if (placeholder) {
    const option = document.createElement("option");
    option.value = placeholder.id;
    option.textContent = placeholder.label;
    if (placeholder.id === selectedId) {
      option.selected = true;
    }
    select.appendChild(option);
  }

  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.label;
    if (entry.id === selectedId) {
      option.selected = true;
    }
    select.appendChild(option);
  }

  select.disabled = select.options.length === 0;
}

function selectedText(selectId) {
  return document.querySelector(`#${selectId} option:checked`)?.textContent ?? "";
}

function previewUrl(kind, tileId, item) {
  const params = new URLSearchParams({
    kind,
    split: state.split,
    tile_id: tileId,
    item,
    size: "768",
  });
  return `/api/preview?${params.toString()}`;
}

function overlayUrl(baseKind, tileId, baseItem, labelItem) {
  const params = new URLSearchParams({
    base_kind: baseKind,
    split: state.split,
    tile_id: tileId,
    base_item: baseItem,
    label_item: labelItem,
    size: "768",
  });
  return `/api/overlay?${params.toString()}`;
}

function parseObservationId(kind, value) {
  if (!value) {
    return null;
  }

  if (kind === "s2") {
    const [year, month] = value.split("-").map(Number);
    return { year, month };
  }

  if (kind === "s1") {
    const [year, month, orbit] = value.split("-");
    return { year: Number(year), month: Number(month), orbit };
  }

  if (kind === "timeline") {
    const [year, month] = value.split("-").map(Number);
    return { year, month };
  }

  if (kind === "aef") {
    return { year: Number(value) };
  }

  return null;
}

function monthKey(year, month) {
  return year * 12 + month;
}

function findNearestTimelineIndex(year, month) {
  const entries = state.manifest?.timeline ?? [];
  if (entries.length === 0) {
    return 0;
  }

  const target = monthKey(year, month);
  let bestIndex = 0;
  let bestDelta = Number.POSITIVE_INFINITY;

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const delta = Math.abs(monthKey(entry.year, entry.month) - target);
    if (delta < bestDelta) {
      bestDelta = delta;
      bestIndex = index;
    }
  }

  return bestIndex;
}

function findNearestMonthlyEntry(entries, targetYear, targetMonth, preferredOrbit = null) {
  if (!entries || entries.length === 0) {
    return null;
  }

  const target = monthKey(targetYear, targetMonth);
  let best = null;

  for (const entry of entries) {
    const delta = Math.abs(monthKey(entry.year, entry.month) - target);
    const orbitPenalty =
      preferredOrbit && entry.orbit ? (entry.orbit === preferredOrbit ? 0 : 1) : 0;
    const candidate = {
      entry,
      delta,
      orbitPenalty,
    };

    if (
      !best ||
      candidate.delta < best.delta ||
      (candidate.delta === best.delta && candidate.orbitPenalty < best.orbitPenalty) ||
      (
        candidate.delta === best.delta &&
        candidate.orbitPenalty === best.orbitPenalty &&
        candidate.entry.id < best.entry.id
      )
    ) {
      best = candidate;
    }
  }

  return best?.entry ?? entries[0];
}

function findNearestYearEntry(entries, targetYear) {
  if (!entries || entries.length === 0) {
    return null;
  }

  let best = null;
  for (const entry of entries) {
    const delta = Math.abs(entry.year - targetYear);
    if (!best || delta < best.delta || (delta === best.delta && entry.year < best.entry.year)) {
      best = { entry, delta };
    }
  }

  return best?.entry ?? entries[0];
}

function setImageState(image, src = null) {
  if (!src) {
    image.removeAttribute("src");
    image.classList.add("is-empty");
    return;
  }

  image.classList.remove("is-empty");
  image.src = src;
}

function clearOverlay(config) {
  if (!config.overlayImageId) {
    return;
  }
  const overlayImage = document.getElementById(config.overlayImageId);
  overlayImage.style.opacity = "";
  setImageState(overlayImage, null);
}

function setPreviewMessage(config, message) {
  const image = document.getElementById(config.imageId);
  const status = document.getElementById(config.statusId);
  setImageState(image, null);
  clearOverlay(config);
  status.textContent = message;
}

function updateOverlayOpacityLabel() {
  const opacityInput = document.getElementById("overlay-opacity");
  const output = document.getElementById("overlay-opacity-value");
  output.textContent = `${opacityInput.value}%`;
}

function currentOverlaySelection() {
  return document.getElementById("overlay-select").value || "";
}

function applyOverlayForPanel(config, tileId, baseItem) {
  if (!config.overlayImageId) {
    return;
  }

  const overlayImage = document.getElementById(config.overlayImageId);
  const opacityInput = document.getElementById("overlay-opacity");
  const overlayItem = currentOverlaySelection();

  if (!baseItem || !overlayItem) {
    clearOverlay(config);
    return;
  }

  overlayImage.style.opacity = String(Number(opacityInput.value) / 100);
  setImageState(overlayImage, overlayUrl(config.kind, tileId, baseItem, overlayItem));
  overlayImage.onerror = () => {
    clearOverlay(config);
  };
}

function setPreviewImage(config, tileId, item) {
  if (!item) {
    setPreviewMessage(config, config.emptyText);
    return;
  }

  const image = document.getElementById(config.imageId);
  const status = document.getElementById(config.statusId);
  const overlayText = currentOverlaySelection() && config.overlayImageId
    ? ` • overlay ${selectedText("overlay-select")} at ${document.getElementById("overlay-opacity").value}%`
    : "";

  setImageState(image, previewUrl(config.kind, tileId, item));
  applyOverlayForPanel(config, tileId, item);
  status.textContent = `${selectedText(config.selectId)}${overlayText}`;

  image.onerror = () => {
    setPreviewMessage(config, "Preview failed to render.");
  };
}

function refreshPreviews() {
  if (!state.tileId || !state.manifest) {
    return;
  }

  for (const config of previewConfigs) {
    const select = document.getElementById(config.selectId);
    setPreviewImage(config, state.tileId, select.value || null);
  }
}

function setTimelineControlsEnabled(enabled) {
  document.getElementById("timeline-prev").disabled = !enabled;
  document.getElementById("timeline-next").disabled = !enabled;
  document.getElementById("timeline-range").disabled = !enabled;
}

function renderTimelineMeta() {
  const entries = state.manifest?.timeline ?? [];
  const label = document.getElementById("timeline-label");
  const status = document.getElementById("timeline-status");
  const range = document.getElementById("timeline-range");

  if (entries.length === 0) {
    label.textContent = "No monthly timeline available";
    status.textContent = "This tile does not have a shared Sentinel-1/Sentinel-2 month list.";
    range.value = "0";
    range.max = "0";
    setTimelineControlsEnabled(false);
    return;
  }

  const entry = entries[state.timelineIndex];
  label.textContent = `${entry.label}`;
  status.textContent = `Step ${state.timelineIndex + 1} of ${entries.length}. Sentinel-2 and Sentinel-1 snap to the nearest available month for this tile.`;
  range.max = String(entries.length - 1);
  range.value = String(state.timelineIndex);
  setTimelineControlsEnabled(true);
}

function syncTimelineToMonth(year, month) {
  if (!state.manifest?.timeline?.length) {
    return;
  }
  state.timelineIndex = findNearestTimelineIndex(year, month);
  renderTimelineMeta();
}

function applyTimeline(index) {
  if (!state.manifest?.timeline?.length) {
    renderTimelineMeta();
    return;
  }

  const clampedIndex = Math.min(
    Math.max(index, 0),
    state.manifest.timeline.length - 1
  );
  state.timelineIndex = clampedIndex;

  const timelineEntry = state.manifest.timeline[clampedIndex];
  const currentS1 = parseObservationId("s1", document.getElementById("s1-select").value);
  const preferredOrbit = currentS1?.orbit ?? null;

  const s2Entry = findNearestMonthlyEntry(
    state.manifest.s2,
    timelineEntry.year,
    timelineEntry.month
  );
  const s1Entry = findNearestMonthlyEntry(
    state.manifest.s1,
    timelineEntry.year,
    timelineEntry.month,
    preferredOrbit
  );
  const aefEntry = findNearestYearEntry(state.manifest.aef, timelineEntry.year);

  if (s2Entry) {
    document.getElementById("s2-select").value = s2Entry.id;
  }
  if (s1Entry) {
    document.getElementById("s1-select").value = s1Entry.id;
  }
  if (aefEntry) {
    document.getElementById("aef-select").value = aefEntry.id;
  }

  renderTimelineMeta();
  refreshPreviews();
}

function syncTimelineFromSelect(kind) {
  const parsed = parseObservationId(kind, document.getElementById(`${kind}-select`).value);
  if (parsed?.year && parsed?.month) {
    syncTimelineToMonth(parsed.year, parsed.month);
  }
}

function wirePreviewSelects() {
  document.getElementById("s2-select").addEventListener("change", () => {
    syncTimelineFromSelect("s2");
    refreshPreviews();
  });

  document.getElementById("s1-select").addEventListener("change", () => {
    syncTimelineFromSelect("s1");
    refreshPreviews();
  });

  document.getElementById("aef-select").addEventListener("change", refreshPreviews);
  document.getElementById("label-select").addEventListener("change", refreshPreviews);
}

function wireInspectorControls() {
  const overlaySelect = document.getElementById("overlay-select");
  const opacityInput = document.getElementById("overlay-opacity");
  const timelineRange = document.getElementById("timeline-range");

  overlaySelect.addEventListener("change", refreshPreviews);

  opacityInput.addEventListener("input", () => {
    updateOverlayOpacityLabel();
    refreshPreviews();
  });

  timelineRange.addEventListener("input", (event) => {
    applyTimeline(Number(event.target.value));
  });

  document.getElementById("timeline-prev").addEventListener("click", () => {
    applyTimeline(state.timelineIndex - 1);
  });

  document.getElementById("timeline-next").addEventListener("click", () => {
    applyTimeline(state.timelineIndex + 1);
  });
}

function populateInspectorControls() {
  fillSelect(
    document.getElementById("overlay-select"),
    state.manifest?.labels ?? [],
    state.manifest?.defaults.overlay ?? "",
    { id: "", label: "No overlay" }
  );
  updateOverlayOpacityLabel();
}

async function loadManifest(tileId) {
  state.tileId = tileId;
  state.manifest = await fetchJson(
    `/api/tile?split=${encodeURIComponent(state.split)}&tile_id=${encodeURIComponent(tileId)}`
  );

  fillSelect(
    document.getElementById("s2-select"),
    state.manifest.s2,
    state.manifest.defaults.s2
  );
  fillSelect(
    document.getElementById("s1-select"),
    state.manifest.s1,
    state.manifest.defaults.s1
  );
  fillSelect(
    document.getElementById("aef-select"),
    state.manifest.aef,
    state.manifest.defaults.aef
  );
  fillSelect(
    document.getElementById("label-select"),
    state.manifest.labels,
    state.manifest.defaults.label
  );

  populateInspectorControls();

  const summary = document.getElementById("dataset-summary");
  summary.textContent = `Tile ${tileId} • years ${state.manifest.years.join(", ")} • ${state.manifest.timeline.length} monthly timeline steps • AEF RGB bands ${state.manifest.aef_rgb_bands.join(", ")}`;

  const initialTimelineIndex = state.manifest.timeline.findIndex(
    (entry) => entry.id === state.manifest.defaults.timeline
  );

  if (state.manifest.timeline.length > 0) {
    applyTimeline(initialTimelineIndex >= 0 ? initialTimelineIndex : 0);
  } else {
    renderTimelineMeta();
    refreshPreviews();
  }
}

function resetInspectorForEmptyTiles() {
  fillSelect(document.getElementById("overlay-select"), [], "", { id: "", label: "No overlay" });
  setTimelineControlsEnabled(false);
  document.getElementById("timeline-label").textContent = "No monthly timeline available";
  document.getElementById("timeline-status").textContent = "No tiles are loaded.";
  document.getElementById("timeline-range").value = "0";
  document.getElementById("timeline-range").max = "0";
}

async function loadTiles() {
  const data = await fetchJson(`/api/tiles?split=${encodeURIComponent(state.split)}`);
  const tileSelect = document.getElementById("tile-select");
  tileSelect.innerHTML = "";

  for (const tile of data.tiles) {
    const option = document.createElement("option");
    option.value = tile.tile_id;
    option.textContent = tile.tile_id;
    tileSelect.appendChild(option);
  }

  tileSelect.disabled = data.tiles.length === 0;

  const summary = document.getElementById("dataset-summary");
  summary.textContent = `${data.tiles.length} tiles discovered in ${state.split}.`;

  if (data.tiles.length > 0) {
    await loadManifest(data.tiles[0].tile_id);
  } else {
    state.tileId = null;
    state.manifest = null;
    state.timelineIndex = 0;
    resetInspectorForEmptyTiles();

    for (const config of previewConfigs) {
      setPreviewMessage(config, "No tiles available.");
      fillSelect(document.getElementById(config.selectId), [], null);
    }
  }
}

function wireTopLevelControls() {
  const splitSelect = document.getElementById("split-select");
  splitSelect.innerHTML = "";

  for (const split of ["train", "test"]) {
    const option = document.createElement("option");
    option.value = split;
    option.textContent = split;
    splitSelect.appendChild(option);
  }

  splitSelect.value = state.split;
  splitSelect.addEventListener("change", async (event) => {
    state.split = event.target.value;
    await loadTiles();
  });

  document.getElementById("tile-select").addEventListener("change", async (event) => {
    await loadManifest(event.target.value);
  });
}

async function bootstrap() {
  wireTopLevelControls();
  wirePreviewSelects();
  wireInspectorControls();
  updateOverlayOpacityLabel();
  await loadTiles();
}

bootstrap().catch((error) => {
  document.getElementById("dataset-summary").textContent = `Viewer failed to load: ${error.message}`;
});
