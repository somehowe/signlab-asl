"use strict";

const byId = id => document.getElementById(id);
const ui = {
  connection: byId("connection"), dataset: byId("dataset-name"),
  sample: byId("sample-select"), detail: byId("sample-detail"),
  kind: byId("input-kind"), status: byId("input-status"),
  analyze: byId("analyze"), file: byId("landmark-file"),
  preview: byId("preview"), previewId: byId("preview-id"),
  placeholder: byId("preview-placeholder"), canvas: byId("landmarks"),
  video: byId("clip-video"), image: byId("clip-image"),
  timeline: byId("timeline"), play: byId("play"),
  slider: byId("frame-slider"), frameNumber: byId("frame-number"),
  saveFrames: byId("save-frames"),
  state: byId("result-state"), decision: byId("decision"),
  prediction: byId("prediction"), predictionDetail: byId("prediction-detail"),
  scores: byId("scores"), predictionMeta: byId("prediction-meta"),
  reportStatus: byId("report-status"), report: byId("report"),
  metrics: byId("metrics"), comparison: byId("comparison"),
  confusion: byId("confusion"), reportNotes: byId("report-notes"),
};
const handEdges = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const poseEdges = [[42, 43], [42, 44], [43, 45], [42, 46], [43, 47], [46, 47], [44, 0], [45, 21]];
const handColors = ["#ddf490", "#84d8c3"];
let samples = new Map();
let pointCount;
let dimensions;
let uploaded = null;
let requestActive = false;
let clipFrames = [];
let frameBounds = null;
let frameIndex = 0;
let playback = null;

function node(tag, text = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  return element;
}

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.dataset.state = type;
}

function pct(value) {
  return Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
}

async function api(path, payload) {
  const options = payload ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  } : {};
  const response = await fetch(path, options);
  const body = await response.json().catch(() => null);
  if (!response.ok || body === null) {
    throw new Error(body?.error || body?.detail || `Request failed (${response.status})`);
  }
  return body;
}

function setConnection(text, state) {
  ui.connection.dataset.state = state;
  ui.connection.replaceChildren(node("span"), document.createTextNode(text));
}

function currentSample() {
  return samples.get(ui.sample.value);
}

function syncControls() {
  ui.sample.disabled = requestActive || !samples.size;
  ui.analyze.disabled = requestActive || !(uploaded || currentSample());
  ui.file.disabled = requestActive;
  ui.analyze.firstChild.nodeValue = requestActive ? "Analyzing… " : "Analyze clip ";
}

function setDecisionStatus(label, state) {
  ui.state.textContent = label;
  ui.state.dataset.state = state;
  ui.decision.dataset.state = state;
}

function resetDecision() {
  setDecisionStatus("Awaiting analysis", "");
  ui.prediction.textContent = "—";
  ui.predictionDetail.textContent = "Analyze this clip to see the model’s decision.";
  ui.scores.replaceChildren(node("li", "Scores will appear here.", "empty"));
  ui.predictionMeta.replaceChildren();
  ui.predictionMeta.hidden = true;
}

function showSelectedSample() {
  uploaded = null;
  ui.file.value = "";
  const sample = currentSample();
  ui.kind.textContent = "Sample library";
  ui.previewId.textContent = sample ? sample.id : "NO CLIP";
  ui.detail.textContent = sample ? "Recorded evaluation sample" : "Select a clip to get started.";
  setStatus(ui.status, "");
  showPreview(sample || {});
  resetDecision();
  syncControls();
}

async function loadCatalog() {
  try {
    const catalog = await api("/api/catalog");
    const dataset = catalog.dataset;
    ui.dataset.textContent = typeof dataset === "string" ? dataset : dataset?.name || "Recorded samples";
    pointCount = catalog.point_count;
    dimensions = catalog.dimensions;
    if (Number.isInteger(pointCount) && Number.isInteger(dimensions)) {
      byId("upload-note").textContent = `JSON with a frames array, ${pointCount} points per frame and ${dimensions} values per point.`;
    }
    samples = new Map((catalog.samples || []).map(sample => [String(sample.id), sample]));
    ui.sample.replaceChildren();
    if (samples.size) {
      let position = 0;
      for (const [id, sample] of samples) {
        const name = `Sample ${String(++position).padStart(2, "0")}`;
        const label = sample.title || [name, sample.label].filter(Boolean).join(" · ");
        const option = node("option", label);
        option.value = id;
        ui.sample.append(option);
      }
    } else {
      ui.sample.append(node("option", "No recorded samples available"));
    }
    setConnection("Local model ready", "ready");
    showSelectedSample();
    if (samples.size) void analyze();
  } catch (error) {
    ui.dataset.textContent = "Unavailable";
    ui.sample.replaceChildren(node("option", "Sample library unavailable"));
    setConnection("Model unavailable", "offline");
    setStatus(ui.status, error.message, "error");
  }
}

function validXY(point) {
  return Number.isFinite(point?.[0]) && Number.isFinite(point?.[1]);
}

function framesFrom(value) {
  if (!Array.isArray(value)) return [];
  return value.filter(frame => Array.isArray(frame));
}

function boundsOf(frames) {
  const bounds = {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity};
  for (const frame of frames) {
    for (const point of frame) {
      if (!validXY(point)) continue;
      bounds.minX = Math.min(bounds.minX, point[0]);
      bounds.maxX = Math.max(bounds.maxX, point[0]);
      bounds.minY = Math.min(bounds.minY, point[1]);
      bounds.maxY = Math.max(bounds.maxY, point[1]);
    }
  }
  return Number.isFinite(bounds.minX) ? bounds : null;
}

function drawFrame() {
  const {width, height} = ui.preview.getBoundingClientRect();
  if (!clipFrames.length || !frameBounds || !width || !height) return;
  const density = Math.min(window.devicePixelRatio || 1, 2);
  ui.canvas.width = Math.round(width * density);
  ui.canvas.height = Math.round(height * density);
  const ctx = ui.canvas.getContext("2d");
  ctx.scale(density, density);
  ctx.lineWidth = 2;
  ctx.lineCap = "round";

  const frame = clipFrames[frameIndex];
  const {minX, maxX, minY, maxY} = frameBounds;
  const scale = .76 * Math.min(width / Math.max(maxX - minX, .001), height / Math.max(maxY - minY, .001));
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const position = point => [
    (point[0] - centerX) * scale + width / 2,
    (point[1] - centerY) * scale + height / 2,
  ];

  function drawEdges(edges, offset, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    for (const [from, to] of edges) {
      const first = frame[offset + from];
      const second = frame[offset + to];
      if (!validXY(first) || !validXY(second)) continue;
      ctx.moveTo(...position(first));
      ctx.lineTo(...position(second));
    }
    ctx.stroke();
  }

  if (frame.length === 48) drawEdges(poseEdges, 0, "#7d9d92");
  handColors.forEach((color, index) => drawEdges(handEdges, index * 21, color));
  for (const [index, point] of frame.entries()) {
    if (!validXY(point)) continue;
    const [x, y] = position(point);
    ctx.fillStyle = handColors[Math.floor(index / 21)] || "#bacdc0";
    ctx.beginPath();
    ctx.arc(x, y, index < 42 ? 2.9 : 3.3, 0, 2 * Math.PI);
    ctx.fill();
  }
  ui.slider.value = frameIndex + 1;
  ui.frameNumber.value = `${frameIndex + 1} / ${clipFrames.length}`;
}

function pause() {
  if (playback !== null) window.clearInterval(playback);
  playback = null;
  ui.play.textContent = "Play";
  ui.play.setAttribute("aria-label", "Play landmarks");
}

function togglePlayback() {
  if (playback !== null) {
    pause();
    return;
  }
  if (clipFrames.length < 2) return;
  ui.play.textContent = "Pause";
  ui.play.setAttribute("aria-label", "Pause landmarks");
  playback = window.setInterval(() => {
    frameIndex = (frameIndex + 1) % clipFrames.length;
    drawFrame();
  }, 100);
}

function localMedia(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, window.location.href);
    return url.origin === window.location.origin ? url : null;
  } catch {
    return null;
  }
}

function showPreview(source) {
  pause();
  ui.video.pause();
  ui.video.hidden = true;
  ui.image.hidden = true;
  ui.canvas.hidden = true;
  ui.timeline.hidden = true;
  ui.placeholder.hidden = false;
  const media = localMedia(source.video_url || source.image_url || source.media_url);
  const isImage = source.image_url || /\.(png|jpe?g|webp|gif|avif)$/i.test(media?.pathname);
  if (media) {
    const target = isImage ? ui.image : ui.video;
    target.src = media.href;
    target.hidden = false;
    ui.placeholder.hidden = true;
    return;
  }

  clipFrames = framesFrom(source.frames);
  frameBounds = boundsOf(clipFrames);
  if (!frameBounds) return;
  frameIndex = 0;
  ui.canvas.hidden = false;
  ui.placeholder.hidden = true;
  ui.timeline.hidden = false;
  ui.slider.max = clipFrames.length;
  ui.slider.disabled = ui.play.disabled = clipFrames.length < 2;
  drawFrame();
}

function validUpload(frames) {
  return Array.isArray(frames) && frames.length > 0 && frames.every(frame =>
    Array.isArray(frame) && frame.length >= 21 && (!pointCount || frame.length === pointCount)
    && frame.every(point => point === null || (
      Array.isArray(point) && point.length >= 2 && (!dimensions || point.length === dimensions)
      && point.every(value => value === null || Number.isFinite(value))
    ))
  ) && frames.some(frame => frame.some(validXY));
}

async function loadFile() {
  const file = ui.file.files[0];
  if (!file) return;
  try {
    if (file.size > 1_000_000) throw new Error("Choose a landmark JSON file smaller than 1 MB.");
    const body = JSON.parse(await file.text());
    const frames = Array.isArray(body) ? body : body?.frames;
    if (!validUpload(frames)) throw new Error("This file does not contain a valid landmark sequence for this model.");
    if (frames.length > 240) throw new Error("Choose a landmark sequence with at most 240 frames.");
    uploaded = {frames: frames.map(frame => frame.map(point => point ?? Array(dimensions || 3).fill(null)))};
    ui.sample.value = "";
    ui.kind.textContent = "Local file";
    ui.detail.textContent = file.name;
    ui.previewId.textContent = "LOCAL LANDMARKS";
    setStatus(ui.status, "");
    showPreview(uploaded);
    resetDecision();
    syncControls();
  } catch (error) {
    setStatus(ui.status, error.message, "error");
    ui.file.value = "";
  }
}

function renderScores(scores) {
  const entries = (Array.isArray(scores) ? scores : Object.entries(scores || {}).map(
    ([label, score]) => ({label, score}),
  )).map(entry => ({label: entry.label, score: entry.score ?? entry.probability}))
    .filter(entry => typeof entry.label === "string" && Number.isFinite(entry.score))
    .sort((a, b) => b.score - a.score);
  ui.scores.replaceChildren();
  if (!entries.length) {
    ui.scores.append(node("li", "No model scores were returned.", "empty"));
    return;
  }
  for (const {label, score} of entries.slice(0, 5)) {
    const row = node("li", "", "score-row");
    const track = node("span", "", "score-track");
    track.setAttribute("aria-hidden", "true");
    const fill = node("span", "", "score-fill");
    fill.style.width = pct(Math.max(0, Math.min(1, score)));
    track.append(fill);
    row.append(node("span", label, "score-label"), track, node("span", pct(score), "score-value"));
    ui.scores.append(row);
  }
}

function renderPrediction(result, sample) {
  const label = result.prediction ?? result.label;
  const abstained = result.abstained === true || typeof label !== "string" || !label;
  const truth = result.truth ?? sample?.label;
  setDecisionStatus(abstained ? "Abstained" : "Accepted", abstained ? "abstained" : "accepted");
  ui.prediction.textContent = abstained ? "No confident match" : label;
  ui.predictionDetail.textContent = abstained
    ? "The model held back its top guess."
    : "The highest ranked sign passed the confidence rule.";
  renderScores(result.scores);

  const metadata = [];
  if (typeof truth === "string") metadata.push(["Reference", truth]);
  if (Number.isFinite(result.threshold)) metadata.push(["Threshold", pct(result.threshold)]);
  if (Number.isFinite(result.latency_ms)) metadata.push(["Model time", `${result.latency_ms.toFixed(1)} ms`]);
  if (sample?.signer !== undefined && sample.signer !== null) metadata.push(["Signer", sample.signer]);
  ui.predictionMeta.replaceChildren(...metadata.map(([key, value]) => {
    const item = node("span", `${key} `);
    item.append(node("strong", value));
    return item;
  }));
  ui.predictionMeta.hidden = !metadata.length;
}

async function analyze() {
  const sample = currentSample();
  if (requestActive || !(uploaded || sample)) return;
  requestActive = true;
  syncControls();
  setStatus(ui.status, "Running the model…");
  try {
    const input = uploaded || {sample_id: sample.id};
    const result = await api("/api/predict", input);
    renderPrediction(result, sample);
    showPreview({...sample, ...input, ...result});
    setStatus(ui.status, "");
  } catch (error) {
    setStatus(ui.status, error.message, "error");
  } finally {
    requestActive = false;
    syncControls();
  }
}

function cell(tag, value, className = "", scope) {
  const element = node(tag, value, className);
  if (scope) element.scope = scope;
  return element;
}

function makeTable(headers, className) {
  const table = node("table", "", className);
  const row = table.createTHead().insertRow();
  headers.forEach((value, index) => row.append(cell("th", value, index ? "number" : "", "col")));
  return table;
}

function renderComparison(report) {
  const modelRows = Array.isArray(report.models)
    ? report.models
    : Object.entries(report.models || {}).map(([name, metrics]) => ({id: name, name, ...metrics}));
  ui.comparison.replaceChildren();
  if (!modelRows.length) {
    ui.comparison.append(node("p", "No comparison was provided.", "empty"));
    return;
  }
  const table = makeTable(["Model", "Macro F1", "Accuracy"], "comparison");
  const body = table.createTBody();
  for (const model of modelRows) {
    const row = body.insertRow();
    row.className = "comparison-row";
    const metrics = model.test || model.metrics || model;
    const name = cell("th", "", "", "row");
    const selected = model.id === report.selected_model;
    name.append(node("strong", model.name || model.label || "Model"));
    if (selected) name.append(node("small", "Selected on validation data"));
    if (model.split || report.protocol) name.append(node("small", model.split || report.protocol));
    row.append(name, cell("td", pct(metrics.macro_f1), "number"), cell("td", pct(metrics.accuracy), "number"));
  }
  ui.comparison.append(table);
}

function renderConfusion(report) {
  const matrix = report.confusion || report.test?.confusion;
  const labels = report.classes || report.labels;
  ui.confusion.replaceChildren();
  if (!Array.isArray(matrix) || !Array.isArray(labels) || !labels.length) {
    ui.confusion.append(node("p", "No confusion matrix was provided.", "empty"));
    return;
  }
  const table = makeTable(["Reference ↓", ...labels], "matrix");
  const body = table.createTBody();
  matrix.forEach((values, rowIndex) => {
    const row = body.insertRow();
    const label = labels[rowIndex];
    const maximum = Math.max(1, ...values);
    row.append(cell("th", label, "", "row"));
    values.forEach((value, columnIndex) => {
      const entry = cell("td", value);
      const alpha = Math.max(0, Math.min(1, value / maximum)) * .28;
      entry.style.backgroundColor = `rgba(73, 151, 95, ${.07 + alpha})`;
      entry.dataset.correct = String(rowIndex === columnIndex);
      entry.title = `${label} → ${labels[columnIndex]}: ${value}`;
      row.append(entry);
    });
  });
  ui.confusion.append(table);
}

function renderReport(report) {
  const test = report.test || report;
  const metricRows = [
    ["Test macro F1", pct(test.macro_f1)],
    ["Coverage", pct(test.coverage)],
    ["Accuracy when accepted", pct(test.selective_accuracy)],
    ["Test clips", test.n ?? test.clips ?? report.split_counts?.test?.clips ?? "—"],
  ];
  ui.metrics.replaceChildren(...metricRows.map(([name, value]) => {
    const metric = node("div", "", "metric");
    metric.append(node("dt", name), node("dd", value));
    return metric;
  }));
  renderComparison(report);
  renderConfusion(report);
  const notes = Array.isArray(report.notes) ? [...report.notes] : [];
  if (report.protocol) notes.unshift(`Evaluation split: ${report.protocol}.`);
  const counts = report.split_counts;
  if (counts?.train?.clips && counts.validation?.clips && counts.test?.clips) {
    notes.push(`Clips by split: ${counts.train.clips} train, ${counts.validation.clips} validation, and ${counts.test.clips} test.`);
  }
  if (notes.length) {
    const list = node("ul");
    list.replaceChildren(...notes.map(note => node("li", note)));
    ui.reportNotes.replaceChildren(list);
    ui.reportNotes.hidden = false;
  }
  ui.report.hidden = false;
  setStatus(ui.reportStatus, "");
}

async function loadReport() {
  try {
    renderReport(await api("/api/report"));
  } catch (error) {
    setStatus(ui.reportStatus, `Evaluation report unavailable. ${error.message}`, "error");
  }
}

ui.sample.addEventListener("change", showSelectedSample);
ui.file.addEventListener("change", loadFile);
ui.analyze.addEventListener("click", analyze);
ui.play.addEventListener("click", togglePlayback);
ui.saveFrames.addEventListener("click", () => {
  if (!clipFrames.length) return;
  const url = URL.createObjectURL(new Blob(
    [JSON.stringify({frames: clipFrames})], {type: "application/json"},
  ));
  const link = node("a");
  link.href = url;
  link.download = "signlab-landmarks.json";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
});
ui.slider.addEventListener("input", () => {
  pause();
  frameIndex = Number(ui.slider.value) - 1;
  drawFrame();
});
window.addEventListener("resize", () => {
  if (!ui.canvas.hidden) drawFrame();
});
for (const media of [ui.video, ui.image]) {
  media.addEventListener("error", () => {
    media.hidden = true;
    ui.placeholder.hidden = false;
  });
}
loadCatalog();
loadReport();
