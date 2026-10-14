const COLOURS = ["#FE8370", "#2A3744", "#7D98A6", "#D9573F"];
const STATE_COLOURS = ["#FE8370", "#2A3744", "#7D98A6", "#F2B134", "#D9573F", "#4E6876", "#B8A17A", "#8C4A1F", "#A3C4A8", "#9B6B9E"];
let compareBy = "state";
const COMPARE_TEXT = {
  state: "Pick which states to compare. Each graph shows all of your measurements while you're in one state, across all of the recordings. The measurements are put on the same scale so they can sit side by side.",
  measurement: "Pick which states to compare. Each graph shows how one measurement looks in each of those states, across all of the recordings.",
};
const UNITS = { noise: "", heartrate: " (bpm)", temperature: " (°C)", motion: "" };
const PY_FILES = ["hmms.py", "synthetic_signal_generator.py", "bridge.py"];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const buttons = [$("generate"), $("train")];

let bridge = null;
let data = null; // { signal_names, raw, scaled }
let result = null; // { predicted, scores, transmat, signals }
let standardised = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function setBusy(busy) {
  buttons.forEach((b) => (b.disabled = busy || !bridge));
}

async function init() {
  setBusy(true);
  setStatus("Loading Python runtime (first visit only, a few MB)...");
  const pyodide = await loadPyodide();
  for (const file of PY_FILES) {
    const text = await (await fetch(file)).text();
    pyodide.FS.writeFile(file, text);
  }
  pyodide.runPython("import bridge");
  bridge = pyodide.pyimport("bridge");
  setBusy(false);
  setStatus("Ready.");
}

// Let the browser repaint before blocking on Python.
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

function seedValue() {
  const raw = $("seed").value.trim();
  return raw === "" ? null : Number(raw);
}

async function onGenerate() {
  setBusy(true);
  setStatus("Generating data...");
  await nextFrame();
  data = JSON.parse(
    bridge.generate(+$("count").value, +$("length").value, seedValue())
  );
  result = null;
  $("view-matrix").hidden = true;
  $("compare-card").hidden = true;
  const select = $("recording");
  select.innerHTML = "";
  data.raw.forEach((_, i) => select.add(new Option(`Recording ${i + 1}`, i)));
  $("results").hidden = false;
  $("matrix-card").hidden = true;
  render();
  setBusy(false);
  setStatus("Data generated. Train a model to see predicted states.");
}

function selectedSignals() {
  return [...document.querySelectorAll("#signals input:checked")].map((i) => i.value);
}

async function onTrain() {
  if (!data) return setStatus("Generate data first.");
  const signals = selectedSignals();
  if (!signals.length) return setStatus("Select at least one signal.");
  setBusy(true);
  setStatus("Training model...");
  await nextFrame();
  try {
    result = JSON.parse(bridge.train(JSON.stringify(signals), +$("states").value, seedValue()));
    result.signals = signals;
    result.stateCount = +$("states").value;
    buildStatePicks(result.stateCount);
    $("compare-card").hidden = false;
    $("matrix-card").hidden = true;
    $("view-matrix").hidden = false;
    $("view-matrix").textContent = "View transition matrix";
    setStatus("Model trained.");
  } catch (e) {
    setStatus("Training failed: " + e.message);
  }
  render();
  setBusy(false);
}

function matrixLatex(m) {
  const rows = m.map((r) => r.map((v) => v.toFixed(3)).join(" & ")).join(" \\\\ ");
  return "P = \\begin{bmatrix}" + rows + "\\end{bmatrix}";
}

const AXIS = { gridcolor: "#D9CFB8", zerolinecolor: "#D9CFB8", linecolor: "#2A3744" };
const BASE_LAYOUT = {
  paper_bgcolor: "#FAF4E4", plot_bgcolor: "#FAF4E4",
  font: { family: "Host Grotesk, sans-serif", color: "#2A3744" },
};
const PLOT_CONFIG = { responsive: true, displaylogo: false };

function values() {
  return standardised ? data.scaled : data.raw;
}

function buildStatePicks(count) {
  const box = $("state-picks");
  box.innerHTML = "";
  for (let s = 0; s < count; s++) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${s}" checked> <span class="state-dot" style="background:${STATE_COLOURS[s % STATE_COLOURS.length]}"></span> State ${s}`;
    box.appendChild(label);
  }
}

function selectedStates() {
  return [...document.querySelectorAll("#state-picks input:checked")].map((i) => +i.value);
}

function renderSignals() {
  const i = +$("recording").value || 0;
  const signals = result ? result.signals : selectedSignals();
  const rec = values()[i];
  const x = rec.map((_, k) => k);
  const hasStates = !!result;
  const traces = [];
  const layout = { ...BASE_LAYOUT, margin: { t: 20, r: 20, b: 50, l: 70 }, showlegend: standardised };

  if (standardised) {
    signals.forEach((name) => {
      const c = data.signal_names.indexOf(name);
      traces.push({
        x, y: rec.map((row) => row[c]), name, type: "scatter", mode: "lines",
        line: { color: COLOURS[c % COLOURS.length], width: 1.8 }, xaxis: "x", yaxis: "y",
      });
    });
    if (hasStates) {
      traces.push({
        x, y: result.predicted[i], name: "Predicted state", type: "scatter", mode: "lines",
        line: { color: "#FE8370", width: 2.5, shape: "hv" }, xaxis: "x2", yaxis: "y2", showlegend: false,
      });
    }
    layout.height = hasStates ? 520 : 340;
    layout.legend = { orientation: "h", y: 1.08 };
    layout.xaxis = { ...AXIS, anchor: "y", matches: hasStates ? "x2" : undefined, showticklabels: !hasStates, title: hasStates ? undefined : "Sample" };
    layout.yaxis = { ...AXIS, title: "Standardised value", domain: hasStates ? [0.38, 1] : [0, 1] };
    if (hasStates) {
      layout.xaxis2 = { ...AXIS, title: "Sample", anchor: "y2" };
      layout.yaxis2 = { ...AXIS, title: "Predicted state", domain: [0, 0.27], dtick: 1, tickmode: "linear" };
    }
  } else {
    // One row per measurement, plus a final row for the predicted states.
    const rows = signals.length + (hasStates ? 1 : 0);
    const gap = 0.04;
    const h = (1 - gap * (rows - 1)) / rows;
    const domainFor = (r) => {
      const top = 1 - r * (h + gap);
      return [Math.max(top - h, 0), top];
    };
    const key = (n) => (n === 0 ? "" : String(n + 1));
    signals.forEach((name, r) => {
      const c = data.signal_names.indexOf(name);
      traces.push({
        x, y: rec.map((row) => row[c]), name, type: "scatter", mode: "lines",
        line: { color: COLOURS[c % COLOURS.length], width: 1.8 },
        xaxis: "x" + key(r), yaxis: "y" + key(r),
      });
      layout["yaxis" + key(r)] = { ...AXIS, title: name + UNITS[name], domain: domainFor(r), anchor: "x" + key(r) };
      layout["xaxis" + key(r)] = { ...AXIS, anchor: "y" + key(r), matches: r < rows - 1 ? "x" + key(rows - 1) : undefined, showticklabels: r === rows - 1 };
    });
    if (hasStates) {
      const r = rows - 1;
      traces.push({
        x, y: result.predicted[i], name: "Predicted state", type: "scatter", mode: "lines",
        line: { color: "#FE8370", width: 2.5, shape: "hv" }, xaxis: "x" + key(r), yaxis: "y" + key(r),
      });
      layout["yaxis" + key(r)] = { ...AXIS, title: "Predicted state", domain: domainFor(r), anchor: "x" + key(r), dtick: 1, tickmode: "linear" };
      layout["xaxis" + key(r)] = { ...AXIS, anchor: "y" + key(r), title: "Sample" };
    } else {
      layout["xaxis" + key(rows - 1)].title = "Sample";
    }
    layout.height = 150 * rows + 80;
  }
  Plotly.react("plot", traces, layout, PLOT_CONFIG);
}

function renderCompare() {
  if (!result) return;
  const states = selectedStates();
  const signals = result.signals;
  const byState = compareBy === "state";
  const vals = byState ? data.scaled : values();
  const traces = [];
  const layout = { ...BASE_LAYOUT, margin: { t: 40, r: 20, b: 50, l: 60 }, height: 380, showlegend: true, legend: { orientation: "h", y: -0.15 }, annotations: [] };
  if (!states.length) { Plotly.purge("compare-plot"); return; }
  const panels = byState ? states : signals;
  const inner = byState ? signals : states;
  const n = panels.length;
  const gap = 0.06;
  const w = (1 - gap * (n - 1)) / n;
  const key = (k) => (k === 0 ? "" : String(k + 1));

  panels.forEach((p, k) => {
    const left = k * (w + gap);
    layout["xaxis" + key(k)] = { ...AXIS, domain: [left, left + w], anchor: "y" + key(k), showticklabels: false };
    layout["yaxis" + key(k)] = { ...AXIS, anchor: "x" + key(k), matches: byState && k > 0 ? "y" : undefined };
    const title = byState ? `State ${p}` : p + UNITS[p];
    layout.annotations.push({ text: title, x: left + w / 2, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: { family: "IBM Plex Mono, monospace", size: 12 } });
    inner.forEach((q, j) => {
      const s = byState ? p : q;
      const name = byState ? q : p;
      const c = data.signal_names.indexOf(name);
      const y = [];
      vals.forEach((rec, r) => rec.forEach((row, t) => { if (result.predicted[r][t] === s) y.push(row[c]); }));
      traces.push({
        type: "box", y, name: byState ? name : `State ${s}`, legendgroup: byState ? name : `s${s}`, showlegend: k === 0,
        marker: { color: byState ? STATE_COLOURS[j % STATE_COLOURS.length] : STATE_COLOURS[s % STATE_COLOURS.length] }, boxmean: true,
        xaxis: "x" + key(k), yaxis: "y" + key(k),
      });
    });
  });
  Plotly.react("compare-plot", traces, layout, PLOT_CONFIG);
}

function render() {
  if (!data) return;
  $("view-mode").textContent = standardised ? "Show separate axes" : "Show standardised values";
  renderSignals();
  renderCompare();

  if (result) {
    const i = +$("recording").value || 0;
    katex.render(matrixLatex(result.transmat), $("matrix"), { displayMode: true, throwOnError: false });
    $("loglik").textContent = `Fit score for this recording: ${result.scores[i].toFixed(2)}`;
  }
}

$("generate").addEventListener("click", onGenerate);
$("train").addEventListener("click", onTrain);
$("compare-mode").addEventListener("click", () => {
  compareBy = compareBy === "state" ? "measurement" : "state";
  $("compare-mode").textContent = compareBy === "state" ? "Group by measurement" : "Group by state";
  $("compare-desc").textContent = COMPARE_TEXT[compareBy];
  renderCompare();
});
$("view-mode").addEventListener("click", () => {
  standardised = !standardised;
  render();
});
$("state-picks").addEventListener("change", renderCompare);
$("view-matrix").addEventListener("click", () => {
  const card = $("matrix-card");
  card.hidden = !card.hidden;
  $("view-matrix").textContent = card.hidden ? "View transition matrix" : "Hide transition matrix";
  if (!card.hidden) {
    render();
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
$("recording").addEventListener("change", render);
$("signals").addEventListener("change", () => !result && render());

init().catch((e) => setStatus("Failed to load Python runtime: " + e.message));
