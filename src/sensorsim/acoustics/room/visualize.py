"""Where the sound went: a self-contained 3D view of the paths from a source to a microphone.

The paths are the renderer's own. ``synth.early_paths`` builds the shoebox image sources and runs
the same obstacle processing that ``synthesize`` (and so every dataset render) applies, and
``obstacles.folded_segments`` unfolds each image back into the room, so every bounce lands on the
surface it reflects from. Drawn delays and levels are therefore exactly the renderer's; only the
drawing stops at a low reflection order. The echogram beside the 3D view also shows every early
arrival the renderer uses up to the mixing time, then the model's late decay (per-octave T60),
which the renderer synthesizes statistically rather than as paths. For a measured pair, the
measured impulse response's energy envelope is overlaid on the same time axis (sample 0 is the
emission; the direct sound sits at distance / c).

The page is one HTML file with its own small canvas renderer: no network access, no third-party
code, fonts or colour maps. It opens offline and can be attached to a report.

    python -m sensorsim.project visualize PROJECT paths.html [--take N] [--order 2]
"""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

import numpy as np

from sensorsim.acoustics.room.model import OCTAVE_HZ, SURFACES, RoomModel
from sensorsim.acoustics.room.obstacles import folded_segments
from sensorsim.acoustics.room.synth import SynthConfig, early_paths

FLOOR_DB = -60.


def _surfaces(point: np.ndarray, dims: np.ndarray, tol: float = 1e-6) -> list[str]:
    """Room surfaces a point lies on (two at an edge, where one bounce reflects off both)."""
    found = []
    for axis in range(3):
        if abs(point[axis]) < tol:
            found.append(SURFACES[2*axis])
        elif abs(point[axis] - dims[axis]) < tol:
            found.append(SURFACES[2*axis + 1])
    return found


def surface_name(surface: str, dims) -> str:
    axis = "xyz".index(surface[0])
    if axis == 2:
        return "floor" if surface.endswith("min") else "ceiling"
    return f"wall {surface[0]} = {0 if surface.endswith('min') else dims[axis]:.2f} m"


def _level_db(gain_octave: np.ndarray, reference: float) -> np.ndarray:
    """Octave-average energy relative to the unobstructed direct sound (1/r at the direct distance)."""
    energy = np.mean(np.atleast_2d(gain_octave)**2, axis=-1)
    with np.errstate(divide="ignore"):
        return 10*np.log10(energy/reference**2)


def scene(model: RoomModel, source_xyz_m, receiver_xyz_m, *, max_order: int = 2, measured: dict | None = None,
          names: tuple[str, str] = ("source", "microphone")) -> dict:
    """Everything the page draws, as plain JSON-ready data. `max_order` (0-3) limits the drawn paths."""
    if not 0 <= max_order <= 3:
        raise ValueError("draw at most third-order reflections (max_order 0 to 3)")
    source, receiver = np.asarray(source_xyz_m, float), np.asarray(receiver_xyz_m, float)
    dims = np.asarray(model.shoebox_dims_m, float)
    reference = 1/np.linalg.norm(source - receiver)
    paths, anchor, report, _ = early_paths(model, source, receiver, SynthConfig(max_order=max_order, late=False))
    owner, starts, ends = folded_segments(anchor.images_m, receiver, dims)
    levels = _level_db(paths.gain_octave, reference)
    drawn = []
    for i in range(len(anchor.delay_s)):
        mine = owner == i
        points = np.vstack([starts[mine], ends[mine][-1:]])[::-1]          # source, bounces..., receiver
        walls = [s for p in points[1:-1] for s in _surfaces(p, dims)]
        if not np.any(paths.wall_pressure[i]):
            status = "dropped"
        elif paths.distance_m[i] > anchor.distance_m[i] + 1e-9:
            status = "detoured"
        else:
            status = "clear"
        drawn.append({"kind": "room", "order": int(anchor.wall_hits[i].sum()), "points": points.round(5).tolist(),
                      "via": [surface_name(s, dims) for s in walls], "surfaces": walls,
                      "delay_s": float(paths.delay_s[i]), "length_m": float(paths.distance_m[i]),
                      "level_db": None if not np.isfinite(levels[i]) else round(float(levels[i]), 2), "status": status})
    records = report.get("first_reflections") or []
    if len(paths.delay_s) - len(anchor.delay_s) == len(records):              # box-face reflections, in order
        for k, record in enumerate(records):
            i = len(anchor.delay_s) + k
            drawn.append({"kind": "box", "order": 1, "points": [source.tolist(), record["point_m"], receiver.tolist()],
                          "via": [f"{record['box']} ({record['face']} face)"], "surfaces": [],
                          "delay_s": float(paths.delay_s[i]), "length_m": float(paths.distance_m[i]),
                          "level_db": None if not np.isfinite(levels[i]) else round(float(levels[i]), 2),
                          "status": "clear"})
    arrivals, note = _arrivals(model, source, receiver, reference)
    mixing = float(model.mixing_time())
    t60 = [float(x) for x in model.late_t60_s]
    k = OCTAVE_HZ.index(1000)
    late = arrivals["delay_s"] >= mixing - .005
    start_db = float(arrivals["level_db"][late].max()) if late.any() else float(arrivals["level_db"].min(initial=0.))
    longest = max([p["delay_s"] for p in drawn] + [mixing])
    view_s = float(min(max(1.6*mixing, 1.15*longest), mixing + t60[k]))
    return {
        "format": "sensorsim-room-paths/1",
        "room": {"dims_m": dims.tolist(), "c_m_s": float(model.sound_speed_m_s), "volume_m3": float(np.prod(dims)),
                 "obstacles": [{"name": b.name, "min_m": list(b.min_xyz), "size_m": list(b.size_xyz), "material": b.material}
                               for b in model.obstacles],
                 "mixing_time_s": mixing, "t60_s": t60, "bands_hz": list(OCTAVE_HZ)},
        "source": {"name": names[0], "xyz_m": source.tolist()},
        "receiver": {"name": names[1], "xyz_m": receiver.tolist()},
        "max_order": max_order,
        "paths": sorted(drawn, key=lambda p: p["delay_s"]),
        "arrivals": {"delay_s": arrivals["delay_s"].round(6).tolist(), "level_db": arrivals["level_db"].round(2).tolist(),
                     "note": note},
        "tail": {"start_s": mixing, "level_db": round(start_db, 2), "t60_s": t60[k], "band_hz": 1000,
                 "slope_db_s": -60/t60[k]},
        "measured": measured,
        "view_s": view_s,
        "level_note": "octave-average energy relative to the unobstructed direct sound",
    }


def _arrivals(model: RoomModel, source, receiver, reference: float) -> tuple[dict, str]:
    """Every early arrival the renderer uses (up to the mixing time), above the drawing floor."""
    try:
        paths = early_paths(model, source, receiver, SynthConfig())[0]
        note = "all early arrivals the renderer uses, up to the mixing time"
    except ValueError as error:                       # e.g. direct sound later than the mixing time
        paths = early_paths(model, source, receiver, SynthConfig(max_order=3, late=False))[0]
        note = f"arrivals up to third order ({error})"
    levels = _level_db(paths.gain_octave, reference)
    keep = np.isfinite(levels) & (levels > FLOOR_DB)
    return {"delay_s": paths.delay_s[keep], "level_db": levels[keep]}, note


def measured_envelope(rir: np.ndarray, fs: float, direct_delay_s: float, *, view_s: float, label: str,
                      step_s: float = 2.5e-4) -> dict:
    """Energy envelope of a measured impulse response, 0 dB at its direct-sound peak, on the scene's time axis.

    Sample 0 must be the emission time (out/fit/rirs are aligned so the direct sound arrives at
    distance / c). Each step keeps the maximum of a 0.5 ms moving average, so short reflections stay visible.
    """
    x = np.asarray(rir, float).ravel()
    window = max(1, round(5e-4*fs))
    energy = np.convolve(x*x, np.ones(window)/window, mode="same")
    a, b = max(0, int((direct_delay_s - 1e-3)*fs)), int((direct_delay_s + 1e-3)*fs) + 1
    peak = float(energy[a:b].max()) if b > a and a < energy.size else float(energy.max())
    step = max(1, round(step_s*fs))
    count = min(int(np.ceil(view_s*fs/step)), energy.size//step)
    blocks = energy[:count*step].reshape(count, step).max(axis=1)
    with np.errstate(divide="ignore"):
        db = np.maximum(10*np.log10(blocks/peak), FLOOR_DB - 10)
    return {"label": label, "step_s": step/fs, "level_db": np.round(db, 1).tolist(),
            "peak_s": float((a + np.argmax(energy[a:b]))/fs) if b > a and a < energy.size else None}


def render_html(data: dict, *, title: str, standalone: bool = True) -> str:
    """The page. `standalone` adds the document wrapper (doctype, head, body) for a file opened directly."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    head = f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n"
    body = (_BODY.replace("__TITLE__", html.escape(title))
            + f'<script type="application/json" id="scene-data">{payload}</script>\n<script>{_JS}</script>\n')
    if not standalone:
        return head + body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head>\n<body>\n{body}</body>\n</html>\n")


def write(path: str | Path, data: dict, *, title: str, standalone: bool = True) -> Path:
    path = Path(path)
    path.write_text(render_html(data, title=title, standalone=standalone), encoding="utf-8")
    return path


def summary(data: dict) -> dict:
    """What the page shows, in numbers (for the command line)."""
    paths = data["paths"]
    room = [p for p in paths if p["kind"] == "room"]
    return {"paths_drawn": len(paths), "by_order": dict(sorted(Counter(p["order"] for p in room).items())),
            "obstacle_reflections": len(paths) - len(room),
            "obstructed": sum(p["status"] != "clear" for p in paths), "early_arrivals": len(data["arrivals"]["delay_s"]),
            "direct_ms": round(1000*paths[0]["delay_s"], 2) if paths else None,
            "mixing_time_ms": round(1000*data["room"]["mixing_time_s"], 1),
            "measured_overlay": data["measured"]["label"] if data.get("measured") else None}


# ------------------------------------------------------------------------------------------ page
_CSS = r"""
/* Layout: one working sheet. The 3D room fills the left, the echogram runs beneath it on the same
   time axis as the animation, and a narrow control column holds the path list. Stacks on phones. */
:root {
  --paper: #f3f6f7;  --panel: #ffffff;  --ink: #1a2630;  --ink-2: #536470;  --rule: #d4dde2;
  --room: #2d4250;   --box: #9a7440;    --src: #b85a00;  --mic: #0a7c72;  --focus: #1f5fbf;
  --t0: #cf3f55; --t1: #d98b1c; --t2: #23978a; --t3: #3567a6; --t4: #6a4aa5;   /* arrival time, early to late */
  --measured: #7a8790;
  --font-ui: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-tech: "Bahnschrift", "DIN Alternate", "Roboto Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-num: ui-monospace, "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0e1418; --panel: #131b21; --ink: #dce5eb; --ink-2: #91a2ad; --rule: #26323a;
    --room: #a7bbc7;  --box: #c9a165;   --src: #ff9f43; --mic: #3cc8b9;  --focus: #82b1ff;
    --t0: #ff6f82; --t1: #ffbe55; --t2: #4fd0c0; --t3: #7aa7ff; --t4: #b495ff; --measured: #8f9ca5;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #0e1418; --panel: #131b21; --ink: #dce5eb; --ink-2: #91a2ad; --rule: #26323a;
  --room: #a7bbc7;  --box: #c9a165;   --src: #ff9f43; --mic: #3cc8b9;  --focus: #82b1ff;
  --t0: #ff6f82; --t1: #ffbe55; --t2: #4fd0c0; --t3: #7aa7ff; --t4: #b495ff; --measured: #8f9ca5;
  color-scheme: dark;
}
html, body { height: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 14px/1.45 var(--font-ui);
       padding-block: 16px; padding-inline: 16px; box-sizing: border-box; }
*, *::before, *::after { box-sizing: inherit; }
.sheet { display: grid; gap: 12px; max-width: 1440px; margin: 0 auto;
         grid-template-columns: minmax(0, 1fr) minmax(16rem, 21rem);
         grid-template-areas: "head head" "view side" "echo side"; }
header { grid-area: head; display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; }
h1 { font: 600 1.45rem/1.15 var(--font-tech); letter-spacing: .01em; margin: 0; text-wrap: balance; }
.block { font: 12px/1.5 var(--font-num); color: var(--ink-2); font-variant-numeric: tabular-nums;
         display: flex; flex-wrap: wrap; gap: 2px 14px; min-width: 0; }
.block b { color: var(--ink); font-weight: 600; }
.frame { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; position: relative; min-width: 0; }
.view { grid-area: view; aspect-ratio: 16 / 10; max-width: 100%; min-height: 260px; touch-action: none; }
.echo { grid-area: echo; height: 230px; }
canvas { display: block; width: 100%; height: 100%; }
.view canvas { cursor: grab; }
.view canvas:active { cursor: grabbing; }
.corner { position: absolute; display: flex; gap: 6px; }
.corner.tl { top: 8px; left: 8px; } .corner.tr { top: 8px; right: 8px; }
.legend { font: 11px/1.3 var(--font-num); color: var(--ink-2); background: var(--panel); padding: 4px 6px;
          border: 1px solid var(--rule); border-radius: 3px; display: grid; gap: 3px; }
.ramp { height: 6px; width: 140px; border-radius: 2px;
        background: linear-gradient(90deg, var(--t0), var(--t1), var(--t2), var(--t3), var(--t4)); }
.ramp-labels { display: flex; justify-content: space-between; }
button, select { font: 12px var(--font-ui); color: var(--ink); background: var(--panel); border: 1px solid var(--rule);
                 border-radius: 3px; padding: 4px 9px; cursor: pointer; }
button:hover, select:hover { border-color: var(--ink-2); }
button[aria-pressed="true"] { background: var(--ink); color: var(--panel); border-color: var(--ink); }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.side { grid-area: side; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.side section { padding: 10px 12px; display: grid; gap: 8px; }
.side h2 { font: 600 11px/1 var(--font-tech); text-transform: uppercase; letter-spacing: .09em; color: var(--ink-2); margin: 0; }
.orders { display: flex; flex-wrap: wrap; gap: 6px 12px; }
.orders label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
.transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.transport input[type=range] { flex: 1 1 8rem; min-width: 0; accent-color: var(--focus); }
.clock { font: 12px var(--font-num); font-variant-numeric: tabular-nums; min-width: 6.5em; text-align: right; }
.list { margin: 0; padding: 0; list-style: none; max-height: 19rem; overflow: auto; border-top: 1px solid var(--rule); }
.list li { display: grid; grid-template-columns: 10px 6.4em 6em minmax(0, 1fr); gap: 8px; align-items: baseline;
           padding: 5px 2px; border-bottom: 1px solid var(--rule); cursor: pointer; font-size: 12px; }
.list li:hover, .list li.on { background: color-mix(in srgb, var(--focus) 10%, transparent); }
.list .sw { width: 10px; height: 10px; border-radius: 2px; align-self: center; }
.list .num { font-family: var(--font-num); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.list .route { min-width: 0; overflow-wrap: anywhere; }
.chip { font: 10px var(--font-num); border: 1px solid var(--ink-2); color: var(--ink-2); border-radius: 2px; padding: 0 3px; margin-left: 4px; }
.info { font-size: 12.5px; min-height: 5.6em; }
.info .route { font-weight: 600; }
.info dl { display: grid; grid-template-columns: auto 1fr; gap: 1px 10px; margin: 6px 0 0; font-family: var(--font-num);
           font-variant-numeric: tabular-nums; }
.info dt { color: var(--ink-2); }
.info dd { margin: 0; }
.note { font-size: 11.5px; color: var(--ink-2); margin: 0; }
@media (max-width: 860px) {
  .sheet { grid-template-columns: minmax(0, 1fr); grid-template-areas: "head" "view" "echo" "side"; }
  .view { aspect-ratio: 4 / 3; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
"""

_BODY = r"""
<main class="sheet">
  <header>
    <h1 id="title">__TITLE__</h1>
    <div class="block" id="titleblock"></div>
  </header>
  <div class="frame view" id="view-frame">
    <canvas id="view" role="img" aria-label="3D view of the room with the sound paths from the source to the microphone"></canvas>
    <div class="corner tl"><div class="legend"><span>arrival time</span><div class="ramp"></div>
      <div class="ramp-labels"><span id="ramp-lo">0 ms</span><span id="ramp-hi"></span></div></div></div>
    <div class="corner tr"><button type="button" id="zoom-in" aria-label="Zoom in">+</button>
      <button type="button" id="zoom-out" aria-label="Zoom out">&minus;</button>
      <button type="button" id="reset-view">Reset view</button></div>
  </div>
  <div class="frame echo"><canvas id="echo" role="img" aria-label="Echogram: level of each arrival against time"></canvas></div>
  <aside class="side">
    <section class="frame">
      <h2>Show</h2>
      <div class="orders" id="orders"></div>
    </section>
    <section class="frame">
      <h2>Sound travel</h2>
      <div class="transport">
        <button type="button" id="play" aria-pressed="false">Play</button>
        <input type="range" id="time" min="0" max="1000" step="1" value="1000" aria-label="Time since the sound left the source">
        <span class="clock" id="clock"></span>
      </div>
      <div class="transport">
        <label for="speed" class="note">Slow motion</label>
        <select id="speed">
          <option value="0.02">20 ms per second</option>
          <option value="0.008" selected>8 ms per second</option>
          <option value="0.002">2 ms per second</option>
        </select>
        <button type="button" id="rest">Show all</button>
      </div>
    </section>
    <section class="frame info" id="info" aria-live="polite"></section>
    <section class="frame">
      <h2>Paths, by arrival</h2>
      <ul class="list" id="list"></ul>
      <p class="note" id="notes"></p>
    </section>
  </aside>
</main>
"""

_JS = r"""
(() => {
"use strict";
const D = JSON.parse(document.getElementById("scene-data").textContent);
const $ = (id) => document.getElementById(id);
const dims = D.room.dims_m, c = D.room.c_m_s, span = Math.hypot(dims[0], dims[1], dims[2]);
const center = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
const paths = D.paths;
const tMax = Math.max(1e-4, ...paths.map((p) => p.delay_s));
const ORDER = ["Direct", "1st order", "2nd order", "3rd order"];
const state = { yaw: -0.75, pitch: 0.5, zoom: 1, t: null, playing: false, hover: -1, pinned: -1,
                orders: new Set(paths.map((p) => p.order)), boxes: true, last: 0 };
let tok = {};

function readTokens() {
  const s = getComputedStyle(document.documentElement);
  for (const k of ["paper", "panel", "ink", "ink-2", "rule", "room", "box", "src", "mic", "focus", "measured",
                   "t0", "t1", "t2", "t3", "t4"]) tok[k] = s.getPropertyValue("--" + k).trim();
}
function hex(c) { const v = parseInt(c.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function rgba(c, a) { const [r, g, b] = hex(c); return `rgba(${r},${g},${b},${a})`; }
function timeColor(delay, a = 1) {
  const stops = [tok.t0, tok.t1, tok.t2, tok.t3, tok.t4].map(hex);
  const u = Math.max(0, Math.min(1, delay / tMax)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(u)), f = u - i;
  const m = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgba(${m[0]},${m[1]},${m[2]},${a})`;
}
const ms = (s) => (1000 * s).toFixed(s < 0.1 ? 2 : 1) + " ms";
const db = (v) => v === null ? "blocked" : (Math.abs(v) < 0.05 ? "" : v > 0 ? "+" : "−") + Math.abs(v).toFixed(1) + " dB";
const visible = (p) => p.kind === "box" ? state.boxes : state.orders.has(p.order);
const route = (p) => [D.source.name, ...p.via, D.receiver.name].join(" → ");
const weight = (p) => p.level_db === null ? 0 : Math.max(0, Math.min(1, 1 + p.level_db / 30));

function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, r.width, r.height];
}

// ------------------------------------------------------------------------------------- 3D view
function projector(w, h) {
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const dist = 2.4 * span, f = 0.95 * Math.min(w, h) * dist / span * state.zoom;
  return (p) => {
    const x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp, depth = y2 + dist;
    return [w / 2 + f * x1 / depth, h / 2 - f * z2 / depth, depth];
  };
}
function line(ctx, P, a, b) { const p = P(a), q = P(b); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
function boxEdges(min, size) {
  const [x0, y0, z0] = min, x1 = x0 + size[0], y1 = y0 + size[1], z1 = z0 + size[2];
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const e = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]];
  return { v, e, faces };
}
function partial(points, frac) {   // the first `frac` of a polyline, by length
  const seg = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) { const d = Math.hypot(...points[i].map((v, k) => v - points[i - 1][k])); seg.push(d); total += d; }
  let left = frac * total;
  const out = [points[0]];
  for (let i = 1; i < points.length; i++) {
    if (left >= seg[i - 1]) { out.push(points[i]); left -= seg[i - 1]; continue; }
    const f = seg[i - 1] > 0 ? left / seg[i - 1] : 0;
    out.push(points[i - 1].map((v, k) => v + (points[i][k] - v) * f));
    return [out, false];
  }
  return [out, true];
}
let screenPaths = [];
function drawView() {
  const [ctx, w, h] = setupCanvas($("view"));
  const P = projector(w, h);
  ctx.clearRect(0, 0, w, h);
  // floor grid, 1 m
  ctx.lineWidth = 1; ctx.strokeStyle = rgba(tok.rule, 0.9); ctx.beginPath();
  for (let x = 1; x < dims[0]; x++) line(ctx, P, [x, 0, 0], [x, dims[1], 0]);
  for (let y = 1; y < dims[1]; y++) line(ctx, P, [0, y, 0], [dims[0], y, 0]);
  ctx.stroke();
  // room
  const room = boxEdges([0, 0, 0], dims);
  ctx.strokeStyle = tok.room; ctx.lineWidth = 1.25; ctx.beginPath();
  for (const [a, b] of room.e) line(ctx, P, room.v[a], room.v[b]);
  ctx.stroke();
  ctx.fillStyle = tok["ink-2"]; ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--font-num");
  const label = (p, text) => { const q = P(p); ctx.fillText(text, q[0] + 4, q[1] - 4); };
  label([dims[0] / 2, 0, 0], `x ${dims[0].toFixed(2)} m`);
  label([0, dims[1] / 2, 0], `y ${dims[1].toFixed(2)} m`);
  label([0, 0, dims[2] / 2], `z ${dims[2].toFixed(2)} m`);
  // obstacles: faces far to near
  for (const b of D.room.obstacles) {
    const g = boxEdges(b.min_m, b.size_m);
    const faces = g.faces.map((f) => ({ f, d: f.reduce((s, i) => s + P(g.v[i])[2], 0) / 4 })).sort((a, b2) => b2.d - a.d);
    for (const { f } of faces) {
      ctx.beginPath(); f.forEach((i, k) => { const q = P(g.v[i]); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.closePath(); ctx.fillStyle = rgba(tok.box, 0.12); ctx.fill();
    }
    ctx.strokeStyle = tok.box; ctx.lineWidth = 1; ctx.beginPath();
    for (const [a, b2] of g.e) line(ctx, P, g.v[a], g.v[b2]);
    ctx.stroke();
    ctx.fillStyle = tok["ink-2"];
    label([b.min_m[0] + b.size_m[0] / 2, b.min_m[1] + b.size_m[1] / 2, b.min_m[2] + b.size_m[2]], b.name);
  }
  // paths: quiet first, the selected one last
  const focus = state.pinned >= 0 ? state.pinned : state.hover;
  const order = paths.map((p, i) => i).filter((i) => visible(paths[i])).sort((a, b) => weight(paths[a]) - weight(paths[b]));
  if (focus >= 0 && order.includes(focus)) { order.splice(order.indexOf(focus), 1); order.push(focus); }
  screenPaths = [];
  for (const i of order) {
    const p = paths[i];
    const frac = state.t === null ? 1 : Math.max(0, Math.min(1, state.t / p.delay_s));
    if (frac <= 0) continue;
    const [pts, arrived] = partial(p.points, frac);
    const scr = pts.map(P);
    const strong = weight(p), on = i === focus, dim = focus >= 0 && !on;
    ctx.save();
    if (p.status !== "clear") ctx.setLineDash([5, 4]);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (on) { ctx.strokeStyle = rgba(tok.panel, 0.9); ctx.lineWidth = 6; ctx.beginPath(); scr.forEach((q, k) => k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke(); }
    ctx.strokeStyle = timeColor(p.delay_s, dim ? 0.18 : (on ? 1 : 0.35 + 0.6 * strong));
    ctx.lineWidth = on ? 3 : 0.8 + 1.8 * strong;
    ctx.beginPath(); scr.forEach((q, k) => k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke();
    ctx.restore();
    for (const q of scr.slice(1, arrived ? -1 : scr.length)) {   // bounce points
      ctx.fillStyle = timeColor(p.delay_s, dim ? 0.25 : 1); ctx.beginPath(); ctx.arc(q[0], q[1], on ? 3.5 : 2.2, 0, 7); ctx.fill();
    }
    if (!arrived) { const q = scr[scr.length - 1]; ctx.fillStyle = timeColor(p.delay_s, 1); ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 7); ctx.fill(); }
    screenPaths.push([i, scr]);
  }
  // source and microphone
  for (const [m, color, tag] of [[D.source, tok.src, "S"], [D.receiver, tok.mic, "M"]]) {
    const q = P(m.xyz_m), floor = P([m.xyz_m[0], m.xyz_m[1], 0]);
    ctx.strokeStyle = rgba(color, 0.6); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(floor[0], floor[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, 7); ctx.fill();
    ctx.fillStyle = tok.panel; ctx.font = "bold 9px " + getComputedStyle(document.body).getPropertyValue("--font-ui");
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(tag, q[0], q[1] + 0.5);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillStyle = tok.ink;
    ctx.font = "12px " + getComputedStyle(document.body).getPropertyValue("--font-ui");
    ctx.strokeStyle = tok.panel; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.strokeText(m.name, q[0] + 10, q[1] - 8);
    ctx.fillText(m.name, q[0] + 10, q[1] - 8);
  }
}

// ----------------------------------------------------------------------------------- echogram
let stemX = [];
function drawEcho() {
  const [ctx, w, h] = setupCanvas($("echo"));
  ctx.clearRect(0, 0, w, h);
  const L = 44, R = 12, T = 12, B = 26, W = w - L - R, H = h - T - B, view = D.view_s;
  const X = (t) => L + W * t / view, Y = (v) => T + H * Math.min(1, Math.max(0, -v / 60));
  const num = getComputedStyle(document.body).getPropertyValue("--font-num");
  ctx.font = "11px " + num; ctx.fillStyle = tok["ink-2"]; ctx.strokeStyle = tok.rule; ctx.lineWidth = 1;
  for (let v = 0; v >= -60; v -= 10) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + W, Y(v)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(v + " dB", L - 5, Y(v) + 4); }
  const stepMs = [1, 2, 5, 10, 20, 50, 100, 200].find((s) => view * 1000 / s <= 8) || 500;
  ctx.textAlign = "center";
  for (let t = 0; t <= view * 1000 + 1e-9; t += stepMs) { const x = X(t / 1000); ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + H); ctx.stroke(); ctx.fillText(t + " ms", x, T + H + 16); }
  if (D.measured) {
    const m = D.measured; ctx.strokeStyle = rgba(tok.measured, 0.85); ctx.lineWidth = 1; ctx.beginPath();
    m.level_db.forEach((v, k) => { const x = X(k * m.step_s); if (x > L + W) return; k ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); }); ctx.stroke();
  }
  ctx.strokeStyle = rgba(tok["ink-2"], 0.35); ctx.lineWidth = 1; ctx.beginPath();
  D.arrivals.delay_s.forEach((t, k) => { if (t <= view) { ctx.moveTo(X(t), Y(-60)); ctx.lineTo(X(t), Y(D.arrivals.level_db[k])); } }); ctx.stroke();
  const tail = D.tail, end = Math.min(view, tail.start_s + (tail.level_db + 60) / -tail.slope_db_s);
  ctx.setLineDash([6, 4]); ctx.strokeStyle = tok.ink; ctx.lineWidth = 1.25; ctx.beginPath();
  ctx.moveTo(X(tail.start_s), Y(tail.level_db)); ctx.lineTo(X(end), Y(tail.level_db + tail.slope_db_s * (end - tail.start_s))); ctx.stroke();
  ctx.setLineDash([2, 3]); ctx.strokeStyle = tok["ink-2"]; ctx.beginPath(); ctx.moveTo(X(tail.start_s), T); ctx.lineTo(X(tail.start_s), T + H); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = tok["ink-2"]; ctx.textAlign = "left";
  let note = `mixing time ${ms(tail.start_s)}; then late decay, T60 ${tail.t60_s.toFixed(2)} s at 1 kHz`;
  if (ctx.measureText(note).width > W - 10) note = `late decay from ${ms(tail.start_s)}`;
  ctx.fillText(note, Math.max(L + 4, Math.min(X(tail.start_s) + 5, L + W - ctx.measureText(note).width - 4)), T + 11);
  const focus = state.pinned >= 0 ? state.pinned : state.hover;
  stemX = [];
  paths.forEach((p, i) => {
    if (!visible(p) || p.level_db === null || p.delay_s > view) return;
    const x = X(p.delay_s), on = i === focus, lit = state.t === null || state.t >= p.delay_s;
    ctx.strokeStyle = timeColor(p.delay_s, lit ? (focus >= 0 && !on ? 0.3 : 1) : 0.15); ctx.lineWidth = on ? 3.5 : 2;
    ctx.beginPath(); ctx.moveTo(x, Y(-60)); ctx.lineTo(x, Y(p.level_db)); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x, Y(p.level_db), on ? 4 : 2.5, 0, 7); ctx.fill();
    stemX.push([i, x]);
  });
  if (state.t !== null) { ctx.strokeStyle = tok.focus; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(state.t), T); ctx.lineTo(X(state.t), T + H); ctx.stroke(); }
  echoGeom = { L, W, view };
}
let echoGeom = { L: 0, W: 1, view: 1 };

// ------------------------------------------------------------------------------------ panels
function fillPanels() {
  const r = D.room, s = D.source.xyz_m, m = D.receiver.xyz_m, f = (v) => v.map((x) => x.toFixed(2)).join(", ");
  const k = r.bands_hz.indexOf(1000);
  $("titleblock").innerHTML = [
    `room <b>${r.dims_m.map((x) => x.toFixed(2)).join(" × ")} m</b>`, `V ${r.volume_m3.toFixed(1)} m³`, `c ${r.c_m_s.toFixed(1)} m/s`,
    `${esc(D.source.name)} at ${f(s)} m`, `${esc(D.receiver.name)} at ${f(m)} m`,
    `T60 ${r.t60_s[k].toFixed(2)} s at 1 kHz`, `mixing time ${ms(r.mixing_time_s)}`].map((x) => `<span>${x}</span>`).join("");
  $("ramp-hi").textContent = ms(tMax);
  const orders = [...new Set(paths.filter((p) => p.kind === "room").map((p) => p.order))].sort();
  const counts = (o) => paths.filter((p) => p.kind === "room" && p.order === o).length;
  $("orders").innerHTML = orders.map((o) => `<label><input type="checkbox" id="order-${o}" data-order="${o}" checked> ${ORDER[o]} <span class="note">${counts(o)}</span></label>`).join("")
    + (paths.some((p) => p.kind === "box") ? `<label><input type="checkbox" id="order-box" checked> Obstacles <span class="note">${paths.filter((p) => p.kind === "box").length}</span></label>` : "");
  $("orders").addEventListener("change", (e) => {
    const box = e.target;
    if (box.id === "order-box") state.boxes = box.checked;
    else if (box.checked) state.orders.add(+box.dataset.order); else state.orders.delete(+box.dataset.order);
    renderList(); draw();
  });
  const notes = [`Level: ${D.level_note}.`, `Grey stems: ${D.arrivals.note}.`];
  if (D.measured) notes.push(`Grey line: measured response (${esc(D.measured.label)}), 0 dB at its direct sound.`);
  if (paths.some((p) => p.status !== "clear")) notes.push("Dashed: the straight path is blocked by an obstacle; the renderer routes it around the box (longer, quieter) or drops it.");
  $("notes").innerHTML = notes.join(" ");
  renderList();
}
function esc(t) { return String(t).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])); }
function renderList() {
  const focus = state.pinned >= 0 ? state.pinned : state.hover;
  $("list").innerHTML = paths.map((p, i) => visible(p) ? `<li data-i="${i}" class="${i === focus ? "on" : ""}" tabindex="0">
    <span class="sw" style="background:${timeColor(p.delay_s)}"></span><span class="num">${ms(p.delay_s)}</span>
    <span class="num">${db(p.level_db)}</span><span class="route">${p.via.length ? esc(p.via.join(" → ")) : "direct"}${p.status !== "clear" ? `<span class="chip">${p.status}</span>` : ""}</span></li>` : "").join("");
}
function showInfo() {
  const i = state.pinned >= 0 ? state.pinned : state.hover;
  if (i < 0) {
    const direct = paths.find((p) => p.order === 0);
    $("info").innerHTML = `<h2>Selected path</h2><p class="note">Point at a path, a stem or a row to see its route. Click to keep it selected. Drag the room to turn it.</p>`
      + (direct ? `<p class="note">The direct sound arrives after ${ms(direct.delay_s)}; ${paths.length - 1} reflections up to ${ORDER[D.max_order].toLowerCase()} follow within ${ms(tMax)}.</p>` : "");
    return;
  }
  const p = paths[i];
  $("info").innerHTML = `<h2>Selected path</h2><div class="route">${esc(route(p))}</div><dl>
    <dt>arrives</dt><dd>${ms(p.delay_s)}</dd><dt>length</dt><dd>${p.length_m.toFixed(2)} m</dd>
    <dt>level</dt><dd>${db(p.level_db)} re direct</dd><dt>bounces</dt><dd>${p.kind === "box" ? "1 (obstacle)" : p.order}</dd>
    ${p.status !== "clear" ? `<dt>obstacle</dt><dd>${p.status === "dropped" ? "blocked, not rendered" : "goes around a box"}</dd>` : ""}</dl>`;
}

// ------------------------------------------------------------------------------- interaction
function draw() { drawView(); drawEcho(); showInfo(); $("clock").textContent = state.t === null ? "all paths" : "t = " + ms(state.t); }
function setHover(i) { if (i !== state.hover) { state.hover = i; renderList(); draw(); } }
function nearestPath(x, y) {
  let best = -1, bestD = 7;
  for (const [i, scr] of screenPaths) for (let k = 1; k < scr.length; k++) {
    const [ax, ay] = scr[k - 1], [bx, by] = scr[k], dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy || 1;
    const u = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L2)), d = Math.hypot(ax + u * dx - x, ay + u * dy - y);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
const view = $("view");
let drag = null;
view.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, yaw: state.yaw, pitch: state.pitch, moved: false }; view.setPointerCapture(e.pointerId); });
view.addEventListener("pointermove", (e) => {
  const r = view.getBoundingClientRect();
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    state.yaw = drag.yaw + dx * 0.008; state.pitch = Math.max(-0.2, Math.min(1.45, drag.pitch + dy * 0.006)); draw();
  } else setHover(nearestPath(e.clientX - r.left, e.clientY - r.top));
});
view.addEventListener("pointerup", (e) => {
  if (drag && !drag.moved) { const r = view.getBoundingClientRect(), i = nearestPath(e.clientX - r.left, e.clientY - r.top); state.pinned = i === state.pinned ? -1 : i; renderList(); draw(); }
  drag = null;
});
view.addEventListener("pointerleave", () => { if (!drag) setHover(-1); });
view.addEventListener("wheel", (e) => { e.preventDefault(); state.zoom = Math.max(0.5, Math.min(4, state.zoom * Math.exp(-e.deltaY * 0.0015))); draw(); }, { passive: false });
$("zoom-in").onclick = () => { state.zoom = Math.min(4, state.zoom * 1.25); draw(); };
$("zoom-out").onclick = () => { state.zoom = Math.max(0.5, state.zoom / 1.25); draw(); };
$("reset-view").onclick = () => { state.yaw = -0.75; state.pitch = 0.5; state.zoom = 1; draw(); };
const echo = $("echo");
const nearestStem = (x) => { let best = -1, bestD = 6; for (const [i, sx] of stemX) { const d = Math.abs(sx - x); if (d < bestD) { bestD = d; best = i; } } return best; };
echo.addEventListener("pointermove", (e) => setHover(nearestStem(e.clientX - echo.getBoundingClientRect().left)));
echo.addEventListener("pointerleave", () => setHover(-1));
echo.addEventListener("click", (e) => {
  const x = e.clientX - echo.getBoundingClientRect().left, i = nearestStem(x);
  if (i >= 0) { state.pinned = i === state.pinned ? -1 : i; renderList(); draw(); return; }
  const g = echoGeom, t = (x - g.L) / g.W * g.view;
  if (t >= 0 && t <= g.view) { stop(); state.t = Math.min(t, tMax * 1.05); syncSlider(); draw(); }
});
$("list").addEventListener("mouseover", (e) => { const li = e.target.closest("li"); if (li) setHover(+li.dataset.i); });
$("list").addEventListener("mouseleave", () => setHover(-1));
const pick = (li) => { const i = +li.dataset.i; state.pinned = i === state.pinned ? -1 : i; renderList(); draw(); };
$("list").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) pick(li); });
$("list").addEventListener("keydown", (e) => { const li = e.target.closest("li"); if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pick(li); } });

// ------------------------------------------------------------------------------------- playback
const tEnd = tMax * 1.05;
function syncSlider() { $("time").value = state.t === null ? 1000 : Math.round(1000 * state.t / tEnd); }
function stop() { state.playing = false; $("play").setAttribute("aria-pressed", "false"); $("play").textContent = "Play"; }
function tick(now) {
  if (!state.playing) return;
  const dt = Math.min(0.1, (now - state.last) / 1000); state.last = now;
  state.t = Math.min(tEnd, (state.t ?? 0) + dt * +$("speed").value);
  syncSlider(); draw();
  if (state.t >= tEnd) { stop(); return; }
  requestAnimationFrame(tick);
}
$("play").onclick = () => {
  if (state.playing) { stop(); return; }
  if (state.t === null || state.t >= tEnd) state.t = 0;
  state.playing = true; state.last = performance.now(); $("play").setAttribute("aria-pressed", "true"); $("play").textContent = "Pause";
  requestAnimationFrame(tick);
};
$("time").oninput = (e) => { stop(); state.t = +e.target.value / 1000 * tEnd; draw(); };
$("rest").onclick = () => { stop(); state.t = null; syncSlider(); draw(); };

readTokens(); fillPanels(); draw();
const resized = new ResizeObserver(() => { drawView(); drawEcho(); });   // canvases only: no layout change, no loop
resized.observe($("view-frame")); resized.observe(echo);
const retheme = () => { readTokens(); renderList(); draw(); };
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
"""
