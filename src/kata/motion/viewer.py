"""Local 3D viewer pages (docs/phases/phase-4-motion.md §7.3): two clips of one song side by side,
synced to an audio file the viewer picks. Pages stay on this machine; audio is never embedded.

The game's axes are left-handed (+z towards the notes); three.js is right-handed. Clips are
converted by flipping z: positions (x, y, -z), quaternions (-qx, -qy, qz, qw). The blade, along
saber-local +z in the game, is then along local -z.
"""

from __future__ import annotations

import html
import json
import random
from pathlib import Path

import numpy as np

from kata.motion.geometry import matrix_to_quat, rot6_to_matrix
from kata.motion.prepare import FPS, TRACK

THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"


def clip(motion: np.ndarray, hits: tuple, frame0: int = 0, notes: list | None = None) -> dict:
    """(T, 27) motion -> viewer JSON: per part flat [x, y, z, qx, qy, qz, qw] per frame (z-flipped)."""
    out = {"fps": FPS, "frame0": int(frame0), "hits": [np.round(np.asarray(h, float), 4).tolist() for h in hits],
           "notes": notes or []}
    for part in ("head", "left", "right"):
        s = TRACK[part].start
        pos = motion[:, s:s + 3].astype(np.float64) * [1, 1, -1]
        q = matrix_to_quat(rot6_to_matrix(motion[:, s + 3:s + 9])) * [-1, -1, 1, 1]
        out[part] = np.round(np.concatenate([pos, q], 1), 4).ravel().tolist()
    return out


def page(title: str, a: dict, b: dict, labels: tuple[str, str], blind: bool, seed: int | None = None) -> str:
    """One HTML page; with blind=True the clips are shown as A/B in random order until revealed.

    The order is fresh on every call unless a seed is given. The map's notes are shared by both
    clips and, in blind mode, hidden until Reveal: only the real clip was played against them, so
    its sabers meet the notes and the generated one's do not."""
    clips = [(a, labels[0]), (b, labels[1])]
    if blind:
        (random.SystemRandom() if seed is None else random.Random(seed)).shuffle(clips)
    notes = a.get("notes") or b.get("notes") or []
    data = {"clips": [{k: v for k, v in c.items() if k != "notes"} for c, _ in clips],
            "labels": [lab for _, lab in clips], "notes": notes, "blind": blind}
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__THREE__", THREE_URL).replace(
        "__DATA__", json.dumps(data, separators=(",", ":")))


def write_page(path: Path, *args, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(*args, **kwargs), encoding="utf-8")
    return path


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #f4f4f2; --fg: #1d1d1b; --muted: #6b6b66; --panel: #ffffff; --line: #d8d8d2; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecece8; --muted: #9a9a94; --panel: #20201e; --line: #34342f; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; }
header { padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
h1 { font-size: 16px; margin: 0 12px 0 0; }
.views { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; padding: 0 16px 16px; }
.view { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.view canvas { width: 100%; height: 420px; display: block; }
.view .tag { padding: 8px 12px; color: var(--muted); }
input[type=range] { flex: 1 1 240px; }
button { font: inherit; padding: 4px 12px; }
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <label>Song audio <input id="file" type="file" accept="audio/*,.egg,.ogg"></label>
  <button id="play">Play</button>
  <input id="seek" type="range" min="0" max="1" step="0.001" value="0">
  <span id="time">0.0 s</span>
  <button id="reveal" hidden>Reveal</button>
</header>
<div class="views"><div class="view"><canvas id="c0"></canvas><div class="tag" id="t0"></div></div>
<div class="view"><canvas id="c1"></canvas><div class="tag" id="t1"></div></div></div>
<audio id="audio"></audio>
<script type="module">
import * as THREE from "__THREE__";
const DATA = __DATA__;
const audio = document.getElementById("audio");
let playing = false, clock0 = 0, offset = 0, showNotes = !DATA.blind;
const duration = Math.max(...DATA.clips.map(c => (c.frame0 + c.head.length / 7) / c.fps));
const now = () => audio.src ? audio.currentTime : (playing ? offset + (performance.now() - clock0) / 1000 : offset);
function makeView(canvas, c) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--panel").trim());
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
  const cam = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
  cam.position.set(0, 1.7, 2.6); cam.lookAt(0, 1.2, -2);
  const grid = new THREE.GridHelper(4, 8, 0x888888, 0x555555); scene.add(grid);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), new THREE.MeshStandardMaterial({color: 0xbbbbbb}));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.12, 12), new THREE.MeshStandardMaterial({color: 0x777777}));
  nose.rotation.x = -Math.PI / 2; nose.position.z = -0.14; head.add(nose); scene.add(head);
  const sabers = [0xd8332a, 0x2a6fd8].map(col => {
    const g = new THREE.BoxGeometry(0.03, 0.03, 1.0); g.translate(0, 0, -0.5);
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: col, emissive: 0x000000}));
    scene.add(m); return m;
  });
  const notes = DATA.notes.map(n => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4),
      new THREE.MeshStandardMaterial({color: n[3] === 0 ? 0xd8332a : 0x2a6fd8, transparent: true, opacity: 0.8}));
    m.position.set((n[1] - 1.5) * 0.6, 0.25 + n[2] * 0.55, 0); scene.add(m); return [n[0], m];
  });
  const parts = [["head", head], ["left", sabers[0]], ["right", sabers[1]]];
  function pose(arr, i, obj) {
    const k = 7 * i; obj.position.set(arr[k], arr[k + 1], arr[k + 2]);
    obj.quaternion.set(arr[k + 3], arr[k + 4], arr[k + 5], arr[k + 6]);
  }
  function draw(t) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
    const n = c.head.length / 7;
    const i = Math.min(n - 1, Math.max(0, Math.round(t * c.fps) - c.frame0));
    for (const [name, obj] of parts) pose(c[name], i, obj);
    c.hits.forEach((hs, s) => {
      const lit = hs.some(x => t >= x && t - x < 0.1);
      sabers[s].material.emissive.setHex(lit ? 0x666666 : 0x000000);
    });
    for (const [tn, m] of notes) { const z = -(tn - t) * 10; m.visible = showNotes && z < 0.5 && z > -30; m.position.z = z; }
    renderer.render(scene, cam);
  }
  return draw;
}
const views = DATA.clips.map((c, k) => makeView(document.getElementById("c" + k), c));
const tags = [0, 1].map(k => document.getElementById("t" + k));
function label(reveal) { tags.forEach((el, k) => el.textContent = DATA.blind && !reveal ? ["A", "B"][k] : DATA.labels[k]); }
label(false);
if (DATA.blind) { const r = document.getElementById("reveal"); r.hidden = false; r.onclick = () => { label(true); showNotes = true; }; }
document.getElementById("file").onchange = e => { audio.src = URL.createObjectURL(e.target.files[0]); };
document.getElementById("play").onclick = () => {
  if (audio.src) { audio.paused ? audio.play() : audio.pause(); playing = !audio.paused; }
  else { if (playing) { offset = now(); playing = false; } else { clock0 = performance.now(); playing = true; } }
};
const seek = document.getElementById("seek");
seek.oninput = () => { const t = seek.value * duration; if (audio.src) audio.currentTime = t; offset = t; clock0 = performance.now(); };
(function loop() {
  const t = now();
  views.forEach(v => v(t));
  document.getElementById("time").textContent = t.toFixed(1) + " s";
  if (document.activeElement !== seek) seek.value = t / duration;
  requestAnimationFrame(loop);
})();
</script></body></html>
"""
