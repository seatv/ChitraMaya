// chitramaya/static/js/sbs_viewer.js
// ── SBS Restoration Viewer ─────────────────────────────────
// Projected equirect-180 SBS preview with original-vs-restored wipe compare.
// ES module (loaded with type="module"); vendors three.js from this folder.
// Sources: /video (original) and /preview-video (restored) — both already
// served with Range support. No server round-trips beyond /api/sbs-status.

import * as THREE from "./three.module.min.js";

const Z = 10500;                       // above every existing overlay (max 9999)
const $id = (i) => document.getElementById(i);

// ── state ──────────────────────────────────────────────────
const S = {
  built: false, open: false,
  eye: 0,
  view: "wipe",                        // "a" (original) | "b" (restored) | "wipe"
  wipeFrac: 0.5,
  rate: 1,                             // base playback speed (both videos)
  yaw: 0, pitch: 0, fov: 75,
  hasA: false, hasB: false,
  offset: 0,                           // seconds added to A when following B
  master: null,
  seeking: false,
};
let renderer, scene, camera, uniforms, videoA, videoB, texA, texB;
let root, stage, divider, raf = 0;

// ── DOM (built lazily on first open) ───────────────────────
const CSS = `
#sbsRoot { position: fixed; inset: 0; z-index: ${Z}; background: #10151b;
  color: #cfd8e3; font: 13px/1.4 "Segoe UI", system-ui, sans-serif; }
#sbsRoot.hidden { display: none; }
#sbsRoot .sbs-bar { position: absolute; left: 0; right: 0; display: flex;
  align-items: center; gap: 10px; padding: 0 10px; background: #1a222c; }
#sbsTop { top: 0; height: 40px; border-bottom: 1px solid #000; }
#sbsBot { bottom: 0; height: 44px; background: #141b23; border-top: 1px solid #000; }
#sbsStage { position: absolute; inset: 40px 0 44px 0; }
#sbsStage canvas { display: block; width: 100%; height: 100%; cursor: grab; }
#sbsStage canvas.dragging { cursor: grabbing; }
#sbsRoot button, #sbsRoot input[type=number], #sbsRoot select { background: #243040; color: #cfd8e3;
  border: 1px solid #33455c; border-radius: 4px; padding: 4px 10px; font: inherit;
  cursor: pointer; }
#sbsRoot button:hover { background: #2d3c50; }
#sbsRoot button.on { background: #4da3ff; border-color: #4da3ff; color: #08121e; }
#sbsRoot button:disabled { opacity: .4; cursor: default; }
#sbsRoot .sbs-grp { display: flex; }
#sbsRoot .sbs-grp button { border-radius: 0; }
#sbsRoot .sbs-grp button:first-child { border-radius: 4px 0 0 4px; }
#sbsRoot .sbs-grp button:last-child { border-radius: 0 4px 4px 0; }
#sbsRoot .sbs-lbl { color: #7e8b9a; }
#sbsRoot .sbs-spacer { flex: 1; }
#sbsOffset { width: 64px; padding: 3px 6px; cursor: text; }
#sbsSeek { flex: 1; accent-color: #4da3ff; }
#sbsTime { min-width: 130px; text-align: center; font-variant-numeric: tabular-nums; }
#sbsDivider { position: absolute; top: 40px; bottom: 44px; width: 3px;
  background: #4da3ff; cursor: ew-resize; display: none; }
#sbsDivider::after { content: "\\25C0 \\25B6"; position: absolute; top: 12px;
  left: 50%; transform: translateX(-50%); background: #4da3ff; color: #08121e;
  padding: 2px 6px; border-radius: 10px; font-size: 11px; white-space: nowrap; }
#sbsTagA, #sbsTagB { position: absolute; top: 48px; display: none;
  padding: 2px 8px; border-radius: 3px; font-size: 11px; background: #0009; }
#sbsTagA { color: #e0a050; } #sbsTagB { color: #59c98a; }
#sbsMsg { position: absolute; left: 50%; top: 46%; transform: translate(-50%,-50%);
  max-width: 460px; text-align: center; color: #7e8b9a; background: #0008;
  padding: 16px 20px; border-radius: 8px; }
#sbsMsg b { color: #cfd8e3; }
#sbsErr { position: absolute; left: 10px; bottom: 54px; max-width: 60%;
  color: #ff9c9c; background: #2a1518; border: 1px solid #5c2b31;
  padding: 6px 10px; border-radius: 4px; display: none; white-space: pre-wrap; }
`;

const HTML = `
<div id="sbsTop" class="sbs-bar">
  <b style="color:#4da3ff">SBS View</b>
  <span class="sbs-lbl">equirect-180 L|R</span>
  <span class="sbs-lbl">|</span>
  <span class="sbs-lbl">Eye</span>
  <div class="sbs-grp">
    <button id="sbsEyeL" class="on" title="Left eye [1]">L</button>
    <button id="sbsEyeR" title="Right eye [2]">R</button>
  </div>
  <span class="sbs-lbl">View</span>
  <div class="sbs-grp">
    <button id="sbsViewA" disabled title="Original full screen [o]">Original</button>
    <button id="sbsViewB" disabled title="Restored full screen [r]">Restored</button>
    <button id="sbsViewW" disabled title="Wipe compare: original left of the divider, restored right [w]">Wipe</button>
  </div>
  <span class="sbs-lbl" title="Seconds added to the original's clock when the restored preview is a segment (auto-filled from the segment start)">Offset</span>
  <input id="sbsOffset" type="number" step="0.1" value="0">
  <span class="sbs-spacer"></span>
  <span class="sbs-lbl" id="sbsFov">FOV 75</span>
  <button id="sbsReset" title="Recenter view [0]">Reset view</button>
  <button id="sbsClose" title="Close [Esc]">Close</button>
</div>
<div id="sbsStage"></div>
<div id="sbsDivider"></div>
<div id="sbsTagA">ORIGINAL</div>
<div id="sbsTagB">RESTORED</div>
<div id="sbsMsg"><b>Loading sources...</b></div>
<div id="sbsErr"></div>
<div id="sbsBot" class="sbs-bar">
  <button id="sbsSkipB" disabled title="Skip back (Left arrow — uses the app's skip amount)">&#171;</button>
  <button id="sbsFrameB" disabled title="Frame step back (,)">&#8249;</button>
  <button id="sbsPlay" disabled>Play</button>
  <button id="sbsFrameF" disabled title="Frame step forward (.)">&#8250;</button>
  <button id="sbsSkipF" disabled title="Skip forward (Right arrow — uses the app's skip amount)">&#187;</button>
  <span id="sbsTime">0:00.0 / 0:00.0</span>
  <input id="sbsSeek" type="range" min="0" max="1000" value="0" disabled>
  <span class="sbs-lbl">Speed</span>
  <select id="sbsRate" title="Playback speed — slower speeds also lighten decode load on big sources">
    <option value="0.1">0.1x</option><option value="0.25">0.25x</option>
    <option value="0.5">0.5x</option><option value="0.75">0.75x</option>
    <option value="1" selected>1x</option>
  </select>
  <button id="sbsMute" class="on" title="Audio follows the restored video when present (M)">Muted</button>
</div>`;

function makeVideo() {
  const v = document.createElement("video");
  v.playsInline = true; v.loop = true; v.muted = true; v.preload = "auto";
  return v;
}

function build() {
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);

  root = document.createElement("div");
  root.id = "sbsRoot"; root.className = "hidden";
  root.innerHTML = HTML;
  document.body.appendChild(root);
  stage = $id("sbsStage");
  divider = $id("sbsDivider");

  videoA = makeVideo(); videoB = makeVideo();

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(S.fov, 1, 0.1, 2000);

  texA = new THREE.VideoTexture(videoA);
  texB = new THREE.VideoTexture(videoB);
  for (const t of [texA, texB]) {
    t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
  }

  // Equirect-180 SBS LR sampled by view direction (projection-correct).
  // A fisheye layout later replaces only the lon/lat -> uv mapping here.
  uniforms = {
    uTexA: { value: texA }, uTexB: { value: texB },
    uHasA: { value: 0 },    uHasB: { value: 0 },
    uEye:  { value: 0.0 },  uWipeX: { value: 1e9 },
    uView: { value: 0 },    // 0 = original full, 1 = restored full, 2 = wipe
  };
  const material = new THREE.ShaderMaterial({
    uniforms, side: THREE.BackSide, depthWrite: false,
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      precision highp float;
      varying vec3 vDir;
      uniform sampler2D uTexA, uTexB;
      uniform float uEye, uWipeX;
      uniform int uHasA, uHasB, uView;
      const float PI = 3.141592653589793;
      void main() {
        vec3 d = normalize(vDir);
        float lon = atan(d.x, -d.z);
        float lat = asin(clamp(d.y, -1.0, 1.0));
        if (abs(lon) > PI * 0.5) {
          gl_FragColor = vec4(0.045, 0.055, 0.07, 1.0);
          return;
        }
        float ue = (lon + PI * 0.5) / PI;
        float v  = (lat + PI * 0.5) / PI;
        vec2 uv  = vec2(uEye + 0.5 * ue, v);
        bool showA = (uView == 2) ? (gl_FragCoord.x < uWipeX) : (uView == 0);
        if (showA && uHasA == 1)       gl_FragColor = texture2D(uTexA, uv);
        else if (!showA && uHasB == 1) gl_FragColor = texture2D(uTexB, uv);
        else if (uHasA == 1)           gl_FragColor = texture2D(uTexA, uv);
        else if (uHasB == 1)           gl_FragColor = texture2D(uTexB, uv);
        else gl_FragColor = vec4(0.045, 0.055, 0.07, 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(500, 96, 64), material));

  new ResizeObserver(resize).observe(stage);
  wireControls();
  S.built = true;
}

function resize() {
  if (!S.built) return;
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  layoutDivider();
}

// ── controls ───────────────────────────────────────────────
function wireControls() {
  const cnv = renderer.domElement;
  let dragging = false, px = 0, py = 0;
  cnv.addEventListener("pointerdown", (e) => {
    dragging = true; px = e.clientX; py = e.clientY;
    cnv.classList.add("dragging"); cnv.setPointerCapture(e.pointerId);
  });
  cnv.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const k = 0.0022 * (S.fov / 75);
    S.yaw   -= (e.clientX - px) * k;
    S.pitch += (e.clientY - py) * k;
    S.pitch = Math.max(-1.55, Math.min(1.55, S.pitch));
    S.yaw   = Math.max(-2.2,  Math.min(2.2,  S.yaw));
    px = e.clientX; py = e.clientY;
  });
  cnv.addEventListener("pointerup", () => { dragging = false; cnv.classList.remove("dragging"); });
  cnv.addEventListener("wheel", (e) => {
    e.preventDefault();
    S.fov = Math.max(30, Math.min(110, S.fov + Math.sign(e.deltaY) * 5));
    camera.fov = S.fov; camera.updateProjectionMatrix();
    $id("sbsFov").textContent = "FOV " + S.fov;
  }, { passive: false });

  let divDrag = false;
  divider.addEventListener("pointerdown", (e) => {
    divDrag = true; divider.setPointerCapture(e.pointerId); e.preventDefault();
  });
  root.addEventListener("pointermove", (e) => {
    if (!divDrag) return;
    S.wipeFrac = Math.max(0.02, Math.min(0.98, e.clientX / root.clientWidth));
    layoutDivider();
  });
  root.addEventListener("pointerup", () => { divDrag = false; });

  $id("sbsEyeL").onclick = () => setEye(0);
  $id("sbsEyeR").onclick = () => setEye(1);
  $id("sbsViewA").onclick = () => setView("a");
  $id("sbsViewB").onclick = () => setView("b");
  $id("sbsViewW").onclick = () => setView("wipe");
  $id("sbsRate").onchange = (e) => setRate(parseFloat(e.target.value) || 1);
  $id("sbsReset").onclick = resetView;
  $id("sbsClose").onclick = close;
  $id("sbsOffset").onchange = (e) => { S.offset = parseFloat(e.target.value) || 0; };

  $id("sbsPlay").onclick = () => {
    const m = S.master; if (!m) return;
    if (m.paused) {
      if (videoA.src) videoA.play().catch(() => {});
      if (videoB.src) videoB.play().catch(() => {});
    } else { videoA.pause(); videoB.pause(); }
  };
  $id("sbsSeek").oninput = (e) => {
    const m = S.master; if (!m || !isFinite(m.duration)) return;
    S.seeking = true;
    seekAll((e.target.value / 1000) * m.duration);
  };
  $id("sbsSkipB").onclick  = () => skipBy(-1);
  $id("sbsSkipF").onclick  = () => skipBy(1);
  $id("sbsFrameB").onclick = () => stepFrame(-1);
  $id("sbsFrameF").onclick = () => stepFrame(1);
  $id("sbsSeek").onchange = () => { S.seeking = false; };
  $id("sbsMute").onclick = () => {
    const m = S.master; if (!m) return;
    m.muted = !m.muted;
    $id("sbsMute").textContent = m.muted ? "Muted" : "Audio";
    $id("sbsMute").classList.toggle("on", m.muted);
  };

  // Capture-phase on window: fires before the app's document-level shortcut
  // handlers (player.js), so viewer keys never leak to the hidden player.
  window.addEventListener("keydown", (e) => {
    if (!S.open) return;
    if (e.target.tagName === "INPUT") return;
    e.stopPropagation();
    if (e.code === "Space") { e.preventDefault(); $id("sbsPlay").click(); }
    else if (e.key === "Escape") close();
    else if (e.key === "1") setEye(0);
    else if (e.key === "2") setEye(1);
    else if (e.key === "o") setView("a");
    else if (e.key === "r") setView("b");
    else if (e.key === "w") setView("wipe");
    else if (e.key === "0") resetView();
    else if (e.key === "ArrowRight") { e.preventDefault(); skipBy(1); }
    else if (e.key === "ArrowLeft")  { e.preventDefault(); skipBy(-1); }
    else if (e.key === ".") { e.preventDefault(); stepFrame(1); }
    else if (e.key === ",") { e.preventDefault(); stepFrame(-1); }
    else if (e.key === "m" || e.key === "M") { e.preventDefault(); $id("sbsMute").click(); }
  }, true);
}

function setEye(i) {
  S.eye = i; uniforms.uEye.value = i ? 0.5 : 0.0;
  $id("sbsEyeL").classList.toggle("on", i === 0);
  $id("sbsEyeR").classList.toggle("on", i === 1);
}
function setView(v) {
  // Guard: each view needs its source(s); fall back to what exists.
  if (v === "wipe" && !(S.hasA && S.hasB)) v = S.hasA ? "a" : "b";
  if (v === "a" && !S.hasA) v = "b";
  if (v === "b" && !S.hasB) v = "a";
  S.view = v;
  uniforms.uView.value = v === "a" ? 0 : v === "b" ? 1 : 2;
  $id("sbsViewA").classList.toggle("on", v === "a");
  $id("sbsViewB").classList.toggle("on", v === "b");
  $id("sbsViewW").classList.toggle("on", v === "wipe");
  layoutDivider();
}
function updateViewButtons() {
  $id("sbsViewA").disabled = !S.hasA;
  $id("sbsViewB").disabled = !S.hasB;
  $id("sbsViewW").disabled = !(S.hasA && S.hasB);
}
function setRate(r) {
  S.rate = r;
  videoA.playbackRate = r;
  videoB.playbackRate = r;
  $id("sbsRate").value = String(r);
}
function resetView() {
  S.yaw = 0; S.pitch = 0; S.fov = 75;
  camera.fov = 75; camera.updateProjectionMatrix();
  $id("sbsFov").textContent = "FOV 75";
}
function layoutDivider() {
  const on = S.view === "wipe" && S.hasA && S.hasB;
  divider.style.display = on ? "block" : "none";
  $id("sbsTagA").style.display = on ? "block" : "none";
  $id("sbsTagB").style.display = on ? "block" : "none";
  if (on) {
    const w = root.clientWidth;
    const x = S.wipeFrac * w;
    divider.style.left = (x - 1.5) + "px";
    $id("sbsTagA").style.left = Math.max(8, x - 90) + "px";
    $id("sbsTagB").style.left = Math.min(w - 90, x + 12) + "px";
  }
}
function showErr(t) { const el = $id("sbsErr"); el.textContent = t; el.style.display = "block"; }

// The app's detected source fps (player.js top-level `let fps`, a global
// lexical binding readable from this module). null when not yet detected.
function appFps() {
  try {
    if (typeof fps !== "undefined" && fps && isFinite(fps) && fps > 0) return fps;
  } catch (e) { /* noop */ }
  return null;
}
// The app's configurable skip amounts (same inputs player.js reads).
function skipAmt(back) {
  const el = $id(back ? "skipBackward" : "skipForward");
  const v = el ? parseFloat(el.value) : 5;
  return (v && isFinite(v) && v > 0) ? v : 5;
}
// Seek both videos, keeping the segment offset on the original.
function seekAll(t) {
  const m = S.master; if (!m) return;
  const d = isFinite(m.duration) ? m.duration : 1e9;
  const nt = Math.max(0, Math.min(d, t));
  if (m === videoB) {
    videoB.currentTime = nt;
    if (videoA.src) videoA.currentTime = nt + S.offset;
  } else {
    videoA.currentTime = nt;
  }
}
// Frame step: pause first (same semantics as the main player's , and .)
function stepFrame(dir) {
  const m = S.master; if (!m) return;
  if (videoA.src && !videoA.paused) videoA.pause();
  if (videoB.src && !videoB.paused) videoB.pause();
  seekAll(m.currentTime + dir / (appFps() || 30));
}
function skipBy(dir) {
  const m = S.master; if (!m) return;
  seekAll(m.currentTime + dir * skipAmt(dir < 0));
}
function fmt(t) {
  if (!isFinite(t)) t = 0;
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return m + ":" + s;
}

// ── sources ────────────────────────────────────────────────
function attach(which, url, label) {
  const v = which === "a" ? videoA : videoB;
  v.src = url;
  v.load();
  v.onerror = () => showErr(
    "Could not decode the " + label + " video in the app's browser view.\n" +
    "8K HEVC may exceed the embedded decoder - a smaller proxy will play.");
  v.onloadedmetadata = () => {
    if (which === "a") { S.hasA = true; uniforms.uHasA.value = 1; }
    else               { S.hasB = true; uniforms.uHasB.value = 1; }
    S.master = S.hasB ? videoB : videoA;
    v.playbackRate = S.rate;
    for (const b of ["sbsPlay", "sbsSeek", "sbsSkipB", "sbsSkipF", "sbsFrameB", "sbsFrameF"])
      $id(b).disabled = false;
    $id("sbsMsg").style.display = "none";
    updateViewButtons();
    setView(S.hasA && S.hasB ? "wipe" : (S.hasA ? "a" : "b"));
  };
}

function detach() {
  for (const v of [videoA, videoB]) {
    try { v.pause(); v.removeAttribute("src"); v.load(); } catch (e) { /* noop */ }
  }
  S.hasA = S.hasB = false; S.master = null;
  uniforms.uHasA.value = 0; uniforms.uHasB.value = 0;
  for (const b of ["sbsPlay", "sbsSeek", "sbsSkipB", "sbsSkipF", "sbsFrameB", "sbsFrameF"])
    $id(b).disabled = true;
  updateViewButtons();
}

// ── open / close ───────────────────────────────────────────
async function open() {
  if (!S.built) build();

  // Pause the main player: no double audio, and free its share of the decoder.
  const mainPlayer = $id("player");
  if (mainPlayer) { try { mainPlayer.pause(); } catch (e) { /* noop */ } }

  root.classList.remove("hidden");
  S.open = true;
  $id("sbsMsg").innerHTML = "<b>Loading sources...</b>";
  $id("sbsMsg").style.display = "block";
  $id("sbsErr").style.display = "none";
  resetView();
  setRate(1);
  resize();

  let st = { has_video: false, has_preview: false, preview_is_segment: false };
  try { st = await (await fetch("/api/sbs-status")).json(); } catch (e) { /* noop */ }

  if (!st.has_video && !st.has_preview) {
    $id("sbsMsg").innerHTML =
      "<b>No video loaded.</b><br><br>Load a video (and run a restore for the " +
      "wipe compare), then reopen SBS View.<br><br>" +
      "Drag to look around - wheel zoom - space play/pause - 1/2 eye - " +
      "o/r/w view (Original / Restored / Wipe) - ,/. frame step - Esc close";
    return;
  }

  // Segment previews start at the segment's own t=0; offset the original.
  let off = 0;
  if (st.preview_is_segment) {
    try { off = (typeof state !== "undefined" && state.segmentStartTime) || 0; }
    catch (e) { off = 0; }
  }
  S.offset = off;
  $id("sbsOffset").value = off.toFixed(1);

  const t = Date.now();
  if (st.has_video)   attach("a", "/video?t=" + t, "original");
  if (st.has_preview) attach("b", "/preview-video?t=" + t, "restored");
  if (st.has_video && !st.has_preview) {
    $id("sbsMsg").innerHTML = "<b>Original only.</b><br><br>" +
      "Run a restore to enable the wipe compare.";
  }

  raf = requestAnimationFrame(tick);
}

function close() {
  S.open = false;
  cancelAnimationFrame(raf);
  detach();
  root.classList.add("hidden");
}

// ── render loop ────────────────────────────────────────────
function tick() {
  if (!S.open) return;
  raf = requestAnimationFrame(tick);

  if (S.hasA && S.hasB && !S.seeking) {
    // Keep A (original) locked to B (restored). Hard-seeking a heavy source
    // every frame flushes its decoder and freezes it — so hard-seek ONLY on
    // big jumps (loop wrap, user seek); converge small drift by nudging A's
    // playbackRate a few percent, the way real players sync.
    const want = videoB.currentTime + S.offset;
    const drift = videoA.currentTime - want;   // >0: A is ahead
    if (Math.abs(drift) > 0.5) {
      videoA.currentTime = want;
      videoA.playbackRate = S.rate;
    } else if (!videoB.paused) {
      const nudge = Math.max(-0.08, Math.min(0.08, -drift * 0.5));
      videoA.playbackRate = S.rate * (1 + nudge);
    } else {
      videoA.playbackRate = S.rate;
      if (Math.abs(drift) > 0.04) videoA.currentTime = want;  // paused: frame-align
    }
    if (videoB.paused !== videoA.paused)
      videoB.paused ? videoA.pause() : videoA.play().catch(() => {});
  }

  const m = S.master;
  if (m) {
    $id("sbsPlay").textContent = m.paused ? "Play" : "Pause";
    const f = appFps();
    $id("sbsTime").textContent = fmt(m.currentTime) + " / " + fmt(m.duration)
      + (f ? "  f" + Math.round(m.currentTime * f) : "");
    if (!S.seeking && isFinite(m.duration) && m.duration > 0)
      $id("sbsSeek").value = Math.round((m.currentTime / m.duration) * 1000);
  }

  uniforms.uWipeX.value = (S.view === "wipe" && S.hasA && S.hasB)
    ? (S.wipeFrac * root.clientWidth / stage.clientWidth) * renderer.domElement.width
    : 1e9;

  camera.rotation.order = "YXZ";
  camera.rotation.y = S.yaw;
  camera.rotation.x = S.pitch;
  renderer.render(scene, camera);
}

// ── entry point ────────────────────────────────────────────
const btn = $id("vcSbsBtn");
if (btn) btn.addEventListener("click", open);
window.SBSViewer = { open, close };   // console / future integration hook