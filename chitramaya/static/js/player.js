// ── ChitraMaya Player — ported from Tilester ───────────────
// Video playback, transport, frame stepping, fullscreen, edge-reveal.

// ── pywebview Integration ─────────────────────────────────
let isPyWebView = false;
let _pvwIsFullscreen = false;

window.__chitramayaPyWebViewReady = false;

function _detectPyWebView() {
  return (typeof window.pywebview !== 'undefined') && window.pywebview && window.pywebview.api;
}

function _markPyWebViewReady() {
  isPyWebView = true;
  window.__chitramayaPyWebViewReady = true;
  try {
    if (typeof window.__chitramayaOnPyWebViewReady === 'function') {
      window.__chitramayaOnPyWebViewReady();
    }
  } catch (e) {
    console.warn('__chitramayaOnPyWebViewReady failed:', e);
  }
}

if (_detectPyWebView()) {
  _markPyWebViewReady();
  console.log('[ChitraMaya] Running in native pywebview window');
} else {
  console.log('[ChitraMaya] Running in browser mode (pywebview not ready yet)');
}

window.addEventListener('pywebviewready', () => {
  _markPyWebViewReady();
  console.log('[ChitraMaya] pywebviewready: native window APIs available');
});

// ── Helper Functions ──────────────────────────────────────
function fmtTime(s) {
  const total = Math.floor(s * 1000);
  const ms = total % 1000;
  const sec = Math.floor(total / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const ss = sec % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

function fmtDuration(s) {
  const sec = Math.round(s);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const r = sec % 60;
  if (m < 60) return `${m}m ${r}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m ${r}s`;
}

function fmtFrameTag(f) {
  if (f === null || f === undefined || isNaN(f)) return '—';
  return String(f);
}

function frameAtTime(t) {
  if (!fps || typeof t !== 'number' || isNaN(t)) return null;
  return Math.round(t * fps);
}

function fmtFpsPair() {
  if (!fps || !isFinite(fps) || fps <= 0) return '—';
  return (Math.abs(fps - Math.round(fps)) < 0.0005) ? String(Math.round(fps)) : fps.toFixed(2);
}

function normalizeFullPath(p) {
  if (!p) return '';
  return String(p).replace(/^["']|["']$/g, '');
}

function basenameFromPath(p) {
  const s = normalizeFullPath(p);
  const parts = s.split(/[\\/]/);
  return parts[parts.length - 1] || s;
}

// ── FPS Detection ─────────────────────────────────────────
function tryDetectFpsFromPlayer() {
  try {
    if (player && typeof player.captureStream === 'function') {
      const stream = player.captureStream();
      const tracks = stream.getVideoTracks ? stream.getVideoTracks() : [];
      if (tracks && tracks.length) {
        const settings = tracks[0].getSettings ? tracks[0].getSettings() : {};
        const fr = settings && settings.frameRate ? Number(settings.frameRate) : null;
        try { tracks[0].stop(); } catch {}
        if (fr && isFinite(fr) && fr > 0) {
          if (!fps) fps = fr;
          if (fpsDisplay) fpsDisplay.innerHTML = `<span class="box-value">${fmtFpsPair()}</span>`;
          return fps;
        }
      }
    }
  } catch (e) { /* ignore */ }
  return fps;
}

// Set FPS from server-provided metadata
function setFps(effective) {
  if (effective && isFinite(effective) && effective > 0) fps = effective;
  if (fpsDisplay) fpsDisplay.innerHTML = `<span class="box-value">${fmtFpsPair()}</span>`;
}

// ── Go-to Frame / Time ────────────────────────────────────
function parseTime(str) {
  str = str.trim();
  if (!str) return NaN;
  if (/^\d+(\.\d+)?$/.test(str)) return parseFloat(str);
  const parts = str.split(':');
  if (parts.length < 2 || parts.length > 3) return NaN;
  let h = 0, m = 0, s = 0;
  if (parts.length === 3) {
    h = parseInt(parts[0], 10); m = parseInt(parts[1], 10); s = parseFloat(parts[2]);
  } else {
    m = parseInt(parts[0], 10); s = parseFloat(parts[1]);
  }
  if (isNaN(h) || isNaN(m) || isNaN(s)) return NaN;
  return h * 3600 + m * 60 + s;
}

function seekToFrame() {
  const f = parseInt(gotoFrame.value, 10);
  if (!fps || isNaN(f) || f < 0) return;
  player.currentTime = f / fps;
  if (!player.paused) player.pause();
}

function seekToTime() {
  const t = parseTime(gotoTime.value);
  if (isNaN(t) || t < 0) return;
  player.currentTime = t;
  if (!player.paused) player.pause();
}

gotoFrame.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); seekToFrame(); gotoFrame.blur(); }
});
gotoTime.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); seekToTime(); gotoTime.blur(); }
});
if (gotoFrameBtn) {
  gotoFrameBtn.addEventListener('click', (e) => { e.preventDefault(); seekToFrame(); try { gotoFrame.blur(); } catch {} });
}
if (gotoTimeBtn) {
  gotoTimeBtn.addEventListener('click', (e) => { e.preventDefault(); seekToTime(); try { gotoTime.blur(); } catch {} });
}

// ── Video Path Setting (from pywebview native drop) ───────
function chitramayaSetVideoFromPath(fullPath) {
  const p = normalizeFullPath(fullPath);
  if (!p) return;
  if (!p.includes('\\') && !p.includes('/')) {
    console.warn('[ChitraMaya] chitramayaSetVideoFromPath: rejected non-path value:', p);
    return;
  }

  // Call the API to load video metadata and set up server state
  // This is async but we fire-and-forget here; onVideoLoad handles the result
  if (typeof onVideoLoad === 'function') {
    onVideoLoad(p);
  }
}

// Expose for Python evaluate_js (native drag/drop)
window.chitramayaSetVideoFromPath = chitramayaSetVideoFromPath;
window.chitramayaSetVideoPath = chitramayaSetVideoFromPath;

// Allow Python-side to surface errors in the UI
window.chitramayaShowError = (title, message) => {
  console.error('[ChitraMaya Error]', title, message);
  // TODO: show error dialog modal
};

// ── Fullscreen ────────────────────────────────────────────
async function toggleFullscreenApp() {
  try {
    if (_detectPyWebView() && window.pywebview.api && window.pywebview.api.toggle_fullscreen) {
      const r = await window.pywebview.api.toggle_fullscreen();
      if (r && r.ok === false) {
        console.warn('[ChitraMaya] Native fullscreen failed:', r.error || r);
        return;
      }
      _pvwIsFullscreen = !_pvwIsFullscreen;
      document.body.classList.toggle('isAppFullscreen', _pvwIsFullscreen);
      if (!_pvwIsFullscreen) {
        document.body.classList.remove('navOpen', 'headerOpen', 'footerOpen', 'controlsOpen');
      }
      return;
    }
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await appRoot.requestFullscreen();
    }
  } catch (err) {
    console.warn('[ChitraMaya] Fullscreen failed:', err);
  }
}

function updateFullscreenClasses() {
  const fsEl = document.fullscreenElement;
  const isAppFs = (fsEl === appRoot) || _pvwIsFullscreen;
  document.body.classList.toggle('isAppFullscreen', isAppFs);
  if (!isAppFs) {
    document.body.classList.remove('navOpen', 'headerOpen', 'footerOpen', 'controlsOpen');
  }
}

document.addEventListener('fullscreenchange', updateFullscreenClasses);
updateFullscreenClasses();

fsBtn.addEventListener('click', toggleFullscreenApp);

// Click video area to toggle play/pause
videoContainer.addEventListener('click', (e) => {
  if (e.target === player || e.target === videoContainer) {
    if (!player.src && !player.currentSrc) return; // no video loaded
    player.paused ? player.play().catch(() => {}) : player.pause();
    player.focus();
  }
});

// Double-click video area to toggle app fullscreen
videoContainer.addEventListener('dblclick', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON')) return;
  toggleFullscreenApp();
});

// ── Edge-Reveal in Fullscreen ─────────────────────────────
const OPEN_ZONE_PX   = 24;
const CLOSE_LEFT_PX  = 480;
const TOP_EDGE_PX    = 24;
const BOTTOM_EDGE_PX = 2;

window.addEventListener('pointermove', (e) => {
  if (!document.body.classList.contains('isAppFullscreen')) return;

  // Right edge → controls panel
  const nearRight = e.clientX >= (window.innerWidth - OPEN_ZONE_PX);
  if (nearRight) document.body.classList.add('navOpen');
  else if (e.clientX <= (window.innerWidth - CLOSE_LEFT_PX)) document.body.classList.remove('navOpen');

  // Top edge → header
  const nearTop = e.clientY <= TOP_EDGE_PX;
  if (nearTop) document.body.classList.add('headerOpen');
  else if (e.clientY > 80) document.body.classList.remove('headerOpen');

  // Bottom edge → footer + controls
  const CONTROLS_ZONE_PX = 80;
  const nearBottom = e.clientY >= (window.innerHeight - CONTROLS_ZONE_PX);
  const atBottom   = e.clientY >= (window.innerHeight - BOTTOM_EDGE_PX);

  if (nearBottom) document.body.classList.add('controlsOpen');
  else if (e.clientY < (window.innerHeight - 120)) document.body.classList.remove('controlsOpen');

  if (atBottom) document.body.classList.add('footerOpen');
  else if (e.clientY < (window.innerHeight - 80)) document.body.classList.remove('footerOpen');
});

// ── Pin Toggle ────────────────────────────────────────────
let isPinned = false;

pinBtn.addEventListener('click', () => {
  isPinned = !isPinned;
  document.body.classList.toggle('pinned', isPinned);
  pinBtn.classList.toggle('pinned', isPinned);
  pinBtn.textContent = isPinned ? '📌 Unpin' : '📌 Pin';
});

// ── Copy Time to Clipboard ────────────────────────────────
if (copyTimeBtn) {
  copyTimeBtn.addEventListener('click', async () => {
    const timeText = currentTime.textContent;
    try {
      await navigator.clipboard.writeText(timeText);
      copyTimeBtn.classList.add('copied');
      copyTimeBtn.textContent = '✓';
      setTimeout(() => {
        copyTimeBtn.classList.remove('copied');
        copyTimeBtn.textContent = '📋';
      }, 1000);
    } catch (err) {
      console.error('[ChitraMaya] Copy failed:', err);
    }
  });
}

// ── Segment Mark ──────────────────────────────────────────
let segStart = null; // start time in seconds, or null
const seekbarSegment = document.getElementById('seekbarSegment');

function _updateSegmentVisual() {
  if (!seekbarSegment) return;

  if (!player.duration) {
    // No video loaded — hide segment
    seekbarSegment.className = 'seekbar-segment';
    return;
  }

  if (segStart !== null && state.segmentEndTime <= 0) {
    // Armed — show start marker
    const startPct = (segStart / player.duration) * 100;
    seekbarSegment.style.left = startPct + '%';
    seekbarSegment.style.width = '2px';
    seekbarSegment.className = 'seekbar-segment visible armed';
  } else if (state.segmentStartTime >= 0 && state.segmentEndTime > state.segmentStartTime) {
    // Complete segment — show colored range
    const startPct = (state.segmentStartTime / player.duration) * 100;
    const endPct = (state.segmentEndTime / player.duration) * 100;
    seekbarSegment.style.left = startPct + '%';
    seekbarSegment.style.width = (endPct - startPct) + '%';
    seekbarSegment.className = 'seekbar-segment visible';
  } else {
    seekbarSegment.className = 'seekbar-segment';
  }
}

function _updateSegMarkBtn() {
  if (!segMarkBtn) return;
  const hasSegment = state.segmentEndTime > state.segmentStartTime;
  if (hasSegment) {
    segMarkBtn.textContent = '✕';
    segMarkBtn.title = 'Clear segment selection';
    segMarkBtn.classList.add('has-segment');
    segMarkBtn.classList.remove('armed');
  } else if (segStart !== null) {
    segMarkBtn.textContent = '▴';
    segMarkBtn.title = `Start: ${fmtTime(segStart)} — click again to set end, right-click to cancel`;
    segMarkBtn.classList.add('armed');
    segMarkBtn.classList.remove('has-segment');
  } else {
    segMarkBtn.textContent = '▴';
    segMarkBtn.title = 'Click to set segment start';
    segMarkBtn.classList.remove('armed', 'has-segment');
  }
}

function clearSegment() {
  segStart = null;
  state.segmentStartTime = 0;
  state.segmentEndTime = 0;
  state.segmentStartFrame = 0;
  state.segmentEndFrame = 0;
  state.previewReady = false;

  // Delete preview file on server
  apiPost('/api/clear-preview');

  _updateSegmentVisual();
  _updateSegMarkBtn();
  if (typeof _updateButtonStates === 'function') _updateButtonStates();

  // Reset preview button
  if (previewBtn) {
    previewBtn.disabled = true;
    previewBtn.classList.remove('highlight');
    previewBtn.textContent = 'Preview';
  }

  console.log('[ChitraMaya] Segment cleared');
}

if (segMarkBtn) {
  segMarkBtn.addEventListener('click', () => {
    const t = player.currentTime;
    const hasSegment = state.segmentEndTime > state.segmentStartTime;

    if (hasSegment) {
      // Segment exists — clear it
      clearSegment();
    } else if (segStart === null) {
      // No segment, not armed — set start
      segStart = t;
      _updateSegMarkBtn();
      _updateSegmentVisual();
      console.log(`[ChitraMaya] Segment start: ${fmtTime(t)}`);
    } else {
      // Armed — set end
      const start = Math.min(segStart, t);
      const end = Math.max(segStart, t);
      segStart = null;

      state.segmentStartTime = start;
      state.segmentEndTime = end;
      state.segmentStartFrame = fps ? Math.round(start * fps) : 0;
      state.segmentEndFrame = fps ? Math.round(end * fps) : 0;

      _updateSegMarkBtn();
      _updateSegmentVisual();
      if (typeof _updateButtonStates === 'function') _updateButtonStates();
      console.log(`[ChitraMaya] Segment: ${fmtTime(start)} → ${fmtTime(end)}`);
    }
  });

  // Right click: cancel arm or clear segment
  segMarkBtn.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (segStart !== null) {
      // Just cancel arm, don't clear segment
      segStart = null;
      _updateSegMarkBtn();
      _updateSegmentVisual();
    } else {
      clearSegment();
    }
  });
}

window.clearSegment = clearSegment;

// ── Playback Updates ──────────────────────────────────────
player.addEventListener('timeupdate', () => {
  currentTime.textContent = fmtTime(player.currentTime);

  if (!fps) tryDetectFpsFromPlayer();

  if (fps) {
    const f = frameAtTime(player.currentTime);
    const v = fmtFrameTag(f);
    if (frameNum) {
      const inner = frameNum.querySelector('.box-value');
      if (inner) inner.textContent = v;
      else frameNum.textContent = v;
    }
  }

  // Update state for other modules
  state.currentFrame = fps ? frameAtTime(player.currentTime) : 0;

  // Update segment highlight on seekbar
  _updateSegmentVisual();
});

// ── Spacebar: capture-phase so it fires before native <video controls> ──
document.addEventListener('keydown', (e) => {
  if (e.key !== ' ') return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  e.preventDefault();
  e.stopPropagation();
  if (!player.src && !player.currentSrc) return;
  player.paused ? player.play().catch(() => {}) : player.pause();
}, true);

// ── Keyboard Shortcuts ────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const skipBack = parseFloat(skipBackward.value) || 5;
  const skipFwd  = parseFloat(skipForward.value)  || 5;

  switch (e.key) {
    case 'f': case 'F':
      e.preventDefault();
      toggleFullscreenApp();
      break;
    case 'p': case 'P':
      e.preventDefault();
      pinBtn.click();
      break;
    case 'm': case 'M':
      e.preventDefault();
      player.muted = !player.muted;
      break;
    case 'ArrowLeft':
      e.preventDefault();
      player.currentTime = Math.max(0, player.currentTime - skipBack);
      break;
    case 'ArrowRight':
      e.preventDefault();
      player.currentTime = Math.min(player.duration || 0, player.currentTime + skipFwd);
      break;
    case ',':
      // Frame step backward
      e.preventDefault();
      if (!player.paused) player.pause();
      if (fps) player.currentTime = Math.max(0, player.currentTime - 1 / fps);
      break;
    case '.':
      // Frame step forward
      e.preventDefault();
      if (!player.paused) player.pause();
      if (fps) player.currentTime = Math.min(player.duration || 0, player.currentTime + 1 / fps);
      break;
  }
});

// ── Drop Zone Click ───────────────────────────────────────
videoDrop.addEventListener('click', () => {
  if (_detectPyWebView() && window.pywebview.api && typeof window.pywebview.api.select_video === 'function') {
    window.pywebview.api.select_video().then(path => {
      if (path) chitramayaSetVideoFromPath(path);
    }).catch(err => { console.error('[ChitraMaya] Video picker error:', err); });
  } else {
    // Browser fallback — prompt for path
    const path = prompt('Enter video path:');
    if (path) chitramayaSetVideoFromPath(path);
  }
});

// Keep dragover for cursor feedback (actual drop handled by pywebview DOM handler)
centerArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

// ── Custom Video Controls ─────────────────────────────────
const vcStartBtn  = document.getElementById('vcStartBtn');
const vcPlayBtn   = document.getElementById('vcPlayBtn');
const vcEndBtn    = document.getElementById('vcEndBtn');
const vcVolumeBtn = document.getElementById('vcVolumeBtn');
const vcFsBtn     = document.getElementById('vcFsBtn');
const seekbar     = document.getElementById('seekbar');
const seekbarNeedle = document.getElementById('seekbarNeedle');

// Seek to start/end
if (vcStartBtn) {
  vcStartBtn.addEventListener('click', () => {
    if (!player.src && !player.currentSrc) return;
    player.currentTime = 0;
    if (!player.paused) player.pause();
  });
}
if (vcEndBtn) {
  vcEndBtn.addEventListener('click', () => {
    if (!player.src && !player.currentSrc) return;
    if (player.duration) player.currentTime = player.duration - 0.1;
    if (!player.paused) player.pause();
  });
}

const _volSteps = [1, 0.75, 0.5, 0.25, 0];
const _volIcons = ['🔊', '🔉', '🔉', '🔈', '🔇'];

function _updateVcPlayBtn() {
  if (vcPlayBtn) vcPlayBtn.textContent = player.paused ? '▶' : '⏸';
}
function _updateVcVolumeBtn() {
  if (!vcVolumeBtn) return;
  if (player.muted || player.volume === 0) { vcVolumeBtn.textContent = '🔇'; return; }
  const idx = _volSteps.findIndex(v => player.volume >= v - 0.01);
  vcVolumeBtn.textContent = _volIcons[idx >= 0 ? idx : 0];
}
function _updateVcFsBtn() {
  if (vcFsBtn) vcFsBtn.textContent = document.body.classList.contains('isAppFullscreen') ? '✕' : '⛶';
}

// Play/pause button
if (vcPlayBtn) {
  vcPlayBtn.addEventListener('click', () => {
    if (!player.src && !player.currentSrc) return;
    player.paused ? player.play().catch(() => {}) : player.pause();
  });
}
player.addEventListener('play',  _updateVcPlayBtn);
player.addEventListener('pause', _updateVcPlayBtn);

// Volume button — cycle through volume levels
if (vcVolumeBtn) {
  vcVolumeBtn.addEventListener('click', () => {
    if (player.muted) {
      player.muted  = false;
      player.volume = _volSteps[0];
    } else {
      const cur  = player.volume;
      const idx  = _volSteps.findIndex(v => cur >= v - 0.01);
      const next = (idx + 1) % _volSteps.length;
      if (_volSteps[next] === 0) { player.muted = true; }
      else { player.muted = false; player.volume = _volSteps[next]; }
    }
  });
}
player.addEventListener('volumechange', _updateVcVolumeBtn);

// Fullscreen button
if (vcFsBtn) vcFsBtn.addEventListener('click', toggleFullscreenApp);

// Watch for fullscreen class changes to update button icon
new MutationObserver(_updateVcFsBtn)
  .observe(document.body, { attributes: true, attributeFilter: ['class'] });

// Seekbar — click to seek, update needle on timeupdate
if (seekbar) {
  seekbar.addEventListener('click', (e) => {
    if (!player.duration) return;
    const rect = seekbar.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    player.currentTime = pct * player.duration;
  });
}

function _updateSeekbarNeedle() {
  if (!seekbarNeedle || !player.duration) return;
  const pct = (player.currentTime / player.duration) * 100;
  seekbarNeedle.style.left = `${pct}%`;
}

player.addEventListener('timeupdate', _updateSeekbarNeedle);

console.log('[ChitraMaya] Player module loaded');
