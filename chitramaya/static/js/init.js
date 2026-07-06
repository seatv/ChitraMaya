// ── Boot Sequence ─────────────────────────────────────────

console.log('[ChitraMaya] UI initializing...');

// ── Reusable Confirm Dialog ──────────────────────────────
function showConfirm(title, message) {
  return new Promise((resolve) => {
    confirmTitle.textContent = title;
    confirmMessage.textContent = message;
    confirmModal.classList.remove('hidden');

    function cleanup() {
      confirmModal.classList.add('hidden');
      confirmYes.removeEventListener('click', onYes);
      confirmNo.removeEventListener('click', onNo);
    }
    function onYes() { cleanup(); resolve(true); }
    function onNo()  { cleanup(); resolve(false); }

    confirmYes.addEventListener('click', onYes);
    confirmNo.addEventListener('click', onNo);
  });
}

// ── Output Folder ────────────────────────────────────────

outputBtn.addEventListener('click', async () => {
  const path = await selectFolder();
  if (!path) return;
  outputPath.value = path;
  state.outputDir = path;
  await apiPost('/api/set-output-dir', { path });
});

// Allow paste + Enter on path fields
outputPath.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const path = outputPath.value.trim();
    if (path) {
      state.outputDir = path;
      await apiPost('/api/set-output-dir', { path });
    }
  }
});

tempBtn.addEventListener('click', async () => {
  const path = await selectFolder();
  if (!path) return;
  tempPath.value = path;
  await apiPost('/api/set-temp-dir', { path });
});

tempPath.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const path = tempPath.value.trim();
    if (path) await apiPost('/api/set-temp-dir', { path });
  }
});

// ── Video Load (called from player.js via chitramayaSetVideoFromPath) ───

async function onVideoLoad(path) {
  console.log('[ChitraMaya] Loading video:', path);
  const result = await loadVideo(path);
  console.log('[ChitraMaya] Video result:', result);
  if (!result || result.error) {
    console.error('[ChitraMaya] Video load failed:', result);
    return;
  }

  // Properly clear old source before setting new (Tilester pattern)
  try { player.pause(); } catch {}
  try { player.removeAttribute('src'); player.load(); } catch {}

  // Show video player
  videoDrop.style.display = 'none';
  videoContainer.style.display = 'flex';

  // Cache-bust: browser caches /video URL, so append timestamp to force re-fetch
  player.src = '/video?t=' + Date.now();
  player.style.display = 'block';
  player.load();

  // Show custom video controls
  const vc = document.getElementById('videoControls');
  if (vc) vc.classList.add('video-ready');

  // Set FPS in player module
  if (result.info) {
    setFps(result.info.fps);
  }

  // Enable detect button
  detectBtn.disabled = false;

  console.log('[ChitraMaya] Video loaded:', result.info);
}

// Expose for player.js
window.onVideoLoad = onVideoLoad;

// ── Detected-region thumbnail ────────────────────────────
// Up/down nav removed. updateCarousel now just refreshes the Mosaic
// (Detected) thumbnail from the current detection. Drop 3 rewires the whole
// detect/preview flow into the FOI preview.
function updateCarousel() {
  const total = state.detectedFaces.length;
  const idx = state.currentFaceIdx;
  if (total > 0 && state.detectedFaces[idx]) {
    _setSlotImage('targetImg', state.detectedFaces[idx].crop_b64);
  } else {
    _clearSlotImage('targetImg');
  }
  updateSwappedDisplay(null);
}

function updateSwappedDisplay(assignResult) {
  if (assignResult && assignResult.swapped_thumb_b64) {
    _setSlotImage('swappedImg', assignResult.swapped_thumb_b64);
  }
}

// ── Face Slot Image Helpers ──────────────────────────────
// These properly update face slot images without breaking DOM references.
// The pattern: if the element is a <div> placeholder, replace it with <img>.
// If it's already an <img>, just update src. Always preserve the id.

function _setSlotImage(id, b64) {
  let el = document.getElementById(id);
  if (!el) return;

  if (el.tagName === 'IMG') {
    el.src = 'data:image/jpeg;base64,' + b64;
  } else {
    // Replace placeholder div with img
    const img = document.createElement('img');
    img.id = id;
    img.className = 'face-slot-img';
    img.src = 'data:image/jpeg;base64,' + b64;
    el.replaceWith(img);
  }
}

function _clearSlotImage(id) {
  let el = document.getElementById(id);
  if (!el) return;

  if (el.tagName === 'IMG') {
    // Replace img with placeholder div
    const div = document.createElement('div');
    div.id = id;
    div.className = 'face-slot-img empty';
    div.textContent = '?';
    el.replaceWith(div);
  }
  // Already a div placeholder — nothing to do
}

// ── Central Button State Manager ──────────────────────────
// Called after every state transition to ensure consistent button states.

function _updateButtonStates() {
  const hasVideo = !!state.videoPath;
  const hasDetected = state.detectedFaces.length > 0;
  const hasAssignment = Object.keys(state.assignments).length > 0;
  const hasSwapped = state.hasSwappedImage || false;
  const hasSegment = state.segmentEndTime > state.segmentStartTime;
  const maskOnly = (document.getElementById('ctrlShowMask')?.checked ?? false);
  const inPreview = state.previewMode || false;

  // (10) Detect: armed when video loaded AND not in Preview Mode
  detectBtn.disabled = !hasVideo || inPreview;

  // (5) Magnifier: armed whenever Restored is populated
  zoomBtn.disabled = !hasSwapped;

  // (9) Swap: armed when Swapped Face AND segment selected, OR mask mode with detection + segment
  swapBtn.disabled = !((hasSwapped && hasSegment) || (hasDetected && maskOnly && hasSegment));

  // (7) Swap & Save: armed when there is a Swapped Face (real swap, not mask)
  swapSaveBtn.disabled = !hasAssignment;

  // (8) Preview: only enabled when preview is ready
  if (!state.previewReady) {
    previewBtn.disabled = true;
    previewBtn.classList.remove('highlight');
    if (!inPreview) previewBtn.textContent = 'Preview';
  }
}

// ── Zoom Preview ─────────────────────────────────────────

zoomBtn.addEventListener('click', async () => {
  if (state.detectedFaces.length === 0) return;

  const maskOnly = (document.getElementById('ctrlShowMask')?.checked ?? false);
  if (!maskOnly && state.assignments[state.currentFaceIdx] === undefined) return;

  state.zoomOpen = true;
  zoomOverlay.classList.remove('hidden');

  // Fetch original and swapped/mask 512×512 faces
  const [origResult, swapResult] = await Promise.all([
    apiPost('/api/original-face', { target_idx: state.currentFaceIdx }),
    previewSwap(state.currentFaceIdx),
  ]);

  if (origResult && origResult.image) {
    zoomOriginal.src = 'data:image/jpeg;base64,' + origResult.image;
  }
  if (swapResult && swapResult.image) {
    zoomImage.src = 'data:image/jpeg;base64,' + swapResult.image;
  }
});

zoomClose.addEventListener('click', () => {
  state.zoomOpen = false;
  zoomOverlay.classList.add('hidden');
});

// ── Action Buttons ───────────────────────────────────────

detectBtn.addEventListener('click', async () => {
  detectBtn.disabled = true;
  detectBtn.textContent = 'Detecting...';

  const result = await detectFaces();

  detectBtn.textContent = 'Detect';

  if (result && result.faces && result.faces.length > 0) {
    state.hasSwappedImage = false;
    state.previewReady = false;
    state.previewMode = false;
    _clearSlotImage('swappedImg');
    updateCarousel();
    console.log(`[ChitraMaya] Detected ${result.faces.length} face(s)`);

    // Auto-populate swapped in mask mode
    const maskOn = (document.getElementById('ctrlShowMask')?.checked ?? false);
    if (maskOn) {
      const maskResult = await previewSwap(state.currentFaceIdx);
      if (maskResult && maskResult.image) {
        _setSlotImage('swappedImg', maskResult.image);
        state.hasSwappedImage = true;
      }
    }
  } else {
    console.log('[ChitraMaya] No faces detected');
  }

  _updateButtonStates();
});

swapBtn.addEventListener('click', async () => {
  const maskOnly = (document.getElementById('ctrlShowMask')?.checked ?? false);
  if (!state.videoPath) {
    alert('Load a video first.');
    return;
  }
  if (!maskOnly && Object.keys(state.assignments).length === 0) {
    alert('Assign at least one face first.');
    return;
  }
  if (state.detectedFaces.length === 0) {
    alert('Detect faces first.');
    return;
  }

  // If in preview mode, exit first
  if (state.previewMode) {
    previewBtn.click();
  }

  // Clear previous preview state
  state.previewReady = false;
  previewBtn.disabled = true;
  previewBtn.classList.remove('highlight');
  previewBtn.textContent = 'Preview';

  const params = gatherParams();
  params.encoder = {
    codec: document.getElementById('ctrlCodec').value,
    preset: document.getElementById('ctrlPreset').value,
    qp: parseInt(document.getElementById('ctrlQP').value),
  };

  // Use segment times if set, otherwise ±5 seconds around current position
  let startTime = state.segmentStartTime || 0;
  let endTime = state.segmentEndTime || 0;
  if (endTime <= startTime) {
    const cur = player.currentTime || 0;
    startTime = Math.max(0, cur - 5);
    endTime = cur + 5;
  }

  console.log(`[ChitraMaya] Swap segment: ${startTime.toFixed(2)}s → ${endTime.toFixed(2)}s`);

  const result = await apiPost('/api/swap-segment', {
    params, start_time: startTime, end_time: endTime,
  });
  if (result.error) {
    alert('Failed to start: ' + result.error);
    _updateButtonStates();
    return;
  }

  // Show progress modal
  progressModal.classList.remove('hidden');
  progressTitle.textContent = 'Processing Segment...';
  progressParams.textContent = buildParamsSummary();
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressFps.textContent = '— fps';
  progressEta.textContent = 'ETA: —';
  swapBtn.disabled = true;
  swapSaveBtn.disabled = true;

  const pollInterval = setInterval(async () => {
    const prog = await apiGet('/api/progress');
    if (!prog || prog.error) return;

    const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;
    progressBar.style.width = pct + '%';
    progressPercent.textContent = `${pct}% (${prog.frame}/${prog.total})`;
    progressFps.textContent = `${prog.fps || '—'} fps`;
    progressEta.textContent = `ETA: ${prog.eta || '—'} | ${prog.faces_swapped || 0} swaps`;

    if (prog.status === 'complete') {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Segment Complete';
      progressPercent.textContent = `${prog.frame} frames, ${prog.faces_swapped || 0} swaps`;
      progressBar.style.width = '100%';
      progressFps.textContent = `${prog.fps || '—'} fps`;
      progressEta.textContent = '';
      progressCancel.textContent = 'Close';
      swapBtn.disabled = false;
      swapSaveBtn.disabled = false;

      // Preview will be enabled when user closes the dialog
      state.previewReady = true;

      console.log('[ChitraMaya] Segment complete — Preview ready');
    } else if (prog.status === 'error' || prog.status === 'cancelled') {
      clearInterval(pollInterval);
      if (prog.status === 'cancelled' && prog.frame > 0) {
        // Partial result available — let user preview it
        progressTitle.textContent = `Cancelled — ${prog.frame} frames processed`;
        progressPercent.textContent = `${prog.faces_swapped || 0} swaps`;
        state.previewReady = true;
      } else {
        progressTitle.textContent = prog.status === 'error' ? 'Error' : 'Cancelled';
        progressPercent.textContent = prog.error || 'No frames processed';
      }
      progressBar.style.width = '0%';
      progressCancel.textContent = 'Close';
      swapBtn.disabled = false;
      swapSaveBtn.disabled = false;
      _updateButtonStates();
    }
  }, 500);

  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled')) {
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
      // Enable Preview if swap completed or has partial result
      if ((prog.status === 'complete' || (prog.status === 'cancelled' && prog.frame > 0)) && state.previewReady) {
        previewBtn.disabled = false;
        previewBtn.classList.add('highlight');
      }
    } else {
      await apiPost('/api/cancel');
    }
  };
});

// ── Preview Mode ─────────────────────────────────────────

previewBtn.addEventListener('click', () => {
  if (!state.previewReady) return;

  if (!state.previewMode) {
    // Enter preview mode — switch player to preview video
    state.previewMode = true;
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}

    player.src = '/preview-video?t=' + Date.now();
    player.load();
    player.play().catch(() => {});

    previewBtn.textContent = '◀ Back';
    previewBtn.classList.remove('highlight');
    previewBtn.classList.add('primary');

    console.log('[ChitraMaya] Entered Preview mode');
  } else {
    // Exit preview mode — switch back to original video
    state.previewMode = false;
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}

    player.src = '/video?t=' + Date.now();
    player.load();

    // Re-render segment highlight once video metadata loads
    player.addEventListener('loadedmetadata', () => {
      _updateSegmentVisual();
    }, { once: true });

    previewBtn.textContent = 'Preview';
    previewBtn.classList.remove('primary');
    if (state.previewReady) previewBtn.classList.add('highlight');

    console.log('[ChitraMaya] Exited Preview mode');
  }
});

swapSaveBtn.addEventListener('click', async () => {
  console.log('[ChitraMaya] Swap & Save clicked');
  console.log('[ChitraMaya]   videoPath:', state.videoPath);
  console.log('[ChitraMaya]   assignments:', JSON.stringify(state.assignments));
  console.log('[ChitraMaya]   detectedFaces:', state.detectedFaces.length);

  if (!state.videoPath || Object.keys(state.assignments).length === 0) {
    alert('Load a video and assign at least one face first.');
    return;
  }

  // Exit preview mode if active
  if (state.previewMode) {
    state.previewMode = false;
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}
    player.src = '/video?t=' + Date.now();
    player.load();
    previewBtn.textContent = 'Preview';
    previewBtn.classList.remove('primary');
  }

  const params = gatherParams();

  // Add encoder params
  params.encoder = {
    codec: document.getElementById('ctrlCodec').value,
    preset: document.getElementById('ctrlPreset').value,
    qp: parseInt(document.getElementById('ctrlQP').value),
  };

  console.log('[ChitraMaya] Starting Swap & Save with params:', params);

  // Start processing
  const result = await apiPost('/api/swap-full', { params });
  if (result.error) {
    alert('Failed to start: ' + result.error);
    return;
  }

  // Show progress modal
  progressModal.classList.remove('hidden');
  progressTitle.textContent = 'Processing Video...';
  progressParams.textContent = buildParamsSummary();
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressFps.textContent = '— fps';
  progressEta.textContent = 'ETA: —';

  // Disable action buttons during processing
  swapBtn.disabled = true;
  swapSaveBtn.disabled = true;
  detectBtn.disabled = true;

  // Poll progress
  const pollInterval = setInterval(async () => {
    const prog = await apiGet('/api/progress');
    if (!prog || prog.error) return;

    const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;
    progressBar.style.width = pct + '%';
    progressPercent.textContent = `${pct}% (${prog.frame}/${prog.total})`;
    progressFps.textContent = `${prog.fps || '—'} fps`;
    progressEta.textContent = `ETA: ${prog.eta || '—'} | ${prog.faces_swapped || 0} swaps`;

    if (prog.status === 'complete') {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Complete!';
      progressPercent.textContent = `Done — ${prog.frame} frames, ${prog.faces_swapped || 0} swaps`;
      progressBar.style.width = '100%';
      progressCancel.textContent = 'Close';

      // Re-enable buttons
      swapBtn.disabled = false;
      swapSaveBtn.disabled = false;
      detectBtn.disabled = false;

      console.log('[ChitraMaya] Swap & Save complete:', prog);
    } else if (prog.status === 'error') {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Error';
      progressPercent.textContent = prog.error || 'Processing failed';
      progressCancel.textContent = 'Close';
      swapBtn.disabled = false;
      swapSaveBtn.disabled = false;
      detectBtn.disabled = false;
    } else if (prog.status === 'cancelled') {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Cancelled';
      progressCancel.textContent = 'Close';
      swapBtn.disabled = false;
      swapSaveBtn.disabled = false;
      detectBtn.disabled = false;
    }
  }, 500);

  // Stop button
  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled')) {
      // Close modal
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
    } else {
      // Cancel processing
      await apiPost('/api/cancel');
    }
  };
});

// (Preview button handler is above with the Swap button)

// ── New Project ──────────────────────────────────────────

async function resetProject() {
  const ok = await showConfirm('New Project', 'Reset all state and start a new project?');
  if (!ok) return;

  // Exit preview mode if active
  if (state.previewMode) {
    state.previewMode = false;
  }

  await apiPost('/api/new-project');
  await apiPost('/api/clear-preview');

  // Clear ALL state
  state.videoPath = null;
  state.videoInfo = null;
  state.detectedFaces = [];
  state.assignments = {};
  state.currentFaceIdx = 0;
  state.previewReady = false;
  state.previewMode = false;
  state.zoomOpen = false;
  state.hasSwappedImage = false;
  state.segmentStartTime = 0;
  state.segmentEndTime = 0;
  state.segmentStartFrame = 0;
  state.segmentEndFrame = 0;

  // Clear player
  try { player.pause(); } catch {}
  try { player.removeAttribute('src'); player.load(); } catch {}
  player.style.display = 'none';

  // Hide video controls
  const vc = document.getElementById('videoControls');
  if (vc) vc.classList.remove('video-ready');

  // Reset UI
  videoDrop.style.display = 'flex';
  videoContainer.style.display = 'none';

  // Clear thumbnails
  _clearSlotImage('targetImg');
  _clearSlotImage('swappedImg');

  // Clear zoom
  zoomOverlay.classList.add('hidden');
  state.zoomOpen = false;
  zoomImage.src = '';
  zoomOriginal.src = '';

  // Clear segment (calls clearSegment in player.js)
  if (typeof clearSegment === 'function') clearSegment();

  // Reset preview button
  previewBtn.textContent = 'Preview';
  previewBtn.classList.remove('highlight', 'primary');

  // Reset FPS
  fps = null;
  if (fpsDisplay) fpsDisplay.innerHTML = '<span class="box-value">—</span>';
  if (frameNum) {
    const inner = frameNum.querySelector('.box-value');
    if (inner) inner.textContent = '—';
  }
  if (currentTime) currentTime.textContent = '00:00:00.000';

  // Update all button states
  _updateButtonStates();
}

newBtn.addEventListener('click', resetProject);

// ── Fullscreen ───────────────────────────────────────────

fsBtn.addEventListener('click', () => {
  const api = getPyWebViewApi();
  if (api && typeof api.toggle_fullscreen === 'function') {
    api.toggle_fullscreen();
  } else {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }
});

// ── PyWebView Ready ──────────────────────────────────────
window.__chitramayaOnPyWebViewReady = () => {
  console.log('[ChitraMaya] PyWebView ready');
};

if (window.__chitramayaPyWebViewReady) {
  window.__chitramayaOnPyWebViewReady();
}

console.log('[ChitraMaya] UI ready');

// ── Startup: Load saved config or server defaults ────────
(async () => {
  try {
    let cfg = await apiGet('/api/load-config');
    if (!cfg || cfg.error || Object.keys(cfg).length === 0) {
      // No saved config — load defaults from models.py
      cfg = await apiGet('/api/default-config');
      console.log('[ChitraMaya] No saved config — loaded defaults from models.py');
    } else {
      console.log('[ChitraMaya] Config loaded from file');
    }
    if (cfg && !cfg.error) {
      applyConfig(cfg);
      if (cfg.outputDir) {
        state.outputDir = cfg.outputDir;
        // Push to the server so self.output_dir matches the loaded config
        // even before the first Restore. The job payload (gatherMosaicParams)
        // is the authoritative source, but this keeps server state consistent.
        await apiPost('/api/set-output-dir', { path: cfg.outputDir });
      }
      if (cfg.tempDir) {
        await apiPost('/api/set-temp-dir', { path: cfg.tempDir });
      }
    }
  } catch (e) {
    console.warn('[ChitraMaya] Failed to load config:', e);
  }
})();

// ── Config Menu (⚙️) — Save / Load / Reset ───────────────
configBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  configMenu.classList.toggle('hidden');
});

// Close menu on click outside
document.addEventListener('click', () => configMenu.classList.add('hidden'));

cfgSave.addEventListener('click', saveConfig);
cfgLoad.addEventListener('click', loadConfig);
cfgReset.addEventListener('click', resetDefaults);

async function saveConfig() {
  const cfg = gatherFullConfig();
  const result = await apiPost('/api/save-config', cfg);
  if (result && !result.error) {
    console.log('[ChitraMaya] Config saved');
    alert('Settings saved to ChitraMaya-config.json');
  } else {
    alert('Failed to save: ' + (result.error || 'unknown error'));
  }
}

async function loadConfig() {
  const cfg = await apiGet('/api/load-config');
  if (cfg && !cfg.error && Object.keys(cfg).length > 0) {
    applyConfig(cfg);
    if (cfg.outputDir) {
      state.outputDir = cfg.outputDir;
      await apiPost('/api/set-output-dir', { path: cfg.outputDir });
    }
    if (cfg.tempDir) {
      await apiPost('/api/set-temp-dir', { path: cfg.tempDir });
    }
    console.log('[ChitraMaya] Config loaded');
  } else {
    alert('No saved config found. Save your settings first.');
  }
}

async function resetDefaults() {
  const ok = await showConfirm('Reset Defaults', 'Reset all settings to their default values?');
  if (!ok) return;
  const defaults = await apiGet('/api/default-config');
  if (defaults && !defaults.error) {
    applyConfig(defaults);
    console.log('[ChitraMaya] Settings reset to defaults from models.py');
  } else {
    console.error('[ChitraMaya] Failed to fetch defaults');
  }
}
