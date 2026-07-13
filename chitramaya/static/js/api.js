// ── Server API ────────────────────────────────────────────

async function apiPost(endpoint, data = {}) {
  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const json = await resp.json();
    if (json.error) {
      console.error(`API error (${endpoint}):`, json.error);
    }
    return json;
  } catch (err) {
    console.error(`API fetch failed (${endpoint}):`, err);
    return { error: err.message };
  }
}

async function apiGet(endpoint) {
  try {
    const resp = await fetch(endpoint, { cache: 'no-store' });
    return await resp.json();
  } catch (err) {
    console.error(`API fetch failed (${endpoint}):`, err);
    return { error: err.message };
  }
}

// ── PyWebView Bridge ─────────────────────────────────────

function getPyWebViewApi() {
  return (typeof window.pywebview !== 'undefined' && window.pywebview) ? window.pywebview.api : null;
}

async function selectFolder() {
  const api = getPyWebViewApi();
  if (api && typeof api.select_folder === 'function') {
    return await api.select_folder();
  }
  // Fallback: prompt
  return prompt('Enter folder path:');
}

async function selectVideo() {
  const api = getPyWebViewApi();
  if (api && typeof api.select_video === 'function') {
    return await api.select_video();
  }
  return prompt('Enter video path:');
}

// ── High-Level API Actions ───────────────────────────────

async function loadVideo(path) {
  const result = await apiPost('/api/load-video', { path });
  if (result.error) return null;

  state.videoPath = path;
  state.videoInfo = result.info;
  state.currentFrame = 0;
  state.detectedFaces = [];
  state.currentFaceIdx = 0;
  state.assignments = {};

  // A new video invalidates any segment marks and preview state from the
  // previous one. Without this reset, a segment marked on video A survives
  // into video B: Restore silently submits A's time range against B, the
  // seekbar paints A's segment on B's timeline, and Preview stays enabled
  // pointing at A's preview file. This lives here (not in onVideoLoad) so
  // every entry path — drag-drop, file picker, browser prompt — is covered.
  state.previewMode = false;
  if (typeof clearSegment === 'function') {
    // clearSegment() also resets the armed start marker, repaints the
    // seekbar, deletes the server-side temp preview, and refreshes button
    // states (with previewMode already false, the Preview button leaves
    // "◀ Back" mode correctly).
    try { clearSegment(); } catch (e) { console.warn('[ChitraMaya] segment reset on load failed:', e); }
  } else {
    state.segmentStartTime = 0;
    state.segmentEndTime = 0;
    state.segmentStartFrame = 0;
    state.segmentEndFrame = 0;
    state.previewReady = false;
  }

  return result;
}

async function loadFacesDir(path) {
  const result = await apiPost('/api/load-faces-dir', { path });
  if (result.error) return null;

  state.facesDir = path;
  state.sourceFaces = result.faces || [];
  return result;
}

async function detectFaces() {
  const time = player.currentTime || 0;
  const result = await apiPost('/api/detect-faces', { time: time });
  if (result.error) return null;

  state.detectedFaces = result.faces || [];
  state.currentFaceIdx = 0;
  state.assignments = {};
  return result;
}

async function assignFace(targetIdx, sourceIdx) {
  const result = await apiPost('/api/assign-face', {
    target_idx: targetIdx,
    source_idx: sourceIdx,
  });
  if (!result.error) {
    state.assignments[targetIdx] = sourceIdx;
  }
  return result;
}

async function previewSwap(targetIdx) {
  const params = gatherParams();
  return await apiPost('/api/preview-swap', {
    target_idx: targetIdx,
    params: params,
  });
}
