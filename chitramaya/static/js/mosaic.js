// ── Restoration tab orchestration ─────────────────────────
// B2: tab toggle, dial formatting, session-status banner, and
// extension of the unified config system to persist mosaic controls.
// B3 will add gatherMosaicParams() + API calls + action button wiring.

// ── Tab toggle ────────────────────────────────────────────
// Driven by a single source of truth on the body element via
// `data-active-tab` so CSS can use attribute selectors if needed
// without growing the JS surface.

function setActiveTab(name) {
  // name: "face-swap" | "restoration"
  document.body.dataset.activeTab = name;

  // Tab strip visual state
  document.querySelectorAll('.ctrl-tab').forEach(btn => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });

  // Tab panel visibility
  document.querySelectorAll('[data-tab-panel]').forEach(panel => {
    panel.hidden = panel.dataset.tabPanel !== name;
  });

  // Footer button visibility (encoder remains visible — shared)
  document.querySelectorAll('[data-mode]').forEach(btn => {
    btn.hidden = btn.dataset.mode !== name;
  });
}

document.querySelectorAll('.ctrl-tab').forEach(btn => {
  btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
});

// Initialize — face-swap is the default per the HTML markup, but call
// setActiveTab so the body attribute and footer-button visibility are
// in a consistent state regardless of which tab the HTML has marked.
setActiveTab('face-swap');


// ── Mosaic dial value formatting ──────────────────────────
// Per-control overrides on top of the generic handler in params.js.

(function attachMosaicDialFormatting() {
  // Score slider: 5-100 displays as 0.05-1.00
  const score = document.getElementById('ctrlMosaicDetScore');
  const scoreVal = document.getElementById('valMosaicDetScore');
  if (score && scoreVal) {
    const fmt = () => { scoreVal.textContent = (parseInt(score.value) / 100).toFixed(2); };
    score.addEventListener('input', fmt);
    fmt();
  }

  // Blend frames: -1 sentinel displays as "auto"
  const blend = document.getElementById('ctrlMosaicBlend');
  const blendVal = document.getElementById('valMosaicBlend');
  if (blend && blendVal) {
    const fmt = () => {
      const v = parseInt(blend.value);
      blendVal.textContent = v < 0 ? 'auto' : String(v);
    };
    blend.addEventListener('input', fmt);
    fmt();
  }
})();


// ── Exclude mosaic controls from face-swap live preview ───
// params.js attaches requestLivePreview to every dial/select/checkbox by
// default. Mosaic controls shouldn't trigger swap preview, so unbind them.
//
// We can't easily unbind, but we can short-circuit: requestLivePreview
// already checks `state.assignments[state.currentFaceIdx] === undefined`
// and returns early. As long as the user isn't in a swap context, mosaic
// dial changes will silently no-op. No explicit unbind needed here — leaving
// this comment as a tripwire for future maintainers who might wonder.


// ── Extend unified config system with mosaic controls ─────
// CONFIG_CONTROLS is defined in params.js; extending here keeps mosaic
// controls auto-persisted via Save Settings / Load Settings without
// duplicating the save/load infrastructure.

const MOSAIC_CONFIG_CONTROLS = [
  'ctrlMosaicDetModel', 'ctrlMosaicDetScore', 'ctrlMosaicDetBatch',
  'ctrlMosaicDetectOnly',
  'ctrlMosaicRestModel', 'ctrlMosaicMaxClip', 'ctrlMosaicOverlap',
  'ctrlMosaicCrossfade', 'ctrlMosaicBlend', 'ctrlMosaicDenoise',
  'ctrlMosaicColorMatch', 'ctrlMosaicFp16', 'ctrlMosaicCompileTrt',
];

if (typeof CONFIG_CONTROLS !== 'undefined') {
  CONFIG_CONTROLS.push(...MOSAIC_CONFIG_CONTROLS);
}


// ── Session-status check on startup ───────────────────────
// If another ChitraMaya job is in progress (CLI batch typically), show a
// non-blocking banner so the user knows UI processing will conflict.

async function checkSessionStatus() {
  try {
    const status = await apiGet('/api/session-status');
    if (status && status.running && !status.is_us) {
      showSessionBanner(status);
    }
  } catch (err) {
    console.warn('session-status check failed:', err);
  }
}

function showSessionBanner(status) {
  // Dismiss any prior banner
  const prior = document.getElementById('sessionBanner');
  if (prior) prior.remove();

  const ageStr = status.age_sec > 60
    ? `${Math.round(status.age_sec / 60)}m`
    : `${Math.round(status.age_sec)}s`;
  const paths = (status.paths || []).join(', ');
  const pathStr = paths ? ` — ${paths}` : '';

  const banner = document.createElement('div');
  banner.id = 'sessionBanner';
  banner.className = 'session-banner';
  banner.innerHTML = `
    <span class="session-banner-icon">⚠</span>
    <span class="session-banner-text">
      Another ChitraMaya job is running (PID ${status.pid}, ${status.mode}, ${ageStr}${pathStr}).
      Starting a UI job will be blocked while it runs.
    </span>
    <button class="session-banner-close" title="Dismiss">×</button>
  `;
  banner.querySelector('.session-banner-close').addEventListener('click', () => {
    banner.remove();
  });

  // Slot in between the header and the main content
  const main = document.querySelector('.main-content');
  if (main && main.parentElement) {
    main.parentElement.insertBefore(banner, main);
  } else {
    document.body.insertBefore(banner, document.body.firstChild);
  }
}

// Check on load and once a minute thereafter — cheap GET, no auth, helps
// catch the case where the CLI batch ends and the user is freed to start
// UI processing.
checkSessionStatus();
setInterval(checkSessionStatus, 60_000);


// ═════════════════════════════════════════════════════════════════════
// B3 — Wiring
// Below this line: model dropdown population, params gathering, action
// button handlers. Pattern mirrors swap-segment / swap-full in init.js.
// state.previewReady is shared with face-swap — whichever job ran most
// recently sets it, and player.js's clearSegment() resets it on segment
// clear. Both modes' Preview buttons key off the same flag.
// ═════════════════════════════════════════════════════════════════════


// ── Model dropdown population ─────────────────────────────
// Calls /api/list-mosaic-models on init. Detection (.pt) and restoration
// (.pth) models live in ./models/ — scanned server-side.

async function populateMosaicModelDropdowns() {
  const data = await apiGet('/api/list-mosaic-models');
  if (!data || data.error) {
    console.warn('list-mosaic-models failed', data && data.error);
    return;
  }

  const detSel = document.getElementById('ctrlMosaicDetModel');
  const restSel = document.getElementById('ctrlMosaicRestModel');

  function fill(select, items, savedValue) {
    if (!select) return;
    const prev = savedValue || select.value;
    select.innerHTML = '<option value="">— select model —</option>';
    for (const item of items || []) {
      const opt = document.createElement('option');
      opt.value = item.path;
      opt.textContent = item.label;
      select.appendChild(opt);
    }
    // Restore previously chosen value if still available
    if (prev && [...select.options].some(o => o.value === prev)) {
      select.value = prev;
    }
  }

  fill(detSel, data.detection);
  fill(restSel, data.restoration);
  _updateRestorationButtonStates();
}


// ── Mosaic params gathering ───────────────────────────────
// Produces the payload the server expects: {mosaic: {...}, encoder: {...}}
// Encoder is shared with face-swap — single source of truth.

function gatherMosaicParams() {
  const score = parseInt(document.getElementById('ctrlMosaicDetScore').value);
  return {
    mosaic: {
      detection_model: document.getElementById('ctrlMosaicDetModel').value,
      restoration_model: document.getElementById('ctrlMosaicRestModel').value,
      mosaic_detection_score: score / 100.0,
      mosaic_detection_batch_size: parseInt(document.getElementById('ctrlMosaicDetBatch').value),
      mosaic_detect_only: document.getElementById('ctrlMosaicDetectOnly').checked,
      mosaic_max_clip_size: parseInt(document.getElementById('ctrlMosaicMaxClip').value),
      mosaic_temporal_overlap: parseInt(document.getElementById('ctrlMosaicOverlap').value),
      mosaic_crossfade: document.getElementById('ctrlMosaicCrossfade').checked,
      mosaic_blend_frames: parseInt(document.getElementById('ctrlMosaicBlend').value),
      mosaic_denoise: document.getElementById('ctrlMosaicDenoise').value,
      mosaic_color_match: document.getElementById('ctrlMosaicColorMatch').checked,
      mosaic_fp16: document.getElementById('ctrlMosaicFp16').checked,
      mosaic_compile_trt: document.getElementById('ctrlMosaicCompileTrt').checked,
    },
    encoder: {
      codec: document.getElementById('ctrlCodec').value,
      preset: document.getElementById('ctrlPreset').value,
      qp: parseInt(document.getElementById('ctrlQP').value),
    },
  };
}

function buildMosaicParamsSummary() {
  const p = gatherMosaicParams();
  const m = p.mosaic;
  const parts = [];
  const detName = document.getElementById('ctrlMosaicDetModel').selectedOptions[0]?.textContent || '—';
  parts.push(`Det: ${detName}`);
  parts.push(`Score: ${m.mosaic_detection_score.toFixed(2)}`);
  if (m.mosaic_detect_only) {
    parts.push('⬛ DETECT ONLY');
  } else {
    const restName = document.getElementById('ctrlMosaicRestModel').selectedOptions[0]?.textContent || '—';
    parts.push(`Rest: ${restName}`);
    parts.push(`Clip: ${m.mosaic_max_clip_size}/${m.mosaic_temporal_overlap}`);
    if (m.mosaic_crossfade) {
      const bf = m.mosaic_blend_frames < 0 ? 'auto' : m.mosaic_blend_frames;
      parts.push(`Crossfade: ${bf}`);
    }
    if (m.mosaic_color_match) parts.push('Color Match');
  }
  parts.push(`Enc: ${p.encoder.codec.toUpperCase()}/${p.encoder.preset}/QP${p.encoder.qp}`);
  return parts.join(' · ');
}


// ── Button enable logic for Restoration mode ──────────────

function _updateRestorationButtonStates() {
  const hasVideo = !!state.videoPath;
  const detModel = document.getElementById('ctrlMosaicDetModel').value;
  const restModel = document.getElementById('ctrlMosaicRestModel').value;
  const detectOnly = document.getElementById('ctrlMosaicDetectOnly').checked;
  const hasModels = detModel && (detectOnly || restModel);
  const hasSegment = state.segmentEndTime > state.segmentStartTime;
  const inPreview = state.previewMode || false;

  const restoreBtn = document.getElementById('restoreBtn');
  const restoreSaveBtn = document.getElementById('restoreSaveBtn');
  const restorePreviewBtn = document.getElementById('restorePreviewBtn');

  // Restore (segment scope): needs video + models + segment marked.
  if (restoreBtn) restoreBtn.disabled = !hasVideo || !hasModels || !hasSegment || inPreview;

  // Restore & Save (full video): only needs video + models.
  if (restoreSaveBtn) restoreSaveBtn.disabled = !hasVideo || !hasModels || inPreview;

  // Preview: enabled + highlighted when a restored preview is ready.
  // state.previewReady is shared with face-swap; clearSegment() in
  // player.js resets it, so we don't need a separate flag.
  if (restorePreviewBtn) {
    if (!state.previewReady) {
      restorePreviewBtn.disabled = true;
      restorePreviewBtn.classList.remove('highlight', 'primary');
      if (!inPreview) restorePreviewBtn.textContent = 'Preview';
    } else {
      restorePreviewBtn.disabled = false;
      if (inPreview) {
        restorePreviewBtn.textContent = '◀ Back';
        restorePreviewBtn.classList.remove('highlight');
        restorePreviewBtn.classList.add('primary');
      } else {
        restorePreviewBtn.textContent = 'Preview';
        restorePreviewBtn.classList.remove('primary');
        restorePreviewBtn.classList.add('highlight');
      }
    }
  }
}

// Wrap the face-swap _updateButtonStates so any call updates both modes.
// init.js declares it with `function _updateButtonStates(){}`, which both
// creates a global binding AND a property on window. Setting only
// window._updateButtonStates was unreliable across browsers (the lexical
// global binding stayed pointing at the original). Reassigning the bare
// name in non-strict global scope updates the same slot.
if (typeof _updateButtonStates === 'function') {
  const _origUpdateButtonStates = _updateButtonStates;
  _updateButtonStates = function _updateButtonStatesWrapped() {
    _origUpdateButtonStates();
    _updateRestorationButtonStates();
  };
  // Mirror onto window in case any browser uses a separate slot for
  // window-property assignment vs lexical reassignment.
  window._updateButtonStates = _updateButtonStates;
  console.log('[mosaic] _updateButtonStates wrapped');
} else {
  console.warn('[mosaic] _updateButtonStates not found — wrap skipped');
}

// Belt-and-braces: also hook loadVideo() directly. If the
// _updateButtonStates wrap above somehow doesn't fire (browser quirk,
// script-load timing), the restoration buttons will still update right
// after a video load via this path.
if (typeof loadVideo === 'function') {
  const _origLoadVideo = loadVideo;
  loadVideo = async function _loadVideoWrapped(path) {
    const result = await _origLoadVideo(path);
    _updateRestorationButtonStates();
    return result;
  };
  window.loadVideo = loadVideo;
  console.log('[mosaic] loadVideo wrapped');
}

// Re-check on relevant changes
['ctrlMosaicDetModel', 'ctrlMosaicRestModel', 'ctrlMosaicDetectOnly'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', _updateRestorationButtonStates);
});


// ── Action button handlers ────────────────────────────────

// Shared progress modal element refs (declared by init.js)
function _showProgressModal(title, summary) {
  progressModal.classList.remove('hidden');
  progressTitle.textContent = title;
  progressParams.textContent = summary;
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressFps.textContent = '— fps';
  progressEta.textContent = 'ETA: —';
}

function _pollMosaicProgress({onComplete, onError}) {
  return setInterval(async () => {
    const prog = await apiGet('/api/progress');
    if (!prog || prog.error) return;

    const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;
    progressBar.style.width = pct + '%';
    progressPercent.textContent = `${pct}% (${prog.frame}/${prog.total})`;
    progressFps.textContent = `${prog.fps || '—'} fps`;
    const det = prog.detections || 0;
    const res = prog.restorations || 0;
    const buf = prog.buffered || 0;
    progressEta.textContent =
      `ETA: ${prog.eta || '—'} | ${det} det, ${res} res, buf=${buf}`;

    if (prog.status === 'complete') {
      onComplete(prog);
    } else if (prog.status === 'error' || prog.status === 'cancelled') {
      onError(prog);
    }
  }, 500);
}

// Restore (segment, with preview)
document.getElementById('restoreBtn').addEventListener('click', async () => {
  if (!state.videoPath) {
    alert('Load a video first.');
    return;
  }

  // Exit face-swap preview if active (player will switch tracks)
  if (state.previewMode && typeof previewBtn !== 'undefined') {
    previewBtn.click();
  }
  state.previewReady = false;
  _updateRestorationButtonStates();

  const params = gatherMosaicParams();

  // Button is segment-gated, so we expect a marked segment by the time
  // we get here; defensive guard anyway.
  const startTime = state.segmentStartTime || 0;
  const endTime = state.segmentEndTime || 0;
  if (endTime <= startTime) {
    alert('Mark a segment first (use the segment marker in the player).');
    _updateRestorationButtonStates();
    return;
  }
  console.log(`[ChitraMaya] Restore segment: ${startTime.toFixed(2)}s → ${endTime.toFixed(2)}s`);

  const result = await apiPost('/api/mosaic-segment', {
    params, start_time: startTime, end_time: endTime,
  });
  if (result.error) {
    alert('Failed to start: ' + result.error);
    _updateRestorationButtonStates();
    return;
  }

  _showProgressModal('Restoring Segment...', buildMosaicParamsSummary());
  const restoreBtn = document.getElementById('restoreBtn');
  const restoreSaveBtn = document.getElementById('restoreSaveBtn');
  restoreBtn.disabled = true;
  restoreSaveBtn.disabled = true;

  const pollInterval = _pollMosaicProgress({
    onComplete: (prog) => {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Segment Complete';
      progressPercent.textContent =
        `${prog.frame} frames, ${prog.detections || 0} det, ${prog.restorations || 0} res`;
      progressBar.style.width = '100%';
      progressEta.textContent = '';
      progressCancel.textContent = 'Close';
      state.previewReady = true;
      _updateRestorationButtonStates();
    },
    onError: (prog) => {
      clearInterval(pollInterval);
      const cancelled = prog.status === 'cancelled' && prog.frame > 0;
      if (cancelled) {
        progressTitle.textContent = `Cancelled — ${prog.frame} frames processed`;
        progressPercent.textContent =
          `${prog.detections || 0} det, ${prog.restorations || 0} res`;
        state.previewReady = true;
      } else {
        progressTitle.textContent = prog.status === 'error' ? 'Error' : 'Cancelled';
        progressPercent.textContent = prog.error || 'No frames processed';
      }
      progressBar.style.width = '0%';
      progressCancel.textContent = 'Close';
      _updateRestorationButtonStates();
    },
  });

  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled')) {
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
      _updateRestorationButtonStates();
    } else {
      await apiPost('/api/cancel');
    }
  };
});


// Restore & Save (full video, no preview)
document.getElementById('restoreSaveBtn').addEventListener('click', async () => {
  if (!state.videoPath) {
    alert('Load a video first.');
    return;
  }

  // Exit any preview mode
  if (state.previewMode && typeof previewBtn !== 'undefined') {
    previewBtn.click();
  }

  const params = gatherMosaicParams();
  console.log('[ChitraMaya] Restore & Save:', params);

  const result = await apiPost('/api/mosaic-full', { params });
  if (result.error) {
    alert('Failed to start: ' + result.error);
    return;
  }

  _showProgressModal('Restoring Video...', buildMosaicParamsSummary());
  const restoreBtn = document.getElementById('restoreBtn');
  const restoreSaveBtn = document.getElementById('restoreSaveBtn');
  restoreBtn.disabled = true;
  restoreSaveBtn.disabled = true;

  const pollInterval = _pollMosaicProgress({
    onComplete: (prog) => {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Complete!';
      progressPercent.textContent =
        `Done — ${prog.frame} frames, ${prog.detections || 0} det, ${prog.restorations || 0} res`;
      progressBar.style.width = '100%';
      progressCancel.textContent = 'Close';
      _updateRestorationButtonStates();
      console.log('[ChitraMaya] Restore & Save complete:', prog);
    },
    onError: (prog) => {
      clearInterval(pollInterval);
      progressTitle.textContent = prog.status === 'error' ? 'Error' : 'Cancelled';
      progressPercent.textContent = prog.error || 'Processing failed';
      progressCancel.textContent = 'Close';
      _updateRestorationButtonStates();
    },
  });

  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled')) {
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
      _updateRestorationButtonStates();
    } else {
      await apiPost('/api/cancel');
    }
  };
});


// Preview (toggle player between /video and /preview-video)
// Mirrors face-swap previewBtn — server.preview_path was set by
// /api/mosaic-segment, /preview-video serves whatever's at that path.
document.getElementById('restorePreviewBtn').addEventListener('click', () => {
  if (!state.previewReady) return;
  const btn = document.getElementById('restorePreviewBtn');

  if (!state.previewMode) {
    // Enter preview — switch player to preview video
    state.previewMode = true;
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}
    player.src = '/preview-video?t=' + Date.now();
    player.load();
    player.play().catch(() => {});

    btn.textContent = '◀ Back';
    btn.classList.remove('highlight');
    btn.classList.add('primary');
    console.log('[ChitraMaya] Entered restoration Preview mode');
  } else {
    // Exit preview — switch back to original video
    state.previewMode = false;
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}
    player.src = '/video?t=' + Date.now();
    player.load();
    player.addEventListener('loadedmetadata', () => {
      if (typeof _updateSegmentVisual === 'function') _updateSegmentVisual();
    }, { once: true });

    btn.textContent = 'Preview';
    btn.classList.remove('primary');
    if (state.previewReady) btn.classList.add('highlight');
    console.log('[ChitraMaya] Exited restoration Preview mode');
  }
});


// Tooltip for the placeholder Detect button
const restoreDetectBtn = document.getElementById('restoreDetectBtn');
if (restoreDetectBtn) {
  restoreDetectBtn.title = 'Coming soon — use the Show Mask Only checkbox + Restore for now';
}


// ── Init: populate dropdowns + first state pass ───────────
populateMosaicModelDropdowns();
_updateRestorationButtonStates();
