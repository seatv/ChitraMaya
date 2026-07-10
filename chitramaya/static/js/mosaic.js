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

// The tab strip was removed — ChitraMaya has a single Restoration view.
// Call setActiveTab once so the body attribute, panel visibility, and
// footer-button visibility (restoration buttons shown, face-swap hidden)
// land in a consistent state on load. Without this the app booted into the
// non-existent "face-swap" mode: controls hidden, dead swap buttons shown.
setActiveTab('restoration');


// ── Mosaic dial value formatting ──────────────────────────
// Per-control overrides on top of the generic handler in params.js.

(function attachMosaicDialFormatting() {
  // Score + IoU sliders: 5-100 display as 0.05-1.00
  [['ctrlMosaicDetScore', 'valMosaicDetScore'],
   ['ctrlMosaicDetIou', 'valMosaicDetIou'],
   ['ctrlMosaicMaskOpacity', 'valMosaicMaskOpacity']].forEach(([ctrlId, valId]) => {
    const dial = document.getElementById(ctrlId);
    const valSpan = document.getElementById(valId);
    if (dial && valSpan) {
      const fmt = () => { valSpan.textContent = (parseInt(dial.value) / 100).toFixed(2); };
      dial.addEventListener('input', fmt);
      fmt();
    }
  });
})();


// ── Extend unified config system with mosaic controls ─────
// CONFIG_CONTROLS is defined in params.js; extending here keeps mosaic
// controls auto-persisted via Save Settings / Load Settings without
// duplicating the save/load infrastructure.

const MOSAIC_CONFIG_CONTROLS = [
  'ctrlMosaicDetModel', 'ctrlMosaicDetScore', 'ctrlMosaicDetIou',
  'ctrlMosaicDetBatch', 'ctrlMosaicDetFp16', 'ctrlMosaicDetTrt',
  'ctrlMosaicSbsSplit',
  'ctrlMosaicRestModel', 'ctrlMosaicMaxClip', 'ctrlMosaicRestFp16',
  'ctrlMosaicRoiDilate', 'ctrlMosaicFeather', 'ctrlMosaicBlendMask',
  'ctrlMosaicSegMasks', 'ctrlMosaicRestTrt',
  'ctrlMosaicMaskPreview', 'ctrlMosaicMaskColor', 'ctrlMosaicMaskOpacity',
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


// ── Compiled-engine awareness (Max Clip defaulting/constraint) ────────────
// Compiled clip sizes per restoration model, keyed by model path:
//   { "<path>": { fp16: [60], fp32: [] }, ... }
// Populated by populateMosaicModelDropdowns() from /api/list-mosaic-models.
let _mosaicRestEngines = {};

// Detection engine availability, keyed by detection model (.pt) path:
//   { "<path>": true|false }  (true = models/engines/<stem>.engine exists)
// Populated by populateMosaicModelDropdowns() from /api/list-mosaic-models.
let _mosaicDetEngines = {};

// Max Clip range when TRT is NOT constraining it (PyTorch chunks arbitrarily).
const MAX_CLIP_FREE = { min: 30, max: 180, step: 10 };

// Default / constrain the Max Clip slider to what is actually compiled for the
// selected restoration model + precision. Mirrors the server-side snap, so the
// value the user sees matches what will run. With TRT off, the range is free.
function updateMaxClipConstraints() {
  const slider = document.getElementById('ctrlMosaicMaxClip');
  const valSpan = document.getElementById('valMosaicMaxClip');
  if (!slider) return;

  const restModel = document.getElementById('ctrlMosaicRestModel').value;
  const fp16 = document.getElementById('ctrlMosaicRestFp16').checked;
  const useTrt = document.getElementById('ctrlMosaicRestTrt').checked;

  if (!useTrt) {
    // PyTorch path — clip size is free.
    slider.min = MAX_CLIP_FREE.min;
    slider.max = MAX_CLIP_FREE.max;
    slider.step = MAX_CLIP_FREE.step;
    slider.disabled = false;
    if (valSpan) valSpan.textContent = slider.value;
    _updateRestorationButtonStates();
    return;
  }

  const info = _mosaicRestEngines[restModel];
  const avail = info ? (fp16 ? info.fp16 : info.fp32) : [];

  if (!avail || avail.length === 0) {
    // TRT requested but nothing compiled for this model + precision. Leave the
    // slider free; the submit-time modal handles the no-engine case (Continue
    // on PyTorch / Manage Models), so we don't block the button here.
    slider.disabled = false;
    if (valSpan) valSpan.textContent = slider.value;
    _updateRestorationButtonStates();
    return;
  }

  if (avail.length === 1) {
    // Single compiled size — lock the slider to it.
    const only = avail[0];
    slider.min = only; slider.max = only; slider.step = 1;
    slider.value = only;
    slider.disabled = true;
    if (valSpan) valSpan.textContent = String(only);
  } else {
    // Multiple compiled sizes — allow the compiled range; snap the current
    // value to the nearest available <= current. The server snaps anything
    // in between as a backstop.
    const lo = avail[0];
    const hi = avail[avail.length - 1];
    slider.min = lo; slider.max = hi; slider.step = 1;
    slider.disabled = false;
    const cur = parseInt(slider.value);
    const le = avail.filter(n => n <= cur);
    const pick = le.length ? le[le.length - 1] : lo;
    slider.value = pick;
    if (valSpan) valSpan.textContent = String(pick);
  }

  _updateRestorationButtonStates();
}


// ── Use Tensor availability + "engine not found" modal ────
// The Use Tensor / FP16 checkboxes are just hardware preferences — they never
// pop a dialog on toggle. Availability is verified at SUBMIT time (Restore /
// Restore & Save), which is robust to engines renamed after the UI loaded.
function _tensorEngineAvailable(kind) {
  if (kind === 'det') {
    const m = document.getElementById('ctrlMosaicDetModel').value;
    if (!m) return false;
    if (/\.engine$/i.test(m)) return true;      // already an engine
    return _mosaicDetEngines[m] === true;
  }
  // restoration: any compiled set for the current precision counts — the Max
  // Clip constraint pins the clip to a compiled size, and the server snaps.
  const m = document.getElementById('ctrlMosaicRestModel').value;
  const fp16 = document.getElementById('ctrlMosaicRestFp16').checked;
  const info = _mosaicRestEngines[m];
  const avail = info ? (fp16 ? info.fp16 : info.fp32) : [];
  return Array.isArray(avail) && avail.length > 0;
}

// Re-fetch engine availability from the server (no dropdown changes). Called at
// submit time so a stale cache (engine renamed/added since load) can't slip a
// missing engine past the check — or falsely flag a present one.
async function _refreshEngineCaches() {
  const data = await apiGet('/api/list-mosaic-models');
  if (!data || data.error) return;
  _mosaicRestEngines = {};
  for (const item of data.restoration || []) {
    if (item.engines) _mosaicRestEngines[item.path] = item.engines;
  }
  _mosaicDetEngines = {};
  for (const item of data.detection || []) {
    _mosaicDetEngines[item.path] = !!item.has_engine;
  }
}

// Custom in-app modal (no window.confirm — OS-inconsistent). Resolves to
// 'continue' (run missing stages on PyTorch) or 'manage' (abort; opens Manage
// Models). Dismiss via backdrop = 'manage' (safe: abort).
function showTensorModal(missing) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('tensorModal');
    const msgEl = document.getElementById('tensorModalMessage');
    const contBtn = document.getElementById('tensorModalContinue');
    const mngBtn = document.getElementById('tensorModalManage');
    const stages = missing.map(k => k === 'det' ? 'detection' : 'restoration').join(' and ');
    const slow = missing.includes('rest')
      ? ' Restoration on PyTorch is much slower (it is the main bottleneck).'
      : '';
    msgEl.textContent =
      'No compiled TensorRT engine was found for the ' + stages +
      ' model at the current settings.\n\nContinue on PyTorch for the ' + stages +
      ' stage, or open Manage Models to compile?' + slow;
    overlay.classList.remove('hidden');

    function cleanup(result) {
      overlay.classList.add('hidden');
      contBtn.removeEventListener('click', onCont);
      mngBtn.removeEventListener('click', onMng);
      overlay.removeEventListener('click', onBackdrop);
      resolve(result);
    }
    function onCont() { cleanup('continue'); }
    function onMng() { cleanup('manage'); }
    function onBackdrop(e) { if (e.target === overlay) cleanup('manage'); }

    contBtn.addEventListener('click', onCont);
    mngBtn.addEventListener('click', onMng);
    overlay.addEventListener('click', onBackdrop);
  });
}

// Submit-time gate. Returns {proceed, override} where override marks which
// stages to force onto PyTorch for this run. Mask Preview skips the restoration
// check (pseudo needs no restoration engine).
async function checkTensorBeforeRun() {
  await _refreshEngineCaches();
  const maskPreview = document.getElementById('ctrlMosaicMaskPreview').checked;
  const missing = [];
  if (document.getElementById('ctrlMosaicDetTrt').checked && !_tensorEngineAvailable('det')) {
    missing.push('det');
  }
  if (!maskPreview && document.getElementById('ctrlMosaicRestTrt').checked
      && !_tensorEngineAvailable('rest')) {
    missing.push('rest');
  }
  if (missing.length === 0) return { proceed: true, override: null };

  const choice = await showTensorModal(missing);
  if (choice === 'continue') {
    return {
      proceed: true,
      override: { det: missing.includes('det'), rest: missing.includes('rest') },
    };
  }
  // 'manage' -> abort now; Manage Models modal lands in a later increment.
  const b = document.getElementById('manageModelsBtn');
  if (b) b.click();
  return { proceed: false, override: null };
}


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

  // Cache compiled clip sizes so Max Clip can default/constrain to them.
  _mosaicRestEngines = {};
  for (const item of data.restoration || []) {
    if (item.engines) _mosaicRestEngines[item.path] = item.engines;
  }

  // Cache detection engine availability for the Use Tensor toggle.
  _mosaicDetEngines = {};
  for (const item of data.detection || []) {
    _mosaicDetEngines[item.path] = !!item.has_engine;
  }

  // Restore the config's saved model selections (see applyConfig): stashed so
  // this survives the startup race with the config load, whichever resolves
  // first. Falls back to the current value if there's no stash.
  const _saved = (typeof window !== 'undefined' && window._pendingMosaicModels) || {};
  fill(detSel, data.detection, _saved.det);
  fill(restSel, data.restoration, _saved.rest);
  updateMaxClipConstraints();
  _updateRestorationButtonStates();
}


// ── Mosaic params gathering ───────────────────────────────
// Produces the payload the server expects: {mosaic: {...}, encoder: {...}}
// Encoder is shared with face-swap — single source of truth.

function gatherMosaicParams() {
  const score = parseInt(document.getElementById('ctrlMosaicDetScore').value);
  // Output/temp dirs travel in the job payload so the value in the box at
  // submit-time is authoritative — the server no longer depends on a prior
  // /api/set-output-dir call having fired (which only happened on Enter or
  // the folder dialog, so a typed-but-not-committed path was silently lost).
  const outDir = (typeof outputPath !== 'undefined' && outputPath && outputPath.value)
    ? outputPath.value.trim() : '';
  const tmpDir = (typeof tempPath !== 'undefined' && tempPath && tempPath.value)
    ? tempPath.value.trim() : '';
  return {
    output_dir: outDir,
    temp_dir: tmpDir,
    mosaic: {
      detection_model: document.getElementById('ctrlMosaicDetModel').value,
      restoration_model: document.getElementById('ctrlMosaicRestModel').value,
      mosaic_detection_score: score / 100.0,
      mosaic_iou: parseInt(document.getElementById('ctrlMosaicDetIou').value) / 100.0,
      mosaic_detection_batch_size: parseInt(document.getElementById('ctrlMosaicDetBatch').value),
      mosaic_detection_fp16: document.getElementById('ctrlMosaicDetFp16').checked,
      mosaic_detection_trt: document.getElementById('ctrlMosaicDetTrt').checked,
      mosaic_sbs_split: document.getElementById('ctrlMosaicSbsSplit').checked,
      mosaic_mask_preview: document.getElementById('ctrlMosaicMaskPreview').checked,
      mosaic_mask_color: document.getElementById('ctrlMosaicMaskColor').value,
      mosaic_mask_opacity: parseInt(document.getElementById('ctrlMosaicMaskOpacity').value) / 100.0,
      mosaic_max_clip_size: parseInt(document.getElementById('ctrlMosaicMaxClip').value),
      mosaic_restoration_fp16: document.getElementById('ctrlMosaicRestFp16').checked,
      mosaic_roi_dilate: parseInt(document.getElementById('ctrlMosaicRoiDilate').value),
      mosaic_feather_radius: parseInt(document.getElementById('ctrlMosaicFeather').value),
      mosaic_blend_mask: document.getElementById('ctrlMosaicBlendMask').value,
      mosaic_use_seg_masks: document.getElementById('ctrlMosaicSegMasks').checked,
      mosaic_restoration_trt: document.getElementById('ctrlMosaicRestTrt').checked,
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
  if (m.mosaic_mask_preview) {
    parts.push('MASK PREVIEW');
  } else {
    const restName = document.getElementById('ctrlMosaicRestModel').selectedOptions[0]?.textContent || '—';
    parts.push(`Rest: ${restName}`);
    parts.push(`Clip: ${m.mosaic_max_clip_size}`);
    parts.push(m.mosaic_restoration_fp16 ? 'FP16' : 'FP32');
    parts.push(m.mosaic_restoration_trt ? 'TRT' : 'PyTorch');
    if (m.mosaic_roi_dilate > 0) parts.push(`Dilate:${m.mosaic_roi_dilate}`);
    if (m.mosaic_feather_radius > 0) parts.push(`Feather:${m.mosaic_feather_radius}`);
    if (m.mosaic_blend_mask && m.mosaic_blend_mask !== 'none') parts.push(`Blend:${m.mosaic_blend_mask}`);
  }
  parts.push(`Enc: ${p.encoder.codec.toUpperCase()}/${p.encoder.preset}/QP${p.encoder.qp}`);
  return parts.join(' · ');
}


// ── Button enable logic for Restoration mode ──────────────

function _updateRestorationButtonStates() {
  const hasVideo = !!state.videoPath;
  const detModel = document.getElementById('ctrlMosaicDetModel').value;
  const restModel = document.getElementById('ctrlMosaicRestModel').value;
  const maskPreview = document.getElementById('ctrlMosaicMaskPreview').checked;
  const hasModels = detModel && (maskPreview || restModel);
  const hasSegment = state.segmentEndTime > state.segmentStartTime;
  const inPreview = state.previewMode || false;

  const restoreBtn = document.getElementById('restoreBtn');
  const restoreSaveBtn = document.getElementById('restoreSaveBtn');
  const restorePreviewBtn = document.getElementById('restorePreviewBtn');

  // A missing TRT engine no longer blocks Restore — the submit-time modal
  // offers Continue-on-PyTorch or Manage Models. So buttons key off video +
  // models (+ segment) only.
  // Restore (segment scope): needs video + models + segment marked.
  if (restoreBtn) {
    restoreBtn.disabled = !hasVideo || !hasModels || !hasSegment || inPreview;
    restoreBtn.title = '';
  }

  // Restore & Save (full video): only needs video + models.
  if (restoreSaveBtn) {
    restoreSaveBtn.disabled = !hasVideo || !hasModels || inPreview;
    restoreSaveBtn.title = '';
  }

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
['ctrlMosaicDetModel', 'ctrlMosaicRestModel', 'ctrlMosaicMaskPreview'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', _updateRestorationButtonStates);
});

// Controls that affect which compiled engine set applies re-evaluate the
// Max Clip constraint (which also refreshes button states).
['ctrlMosaicRestModel', 'ctrlMosaicRestFp16'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', updateMaxClipConstraints);
});

// Use Tensor toggles run the availability / compile-permission check, then
// refresh the relevant constraints + button states.
// Use Tensor toggles are plain preferences — no dialog on toggle. Detection
// affects nothing but button state; restoration re-runs the Max Clip
// constraint. Availability is checked at submit time.
const _detTrtEl = document.getElementById('ctrlMosaicDetTrt');
if (_detTrtEl) _detTrtEl.addEventListener('change', _updateRestorationButtonStates);
const _restTrtEl = document.getElementById('ctrlMosaicRestTrt');
if (_restTrtEl) _restTrtEl.addEventListener('change', updateMaxClipConstraints);


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
    // Bail only on a TRANSPORT error (apiGet returns {error} with no status).
    // A job error is {status:"error", error:"..."} and must reach onError.
    if (!prog || (prog.error && !prog.status)) return;

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

  // Verify TRT engines at submit time. If missing, the modal offers Continue
  // (PyTorch for that stage) or Manage Models (abort).
  const gate = await checkTensorBeforeRun();
  if (!gate.proceed) { _updateRestorationButtonStates(); return; }
  if (gate.override) {
    if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
    if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
  }

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


// Restore & Save (full video). On completion, Preview plays the full output.
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

  // Verify TRT engines at submit time (see segment handler).
  const gate = await checkTensorBeforeRun();
  if (!gate.proceed) { _updateRestorationButtonStates(); return; }
  if (gate.override) {
    if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
    if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
  }

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
      // Server pointed preview at the full output — enable Preview so clicking
      // it plays the completed file (same mechanism as segment preview).
      state.previewReady = true;
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


// (restoreDetectBtn is repurposed as the "Test Frame N" trigger — wired in the
// Frame-of-Interest section below.)


// ── Manage Models window (list + select + compile) ────────
// Lists local models with a compiled? badge, lets you multi-select rows, set
// Image Size (detection) + Max Clip Length (restoration), and compile the
// selected models for THIS machine's GPU (server shells out to the exe's
// -compile-* subcommands, streaming the log). Reached from the header button
// and the "engine not found" modal's Manage Models button.
let _mmPoll = null;
let _mmModels = [];             // [{path,label,kind:'det'|'rest',compiled,detail}]
const _mmSelected = new Set();  // selected model paths

async function mmPopulateList() {
  const el = document.getElementById('mmModelList');
  if (!el) return;
  el.textContent = 'Loading…';
  const data = await apiGet('/api/list-mosaic-models');
  if (!data || data.error) { el.textContent = 'Failed to list models.'; return; }
  _mmModels = [];
  (data.detection || []).forEach(m =>
    _mmModels.push({ path: m.path, label: m.label, kind: 'det',
                     compiled: !!m.has_engine, detail: '' }));
  (data.restoration || []).forEach(m => {
    const sizes = [...new Set([...((m.engines && m.engines.fp16) || []),
                               ...((m.engines && m.engines.fp32) || [])])].sort((a, b) => a - b);
    _mmModels.push({ path: m.path, label: m.label, kind: 'rest',
                     compiled: sizes.length > 0,
                     detail: sizes.length ? ('clips ' + sizes.join(',')) : '' });
  });
  for (const p of [..._mmSelected]) if (!_mmModels.some(m => m.path === p)) _mmSelected.delete(p);
  mmRenderList();
}

function mmRenderList() {
  const el = document.getElementById('mmModelList');
  if (!el) return;
  if (!_mmModels.length) {
    el.innerHTML = '<div class="mm-empty">No .pt / .pth models in models/. Add some (Download lands next), then Refresh.</div>';
    mmUpdateControls();
    return;
  }
  const badge = (ok, detail) =>
    `<span class="mm-badge ${ok ? 'mm-ok' : 'mm-no'}">`
    + (ok ? ('Compiled' + (detail ? ' · ' + detail : '')) : 'Not compiled') + '</span>';
  el.innerHTML = _mmModels.map(m => {
    const sel = _mmSelected.has(m.path) ? ' mm-selected' : '';
    const ext = m.kind === 'det' ? '.pt detection' : '.pth restoration';
    return `<div class="mm-row${sel}" data-path="${encodeURIComponent(m.path)}">`
      + `<span class="mm-name">${m.label} <span class="mm-ext">${ext}</span></span>`
      + badge(m.compiled, m.detail) + '</div>';
  }).join('');
  el.querySelectorAll('.mm-row').forEach(row => {
    row.addEventListener('click', () => {
      const p = decodeURIComponent(row.dataset.path);
      if (_mmSelected.has(p)) _mmSelected.delete(p); else _mmSelected.add(p);
      row.classList.toggle('mm-selected', _mmSelected.has(p));
      mmUpdateControls();
    });
  });
  mmUpdateControls();
}

function mmUpdateControls() {
  const btn = document.getElementById('mmCompile');
  const n = _mmSelected.size;
  if (btn && !btn.classList.contains('mm-compiling')) {
    btn.disabled = n === 0;
    btn.textContent = `Compile (${n} selected)`;
  }
  // Image Size applies to detection, Max Clip to restoration. Dim the slider
  // that no selected model uses (only once a selection excludes that kind).
  const kinds = new Set(_mmModels.filter(m => _mmSelected.has(m.path)).map(m => m.kind));
  const imgRow = document.getElementById('mmImgszRow');
  const clipRow = document.getElementById('mmMaxClipRow');
  if (imgRow)  imgRow.classList.toggle('mm-inert',  n > 0 && !kinds.has('det'));
  if (clipRow) clipRow.classList.toggle('mm-inert', n > 0 && !kinds.has('rest'));
}

function mmSelectMissing() {
  _mmSelected.clear();
  _mmModels.forEach(m => { if (!m.compiled) _mmSelected.add(m.path); });
  mmRenderList();
}

function _mmSetCompiling(on) {
  const btn = document.getElementById('mmCompile');
  if (!btn) return;
  btn.classList.toggle('mm-compiling', on);
  btn.disabled = on || _mmSelected.size === 0;
  btn.textContent = on ? '⏳ Compiling…' : `Compile (${_mmSelected.size} selected)`;
}

function _mmStartPolling() {
  const log = document.getElementById('mmLog');
  if (_mmPoll) clearInterval(_mmPoll);
  _mmSetCompiling(true);
  _mmPoll = setInterval(async () => {
    const s = await apiGet('/api/compile-log');
    if (!s) return;
    if (log) { log.textContent = s.log || ''; log.scrollTop = log.scrollHeight; }
    if (!s.running) {
      clearInterval(_mmPoll); _mmPoll = null;
      _mmSetCompiling(false);
      mmPopulateList();                        // Manage Models badges refresh
      if (typeof populateMosaicModelDropdowns === 'function') populateMosaicModelDropdowns();  // Control Panel dropdowns + engine caches
    }
  }, 1000);
}

async function mmCompile() {
  if (_mmSelected.size === 0) return;
  const log = document.getElementById('mmLog');
  const imgsz = parseInt(document.getElementById('mmImgsz').value, 10) || 640;
  const maxClip = parseInt(document.getElementById('mmMaxClip').value, 10) || 60;
  const force = document.getElementById('mmForce').checked;
  _mmSetCompiling(true);
  if (log) log.textContent = 'Starting…';
  const res = await apiPost('/api/compile-engines',
    { models: [..._mmSelected], imgsz, max_clip: maxClip, force });
  if (!res || res.error) {
    if (log) log.textContent = 'Error: ' + (res ? res.error : 'no response');
    _mmSetCompiling(false);
    return;
  }
  _mmStartPolling();
}

async function openManageModels() {
  const modal = document.getElementById('manageModelsModal');
  if (!modal) return;
  modal.classList.remove('hidden');
  await mmLoadSources();
  await mmPopulateList();
  const s = await apiGet('/api/compile-log');   // resume if a compile is running
  if (s && s.running) {
    const log = document.getElementById('mmLog');
    if (log) log.textContent = s.log || '';
    _mmStartPolling();
  }
}

function closeManageModels() {
  const modal = document.getElementById('manageModelsModal');
  if (modal) modal.classList.add('hidden');
  if (_mmPoll) { clearInterval(_mmPoll); _mmPoll = null; }  // leave compile running server-side
}

// ── Download (Hugging Face) ───────────────────────────────
let _mmDlPoll = null;
let _mmFetchFiles = [];            // [{path, size}]
let _mmCurrentRepoUrl = '';
const _mmDlSelected = new Set();   // selected repo file paths

async function mmLoadSources() {
  const sel = document.getElementById('mmSource');
  if (!sel) return;
  const data = await apiGet('/api/model-sources');
  const sources = (data && data.sources) || [];
  sel.innerHTML = sources.map(s =>
    `<option value="${encodeURIComponent(s.url)}">${s.name || s.url}</option>`).join('');
}

async function mmAddSource() {
  const url = (window.prompt('Hugging Face repo URL (e.g. https://huggingface.co/owner/repo):') || '').trim();
  if (!url) return;
  const name = (window.prompt('A short name for this source (optional):') || '').trim();
  const res = await apiPost('/api/model-sources', { url, name });
  if (!res || res.error) { alert('Could not add source: ' + (res ? res.error : 'no response')); return; }
  await mmLoadSources();
  const sel = document.getElementById('mmSource');
  if (sel) sel.value = encodeURIComponent(url);   // select the newly added
}

function _mmFmtSize(b) {
  if (!b) return '';
  return b >= 1048576 ? (b / 1048576).toFixed(0) + ' MB' : (b / 1024).toFixed(0) + ' KB';
}

async function mmFetch() {
  const sel = document.getElementById('mmSource');
  const list = document.getElementById('mmFetchList');
  const btns = document.getElementById('mmDlBtns');
  const fetchBtn = document.getElementById('mmFetch');
  if (!sel || !sel.value) return;
  _mmCurrentRepoUrl = decodeURIComponent(sel.value);
  if (fetchBtn) { fetchBtn.disabled = true; fetchBtn.textContent = '⏳ Fetching…'; }
  if (list) { list.classList.remove('mm-hidden'); list.innerHTML = 'Fetching…'; }
  if (btns) btns.classList.add('mm-hidden');
  try {
    const res = await apiPost('/api/fetch-model-list', { url: _mmCurrentRepoUrl });
    if (!res || res.error) { if (list) list.innerHTML = '<div class="mm-empty">' + ((res && res.error) || 'Fetch failed') + '</div>'; return; }
    _mmFetchFiles = res.files || [];
    _mmDlSelected.clear();
    mmFetchRender();
    if (btns) btns.classList.toggle('mm-hidden', _mmFetchFiles.length === 0);
  } finally {
    if (fetchBtn) { fetchBtn.disabled = false; fetchBtn.textContent = 'Fetch'; }
  }
}

function mmFetchRender() {
  const list = document.getElementById('mmFetchList');
  if (!list) return;
  if (!_mmFetchFiles.length) {
    list.innerHTML = '<div class="mm-empty">No .pt / .pth files in this repo.</div>';
    mmDlUpdateControls();
    return;
  }
  list.innerHTML = _mmFetchFiles.map(f => {
    const sel = _mmDlSelected.has(f.path) ? ' mm-selected' : '';
    return `<div class="mm-row${sel}" data-path="${encodeURIComponent(f.path)}">`
      + `<span class="mm-name">${f.path}</span>`
      + `<span class="mm-size">${_mmFmtSize(f.size)}</span></div>`;
  }).join('');
  list.querySelectorAll('.mm-row').forEach(row => {
    row.addEventListener('click', () => {
      const p = decodeURIComponent(row.dataset.path);
      if (_mmDlSelected.has(p)) _mmDlSelected.delete(p); else _mmDlSelected.add(p);
      row.classList.toggle('mm-selected', _mmDlSelected.has(p));
      mmDlUpdateControls();
    });
  });
  mmDlUpdateControls();
}

function mmDlUpdateControls() {
  const btn = document.getElementById('mmDownload');
  const n = _mmDlSelected.size;
  if (btn && !btn.classList.contains('mm-downloading')) {
    btn.disabled = n === 0;
    btn.textContent = `Download (${n} selected)`;
  }
}

function mmDlSelectAll() {
  _mmFetchFiles.forEach(f => _mmDlSelected.add(f.path));
  mmFetchRender();
}

function _mmSetDownloading(on) {
  const btn = document.getElementById('mmDownload');
  if (!btn) return;
  btn.classList.toggle('mm-downloading', on);
  btn.disabled = on || _mmDlSelected.size === 0;
  btn.textContent = on ? '⏳ Downloading…' : `Download (${_mmDlSelected.size} selected)`;
}

async function mmDownload() {
  if (_mmDlSelected.size === 0) return;
  const log = document.getElementById('mmLog');
  _mmSetDownloading(true);
  if (log) log.textContent = 'Starting download…';
  const res = await apiPost('/api/download-models',
    { url: _mmCurrentRepoUrl, files: [..._mmDlSelected] });
  if (!res || res.error) {
    if (log) log.textContent = 'Error: ' + (res ? res.error : 'no response');
    _mmSetDownloading(false);
    return;
  }
  if (_mmDlPoll) clearInterval(_mmDlPoll);
  _mmDlPoll = setInterval(async () => {
    const s = await apiGet('/api/download-log');
    if (!s) return;
    if (log) { log.textContent = s.log || ''; log.scrollTop = log.scrollHeight; }
    if (!s.running) {
      clearInterval(_mmDlPoll); _mmDlPoll = null;
      _mmSetDownloading(false);
      mmPopulateList();                        // downloaded models appear in the list below
      if (typeof populateMosaicModelDropdowns === 'function') populateMosaicModelDropdowns();  // AND in the Control Panel dropdowns (so they're selectable)
    }
  }, 800);
}

// Slider readouts — self-contained (kept out of the config system).
['mmImgsz', 'mmMaxClip'].forEach(id => {
  const dial = document.getElementById(id);
  const val = document.getElementById(id + 'Val');
  if (dial && val) { const upd = () => { val.textContent = dial.value; }; dial.addEventListener('input', upd); upd(); }
});

const manageModelsBtn = document.getElementById('manageModelsBtn');
if (manageModelsBtn) manageModelsBtn.addEventListener('click', openManageModels);
[['mmRefresh', mmPopulateList], ['mmSelectMissing', mmSelectMissing],
 ['mmCompile', mmCompile], ['mmClose', closeManageModels],
 ['mmAddSource', mmAddSource], ['mmFetch', mmFetch],
 ['mmDlSelectAll', mmDlSelectAll], ['mmDownload', mmDownload]].forEach(([id, fn]) => {
  const b = document.getElementById(id);
  if (b) b.addEventListener('click', fn);
});


// ── Frame-of-Interest "Test Frame" ────────────────────────
// Restore a short window centered on the current playhead frame and show each
// detected region as a [Mosaic | Restored] pair in the top strip. Click a pair
// to enlarge it over the player (controls stay visible). Re-running after a
// knob change refreshes the strip and the open enlarge in place — dial → test
// → see, without the overlay closing.
let _foiRegions = [];      // latest results
let _foiOpenIdx = null;    // region index currently enlarged over the player, or null
let _foiRunning = false;   // true while a test is in flight (protects the label)

// Button reads "Test Frame N" for the current playhead, so it's unambiguous
// which frame the click will process.
function _updateTestFrameLabel() {
  const btn = document.getElementById('restoreDetectBtn');
  if (!btn) return;
  if (_foiRunning) return;   // keep "⏳ Refreshing…" while a test is in flight
  let n = null;
  if (typeof frameAtTime === 'function' && typeof player !== 'undefined') {
    n = frameAtTime(player.currentTime);
  } else if (state.currentFrame != null) {
    n = state.currentFrame;
  }
  btn.textContent = (n === null || n === undefined) ? 'Test Frame' : `Test Frame ${n}`;
}

function _foiCurrentFrame() {
  if (typeof frameAtTime === 'function' && typeof player !== 'undefined') {
    return frameAtTime(player.currentTime);
  }
  return state.currentFrame || 0;
}

async function runFoiPreview() {
  const btn = document.getElementById('restoreDetectBtn');
  const status = document.getElementById('foiStatus');
  const container = document.getElementById('foiRegions');
  if (!container) return;

  if (!state.videoPath && !state.videoInfo) {
    status.textContent = 'Load a video first';
    return;
  }
  const frame = _foiCurrentFrame();
  if (frame === null || frame === undefined) {
    status.textContent = 'Pause on a frame first';
    return;
  }

  // Honor the same TRT-engine gate as Restore / Restore & Save: if Use Tensor
  // is on but the engine is missing, prompt (Manage vs Continue-on-PyTorch)
  // instead of silently falling back to PyTorch.
  const gate = await checkTensorBeforeRun();
  if (!gate.proceed) return;             // user chose Manage Models — abort
  const params = gatherMosaicParams();
  if (gate.override) {
    if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
    if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
  }

  // The button itself is the working indicator — no modal, no poll — so the
  // images (strip + open enlarge) stay put and your eye keeps its reference for
  // the before/after when the new result swaps in.
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Refreshing…'; }
  _foiRunning = true;

  try {
    const res = await apiPost('/api/mosaic-foi', {
      frame: frame,
      params: params,
    });
    if (!res || res.error) {
      const msg = (res && res.error) ? res.error : 'no response';
      if (/busy/i.test(msg)) {
        // Another op is running — leave the current view untouched.
        status.textContent = 'Still working — try again in a moment.';
      } else {
        status.textContent = 'Error: ' + msg;
        _foiRegions = [];
        _closeFoiEnlarge();
      }
      return;
    }
    _foiRegions = res.regions || [];
    const win = res.window || [];
    status.textContent =
      `Frame ${res.frame} · ${_foiRegions.length} region${_foiRegions.length === 1 ? '' : 's'}`
      + (win.length === 2 ? ` · window ${win[0]}–${win[1]}` : '')
      + ' · click a pair to enlarge';
    if (_foiRegions.length === 0) {
      container.innerHTML = '<div class="foi-empty">No mosaics detected on this frame.</div>';
      _closeFoiEnlarge();
      return;
    }
    const cell = (label, b64) => {
      const img = b64
        ? `<img class="foi-img" src="data:image/jpeg;base64,${b64}" alt="${label}">`
        : '<div class="foi-img foi-na">—</div>';
      return `<div class="foi-cell"><div class="foi-clabel">${label}</div>${img}</div>`;
    };
    // Rebuild the strip now that the new result is in hand (swap in place, so
    // it never blanks while refreshing).
    container.innerHTML = '';
    _foiRegions.forEach((r, i) => {
      const row = document.createElement('div');
      row.className = 'foi-row';
      row.title = 'Click to enlarge over the player';
      if (i === _foiOpenIdx) row.classList.add('foi-row-active');
      row.innerHTML = cell('Mosaic', r.mosaic) + cell('Restored', r.restored);
      row.addEventListener('click', () => openFoiEnlarge(i));
      container.appendChild(row);
    });

    // Refresh the open enlarge in place (same region index), or close it if the
    // new result no longer has that region.
    if (_foiOpenIdx !== null) {
      if (_foiOpenIdx < _foiRegions.length) openFoiEnlarge(_foiOpenIdx);
      else _closeFoiEnlarge();
    }
  } catch (e) {
    status.textContent = 'Error: ' + (e && e.message ? e.message : e);
  } finally {
    _foiRunning = false;
    if (btn) { btn.disabled = false; _updateTestFrameLabel(); }
  }
}

// Enlarge one region's Mosaic|Restored over the player, reusing the center-area
// zoom overlay (absolute inset:0 in .center-area) so the side controls stay
// visible. Panels scale to the crop's aspect (foi-mode overrides the fixed
// 512 square). Does not close on re-run — it refreshes.
function openFoiEnlarge(idx) {
  const r = _foiRegions[idx];
  if (!r) return;
  _foiOpenIdx = idx;
  const overlay = document.getElementById('zoomOverlay');
  const orig = document.getElementById('zoomOriginal');
  const rest = document.getElementById('zoomImage');
  if (!overlay || !orig || !rest) return;

  // Labels: first panel Mosaic, second Restored.
  const labels = overlay.querySelectorAll('.zoom-label');
  if (labels[0]) labels[0].textContent = 'Mosaic';
  if (labels[1]) labels[1].textContent = 'Restored';

  orig.src = r.mosaic ? ('data:image/jpeg;base64,' + r.mosaic) : '';
  rest.src = r.restored ? ('data:image/jpeg;base64,' + r.restored) : '';
  overlay.classList.add('foi-mode');
  overlay.classList.remove('hidden');
  if (typeof state !== 'undefined') state.zoomOpen = true;

  // Mark the active strip row.
  document.querySelectorAll('#foiRegions .foi-row').forEach((el, i) => {
    el.classList.toggle('foi-row-active', i === idx);
  });
}

function _closeFoiEnlarge() {
  _foiOpenIdx = null;
  const overlay = document.getElementById('zoomOverlay');
  if (overlay) {
    overlay.classList.add('hidden');
    overlay.classList.remove('foi-mode');
  }
  const orig = document.getElementById('zoomOriginal');
  const rest = document.getElementById('zoomImage');
  if (orig) orig.src = '';
  if (rest) rest.src = '';
  if (typeof state !== 'undefined') state.zoomOpen = false;
  document.querySelectorAll('#foiRegions .foi-row').forEach((el) => {
    el.classList.remove('foi-row-active');
  });
}

const _foiBtn = document.getElementById('restoreDetectBtn');
if (_foiBtn) _foiBtn.addEventListener('click', runFoiPreview);

// Full reset of the Test Frame state — regions cache, strip, and any open
// enlarge. Called from resetProject (New) so a new video starts clean.
function _foiReset() {
  _foiRegions = [];
  _foiOpenIdx = null;
  if (_foiRunning) _foiRunning = false;
  const c = document.getElementById('foiRegions');
  if (c) c.innerHTML = '';
  const st = document.getElementById('foiStatus');
  if (st) st.textContent = 'Pause on a frame, then press Test Frame (below)';
  _closeFoiEnlarge();
}
const _foiZoomClose = document.getElementById('zoomClose');
if (_foiZoomClose) _foiZoomClose.addEventListener('click', _closeFoiEnlarge);

// Keep the button label in sync with the playhead.
if (typeof player !== 'undefined' && player) {
  player.addEventListener('timeupdate', _updateTestFrameLabel);
  player.addEventListener('seeked', _updateTestFrameLabel);
  player.addEventListener('loadedmetadata', _updateTestFrameLabel);
}
_updateTestFrameLabel();


// ── Init: populate dropdowns + first state pass ───────────
populateMosaicModelDropdowns();
_updateRestorationButtonStates();
