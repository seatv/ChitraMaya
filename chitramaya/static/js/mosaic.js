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

  fill(detSel, data.detection);
  fill(restSel, data.restoration);
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


// Tooltip for the placeholder Detect button
const restoreDetectBtn = document.getElementById('restoreDetectBtn');
if (restoreDetectBtn) {
  restoreDetectBtn.title = 'Coming soon — use the Show Mask Only checkbox + Restore for now';
}


// ── Manage Models button ──────────────────────────────────
// No-op placeholder. A later increment replaces this with a modal for
// downloading source checkpoints (.pt / .pth) and compiling TRT engines.
// The "Use Tensor" checkboxes (increment 2) will route here when a
// requested engine is missing and no source checkpoint is present.
const manageModelsBtn = document.getElementById('manageModelsBtn');
if (manageModelsBtn) {
  manageModelsBtn.addEventListener('click', () => {
    console.info('[ChitraMaya] Manage Models: model download/compile modal is not implemented yet.');
  });
}


// ── Init: populate dropdowns + first state pass ───────────
populateMosaicModelDropdowns();
_updateRestorationButtonStates();
