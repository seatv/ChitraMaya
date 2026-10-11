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
  // Censor Block is an integer px value (not a 0-1 ratio).
  const bdial = document.getElementById('ctrlMosaicCensorBlock');
  const bval = document.getElementById('valMosaicCensorBlock');
  if (bdial && bval) {
    const bf = () => { bval.textContent = bdial.value; };
    bdial.addEventListener('input', bf);
    bf();
  }
})();


// ── Extend unified config system with mosaic controls ─────
// CONFIG_CONTROLS is defined in params.js; extending here keeps mosaic
// controls auto-persisted via Save Settings / Load Settings without
// duplicating the save/load infrastructure.

const MOSAIC_CONFIG_CONTROLS = [
  'ctrlMosaicDetModel', 'ctrlMosaicDetScore', 'ctrlMosaicDetIou',
  'ctrlMosaicDetBatch', 'ctrlMosaicDetImgsz', 'ctrlMosaicDetFp16', 'ctrlMosaicDetTrt',
  'ctrlMosaicSbsSplit', 'ctrlMosaicVrProjection',
  'ctrlMosaicRestModel', 'ctrlMosaicMaxClip', 'ctrlMosaicRestFp16',
  'ctrlMosaicOverlap', 'ctrlMosaicCrossfade',   // CM-213
  'ctrlMosaicSecondary',
  'ctrlMosaicSecondaryDenoise',  // Batch 74 (CM-146 fix): was missing -- Save/Load Settings silently reset denoise to none
  'ctrlMosaicSecondaryQuality',  // CM-217
  'ctrlMosaicTemporalFix',
  'ctrlMosaicRoiDilate', 'ctrlMosaicFeather', 'ctrlMosaicBlendMask',
  'ctrlMosaicCropCap',   // CM-227: open-scene crop storage cap (Off / 1024 / 768 / 512)
  'ctrlMosaicSegMasks', 'ctrlMosaicResidualScan', 'ctrlMosaicRestTrt',
  'ctrlMosaicMaskPreview', 'ctrlMosaicMaskColor', 'ctrlMosaicMaskOpacity',
  'ctrlMosaicCensor', 'ctrlMosaicCensorBlock',
  'ctrlAsyncEncoder',
  'ctrlOutputUpscale',  // CM-219/CM-228: whole-frame upscale before encode (factor or target)
  'ctrlUpscaleScaler',  // CM-228: esrgan | rtx
  'ctrlMosaicUpscale',  // CM-228: the Upscale alternate execution mode (decode -> scale -> encode, no models)
  'ctrlStoreBackend',   // CM-084 (Batch 38): FrameStore backend
  'ctrlHwDecode',       // T20d (CM-234): Hardware decode on the ffmpeg lane (auto | off)
  'ctrlOutputSuffix',   // CM-202 (GitHub #10): output file suffix
  'ctrlRunReport',      // CM-202 (GitHub #10): run files beside | temp | off
  'mmEndpoint',         // CM-199 (GitHub #13): model download endpoint (Manage Models)
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


// Batch 54 (field, 2026-08-23): restore a saved model selection by FILENAME
// when the exact path is not among the options. A config carried from
// another machine (or an install moved to a new drive) stores the OTHER
// box's absolute model paths -- exact match failed and the model fields sat
// empty on first run even though the same model files existed locally.
// Same model, new home: match on basename (case-insensitive, / or \).
// Saving afterwards writes the local path and exact matching takes over.
// Returns true if a match was applied. Used by fill() in
// populateMosaicModelDropdowns() and by applyConfig() in params.js
// (guarded with typeof, so script/config load order does not matter).
function matchModelOptionByName(select, wantedPath) {
  if (!select || !wantedPath) return false;
  const base = String(wantedPath).split(/[\\/]/).pop().toLowerCase();
  if (!base) return false;
  const opt = [...select.options].find(o =>
    o.value && o.value.split(/[\\/]/).pop().toLowerCase() === base);
  if (!opt) return false;
  select.value = opt.value;
  console.log('[ChitraMaya] model restored by filename match:', base,
              '(saved path pointed at another location)');
  return true;
}

// ── Compiled-engine awareness (Max Clip defaulting/constraint) ────────────
// Compiled clip sizes per restoration model, keyed by model path:
//   { "<path>": { fp16: [60], fp32: [] }, ... }
// Populated by populateMosaicModelDropdowns() from /api/list-mosaic-models.
let _mosaicRestEngines = {};

// Detection engine availability, keyed by detection model (.pt) path:
//   { "<path>": true|false }  (true = models/engines/<stem>.engine exists)
// Populated by populateMosaicModelDropdowns() from /api/list-mosaic-models.
let _mosaicDetEngines = {};
// CM-148 (Batch 76): compiled imgsz per detection model (null = unknown).
// CM-218: several flavors can coexist -- this holds the LIST of compiled
// Image Sizes (sorted) so the gate fires only when none matches the dial.
let _mosaicDetEngineImgsz = {};

// Max Clip range when TRT is NOT constraining it. Batch 42: on the
// PyTorch path the clip now feeds BasicVSR++ whole (lada semantics --
// clip length IS the temporal window), so the only ceiling is memory:
// VRAM during the forward scales with clip length. Dial unlocked to 600;
// a too-long clip fails with a catchable torch OOM, not a native crash.
const MAX_CLIP_FREE = { min: 30, max: 600, step: 10 };
// v1.60 (CM-104): with Tensor on, engines are clip-size independent — one
// compiled set serves any Max Clip, so the dial runs free up to 720
// (Jasna's validated ceiling; past ~180 wall time stops improving and only
// activation VRAM grows).
const MAX_CLIP_TRT = { min: 30, max: 720, step: 10 };

// Set the Max Clip slider range for the selected restoration model +
// precision. v1.60: engines no longer constrain the dial — the range is
// free on both paths; only the ceilings differ (PyTorch 600 / Tensor 720).
function updateMaxClipConstraints() {
  const slider = document.getElementById('ctrlMosaicMaxClip');
  const valSpan = document.getElementById('valMosaicMaxClip');
  if (!slider) return;

  const useTrt = document.getElementById('ctrlMosaicRestTrt').checked;
  const range = useTrt ? MAX_CLIP_TRT : MAX_CLIP_FREE;

  slider.min = range.min;
  slider.max = range.max;
  slider.step = range.step;
  slider.disabled = false;
  if (parseInt(slider.value) > range.max) slider.value = range.max;
  if (valSpan) valSpan.textContent = slider.value;

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
  _mosaicDetEngineImgsz = {};
  for (const item of data.detection || []) {
    _mosaicDetEngines[item.path] = !!item.has_engine;
    _mosaicDetEngineImgsz[item.path] = _detEngineSizes(item);   // CM-218
  }
}

// CM-218: the compiled Image Sizes of a detection model's engine flavors
// (falls back to the single engine_imgsz of an older server). null = none known.
// T20b: the Manage Models button says what a new user must do next, in colour.
// GREEN "Get Models" when there is nothing to run with (no detection or no
// restoration model); AMBER "Compile Models" on the NVIDIA edition when
// models are present, TensorRT is ticked, and no detection engine matches
// the panel's Image Size (or no restoration engine exists) -- the run would
// work, in PyTorch, at a fraction of the speed, and nothing said so. Grey
// "Manage Models" otherwise. Off-CUDA there is nothing to compile: never amber.
let _mmLastModelData = null;
function _updateManageModelsButton() {
  const btn = document.getElementById('manageModelsBtn');
  if (!btn) return;
  const data = _mmLastModelData;
  btn.classList.remove('mm-btn-get', 'mm-btn-compile');
  if (!data) return;
  const det = Array.isArray(data.detection) ? data.detection : [];
  const rest = Array.isArray(data.restoration) ? data.restoration : [];
  if (!det.length || !rest.length) {
    btn.classList.add('mm-btn-get');
    btn.textContent = 'Get Models';
    btn.title = 'No models yet. Opens Manage Models; the Standard models section downloads the set in one click'
      + ((document.documentElement.dataset.edition || 'cuda') === 'cuda' ? ' and compiles the engines for this GPU.' : '.');
    return;
  }
  const ed = document.documentElement.dataset.edition || '';
  const trt = document.getElementById('ctrlMosaicDetTrt');
  const imgsz = parseInt((document.getElementById('ctrlMosaicDetImgsz') || {}).value || '640', 10) || 640;
  if ((!ed || ed === 'cuda') && trt && trt.checked) {
    const detOk = det.some(it => (_detEngineSizes(it) || []).includes(imgsz));
    const restOk = rest.some(it => it.engines && (it.engines.fixed
      || ((it.engines.fp16 || []).length + (it.engines.fp32 || []).length) > 0));
    if (!detOk || !restOk) {
      btn.classList.add('mm-btn-compile');
      btn.textContent = 'Compile Models';
      btn.title = (!detOk
        ? `No detection engine is compiled for Image Size ${imgsz}; `
        : 'No restoration engine is compiled; ')
        + 'the run works in PyTorch at a fraction of the speed. Opens Manage Models to compile (minutes, once per GPU).';
      return;
    }
  }
  btn.textContent = 'Manage Models';
  btn.title = 'Download, compile and manage detection / restoration models and custom scalers';
}

function _detEngineSizes(item) {
  if (Array.isArray(item.engine_imgszs) && item.engine_imgszs.length) {
    return item.engine_imgszs.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  }
  return item.engine_imgsz == null ? null : [Number(item.engine_imgsz)];
}

// Custom in-app modal (no window.confirm — OS-inconsistent). Resolves to
// 'continue' (run missing stages on PyTorch) or 'manage' (abort; opens Manage
// Models). Dismiss via backdrop = 'manage' (safe: abort).
function showTensorModal(missing, mismatch) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('tensorModal');
    const titleEl = document.getElementById('tensorModalTitle');
    const msgEl = document.getElementById('tensorModalMessage');
    const contBtn = document.getElementById('tensorModalContinue');
    const mngBtn = document.getElementById('tensorModalManage');
    const engBtn = document.getElementById('tensorModalEngineSize');

    if (mismatch) {
      // CM-148 (Batch 76): engine exists but was compiled at a different
      // Image Size than the dial. Previously this ran the ENTIRE job on
      // PyTorch detection with only a console line -- a 2h45m benchmark
      // spent 57 min detecting on PyTorch before anyone noticed.
      if (titleEl) titleEl.textContent = 'TensorRT engine size mismatch';
      const compiledAt = (mismatch.all && mismatch.all.length > 1)
        ? ('The compiled detection engines were built at Image Sizes ' + mismatch.all.join(', '))
        : ('The compiled detection engine was built at Image Size ' + mismatch.have);
      msgEl.textContent =
        compiledAt + ', but the dial is set to ' + mismatch.want +
        '.\n\nUse engine size runs this job at ' + mismatch.have +
        ' with TensorRT (fast; detection behavior matches the compiled ' +
        'size). Continue on PyTorch keeps ' + mismatch.want +
        ' but detection runs without TensorRT (slower). Manage Models ' +
        'aborts so you can compile a ' + mismatch.want + ' flavor (the compiled ones are kept).';
      if (engBtn) {
        engBtn.textContent = 'Use engine size (' + mismatch.have + ')';
        engBtn.classList.remove('hidden');
      }
    } else {
      if (titleEl) titleEl.textContent = 'TensorRT engine not found';
      const stages = missing.map(k => k === 'det' ? 'detection' : 'restoration').join(' and ');
      const slow = missing.includes('rest')
        ? ' Restoration on PyTorch is much slower (it is the main bottleneck).'
        : '';
      msgEl.textContent =
        'No compiled TensorRT engine was found for the ' + stages +
        ' model at the current settings.\n\nContinue on PyTorch for the ' + stages +
        ' stage, or open Manage Models to compile?' + slow;
      if (engBtn) engBtn.classList.add('hidden');
    }
    overlay.classList.remove('hidden');

    function cleanup(result) {
      overlay.classList.add('hidden');
      if (engBtn) engBtn.classList.add('hidden');
      contBtn.removeEventListener('click', onCont);
      mngBtn.removeEventListener('click', onMng);
      if (engBtn) engBtn.removeEventListener('click', onEng);
      overlay.removeEventListener('click', onBackdrop);
      resolve(result);
    }
    function onCont() { cleanup('continue'); }
    function onMng() { cleanup('manage'); }
    function onEng() { cleanup('engine-size'); }
    function onBackdrop(e) { if (e.target === overlay) cleanup('manage'); }

    contBtn.addEventListener('click', onCont);
    mngBtn.addEventListener('click', onMng);
    if (engBtn) engBtn.addEventListener('click', onEng);
    overlay.addEventListener('click', onBackdrop);
  });
}

// CM-236 (T22): the VRAM preflight BEFORE the run. Asks the server for the
// estimate (free VRAM now minus the resident-models footprint this card's
// earlier runs measured, against the working set the run will add) and,
// when it does not fit, shows the numbers with three choices. "Apply levers"
// changes the PANEL CONTROLS (Crop Cap to the value that fits, Async Encode
// off) so what runs is what the panel shows; the caller re-gathers params.
// Resolves {proceed, applied}. Any failure of the check itself = proceed
// (the run-time preflight in the console still prints).
function showVramModal(res) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('vramModal');
    const msgEl = document.getElementById('vramModalMessage');
    const cancelBtn = document.getElementById('vramModalCancel');
    const anywayBtn = document.getElementById('vramModalAnyway');
    const applyBtn = document.getElementById('vramModalApply');
    if (!overlay || !msgEl || !cancelBtn || !anywayBtn || !applyBtn) { resolve('anyway'); return; }
    const applicable = (res.levers || []).filter(l => l.kind === 'crop_cap' || l.kind === 'async_off');
    const advice = (res.levers || []).filter(l => l.kind !== 'crop_cap' && l.kind !== 'async_off');
    const src = res.footprint_source === 'measured'
      ? ('measured on this card\'s last run of this shape' + (res.ledger_when ? ' (' + res.ledger_when + ')' : ''))
      : 'estimated from the edition table (no run of this shape on this card yet)';
    let txt = (res.files_checked > 1 ? ('Largest file in the plan: ' + res.file + ' ') : '')
      + res.width + 'x' + res.height + '. The run may need about ' + res.need_mb + ' MB ('
      + res.parts + ') but about ' + res.free_mb + ' MB will be free once the models are loaded ('
      + res.free_now_mb + ' MB free now, models ' + res.footprint_mb + ' MB, ' + src
      + ') on this ' + res.total_mb + ' MB card. An out-of-memory stop at the first large scene is likely.';
    if (applicable.length) {
      txt += '\n\nApply levers sets: ' + applicable.map(l => l.text).join('; ') + '.';
    }
    if (advice.length) {
      txt += '\n\nFurther levers, by hand: ' + advice.map(l => l.text).join('; ') + '.';
    }
    msgEl.textContent = txt;
    if (!applicable.length) applyBtn.classList.add('hidden'); else applyBtn.classList.remove('hidden');
    overlay.classList.remove('hidden');
    function cleanup(result) {
      overlay.classList.add('hidden');
      cancelBtn.removeEventListener('click', onCancel);
      anywayBtn.removeEventListener('click', onAnyway);
      applyBtn.removeEventListener('click', onApply);
      overlay.removeEventListener('click', onBackdrop);
      resolve(result);
    }
    function onCancel() { cleanup('cancel'); }
    function onAnyway() { cleanup('anyway'); }
    function onApply() { cleanup('apply'); }
    function onBackdrop(e) { if (e.target === overlay) cleanup('cancel'); }
    cancelBtn.addEventListener('click', onCancel);
    anywayBtn.addEventListener('click', onAnyway);
    applyBtn.addEventListener('click', onApply);
    overlay.addEventListener('click', onBackdrop);
  });
}

// T23d (CM-239): one pre-run check in flight at a time, and the button that
// started it says so. The check reads free VRAM from the driver and can take
// a few seconds on a busy card; a second click meanwhile used to start a
// second check (and, with two "Run anyway" clicks, could start two runs).
let _vramCheckInFlight = false;

function _holdButton(btn, text) {
  if (!btn) return { release: function () {} };
  const wasDisabled = !!btn.disabled;
  const wasText = btn.textContent;
  btn.disabled = true;
  try {
    btn.textContent = (window.I18N && typeof I18N.tr === 'function') ? I18N.tr(text) : text;
  } catch (e) { btn.textContent = text; }
  btn.classList.add('is-checking');
  return {
    release: function () {
      btn.textContent = wasText;
      btn.disabled = wasDisabled;
      btn.classList.remove('is-checking');
    },
  };
}

async function checkVramBeforeRun(params, files, btn) {
  if (_vramCheckInFlight) {
    console.log('[VRAM] pre-run check already in flight; this click is ignored');
    return { proceed: false, applied: false, busy: true };
  }
  _vramCheckInFlight = true;
  const held = _holdButton(btn, 'Checking VRAM...');
  try {
    const r = await fetch('/api/vram-preflight', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: params, files: files || null }),
    });
    if (!r.ok) return { proceed: true, applied: false };
    const res = await r.json();
    if (!res || res.skipped || res.fits !== false) {
      if (res && res.skipped) console.log('[VRAM] pre-run check skipped: ' + res.skipped);
      else if (res) console.log('[VRAM] pre-run check: ~' + res.need_mb + ' MB needed, ~' + res.free_mb
        + ' MB free after models (' + res.footprint_source + '), fits');
      return { proceed: true, applied: false };
    }
    console.log('[VRAM] pre-run check: ~' + res.need_mb + ' MB needed vs ~' + res.free_mb
      + ' MB free after models (' + res.footprint_source + ') -- asking');
    // The button reads normally again while the dialog is up (the dialog
    // owns the decision now); the in-flight guard stays until we return.
    held.release();
    const choice = await showVramModal(res);
    if (choice === 'cancel') return { proceed: false, applied: false };
    if (choice === 'apply') {
      let applied = false;
      for (const l of (res.levers || [])) {
        if (l.kind === 'crop_cap') {
          const el = document.getElementById('ctrlMosaicCropCap');
          if (el) {
            el.value = String(l.value);
            if (el.value !== String(l.value)) { el.value = '512'; }
            el.dispatchEvent(new Event('change', { bubbles: true }));
            applied = true;
          }
        } else if (l.kind === 'async_off') {
          const el = document.getElementById('ctrlAsyncEncoder');
          if (el && el.checked) {
            el.checked = false;
            el.dispatchEvent(new Event('change', { bubbles: true }));
            applied = true;
          }
        }
      }
      if (applied && typeof showToast === 'function') {
        showToast('Levers applied to the panel for this run (Crop Cap / Async Encode); save settings to keep them.');
      }
      return { proceed: true, applied: applied };
    }
    return { proceed: true, applied: false };
  } catch (e) {
    console.warn('[VRAM] pre-run check failed: ' + e);
    return { proceed: true, applied: false };
  } finally {
    held.release();
    _vramCheckInFlight = false;
  }
}

// Submit-time gate. Returns {proceed, override} where override marks which
// stages to force onto PyTorch for this run. Mask Preview skips the restoration
// check (pseudo needs no restoration engine).
async function checkTensorBeforeRun() {
  // CM-228: Upscale mode runs no detector and no restorer -- nothing to gate.
  // (T16 returned a bare `true` here; every caller reads `gate.proceed`, so
  // Restore & Save silently bailed. The click-path smoke now covers this.)
  if (_upscaleModeOn()) return { proceed: true, override: null };
  await _refreshEngineCaches();
  const maskPreview = document.getElementById('ctrlMosaicMaskPreview').checked;
  // Add Mosaic (censor) skips restoration entirely, like Preview Detection —
  // don't gate on a restoration engine that will never run.
  const censor = document.getElementById('ctrlMosaicCensor').checked;
  const missing = [];
  if (document.getElementById('ctrlMosaicDetTrt').checked && !_tensorEngineAvailable('det')) {
    missing.push('det');
  }
  if (!maskPreview && !censor && document.getElementById('ctrlMosaicRestTrt').checked
      && !_tensorEngineAvailable('rest')) {
    missing.push('rest');
  }
  // CM-148 (Batch 76): the engine may EXIST but be compiled at a different
  // Image Size than the dial -- the server then silently runs detection on
  // PyTorch. Surface it here, at submit time, like a missing engine.
  let mismatch = null;
  if (!missing.includes('det')
      && document.getElementById('ctrlMosaicDetTrt').checked) {
    const mv = document.getElementById('ctrlMosaicDetModel').value;
    const dial = parseInt(
      (document.getElementById('ctrlMosaicDetImgsz') || {}).value || '640', 10);
    const sizes = (mv && !/\.engine$/i.test(mv)) ? _mosaicDetEngineImgsz[mv] : null;
    if (sizes && sizes.length && Number.isFinite(dial) && !sizes.includes(dial)) {
      // CM-218: no compiled flavor at the dial's size. Offer the nearest one.
      const nearest = sizes.reduce((a, b) => (Math.abs(b - dial) < Math.abs(a - dial) ? b : a), sizes[0]);
      mismatch = { want: dial, have: nearest, all: sizes };
    }
  }

  if (missing.length === 0 && !mismatch) return { proceed: true, override: null };

  if (missing.length > 0) {
    const choice = await showTensorModal(missing);
    if (choice !== 'continue') {
      const b = document.getElementById('manageModelsBtn');
      if (b) b.click();
      return { proceed: false, override: null };
    }
    return {
      proceed: true,
      override: { det: missing.includes('det'), rest: missing.includes('rest') },
    };
  }

  // Pure imgsz mismatch (engine present, wrong size).
  const choice = await showTensorModal([], mismatch);
  if (choice === 'engine-size') {
    // Snap the dial to the engine's compiled size so the run keeps TRT and
    // the UI honestly shows the size that will execute.
    const dialEl = document.getElementById('ctrlMosaicDetImgsz');
    if (dialEl) {
      dialEl.value = String(mismatch.have);
      dialEl.dispatchEvent(new Event('input', { bubbles: true }));
      dialEl.dispatchEvent(new Event('change', { bubbles: true }));
    }
    // Callers gathered params BEFORE this gate ran -- hand the size back so
    // they can patch the already-built payload.
    return { proceed: true, override: null, imgsz: mismatch.have };
  }
  if (choice === 'continue') {
    // Keep the dial size; force detection onto PyTorch for this run so the
    // outcome matches what the user just approved.
    return { proceed: true, override: { det: true, rest: false } };
  }
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
    // (matchModelOptionByName below handles the cross-machine case.)
    if (!select) return;
    const prev = savedValue || select.value;
    // CM-102 (v1.50.00): a fresh install has NO models yet, and a bare
    // "— select model —" over an empty list reads as a broken app (two
    // field users stalled here). When the list is empty, the placeholder
    // itself says what to do; Manage Models is the button top-right.
    if (!items || items.length === 0) {
      // CM-232: Manage Models now has one button for the standard set
      select.innerHTML =
        '<option value="">No models yet \u2014 Manage Models (top right) \u2192 Get the standard models</option>';
      return;
    }
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
    } else if (prev) {
      matchModelOptionByName(select, prev);
    }
  }

  // Cache compiled clip sizes so Max Clip can default/constrain to them.
  _mosaicRestEngines = {};
  for (const item of data.restoration || []) {
    if (item.engines) _mosaicRestEngines[item.path] = item.engines;
  }

  // Cache detection engine availability for the Use Tensor toggle.
  _mosaicDetEngines = {};
  _mosaicDetEngineImgsz = {};  // CM-148 (Batch 76) / CM-218 (list of flavors)
  for (const item of data.detection || []) {
    _mosaicDetEngines[item.path] = !!item.has_engine;
    _mosaicDetEngineImgsz[item.path] = _detEngineSizes(item);
  }

  // Restore the config's saved model selections (see applyConfig): stashed so
  // this survives the startup race with the config load, whichever resolves
  // first. Falls back to the current value if there's no stash.
  //
  // CONSUME-ONCE: the stash must be deleted after use. This function re-runs
  // whenever a compile or download finishes (to pick up new engines/models);
  // if the stash survived, every re-run would snap the dropdowns back to the
  // STARTUP config's models, silently reverting the user's live selection —
  // the next Restore would run the wrong model.
  const _saved = (typeof window !== 'undefined' && window._pendingMosaicModels) || {};
  if (typeof window !== 'undefined') delete window._pendingMosaicModels;
  fill(detSel, data.detection, _saved.det);
  fill(restSel, data.restoration, _saved.rest);
  _mmLastModelData = data;                       // T20b: for the Manage Models button colour
  _updateManageModelsButton();
  updateMaxClipConstraints();
  _updateRestorationButtonStates();
  // Engine availability just refreshed → recompute which controls are inert
  // (e.g. Det FP16 when a compiled engine exists for the selected model).
  _updateControlEnableStates();
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
    // CM-180: the whole control panel (preset schema) rides with the job and
    // lands verbatim in <output stem>.run.json -- "what did I run", answered.
    panel: (typeof gatherFullConfig === 'function') ? gatherFullConfig() : null,
    mosaic: {
      detection_model: document.getElementById('ctrlMosaicDetModel').value,
      restoration_model: document.getElementById('ctrlMosaicRestModel').value,
      mosaic_detection_score: score / 100.0,
      mosaic_iou: parseInt(document.getElementById('ctrlMosaicDetIou').value) / 100.0,
      mosaic_detection_batch_size: parseInt(document.getElementById('ctrlMosaicDetBatch').value),
      mosaic_det_imgsz: parseInt(document.getElementById('ctrlMosaicDetImgsz').value) || 640,
      mosaic_detection_fp16: document.getElementById('ctrlMosaicDetFp16').checked,
      mosaic_detection_trt: document.getElementById('ctrlMosaicDetTrt').checked,
      mosaic_sbs_split: document.getElementById('ctrlMosaicSbsSplit').checked,
      // CM-045: projection is only meaningful with Split SBS on; send "none"
      // otherwise so a stale dropdown can never silently activate it.
      mosaic_vr_projection: (document.getElementById('ctrlMosaicSbsSplit').checked
        ? ((document.getElementById('ctrlMosaicVrProjection') || {}).value || 'none')
        : 'none'),
      mosaic_mask_preview: document.getElementById('ctrlMosaicMaskPreview').checked,
      mosaic_mask_color: document.getElementById('ctrlMosaicMaskColor').value,
      mosaic_mask_opacity: parseInt(document.getElementById('ctrlMosaicMaskOpacity').value) / 100.0,
      mosaic_censor: document.getElementById('ctrlMosaicCensor').checked,
      mosaic_censor_block: parseInt(document.getElementById('ctrlMosaicCensorBlock').value) || 16,
      mosaic_max_clip_size: parseInt(document.getElementById('ctrlMosaicMaxClip').value),
      // CM-213: overlap + crossfade at Max Clip cuts (blend_frames stays auto = overlap / 3)
      mosaic_temporal_overlap: parseInt((document.getElementById('ctrlMosaicOverlap') || {}).value || '0', 10) || 0,
      mosaic_crossfade: !!((document.getElementById('ctrlMosaicCrossfade') || {}).checked),
      mosaic_restoration_fp16: document.getElementById('ctrlMosaicRestFp16').checked,
      mosaic_roi_dilate: parseInt(document.getElementById('ctrlMosaicRoiDilate').value),
      mosaic_crop_cap: parseInt((document.getElementById('ctrlMosaicCropCap') || {}).value || '0', 10) || 0,   // CM-227
      mosaic_feather_radius: parseInt(document.getElementById('ctrlMosaicFeather').value),
      mosaic_blend_mask: document.getElementById('ctrlMosaicBlendMask').value,
      mosaic_use_seg_masks: document.getElementById('ctrlMosaicSegMasks').checked,
      mosaic_residual_scan: document.getElementById('ctrlMosaicResidualScan').checked,   // CM-205
      mosaic_restoration_trt: document.getElementById('ctrlMosaicRestTrt').checked,
      // CM-077: secondary restoration (RTX Super-Res upscale before paste-back)
      mosaic_secondary: ((document.getElementById('ctrlMosaicSecondary') || {}).value || 'none'),
      // CM-146 (Batch 70): Maxine denoise pass chained after the RTX upscale
      mosaic_secondary_denoise: ((document.getElementById('ctrlMosaicSecondaryDenoise') || {}).value || 'none'),
      // CM-217: Maxine Super-Res quality level (RTX modes)
      mosaic_secondary_quality: ((document.getElementById('ctrlMosaicSecondaryQuality') || {}).value || 'high'),
      mosaic_temporal_stability: parseInt((document.getElementById('ctrlMosaicTemporalFix') || {}).value || '0', 10) || 0,
      // CM-084 (Batch 38): FrameStore backend (auto | device | host)
      mosaic_store_backend: ((document.getElementById('ctrlStoreBackend') || {}).value || 'auto'),
      // T20d (CM-234): Hardware decode on the ffmpeg lane (auto | off)
      mosaic_hw_decode: ((document.getElementById('ctrlHwDecode') || {}).value || 'auto'),
      // CM-202 (GitHub #10): output suffix + the run-files switch
      mosaic_output_suffix: (document.getElementById('ctrlOutputSuffix') ? document.getElementById('ctrlOutputSuffix').value : '-restored'),
      mosaic_run_report: ((document.getElementById('ctrlRunReport') || {}).value || ''),
      // CM-219/CM-228: Output Upscale (none | 2x | 4x | 1080p | 1440p | 2160p), Encoder section
      mosaic_output_upscale: ((document.getElementById('ctrlOutputUpscale') || {}).value || 'none'),
      // CM-228: the whole-frame scaler and the upscale-only mode
      mosaic_upscale_scaler: ((document.getElementById('ctrlUpscaleScaler') || {}).value || 'esrgan'),
      mosaic_upscale_only: _upscaleModeOn(),
    },
    encoder: {
      codec: document.getElementById('ctrlCodec').value,
      preset: document.getElementById('ctrlPreset').value,
      qp: parseInt(document.getElementById('ctrlQP').value),
      async_encoder: !!document.getElementById('ctrlAsyncEncoder')?.checked,
    },
  };
}

function buildMosaicParamsSummary() {
  const p = gatherMosaicParams();
  const m = p.mosaic;
  const parts = [];
  if (m.mosaic_upscale_only) {
    // CM-228: no detector, no restorer -- the summary says only what runs
    parts.push('UPSCALE ONLY');
    parts.push(`Enc: ${p.encoder.codec.toUpperCase()}/${p.encoder.preset}/QP${p.encoder.qp}`);
    parts.push((m.mosaic_output_upscale && m.mosaic_output_upscale !== 'none')
      ? `Upscale:${m.mosaic_output_upscale}/${_scalerTag(m.mosaic_upscale_scaler)}`
      : 'Upscale:off (re-encode)');
    return parts.join(' · ');
  }
  const detName = document.getElementById('ctrlMosaicDetModel').selectedOptions[0]?.textContent || '—';
  parts.push(`Det: ${detName}`);
  parts.push(`Score: ${m.mosaic_detection_score.toFixed(2)}`);
  if (m.mosaic_mask_preview) {
    parts.push('MASK PREVIEW');
  } else {
    const restName = document.getElementById('ctrlMosaicRestModel').selectedOptions[0]?.textContent || '—';
    parts.push(`Rest: ${restName}`);
    parts.push(`Clip: ${m.mosaic_max_clip_size}`);
    if (m.mosaic_temporal_overlap > 0) {
      parts.push(`Overlap:${m.mosaic_temporal_overlap}${m.mosaic_crossfade ? '+xf' : ''}`);   // CM-213
    }
    parts.push(m.mosaic_restoration_fp16 ? 'FP16' : 'FP32');
    parts.push(m.mosaic_restoration_trt ? 'TRT' : 'PyTorch');
    // Batch 73: surface the scaler + denoise in the progress-modal summary.
    // Field case 2026-09-01: three denoise A/B runs went out with denoise
    // silently defaulting to none and nothing on screen said so.
    if (m.mosaic_secondary && m.mosaic_secondary !== 'none') {
      parts.push(`Scaler:${m.mosaic_secondary.startsWith('custom:') ? _scalerTag(m.mosaic_secondary) : m.mosaic_secondary}`
        + ((m.mosaic_secondary.startsWith('rtx') && m.mosaic_secondary_quality && m.mosaic_secondary_quality !== 'high')
           ? `/${m.mosaic_secondary_quality}` : '')   // CM-217: only when off the High default
        + ((m.mosaic_secondary_denoise && m.mosaic_secondary_denoise !== 'none')
           ? `+dn:${m.mosaic_secondary_denoise}` : ''));
    }
    if (m.mosaic_roi_dilate > 0) parts.push(`Margin:${m.mosaic_roi_dilate}`);
    if (m.mosaic_crop_cap > 0) parts.push(`CropCap:${m.mosaic_crop_cap}`);   // CM-227: only when on
    if (m.mosaic_blend_mask && m.mosaic_blend_mask !== 'none') parts.push(`Blend:${m.mosaic_blend_mask}`);
    // Feather only takes effect with the facefusion blend mask — only surface
    // it in the summary when it's actually active, so it can't read as "on"
    // under Blend Mask = None.
    if (m.mosaic_feather_radius > 0 && m.mosaic_blend_mask === 'facefusion') {
      parts.push(`Feather:${m.mosaic_feather_radius}`);
    }
  }
  parts.push(`Enc: ${p.encoder.codec.toUpperCase()}/${p.encoder.preset}/QP${p.encoder.qp}`);
  if (m.mosaic_output_upscale && m.mosaic_output_upscale !== 'none' && !m.mosaic_mask_preview && !m.mosaic_censor) {
    parts.push(`Upscale:${m.mosaic_output_upscale}`
      + (m.mosaic_upscale_scaler === 'rtx' ? '/rtx'
         : (String(m.mosaic_upscale_scaler || '').startsWith('custom:') ? '/' + _scalerTag(m.mosaic_upscale_scaler) : '')));   // CM-219/CM-228/CM-229
  }
  return parts.join(' · ');
}


// ── Button enable logic for Restoration mode ──────────────

// CM-229: a short tag for the summary line -- 'rtx', 'esrgan', or the custom file's stem
function _scalerTag(v) {
  const s = String(v || '');
  if (s === 'rtx') return 'rtx';
  if (s.startsWith('custom:')) return s.slice(7).replace(/\.[^.]+$/, '');
  return 'esrgan';
}

function _upscaleModeOn() {
  // CM-228 / T17: the Upscale alternate execution mode (no detector, no restorer)
  return !!document.getElementById('ctrlMosaicUpscale')?.checked;
}

function _updateRestorationButtonStates() {
  const hasVideo = !!state.videoPath;
  const detModel = document.getElementById('ctrlMosaicDetModel').value;
  const restModel = document.getElementById('ctrlMosaicRestModel').value;
  const maskPreview = document.getElementById('ctrlMosaicMaskPreview').checked;
  const censor = document.getElementById('ctrlMosaicCensor').checked;
  const upscale = _upscaleModeOn();
  // Censor and Mask Preview both skip restoration, so they only need a
  // detection model; normal restore needs a restoration model; Upscale
  // (T17) needs no model at all.
  const hasModels = upscale || (detModel && (maskPreview || censor || restModel));
  const hasSegment = state.segmentEndTime > state.segmentStartTime;
  const inPreview = state.previewMode || false;
  // T17: a segment preview on screen can be saved as a file from the & Save button
  const canSaveSegment = inPreview && !!state.previewReady && !!state.previewSegment;

  const restoreBtn = document.getElementById('restoreBtn');
  const restoreSaveBtn = document.getElementById('restoreSaveBtn');
  const restorePreviewBtn = document.getElementById('restorePreviewBtn');
  const testFrameBtn = document.getElementById('restoreDetectBtn');

  // A missing TRT engine no longer blocks Restore — the submit-time modal
  // offers Continue-on-PyTorch or Manage Models. So buttons key off video +
  // models (+ segment) only.
  // Restore / Upscale (segment scope): needs video + models + segment marked.
  if (restoreBtn) {
    restoreBtn.disabled = !hasVideo || !hasModels || !hasSegment || inPreview;
    restoreBtn.title = '';
    restoreBtn.textContent = upscale ? 'Upscale' : 'Restore';   // T17
  }

  // Restore & Save / Upscale & Save (full video): only needs video + models.
  // In a segment preview it becomes Save Segment (T17).
  if (restoreSaveBtn) {
    if (canSaveSegment) {
      restoreSaveBtn.disabled = false;
      restoreSaveBtn.textContent = 'Save Segment';
      restoreSaveBtn.title = 'Save the previewed segment to the output folder (no re-encode); start-end times go into the file name';
    } else {
      restoreSaveBtn.disabled = !hasVideo || !hasModels || inPreview;
      restoreSaveBtn.title = '';
      restoreSaveBtn.textContent = upscale ? 'Upscale & Save' : 'Restore & Save';
    }
  }

  // T17: Test Frame restores one frame -- it means nothing in Upscale mode.
  if (testFrameBtn) {
    const busy = (typeof _foiRunning !== 'undefined') && _foiRunning;
    if (upscale) {
      testFrameBtn.disabled = true;
      testFrameBtn.title = 'Test Frame does not apply in Upscale mode (nothing is detected or restored). Mark a segment and use Upscale, then Preview.';
    } else if (!busy) {
      testFrameBtn.disabled = false;
      testFrameBtn.title = 'Process the current playhead frame with the current settings — restores a short window around it and shows each detected region (Mosaic vs Restored) so you can judge before a full run';
    }
  }

  // Add Mosaic: active whenever a video is loaded; inactive only while an
  // add-mosaic is actively in progress (draw mode, modal, or encoding).
  // No models needed — it's a pure decode->pixelate->encode pass.
  const addMosaicBtn = document.getElementById('addMosaicBtn');
  if (addMosaicBtn) {
    const busy = (typeof _amIsBusy === 'function') ? _amIsBusy() : false;
    addMosaicBtn.disabled = !hasVideo || busy;
    addMosaicBtn.classList.toggle('primary', hasVideo && !busy);
    addMosaicBtn.classList.toggle('secondary', !(hasVideo && !busy));
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

// ── Control enable/disable (gray out knobs that don't apply) ──────────────
// A control that can't affect the current run should be disabled, not left
// looking live. Set the input disabled AND dim its row so it reads as inert.
function _setCtrlEnabled(el, enabled) {
  if (!el) return;
  el.disabled = !enabled;
  const row = el.closest('.ctrl-row');
  if (row) row.style.opacity = enabled ? '' : '0.45';
}

function _updateControlEnableStates() {
  _updateManageModelsButton();   // T20b: Image Size / Use Tensor changed -> the button's colour may
  // Feather only applies with Blend Mask = Face Swap (internal value
  // "facefusion" kept for saved-config compat; the legacy "none" mask
  // has no feather parameter). Gray it out under None so it can't read as on.
  const blend = document.getElementById('ctrlMosaicBlendMask');
  const feather = document.getElementById('ctrlMosaicFeather');
  _setCtrlEnabled(feather, !!(blend && blend.value === 'facefusion'));

  // Mask Colour + Opacity only matter in Preview Detection (pseudo) mode.
  const maskPrev = document.getElementById('ctrlMosaicMaskPreview');
  const previewOn = !!(maskPrev && maskPrev.checked);
  _setCtrlEnabled(document.getElementById('ctrlMosaicMaskColor'), previewOn);
  _setCtrlEnabled(document.getElementById('ctrlMosaicMaskOpacity'), previewOn);

  // Block only matters in Add Mosaic (censor) mode.
  const censorEl = document.getElementById('ctrlMosaicCensor');
  _setCtrlEnabled(document.getElementById('ctrlMosaicCensorBlock'),
                  !!(censorEl && censorEl.checked));

  // Detection FP16 is baked into a compiled detection .engine — AutoBackend
  // reads the engine's binding precision and ignores the constructor flag, so
  // the toggle only affects the PyTorch (.pt) path. Gray it when Use Tensor is
  // on AND an engine exists for the selected model (if no engine, it falls
  // back to .pt where FP16 *does* matter, so leave it live there).
  const detTrt = document.getElementById('ctrlMosaicDetTrt');
  const detModel = document.getElementById('ctrlMosaicDetModel');
  const detFp16 = document.getElementById('ctrlMosaicDetFp16');
  // CM-164: keep the hidden TensorRT toggles unchecked off-CUDA even after a
  // Load Settings that carried them checked from an NVIDIA machine.
  const _edNow = document.documentElement.dataset.edition || '';
  if (_edNow && _edNow !== 'cuda') {
    if (detTrt) detTrt.checked = false;
    const _rt = document.getElementById('ctrlMosaicRestTrt');
    if (_rt) _rt.checked = false;
  }
  const mv = detModel ? (detModel.value || '') : '';
  const detHasEngine = /\.engine$/i.test(mv) || _mosaicDetEngines[mv] === true;
  let detFp16Inert = !!(detTrt && detTrt.checked && detHasEngine);
  // CM-169: on the AMD (ROCm) edition FP16 detection stops detecting after
  // the first frames (measured on an RX 9060 XT, 2026-09-06); the pipeline
  // forces FP32 regardless, so the toggle is shown off and greyed with the
  // reason. Restoration FP16 stays a live choice (its output was clean).
  if (document.documentElement.dataset.edition === 'rocm' && detFp16) {
    detFp16.checked = false;
    detFp16.title = 'Not available on the AMD (ROCm) edition: FP16 detection ' +
      'stops finding anything after the first frames (measured on an RX 9060 XT, ' +
      '2026-09-06). Detection runs in FP32 here; Restoration FP16 is unaffected.';
    detFp16Inert = true;
  }
  _setCtrlEnabled(detFp16, !detFp16Inert);
  // NOTE: Restoration FP16 is intentionally NOT grayed — it selects the
  // fp16-vs-fp32 engine SET (different compiled files), so it has a real
  // effect even with Use Tensor on.

  // CM-045: VR Projection depends on Split SBS (an eye = half the frame).
  // gatherMosaicParams additionally forces "none" when SBS is off, so a
  // stale dropdown can never silently activate projection.
  const sbsEl = document.getElementById('ctrlMosaicSbsSplit');
  const vrpEl = document.getElementById('ctrlMosaicVrProjection');
  const sbsOn = !!(sbsEl && sbsEl.checked);
  _setCtrlEnabled(vrpEl, sbsOn);
  if (vrpEl) vrpEl.title = sbsOn ? '' : 'Enable Split SBS to use VR Projection.';

  _applyUpscaleModeGreying();   // T17: last, so it wins over the per-control rules above
}

// T17 (CM-228): in Upscale mode nothing is detected or restored, so the
// Detection and Restoration sections grey out -- except the rows the scaler
// still reads. Controls greyed here are tagged so leaving the mode restores
// exactly them and nothing the rules above decided.
const _UPSCALE_LIVE_IDS = new Set(['ctrlOutputUpscale', 'ctrlUpscaleScaler',
                                   'ctrlMosaicSecondaryQuality', 'ctrlMosaicSecondaryDenoise']);
function _applyUpscaleModeGreying() {
  const on = _upscaleModeOn();
  ['sectMosaicDetection', 'sectMosaicRestoration'].forEach(sid => {
    const sect = document.getElementById(sid);
    if (!sect) return;
    sect.querySelectorAll('input, select, button').forEach(el => {
      if (!el.id || _UPSCALE_LIVE_IDS.has(el.id)) return;
      if (on) {
        if (!el.disabled) { el.dataset.upscaleGrey = '1'; _setCtrlEnabled(el, false); }
      } else if (el.dataset.upscaleGrey === '1') {
        delete el.dataset.upscaleGrey;
        _setCtrlEnabled(el, true);
      }
    });
  });
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
    if (typeof populateCustomScalers === 'function') populateCustomScalers();   // CM-229: pick up new files
    return result;
  };
  window.loadVideo = loadVideo;
  console.log('[mosaic] loadVideo wrapped');
}

// Alternate Execution Modes are mutually exclusive: Add Mosaic (censor) and
// Preview Detection (mask preview) can't both be on. Checking one clears the
// other, then refresh dependent UI state.
(function _wireModeExclusivity() {
  const censor = document.getElementById('ctrlMosaicCensor');
  const preview = document.getElementById('ctrlMosaicMaskPreview');
  const upscale = document.getElementById('ctrlMosaicUpscale');   // T17
  const refresh = () => { _updateControlEnableStates(); _updateRestorationButtonStates(); };
  if (censor) censor.addEventListener('change', () => {
    if (censor.checked) { if (preview) preview.checked = false; if (upscale) upscale.checked = false; }
    refresh();
  });
  if (preview) preview.addEventListener('change', () => {
    if (preview.checked) { if (censor) censor.checked = false; if (upscale) upscale.checked = false; }
    refresh();
  });
  if (upscale) upscale.addEventListener('change', () => {
    if (upscale.checked) { if (censor) censor.checked = false; if (preview) preview.checked = false; }
    refresh();
  });
})();


// Re-check on relevant changes
['ctrlMosaicDetModel', 'ctrlMosaicRestModel', 'ctrlMosaicMaskPreview',
 'ctrlMosaicCensor', 'ctrlMosaicUpscale'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', _updateRestorationButtonStates);
});

// Controls whose value changes which OTHER controls are inert → recompute
// enable/disable state. (Blend Mask → Feather; Mask Preview → colour/opacity;
// Add Mosaic → Block; Det Use Tensor / Det Model → Det FP16.)
['ctrlMosaicBlendMask', 'ctrlMosaicMaskPreview', 'ctrlMosaicCensor', 'ctrlMosaicUpscale',
 'ctrlMosaicDetTrt', 'ctrlMosaicDetModel', 'ctrlMosaicSbsSplit',
 'ctrlMosaicDetImgsz'].forEach(id => {       // T20b: Image Size decides the Manage Models button colour
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', _updateControlEnableStates);
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

// CM-164: TensorRT exists only on the NVIDIA edition. On AMD (ROCm) and
// Intel (XPU) the two "Use Tensor" rows were shown checked and meant nothing
// (the pipeline fell through to PyTorch and said so in the console). Hide the
// rows and send them unchecked so the request line and the misses JSON tell
// the truth. <html data-edition> is set by the server from torch.version.
(function _hideTensorRtOffCuda() {
  const ed = document.documentElement.dataset.edition || '';
  if (!ed || ed === 'cuda') return;
  [_detTrtEl, _restTrtEl].forEach(el => {
    if (!el) return;
    el.checked = false;
    const row = el.closest('.ctrl-row');
    if (row) row.style.display = 'none';
  });
  // CM-228: Maxine is NVIDIA-only -- off-CUDA the Upscale Scaler has one
  // honest choice; drop the RTX option so a loaded settings file cannot
  // select it (the pipeline would fall back and say so, but why ask).
  const sc = document.getElementById('ctrlUpscaleScaler');
  if (sc) {
    Array.from(sc.options).forEach(o => { if (o.value === 'rtx') o.remove(); });
    sc.value = 'esrgan';
  }
  // T20b: the same for the Secondary -- RTX Super-Res 2x/4x cannot exist off-CUDA, and
  // the recommended default is now a scaler, so the markup's rtx-4x becomes esrgan-4x
  // here (the server's default-config route hands out esrgan-4x for this edition too).
  const sec = document.getElementById('ctrlMosaicSecondary');
  if (sec) {
    const wasRtx = String(sec.value).startsWith('rtx');
    Array.from(sec.options).forEach(o => { if (String(o.value).startsWith('rtx')) o.remove(); });
    if (wasRtx || !sec.value) sec.value = 'esrgan-4x';
  }
  // T20a/T20b: Manage Models' compile section is TensorRT, i.e. NVIDIA-only. Off-CUDA
  // a press launched -compile-det and died in the log. Greyed, not hidden (Gman,
  // 09-23): the section exists and is not yours yet; the note under it says why.
  const cs = document.getElementById('mmCompileSection');
  if (cs) {
    cs.classList.add('mm-inert');
    cs.querySelectorAll('input, button, select').forEach(el => { el.disabled = true; });
  }
  const nc = document.getElementById('mmNoCompileNote');
  if (nc) nc.classList.remove('mm-hidden');
})();


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
    // CM-135: current + average fps, both measured over COMPLETED frames
    // (restored + passthrough), so the number is delivered speed, not the
    // decode/detect front racing ahead of restoration. 0 renders as 0.0
    // (a window with no completions is real on chunked backends), and a
    // numeric-always display keeps the row width stable (Batch 69 wobble
    // fix, together with the wider modal + tabular digits in ui.html).
    const _fpsTxt = (v) => (v == null ? '—' : Number(v).toFixed(1));
    // Batch 77: the current value is the last delivery-cycle rate, HELD
    // between clip deliveries (chunked TRT delivers in bursts; a per-emit
    // window read 0.0 almost always). When the held value is older than a
    // few seconds, mark it as refreshing so a long clip in progress (or a
    // genuine stall) is visible without lying about the number.
    const _stale = Number(prog.fps_stale || 0) > 5 ? '…' : '';
    progressFps.textContent =
      `${_fpsTxt(prog.fps)}${_stale} fps (avg ${_fpsTxt(prog.fps_avg)})`;
    const det = prog.detections || 0;
    const res = prog.restorations || 0;
    const buf = prog.buffered || 0;
    progressEta.textContent =
      `ETA: ${prog.eta || '—'} | ${det} det, ${res} res, buf=${buf}`;

    if (prog.status === 'complete') {
      // Batch 71: honest FINAL readout. The last live window is often
      // legitimately 0.0 fps (the tail frames were delivered before the
      // final poll), so the closing display substitutes the run average
      // as the headline number when current is zero/absent. The ETA is
      // meaningless on a finished run and was going stale ("ETA: 14s"
      // on a done job, field 2026-09-01) -- replace it with the final
      // counts only.
      const _fin = (Number(prog.fps) > 0) ? prog.fps : prog.fps_avg;
      progressFps.textContent =
        `${_fpsTxt(_fin)} fps (avg ${_fpsTxt(prog.fps_avg)})`;
      progressEta.textContent = `${det} det, ${res} res`;
      onComplete(prog);
    } else if (prog.status === 'error' || prog.status === 'cancelled' ||
               prog.status === 'partial') {
      // CM-107 (Batch 50): 'partial' = the run errored but a playable
      // partial output was saved. Routed through onError; handlers branch
      // on prog.status to present it honestly.
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
  // CM-236: the VRAM preflight, before the run (levers change the panel; re-gather).
  const vgate = await checkVramBeforeRun(params, null, document.getElementById('restoreBtn'));
  if (!vgate.proceed) { if (!vgate.busy) _updateRestorationButtonStates(); return; }   // T23d: busy = a check is already running; leave its button alone
  if (vgate.applied) { const _re = gatherMosaicParams(); params.mosaic = _re.mosaic; params.encoder = _re.encoder; params.panel = _re.panel; }
  if (gate.override) {
    if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
    if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
  }
  // CM-148 (Batch 76): user chose "Use engine size" -- run at the engine's
  // compiled Image Size (params were gathered before the gate snapped the dial).
  if (gate.imgsz) {
    params.mosaic.mosaic_det_imgsz = gate.imgsz;
    // CM-196 report nit: the panel copy in the run report must show the size that ran.
    if (params.panel && typeof params.panel === 'object') params.panel.ctrlMosaicDetImgsz = String(gate.imgsz);
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
      state.previewSegment = { start: startTime, end: endTime };   // T17: Save Segment applies
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
        state.previewSegment = { start: startTime, end: endTime, partial: true };
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
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled' || prog.status === 'partial')) {
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

  // T17: while watching a SEGMENT preview this button is "Save Segment" --
  // copy the previewed file into the output folder, no re-encode.
  if (state.previewMode && state.previewReady && state.previewSegment) {
    await _saveSegment();
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
  // CM-236: the VRAM preflight, before the run (levers change the panel; re-gather).
  const vgate = await checkVramBeforeRun(params, null, document.getElementById('restoreSaveBtn'));
  if (!vgate.proceed) { if (!vgate.busy) _updateRestorationButtonStates(); return; }   // T23d: busy = a check is already running; leave its button alone
  if (vgate.applied) { const _re = gatherMosaicParams(); params.mosaic = _re.mosaic; params.encoder = _re.encoder; params.panel = _re.panel; }
  if (gate.override) {
    if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
    if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
  }
  // CM-148 (Batch 76): user chose "Use engine size" -- run at the engine's
  // compiled Image Size (params were gathered before the gate snapped the dial).
  if (gate.imgsz) {
    params.mosaic.mosaic_det_imgsz = gate.imgsz;
    // CM-196 report nit: the panel copy in the run report must show the size that ran.
    if (params.panel && typeof params.panel === 'object') params.panel.ctrlMosaicDetImgsz = String(gate.imgsz);
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
      // T20c: a completed run with warnings (CM-186 guard hits, stabilizer windows
      // skipped) says so here -- "Complete!" over 401 refused frames was a lie (09-23)
      const _warns = Array.isArray(prog.warnings) ? prog.warnings : [];
      progressTitle.textContent = _warns.length ? `Complete \u2014 with ${_warns.length} warning${_warns.length > 1 ? 's' : ''}` : 'Complete!';
      progressPercent.textContent =
        `Done \u2014 ${prog.frame} frames, ${prog.detections || 0} det, ${prog.restorations || 0} res`
        + (_warns.length ? '\n\u26a0 ' + _warns.join('\n\u26a0 ') : '');
      if (_warns.length) { progressPercent.style.whiteSpace = 'pre-line'; progressPercent.style.textAlign = 'left'; }
      progressBar.style.width = '100%';
      progressCancel.textContent = 'Close';
      // Server pointed preview at the full output — enable Preview so clicking
      // it plays the completed file (same mechanism as segment preview).
      state.previewReady = true;
      state.previewSegment = null;          // T17: already saved; no Save Segment
      _updateRestorationButtonStates();
      console.log('[ChitraMaya] Restore & Save complete:', prog);
    },
    onError: (prog) => {
      clearInterval(pollInterval);
      if (prog.status === 'partial') {
        // CM-107 (Batch 50): the run errored, but the output file holds
        // everything encoded up to the failure and is playable. Field
        // case 2026-08-19: 135,584 frames / 37 minutes survived an NVENC
        // death and the UI said only "Error". Keep the bar at the real
        // percentage -- the work shown was NOT lost.
        const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;
        progressTitle.textContent =
          `Partial result — ${prog.frame} of ${prog.total} frames saved`;
        progressPercent.textContent = prog.error || 'Partial output saved.';
        progressBar.style.width = pct + '%';
      } else {
        progressTitle.textContent = prog.status === 'error' ? 'Error' : 'Cancelled';
        progressPercent.textContent = prog.error || 'Processing failed';
      }
      progressCancel.textContent = 'Close';
      _updateRestorationButtonStates();
    },
  });

  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled' || prog.status === 'partial')) {
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
      _updateRestorationButtonStates();
    } else {
      await apiPost('/api/cancel');
    }
  };
});


// ── Batch Folder (CM-079 / Batch 23 queue UI) ─────────────
// Folder picker + dual-list queue builder + live status board. The modal
// calls /api/mosaic-folder-plan to see exactly what the batch planner sees
// (own outputs excluded, output-exists annotated), the user builds the queue
// by moving files, and the run view turns that queue into per-file status
// rows driven by /api/progress's batch block.
(function _wireBatchFolder() {
  const openBtn   = document.getElementById('processFolderBtn');
  const modal     = document.getElementById('batchModal');
  if (!openBtn || !modal) return;

  const setupView  = document.getElementById('batchSetup');
  const runView    = document.getElementById('batchRun');
  const folderEl   = document.getElementById('batchFolder');
  const browseBtn  = document.getElementById('batchBrowseBtn');
  const rescanBtn  = document.getElementById('batchRescanBtn');
  const recurEl    = document.getElementById('batchRecursive');
  const leftList   = document.getElementById('bmLeftList');
  const rightList  = document.getElementById('bmRightList');
  const leftCount  = document.getElementById('bmLeftCount');
  const rightCount = document.getElementById('bmRightCount');
  const addBtn     = document.getElementById('bmAddBtn');
  const addAllBtn  = document.getElementById('bmAddAllBtn');
  const removeBtn  = document.getElementById('bmRemoveBtn');
  const startBtn   = document.getElementById('batchStartBtn');
  const cancelBtn  = document.getElementById('batchCancelBtn');
  const stopBtn    = document.getElementById('batchStopBtn');
  const queueEl    = document.getElementById('bmQueueStatus');
  const barEl      = document.getElementById('bmProgressBar');
  const pctEl      = document.getElementById('bmProgressPct');
  const fpsEl      = document.getElementById('bmProgressFps');
  const sumEl      = document.getElementById('bmSummary');

  let left = [];    // [{name, path, output_name, output_exists}] still in folder
  let right = [];   // the queue, in processing order
  let poll = null;

  function _fmtSecs(s) {
    s = Math.round(s);
    const m = Math.floor(s / 60), r = s % 60;
    return m ? `${m}m ${r}s` : `${r}s`;
  }

  function renderLists() {
    leftList.innerHTML = '';
    left.forEach((f, i) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = f.output_exists ? `${f.name}   [output exists]` : f.name;
      if (f.output_exists) o.classList.add('bm-exists');
      o.title = f.path;
      leftList.appendChild(o);
    });
    rightList.innerHTML = '';
    right.forEach((f, i) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = f.output_exists ? `${f.name}   [re-process -> -2]` : f.name;
      o.title = f.path;
      rightList.appendChild(o);
    });
    leftCount.textContent = String(left.length);
    rightCount.textContent = String(right.length);
    startBtn.disabled = right.length === 0;
  }

  async function rescan() {
    const folder = (folderEl.value || '').trim();
    if (!folder) { left = []; right = []; renderLists(); return; }
    const params = gatherMosaicParams();
    params.folder = folder;
    params.recursive = recurEl.checked;
    const res = await apiPost('/api/mosaic-folder-plan', { params });
    if (res.error) { alert('Scan failed: ' + res.error); return; }
    const files = res.files || [];
    // Already-processed files start on the left (visible, grey); fresh ones
    // go straight into the queue. This replaces the old invisible
    // skip-existing checkbox with something you can see and override.
    left  = files.filter(f => f.output_exists);
    right = files.filter(f => !f.output_exists);
    renderLists();
  }

  function moveSelected(fromList, fromArr, toArr) {
    const idxs = Array.from(fromList.selectedOptions).map(o => parseInt(o.value, 10));
    if (!idxs.length) return;
    const moved = idxs.map(i => fromArr[i]);
    idxs.sort((a, b) => b - a).forEach(i => fromArr.splice(i, 1));
    moved.forEach(f => { if (!toArr.some(g => g.path === f.path)) toArr.push(f); });
    renderLists();
  }

  openBtn.addEventListener('click', () => {
    modal.classList.remove('hidden');
    setupView.style.display = '';
    runView.style.display = 'none';
    if ((folderEl.value || '').trim()) rescan(); else folderEl.focus();
  });
  cancelBtn.addEventListener('click', () => modal.classList.add('hidden'));
  browseBtn.addEventListener('click', async () => {
    const folder = await selectFolder();
    if (folder) { folderEl.value = folder; await rescan(); }
  });
  rescanBtn.addEventListener('click', rescan);
  folderEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') rescan(); });
  recurEl.addEventListener('change', rescan);
  addBtn.addEventListener('click', () => moveSelected(leftList, left, right));
  removeBtn.addEventListener('click', () => moveSelected(rightList, right, left));
  addAllBtn.addEventListener('click', () => {
    left.forEach(f => { if (!right.some(g => g.path === f.path)) right.push(f); });
    left = [];
    renderLists();
  });
  leftList.addEventListener('dblclick', () => moveSelected(leftList, left, right));
  rightList.addEventListener('dblclick', () => moveSelected(rightList, right, left));

  function buildQueueRows(files) {
    queueEl.innerHTML = '';
    files.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'bm-qrow bm-q-pending';
      const nm = document.createElement('span');
      nm.className = 'bm-qname'; nm.textContent = f.name; nm.title = f.path || f.name;
      const st = document.createElement('span');
      st.className = 'bm-qstate'; st.textContent = 'queued';
      row.appendChild(nm); row.appendChild(st);
      queueEl.appendChild(row);
    });
  }

  function setRow(i, cls, text, titleText) {
    const row = queueEl.children[i];
    if (!row) return;
    row.className = 'bm-qrow bm-q-' + cls;
    const st = row.querySelector('.bm-qstate');
    if (st) { st.textContent = text; if (titleText) st.title = titleText; }
  }

  startBtn.addEventListener('click', async () => {
    if (!right.length) return;
    const params = gatherMosaicParams();
    params.folder = (folderEl.value || '').trim();
    params.files = right.map(f => f.path);   // explicit queue, in order

    // Same TRT-engine gate as the single-file runs.
    const gate = await checkTensorBeforeRun();
    if (!gate.proceed) { return; }
    // CM-236: the VRAM preflight over the whole plan (largest file decides).
    const vgate = await checkVramBeforeRun(params, params.files, startBtn);
    if (!vgate.proceed) { return; }
    if (vgate.applied) {
      const _re = gatherMosaicParams();
      params.mosaic = _re.mosaic; params.encoder = _re.encoder; params.panel = _re.panel;
    }
    if (gate.override) {
      if (gate.override.det) params.mosaic.mosaic_detection_trt = false;
      if (gate.override.rest) params.mosaic.mosaic_restoration_trt = false;
    }
    // CM-148 (Batch 76): honor "Use engine size" (see segment handler).
    if (gate.imgsz) {
      params.mosaic.mosaic_det_imgsz = gate.imgsz;
      // CM-196 report nit: the panel copy in the run report must show the size that ran.
      if (params.panel && typeof params.panel === 'object') params.panel.ctrlMosaicDetImgsz = String(gate.imgsz);
    }

    const result = await apiPost('/api/mosaic-folder', { params });
    if (result.error) { alert('Failed to start: ' + result.error); return; }

    // Switch the modal to the run view: the queue becomes the status board.
    setupView.style.display = 'none';
    runView.style.display = '';
    stopBtn.textContent = 'Stop';
    buildQueueRows(right);
    barEl.style.width = '0%';
    pctEl.textContent = '0%';
    fpsEl.textContent = '— fps';
    sumEl.textContent = '';

    poll = setInterval(async () => {
      const prog = await apiGet('/api/progress');
      if (!prog || (prog.error && !prog.status)) return;
      const b = prog.batch || {};
      const files = b.files || [];
      const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;

      files.forEach((f, i) => {
        if (f.status === 'processing') {
          setRow(i, 'processing', `${pct}% (${prog.frame}/${prog.total}) @ ${prog.fps || '--'} fps`);
        } else if (f.status === 'done') {
          const secs = f.seconds ? ` in ${_fmtSecs(f.seconds)}` : '';
          setRow(i, 'done', `done${secs}`);
        } else if (f.status === 'skipped') {
          setRow(i, 'skipped', `skipped (${f.error || 'output exists'})`);
        } else if (f.status === 'error') {
          setRow(i, 'error', 'FAILED -- hover for detail', f.error || '');
        } else if (f.status === 'cancelled') {
          setRow(i, 'cancelled', 'cancelled');
        } else {
          setRow(i, 'pending', 'queued');
        }
      });

      barEl.style.width = pct + '%';
      pctEl.textContent = `File ${b.index || 0}/${b.total || files.length} — ${pct}%`;
      fpsEl.textContent = `${prog.fps || '—'} fps`;
      sumEl.textContent =
        `${b.done || 0} done, ${b.skipped || 0} skipped, ${b.errors || 0} error(s)`;

      if (prog.status === 'complete' || prog.status === 'cancelled' || prog.status === 'error') {
        clearInterval(poll);
        poll = null;
        barEl.style.width = '100%';
        pctEl.textContent =
          prog.status === 'complete' ? 'Batch complete' :
          prog.status === 'cancelled' ? 'Batch cancelled' : 'Batch error';
        sumEl.textContent = b.summary || prog.error || sumEl.textContent;
        stopBtn.textContent = 'Close';
        _updateRestorationButtonStates();
      }
    }, 500);

    stopBtn.onclick = async () => {
      if (!poll) {  // terminal state -- Close resets the modal for next time
        modal.classList.add('hidden');
        setupView.style.display = '';
        runView.style.display = 'none';
        return;
      }
      // Cancel stops the batch between files (the in-flight file finishes).
      await apiPost('/api/cancel');
      stopBtn.textContent = 'Stopping after current file...';
    };
  });
})();


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
    _updateRestorationButtonStates();   // T17: & Save -> Save Segment
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
    _updateRestorationButtonStates();   // T17: back to Restore/Upscale & Save
    console.log('[ChitraMaya] Exited restoration Preview mode');
  }
});

// T17: Save Segment -- the previewed segment file goes to the output folder
// under <stem><suffix>-<start>-<end>.mp4; the server copies, no re-encode.
async function _saveSegment() {
  const seg = state.previewSegment;
  if (!seg) return;
  const btn = document.getElementById('restoreSaveBtn');
  const oldText = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
  try {
    const params = gatherMosaicParams();
    const res = await apiPost('/api/save-segment', {
      start_time: seg.start, end_time: seg.end,
      output_dir: params.output_dir || '',
      params,
    });
    if (res && res.status === 'ok') {
      console.log('[ChitraMaya] Segment saved:', res.output);
      alert(`Saved ${res.name}\nin ${res.folder}`);
    } else {
      alert('Could not save the segment: ' + ((res && (res.error || res.message)) || 'unknown error'));
    }
  } catch (e) {
    alert('Could not save the segment: ' + e);
  } finally {
    if (btn) { btn.textContent = oldText; }
    _updateRestorationButtonStates();
  }
}


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
  (data.detection || []).forEach(m => {
    // CM-203: say what the engine on disk accepts. The run clamps the
    // Detection Batch to this; the badge is where the user learns it.
    let detail = '';
    if (m.has_engine) {
      // CM-218: one badge entry per compiled flavor: imgsz/bN (fp32, * = legacy file).
      if (Array.isArray(m.engine_flavors) && m.engine_flavors.length) {
        detail = m.engine_flavors.map(f => f.label).join(', ');
      } else {
        const parts = [];
        if (m.engine_imgsz) parts.push('imgsz ' + m.engine_imgsz);
        if (m.engine_max_batch) parts.push('batch 1..' + m.engine_max_batch);
        detail = parts.join(', ');
      }
    }
    _mmModels.push({ path: m.path, label: m.label, kind: 'det',
                     compiled: !!m.has_engine, detail });
  });
  (data.restoration || []).forEach(m => {
    // v1.60 (CM-104): a usable set serves ANY clip length. Badge wording:
    // canonical fixed-batch set = "any clip length"; legacy ladder files =
    // still usable, but flag that one recompile shrinks them.
    const usable = !!(m.usable && (m.usable.fp16 || m.usable.fp32));
    const fixed = !!(m.fixed && (m.fixed.fp16 || m.fixed.fp32));
    const sizes = [...new Set([...((m.engines && m.engines.fp16) || []),
                               ...((m.engines && m.engines.fp32) || [])])].sort((a, b) => a - b);
    let detail = '';
    if (usable) {
      detail = fixed ? 'any clip length'
                     : ('legacy b' + sizes.join(',') + ' - recompile to shrink');
    }
    _mmModels.push({ path: m.path, label: m.label, kind: 'rest',
                     compiled: usable, detail });
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
  // T20a: off-CUDA there is nothing to compile -- "Not compiled" read as a fault
  const _edMM = document.documentElement.dataset.edition || '';
  const badge = (ok, detail) =>
    (_edMM && _edMM !== 'cuda')
      ? '<span class="mm-badge mm-ok">PyTorch (this edition)</span>'
      : `<span class="mm-badge ${ok ? 'mm-ok' : 'mm-no'}">`
        + (ok ? ('Compiled' + (detail ? ' \u00b7 ' + detail : '')) : 'Not compiled') + '</span>';
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
  // T20b: off-CUDA the whole compile section is inert; the button stays disabled
  const _cs = document.getElementById('mmCompileSection');
  const _inert = !!(_cs && _cs.classList.contains('mm-inert'));
  if (btn && !btn.classList.contains('mm-compiling')) {
    btn.disabled = _inert || n === 0;
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
      _mmSelected.clear();                     // T20b: every compile starts from an empty pick
      mmPopulateList();                        // Manage Models badges refresh
      if (typeof populateMosaicModelDropdowns === 'function') {                                  // Control Panel dropdowns + engine caches
        Promise.resolve(populateMosaicModelDropdowns()).catch(() => {})
          .then(() => { if (typeof populateCustomScalers === 'function') populateCustomScalers(); });   // CM-229, after
      }
    }
  }, 1000);
}

async function mmCompile() {
  if (_mmSelected.size === 0) return;
  const log = document.getElementById('mmLog');
  // Compile and download share the single #mmLog pane and the GPU; don't let
  // them run at once (their logs would overwrite each other tick-by-tick).
  if (_mmDlPoll) {
    if (log) log.textContent = 'A download is running — wait for it to finish before compiling.';
    return;
  }
  const imgsz = parseInt(document.getElementById('mmImgsz').value, 10) || 640;
  // v1.60 (CM-104): restoration engines are clip-size independent; the
  // Manage Models Max Clip slider is gone. max_clip is kept in the API
  // for compatibility and ignored by the engine layer.
  const maxClip = 60;
  const force = document.getElementById('mmForce').checked;
  _mmSetCompiling(true);
  if (log) log.textContent = 'Starting…';
  // CM-201 / T20a: the detector engine's batch profile is the modal's own
  // Detection Batch slider (2..16). It USED to follow the panel's run-time
  // slider silently -- a 640/b4 engine came out of a compile the user
  // expected to be b8 (clean-machine test, 09-23). Nothing hidden now.
  const maxBatch = _mmDetBatchValue();
  const res = await apiPost('/api/compile-engines',
    { models: [..._mmSelected], imgsz, max_clip: maxClip, force, max_batch: maxBatch });
  if (!res || res.error) {
    if (log) log.textContent = 'Error: ' + (res ? res.error : 'no response');
    _mmSetCompiling(false);
    return;
  }
  _mmStartPolling();
}

async function openManageModels() {
  mmSyncDetBatchHint();   // CM-203: settings may have been loaded after the dial listener
  const modal = document.getElementById('manageModelsModal');
  if (!modal) return;
  modal.classList.remove('hidden');
  await mmLoadSources();
  await mmPopulateList();
  await mmUpPopulate();          // CM-232: custom scalers
  await mmStdRefresh();          // CM-232: standard set status
  // Resume whichever background job is still running server-side (either, not
  // both — they can't run concurrently). Standard job first (it owns the other
  // two while it runs), then compile, then download.
  const ss = await apiGet('/api/standard-models');
  if (ss && ss.job && ss.job.running) { _mmStdStartPolling(); return; }
  const cs = await apiGet('/api/compile-log');
  if (cs && cs.running) {
    const log = document.getElementById('mmLog');
    if (log) log.textContent = cs.log || '';
    _mmStartPolling();
  } else {
    const ds = await apiGet('/api/download-log');
    if (ds && ds.running) {
      const log = document.getElementById('mmLog');
      if (log) log.textContent = ds.log || '';
      _mmStartDlPolling();
    }
  }
}

// CM-109 (Batch 50, GenSRT precedent): the compile log is THE artifact users
// paste for support (field: the dual-GPU compile forensics ran on phone
// screenshots because the pane could not be copied). One click copies the
// whole pane. Clipboard API first (127.0.0.1 is a secure context in
// WebView2/Chromium); hidden-textarea execCommand fallback for older
// runtimes. The pane itself is also selectable now (ui.html user-select).
(function _mmWireLogCopy() {
  const btn = document.getElementById('mmLogCopy');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const log = document.getElementById('mmLog');
    const text = log ? log.textContent : '';
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch (e) {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
      } catch (e2) { ok = false; }
    }
    const prev = 'Copy';
    btn.textContent = ok ? 'Copied' : 'Copy failed';
    setTimeout(() => { btn.textContent = prev; }, 1500);
  });
})();

function closeManageModels() {
  const modal = document.getElementById('manageModelsModal');
  if (modal) modal.classList.add('hidden');
  // Stop BOTH polls (the jobs keep running server-side; openManageModels
  // resumes whichever is still active). Previously only _mmPoll was cleared,
  // so a download poll kept firing every 800ms against the hidden modal.
  if (_mmPoll) { clearInterval(_mmPoll); _mmPoll = null; }
  if (_mmDlPoll) { clearInterval(_mmDlPoll); _mmDlPoll = null; }
}

// Issue #3: Escape closes Manage Models (any in-progress compile/download
// keeps running server-side and resumes on reopen). Only fires while the
// modal is open, so it never steals Escape from other contexts.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  const modal = document.getElementById('manageModelsModal');
  if (modal && !modal.classList.contains('hidden')) {
    e.preventDefault();
    closeManageModels();
  }
});

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
    // CM-199 (GitHub #13): the download endpoint rides with the request.
    const _ep = (document.getElementById('mmEndpoint') || {}).value || '';
    const res = await apiPost('/api/fetch-model-list', { url: _mmCurrentRepoUrl, endpoint: _ep });
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

function _mmStartDlPolling() {
  const log = document.getElementById('mmLog');
  if (_mmDlPoll) clearInterval(_mmDlPoll);
  _mmSetDownloading(true);
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

async function mmDownload() {
  if (_mmDlSelected.size === 0) return;
  const log = document.getElementById('mmLog');
  // Shared #mmLog / GPU — don't run alongside a compile.
  if (_mmPoll) {
    if (log) log.textContent = 'A compile is running — wait for it to finish before downloading.';
    return;
  }
  _mmSetDownloading(true);
  if (log) log.textContent = 'Starting download…';
  const res = await apiPost('/api/download-models',
    { url: _mmCurrentRepoUrl, files: [..._mmDlSelected],
      endpoint: ((document.getElementById('mmEndpoint') || {}).value || '') });   // CM-199
  if (!res || res.error) {
    if (log) log.textContent = 'Error: ' + (res ? res.error : 'no response');
    _mmSetDownloading(false);
    return;
  }
  _mmStartDlPolling();
}

// Slider readouts — self-contained (kept out of the config system).
// (mmMaxClip removed in v1.60 — engines are clip-size independent; the
// forEach tolerates the missing element either way.)
['mmImgsz', 'mmMaxClip', 'mmDetBatch'].forEach(id => {
  const dial = document.getElementById(id);
  const val = document.getElementById(id + 'Val');
  if (dial && val) { const upd = () => { val.textContent = dial.value; }; dial.addEventListener('input', upd); upd(); }
});
// CM-232: the standard-set status is per flavor (Image Size x Detection Batch)
(function _wireStdFlavorRefresh() {
  const modal = document.getElementById('manageModelsModal');
  const refresh = () => { if (modal && !modal.classList.contains('hidden') && typeof mmStdRefresh === 'function') mmStdRefresh(); };
  ['mmImgsz', 'mmDetBatch'].forEach(id => { const el = document.getElementById(id); if (el) el.addEventListener('change', refresh); });
})();

// CM-203: the compile section shows the batch the next detector engine
// will be built for -- the panel's Detection Batch, live.
function mmSyncDetBatchHint() {
  // T20a: the modal has its own Detection Batch slider (mmDetBatch); the
  // panel's run-time slider no longer drives the compile. Kept as a no-op
  // for the callers that still name it.
}

// ── CM-232: Standard models (one button) ───────────────────
// Four files every install needs; on NVIDIA their engines too. The server
// sequences the existing download + compile jobs; we poll /api/standard-models
// for step / stage / combined log and show Done as a health check.
let _mmStdPoll = null;
function _mmDetBatchValue() {
  const el = document.getElementById('mmDetBatch');
  return Math.max(1, parseInt(el ? el.value : '8', 10) || 8);
}
function _mmStdFlavor() {
  const imgsz = parseInt((document.getElementById('mmImgsz') || {}).value || '640', 10) || 640;
  return { imgsz, max_batch: _mmDetBatchValue() };
}
function _mmStdRender(st) {
  const list = document.getElementById('mmStdList');
  const status = document.getElementById('mmStdStatus');
  const btn = document.getElementById('mmStdRun');
  if (!list || !st) return;
  const compiles = !!st.compiles;
  list.innerHTML = (st.models || []).map(m => {
    let badge;
    if (!m.present) badge = '<span class="mm-badge mm-no">Not downloaded</span>';
    else if (!compiles) badge = '<span class="mm-badge mm-ok">Downloaded</span>';
    else if (m.compiled) badge = '<span class="mm-badge mm-ok">Compiled' + (m.flavors && m.flavors.length ? ' · ' + m.flavors.join(', ') : '') + '</span>';
    else badge = '<span class="mm-badge mm-no">Downloaded · not compiled for this flavor</span>';
    return `<div class="mm-row"><span class="mm-name">${m.label} <span class="mm-ext">${m.file}</span></span>${badge}</div>`;
  }).join('');
  const job = st.job || {};
  const done = st.all_present && (!compiles || st.all_compiled);
  if (btn) {
    btn.disabled = !!job.running;
    btn.textContent = job.running ? `\u23f3 ${job.step}/${job.total}\u2026`
                    : (done ? '\u2713 Done \u2014 check again' : 'Get the standard models');
  }
  if (status) {
    if (job.running) status.textContent = job.stage || 'working\u2026';
    else if (done) status.textContent = compiles
      ? `All four present and compiled for ${_mmStdFlavor().imgsz}/b${_mmStdFlavor().max_batch}.`
      : 'All four present (no engines on this edition).';
    else if (job.done === false) status.textContent = job.stage || 'finished with problems (see the log)';
    else {
      const miss = (st.models || []).filter(m => !m.present).length;
      const unc = (st.models || []).filter(m => m.present && !m.compiled).length;
      status.textContent = (miss ? `${miss} to download` : '') + (miss && compiles && unc ? ', ' : '')
        + (compiles && unc ? `${unc} to compile for ${_mmStdFlavor().imgsz}/b${_mmStdFlavor().max_batch}` : '');
    }
  }
}
async function mmStdRefresh() {
  const f = _mmStdFlavor();
  const st = await apiGet(`/api/standard-models?imgsz=${f.imgsz}&max_batch=${f.max_batch}`);
  if (!st || st.error) { const l = document.getElementById('mmStdList'); if (l) l.textContent = 'Could not check the standard models.'; return null; }
  _mmStdRender(st);
  return st;
}
function _mmStdStartPolling() {
  const log = document.getElementById('mmLog');
  if (_mmStdPoll) clearInterval(_mmStdPoll);
  _mmStdPoll = setInterval(async () => {
    const st = await mmStdRefresh();
    if (!st) return;
    if (log && st.log) { log.textContent = st.log; log.scrollTop = log.scrollHeight; }
    if (!(st.job && st.job.running)) {
      clearInterval(_mmStdPoll); _mmStdPoll = null;
      mmPopulateList();
      if (typeof populateMosaicModelDropdowns === 'function') {
        Promise.resolve(populateMosaicModelDropdowns()).catch(() => {})
          .then(() => { if (typeof populateCustomScalers === 'function') populateCustomScalers(); });
      }
    }
  }, 1000);
}
async function mmStdRun() {
  const log = document.getElementById('mmLog');
  if (_mmPoll || _mmDlPoll) {
    if (log) log.textContent = 'A compile or download is running \u2014 wait for it to finish first.';
    return;
  }
  const f = _mmStdFlavor();
  const endpoint = (document.getElementById('mmEndpoint') || {}).value || undefined;
  if (log) log.textContent = 'Starting\u2026';
  const res = await apiPost('/api/standard-models/run', { imgsz: f.imgsz, max_batch: f.max_batch, endpoint });
  if (!res || res.error) { if (log) log.textContent = (res && res.error) || 'Could not start.'; return; }
  _mmStdStartPolling();
}

// ── CM-232: Custom scalers (models/upscalers) ──────────────
let _mmUpModels = [];
let _mmUpSelected = null;
function _mmUpRender() {
  const el = document.getElementById('mmUpList');
  const rm = document.getElementById('mmUpRemove');
  if (!el) return;
  if (!_mmUpModels.length) {
    el.innerHTML = '<div class="mm-empty">No custom scalers. Add\u2026 a .pth / .safetensors super-resolution model, or drop one into models/upscalers and Rescan.</div>';
    _mmUpSelected = null;
    if (rm) rm.disabled = true;
    return;
  }
  el.innerHTML = _mmUpModels.map(m => {
    const sel = _mmUpSelected === m.file ? ' mm-selected' : '';
    const detail = `${m.name || m.arch} ${m.scale}x \u00b7 ${m.half ? 'fp16' : 'fp32'}` + (m.size_mb ? ` \u00b7 ${m.size_mb} MB` : '');
    const badge = m.slow ? '<span class="mm-badge mm-no">slow (attention model)</span>' : '<span class="mm-badge mm-ok">ready</span>';
    return `<div class="mm-row${sel}" data-file="${encodeURIComponent(m.file)}">`
      + `<span class="mm-name">${m.file} <span class="mm-ext">${detail}</span></span>${badge}</div>`;
  }).join('');
  el.querySelectorAll('.mm-row').forEach(row => {
    row.addEventListener('click', () => {
      const f = decodeURIComponent(row.dataset.file);
      _mmUpSelected = _mmUpSelected === f ? null : f;
      el.querySelectorAll('.mm-row').forEach(r => r.classList.toggle('mm-selected', decodeURIComponent(r.dataset.file) === _mmUpSelected));
      if (rm) rm.disabled = !_mmUpSelected;
    });
  });
  if (rm) rm.disabled = !_mmUpSelected;
}
async function mmUpPopulate(rescan) {
  const el = document.getElementById('mmUpList');
  if (el && !_mmUpModels.length) el.textContent = 'Loading\u2026';
  const data = await apiGet('/api/upscale-models' + (rescan ? '?rescan=1' : ''));
  _mmUpModels = (data && Array.isArray(data.models)) ? data.models : [];
  if (_mmUpSelected && !_mmUpModels.some(m => m.file === _mmUpSelected)) _mmUpSelected = null;
  _mmUpRender();
  if (data && data.error) { const log = document.getElementById('mmLog'); if (log) log.textContent = data.error; }
}
async function mmUpAdd() {
  const log = document.getElementById('mmLog');
  const api = (typeof window.pywebview !== 'undefined' && window.pywebview) ? window.pywebview.api : null;
  if (!api || typeof api.select_model_file !== 'function') {
    if (log) log.textContent = 'The file picker needs the desktop app. Drop the file into models/upscalers and press Rescan instead.';
    return;
  }
  let path = null;
  try { path = await api.select_model_file(); } catch (e) { path = null; }
  if (!path) return;
  if (log) log.textContent = `Adding ${path}\u2026`;
  const res = await apiPost('/api/upscale-models/add', { path });
  if (!res || res.error) { if (log) log.textContent = (res && res.error) || 'Could not add the file.'; return; }
  if (log) log.textContent = res.usable
    ? `Added ${res.file}: ${res.entry ? res.entry.label : 'usable'} \u2014 now in the Secondary and Upscale Scaler dropdowns.`
    : `Added ${res.file}, but it is not offered: ${res.reason}`;
  await mmUpPopulate();
  if (typeof populateCustomScalers === 'function') populateCustomScalers();
}
async function mmUpRemove() {
  if (!_mmUpSelected) return;
  const log = document.getElementById('mmLog');
  if (!confirm(`Remove ${_mmUpSelected} from the scaler dropdowns?\n\nThe file is moved to models/upscalers/_removed, not deleted.`)) return;
  const res = await apiPost('/api/upscale-models/remove', { file: _mmUpSelected });
  if (!res || res.error) { if (log) log.textContent = (res && res.error) || 'Could not remove.'; return; }
  if (log) log.textContent = `Moved to ${res.moved_to}`;
  _mmUpSelected = null;
  await mmUpPopulate();
  if (typeof populateCustomScalers === 'function') populateCustomScalers();
}
async function mmUpOpen() {
  const res = await apiPost('/api/upscale-models/open', {});
  const log = document.getElementById('mmLog');
  if (res && res.error && log) log.textContent = `${res.error} (folder: ${res.dir})`;
}

const manageModelsBtn = document.getElementById('manageModelsBtn');
if (manageModelsBtn) manageModelsBtn.addEventListener('click', openManageModels);
[['mmRefresh', () => { _mmSelected.clear(); return mmPopulateList(); }], ['mmSelectMissing', mmSelectMissing],
 ['mmCompile', mmCompile], ['mmClose', closeManageModels],
 ['mmAddSource', mmAddSource], ['mmFetch', mmFetch],
 ['mmDlSelectAll', mmDlSelectAll], ['mmDownload', mmDownload],
 ['mmStdRun', mmStdRun],                                                   // CM-232
 ['mmUpAdd', mmUpAdd], ['mmUpRescan', () => mmUpPopulate(true)],
 ['mmUpOpen', mmUpOpen], ['mmUpRemove', mmUpRemove]].forEach(([id, fn]) => {
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

// Batch 28: Test Frame result history -- the last N successful results for
// the SAME frame, so consecutive runs with different knobs (scaler on/off,
// temporal stability, feather...) can be flipped through in the enlarge's
// right pane via the < > navigator next to its label. The left (mosaic)
// pane always shows the current run; only the right pane time-travels.
// The stack clears whenever the tested frame number changes.
const _FOI_HIST_MAX = 5;
let _foiHistory = [];      // [{regions, tag}], oldest -> newest
let _foiHistFrame = null;  // frame number the history belongs to
let _foiHistPos = 0;       // entry currently shown in the right pane

// Compact per-run tag (tooltip on the counter): capture time + the knobs
// most likely to differ between consecutive test runs on one frame.
function _foiRunTag(params) {
  const m = (params && params.mosaic) || {};
  const parts = [];
  parts.push(m.mosaic_restoration_trt ? 'TRT' : 'PyTorch');
  if (m.mosaic_secondary && m.mosaic_secondary !== 'none') {
    parts.push('Scaler:' + m.mosaic_secondary
      + ((m.mosaic_secondary_denoise && m.mosaic_secondary_denoise !== 'none')
         ? ('+dn:' + m.mosaic_secondary_denoise) : ''));
  }
  if (m.mosaic_temporal_stability > 0) parts.push('TS:' + m.mosaic_temporal_stability);
  if (m.mosaic_blend_mask && m.mosaic_blend_mask !== 'none') {
    parts.push('Blend:' + m.mosaic_blend_mask);
    if (m.mosaic_feather_radius > 0) parts.push('Feather:' + m.mosaic_feather_radius);
  }
  if (m.mosaic_censor) parts.push('Censor:' + m.mosaic_censor_block);
  const t = new Date();
  const pad = (x) => String(x).padStart(2, '0');
  return `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}`
    + ' · ' + parts.join(' · ');
}

// Render the enlarge's right pane + navigator from the history stack.
// Batch 28b: numbered slots (1 = oldest kept, rightmost = newest) between
// the < > steppers give one-click random access; each slot's tooltip is
// that run's time + settings tag, readable without leaving the current run.
function _foiHistRender() {
  const rest = document.getElementById('zoomImage');
  const nav = document.getElementById('foiHistNav');
  const dots = document.getElementById('foiHistDots');
  const prev = document.getElementById('foiHistPrev');
  const next = document.getElementById('foiHistNext');
  if (!rest) return;
  const n = _foiHistory.length;
  if (_foiOpenIdx === null || n === 0) {
    if (nav) nav.classList.add('hidden');
    return;
  }
  if (_foiHistPos < 0) _foiHistPos = 0;
  if (_foiHistPos >= n) _foiHistPos = n - 1;
  const entry = _foiHistory[_foiHistPos];
  const r = entry.regions[_foiOpenIdx];
  rest.src = (r && r.restored) ? ('data:image/jpeg;base64,' + r.restored) : '';
  if (nav) {
    nav.classList.remove('hidden');
    if (dots) {
      dots.innerHTML = '';
      _foiHistory.forEach((en, i) => {
        const d = document.createElement('span');
        d.className = 'foi-hist-dot' + (i === _foiHistPos ? ' foi-hist-cur' : '');
        d.textContent = String(i + 1);
        const rr = en.regions[_foiOpenIdx];
        d.title = en.tag
          + (i === n - 1 ? ' · newest' : '')
          + ((rr && rr.restored) ? '' : ' · (no such region in this run)');
        d.addEventListener('click', (e) => {
          e.stopPropagation();
          _foiHistPos = i;
          _foiHistRender();
        });
        dots.appendChild(d);
      });
    }
    if (prev) prev.classList.toggle('foi-hist-off', _foiHistPos === 0);
    if (next) next.classList.toggle('foi-hist-off', _foiHistPos === n - 1);
  }
}

// Test Frame pane labels: in Add Mosaic (censor) mode the "before" is the
// clean original and the "after" is the pixelated result.
function _foiPaneLabels() {
  const c = document.getElementById('ctrlMosaicCensor');
  return (c && c.checked) ? ['Original', 'Censored'] : ['Mosaic', 'Restored'];
}
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

// CM-085 (Batch 38): collapsible Test Frame strip. The toggle hides the
// image rows (the bar stays, one line tall); a fresh Test Frame result
// always re-expands so the run the user just asked for is never hidden.
function _setFoiCollapsed(collapsed) {
  const panel = document.getElementById('faceStrip');
  const btn = document.getElementById('foiToggle');
  if (!panel || !btn) return;
  panel.classList.toggle('foi-collapsed', collapsed);
  btn.innerHTML = collapsed ? '&#9656;' : '&#9662;';
  btn.title = collapsed ? 'Expand the comparison strip'
                        : 'Collapse the comparison strip';
}
(function _wireFoiToggle() {
  const btn = document.getElementById('foiToggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const panel = document.getElementById('faceStrip');
    _setFoiCollapsed(panel && !panel.classList.contains('foi-collapsed'));
  });
})();

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
  // CM-148 (Batch 76): user chose "Use engine size" -- run at the engine's
  // compiled Image Size (params were gathered before the gate snapped the dial).
  if (gate.imgsz) {
    params.mosaic.mosaic_det_imgsz = gate.imgsz;
    // CM-196 report nit: the panel copy in the run report must show the size that ran.
    if (params.panel && typeof params.panel === 'object') params.panel.ctrlMosaicDetImgsz = String(gate.imgsz);
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
    _setFoiCollapsed(false);   // CM-085: fresh result -> always visible
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

    // Batch 28: push this result onto the frame's history stack. A new
    // frame starts a fresh stack; the cap drops the oldest. Position snaps
    // to the newest so the enlarge always lands on what just ran.
    if (_foiHistFrame !== res.frame) {
      _foiHistory = [];
      _foiHistFrame = res.frame;
    }
    _foiHistory.push({ regions: _foiRegions, tag: _foiRunTag(params) });
    if (_foiHistory.length > _FOI_HIST_MAX) _foiHistory.shift();
    _foiHistPos = _foiHistory.length - 1;
    const cell = (label, b64) => {
      const img = b64
        ? `<img class="foi-img" src="data:image/jpeg;base64,${b64}" alt="${label}">`
        : '<div class="foi-img foi-na">—</div>';
      return `<div class="foi-cell"><div class="foi-clabel">${label}</div>${img}</div>`;
    };
    // Rebuild the strip now that the new result is in hand (swap in place, so
    // it never blanks while refreshing).
    container.innerHTML = '';
    const _lbl = _foiPaneLabels();
    _foiRegions.forEach((r, i) => {
      const row = document.createElement('div');
      row.className = 'foi-row';
      row.title = 'Click to enlarge over the player';
      if (i === _foiOpenIdx) row.classList.add('foi-row-active');
      row.innerHTML = cell(_lbl[0], r.mosaic) + cell(_lbl[1], r.restored);
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

  // Labels: censor mode = Original|Censored, else Mosaic|Restored.
  // Batch 28: the right label text lives in its own span (#zoomLabelRight)
  // so setting it can't clobber the history navigator markup next to it.
  const _lbl = _foiPaneLabels();
  const labels = overlay.querySelectorAll('.zoom-label');
  if (labels[0]) labels[0].textContent = _lbl[0];
  const rlab = document.getElementById('zoomLabelRight');
  if (rlab) rlab.textContent = _lbl[1];

  orig.src = r.mosaic ? ('data:image/jpeg;base64,' + r.mosaic) : '';
  _foiHistRender();   // right pane + < > navigator from the history stack
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
  // Batch 28: hide the history navigator so it can't linger if the overlay
  // is ever reused outside foi-mode. The stack itself survives -- reopening
  // a pair on the same frame resumes where you were.
  const _nav = document.getElementById('foiHistNav');
  if (_nav) _nav.classList.add('hidden');
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
  _foiHistory = [];        // Batch 28: new video, new history
  _foiHistFrame = null;
  _foiHistPos = 0;
  if (_foiRunning) _foiRunning = false;
  const c = document.getElementById('foiRegions');
  if (c) c.innerHTML = '';
  const st = document.getElementById('foiStatus');
  if (st) st.textContent = 'Pause on a frame, then press Test Frame (below)';
  _closeFoiEnlarge();
}
const _foiZoomClose = document.getElementById('zoomClose');
if (_foiZoomClose) _foiZoomClose.addEventListener('click', _closeFoiEnlarge);

// Batch 28: history navigator clicks (< older, > newer). Bounds are
// clamped and the end buttons disabled inside _foiHistRender.
const _foiHistPrevBtn = document.getElementById('foiHistPrev');
const _foiHistNextBtn = document.getElementById('foiHistNext');
if (_foiHistPrevBtn) _foiHistPrevBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  _foiHistPos -= 1;
  _foiHistRender();
});
if (_foiHistNextBtn) _foiHistNextBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  _foiHistPos += 1;
  _foiHistRender();
});

// Keep the button label in sync with the playhead.
if (typeof player !== 'undefined' && player) {
  player.addEventListener('timeupdate', _updateTestFrameLabel);
  player.addEventListener('seeked', _updateTestFrameLabel);
  player.addEventListener('loadedmetadata', _updateTestFrameLabel);
}
_updateTestFrameLabel();


// ── Add Mosaic (SFW censor) ───────────────────────────────
// Decode -> pixelate up to 3 rectangles -> encode. No detection/restoration.
// Uses the marked segment if one is set, otherwise the whole video. The
// saved output becomes the preview (and the SBS viewer's "restored" side,
// so you can wipe-compare original vs censored).

const _amModal = document.getElementById('addMosaicModal');

function _amFmtTime(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

function _amOpenModal() {
  // Prefill SBS from the detection Split SBS control; scope from segment marks.
  const sbsCtl = document.getElementById('ctrlMosaicSbsSplit');
  document.getElementById('amSbs').checked = !!(sbsCtl && sbsCtl.checked);
  const seg = state.segmentEndTime > state.segmentStartTime;
  document.getElementById('amScope').textContent = seg
    ? `Scope: marked segment ${_amFmtTime(state.segmentStartTime)} - ${_amFmtTime(state.segmentEndTime)}`
    : 'Scope: whole video (mark a segment first to limit it).';
  _amModal.classList.remove('hidden');
  _updateRestorationButtonStates();
}

document.getElementById('addMosaicBtn').addEventListener('click', () => {
  if (!state.videoPath) {
    alert('Load a video first.');
    return;
  }
  // Draw-first flow: drag rectangles on the paused video, then the modal
  // opens pre-filled for numeric fine-tuning. "Type coordinates" skips
  // straight to the modal.
  _amEnterDraw();
});

document.getElementById('amCancel').addEventListener('click', () => {
  _amModal.classList.add('hidden');
  _updateRestorationButtonStates();
});

const _amRedrawBtn = document.getElementById('amRedraw');
if (_amRedrawBtn) _amRedrawBtn.addEventListener('click', () => {
  _amModal.classList.add('hidden');
  _amEnterDraw();   // previous rectangles are kept for adjustment
});

// ── Add Mosaic: draw rectangles on the player ─────────────
// A fixed-position overlay tracks the <video>'s displayed content box
// (object-fit: contain math, so letterbox bars are excluded). It swallows
// all mouse events while active — no play/pause conflict with the player's
// click handler. Up to 3 rects, drawn in FULL-FRAME video pixels; per-eye
// conversion happens when the modal is filled. With Split SBS on, a dashed
// ghost mirrors each rect on the other eye live.

var _amDraw = { active: false, rects: [], drag: null, layer: null, banner: null };
var _amJobRunning = false;

// True while the user is actively working on a mosaic: drawing, in the
// modal, or the encode job is running. Drives the Add Mosaic button state.
function _amIsBusy() {
  const modalOpen = _amModal && !_amModal.classList.contains('hidden');
  return _amDraw.active || modalOpen || _amJobRunning;
}

// Full reset — called by New (resetProject in init.js) so rectangles never
// leak from one project/video into the next.
function _amReset() {
  if (_amDraw.active) _amExitDraw(false);
  _amDraw.rects = [];
  _amDraw.drag = null;
  _amJobRunning = false;
  for (let i = 1; i <= 3; i++) {
    ['t', 'l', 'b', 'r'].forEach(k => {
      const el = document.getElementById(`am${i}${k}`);
      if (el) el.value = '';
    });
  }
  if (_amModal) _amModal.classList.add('hidden');
}

(function _amInjectDrawCss() {
  const st = document.createElement('style');
  st.textContent = `
    #amDrawLayer { position: fixed; z-index: 9000; cursor: crosshair;
      touch-action: none; }
    #amDrawLayer .am-rect { position: absolute; border: 2px solid #ff4444;
      background: rgba(255,68,68,0.12); box-sizing: border-box; }
    #amDrawLayer .am-rect .am-x { position: absolute; top: -10px; right: -10px;
      width: 20px; height: 20px; line-height: 18px; text-align: center;
      background: #ff4444; color: #fff; border-radius: 50%; font-size: 12px;
      cursor: pointer; user-select: none; }
    #amDrawLayer .am-rect .am-dim { position: absolute; left: 0; bottom: -18px;
      font: 11px monospace; color: #ffb3b3; background: #000a;
      padding: 0 4px; white-space: nowrap; }
    #amDrawLayer .am-ghost { position: absolute; border: 2px dashed #ff8888;
      box-sizing: border-box; pointer-events: none; opacity: .7; }
    #amDrawBanner { position: fixed; z-index: 9001;
      transform: translateX(-50%); display: flex; gap: 10px;
      align-items: center; background: #1a222ce6; border: 1px solid #33455c;
      border-radius: 6px; padding: 6px 12px; font-size: 12px; color: #cfd8e3;
      pointer-events: none; /* drawing works "through" the banner... */ }
    #amDrawBanner button { background: #243040; color: #cfd8e3;
      border: 1px solid #33455c; border-radius: 4px; padding: 3px 10px;
      font: inherit; cursor: pointer;
      pointer-events: auto; /* ...but its buttons stay clickable */ }
    #amDrawBanner button:hover { background: #2d3c50; }
    #amDrawBanner button.primary { background: #4da3ff; border-color: #4da3ff;
      color: #08121e; }`;
  document.head.appendChild(st);
})();

// Displayed content box of the <video> (object-fit: contain).
function _amMetrics() {
  const vw = player.videoWidth, vh = player.videoHeight;
  if (!vw || !vh) return null;
  const r = player.getBoundingClientRect();
  const scale = Math.min(r.width / vw, r.height / vh);
  return {
    vw, vh, scale,
    x: r.left + (r.width - vw * scale) / 2,
    y: r.top + (r.height - vh * scale) / 2,
    w: vw * scale, h: vh * scale,
  };
}

function _amSbsOn() {
  const c = document.getElementById('ctrlMosaicSbsSplit');
  return !!(c && c.checked);
}

function _amEnterDraw() {
  if (_amDraw.active) return;
  const m = _amMetrics();
  if (!m) {
    // Metadata not ready (or no playable video) — fall back to manual entry.
    _amOpenModal();
    return;
  }
  try { player.pause(); } catch (e) { /* noop */ }

  // Drop stale rectangles that don't fit the current video (e.g. a new
  // video of a different size was loaded without pressing New).
  if (_amDraw.rects.some(rc => rc.r >= m.vw || rc.b >= m.vh)) {
    _amDraw.rects = [];
  }

  if (!_amDraw.layer) {
    const layer = document.createElement('div');
    layer.id = 'amDrawLayer';
    document.body.appendChild(layer);
    _amDraw.layer = layer;

    const banner = document.createElement('div');
    banner.id = 'amDrawBanner';
    banner.innerHTML =
      '<span id="amDrawHint"></span>' +
      '<button id="amDrawDone" class="primary" title="Enter">Done</button>' +
      '<button id="amDrawAuto" title="Detect regions automatically instead of drawing">Auto-detect</button>' +
      '<button id="amDrawType">Type coordinates</button>' +
      '<button id="amDrawCancel" title="Esc">Cancel</button>';
    document.body.appendChild(banner);
    _amDraw.banner = banner;

    banner.querySelector('#amDrawDone').onclick = () => _amExitDraw(true);
    banner.querySelector('#amDrawAuto').onclick = () => {
      _amExitDraw(false);
      const a = document.getElementById('amAuto'); if (a) a.checked = true;
      _amOpenModal();
    };
    banner.querySelector('#amDrawType').onclick = () => { _amExitDraw(false); _amOpenModal(); };
    banner.querySelector('#amDrawCancel').onclick = () => { _amDraw.rects = []; _amExitDraw(false); };

    layer.addEventListener('pointerdown', _amDrawDown);
    layer.addEventListener('pointermove', _amDrawMove);
    layer.addEventListener('pointerup', _amDrawUp);
    window.addEventListener('resize', _amRender);
  }

  _amDraw.active = true;
  _amDraw.layer.style.display = 'block';
  _amDraw.banner.style.display = 'flex';
  window.addEventListener('keydown', _amDrawKeys, true);
  _amRender();
  _updateRestorationButtonStates();
}

function _amExitDraw(openModal) {
  _amDraw.active = false;
  _amDraw.drag = null;
  if (_amDraw.layer) _amDraw.layer.style.display = 'none';
  if (_amDraw.banner) _amDraw.banner.style.display = 'none';
  window.removeEventListener('keydown', _amDrawKeys, true);
  if (openModal) {
    _amFillFromRects();
    _amOpenModal();
  }
  _updateRestorationButtonStates();
}

function _amDrawKeys(e) {
  if (!_amDraw.active) return;
  if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); _amDraw.rects = []; _amExitDraw(false); }
  else if (e.key === 'Enter') { e.stopPropagation(); e.preventDefault(); _amExitDraw(true); }
}

function _amClientToVideo(e, m) {
  return {
    x: Math.max(0, Math.min(m.vw - 1, Math.round((e.clientX - m.x) / m.scale))),
    y: Math.max(0, Math.min(m.vh - 1, Math.round((e.clientY - m.y) / m.scale))),
  };
}

function _amDrawDown(e) {
  if (e.target.classList && e.target.classList.contains('am-x')) {
    const idx = parseInt(e.target.dataset.idx, 10);
    _amDraw.rects.splice(idx, 1);
    _amRender();
    return;
  }
  if (_amDraw.rects.length >= 3) { _amRender(); return; }
  const m = _amMetrics(); if (!m) return;
  _amDraw.drag = { start: _amClientToVideo(e, m), end: _amClientToVideo(e, m) };
  _amDraw.layer.setPointerCapture(e.pointerId);
  e.preventDefault();
}

function _amDrawMove(e) {
  if (!_amDraw.drag) return;
  const m = _amMetrics(); if (!m) return;
  _amDraw.drag.end = _amClientToVideo(e, m);
  _amRender();
}

function _amDrawUp() {
  if (!_amDraw.drag) return;
  const { start, end } = _amDraw.drag;
  _amDraw.drag = null;
  const t = Math.min(start.y, end.y), b = Math.max(start.y, end.y);
  const l = Math.min(start.x, end.x), r = Math.max(start.x, end.x);
  if ((b - t) >= 8 && (r - l) >= 8 && _amDraw.rects.length < 3) {
    _amDraw.rects.push({ t, l, b, r });
  }
  _amRender();
}

function _amRender() {
  if (!_amDraw.active || !_amDraw.layer) return;
  const m = _amMetrics(); if (!m) return;
  const layer = _amDraw.layer;
  layer.style.left = m.x + 'px';
  layer.style.top = m.y + 'px';
  layer.style.width = m.w + 'px';
  layer.style.height = m.h + 'px';
  _amDraw.banner.style.top = Math.max(8, m.y + 10) + 'px';
  _amDraw.banner.style.left = (m.x + m.w / 2) + 'px';   // centered over content

  const sbs = _amSbsOn();
  const half = m.vw / 2;
  let html = '';
  const all = _amDraw.rects.slice();
  if (_amDraw.drag) {
    const { start, end } = _amDraw.drag;
    all.push({ t: Math.min(start.y, end.y), l: Math.min(start.x, end.x),
               b: Math.max(start.y, end.y), r: Math.max(start.x, end.x), _live: true });
  }
  all.forEach((rc, i) => {
    const px = (v) => (v * m.scale);
    html += `<div class="am-rect" style="left:${px(rc.l)}px;top:${px(rc.t)}px;` +
            `width:${px(rc.r - rc.l)}px;height:${px(rc.b - rc.t)}px;">` +
            (rc._live ? '' : `<span class="am-x" data-idx="${i}" title="Remove">x</span>`) +
            `<span class="am-dim">${rc.r - rc.l}x${rc.b - rc.t}px</span></div>`;
    if (sbs) {
      // Dashed ghost on the other eye (same per-eye position).
      const onRight = ((rc.l + rc.r) / 2) >= half;
      const gl = onRight ? rc.l - half : rc.l + half;
      const gr = onRight ? rc.r - half : rc.r + half;
      if (gr > 0 && gl < m.vw) {
        html += `<div class="am-ghost" style="left:${px(gl)}px;top:${px(rc.t)}px;` +
                `width:${px(gr - gl)}px;height:${px(rc.b - rc.t)}px;"></div>`;
      }
    }
  });
  layer.innerHTML = html;

  const hint = _amDraw.banner.querySelector('#amDrawHint');
  const n = _amDraw.rects.length;
  hint.textContent = n >= 3
    ? 'Add Mosaic: 3/3 rectangles (x to remove one).'
    : `Add Mosaic: drag to draw a rectangle (${n}/3)` +
      (sbs ? ' — either eye; the dashed ghost mirrors it' : '') + '.';
}

// Convert drawn full-frame rects to modal values (per-eye when SBS is on).
function _amFillFromRects() {
  const m = _amMetrics();
  const sbs = _amSbsOn();
  const half = m ? Math.floor(m.vw / 2) : 0;
  for (let i = 1; i <= 3; i++) {
    const rc = _amDraw.rects[i - 1];
    ['t', 'l', 'b', 'r'].forEach(k => {
      const el = document.getElementById(`am${i}${k}`);
      if (!rc) { el.value = ''; return; }
      let v = rc[k];
      if (sbs && (k === 'l' || k === 'r')) {
        const onRight = ((rc.l + rc.r) / 2) >= half;
        v = onRight ? v - half : Math.min(v, half - 1);
        v = Math.max(0, v);
      }
      el.value = v;
    });
  }
}

function _amGatherRois() {
  const rois = [];
  const errs = [];
  for (let i = 1; i <= 3; i++) {
    const vals = ['t', 'l', 'b', 'r'].map(k =>
      document.getElementById(`am${i}${k}`).value.trim());
    if (vals.every(v => v === '')) continue;            // unused row
    if (vals.some(v => v === '')) {
      errs.push(`Rect ${i}: all four values are required.`);
      continue;
    }
    const [t, l, b, r] = vals.map(v => parseInt(v, 10));
    if ([t, l, b, r].some(v => isNaN(v) || v < 0)) {
      errs.push(`Rect ${i}: values must be non-negative integers.`);
    } else if (b <= t || r <= l) {
      errs.push(`Rect ${i}: needs bottom > top and right > left.`);
    } else {
      rois.push([t, l, b, r]);
    }
  }
  return { rois, errs };
}

document.getElementById('amSubmit').addEventListener('click', async () => {
  const block = Math.max(2, parseInt(document.getElementById('amBlock').value, 10) || 16);
  const gp = gatherMosaicParams();
  const startTime = state.segmentStartTime || 0;
  const endTime = state.segmentEndTime || 0;
  const auto = document.getElementById('amAuto').checked;

  let result, summary;
  if (auto) {
    // Auto-detect: pixelate regions found by the selected Detection model.
    if (!gp.mosaic.detection_model) {
      alert('Select a Detection model first (e.g. the NSFW detection model) in the Control Panel.');
      return;
    }
    const mosaic = Object.assign({}, gp.mosaic, { mosaic_censor_block: block });
    result = await apiPost('/api/auto-mosaic', {
      params: { mosaic, encoder: gp.encoder, output_dir: gp.output_dir, temp_dir: gp.temp_dir },
      start_time: startTime, end_time: endTime,
    });
    summary = `auto-detect, block=${block}${gp.mosaic.mosaic_sbs_split ? ', SBS per-eye' : ''}`;
  } else {
    const { rois, errs } = _amGatherRois();
    if (errs.length) { alert(errs.join('\n')); return; }
    if (!rois.length) { alert('Enter at least one rectangle (t,l,b,r), or tick Auto-detect.'); return; }
    const sbs = document.getElementById('amSbs').checked;
    result = await apiPost('/api/add-mosaic', {
      params: { rois, block, sbs, output_dir: gp.output_dir, temp_dir: gp.temp_dir, encoder: gp.encoder,
                hw_decode: gp.mosaic.mosaic_hw_decode },   // T20d: same decode lane rules as a run
      start_time: startTime, end_time: endTime,
    });
    summary = `${rois.length} rect(s), block=${block}${sbs ? ', both SBS eyes' : ''}`;
  }
  if (result.error) {
    alert('Failed to start: ' + result.error);
    return;
  }
  _amModal.classList.add('hidden');
  _amJobRunning = true;
  _updateRestorationButtonStates();

  _showProgressModal(auto ? 'Auto Mosaic (detecting)...' : 'Adding Mosaic...', summary);

  const pollInterval = _pollMosaicProgress({
    onComplete: (prog) => {
      clearInterval(pollInterval);
      progressTitle.textContent = 'Mosaic Added';
      progressPercent.textContent =
        `${prog.frame} frames -> ${prog.output_path || 'saved'}`;
      progressBar.style.width = '100%';
      progressEta.textContent = '';
      progressCancel.textContent = 'Close';
      state.previewReady = true;
      _amJobRunning = false;
      _updateRestorationButtonStates();
    },
    onError: (prog) => {
      clearInterval(pollInterval);
      const cancelled = prog.status === 'cancelled' && prog.frame > 0;
      if (prog.status === 'partial') {
        // CM-107 (Batch 50): errored, but a playable partial output exists.
        const pct = prog.total > 0 ? Math.round((prog.frame / prog.total) * 100) : 0;
        progressTitle.textContent =
          `Partial result — ${prog.frame} of ${prog.total} frames saved`;
        progressPercent.textContent = prog.error || 'Partial output saved.';
        progressBar.style.width = pct + '%';
      } else if (cancelled) {
        progressTitle.textContent = `Cancelled — ${prog.frame} frames processed`;
        progressPercent.textContent = 'Partial output was saved and remuxed.';
        state.previewReady = true;
        progressBar.style.width = '0%';
      } else {
        progressTitle.textContent = prog.status === 'error' ? 'Error' : 'Cancelled';
        progressPercent.textContent = prog.error || 'No frames processed';
        progressBar.style.width = '0%';
      }
      progressCancel.textContent = 'Close';
      _amJobRunning = false;
      _updateRestorationButtonStates();
    },
  });

  progressCancel.textContent = 'Stop';
  progressCancel.onclick = async () => {
    const prog = await apiGet('/api/progress');
    if (prog && (prog.status === 'complete' || prog.status === 'error' || prog.status === 'cancelled' || prog.status === 'partial')) {
      progressModal.classList.add('hidden');
      clearInterval(pollInterval);
      _updateRestorationButtonStates();
    } else {
      await apiPost('/api/cancel');
    }
  };
});


// ── CM-229: custom scalers (models\upscalers) in the Secondary and Upscale
// Scaler dropdowns. Options carry value "custom:<file>" and read
// "<file> (RealPLKSR 2x)". Re-populated at startup, after Manage Models
// activity, and on every video load (a file dropped into the folder shows
// up without a restart). A saved value naming a file that is gone falls
// back to the built-in default with one console line -- never a dead run.
// T20b: a missing custom secondary falls back to the edition's DEFAULT scaler (the
// recommended set), not to Off -- the point of the fallback is "never a dead run".
const _CUSTOM_SCALER_SELECTS = {
  ctrlMosaicSecondary: { fallback: ((document.documentElement.dataset.edition || 'cuda') === 'cuda') ? 'rtx-4x' : 'esrgan-4x' },
  ctrlUpscaleScaler: { fallback: 'esrgan' },
};
async function populateCustomScalers() {
  let data = null;
  try { data = await apiGet('/api/upscale-models'); } catch (e) { data = null; }
  const models = (data && Array.isArray(data.models)) ? data.models : [];
  const pending = window._pendingCustomScalers || {};
  for (const [id, spec] of Object.entries(_CUSTOM_SCALER_SELECTS)) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    const wanted = pending[id] !== undefined ? String(pending[id]) : sel.value;
    Array.from(sel.options).forEach(o => { if (String(o.value).startsWith('custom:')) o.remove(); });
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = m.label;
      o.dataset.customScaler = '1';
      o.title = `${m.file}: ${m.name || m.arch} ${m.scale}x, ${m.half ? 'fp16' : 'fp32'}`;
      sel.appendChild(o);
    }
    if (wanted && String(wanted).startsWith('custom:')) {
      if (Array.from(sel.options).some(o => o.value === wanted)) {
        sel.value = wanted;
      } else {
        sel.value = spec.fallback;
        console.warn(`[Scalers] ${id}: saved scaler "${wanted.slice(7)}" is not in ${(data && data.dir) || 'models\\upscalers'}; using ${spec.fallback}`);
      }
    } else if (!sel.value) {
      sel.value = spec.fallback;
    }
  }
  delete window._pendingCustomScalers;
  if (models.length) console.log(`[Scalers] ${models.length} custom model(s) in ${data.dir}: ${models.map(m => m.label).join(', ')}`);
  if (data && data.error) console.warn('[Scalers] ' + data.error);
  if (typeof _updateControlEnableStates === 'function') _updateControlEnableStates();
}

// ── Init: populate dropdowns + first state pass ───────────
// the two fills run one after the other, not concurrently: both make the
// server import torchvision, and two threads importing it at once raced
// into a circular-import error in the field (09-21)
Promise.resolve(populateMosaicModelDropdowns()).catch(() => {}).then(() => populateCustomScalers());
_updateRestorationButtonStates();
