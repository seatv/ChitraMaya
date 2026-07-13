// ── Dial Value Display ────────────────────────────────────
// Auto-wire all ctrl-dial inputs to update their value display

document.querySelectorAll('.ctrl-dial').forEach(dial => {
  const valSpan = document.getElementById('val' + dial.id.replace('ctrl', ''));
  if (!valSpan) return;

  function updateValue() {
    let display = dial.value;
    // Special formatting
    if (dial.id === 'ctrlGamma') {
      display = (parseInt(dial.value) / 100).toFixed(1);
    } else if (dial.id === 'ctrlRestorerAlpha') {
      display = dial.value + '%';
    }
    valSpan.textContent = display;
  }

  dial.addEventListener('input', updateValue);
  updateValue(); // Initialize
});

// ── Gather All Parameters ────────────────────────────────
// Collects current UI state into the params structure expected by the server

function gatherParams() {
  return {
    swap: {
      swapper_resolution: parseInt(document.getElementById('ctrlResolution').value),
      strength: parseInt(document.getElementById('ctrlStrength').value),
      match_threshold: parseInt(document.getElementById('ctrlThreshold').value),
      face_adj_enabled: document.getElementById('ctrlFaceAdjOn').checked,
      kps_x_offset: parseInt(document.getElementById('ctrlKpsX').value),
      kps_y_offset: parseInt(document.getElementById('ctrlKpsY').value),
      kps_scale: parseInt(document.getElementById('ctrlKpsScale').value),
      face_scale: parseInt(document.getElementById('ctrlFaceScale').value),
      color_enabled: document.getElementById('ctrlColorOn').checked,
      color_gamma: parseInt(document.getElementById('ctrlGamma').value) / 100.0,
      color_red: parseInt(document.getElementById('ctrlRed').value),
      color_green: parseInt(document.getElementById('ctrlGreen').value),
      color_blue: parseInt(document.getElementById('ctrlBlue').value),
    },
    mask: {
      border_top: parseInt(document.getElementById('ctrlBorderTop').value),
      border_bottom: parseInt(document.getElementById('ctrlBorderBot').value),
      border_sides: parseInt(document.getElementById('ctrlBorderSides').value),
      border_blur: parseInt(document.getElementById('ctrlBorderBlur').value),
      blend_amount: parseInt(document.getElementById('ctrlBlend').value),
      diff_enabled: document.getElementById('ctrlDiffOn').checked,
      diff_threshold: parseInt(document.getElementById('ctrlDiffThresh').value),
      occluder_enabled: document.getElementById('ctrlOccluderOn').checked,
      occluder_amount: parseInt(document.getElementById('ctrlOccluderAmt').value),
      face_parser_enabled: document.getElementById('ctrlFaceParserOn').checked,
      face_parser_amount: parseInt(document.getElementById('ctrlFaceParserAmt').value),
      mouth_parser_amount: parseInt(document.getElementById('ctrlMouthParser').value),
    },
    restorer: {
      enabled: document.getElementById('ctrlRestorerOn').checked,
      restorer_type: document.getElementById('ctrlRestorerType').value,
      det_mode: document.getElementById('ctrlRestorerDet').value,
      blend_alpha: parseInt(document.getElementById('ctrlRestorerAlpha').value) / 100.0,
    },
    orient: {
      enabled: document.getElementById('ctrlOrientOn').checked,
      angle: parseInt(document.getElementById('ctrlOrientAngle').value),
    },
    mask_only: (document.getElementById('ctrlShowMask')?.checked ?? false),
  };
}

// Build a readable summary of current swap settings for the progress dialog
function buildParamsSummary() {
  const p = gatherParams();
  const parts = [];

  parts.push(`Res: ${p.swap.swapper_resolution}`);
  parts.push(`Strength: ${p.swap.strength}`);
  parts.push(`Threshold: ${p.swap.match_threshold}`);
  if (p.mask_only) parts.push('⬛ MASK ONLY');

  if (p.restorer.enabled) {
    parts.push(`Restorer: ${p.restorer.restorer_type.toUpperCase()} (${Math.round(p.restorer.blend_alpha * 100)}%)`);
  }
  if (p.mask.occluder_enabled) parts.push('Occluder: ON');
  if (p.mask.face_parser_enabled) parts.push('Parser: ON');
  if (p.mask.diff_enabled) parts.push(`Diff: ${p.mask.diff_threshold}`);

  const codec = document.getElementById('ctrlCodec').value.toUpperCase();
  const preset = document.getElementById('ctrlPreset').value;
  const qp = document.getElementById('ctrlQP').value;
  parts.push(`Enc: ${codec}/${preset}/QP${qp}`);

  return parts.join(' · ');
}

// ── Unified Config System ─────────────────────────────────
// Uses control element IDs as config keys — no mapping needed.
// Each control type (dial, select, checkbox) is handled automatically.

// All configurable control IDs. Mosaic controls are appended by mosaic.js
// (MOSAIC_CONFIG_CONTROLS). Face-swap control IDs were removed here — they
// do not exist in ChitraMaya's UI and were silently skipped on save/load.
const CONFIG_CONTROLS = [
  // Encoder
  'ctrlCodec', 'ctrlPreset', 'ctrlQP',
  // Transport
  'skipBackward', 'skipForward',
];

function gatherFullConfig() {
  const cfg = {};

  // Folder paths
  cfg.outputDir = outputPath.value || '';
  cfg.tempDir = tempPath.value || '';

  // All UI controls
  for (const id of CONFIG_CONTROLS) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (el.type === 'checkbox') cfg[id] = el.checked;
    else cfg[id] = el.value;
  }

  // Developer flags — preserve from last loaded config
  cfg.debug = window._chitramayaDevFlags?.debug ?? false;
  cfg.perf_test = window._chitramayaDevFlags?.perf_test ?? false;

  return cfg;
}

function applyConfig(cfg) {
  if (!cfg) return;

  // Store developer flags for later save
  window._chitramayaDevFlags = {
    debug: cfg.debug ?? false,
    perf_test: cfg.perf_test ?? false,
  };

  // Folder paths
  if (cfg.outputDir !== undefined) outputPath.value = cfg.outputDir;
  if (cfg.tempDir !== undefined) tempPath.value = cfg.tempDir;

  // All controls
  for (const id of CONFIG_CONTROLS) {
    if (cfg[id] === undefined) continue;
    const el = document.getElementById(id);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!cfg[id];
    else el.value = String(cfg[id]);
  }

  // The mosaic model dropdowns are filled asynchronously (populateMosaic-
  // ModelDropdowns fetches the list). At startup this can race the config
  // load: if the config applies first, the setValue above no-ops because the
  // <option>s don't exist yet. Stash the desired values so populate can
  // restore them once the options are in. Robust to either resolve order.
  //
  // Only stash values that FAILED to apply (option not present yet) — and
  // clear any stale stash when everything applied. populateMosaicModel-
  // Dropdowns() consumes the stash exactly once. An unconditional stash
  // lingered forever and snapped the dropdowns back to the startup config's
  // models every time a compile/download finished repopulating the lists.
  if (typeof window !== 'undefined') {
    const detSel = document.getElementById('ctrlMosaicDetModel');
    const restSel = document.getElementById('ctrlMosaicRestModel');
    const detWanted = cfg.ctrlMosaicDetModel;
    const restWanted = cfg.ctrlMosaicRestModel;
    const detMissed = detWanted !== undefined && detWanted !== '' &&
                      detSel && detSel.value !== String(detWanted);
    const restMissed = restWanted !== undefined && restWanted !== '' &&
                       restSel && restSel.value !== String(restWanted);
    if (detMissed || restMissed) {
      window._pendingMosaicModels = {
        det: detMissed ? detWanted : undefined,
        rest: restMissed ? restWanted : undefined,
      };
    } else {
      delete window._pendingMosaicModels;
    }
  }

  // Update all dial value displays
  document.querySelectorAll('.ctrl-dial').forEach(dial => {
    dial.dispatchEvent(new Event('input'));
  });
}

// getDefaultConfig removed — defaults come from server /api/default-config
// which derives them from models.py dataclasses (single source of truth)
