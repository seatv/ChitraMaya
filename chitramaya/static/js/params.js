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
    mask_only: document.getElementById('ctrlShowMask').checked,
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

// All configurable control IDs
const CONFIG_CONTROLS = [
  // Detection
  'ctrlDetectType', 'ctrlDetectScore', 'ctrlShowMask',
  // Swap
  'ctrlResolution', 'ctrlStrength', 'ctrlThreshold', 'ctrlMergeMath',
  // Restorer
  'ctrlRestorerOn', 'ctrlRestorerType', 'ctrlRestorerDet', 'ctrlRestorerAlpha',
  // Occluder
  'ctrlOccluderOn', 'ctrlOccluderAmt',
  // Face Parser
  'ctrlFaceParserOn', 'ctrlFaceParserAmt', 'ctrlMouthParser',
  // Diff
  'ctrlDiffOn', 'ctrlDiffThresh',
  // Orientation
  'ctrlOrientOn', 'ctrlOrientAngle',
  // Border & Blend
  'ctrlBorderTop', 'ctrlBorderBot', 'ctrlBorderSides', 'ctrlBorderBlur', 'ctrlBlend',
  // Color
  'ctrlColorOn', 'ctrlGamma', 'ctrlRed', 'ctrlGreen', 'ctrlBlue',
  // Face Adjustment
  'ctrlFaceAdjOn', 'ctrlKpsX', 'ctrlKpsY', 'ctrlKpsScale', 'ctrlFaceScale',
  // Encoder
  'ctrlCodec', 'ctrlPreset', 'ctrlQP',
  // Transport
  'skipBackward', 'skipForward',
];

function gatherFullConfig() {
  const cfg = {};

  // Folder paths
  cfg.facesDir = facesPath.value || '';
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
  if (cfg.facesDir !== undefined) facesPath.value = cfg.facesDir;
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

  // Update all dial value displays
  document.querySelectorAll('.ctrl-dial').forEach(dial => {
    dial.dispatchEvent(new Event('input'));
  });
}

// getDefaultConfig removed — defaults come from server /api/default-config
// which derives them from models.py dataclasses (single source of truth)

// ── Apply Settings / Live Preview ─────────────────────────
// Two modes:
// 1. Live OFF (default): user adjusts dials, clicks [Apply Settings]
// 2. Live ON: auto-update on every dial change (debounced 50ms)

let _previewTimer = null;

async function applySettings() {
  if (state.detectedFaces.length === 0) return;

  const maskOnly = document.getElementById('ctrlShowMask').checked;
  if (!maskOnly && state.assignments[state.currentFaceIdx] === undefined) return;

  applySettingsBtn.disabled = true;
  applySettingsBtn.innerHTML = 'Applying<br>...';

  const result = await previewSwap(state.currentFaceIdx);

  if (result && result.image) {
    // Update swapped thumbnail in carousel
    _setSlotImage('swappedImg', result.image);
    state.hasSwappedImage = true;

    // Update zoom preview if open
    if (state.zoomOpen) {
      zoomImage.src = 'data:image/jpeg;base64,' + result.image;
    }
  }

  applySettingsBtn.innerHTML = 'Apply<br>Settings';
  applySettingsBtn.disabled = false;
}

function requestLivePreview() {
  if (!livePreviewCheck || !livePreviewCheck.checked) return;
  if (state.assignments[state.currentFaceIdx] === undefined) return;
  if (_previewTimer) clearTimeout(_previewTimer);
  _previewTimer = setTimeout(() => applySettings(), 50);
}

// Wire all controls to trigger live preview when enabled
document.querySelectorAll('.ctrl-dial, .ctrl-select, .ctrl-checkbox').forEach(el => {
  if (el.id === 'ctrlShowMask') return; // Has its own handler
  const event = (el.type === 'checkbox' || el.tagName === 'SELECT') ? 'change' : 'input';
  el.addEventListener(event, requestLivePreview);
});

// Mask checkbox also updates button states and auto-populates swapped
document.getElementById('ctrlShowMask').addEventListener('change', async () => {
  const maskOn = document.getElementById('ctrlShowMask').checked;

  if (maskOn && state.detectedFaces.length > 0) {
    // Clear source face assignment — mask mode doesn't need it
    state.assignments = {};
    _clearSlotImage('sourceImg');

    // Auto-populate swapped with mask preview
    const result = await previewSwap(state.currentFaceIdx);
    if (result && result.image) {
      _setSlotImage('swappedImg', result.image);
      state.hasSwappedImage = true;
    }
  } else if (!maskOn) {
    // Toggled off — clear the mask preview
    _clearSlotImage('swappedImg');
    state.hasSwappedImage = false;
  }

  _updateButtonStates();
});
