// chitramaya/static/js/training.js
// CM-112 Training UI (Batch 83): Dataset Builder + Train Detector panels.
// Drives the /api/train/* endpoints, which shell out to the app's own
// -make-dataset / -train-det subcommands (the -compile-det pattern).
// Content never leaves the machine; datasets and runs land in user-chosen
// folders. Progress rides the Batch 82 machine-parsable lines.

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const modal = $('trainingModal');
  if (!modal) return;

  let _pollTimer = null;
  let _wasRunning = false;

  // ── Open / close ──────────────────────────────────────────
  const btn = $('trainingBtn');
  if (btn) btn.addEventListener('click', () => {
    modal.classList.remove('hidden');
    _startPolling();       // pick up an in-flight job on reopen
  });
  $('trCloseBtn').addEventListener('click', () => {
    modal.classList.add('hidden');
    // Polling continues while a job runs so reopening shows live state.
    if (!_wasRunning) _stopPolling();
  });
  modal.addEventListener('click', (e) => {
    if (e.target === modal && !_wasRunning) modal.classList.add('hidden');
  });

  // ── Native pickers (pywebview) with manual-path fallback ─────
  async function _pickVideo() {
    try {
      if (window.pywebview && window.pywebview.api &&
          window.pywebview.api.select_video) {
        return await window.pywebview.api.select_video();
      }
    } catch (e) { console.warn('select_video failed', e); }
    return prompt('Full path to a source video:') || null;
  }
  async function _pickFolder() {
    try {
      if (window.pywebview && window.pywebview.api &&
          window.pywebview.api.select_folder) {
        return await window.pywebview.api.select_folder();
      }
    } catch (e) { console.warn('select_folder failed', e); }
    return prompt('Full path to a folder:') || null;
  }

  // ── Panel 1: source video list ────────────────────────────
  const videoList = $('trVideoList');
  $('trAddVideoBtn').addEventListener('click', async () => {
    const p = await _pickVideo();
    if (!p) return;
    const exists = [...videoList.options].some(o => o.value === p);
    if (exists) return;
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    videoList.appendChild(opt);
  });
  $('trRemoveVideoBtn').addEventListener('click', () => {
    [...videoList.selectedOptions].forEach(o => o.remove());
  });
  $('trOutBrowseBtn').addEventListener('click', async () => {
    const p = await _pickFolder();
    if (p) $('trOutDir').value = p;
  });
  $('trRunBrowseBtn').addEventListener('click', async () => {
    const p = await _pickFolder();
    if (p) $('trRunDir').value = p;
  });

  // Batch 85: data.yaml picker (feedback #1 -- no more manual typing on
  // return to this screen).
  $('trDataYamlBrowseBtn').addEventListener('click', async () => {
    let p = null;
    try {
      if (window.pywebview && window.pywebview.api &&
          window.pywebview.api.select_yaml) {
        p = await window.pywebview.api.select_yaml();
      }
    } catch (e) { console.warn('select_yaml failed', e); }
    if (p === null) p = prompt('Full path to a data.yaml:') || null;
    if (p) $('trDataYaml').value = p;
  });

  // ── Panel 2: base model custom-path toggle ────────────────
  $('trBase').addEventListener('change', () => {
    $('trBaseCustom').classList.toggle('hidden',
        $('trBase').value !== 'custom');
  });

  // ── Start actions ─────────────────────────────────────────
  // Batch 85 (feedback #2): button state is authoritative from the CLICK,
  // not from the racy poll. The old code re-enabled Start Training during
  // the run (a poll gap right after click, or the long silent AMP/AutoBatch
  // startup), so the user thought training was done and re-launched it --
  // the run folder kept getting recreated. Now: _activeKind is set on click
  // and cleared ONLY after we have observed running=true then running=false
  // (a real completion), so an early/racy running=false can never re-enable.
  let _activeKind = null;   // 'dataset' | 'train' while a job is ours
  let _sawRunning = false;  // observed running=true since we started?

  function _applyButtons() {
    const busy = !!_activeKind;
    const b = $('trBuildBtn'), t = $('trTrainBtn');
    b.disabled = busy;
    t.disabled = busy;
    b.textContent = (_activeKind === 'dataset') ? 'Building...' : 'Build Dataset';
    t.textContent = (_activeKind === 'train') ? 'Training...' : 'Start Training';
  }

  $('trBuildBtn').addEventListener('click', async () => {
    const inputs = [...videoList.options].map(o => o.value);
    const body = {
      inputs: inputs,
      out_dir: $('trOutDir').value.trim(),
      frames_per_video: parseInt($('trFrames').value, 10) || 400,
      negatives_pct: parseInt($('trNegPct').value, 10) || 15,
    };
    _activeKind = 'dataset'; _sawRunning = false; _applyButtons();
    $('trStatusLine').textContent = 'Starting dataset build...';
    const r = await apiPost('/api/train/build-dataset', body);
    if (r && r.error) { alert(r.error); _activeKind = null; _applyButtons(); return; }
    _startPolling();
  });

  $('trTrainBtn').addEventListener('click', async () => {
    let base = $('trBase').value;
    if (base === 'custom') base = $('trBaseCustom').value.trim();
    const body = {
      data_yaml: $('trDataYaml').value.trim(),
      base: base,
      epochs: parseInt($('trEpochs').value, 10) || 60,
      imgsz: parseInt($('trImgsz').value, 10) || 800,
      run_dir: $('trRunDir').value.trim(),
      name: $('trRunName').value.trim() || 'mosaic-det',
    };
    _activeKind = 'train'; _sawRunning = false; _applyButtons();
    $('trStatusLine').textContent = 'Starting training (first run downloads base weights)...';
    const r = await apiPost('/api/train/train-det', body);
    if (r && r.error) { alert(r.error); _activeKind = null; _applyButtons(); return; }
    _startPolling();
  });

  // ── Copy log (Batch 84; standing rule: every log box gets one) ──────
  // Mirrors the console drawer / compile-log (CM-109) pattern exactly:
  // clipboard API first, execCommand fallback, 1.5s button feedback.
  const copyBtn = $('trLogCopy');
  if (copyBtn) copyBtn.addEventListener('click', () => {
    const text = $('trLog').textContent || '';
    const done = (ok) => {
      copyBtn.textContent = ok ? 'Copied!' : 'Copy failed';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
    } else {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        done(ok);
      } catch (e) { done(false); }
    }
  });

  // ── Status polling ────────────────────────────────────────
  function _fmtEta(s) {
    s = Math.max(0, Math.round(s));
    const m = Math.floor(s / 60), sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  }

  function _render(st) {
    const p = st.progress || {};
    let pct = 0;
    let line = 'Idle.';
    if (st.kind === 'dataset') {
      const vids = p.videos || 1;
      const vi = Math.max(1, p.video_i || 1);
      const within = (p.frames > 0) ? (p.frame || 0) / p.frames : 0;
      pct = Math.round(((vi - 1 + within) / vids) * 100);
      line = st.running
        ? `Building dataset: video ${vi}/${vids}` +
          (p.video ? ` (${p.video})` : '') +
          ` — ${p.frame || 0}/${p.frames || '?'} frames`
        : (st.returncode === 0 && p.images != null
            ? `Dataset done: ${p.images} images, ${p.boxes} boxes, ` +
              `${p.negatives} negatives.`
            : (st.returncode == null ? 'Idle.'
               : `Dataset build FAILED (exit ${st.returncode}) — see log.`));
      if (!st.running && st.returncode === 0) pct = 100;
    } else if (st.kind === 'train') {
      const e = p.epoch || 0, n = p.epochs || 0;
      pct = n > 0 ? Math.round((e / n) * 100) : 0;
      const met = (p.map50 != null)
        ? ` — mAP50 ${Number(p.map50).toFixed(4)}` : '';
      const eta = (st.running && p.eta_s != null)
        ? ` — ETA ${_fmtEta(p.eta_s)}` : '';
      line = st.running
        ? `Training: epoch ${e}/${n}${met}${eta}`
        : (st.returncode === 0
            ? `Training done: ${n} epochs${met}. best.pt is in the run ` +
              `folder — copy it into models\\ and Compile in Manage Models.`
            : (st.returncode == null ? 'Idle.'
               : `Training FAILED (exit ${st.returncode}) — see log.`));
      if (!st.running && st.returncode === 0) pct = 100;
    }
    $('trBarFill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
    $('trStatusLine').textContent = line;

    const logEl = $('trLog');
    const stick = (logEl.scrollTop + logEl.clientHeight >=
                   logEl.scrollHeight - 20);
    if (logEl.textContent !== st.log) {
      logEl.textContent = st.log || '';
      if (stick) logEl.scrollTop = logEl.scrollHeight;
    }
  }

  async function _poll() {
    const st = await apiGet('/api/train/status');
    if (!st || st.error) return;
    _render(st);
    // Batch 85: authoritative button state (see _applyButtons rationale).
    if (st.running) {
      _sawRunning = true;
      if (!_activeKind) _activeKind = st.kind;   // adopt an in-flight job on reopen
    } else if (_activeKind && _sawRunning) {
      _activeKind = null;                        // real completion -> release
    }
    _applyButtons();
    if (_wasRunning && !st.running && st.kind === 'dataset' &&
        st.returncode === 0) {
      // Glue: a finished build hands its data.yaml straight to Panel 2.
      const out = $('trOutDir').value.trim();
      if (out && !$('trDataYaml').value.trim()) {
        const sep = out.includes('/') ? '/' : '\\';
        $('trDataYaml').value = out.replace(/[\\/]+$/, '') + sep + 'data.yaml';
      }
    }
    _wasRunning = !!st.running;
    if (!st.running && _pollTimer && modal.classList.contains('hidden')) {
      _stopPolling();   // nothing live and nobody watching
    }
  }

  function _startPolling() {
    if (_pollTimer) return;
    _poll();
    _pollTimer = setInterval(_poll, 1000);
  }
  function _stopPolling() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  }
})();
