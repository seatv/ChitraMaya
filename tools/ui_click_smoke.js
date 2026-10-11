// tools/ui_click_smoke.js
// CM-230 (T17): DEV-ONLY click-path smoke in jsdom -- the check the T16
// dead button would have failed. For every execution mode (restore, Preview
// Detection, Add Mosaic, Upscale) it loads the page, gives it a video and
// models, sets the mode, clicks Restore & Save and asserts that the request
// to /api/mosaic-full actually left the page with the mode's payload. Then
// the T17 flow: mode labels, greying, Test Frame, a segment run's Preview,
// Save Segment.
//
//   node tools/ui_click_smoke.js <repo root> [cuda|rocm|xpu]
// Exit code 2 on any failed assertion or load-time error.
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const path = require('path');
const ROOT = process.argv[2];
const EDITION = process.argv[3] || 'cuda';
let html = fs.readFileSync(path.join(ROOT, 'chitramaya/templates/ui.html'), 'utf8');
html = html.replace(/\{\{\s*url_for\('static',\s*filename='([^']+)'\)\s*\}\}/g, '/static/$1');
html = html.replace(/\{\{\s*edition\s*\}\}/g, EDITION);
html = html.replace(/\{%[^%]*%\}/g, '');
html = html.replace(/\{\{[^}]*\}\}/g, '');
html = html.replace(/<script src="\/static\/js\/([\w.]+)[^"]*"><\/script>/g, (m, f) =>
  '<script>\n' + fs.readFileSync(path.join(ROOT, 'chitramaya/static/js', f), 'utf8') + '\n</script>');
html = html.replace(/<script type="module"[^>]*><\/script>/g, '');
html = html.replace(/<link[^>]*href="\/static\/swap\.css[^"]*"[^>]*>/, '<style>\n' + fs.readFileSync(path.join(ROOT, 'chitramaya/static/swap.css'), 'utf8') + '\n</style>');
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + String(e && (e.detail || e.message || e)).split('\n')[0]));
vc.on('error', (...a) => errors.push('console.error: ' + a.map(String).join(' ').split('\n')[0]));
vc.on('log', () => {}); vc.on('warn', () => {}); vc.on('info', () => {});
const posts = [];   // {url, body}
let progress = { status: 'idle' };
const DET = 'C:/m/det_v4_fast.pt', REST = 'C:/m/rest_generic.pth';
const dom = new JSDOM(html, {
  runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://127.0.0.1:5000/', virtualConsole: vc,
  beforeParse(window) {
    window.fetch = async (url, opts = {}) => {
      const method = (opts.method || 'GET').toUpperCase();
      let body = {};
      if (method === 'POST') {
        let parsed = null; try { parsed = JSON.parse(opts.body || '{}'); } catch {}
        posts.push({ url, body: parsed });
      }
      if (url.startsWith('/api/list-mosaic-models')) body = {
        detection: [{ path: DET, name: 'det_v4_fast.pt', has_engine: true, engine_imgszs: [640], engine_flavors: [{ imgsz: 640, max_batch: 8, fp16: true, label: '640/b8' }] }],
        restoration: [{ path: REST, name: 'rest_generic.pth', engines: { fixed: true } }] };
      else if (url === '/api/presets') body = { presets: [], dir: 'presets' };
      // CM-229: two custom scalers in models/upscalers
      else if (url.startsWith('/api/upscale-models')) body = { dir: 'X:/app/models/upscalers', models: [
        { id: 'custom:RealPLKSR_x2.pth', file: 'RealPLKSR_x2.pth', label: 'RealPLKSR_x2 (RealPLKSR 2x)', arch: 'PLKSR', name: 'RealPLKSR', scale: 2, half: true },
        { id: 'custom:ATD_light_x2.safetensors', file: 'ATD_light_x2.safetensors', label: 'ATD_light_x2 (ATD 2x)', arch: 'ATD', name: 'ATD', scale: 2, half: true } ] };
      // CM-232: standard set status + upscaler add/remove/open
      else if (url.startsWith('/api/standard-models/run')) body = { ok: true, started: true };
      // T20a: the server says compiles=false off-CUDA (no TensorRT there)
      else if (url.startsWith('/api/standard-models')) body = { edition: EDITION, compiles: EDITION === 'cuda', all_present: false, all_compiled: false,
        job: { running: false, step: 0, total: 0, stage: '', done: null }, log: '', models: [
          { file: 'lada_mosaic_detection_model_v2.pt', label: 'lada detection v2', kind: 'detection', present: true, compiled: true, flavors: ['640/b8'], size_mb: 52 },
          { file: 'lada_mosaic_detection_model_v4_fast.pt', label: 'lada detection v4 fast', kind: 'detection', present: false, compiled: false, flavors: [], size_mb: 0 },
          { file: 'lada_vr_mosaic_detection_model_v2_accurate.pt', label: 'zelefans VR detection v2 accurate', kind: 'detection', present: true, compiled: false, flavors: [], size_mb: 52 },
          { file: 'lada_mosaic_restoration_model_generic_v1.2.pth', label: 'lada restoration generic v1.2', kind: 'restoration', present: true, compiled: true, flavors: [], size_mb: 60 } ] };
      else if (url === '/api/upscale-models/remove') body = { ok: true, moved_to: 'X:/app/models/upscalers/_removed/RealPLKSR_x2.pth' };
      else if (url === '/api/upscale-models/open') body = { ok: true, dir: 'X:/app/models/upscalers' };
      // the saved config names a custom secondary that exists and an upscale scaler that is gone
      else if (url === '/api/load-config' || url === '/api/default-config') body = { ctrlMosaicSecondary: 'custom:RealPLKSR_x2.pth', ctrlUpscaleScaler: 'custom:Gone_x4.pth' };
      else if (url.startsWith('/api/progress')) body = progress;
      else if (url.startsWith('/api/session-status')) body = { running: false };
      else if (url === '/api/mosaic-full' || url === '/api/mosaic-segment') body = { status: 'started' };
      // CM-236 (T22): the pre-run VRAM check -- fits by default; the test flips it
      else if (url === '/api/vram-preflight') {
        // T23d (CM-239): the test can make the check slow to watch the button
        if (window.__vramDelayMs) await new Promise(r => setTimeout(r, window.__vramDelayMs));
        body = window.__vramMock || { fits: true, need_mb: 1000, free_mb: 4000, footprint_source: 'measured' };
      }
      else if (url === '/api/save-segment') body = { status: 'ok', output: 'X:/out/clip-upscaled-00m01.000s-00m03.000s.mp4', name: 'clip-upscaled-00m01.000s-00m03.000s.mp4', folder: 'X:/out' };
      else if (url.startsWith('/api/console')) body = { lines: [], next: 0 };
      else body = {};
      return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
    };
    window.alert = (m) => posts.push({ url: 'alert', body: String(m) });
    window.HTMLCanvasElement.prototype.getContext = () => null;
    window.HTMLMediaElement.prototype.play = async () => {};
    window.HTMLMediaElement.prototype.pause = () => {};
    window.HTMLMediaElement.prototype.load = () => {};
    window.pywebview = { api: {} };
  },
});
dom.window.addEventListener('error', (e) => errors.push('window.onerror: ' + (e.message || e)));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const fails = [];
function check(name, cond, detail = '') { console.log(`[${cond ? 'OK ' : 'BAD'}] ${name}${detail ? '  ' + detail : ''}`); if (!cond) fails.push(name); }

setTimeout(async () => {
  const w = dom.window, d = w.document;
  const $ = (id) => d.getElementById(id);
  const fire = (el, type = 'change') => el.dispatchEvent(new w.Event(type, { bubbles: true }));
  const CUDA = EDITION === 'cuda';
  const S = w.eval('state');   // `const state` in state.js: not a window property, same object via eval
  console.log(`edition=${EDITION} load-time errors: ${errors.length}`); errors.forEach(e => console.log('  ' + e));
  // a video and models, like a user who just loaded a file
  S.videoPath = 'X:/in/clip.mp4';
  S.segmentStartTime = 0; S.segmentEndTime = 0;
  const addOpt = (sel, v) => { const o = d.createElement('option'); o.value = v; o.textContent = path.basename(v); sel.appendChild(o); sel.value = v; };
  addOpt($('ctrlMosaicDetModel'), DET); addOpt($('ctrlMosaicRestModel'), REST);
  $('ctrlMosaicDetTrt').checked = false; $('ctrlMosaicRestTrt').checked = false;   // no engine gate in this smoke
  fire($('ctrlMosaicDetModel')); fire($('ctrlMosaicRestModel'));
  if (typeof w._updateRestorationButtonStates === 'function') w._updateRestorationButtonStates();
  await sleep(50);

  // CM-229 (T19): the custom scaler options arrived, the saved existing one applied, the
  // saved missing one fell back to the default (and the payload carries the id verbatim)
  {
    const sec = $('ctrlMosaicSecondary'), ups = $('ctrlUpscaleScaler');
    check('custom scalers: options added to Secondary', Array.from(sec.options).filter(o => o.value.startsWith('custom:')).length === 2);
    check('custom scalers: options added to Upscale Scaler', Array.from(ups.options).filter(o => o.value.startsWith('custom:')).length === 2);
    check('custom scalers: option label reads file (arch Nx)', Array.from(sec.options).some(o => o.textContent === 'RealPLKSR_x2 (RealPLKSR 2x)'));
    check('custom scalers: saved existing custom secondary applied', sec.value === 'custom:RealPLKSR_x2.pth', `got=${sec.value}`);
    check('custom scalers: saved missing custom upscale scaler fell back to esrgan', ups.value === 'esrgan', `got=${ups.value}`);
    const p = w.eval('gatherMosaicParams()');
    check('custom scalers: payload mosaic_secondary carries the id verbatim', !!(p.mosaic && p.mosaic.mosaic_secondary === 'custom:RealPLKSR_x2.pth'), `got=${p.mosaic && p.mosaic.mosaic_secondary}`);
    sec.value = 'none';
  }

  // CM-232 (T20): Manage Models -- standard set section and custom scalers section
  {
    $('manageModelsBtn').click(); await sleep(400);
    const rows = Array.from($('mmStdList').querySelectorAll('.mm-row'));
    check('std: four standard models listed', rows.length === 4, `got=${rows.length}`);
    check('std: missing one badged Not downloaded', rows.some(r => /Not downloaded/.test(r.textContent)));
    if (CUDA) {
      check('std: present-not-compiled badged for this flavor', rows.some(r => /not compiled/.test(r.textContent)));
      check('std: status counts work to do', /1 to download/.test($('mmStdStatus').textContent) && /1 to compile/.test($('mmStdStatus').textContent), `"${$('mmStdStatus').textContent}"`);
      check('cuda: compile section live, no-compile note hidden', !$('mmCompileSection').classList.contains('mm-inert') && !$('mmCompile').disabled === ($('mmCompile').textContent.indexOf('(0 selected)') < 0) && $('mmNoCompileNote').classList.contains('mm-hidden'));
      check('cuda: models list badges read Compiled', /Compiled/.test($('mmModelList').textContent) && !/PyTorch \(this edition\)/.test($('mmModelList').textContent));
    } else {
      // T20a: off-CUDA nothing compiles -- no "not compiled" anywhere, compile section gone, note shown
      check('off-cuda: present models badged Downloaded, never "not compiled"', rows.some(r => /Downloaded/.test(r.textContent)) && !rows.some(r => /not compiled/.test(r.textContent)));
      check('off-cuda: status counts downloads only', /1 to download/.test($('mmStdStatus').textContent) && !/compile/.test($('mmStdStatus').textContent), `"${$('mmStdStatus').textContent}"`);
      // T20b: greyed, not hidden -- the section exists and is not yours yet
      check('off-cuda: compile section greyed (mm-inert), every control disabled', $('mmCompileSection').classList.contains('mm-inert') && Array.from($('mmCompileSection').querySelectorAll('input, button')).every(el => el.disabled));
      check('off-cuda: no-compile note shown', !$('mmNoCompileNote').classList.contains('mm-hidden'));
      check('off-cuda: models list badges read PyTorch (this edition)', /PyTorch \(this edition\)/.test($('mmModelList').textContent) && !/Not compiled/.test($('mmModelList').textContent), `"${$('mmModelList').textContent.slice(0, 80)}"`);
    }
    check('std: button reads Get the standard models (not done)', /Get the standard models/.test($('mmStdRun').textContent), `"${$('mmStdRun').textContent}"`);
    const upRows = Array.from($('mmUpList').querySelectorAll('.mm-row'));
    check('upscalers: two custom models listed', upRows.length === 2, `got=${upRows.length}`);
    check('upscalers: row shows arch, scale, precision', upRows.some(r => /RealPLKSR 2x/.test(r.textContent) && /fp16/.test(r.textContent)));
    check('upscalers: Remove disabled with nothing selected', $('mmUpRemove').disabled === true);
    upRows[0].click(); await sleep(30);
    check('upscalers: Remove enabled after selecting a row', $('mmUpRemove').disabled === false);
    w.confirm = () => true;
    const before = posts.length;
    $('mmUpRemove').click(); await sleep(200);
    const rm = posts.slice(before).find(x => x.url === '/api/upscale-models/remove');
    check('upscalers: Remove posts the selected file', !!rm && rm.body.file === 'RealPLKSR_x2.pth', rm ? `file=${rm.body.file}` : 'no post');
    const b2 = posts.length;
    $('mmStdRun').click(); await sleep(200);
    const run = posts.slice(b2).find(x => x.url === '/api/standard-models/run');
    check('std: button posts the run with the modal flavor (b8 default)', !!run && run.body.imgsz === 640 && run.body.max_batch === 8, run ? `imgsz=${run.body.imgsz} b=${run.body.max_batch}` : 'no post');
    // T20a: the modal's Detection Batch slider drives the flavor, not the panel's run-time slider
    $('mmDetBatch').value = '12'; fire($('mmDetBatch'), 'input'); fire($('mmDetBatch')); await sleep(30);
    if (CUDA) check('std: Detection Batch slider 12 shows in the status flavor', /b12/.test($('mmStdStatus').textContent), `"${$('mmStdStatus').textContent}"`);
    check('std: Detection Batch slider value label follows', $('mmDetBatchVal').textContent === '12', `"${$('mmDetBatchVal').textContent}"`);
    check('std: modal slider is 2..16 step 2', $('mmDetBatch').min === '2' && $('mmDetBatch').max === '16' && $('mmDetBatch').step === '2');
    check('layout: Custom scalers section sits below the compile log', $('mmLog').compareDocumentPosition($('mmUpSection')) & 4);
    $('mmDetBatch').value = '8'; fire($('mmDetBatch'), 'input'); fire($('mmDetBatch')); await sleep(30);
    // T20b: a row pick does not survive Refresh (the surprise 2-model recompile of 09-23)
    const mrows = Array.from($('mmModelList').querySelectorAll('.mm-row'));
    if (mrows.length) { mrows[0].click(); await sleep(20); }
    check('mm: row selected before Refresh', w.eval('_mmSelected.size') === 1, `size=${w.eval('_mmSelected.size')}`);
    $('mmRefresh').click(); await sleep(300);
    check('mm: selection cleared by Refresh (T20b)', w.eval('_mmSelected.size') === 0, `size=${w.eval('_mmSelected.size')}`);
    if (CUDA) {
      // off-CUDA the compile button is inert whatever the selection
    } else {
      mrows.length && (Array.from($('mmModelList').querySelectorAll('.mm-row'))[0].click(), await sleep(20));
      check('off-cuda: compile button stays disabled with a row selected', $('mmCompile').disabled === true);
      w.eval('_mmSelected.clear()');
    }
    $('mmClose').click(); await sleep(50);
  }

  // T20b: the Manage Models button colour says what comes next
  {
    const mmb = $('manageModelsBtn');
    $('ctrlMosaicDetTrt').checked = true; fire($('ctrlMosaicDetTrt')); await sleep(30);   // the smoke unticked it above
    check('mm-btn: models + engine at 640 -> plain Manage Models', !mmb.classList.contains('mm-btn-get') && !mmb.classList.contains('mm-btn-compile') && mmb.textContent === 'Manage Models', `"${mmb.textContent}" ${mmb.className}`);
    $('ctrlMosaicDetImgsz').value = '800'; fire($('ctrlMosaicDetImgsz'), 'input'); fire($('ctrlMosaicDetImgsz')); await sleep(30);
    if (CUDA) {
      check('mm-btn: cuda, TRT ticked, no engine at 800 -> amber Compile Models', mmb.classList.contains('mm-btn-compile') && mmb.textContent === 'Compile Models', `"${mmb.textContent}" ${mmb.className}`);
      $('ctrlMosaicDetTrt').checked = false; fire($('ctrlMosaicDetTrt')); await sleep(30);
      check('mm-btn: cuda, TRT unticked -> plain again (PyTorch is a choice)', !mmb.classList.contains('mm-btn-compile'), `${mmb.className}`);
      $('ctrlMosaicDetTrt').checked = true; fire($('ctrlMosaicDetTrt')); await sleep(30);
    } else {
      check('mm-btn: off-cuda never amber', !mmb.classList.contains('mm-btn-compile'), `${mmb.className}`);
    }
    $('ctrlMosaicDetImgsz').value = '640'; fire($('ctrlMosaicDetImgsz'), 'input'); fire($('ctrlMosaicDetImgsz')); await sleep(30);
    w.eval("_mmLastModelData = {detection: [], restoration: []}; _updateManageModelsButton();");
    check('mm-btn: no models -> green Get Models', mmb.classList.contains('mm-btn-get') && mmb.textContent === 'Get Models', `"${mmb.textContent}" ${mmb.className}`);
    w.eval("_mmLastModelData = null; _updateManageModelsButton();");
    if (typeof w.populateMosaicModelDropdowns === 'function') { await w.populateMosaicModelDropdowns(); await sleep(50); }
    check('mm-btn: back to plain after the list refills', mmb.textContent === 'Manage Models', `"${mmb.textContent}"`);
    $('ctrlMosaicDetTrt').checked = false; fire($('ctrlMosaicDetTrt')); await sleep(30);   // back to no engine gate
  }

  // T20b: the baked defaults reach the panel (markup + edition)
  {
    check('defaults: Detection Batch 8', $('ctrlMosaicDetBatch').value === '8', `got=${$('ctrlMosaicDetBatch').value}`);
    check('defaults: Score 0.20', $('ctrlMosaicDetScore').value === '20', `got=${$('ctrlMosaicDetScore').value}`);
    check('defaults: Temporal Fix 1', $('ctrlMosaicTemporalFix').value === '1', `got=${$('ctrlMosaicTemporalFix').value}`);
    const secOpts = Array.from($('ctrlMosaicSecondary').options).map(o => o.value);
    if (CUDA) check('defaults: Secondary offers RTX on cuda', secOpts.includes('rtx-4x'));
    else check('defaults: Secondary has no RTX entries off-cuda', !secOpts.some(v => v.startsWith('rtx')), secOpts.join(','));
  }

  const modes = [
    { name: 'restore', set: () => {}, key: null },
    { name: 'preview-detection', set: () => { $('ctrlMosaicMaskPreview').checked = true; fire($('ctrlMosaicMaskPreview')); }, key: 'mosaic_mask_preview' },
    { name: 'add-mosaic', set: () => { $('ctrlMosaicCensor').checked = true; fire($('ctrlMosaicCensor')); }, key: 'mosaic_censor' },
    { name: 'upscale', set: () => { $('ctrlMosaicUpscale').checked = true; fire($('ctrlMosaicUpscale')); }, key: 'mosaic_upscale_only' },
  ];
  for (const m of modes) {
    ['ctrlMosaicMaskPreview', 'ctrlMosaicCensor', 'ctrlMosaicUpscale'].forEach(id => { $(id).checked = false; fire($(id)); });
    m.set(); await sleep(30);
    const btn = $('restoreSaveBtn');
    check(`${m.name}: Restore & Save enabled`, !btn.disabled, `label="${btn.textContent}"`);
    const before = posts.length;
    btn.click(); await sleep(300);
    const p = posts.slice(before).find(x => x.url === '/api/mosaic-full');
    check(`${m.name}: click reaches POST /api/mosaic-full`, !!p);
    if (p && m.key) check(`${m.name}: payload ${m.key} = true`, !!(p.body && p.body.params && p.body.params.mosaic && p.body.params.mosaic[m.key] === true));
    // CM-227 (T18): Crop Cap rides in every payload and defaults to 0 = Off
    if (p && m.name === 'restore') check('restore: payload mosaic_crop_cap = 0 (Off by default)', !!(p.body && p.body.params && p.body.params.mosaic) && p.body.params.mosaic.mosaic_crop_cap === 0, `got=${p.body && p.body.params && p.body.params.mosaic && p.body.params.mosaic.mosaic_crop_cap}`);
    // let the fake job "finish" so the modal closes and state resets
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200);
    progress = { status: 'idle' };
    try { $('progressCancel').click(); } catch {}
    S.previewMode = false; S.previewReady = false; S.previewSegment = null;
    if (typeof w._updateRestorationButtonStates === 'function') w._updateRestorationButtonStates();
  }

  // T17: Upscale mode labels, greying, Test Frame
  ['ctrlMosaicMaskPreview', 'ctrlMosaicCensor'].forEach(id => { $(id).checked = false; fire($(id)); });
  $('ctrlMosaicUpscale').checked = true; fire($('ctrlMosaicUpscale')); await sleep(30);
  check('upscale: Restore reads Upscale', $('restoreBtn').textContent === 'Upscale', `"${$('restoreBtn').textContent}"`);
  check('upscale: Restore & Save reads Upscale & Save', $('restoreSaveBtn').textContent === 'Upscale & Save', `"${$('restoreSaveBtn').textContent}"`);
  check('upscale: Test Frame disabled', $('restoreDetectBtn').disabled === true);
  check('upscale: Detection model greyed', $('ctrlMosaicDetModel').disabled === true);
  check('upscale: Max Clip greyed', $('ctrlMosaicMaxClip').disabled === true);
  check('upscale: Output Upscale live', $('ctrlOutputUpscale').disabled === false);
  check('upscale: Upscale Scaler live', $('ctrlUpscaleScaler').disabled === false);
  check('upscale: Scaler Quality live', $('ctrlMosaicSecondaryQuality').disabled === false);
  check('upscale: Crop Cap greyed (no scenes in Upscale mode)', $('ctrlMosaicCropCap').disabled === true);   // CM-227 (T18)
  check('upscale: Add Mosaic / Preview Detection unchecked (exclusive)', !$('ctrlMosaicCensor').checked && !$('ctrlMosaicMaskPreview').checked);
  check('upscale: Output Upscale sits under Restoration', !!$('ctrlOutputUpscale').closest('#sectMosaicRestoration'));
  check('upscale: Run Files still in Encoder', !!$('ctrlRunReport').closest('#sectEncoder'));
  // leave the mode: everything back
  $('ctrlMosaicUpscale').checked = false; fire($('ctrlMosaicUpscale')); await sleep(30);
  check('leave upscale: Detection model live again', $('ctrlMosaicDetModel').disabled === false);
  check('leave upscale: Crop Cap live again', $('ctrlMosaicCropCap').disabled === false);                    // CM-227 (T18)
  // CM-227 (T18): Crop Cap 512 reaches the full-run payload and the summary; Off leaves the summary alone
  {
    $('ctrlMosaicCropCap').value = '512'; fire($('ctrlMosaicCropCap')); await sleep(30);
    const before = posts.length;
    $('restoreSaveBtn').click(); await sleep(300);
    const p = posts.slice(before).find(x => x.url === '/api/mosaic-full');
    check('crop cap: 512 reaches the payload', !!(p && p.body && p.body.params && p.body.params.mosaic && p.body.params.mosaic.mosaic_crop_cap === 512));
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200);
    progress = { status: 'idle' };
    try { w.mosaicFullModal && w.mosaicFullModal.classList && w.mosaicFullModal.classList.add('hidden'); } catch (e) {}
    $('ctrlMosaicCropCap').value = '0'; fire($('ctrlMosaicCropCap')); await sleep(30);
  }
  // CM-236 (T22): the pre-run VRAM check. Fits -> no dialog, the run starts. Does not fit ->
  // the dialog; Cancel aborts (no POST); Apply sets Crop Cap + Async on the PANEL and the
  // payload carries them; Run anyway posts the unchanged settings.
  {
    check('vram: modal markup present', !!$('vramModal') && !!$('vramModalApply') && !!$('vramModalAnyway') && !!$('vramModalCancel'));
    $('ctrlMosaicCropCap').value = '0'; fire($('ctrlMosaicCropCap'));
    $('ctrlAsyncEncoder').checked = true; fire($('ctrlAsyncEncoder')); await sleep(30);
    const NOFIT = { fits: false, need_mb: 8080, free_mb: 4291, free_now_mb: 7014, total_mb: 8123, footprint_mb: 2723,
      footprint_source: 'measured', ledger_when: '2026-09-26 15:32', width: 3840, height: 2160, file: 'a.mp4', files_checked: 1,
      parts: 'scenes 5512 (crops up to 1920 px) + detector 248 + decode 379 + paste 379 + async 506 + slack 1053',
      levers: [ { kind: 'crop_cap', value: 1024, need_mb: 3544, text: 'Crop Cap 1024 px (...)' },
                { kind: 'async_off', value: false, need_mb: null, text: 'Async Encode off (-506 MB)' },
                { kind: 'det_batch', value: 4, need_mb: null, text: 'Detection Batch 4' } ] };
    // fits: no dialog
    w.__vramMock = null;
    let before = posts.length;
    $('restoreSaveBtn').click(); await sleep(300);
    check('vram fits: no dialog, run posts', $('vramModal').classList.contains('hidden') && !!posts.slice(before).find(x => x.url === '/api/mosaic-full'));
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200); progress = { status: 'idle' };
    // does not fit: dialog, cancel
    w.__vramMock = NOFIT;
    before = posts.length;
    $('restoreSaveBtn').click(); await sleep(300);
    check('vram no-fit: dialog shown', !$('vramModal').classList.contains('hidden'));
    check('vram no-fit: message carries the numbers', /8080 MB/.test($('vramModalMessage').textContent) && /4291 MB/.test($('vramModalMessage').textContent) && /measured/.test($('vramModalMessage').textContent));
    check('vram no-fit: message names the applicable levers and the advice', /Apply levers sets: Crop Cap 1024/.test($('vramModalMessage').textContent) && /by hand: Detection Batch 4/.test($('vramModalMessage').textContent));
    $('vramModalCancel').click(); await sleep(200);
    check('vram cancel: hidden, no run posted', $('vramModal').classList.contains('hidden') && !posts.slice(before).find(x => x.url === '/api/mosaic-full'));
    check('vram cancel: panel untouched', $('ctrlMosaicCropCap').value === '0' && $('ctrlAsyncEncoder').checked === true);
    // does not fit: run anyway
    before = posts.length;
    $('restoreSaveBtn').click(); await sleep(300);
    $('vramModalAnyway').click(); await sleep(300);
    let p = posts.slice(before).find(x => x.url === '/api/mosaic-full');
    check('vram anyway: run posts with the unchanged settings', !!p && p.body.params.mosaic.mosaic_crop_cap === 0 && p.body.params.encoder.async_encoder === true);
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200); progress = { status: 'idle' };
    // does not fit: apply levers
    before = posts.length;
    $('restoreSaveBtn').click(); await sleep(300);
    $('vramModalApply').click(); await sleep(300);
    p = posts.slice(before).find(x => x.url === '/api/mosaic-full');
    check('vram apply: panel shows Crop Cap 1024 and Async off', $('ctrlMosaicCropCap').value === '1024' && $('ctrlAsyncEncoder').checked === false);
    check('vram apply: payload carries the levers', !!p && p.body.params.mosaic.mosaic_crop_cap === 1024 && p.body.params.encoder.async_encoder === false);
    check('vram apply: panel copy in the report matches', !!p && p.body.params.panel && String(p.body.params.panel.ctrlMosaicCropCap) === '1024');
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200); progress = { status: 'idle' };
    // the check is asked with the params (single file: files null)
    const q = posts.slice(before).find(x => x.url === '/api/vram-preflight');
    check('vram: the check posts the run params', !!q && q.body.params && q.body.params.mosaic && q.body.files === null);
    w.__vramMock = null;
    $('ctrlMosaicCropCap').value = '0'; fire($('ctrlMosaicCropCap')); $('ctrlAsyncEncoder').checked = false; fire($('ctrlAsyncEncoder')); await sleep(30);
  }
  // T23d (CM-239): while the pre-run VRAM check runs, the button says so and
  // is disabled; a second click is ignored; exactly ONE run posts; the label
  // comes back afterwards.
  {
    const btn = $('restoreSaveBtn');
    const label0 = btn.textContent;
    w.__vramDelayMs = 400;
    const before = posts.length;
    btn.click(); await sleep(120);
    check('vram guard: button disabled while checking', btn.disabled === true, `disabled=${btn.disabled}`);
    check('vram guard: button reads Checking VRAM...', /Checking VRAM/.test(btn.textContent), `text=${btn.textContent}`);
    btn.click(); btn.click(); await sleep(700);
    const runs = posts.slice(before).filter(x => x.url === '/api/mosaic-full').length;
    const checks = posts.slice(before).filter(x => x.url === '/api/vram-preflight').length;
    check('vram guard: one check, one run for three clicks', runs === 1 && checks === 1, `runs=${runs} checks=${checks}`);
    check('vram guard: label restored after the check', btn.textContent === label0, `text=${btn.textContent}`);
    w.__vramDelayMs = 0;
    progress = { status: 'complete', frame: 10, detections: 0, restorations: 0 }; await sleep(1200); progress = { status: 'idle' };
  }
  // T20d (CM-234): Hardware decode -- Auto by default, Off reaches the payload and the Add Mosaic request
  {
    check('hw decode: control present, default auto', !!$('ctrlHwDecode') && $('ctrlHwDecode').value === 'auto', `got=${$('ctrlHwDecode') && $('ctrlHwDecode').value}`);
    check('hw decode: exactly auto|off offered', Array.from($('ctrlHwDecode').options).map(o => o.value).join(',') === 'auto,off');
    $('ctrlHwDecode').value = 'off'; fire($('ctrlHwDecode')); await sleep(30);
    const p0 = w.eval('gatherMosaicParams()');
    check('hw decode: off reaches mosaic_hw_decode', !!(p0.mosaic && p0.mosaic.mosaic_hw_decode === 'off'), `got=${p0.mosaic && p0.mosaic.mosaic_hw_decode}`);
    check('hw decode: in the preset schema', w.eval("MOSAIC_CONFIG_CONTROLS.indexOf('ctrlHwDecode') >= 0"));
    $('ctrlHwDecode').value = 'auto'; fire($('ctrlHwDecode')); await sleep(30);
  }
  check('leave upscale: Test Frame live again', $('restoreDetectBtn').disabled === false);
  check('leave upscale: labels back', $('restoreBtn').textContent === 'Restore' && $('restoreSaveBtn').textContent === 'Restore & Save');

  // T17: segment -> Preview -> Save Segment (in upscale mode)
  $('ctrlMosaicUpscale').checked = true; fire($('ctrlMosaicUpscale'));
  S.segmentStartTime = 1.0; S.segmentEndTime = 3.0;
  if (typeof w._updateRestorationButtonStates === 'function') w._updateRestorationButtonStates();
  check('segment: Upscale (segment) enabled with a marked segment', !$('restoreBtn').disabled);
  let before = posts.length;
  $('restoreBtn').click(); await sleep(300);
  const seg = posts.slice(before).find(x => x.url === '/api/mosaic-segment');
  check('segment: click reaches POST /api/mosaic-segment', !!seg, seg ? `start=${seg.body.start_time} end=${seg.body.end_time}` : '');
  progress = { status: 'complete', frame: 120, detections: 0, restorations: 0 }; await sleep(1200); progress = { status: 'idle' };
  try { $('progressCancel').click(); } catch {}
  check('segment: Preview lights up', !$('restorePreviewBtn').disabled && !!S.previewReady);
  check('segment: previewSegment recorded', !!S.previewSegment && S.previewSegment.start === 1.0 && S.previewSegment.end === 3.0);
  $('restorePreviewBtn').click(); await sleep(50);
  check('preview: & Save reads Save Segment', $('restoreSaveBtn').textContent === 'Save Segment' && !$('restoreSaveBtn').disabled, `"${$('restoreSaveBtn').textContent}"`);
  before = posts.length;
  $('restoreSaveBtn').click(); await sleep(200);
  const sv = posts.slice(before).find(x => x.url === '/api/save-segment');
  check('preview: Save Segment posts /api/save-segment with the times', !!sv && sv.body.start_time === 1.0 && sv.body.end_time === 3.0);
  check('preview: confirmation shown', posts.slice(before).some(x => x.url === 'alert' && /Saved /.test(x.body)));
  $('restorePreviewBtn').click(); await sleep(50);   // Back
  check('back: & Save reads Upscale & Save again', $('restoreSaveBtn').textContent === 'Upscale & Save', `"${$('restoreSaveBtn').textContent}"`);

  console.log(`\n${fails.length ? 'FAIL' : 'PASS'}: ${fails.length} failed, load-time errors ${errors.length}`);
  process.exit((fails.length || errors.length) ? 2 : 0);
}, 1500);
