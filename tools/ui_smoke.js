// tools/ui_smoke.js
// DEV-ONLY UI load + click-through in jsdom (no browser, no server, no GPU).
// Loads templates/ui.html with every static script inlined, a stub fetch for
// the endpoints the UI calls at load and in the gear menu, and then clicks
// through: TRT rows per edition, saved configurations (list / one-click load
// / save-as / delete with its confirm ON TOP / Enter / Esc), Save-Load-Reset
// Settings, the Console drawer (open, poll, Esc, X), Save The Children,
// fullscreen. Exit code 2 on any load-time error (a ReferenceError anywhere
// in the bundle kills everything after it -- the T9e regression, 09-09).
//
//   npm install jsdom          (once, anywhere; or in a scratch folder)
//   node tools/ui_smoke.js <repo root> cuda
//   node tools/ui_smoke.js <repo root> rocm
//   node tools/ui_smoke.js <repo root> xpu
//
// Pair with `python -m tools.verify_ui_js` (static, no node needed).
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
const presets = { 'bench A': { ctrlCodec: 'av1', ctrlQP: '30', ctrlMosaicMaxClip: '90', ctrlMosaicDetTrt: true, ctrlMosaicRestTrt: true, outputDir: 'X:\\outA', tempDir: 'X:\\tmp', debug: false, perf_test: false } };
const calls = [];
let consoleCursor = 0;
const dom = new JSDOM(html, {
  runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://127.0.0.1:5000/', virtualConsole: vc,
  beforeParse(window) {
    window.fetch = async (url, opts = {}) => {
      const method = (opts.method || 'GET').toUpperCase();
      calls.push(method + ' ' + url);
      let body = {};
      if (url === '/api/presets') body = { presets: Object.keys(presets), dir: 'presets' };
      else if (url.startsWith('/api/presets/')) {
        const name = decodeURIComponent(url.slice('/api/presets/'.length));
        if (method === 'GET') body = presets[name] || { error: 'no such preset' };
        else if (method === 'POST') { presets[name] = JSON.parse(opts.body); body = { ok: true, name, keys: Object.keys(presets[name]).length }; }
        else if (method === 'DELETE') { delete presets[name]; body = { ok: true }; }
      } else if (url === '/api/load-config') body = { ctrlCodec: 'hevc', ctrlQP: '18', ctrlMosaicMaxClip: '300', ctrlMosaicDetTrt: true, ctrlMosaicRestTrt: true, outputDir: 'X:\\out', tempDir: 'X:\\tmp' };
      else if (url === '/api/default-config') body = { ctrlCodec: 'hevc', ctrlQP: '20', ctrlMosaicMaxClip: '90' };
      else if (url === '/api/save-config') body = { ok: true };
      else if (url.startsWith('/api/console')) { consoleCursor += 2; body = { lines: ['[line ' + consoleCursor + ']', '[line ' + (consoleCursor + 1) + ']'], next: consoleCursor }; }
      else if (url.startsWith('/api/list-mosaic-models')) body = { detection: [], restoration: [] };
      else if (url.startsWith('/api/session-status')) body = { running: false };
      else body = {};
      return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
    };
    window.alert = (m) => calls.push('alert ' + m);
    window.HTMLCanvasElement.prototype.getContext = () => null;
    window.HTMLMediaElement.prototype.play = async () => {};
    window.HTMLMediaElement.prototype.pause = () => {};
    window.HTMLMediaElement.prototype.load = () => {};
    window.pywebview = { api: { open_url: (u) => calls.push('open_url ' + u), toggle_fullscreen: () => calls.push('toggle_fullscreen') } };
  },
});
dom.window.addEventListener('error', (e) => errors.push('window.onerror: ' + (e.message || e)));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
setTimeout(async () => {
  const w = dom.window, d = w.document, out = [];
  const $ = (id) => d.getElementById(id);
  const vis = (el) => el && !el.classList.contains('hidden') && el.style.display !== 'none';
  out.push(`edition=${EDITION} load-time errors: ${errors.length}`);
  errors.forEach(e => out.push('  ' + e));
  // TRT rows
  const detRow = $('ctrlMosaicDetTrt').closest('.ctrl-row'), restRow = $('ctrlMosaicRestTrt').closest('.ctrl-row');
  out.push(`TRT rows visible: det=${detRow.style.display !== 'none'} rest=${restRow.style.display !== 'none'}  checked: det=${$('ctrlMosaicDetTrt').checked} rest=${$('ctrlMosaicRestTrt').checked}  (startup config carried both TRUE)`);
  // gear menu + presets
  $('configBtn').click(); await sleep(150);
  const items = [...d.querySelectorAll('#cfgPresetList .config-menu-item.preset')].map(b => b.textContent);
  out.push(`gear open: ${vis($('configMenu'))}  preset items: ${JSON.stringify(items)}  sep visible: ${$('cfgPresetSep').style.display !== 'none'}`);
  d.querySelector('#cfgPresetList .config-menu-item.preset').click(); await sleep(250);
  out.push(`after one-click load: ctrlQP=${$('ctrlQP').value} maxclip=${$('ctrlMosaicMaxClip').value} outputPath=${$('outputPath').value} TRT checked det=${$('ctrlMosaicDetTrt').checked} rest=${$('ctrlMosaicRestTrt').checked} menu hidden=${!vis($('configMenu'))}`);
  // manage modal: save as, delete (confirm must be reachable = on top)
  $('configBtn').click(); $('cfgPresets').click(); await sleep(150);
  out.push(`preset modal open: ${vis($('presetModal'))}  select options: ${$('presetSelect').options.length}`);
  $('presetName').value = 'bench B'; $('presetSave').click(); await sleep(200);
  out.push(`save-as: hint="${$('presetHint').textContent.slice(0, 40)}"  select options now: ${$('presetSelect').options.length}`);
  $('presetSelect').value = 'bench B'; $('presetDelete').click(); await sleep(150);
  const cm = $('confirmModal'), pm = $('presetModal');
  const order = [...d.querySelectorAll('.modal-overlay')].map(x => x.id);
  const zc = w.getComputedStyle(cm).zIndex, zp = w.getComputedStyle(pm).zIndex;
  out.push(`delete -> confirm visible: ${vis(cm)}  z(confirm)=${zc} z(preset)=${zp}  DOM order confirm<preset: ${order.indexOf('confirmModal') < order.indexOf('presetModal')}  => confirm ON TOP: ${Number(zc) > Number(zp) || (zc === zp && order.indexOf('confirmModal') > order.indexOf('presetModal'))}`);
  $('confirmYes').click(); await sleep(200);
  $('presetName').value = 'bench C'; $('presetName').dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); await sleep(200);
  out.push(`Enter saves: ${'bench C' in presets}`);
  d.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); out.push(`Esc closes preset modal: ${!vis(pm)}`);
  $('configBtn').click(); $('cfgPresets').click(); await sleep(100);
  out.push(`after confirm: options=${$('presetSelect').options.length} hint="${$('presetHint').textContent.slice(0, 30)}" presets left=${JSON.stringify(Object.keys(presets))}`);
  $('presetClose').click(); out.push(`modal closed: ${!vis(pm)}`);
  // Save / Load / Reset Settings
  $('configBtn').click(); $('cfgSave').click(); await sleep(150);
  out.push(`Save Settings alert: ${calls.some(c => c.startsWith('alert Settings saved'))}`);
  $('configBtn').click(); $('cfgLoad').click(); await sleep(150);
  out.push(`Load Settings: ctrlQP=${$('ctrlQP').value} (expect 18) TRT det=${$('ctrlMosaicDetTrt').checked} rest=${$('ctrlMosaicRestTrt').checked}`);
  $('configBtn').click(); $('cfgReset').click(); await sleep(100); $('confirmYes').click(); await sleep(150);
  out.push(`Reset Defaults: ctrlQP=${$('ctrlQP').value} (expect 20)`);
  // console drawer
  $('cdToggle').click(); await sleep(1300);
  out.push(`console drawer open: ${$('cdPanel').classList.contains('cd-open')} lines polled: ${$('cdBody').textContent.split('\n').filter(Boolean).length} icon=${$('cdToggleIcon').textContent}`);
  d.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  out.push(`Esc closes console: ${!$('cdPanel').classList.contains('cd-open')}`);
  $('cdToggle').click(); $('cdClose').click(); out.push(`X closes console: ${!$('cdPanel').classList.contains('cd-open')}`);
  // Save The Children, fullscreen
  $('stcBtn').click(); out.push(`STC open_url: ${calls.some(c => c.startsWith('open_url'))}`);
  $('fsBtn').click(); out.push(`fullscreen via bridge: ${calls.includes('toggle_fullscreen')}`);
  // elapsed clock wiring present?
  out.push(`elapsed clock elements: ${['progressElapsed', 'batchElapsed'].map(i => i + '=' + !!$(i)).join(' ')}`);
  console.log(out.join('\n'));
  process.exit(errors.length ? 2 : 0);
}, 1500);
