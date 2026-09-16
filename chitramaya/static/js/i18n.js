// chitramaya/static/js/i18n.js -- CM-182: UI language switch + string loader.
//
// The English strings were extracted from ui.html / the JS message calls
// into static/i18n/en.json (scratch/i18n_extract.py); translators fill the
// same keys in <lang>.json. This file applies a language at runtime by the
// SAME rules the extractor used to build the keys, matching on the English
// text where an element has no id, so nothing in the markup had to change:
//
//   <id>.title | .placeholder | .aria-label | .alt   -> the attribute on #id
//   <id>.value                                       -> input[type=button] value
//   <select>.opt.<value>                             -> that option's text
//   <id>.text                                        -> #id's own text nodes
//                                                       (innerHTML when the
//                                                       English carries tags)
//   <ctx>.<slug>.<attr> / <ctx>.<slug>               -> inside #ctx (or the
//                                                       document for "root"),
//                                                       the element whose
//                                                       attribute / own text
//                                                       equals the English
//   js.<file>.L<n>.<i>                               -> alert/showToast/... :
//                                                       the message text is
//                                                       looked up by English
//
// Every lookup falls back to English per key: a missing or stale
// translation never blanks a control. Language is the ctrlUiLanguage
// select in the gear menu, persisted with Save Settings (ChitraMaya-
// config.json, key ctrlUiLanguage). ASCII-only source; the strings live in
// the JSON files.
(function () {
  'use strict';
  const LANGS = { 'en': 'English', 'zh-CN': '简体中文', 'zh-TW': '繁體中文', 'th': 'ไทย' };  // th = machine draft (09-15)
  const ATTRS = ['title', 'placeholder', 'aria-label', 'alt', 'data-tooltip'];
  let EN = null;            // key -> english
  let CUR = null;           // key -> translation (current language)
  let curLang = 'en';
  const originals = new Map();  // element -> {attr/text: english} so switching back restores
  let jsMap = new Map();        // english message -> translation
  let patched = false;

  function norm(s) { return String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); }

  async function loadLang(lang) {
    const r = await fetch('/static/i18n/' + encodeURIComponent(lang) + '.json?v=' + Date.now(), { cache: 'no-store' });
    if (!r.ok) throw new Error('i18n: ' + lang + ' HTTP ' + r.status);
    return await r.json();
  }

  function remember(el, slot, value) {
    let m = originals.get(el);
    if (!m) { m = {}; originals.set(el, m); }
    if (!(slot in m)) m[slot] = value;
  }

  function ownText(el) {
    let t = '';
    for (const c of el.childNodes) if (c.nodeType === 3) t += c.nodeValue;
    return norm(t);
  }

  function setOwnText(el, text) {
    // Replace the element's own text nodes with one node carrying `text`,
    // leaving child elements (inputs, spans, icons) in place.
    let first = null;
    for (const c of Array.from(el.childNodes)) {
      if (c.nodeType !== 3) continue;
      if (!norm(c.nodeValue)) continue;
      if (first === null) { first = c; c.nodeValue = text; }
      else c.nodeValue = '';
    }
    if (first === null) el.appendChild(document.createTextNode(text));
  }

  function scopeOf(ctx) {
    if (!ctx || ctx === 'root' || ctx === 'main') return document;
    return document.getElementById(ctx) || null;
  }

  function findByAttr(scope, attr, english) {
    if (!scope) return null;
    const q = scope.querySelectorAll('[' + attr + ']');
    for (const el of q) {
      const m = originals.get(el);
      const base = (m && ('attr:' + attr) in m) ? m['attr:' + attr] : el.getAttribute(attr);
      if (norm(base) === english) return el;
    }
    return null;
  }

  function findByText(scope, english) {
    if (!scope) return null;
    const all = scope.querySelectorAll('*');
    for (const el of all) {
      if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') continue;
      const m = originals.get(el);
      if (m && 'text' in m) { if (m.text === english) return el; continue; }
      if (m && 'html' in m) { if (m.html === english) return el; continue; }
      if (ownText(el) === english) return el;
      if (english.indexOf('<') >= 0 && norm(el.innerHTML) === english) return el;
    }
    return null;
  }

  function applyKey(key, english, value) {
    // value === english restores the original (English) form.
    const parts = key.split('.');
    if (parts[0] === 'js') return;  // messages: handled through jsMap
    const last = parts[parts.length - 1];
    // <select>.opt.<value>
    const oi = parts.indexOf('opt');
    if (oi > 0) {
      const sel = document.getElementById(parts.slice(0, oi).join('.'));
      if (!sel) return;
      const optVal = parts.slice(oi + 1).join('.');
      for (const o of sel.options) {
        if (o.value === optVal) { remember(o, 'text', norm(o.textContent)); o.textContent = value; return; }
      }
      return;
    }
    if (ATTRS.indexOf(last) >= 0 || last === 'value') {
      const attr = last;
      if (parts.length === 2) {
        const el = document.getElementById(parts[0]);
        if (!el) return;
        if (attr === 'value') { remember(el, 'attr:value', el.value); el.value = value; return; }
        remember(el, 'attr:' + attr, el.getAttribute(attr)); el.setAttribute(attr, value); return;
      }
      // <ctx>.<slug>.<attr>: locate by English text inside the scope
      const el = findByAttr(scopeOf(parts[0]), attr, english);
      if (!el) return;
      remember(el, 'attr:' + attr, el.getAttribute(attr)); el.setAttribute(attr, value); return;
    }
    if (last === 'text' && parts.length === 2) {
      const el = document.getElementById(parts[0]);
      if (!el) return;
      if (english.indexOf('<') >= 0) { remember(el, 'html', norm(el.innerHTML)); el.innerHTML = value; return; }
      remember(el, 'text', ownText(el)); setOwnText(el, value); return;
    }
    // <ctx>.<slug>: own text (or inline html) inside the scope
    const el = findByText(scopeOf(parts[0]), english);
    if (!el) return;
    if (english.indexOf('<') >= 0) { remember(el, 'html', norm(el.innerHTML)); el.innerHTML = value; }
    else { remember(el, 'text', ownText(el)); setOwnText(el, value); }
  }

  function applyAll() {
    if (!EN) return;
    const dict = CUR || {};
    for (const key of Object.keys(EN)) {
      if (key === '_meta') continue;
      const english = norm(EN[key]);
      let v = dict[key];
      if (v == null || !norm(v)) v = EN[key];   // per-key English fallback
      try { applyKey(key, english, String(v)); } catch (e) { /* one bad key never stops the rest */ }
    }
    // JS messages: English text -> translation
    jsMap = new Map();
    for (const key of Object.keys(EN)) {
      if (key.indexOf('js.') !== 0) continue;
      const v = dict[key];
      if (v && norm(v) && v !== EN[key]) jsMap.set(norm(EN[key]), String(v));
    }
    patchMessageFns();
    try { document.documentElement.setAttribute('lang', curLang); } catch (e) {}
  }

  function tr(s) {
    if (typeof s !== 'string') return s;
    const hit = jsMap.get(norm(s));
    return hit == null ? s : hit;
  }

  function patchMessageFns() {
    if (patched) return;
    patched = true;
    const names = ['showToast', 'showConfirm', 'setStatus', 'showError'];
    for (const n of names) {
      const fn = window[n];
      if (typeof fn !== 'function' || fn.__i18n) continue;
      const wrapped = function () {
        const args = Array.from(arguments).map(a => tr(a));
        return fn.apply(this, args);
      };
      wrapped.__i18n = true;
      window[n] = wrapped;
    }
    if (!window.alert.__i18n) {
      const a = window.alert.bind(window);
      const w = function (m) { return a(tr(m)); };
      w.__i18n = true;
      window.alert = w;
    }
  }

  async function setLanguage(lang) {
    lang = LANGS[lang] ? lang : 'en';
    try {
      if (!EN) EN = await loadLang('en');
      CUR = (lang === 'en') ? null : await loadLang(lang);
      curLang = lang;
      applyAll();
      const sel = document.getElementById('ctrlUiLanguage');
      if (sel && sel.value !== lang) sel.value = lang;
      console.log('[i18n] language: ' + lang + (CUR ? ' (' + (Object.keys(CUR).length - 1) + ' strings)' : ''));
    } catch (e) {
      console.warn('[i18n] could not apply ' + lang + ': ' + e);
    }
  }

  window.I18N = { setLanguage, tr, get language() { return curLang; }, LANGS };

  document.addEventListener('DOMContentLoaded', () => {
    const sel = document.getElementById('ctrlUiLanguage');
    if (!sel) return;
    // The gear menu closes on any document click; keep the select usable.
    sel.addEventListener('click', e => e.stopPropagation());
    sel.addEventListener('change', async e => {
      e.stopPropagation();
      await setLanguage(sel.value);
      // Persist with the rest of the settings (ChitraMaya-config.json).
      if (typeof window.saveConfig === 'function') { try { window.saveConfig(); } catch (err) {} }
    });
  });
})();
