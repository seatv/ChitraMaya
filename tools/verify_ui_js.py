# tools/verify_ui_js.py
"""Static wiring check for the UI's JavaScript bundle (no browser needed).

Field event 2026-09-09: Batch T9e rewrote the saved-configurations block in
init.js and deleted saveConfig / loadConfig / resetDefaults on the way out.
Those are function declarations referenced BY NAME a few lines earlier
(`cfgSave.addEventListener('click', saveConfig)`), so the file parsed, but at
load that line threw ReferenceError and every statement after it -- the whole
saved-configurations block, Save The Children -- never ran. The gear menu
opened and nothing in it did anything. Nothing in the tree would have said so.

This catches that class of break before a zip goes out:

  1. every <script src> the UI page loads exists;
  2. `node --check` (syntax) on each file when node is on PATH;
  3. every bare identifier handed to addEventListener(...) or assigned to an
     on<event> property is declared at top level somewhere in the bundle;
  4. every document.getElementById('literal') in the bundle names an id that
     exists in the page (a WinMerge that took the .js but not the .html).

    python -m tools.verify_ui_js
    python -m tools.verify_ui_js --template chitramaya/templates/ui.html

Exit code 0 = clean; 1 = at least one finding. ASCII-only output.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]

# Browser / library globals that legitimately appear as bare handler names.
_KNOWN_GLOBALS = {
    "alert", "console", "window", "document", "fetch", "requestAnimationFrame",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval", "THREE",
    "location", "history", "navigator", "Promise", "Object", "Array", "JSON",
    "Math", "Number", "String", "Boolean", "Date", "Event", "CustomEvent",
    "URL", "Blob", "FormData", "encodeURIComponent", "decodeURIComponent",
    "parseInt", "parseFloat", "isNaN", "undefined", "null", "true", "false",
    "pywebview", "e", "ev", "evt", "event",
}

_SCRIPT_SRC_RE = re.compile(r"""<script[^>]*\ssrc\s*=\s*["']([^"']+)["']""", re.I)
_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\ssrc=)[^>]*>(.*?)</script>", re.I | re.S)
_DECL_RE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"
    r"|^[ \t]*(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"|^[ \t]*(?:const|let|var)\s*\{([^}]*)\}"
    r"|^[ \t]*(?:const|let|var)\s*\[([^\]]*)\]"
    r"|^[ \t]*class\s+([A-Za-z_$][\w$]*)"
    r"|^[ \t]*(?:window\.)?([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
    re.M,
)
_LISTENER_RE = re.compile(
    r"""\.addEventListener\(\s*['"][\w:-]+['"]\s*,\s*([A-Za-z_$][\w$]*)\s*[,)]"""
)
_ONPROP_RE = re.compile(r"""\.on[a-z]+\s*=\s*([A-Za-z_$][\w$]*)\s*;""")
_GETID_RE = re.compile(r"""document\.getElementById\(\s*['"]([^'"]+)['"]\s*\)""")
_HTML_ID_RE = re.compile(r"""\sid\s*=\s*["']([^"']+)["']""")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _strip_comments(src: str) -> str:
    # Good enough for this bundle: no regex literals or strings containing "//"
    # that matter for declarations / listener names.
    return _COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), src)


def _declared_names(src: str) -> Set[str]:
    names: Set[str] = set()
    for m in _DECL_RE.finditer(src):
        fn, single, destr_obj, destr_arr, cls, assigned = m.groups()
        for n in (fn, single, cls, assigned):
            if n:
                names.add(n)
        for group in (destr_obj, destr_arr):
            if group:
                for part in group.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    # {a, b: c, d = 1} -> a, c, d ; [x, y] -> x, y
                    tail = part.split(":")[-1].split("=")[0].strip()
                    if re.match(r"^[A-Za-z_$][\w$]*$", tail):
                        names.add(tail)
    # Names bound mid-line or in parameter lists: `if (x) { const upd = ... }`,
    # `([id, fn]) => {...}`, `function (a, b) {`, `e => ...`. Over-approximate
    # on purpose -- a handler name that appears NOWHERE is the bug we hunt.
    for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", src):
        names.add(m.group(1))
    for m in re.finditer(r"\(([^()]*)\)\s*(?:=>|\{)", src):
        names.update(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
    for m in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*=>", src):
        names.add(m.group(1))
    return names


def _script_files(template: Path) -> Tuple[List[Path], List[str], str]:
    html = template.read_text(encoding="utf-8", errors="replace")
    files: List[Path] = []
    missing: List[str] = []
    for src in _SCRIPT_SRC_RE.findall(html):
        if src.startswith(("http://", "https://", "//")):
            continue
        rel = src.split("?")[0]
        # Flask url_for('static', filename=...) or a literal /static/... path.
        m = re.search(r"filename\s*=\s*['\"]([^'\"]+)['\"]", rel)
        if m:
            rel = "static/" + m.group(1)
        rel = rel.lstrip("/")
        cand = _ROOT / "chitramaya" / rel
        if cand.exists():
            files.append(cand)
        else:
            missing.append(src)
    return files, missing, html


def _node_check(files: List[Path]) -> List[str]:
    node = shutil.which("node")
    if not node:
        print("[verify-ui-js] node not on PATH; syntax check skipped (regex checks still run)")
        return []
    findings: List[str] = []
    for f in files:
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            first = (r.stderr or r.stdout or "").strip().splitlines()
            findings.append(f"SYNTAX  {f.relative_to(_ROOT)}: {first[0] if first else 'node --check failed'}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="static wiring check for the UI JavaScript bundle")
    ap.add_argument("--template", default=str(_ROOT / "chitramaya" / "templates" / "ui.html"))
    ap.add_argument("--no-ids", action="store_true", help="skip the getElementById-vs-page check")
    ap.add_argument("--strict-ids", action="store_true",
                    help="unknown getElementById ids FAIL (default: listed as warnings -- the bundle still "
                         "carries null-guarded lookups of retired face-swap controls)")
    args = ap.parse_args()

    template = Path(args.template)
    if not template.exists():
        print(f"[verify-ui-js] template not found: {template}")
        return 1

    files, missing, html = _script_files(template)
    findings: List[str] = [f"MISSING <script src> {m}" for m in missing]
    print(f"[verify-ui-js] {template.relative_to(_ROOT) if template.is_relative_to(_ROOT) else template}: "
          f"{len(files)} script files" + (f", {len(missing)} missing" if missing else ""))

    findings += _node_check(files)

    sources: Dict[Path, str] = {f: _strip_comments(f.read_text(encoding="utf-8", errors="replace")) for f in files}
    inline = "\n".join(_strip_comments(s) for s in _INLINE_SCRIPT_RE.findall(html))
    declared: Set[str] = set(_KNOWN_GLOBALS)
    for src in sources.values():
        declared |= _declared_names(src)
    declared |= _declared_names(inline)

    # 3. handler identifiers
    n_handlers = 0
    for f, src in list(sources.items()) + [(template, inline)]:
        for regex in (_LISTENER_RE, _ONPROP_RE):
            for m in regex.finditer(src):
                name = m.group(1)
                n_handlers += 1
                if name not in declared:
                    line = src.count("\n", 0, m.start()) + 1
                    rel = f.relative_to(_ROOT) if f.is_relative_to(_ROOT) else f
                    findings.append(f"UNDECLARED handler '{name}' at {rel}:{line} "
                                    f"(ReferenceError at load; everything after it in that file is dead)")
    print(f"[verify-ui-js] {n_handlers} named handlers checked against {len(declared)} declarations")

    # 4. getElementById literals vs page ids
    warnings: List[str] = []
    if not args.no_ids:
        page_ids = set(_HTML_ID_RE.findall(html))
        # ids created by JS (createElement + .id = '...') count too
        for src in sources.values():
            page_ids |= set(re.findall(r"""\.id\s*=\s*['"]([^'"]+)['"]""", src))
        n_ids = 0
        for f, src in sources.items():
            for m in _GETID_RE.finditer(src):
                n_ids += 1
                if m.group(1) not in page_ids:
                    line = src.count("\n", 0, m.start()) + 1
                    (findings if args.strict_ids else warnings).append(
                        f"NO SUCH ID '{m.group(1)}' referenced at {f.relative_to(_ROOT)}:{line} "
                        f"(page markup missing or not merged -- or a retired control behind a null guard)")
        print(f"[verify-ui-js] {n_ids} getElementById literals checked against {len(page_ids)} page ids")

    if warnings:
        print(f"[verify-ui-js] {len(warnings)} warning(s) (not failing; --strict-ids to fail):")
        for x in warnings:
            print("  " + x)
    if findings:
        print(f"[verify-ui-js] FAIL: {len(findings)} finding(s)")
        for x in findings:
            print("  " + x)
        return 1
    print("[verify-ui-js] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
