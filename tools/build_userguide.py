"""Render docs/UserGuide.md and docs/ScalingGuide.md to self-contained HTML
(inline CSS, no scripts, no external requests) for shipping next to the exe
and for get.chitramaya.org.

    python -m tools.build_userguide            # writes docs/UserGuide.html, docs/ScalingGuide.html
    python -m tools.build_userguide --check    # exit 1 if the HTML is older than the Markdown

Needs the `markdown` package (pip install markdown); nothing else. The
generated files are committed so the packager (and a reader on GitLab) never
needs this tool -- re-run it after editing the Markdown. Links between the two
documents are rewritten from .md to .html; everything else is left alone.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

DOCS = [("UserGuide.md", "UserGuide.html", "ChitraMaya User Guide"),
        ("ScalingGuide.md", "ScalingGuide.html", "ChitraMaya Scaling Guide")]

CSS = """
:root { --fg:#1f2328; --bg:#ffffff; --muted:#59636e; --line:#d1d9e0; --code:#f6f8fa; --link:#0969da; --note:#ddf4ff; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --bg:#0d1117; --muted:#9198a1; --line:#30363d; --code:#161b22; --link:#4493f8; --note:#0c2d6b; }
}
html { -webkit-text-size-adjust: 100%; }
body { margin:0; padding:0 16px 48px; background:var(--bg); color:var(--fg);
       font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 2em; margin: 1.2em 0 .4em; border-bottom: 1px solid var(--line); padding-bottom: .3em; }
h2 { font-size: 1.5em; margin: 1.8em 0 .5em; border-bottom: 1px solid var(--line); padding-bottom: .3em; }
h3 { font-size: 1.2em; margin: 1.4em 0 .4em; }
p, li { max-width: 72ch; }
a { color: var(--link); }
code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: .92em; }
code { background: var(--code); padding: .1em .35em; border-radius: 4px; }
pre { background: var(--code); padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: 6px 12px; text-align: left; vertical-align: top; }
th { background: var(--code); }
blockquote { margin: 1em 0; padding: .6em 1em; border-left: 4px solid var(--link); background: var(--note); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2em 0; }
img { max-width: 100%; height: auto; }
.toc { background: var(--code); padding: 10px 18px; border-radius: 6px; margin: 1em 0; }
.toc ul { margin: 0; padding-left: 1.2em; }
.stamp { color: var(--muted); font-size: .85em; margin-top: 3em; border-top: 1px solid var(--line); padding-top: .6em; }
"""


def _rewrite_links(html: str) -> str:
    # ScalingGuide.md -> ScalingGuide.html etc. (same folder, both shipped)
    for src, dst, _ in DOCS:
        html = html.replace(f'href="{src}', f'href="{dst}')
    return html


def render(md_path: Path, title: str) -> str:
    try:
        import markdown
    except ImportError:
        raise SystemExit("pip install markdown  (the renderer; the generated HTML is committed, so only "
                         "whoever edits the Markdown needs it)")
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
    body = _rewrite_links(body)
    stamp = _dt.date.today().isoformat()
    ver = ""
    try:
        sys.path.insert(0, str(md_path.resolve().parents[1]))
        from chitramaya import __version__ as ver  # noqa
    except Exception:
        ver = ""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
        f"{body}\n<p class=\"stamp\">Generated from docs/{md_path.name} on {stamp}"
        f"{(' for ChitraMaya ' + ver) if ver else ''}. The Markdown is the source; edit that.</p>\n"
        "</main>\n</body>\n</html>\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=None, help="docs folder (default: <repo>/docs)")
    ap.add_argument("--check", action="store_true", help="only check that the HTML is newer than the Markdown")
    a = ap.parse_args(argv)
    docs = Path(a.docs) if a.docs else Path(__file__).resolve().parents[1] / "docs"
    rc = 0
    for src, dst, title in DOCS:
        md = docs / src
        out = docs / dst
        if not md.is_file():
            print(f"[userguide] missing {md}")
            rc = 1
            continue
        if a.check:
            stale = (not out.is_file()) or out.stat().st_mtime < md.stat().st_mtime
            print(f"[userguide] {dst}: {'STALE -- re-run tools/build_userguide.py' if stale else 'up to date'}")
            rc |= int(stale)
            continue
        out.write_text(render(md, title), encoding="utf-8")
        print(f"[userguide] wrote {out} ({out.stat().st_size // 1024} KB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
