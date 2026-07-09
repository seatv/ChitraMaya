# packaging/windows/chitramaya_entrypoint.py
#
# Tiny bootstrap so PyInstaller freezes ChitraMaya as a PACKAGE import, not by
# pointing at chitramaya/__main__.py directly. __main__ and its imports
# (chitramaya.server, tools.*) use package-relative resolution; freezing the
# module itself as the entry script breaks that ("attempted relative import
# with no known parent package"). Importing chitramaya.__main__ here keeps the
# package context intact.
from __future__ import annotations


def main() -> int:
    from chitramaya.__main__ import main as _main
    return int(_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
