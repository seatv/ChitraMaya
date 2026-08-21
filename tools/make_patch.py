# tools/make_patch.py
"""Build a ChitraMaya patch zip by diffing two EXTRACTED release trees.

CM-097 (v1.60.00), ported from the GenSRT patch toolchain. The design
decision that simplifies everything: diff the extracted RELEASED installer
against the extracted NEW installer -- never the local dist/ tree. The
released installer is what users actually have; dist/ may have been
touched since. See claude/ChitraMaya-PatchingNotes-GenSRT.md.

The patch zip contains:
    patch-manifest.json   from/to versions + per-file SHA-256 (old and new)
                          + deletion list
    Apply-Patch.ps1       the apply script (copied from tools/, PURE ASCII
                          -- enforced here; Windows PowerShell 5.1 decodes
                          BOM-less non-ASCII as cp1252 and mangles it)
    payload/...           every changed or added file

Safety properties (each one exists because its failure is expensive):
  * Version guard: Apply verifies every to-be-overwritten file's SHA-256
    matches the source release before touching anything.
  * Refuse from-version == to-version (a patch that does not change the
    version cannot be verified on the user's side).
  * Deletions are recorded and applied (a stale module that still imports
    is worse than a missing one).
  * User data is excluded from the diff entirely: models/ (checkpoints,
    detection engines, TRT sub-engine sets -- machine-specific, must
    NEVER be patched), ChitraMaya-config.json, console/compile logs,
    hf-token.txt, prior patch backups.

Usage (from the repo root, any Python 3.10+):
    python tools/make_patch.py --old <extracted-1.50-dir> \
        --new <extracted-1.60-dir> --out ChitraMaya-patch.zip

Create-chitramaya-patch.ps1 orchestrates the download/extract steps and
calls this. This tool publishes nothing.

ASCII-only stdout, as everywhere in ChitraMaya.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

# Paths (relative, posix-style, case-preserving) excluded from the diff on
# BOTH sides and protected from deletion. Directory prefixes end with "/".
EXCLUDE_PREFIXES = (
    "models/",        # checkpoints + detection engines + TRT sub-engine sets
    "backup-",        # backups left by prior patch applications
)
EXCLUDE_NAMES = (
    "ChitraMaya-config.json",
    "hf-token.txt",
)
EXCLUDE_PATTERNS = (
    "ChitraMaya-console*.log",
    "*-compile-*.log",
    "*.misses.json",
    "Thumbs.db",
    "desktop.ini",
)

_MAIN_EXES = ("ChitraMaya.exe", "ChitraMaya-cli.exe")


def _excluded(rel: str) -> bool:
    for p in EXCLUDE_PREFIXES:
        if rel.startswith(p):
            return True
    name = rel.rsplit("/", 1)[-1]
    if name in EXCLUDE_NAMES:
        return True
    for pat in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_tree(root: Path) -> dict[str, Path]:
    """Map relative posix path -> absolute path for every included file."""
    out: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            ap = Path(dirpath) / name
            rel = ap.relative_to(root).as_posix()
            if not _excluded(rel):
                out[rel] = ap
    return out


def _hash_tree(files: dict[str, Path], label: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    total = len(files)
    for i, (rel, ap) in enumerate(sorted(files.items()), 1):
        hashes[rel] = _sha256(ap)
        if i % 500 == 0 or i == total:
            print(f"[make_patch] hashed {label}: {i}/{total}")
    return hashes


def _read_version(root: Path) -> str | None:
    vp = root / "VERSION.txt"
    if vp.is_file():
        try:
            v = vp.read_text(encoding="ascii", errors="strict").strip()
            return v or None
        except Exception:
            return None
    return None


def _check_ascii(path: Path) -> None:
    data = path.read_bytes()
    for i, b in enumerate(data):
        if b > 0x7F:
            raise SystemExit(
                f"[make_patch] REFUSING: {path} contains a non-ASCII byte "
                f"at offset {i} (0x{b:02x}). Windows PowerShell 5.1 decodes "
                f"BOM-less non-ASCII .ps1 files as cp1252 and mangles them. "
                f"Make the apply script pure ASCII."
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff two extracted ChitraMaya release trees into a patch zip.",
    )
    ap.add_argument("--old", required=True, help="Extracted RELEASED tree (what users have)")
    ap.add_argument("--new", required=True, help="Extracted NEW tree (what they should get)")
    ap.add_argument("--out", default=None, help="Output patch zip path (default: auto-named)")
    ap.add_argument("--from-version", default=None, help="Override old version label")
    ap.add_argument("--to-version", default=None, help="Override new version label")
    ap.add_argument(
        "--apply-script",
        default=str(Path(__file__).parent / "Apply-Patch.ps1"),
        help="Apply script to embed (default: tools/Apply-Patch.ps1)",
    )
    ap.add_argument(
        "--allow-same-version", action="store_true",
        help="TESTING ONLY: skip the from==to version refusal.",
    )
    args = ap.parse_args()

    old_root = Path(args.old).resolve()
    new_root = Path(args.new).resolve()
    for r, label in ((old_root, "--old"), (new_root, "--new")):
        if not r.is_dir():
            raise SystemExit(f"[make_patch] {label} is not a directory: {r}")
        if not any((r / e).is_file() for e in _MAIN_EXES):
            print(f"[make_patch] WARNING: {r} does not contain "
                  f"{' or '.join(_MAIN_EXES)} -- is this an extracted "
                  f"ChitraMaya tree?")

    apply_script = Path(args.apply_script).resolve()
    if not apply_script.is_file():
        raise SystemExit(f"[make_patch] Apply script not found: {apply_script}")
    _check_ascii(apply_script)

    from_v = args.from_version or _read_version(old_root)
    to_v = args.to_version or _read_version(new_root)

    print(f"[make_patch] old: {old_root}  (version: {from_v or 'unknown'})")
    print(f"[make_patch] new: {new_root}  (version: {to_v or 'unknown'})")

    # Version guard (GenSRT: caught a forgotten version bump on the very
    # first real run). When versions are unknown (pre-VERSION.txt
    # releases), fall back to comparing the main exe -- identical
    # executables mean identical embedded __version__.
    if from_v and to_v and from_v == to_v and not args.allow_same_version:
        raise SystemExit(
            f"[make_patch] REFUSING: from-version == to-version "
            f"({from_v}). Bump __version__ / rebuild before patching."
        )
    if not (from_v and to_v):
        for exe in _MAIN_EXES:
            op, np_ = old_root / exe, new_root / exe
            if op.is_file() and np_.is_file() and _sha256(op) == _sha256(np_):
                if not args.allow_same_version:
                    raise SystemExit(
                        f"[make_patch] REFUSING: versions unknown and {exe} "
                        f"is byte-identical in both trees -- this looks like "
                        f"the same release. Bump __version__ / rebuild."
                    )

    old_files = _walk_tree(old_root)
    new_files = _walk_tree(new_root)
    print(f"[make_patch] included files: old={len(old_files)} new={len(new_files)}")

    old_hashes = _hash_tree(old_files, "old")
    new_hashes = _hash_tree(new_files, "new")

    updates: list[dict] = []
    adds: list[dict] = []
    deletions: list[str] = []
    for rel, nh in sorted(new_hashes.items()):
        oh = old_hashes.get(rel)
        if oh is None:
            adds.append({"path": rel, "action": "add", "sha256_new": nh,
                         "size_new": new_files[rel].stat().st_size})
        elif oh != nh:
            updates.append({"path": rel, "action": "update", "sha256_old": oh,
                            "sha256_new": nh,
                            "size_new": new_files[rel].stat().st_size})
    for rel in sorted(old_hashes):
        if rel not in new_hashes:
            deletions.append(rel)

    # Stamp VERSION.txt into the payload even when the OLD release predates
    # it, so patched installs always carry a readable version afterwards.
    synth_version: bytes | None = None
    if "VERSION.txt" not in new_hashes and to_v:
        synth_version = (to_v + "\n").encode("ascii")
        adds.append({"path": "VERSION.txt", "action": "add",
                     "sha256_new": hashlib.sha256(synth_version).hexdigest(),
                     "size_new": len(synth_version)})

    if not updates and not adds and not deletions:
        raise SystemExit("[make_patch] REFUSING: trees are identical after "
                         "exclusions -- nothing to patch.")

    out = Path(args.out) if args.out else Path(
        f"ChitraMaya-patch-{from_v or 'old'}-to-{to_v or 'new'}.zip"
    )
    files_entries = updates + adds
    manifest = {
        "format": 1,
        "product": "ChitraMaya",
        "from_version": from_v,
        "to_version": to_v,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
        "files": files_entries,
        "deletions": deletions,
    }

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("patch-manifest.json", json.dumps(manifest, indent=2))
        z.write(apply_script, "Apply-Patch.ps1")
        for e in files_entries:
            arc = "payload/" + e["path"]
            if e["path"] == "VERSION.txt" and synth_version is not None:
                z.writestr(arc, synth_version)
            else:
                z.write(new_files[e["path"]], arc)

    total_new = sum(e["size_new"] for e in files_entries)
    print(f"[make_patch] DONE: {out}")
    print(f"[make_patch]   updated: {len(updates)}  added: {len(adds)}  "
          f"deleted: {len(deletions)}")
    print(f"[make_patch]   payload (uncompressed): {total_new / (1024*1024):.1f} MB; "
          f"patch zip: {out.stat().st_size / (1024*1024):.1f} MB")
    print(f"[make_patch] Test before publishing: extract the zip, run "
          f"Apply-Patch.ps1 against a COPY of the old tree, then run "
          f"ChitraMaya-cli.exe -self-check in the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
