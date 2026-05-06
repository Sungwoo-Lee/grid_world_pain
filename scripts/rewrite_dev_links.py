#!/usr/bin/env python3
"""ONE-SHOT: rewrite incoming links to docs/develop/* after Phase C migration.

Walks docs/develop/{active,archive}/ to build a filename → new-path map,
then rewrites references like `../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md` to
`../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` in:
  - docs/project/**/*.md
  - docs/environment/**/*.md
  - docs/develop/**/*.md (handles intra-develop links, e.g. plan doc)
  - CLAUDE.md

Usage:
    python scripts/rewrite_dev_links.py            # dry run
    python scripts/rewrite_dev_links.py --apply    # write changes

Idempotent: re-running on already-rewritten paths is a no-op.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVELOP = ROOT / "docs" / "develop"

# Filenames that exist in docs/environment/ rather than docs/develop/ — these
# were mis-linked as `../develop/<file>` but the file actually lives elsewhere.
# Fix to point at the correct location.
KNOWN_MISPLACED = {
    "ENVIRONMENT_SUMMARY.md": "docs/environment/ENVIRONMENT_SUMMARY.md",
}


def build_map() -> dict[str, str]:
    """Map filename -> path relative to docs/develop/ (e.g. 'active/neuromodulation/X.md')."""
    m: dict[str, str] = {}
    for sub in ("active", "archive"):
        base = DEVELOP / sub
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            rel = md.relative_to(DEVELOP)
            m[md.name] = str(rel)
    return m


# Match patterns like "(../)*develop/FILENAME.md" inside link targets.
# Capture: prefix (e.g. "../" or "../../"), filename.
DEVELOP_LINK_RE = re.compile(r"((?:\.\./)+)develop/([A-Za-z0-9_.-]+\.md)")


def rewrite_text(text: str, fmap: dict[str, str], file_in_develop: bool) -> tuple[str, list[str]]:
    """Return (new_text, list of (old, new) substitutions made)."""
    changes: list[str] = []

    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        fname = m.group(2)

        # Misplaced (env file linked as develop) — fix to environment/ path.
        if fname in KNOWN_MISPLACED:
            target = KNOWN_MISPLACED[fname]
            # Re-anchor: prefix already goes up out of the source file's dir to
            # something near `docs/`. We can rebuild the absolute-from-root
            # target. For simplicity: if prefix is "../" assume source is in
            # docs/<sub>/, so "../environment/ENV..." works. If prefix is
            # "../../" assume source is in docs/<sub>/<sub2>/, so "../../environment/...".
            # The KNOWN_MISPLACED entry already starts with "docs/", so strip
            # that and prepend the same number of ../ as the prefix.
            stripped = target.removeprefix("docs/")
            new_link = f"{prefix}{stripped}"
            old_link = m.group(0)
            if new_link != old_link:
                changes.append(f"{old_link} -> {new_link}")
            return new_link

        new_rel = fmap.get(fname)
        if new_rel is None:
            return m.group(0)  # unknown filename — leave alone
        if file_in_develop:
            # Source file is itself under docs/develop/, so its links to
            # other develop docs use a different relative form. The pattern
            # we matched starts with "../" or "../../" — we leave the prefix
            # but the path AFTER develop/ becomes new_rel.
            new_link = f"{prefix}develop/{new_rel}"
        else:
            new_link = f"{prefix}develop/{new_rel}"
        if new_link != m.group(0):
            changes.append(f"{m.group(0)} -> {new_link}")
        return new_link

    new_text = DEVELOP_LINK_RE.sub(repl, text)
    return new_text, changes


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    for sub in ("project", "environment", "develop"):
        base = ROOT / "docs" / sub
        if base.exists():
            targets.extend(sorted(base.rglob("*.md")))
    if (ROOT / "CLAUDE.md").exists():
        targets.append(ROOT / "CLAUDE.md")
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    fmap = build_map()
    if not fmap:
        print("No develop docs found — was the migration run?", file=sys.stderr)
        return 1

    targets = collect_targets()
    total_changes = 0
    files_changed = 0

    for path in targets:
        text = path.read_text()
        is_dev = path.is_relative_to(DEVELOP)
        new_text, changes = rewrite_text(text, fmap, file_in_develop=is_dev)
        if not changes:
            continue
        files_changed += 1
        total_changes += len(changes)
        rel = path.relative_to(ROOT)
        print(f"\n{rel}: {len(changes)} substitution(s)")
        for c in changes[:5]:
            print(f"  {c}")
        if len(changes) > 5:
            print(f"  ... and {len(changes) - 5} more")

        if args.apply:
            path.write_text(new_text)

    print(f"\nTotal: {total_changes} substitutions across {files_changed} files")
    if not args.apply:
        print("(dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
