#!/usr/bin/env python3
"""Record that a wiki entry was actually read: bump use_count, set last_used.

Reinforcement WITHOUT decay. The 2026 agent-memory literature pairs use-reinforcement
with Ebbinghaus-style time decay; decay is deliberately NOT implemented here, because
in a research log a refuted hypothesis stays refuted — age is not evidence of
irrelevance.

COVERAGE LIMIT (important): this counts skill-mediated reads only — /wiki-read at
L3. An ad-hoc grep, a direct Read of the file, or a human opening it in an editor is
invisible. use_count is therefore a LOWER BOUND on an entry's usefulness and must
never be used to delete or demote entries. It is for breaking ties in recall order
and for spotting folders nothing ever reads.

Usage:  python scripts/claude/wiki_touch.py <entry_id> [<entry_id> ...]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "docs" / "llm_wiki"


def touch(path: Path, today: str) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: no frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path.name}: unterminated frontmatter")
    fm, body = text[4:end], text[end + 5:]

    m = re.search(r"^use_count:[ \t]*(\d+)[ \t]*$", fm, re.M)
    old = int(m.group(1)) if m else 0
    new = old + 1

    if m:
        fm = fm[: m.start()] + f"use_count: {new}" + fm[m.end():]
    else:
        # Insert after raw_completeness (last standard field) to keep field order stable.
        anchor = re.search(r"^raw_completeness:.*$", fm, re.M)
        ins = f"\nuse_count: {new}\nlast_used: {today}"
        if anchor:
            fm = fm[: anchor.end()] + ins + fm[anchor.end():]
        else:
            fm = fm.rstrip("\n") + ins
        path.write_text("---\n" + fm + "\n---\n" + body, encoding="utf-8")
        return old, new

    lm = re.search(r"^last_used:[ \t]*.*$", fm, re.M)
    if lm:
        fm = fm[: lm.start()] + f"last_used: {today}" + fm[lm.end():]
    else:
        um = re.search(r"^use_count:.*$", fm, re.M)
        fm = fm[: um.end()] + f"\nlast_used: {today}" + fm[um.end():]

    path.write_text("---\n" + fm + "\n---\n" + body, encoding="utf-8")
    return old, new


def main() -> int:
    ap = argparse.ArgumentParser(description="Bump use_count / last_used on wiki entries that were read.")
    ap.add_argument("ids", nargs="+", help="Entry IDs (filename stems), e.g. 20260728_1642_slug")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    today = _dt.date.today().isoformat()
    entries = args.root / "entries"
    rc = 0
    for eid in args.ids:
        hits = list(entries.rglob(f"{eid}.md"))
        if not hits:
            print(f"not found: {eid}", file=sys.stderr)
            rc = 1
            continue
        old, new = touch(hits[0], today)
        print(f"touched: {eid}  use_count {old} -> {new}, last_used {today}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
