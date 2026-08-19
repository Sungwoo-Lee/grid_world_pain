#!/usr/bin/env python3
"""gen_schema_doc.py — regenerate the schema tables inside TRAJECTORY_STORE_SCHEMA.md.

The schema doc is half the deliverable of the trajectory-collection pipeline, and a doc
that disagrees with the code is worse than no doc, because it is trusted.  So the two
column tables are GENERATED from `STEP_COLUMNS` / `EPISODE_COLUMNS` in
`src/utils/trajectory_store.py` rather than hand-transcribed, written between marker
comments, and a test (`tests/test_trajectory_collection.py::test_schema_doc_matches_code`)
asserts the committed doc matches freshly-generated output.

Usage
-----
    python scripts/eval/traj_collect/gen_schema_doc.py            # rewrite in place
    python scripts/eval/traj_collect/gen_schema_doc.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # scripts/eval/traj_collect -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.trajectory_store import (  # noqa: E402
    EPISODE_COLUMNS, SCHEMA_VERSION, STEP_COLUMNS, render_arrow_type,
)

DOC_PATH = PROJECT_ROOT / "docs" / "environment" / "TRAJECTORY_STORE_SCHEMA.md"

BEGIN = "<!-- BEGIN GENERATED: {} — do not edit by hand; run scripts/eval/traj_collect/gen_schema_doc.py -->"
END = "<!-- END GENERATED: {} -->"


def _esc(s: str) -> str:
    return s.replace("|", "\\|")


def step_table() -> str:
    rows = ["| # | Column | Arrow type | Row-convention timing | Source |",
            "|---:|---|---|---|---|"]
    for i, c in enumerate(STEP_COLUMNS, 1):
        rows.append(f"| {i} | `{c.name}` | `{_esc(render_arrow_type(c))}` | "
                    f"{_esc(c.timing)} | {_esc(c.source)} |")
    return "\n".join(rows)


def episode_table() -> str:
    rows = ["| # | Column | Arrow type | Meaning |", "|---:|---|---|---|"]
    for i, c in enumerate(EPISODE_COLUMNS, 1):
        rows.append(f"| {i} | `{c.name}` | `{_esc(render_arrow_type(c))}` | {_esc(c.source)} |")
    return "\n".join(rows)


def version_line() -> str:
    return (f"`SCHEMA_VERSION = {SCHEMA_VERSION}` — "
            f"{len(STEP_COLUMNS)} step columns, {len(EPISODE_COLUMNS)} episode columns.")


BLOCKS = {
    "schema_version": version_line,
    "step_columns": step_table,
    "episode_columns": episode_table,
}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        begin, end = BEGIN.format(name), END.format(name)
        pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pat.search(text):
            raise ValueError(
                f"{DOC_PATH}: missing generated block {name!r}. Expected the marker "
                f"pair:\n{begin}\n{end}")
        text = pat.sub(lambda _m: f"{begin}\n\n{fn()}\n\n{end}", text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed doc differs from generated output")
    ap.add_argument("--path", type=Path, default=DOC_PATH)
    a = ap.parse_args(argv)

    have = a.path.read_text()
    want = render(have)
    if have == want:
        print(f"{a.path}: up to date")
        return 0
    if a.check:
        print(f"{a.path}: STALE — regenerate with "
              f"`python scripts/eval/traj_collect/gen_schema_doc.py`", file=sys.stderr)
        return 1
    a.path.write_text(want)
    print(f"{a.path}: regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
