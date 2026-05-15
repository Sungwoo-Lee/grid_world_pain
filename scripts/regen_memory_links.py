#!/usr/bin/env python3
"""Walk docs/memory/memories/, extract [[id]] wikilinks from bodies, rewrite related: frontmatter."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "docs" / "memory"

# Matches the canonical insight ID embedded in [[...]] tokens.
# Coordinate arrays like [[1,1],[5,5]] do not match because the inner content
# includes commas and digits, not the YYYYMMDD_HHMM_slug pattern.
ID_PATTERN = re.compile(r"\[\[(\d{8}_\d{4}_[a-z0-9_]+)(?:\|[^\]]+)?\]\]")

# Matches the entire related: line (single-line only; multi-line YAML not used).
RELATED_LINE_RE = re.compile(r"^related:[ \t]*(.*)", re.MULTILINE)


def _canonical(ids: list[str]) -> str:
    """Format a deduplicated sorted list of IDs as canonical related: value."""
    unique = sorted(set(ids))
    if not unique:
        return "[]"
    inner = ", ".join(f'"{i}"' for i in unique)
    return f"[{inner}]"


def _parse_existing_related(raw: str) -> list[str]:
    """Parse any hand-typed related: value into a list of ID strings.

    Handles: [], [id], [id, id], ["id"], ["id", "id"], multi-word forms,
    and the compact bracket form without quotes.
    """
    raw = raw.strip()
    if raw in ("[]", ""):
        return []
    # Strip outer brackets
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        # Split on commas; strip whitespace and quotes from each token
        return [
            tok.strip().strip('"').strip("'")
            for tok in inner.split(",")
            if tok.strip().strip('"').strip("'")
        ]
    # Fallback: treat the whole thing as a bare ID
    return [raw.strip().strip('"').strip("'")]


def process_file(path: Path, dry_run: bool = False) -> bool:
    """Update related: in one insight file. Returns True if the file was changed."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return False

    # Locate frontmatter end
    fm_end = text.find("\n---\n", 4)
    if fm_end == -1:
        return False
    fm_block = text[4:fm_end]
    body = text[fm_end + 5:]

    # Find related: within frontmatter
    m = RELATED_LINE_RE.search(fm_block)
    if m is None:
        return False

    existing_raw = m.group(1).strip()
    existing_ids = _parse_existing_related(existing_raw)

    # Extract [[id]] tokens from the body only (not from frontmatter).
    # [[id|alias]] form: capture the id, log the alias presence.
    wikilink_ids = ID_PATTERN.findall(body)
    for alias_match in re.finditer(r"\[\[(\d{8}_\d{4}_[a-z0-9_]+)\|([^\]]+)\]\]", body):
        print(f"note: {path.name} has alias form [[{alias_match.group(1)}|{alias_match.group(2)}]] — treating as [[{alias_match.group(1)}]]")

    # Union: preserve existing related IDs and add any body [[id]] tokens.
    # --normalise-existing is the same algorithm but is documented as the
    # one-time migration pass; the flag controls nothing at runtime.
    all_ids = list(set(existing_ids) | set(wikilink_ids))

    new_value = _canonical(all_ids)

    # Skip writing when the file already has exactly the right canonical string.
    # We compare the new value against the current raw value directly — this
    # catches both content differences (new IDs to add) and format differences
    # (same IDs but written without quotes or in wrong order).
    if new_value == existing_raw:
        return False

    # Rewrite only the related: line
    new_fm_block = fm_block[: m.start()] + f"related: {new_value}" + fm_block[m.end():]
    new_text = "---\n" + new_fm_block + "\n---\n" + body

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
        print(f"updated: {path.relative_to(ROOT)}  ({existing_raw!r} → {new_value})")
    return True


def collect_insights(memory_root: Path) -> list[Path]:
    memories = memory_root / "memories"
    if not memories.is_dir():
        return []
    return sorted(
        p
        for p in memories.rglob("*.md")
        if p.name not in ("_topic_index.md", "_global_tags.md")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite related: frontmatter in memory insights from [[id]] body tokens."
    )
    parser.add_argument(
        "--normalise-existing",
        action="store_true",
        help="Merge hand-typed related: values with body [[id]] tokens into canonical form.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only; exit 1 if any file would be changed.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root of the memory layer (default: docs/memory).",
    )
    args = parser.parse_args()

    insights = collect_insights(args.root)
    if not insights:
        print(f"No insight files found under {args.root}", file=sys.stderr)
        return 1

    changed = 0
    for path in insights:
        if process_file(path, dry_run=args.check):
            changed += 1
            if args.check:
                print(f"would change: {path.relative_to(ROOT)}")

    if args.check:
        if changed:
            print(f"{changed} file(s) would be changed.", file=sys.stderr)
            return 1
        print("OK — no files would change.")
        return 0

    if not changed:
        print("No files changed.")
    else:
        print(f"{changed} file(s) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
