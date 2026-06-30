#!/usr/bin/env python3
"""Resolve an insight ID or session UUID to its source JSONL and print restoration commands.

Usage:
    python scripts/open_conversation.py <insight_id>
    python scripts/open_conversation.py <session_uuid>

Exit codes:
    0 — success
    1 — insight / UUID not found
    2 — insight found but raw_source is a pre-sync archive (no synced JSONL)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORIES_ROOT = ROOT / "docs" / "memory" / "memories"

# Insight ID pattern: YYYYMMDD_HHMM_<slug>
INSIGHT_ID_RE = re.compile(r"^\d{8}_\d{4}_[a-z0-9_]+$")

# Session UUID pattern: 8-4-4-4-12 hex
UUID_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)

# UUID anywhere in a string (for raw_source extraction)
UUID_ANYWHERE_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    if not text.startswith("---\n"):
        return None
    fm_end = text.find("\n---\n", 4)
    if fm_end == -1:
        return None
    fm_block = text[4:fm_end]
    result: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"')
    return result


def _collect_all_insights() -> list[Path]:
    """Return all insight .md paths (excluding index files and archive)."""
    if not MEMORIES_ROOT.is_dir():
        return []
    paths = []
    for p in sorted(MEMORIES_ROOT.rglob("*.md")):
        if p.name in ("_topic_index.md", "_global_tags.md"):
            continue
        parts = set(p.parts)
        if "_archive" in parts or ".trash" in parts:
            continue
        paths.append(p)
    return paths


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def _jsonl_path_for_uuid(uuid: str) -> Path:
    return (
        ROOT
        / "claude_data"
        / ".claude"
        / "projects"
        / "-media-nas01-projects-Interoceptive-AI-grid-world-pain"
        / f"{uuid.lower()}.jsonl"
    )


def _handle_insight(insight_path: Path, insight_id: str, all_paths: list[Path]) -> int:
    text = insight_path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    if fm is None:
        print(f"Error: could not parse frontmatter in {insight_path}", file=sys.stderr)
        return 1

    raw_source = fm.get("raw_source", "").strip()
    folder = fm.get("folder", insight_path.parent.name)

    print(f"Insight  : docs/memory/memories/{folder}/{insight_id}.md")

    # Detect pre-sync / archive source
    if not raw_source or raw_source == "none":
        print("This insight has no synced raw source (pre-sync genesis insight).")
        sys.exit(2)

    uuid_m = UUID_ANYWHERE_RE.search(raw_source)
    if not uuid_m:
        # _archive/... path — not a UUID-based JSONL
        print("This insight has no synced raw source (pre-sync genesis insight).")
        sys.exit(2)

    uuid = uuid_m.group(1).lower()
    return _handle_uuid(uuid, highlight_id=insight_id, all_paths=all_paths)


def _handle_uuid(uuid: str, highlight_id: str | None, all_paths: list[Path]) -> int:
    """Print provenance info for a session UUID. highlight_id marks '(this one)' in sibling list."""
    jsonl_path = _jsonl_path_for_uuid(uuid)

    print(f"Session  : {uuid}")

    if jsonl_path.exists():
        size_str = _human_size(jsonl_path.stat().st_size)
        jsonl_display = str(jsonl_path.relative_to(ROOT))
        print(f"JSONL    : {jsonl_display} ({size_str})")
    else:
        jsonl_display = str(jsonl_path.relative_to(ROOT))
        print(f"JSONL    : {jsonl_display} (not present locally)")

    # Collect sibling insights from same session UUID
    siblings: list[tuple[str, str]] = []  # (insight_id, topic_folder)
    for p in all_paths:
        t = p.read_text(encoding="utf-8")
        fm = _parse_frontmatter(t)
        if fm is None:
            continue
        rs = fm.get("raw_source", "")
        m = UUID_ANYWHERE_RE.search(rs)
        if m and m.group(1).lower() == uuid:
            iid = fm.get("id", p.stem)
            fld = fm.get("folder", p.parent.name)
            siblings.append((iid, fld))

    siblings.sort(key=lambda x: x[0])
    print(f"Sibling insights from this session ({len(siblings)}):")
    for sibling_id, sibling_folder in siblings:
        marker = " (this one)" if sibling_id == highlight_id else ""
        print(f"  - [[{sibling_id}]]{marker}")

    print("")
    print("To restore the conversation:")
    uuid_prefix = uuid[:8]
    print(f"  Option A — re-enter in Claude Code:    claude --resume {uuid}")
    if jsonl_path.exists():
        print(
            f"  Option B — one-shot markdown view:     "
            f"python scripts/claude_jsonl_to_md.py {jsonl_display} /tmp/{uuid_prefix}.md"
        )
    else:
        print(
            f"  Option B — one-shot markdown view:     "
            f"python scripts/claude_jsonl_to_md.py {jsonl_display} /tmp/{uuid_prefix}.md"
            f"  (JSONL not synced locally)"
        )
    print("")
    if not jsonl_path.exists():
        print("If the JSONL is missing locally: ./sync-agent-data.sh claude pull")

    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/open_conversation.py <insight_id_or_session_uuid>",
            file=sys.stderr,
        )
        return 1

    arg = sys.argv[1].strip()
    all_paths = _collect_all_insights()

    # Determine whether arg is an insight ID or a session UUID
    if UUID_RE.match(arg):
        # UUID mode — verify at least one insight points at this session
        uuid = arg.lower()
        found_any = False
        for p in all_paths:
            t = p.read_text(encoding="utf-8")
            fm = _parse_frontmatter(t)
            if fm is None:
                continue
            rs = fm.get("raw_source", "")
            m = UUID_ANYWHERE_RE.search(rs)
            if m and m.group(1).lower() == uuid:
                found_any = True
                break
        if not found_any:
            print(f"No insights found for session UUID: {uuid}", file=sys.stderr)
            return 1
        return _handle_uuid(uuid, highlight_id=None, all_paths=all_paths)

    elif INSIGHT_ID_RE.match(arg):
        # Insight ID mode
        insight_id = arg
        # Search for the file
        for p in all_paths:
            t = p.read_text(encoding="utf-8")
            fm = _parse_frontmatter(t)
            if fm is None:
                continue
            if fm.get("id", "").strip() == insight_id:
                return _handle_insight(p, insight_id, all_paths)
        print(f"Insight not found: {insight_id}", file=sys.stderr)
        return 1

    else:
        print(
            f"Argument '{arg}' does not match an insight ID (YYYYMMDD_HHMM_slug) "
            "or a session UUID (8-4-4-4-12 hex).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
