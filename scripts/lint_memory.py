#!/usr/bin/env python3
"""Read-only lint script for docs/memory/: 9 checks, punch-list output, exit 0/1."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ─── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "docs" / "memory"
TODAY = date(2026, 5, 16)  # canonical today per task spec
ORPHAN_DAYS = 30
FOLDER_OVERLAP_THRESHOLD = 0.7

# Matches canonical insight IDs inside [[...]] tokens.
# Coordinate arrays like [[1,1],[5,5]] do not match (no slug pattern).
ID_PATTERN = re.compile(r"\[\[(\d{8}_\d{4}_[a-z0-9_]+)(?:\|[^\]]+)?\]\]")

# Strips auto-generated BACKLINKS blocks before scanning body for [[id]] tokens.
BACKLINKS_BLOCK_RE = re.compile(r"<!-- BACKLINKS.*?<!-- END BACKLINKS -->\s*", re.DOTALL)

# Matches the related: line in frontmatter (single-line only).
RELATED_LINE_RE = re.compile(r"^related:[ \t]*(.*)", re.MULTILINE)

# Legacy _archive/raw_conversations/... paths are pre-sync placeholders — skip resolution.
ARCHIVE_RAW_RE = re.compile(r"_archive/raw_conversations/.*\.md$")

# UUID in raw_source (8-4-4-4-12 hex).
UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.IGNORECASE
)

# ─── Helpers ────────────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """Return frontmatter key→raw-value dict or None if no valid frontmatter."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    fm_block = text[4:end]
    result: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _parse_related(raw: str) -> list[str]:
    """Parse related: [...] into a list of ID strings."""
    raw = raw.strip()
    if raw in ("[]", ""):
        return []
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [
            tok.strip().strip('"').strip("'")
            for tok in inner.split(",")
            if tok.strip().strip('"').strip("'")
        ]
    return [raw.strip().strip('"').strip("'")]


def _parse_tags(raw: str) -> list[str]:
    """Parse tags: [a, b, c] into a list of strings."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        return [t.strip().strip('"').strip("'") for t in inner.split(",") if t.strip()]
    if raw:
        return [raw.strip().strip('"').strip("'")]
    return []


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two definition strings.

    Word-level (not character-level) is used because single-character Jaccard
    produces false positives for short English strings that share common letters
    but are semantically unrelated (e.g., 'DreamerV3 failure investigation' vs
    'Lab cluster ops and env mgmt' share letters e/a/i/n/o but share 0 words).
    Word-level Jaccard correctly identifies genuinely overlapping definitions.
    """
    if not a and not b:
        return 1.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def collect_insights(memory_root: Path) -> list[Path]:
    """Return all insight .md files excluding index/archive/trash."""
    memories = memory_root / "memories"
    if not memories.is_dir():
        return []
    paths = []
    for p in sorted(memories.rglob("*.md")):
        if p.name in ("_topic_index.md", "_global_tags.md"):
            continue
        parts = set(p.parts)
        if "_archive" in parts or ".trash" in parts:
            continue
        paths.append(p)
    return paths


# ─── Data model for a parsed insight ────────────────────────────────────────


class Insight:
    __slots__ = (
        "path", "id", "folder_fm", "folder_actual", "date_str",
        "tags", "related_ids", "raw_source", "outbound_wikilinks",
    )

    def __init__(self, path: Path, fm: dict[str, str], body_clean: str) -> None:
        self.path = path
        self.id = fm.get("id", "").strip()
        self.folder_fm = fm.get("folder", "").strip()
        self.folder_actual = path.parent.name
        self.date_str = fm.get("date", "").strip()
        self.tags = _parse_tags(fm.get("tags", ""))
        rel_raw = fm.get("related", "[]")
        self.related_ids = _parse_related(rel_raw)
        self.raw_source = fm.get("raw_source", "").strip()
        self.outbound_wikilinks: list[str] = list(dict.fromkeys(ID_PATTERN.findall(body_clean)))


def load_all_insights(paths: list[Path]) -> list[Insight]:
    """Parse all insight files into Insight objects."""
    result = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        if fm is None:
            continue
        fm_end = text.find("\n---\n", 4)
        body = text[fm_end + 5:] if fm_end != -1 else ""
        body_clean = BACKLINKS_BLOCK_RE.sub("", body)
        result.append(Insight(path, fm, body_clean))
    return result


# ─── Check implementations ──────────────────────────────────────────────────


def check_wikilink_resolution(
    insights: list[Insight],
    id_set: set[str],
) -> list[str]:
    """[1/9] Every [[id]] token in body resolves to a real insight."""
    issues = []
    for ins in insights:
        for target in ins.outbound_wikilinks:
            if target not in id_set:
                rel = ins.path.relative_to(ROOT)
                issues.append(f"  {rel}: broken [[{target}]]")
    return issues


def check_related_resolution(
    insights: list[Insight],
    id_set: set[str],
) -> list[str]:
    """[2/9] Every related: ID resolves to a real insight."""
    issues = []
    for ins in insights:
        for rid in ins.related_ids:
            if rid and rid not in id_set:
                rel = ins.path.relative_to(ROOT)
                issues.append(f"  {rel}: related: contains unknown id '{rid}'")
    return issues


def check_folder_frontmatter(insights: list[Insight]) -> list[str]:
    """[3/9] folder: frontmatter matches the parent directory name."""
    issues = []
    for ins in insights:
        if ins.folder_fm != ins.folder_actual:
            rel = ins.path.relative_to(ROOT)
            issues.append(
                f"  {rel}: folder: '{ins.folder_fm}' but parent dir is '{ins.folder_actual}'"
            )
    return issues


def check_unique_ids(insights: list[Insight]) -> list[str]:
    """[4/9] No two insights share the same id: value."""
    seen: dict[str, list[Path]] = {}
    for ins in insights:
        if ins.id:
            seen.setdefault(ins.id, []).append(ins.path)
    issues = []
    for iid, paths in seen.items():
        if len(paths) > 1:
            rels = ", ".join(str(p.relative_to(ROOT)) for p in paths)
            issues.append(f"  Duplicate id '{iid}': {rels}")
    return issues


def check_tags_in_dictionary(
    insights: list[Insight],
    global_tags: set[str],
) -> list[str]:
    """[5/9] Every tag used in any insight is present in _global_tags.md."""
    issues = []
    for ins in insights:
        for tag in ins.tags:
            if tag and tag not in global_tags:
                rel = ins.path.relative_to(ROOT)
                issues.append(f"  {rel}: unknown tag '{tag}'")
    return issues


def check_folder_definition_overlap(memory_root: Path) -> list[str]:
    """[6/9] No two folder definitions in ROOT_INDEX.md have ≥ 0.7 character Jaccard overlap."""
    root_index = memory_root / "ROOT_INDEX.md"
    if not root_index.exists():
        return ["  ROOT_INDEX.md not found — cannot run folder-definition overlap check."]

    text = root_index.read_text(encoding="utf-8")
    # Extract the Active folders table: rows of the form | `folder` | definition | ... |
    TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+)\|", re.MULTILINE)
    folders: list[tuple[str, str]] = []
    for m in TABLE_ROW_RE.finditer(text):
        folder_name = m.group(1).strip()
        definition = m.group(2).strip()
        if folder_name and definition:
            folders.append((folder_name, definition))

    if len(folders) < 2:
        return []

    issues = []
    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):
            name_a, def_a = folders[i]
            name_b, def_b = folders[j]
            sim = _jaccard_similarity(def_a.lower(), def_b.lower())
            if sim >= FOLDER_OVERLAP_THRESHOLD:
                issues.append(
                    f"  '{name_a}' vs '{name_b}': definition overlap {sim:.2f} ≥ {FOLDER_OVERLAP_THRESHOLD}\n"
                    f"    '{def_a}'\n"
                    f"    '{def_b}'"
                )
    return issues


def check_old_orphans(insights: list[Insight]) -> list[str]:
    """[7/9] Orphan insights (0 inbound AND 0 outbound links) older than 30 days."""
    # Build outbound set and inbound counts
    id_set = {ins.id for ins in insights if ins.id}
    inbound: dict[str, int] = {ins.id: 0 for ins in insights if ins.id}

    for ins in insights:
        for target in ins.outbound_wikilinks:
            if target in inbound:
                inbound[target] += 1

    issues = []
    for ins in insights:
        has_outbound = bool(ins.outbound_wikilinks)
        has_inbound = inbound.get(ins.id, 0) > 0
        if has_outbound or has_inbound:
            continue
        # Truly an orphan — check age
        if not ins.date_str:
            continue
        try:
            insight_date = datetime.strptime(ins.date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        age_days = (TODAY - insight_date).days
        if age_days > ORPHAN_DAYS:
            rel = ins.path.relative_to(ROOT)
            issues.append(f"  {rel} (date {ins.date_str}, {age_days} days old)")
    return issues


def check_raw_source_resolution(
    insights: list[Insight],
    memory_root: Path,
) -> tuple[list[str], bool]:
    """[8/9] Every non-none raw_source resolves on disk.

    Returns (issues, skipped_due_to_missing_claude_data).
    """
    claude_data = ROOT / "claude_data"
    if not claude_data.is_dir():
        return (
            ["  claude_data/ not present locally — run ./sync-agent-data.sh claude pull "
             "to populate; skipping per-insight raw_source resolution checks."],
            True,
        )

    issues = []
    for ins in insights:
        rs = ins.raw_source
        if not rs or rs.lower() == "none":
            continue
        # Legacy _archive/raw_conversations/... paths are pre-sync placeholders; skip.
        if ARCHIVE_RAW_RE.search(rs):
            continue
        # Resolve relative to repo root
        target = ROOT / rs
        if not target.exists():
            rel = ins.path.relative_to(ROOT)
            issues.append(f"  {rel}: raw_source '{rs}' not found on disk")
    return issues, False


def check_raw_source_reverse(
    insights: list[Insight],
    memory_root: Path,
) -> list[str]:
    """[9/9] Every JSONL in claude_data/.claude/projects/.../*.jsonl referenced by an insight
    appears in the Conversation provenance section of GRAPH_REPORT.md.
    """
    claude_data = ROOT / "claude_data"
    if not claude_data.is_dir():
        return ["  claude_data/ not present locally — skipping reverse check."]

    graph_report = memory_root / "GRAPH_REPORT.md"
    if not graph_report.exists():
        return ["  GRAPH_REPORT.md not found — run scripts/regen_memory_graph.py to generate it."]

    report_text = graph_report.read_text(encoding="utf-8")

    # Collect UUIDs referenced by insights
    referenced_uuids: set[str] = set()
    for ins in insights:
        rs = ins.raw_source
        if not rs or rs.lower() == "none":
            continue
        m = UUID_PATTERN.search(rs)
        if m:
            referenced_uuids.add(m.group(1).lower())

    issues = []
    for uuid in sorted(referenced_uuids):
        if uuid not in report_text.lower():
            issues.append(
                f"  Session {uuid} is referenced in raw_source but not found in "
                f"GRAPH_REPORT.md Conversation provenance section."
            )
    return issues


# ─── Global tags loading ─────────────────────────────────────────────────────


def load_global_tags(memory_root: Path) -> set[str]:
    """Extract active tag names from _global_tags.md."""
    tags_file = memory_root / "memories" / "_global_tags.md"
    if not tags_file.exists():
        return set()
    text = tags_file.read_text(encoding="utf-8")
    # Active tags table rows: | `tag` | ... |
    TAG_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)
    tags = set()
    in_active_section = False
    for line in text.splitlines():
        if "## Active tags" in line:
            in_active_section = True
            continue
        if in_active_section and line.startswith("##"):
            break
        if in_active_section:
            m = TAG_ROW_RE.match(line)
            if m:
                tag = m.group(1).strip()
                if tag and tag != "Tag":  # skip header row
                    tags.add(tag)
    return tags


# ─── Output helpers ──────────────────────────────────────────────────────────


def _format_section(
    num: int,
    total: int,
    title: str,
    issues: list[str],
    ok_msg: str,
    quiet: bool,
) -> list[str]:
    lines = []
    if not quiet:
        lines.append(f"\n[{num}/{total}] {title}")
    if issues:
        lines.extend(issues)
    elif not quiet:
        lines.append(f"  {ok_msg}")
    return lines


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only lint for docs/memory/ — emits a punch list of issues."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root of the memory layer (default: docs/memory).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_out",
        help="Emit JSON list of issues for machine consumption.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print issues, no per-check headers.",
    )
    args = parser.parse_args()

    memory_root: Path = args.root

    # ── Load data ────────────────────────────────────────────────────────────
    insight_paths = collect_insights(memory_root)
    if not insight_paths:
        print(f"No insight files found under {memory_root}", file=sys.stderr)
        return 1

    insights = load_all_insights(insight_paths)
    id_set: set[str] = {ins.id for ins in insights if ins.id}
    global_tags = load_global_tags(memory_root)

    # ── Run checks ───────────────────────────────────────────────────────────
    TOTAL = 9
    now_str = datetime.now().strftime("%H:%M")
    header = (
        f"{'='*40}\n"
        f"docs/memory/ lint report — {TODAY} {now_str}\n"
        f"{'='*40}"
    )

    issues_wikilink = check_wikilink_resolution(insights, id_set)
    issues_related = check_related_resolution(insights, id_set)
    issues_folder = check_folder_frontmatter(insights)
    issues_dup_id = check_unique_ids(insights)
    issues_tags = check_tags_in_dictionary(insights, global_tags)
    issues_overlap = check_folder_definition_overlap(memory_root)
    issues_orphans = check_old_orphans(insights)
    issues_raw_src, skipped_raw = check_raw_source_resolution(insights, memory_root)
    issues_reverse = check_raw_source_reverse(insights, memory_root)

    n_insights = len(insights)

    # ── JSON output mode ─────────────────────────────────────────────────────
    if args.json_out:
        records: list[dict] = []

        def _add(check: str, items: list[str], severity: str = "error") -> None:
            for msg in items:
                records.append({"check": check, "severity": severity, "message": msg.strip()})

        _add("wikilink_resolution", issues_wikilink)
        _add("related_resolution", issues_related)
        _add("folder_frontmatter", issues_folder)
        _add("duplicate_ids", issues_dup_id)
        _add("tag_dictionary", issues_tags)
        _add("folder_definition_overlap", issues_overlap)
        _add("old_orphans", issues_orphans, "warning")
        _add("raw_source_resolution", issues_raw_src, "warning" if skipped_raw else "error")
        _add("raw_source_reverse", issues_reverse, "warning")

        print(json.dumps(records, indent=2))
        # Exit code: 1 if any genuine errors (not just warnings about missing claude_data)
        genuine_errors = (
            issues_wikilink
            or issues_related
            or issues_folder
            or issues_dup_id
            or issues_tags
            or issues_overlap
            or (issues_raw_src and not skipped_raw)
        )
        return 1 if genuine_errors else 0

    # ── Human-readable output ─────────────────────────────────────────────────
    out: list[str] = []
    if not args.quiet:
        out.append(header)

    out.extend(_format_section(
        1, TOTAL, "Wikilink [[id]] resolution", issues_wikilink,
        "All [[id]] tokens resolve.", args.quiet,
    ))
    out.extend(_format_section(
        2, TOTAL, "related: resolution", issues_related,
        "All related: IDs resolve.", args.quiet,
    ))
    out.extend(_format_section(
        3, TOTAL, f"folder: frontmatter matches parent", issues_folder,
        f"All {n_insights} insights have correct folder: frontmatter.", args.quiet,
    ))
    out.extend(_format_section(
        4, TOTAL, "No duplicate id:", issues_dup_id,
        f"All {n_insights} IDs are unique.", args.quiet,
    ))
    out.extend(_format_section(
        5, TOTAL, "Tag dictionary completeness", issues_tags,
        "All tags used in insights are present in _global_tags.md.", args.quiet,
    ))
    out.extend(_format_section(
        6, TOTAL, "Near-duplicate folder definitions", issues_overlap,
        "No folder-definition overlaps above the 0.7 threshold.", args.quiet,
    ))

    # Section [7] always prints the list (warnings, not errors)
    if not args.quiet:
        out.append(f"\n[7/{TOTAL}] Old orphan insights (> {ORPHAN_DAYS} days, no links)")
    if issues_orphans:
        icon = "  ⚠️ " if not args.quiet else ""
        if not args.quiet:
            out.append(f"  ⚠️  {len(issues_orphans)} orphan insight(s) older than {ORPHAN_DAYS} days:")
        out.extend(issues_orphans)
    elif not args.quiet:
        out.append("  ✅ No old orphan insights.")

    # Section [8]
    if not args.quiet:
        out.append(f"\n[8/{TOTAL}] raw_source resolution")
    if skipped_raw:
        out.extend(f"  ⚠️  {ln}" if not args.quiet else ln for ln in issues_raw_src)
    elif issues_raw_src:
        out.extend(issues_raw_src)
    elif not args.quiet:
        out.append("  ✅ All raw_source pointers resolve.")

    # Section [9]
    if not args.quiet:
        out.append(f"\n[9/{TOTAL}] raw_source reverse check")
    if issues_reverse and issues_reverse[0].strip().startswith("claude_data/ not present"):
        out.extend(f"  ⚠️  {ln}" if not args.quiet else ln for ln in issues_reverse)
    elif issues_reverse:
        out.extend(issues_reverse)
    elif not args.quiet:
        out.append("  ✅ All referenced sessions appear in GRAPH_REPORT.md.")

    # ── Summary ───────────────────────────────────────────────────────────────
    genuine_errors = (
        issues_wikilink
        or issues_related
        or issues_folder
        or issues_dup_id
        or issues_tags
        or issues_overlap
        or (issues_raw_src and not skipped_raw)
    )
    warnings_only = issues_orphans or skipped_raw or (not issues_reverse or issues_reverse[0].strip().startswith("claude_data/"))

    total_issues = (
        len(issues_wikilink)
        + len(issues_related)
        + len(issues_folder)
        + len(issues_dup_id)
        + len(issues_tags)
        + len(issues_overlap)
        + (len(issues_raw_src) if not skipped_raw else 0)
    )
    total_warnings = (
        len(issues_orphans)
        + (1 if skipped_raw else 0)
        + (1 if issues_reverse and issues_reverse[0].strip().startswith("claude_data/") else 0)
        + (len(issues_reverse) if issues_reverse and not issues_reverse[0].strip().startswith("claude_data/") else 0)
    )

    if not args.quiet:
        out.append(f"\n{'='*40}")
        if genuine_errors:
            checks_with_errors = sum([
                bool(issues_wikilink), bool(issues_related), bool(issues_folder),
                bool(issues_dup_id), bool(issues_tags), bool(issues_overlap),
                bool(issues_raw_src and not skipped_raw),
            ])
            out.append(
                f"Summary: {total_issues} error(s) across {checks_with_errors} check(s)"
                + (f", {total_warnings} warning(s)." if total_warnings else ".")
            )
        elif warnings_only:
            out.append(f"Summary: 0 errors. {total_warnings} warning(s) (see above).")
        else:
            out.append("Summary: ✅ All checks passed — memory layer is clean.")
        out.append("=" * 40)

    print("\n".join(out))
    return 1 if genuine_errors else 0


if __name__ == "__main__":
    sys.exit(main())
