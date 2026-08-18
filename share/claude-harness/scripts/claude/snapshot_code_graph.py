#!/usr/bin/env python3
"""Capture a point-in-time code-graph snapshot into docs/llm_wiki/code_snapshots/.

Run:
    python scripts/snapshot_code_graph.py <label> [--scope src|all] [--no-regen] [--no-commit]

Behavior:
1. Unless --no-regen, run regen_code_graph.py to refresh the live graph.
2. Read <repo_root>/<scope>/graphify-out/GRAPH_REPORT.md.
3. Write to docs/llm_wiki/code_snapshots/YYYYMMDD_HHMM_<label>.md with a
   frontmatter header.
4. Update docs/llm_wiki/code_snapshots/README.md index.
5. Unless --no-commit, stage + commit the snapshot and updated index.

Idempotency: if the target filename already exists, append _2, _3, etc.
Exit codes: 0 = success, 1 = validation / regen / missing-file error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = REPO_ROOT / "docs" / "llm_wiki" / "code_snapshots"
CONDA_PIP = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/pip"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _validate_label(label: str) -> None:
    """Raise SystemExit if label does not match [a-z0-9_]+."""
    if not re.fullmatch(r"[a-z0-9_]+", label):
        print(
            f"ERROR: label '{label}' is invalid. "
            "Use only lowercase letters, digits, and underscores (kebab mapped: use _ not -).",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_graphify_version() -> str:
    """Return the installed graphifyy version string, or 'unknown'."""
    try:
        result = subprocess.run(
            [CONDA_PIP, "show", "graphifyy"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _get_short_sha() -> str:
    """Return `git rev-parse --short HEAD` or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _parse_summary_stats(content: str) -> tuple[str, str, str]:
    """Parse '## Summary' section for node/edge/community counts.

    Matches lines like: '735 nodes · 1009 edges · 59 communities ...'
    Returns (node_count, edge_count, community_count) as strings, or
    ('?', '?', '?') if the section / pattern is not found.
    """
    # Find the ## Summary section
    summary_match = re.search(r"^## Summary\s*\n(.*)", content, re.MULTILINE)
    if not summary_match:
        return "?", "?", "?"
    summary_line = summary_match.group(1).strip()
    m = re.search(
        r"(\d+)\s+nodes\s*[·\-]\s*(\d+)\s+edges\s*[·\-]\s*(\d+)\s+communities",
        summary_line,
    )
    if not m:
        return "?", "?", "?"
    return m.group(1), m.group(2), m.group(3)


def _unique_filename(label: str, timestamp: str) -> str:
    """Return a unique filename stem, appending _2/_3/... if needed."""
    base = f"{timestamp}_{label}"
    stem = base
    counter = 2
    while (SNAPSHOTS_DIR / f"{stem}.md").exists():
        stem = f"{base}_{counter}"
        counter += 1
    return f"{stem}.md"


def _build_header(
    label: str,
    captured: str,
    short_sha: str,
    scope: str,
    graphify_version: str,
    session_id: str,
    snapshot_filename: str,
) -> str:
    """Build the frontmatter + intro block prepended to the snapshot."""
    return f"""---
snapshot_label: {label}
captured: {captured}
source_commit: {short_sha}
scope: {scope}
graphify_version: {graphify_version}
session_id: {session_id}
---

# Code-graph snapshot — `{label}`

Captured `{captured}` from commit `{short_sha}`.

To compare with the current code state: `python scripts/regen_code_graph.py && diff {scope}/graphify-out/GRAPH_REPORT.md docs/llm_wiki/code_snapshots/{snapshot_filename}`.

To re-enter the originating conversation: `python scripts/open_conversation.py {session_id}` (if the session was a `/wiki-write` invocation).

---

"""


def _update_readme(
    captured: str,
    label: str,
    short_sha: str,
    node_count: str,
    edge_count: str,
    community_count: str,
    snapshot_filename: str,
) -> None:
    """Prepend a row to the snapshots table in docs/llm_wiki/code_snapshots/README.md."""
    readme = SNAPSHOTS_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")

    # The table header we're looking for:
    header_marker = "| Captured | Label | Source commit | Stats | Link |"
    separator = "|---|---|---|---|---|"
    new_row = (
        f"| {captured} | {label} | {short_sha} | "
        f"{node_count} nodes / {edge_count} edges / {community_count} communities | "
        f"[snapshot]({snapshot_filename}) |"
    )

    # Insert the new row right after the separator line of the table
    sep_pos = text.find(separator)
    if sep_pos == -1:
        # Fallback: append to end of file
        text = text.rstrip() + "\n" + new_row + "\n"
    else:
        insert_pos = sep_pos + len(separator)
        # Check if there's already a newline
        after = text[insert_pos:]
        text = text[:insert_pos] + "\n" + new_row + after

    readme.write_text(text, encoding="utf-8")
    print(f"Updated: docs/llm_wiki/code_snapshots/README.md")


def _run_regen(scope: str) -> None:
    """Run regen_code_graph.py --scope <scope>. Exit 1 on failure."""
    regen_script = REPO_ROOT / "scripts" / "regen_code_graph.py"
    conda_python = Path(sys.executable)
    cmd = [str(conda_python), str(regen_script), "--scope", scope]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(
            f"ERROR: regen_code_graph.py exited with code {result.returncode}. "
            "Fix the issue or use --no-regen to skip regeneration.",
            file=sys.stderr,
        )
        sys.exit(1)


def _git_commit(snapshot_path: Path, scope: str, label: str, short_sha: str,
                node_count: str, edge_count: str, community_count: str) -> None:
    """Stage snapshot + README and commit."""
    readme_path = SNAPSHOTS_DIR / "README.md"
    files = [str(snapshot_path.relative_to(REPO_ROOT)), str(readme_path.relative_to(REPO_ROOT))]

    # Stage by name
    stage_cmd = ["git", "add"] + files
    subprocess.run(stage_cmd, cwd=str(REPO_ROOT), check=True)

    commit_msg = (
        f"docs(wiki): \U0001f4ca code-graph snapshot — {label}\n\n"
        f"Source commit: {short_sha}\n"
        f"Scope: {scope}\n"
        f"Stats: {node_count} nodes / {edge_count} edges / {community_count} communities\n\n"
        "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
    )
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(
            f"WARNING: git commit exited with code {result.returncode}. "
            "Files are staged; commit manually if needed.",
            file=sys.stderr,
        )
    else:
        # Print the short hash of the new commit
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        new_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "?"
        print(f"Committed as {new_sha}.")


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture a code-graph snapshot into docs/llm_wiki/code_snapshots/.",
    )
    p.add_argument(
        "label",
        help="Short snake_case slug describing why this snapshot exists "
             "(e.g. v1_4_ship, memory_v2_complete, pre_jax_dreamer_pivot). "
             "Allowed chars: [a-z0-9_].",
    )
    p.add_argument(
        "--scope",
        choices=["src", "all"],
        default="src",
        help="Scope of the graphify scan passed to regen_code_graph.py (default: src).",
    )
    p.add_argument(
        "--no-regen",
        action="store_true",
        help="Skip live graphify regeneration; copy whatever is currently in "
             "<scope>/graphify-out/GRAPH_REPORT.md.",
    )
    p.add_argument(
        "--no-commit",
        action="store_true",
        help="Skip the auto-commit. Useful for testing.",
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # 0. Validate label
    _validate_label(args.label)

    # 1. Optionally regenerate the live graph
    if not args.no_regen:
        _run_regen(args.scope)

    # 2. Locate the source GRAPH_REPORT.md
    if args.scope == "src":
        graph_report_path = REPO_ROOT / "src" / "graphify-out" / "GRAPH_REPORT.md"
    else:
        graph_report_path = REPO_ROOT / "graphify-out" / "GRAPH_REPORT.md"

    if not graph_report_path.exists():
        print(
            f"ERROR: {graph_report_path} not found. "
            "Run `python scripts/regen_code_graph.py --scope {args.scope}` first, "
            "or use --no-regen to reuse an existing file.",
            file=sys.stderr,
        )
        sys.exit(1)

    report_content = graph_report_path.read_text(encoding="utf-8")

    # 3. Compute filename and destination
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    captured = now.strftime("%Y-%m-%d %H:%M")
    short_sha = _get_short_sha()
    graphify_version = _get_graphify_version()
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_filename = _unique_filename(args.label, timestamp)
    snapshot_path = SNAPSHOTS_DIR / snapshot_filename

    # 4. Parse stats from the source report
    node_count, edge_count, community_count = _parse_summary_stats(report_content)

    # 5. Build and write the snapshot file
    header = _build_header(
        label=args.label,
        captured=captured,
        short_sha=short_sha,
        scope=args.scope,
        graphify_version=graphify_version,
        session_id=session_id,
        snapshot_filename=snapshot_filename,
    )
    snapshot_path.write_text(header + report_content, encoding="utf-8")
    print(f"Snapshot written: {snapshot_path.relative_to(REPO_ROOT)}")
    print(f"  Stats: {node_count} nodes / {edge_count} edges / {community_count} communities")

    # 6. Update the index README
    _update_readme(
        captured=captured,
        label=args.label,
        short_sha=short_sha,
        node_count=node_count,
        edge_count=edge_count,
        community_count=community_count,
        snapshot_filename=snapshot_filename,
    )

    # 7. Commit unless --no-commit
    if not args.no_commit:
        _git_commit(
            snapshot_path=snapshot_path,
            scope=args.scope,
            label=args.label,
            short_sha=short_sha,
            node_count=node_count,
            edge_count=edge_count,
            community_count=community_count,
        )
    else:
        print("--no-commit: skipping git commit. Stage and commit manually if needed.")

    print("Done.")


if __name__ == "__main__":
    main()
