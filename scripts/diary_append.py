"""Append an event to the project diary at docs/diary/YYYY-MM-DD.md.

Holds an exclusive flock while doing the read-modify-write so concurrent
updates from parallel Claude sessions queue rather than clobber.

CLI subcommands:
    session-start    --label LBL --summary SUM [--link L] [--time HH:MM]
    session-end      --label LBL [--commits C] [--time HH:MM]
    implemented      --subject S --link L [--time HH:MM]
    verified         --subject S --link L [--time HH:MM]
    insight          --subject S --link L [--time HH:MM]
    training-start   --tag T --node N --gpu G --cell C --wandb W --doc D [--time HH:MM]
    training-done    --tag T --result R [--analysis A] [--time HH:MM]
    progress-report  --title T --what-this-did W --headline H --whats-next N
                     --sources S [--session SESS] [--time HH:MM]
    note             --text T

All commands accept --date YYYY-MM-DD (default: today, local Asia/Seoul time).

The script creates the daily file from docs/diary/TEMPLATE.md if missing, then
inserts/edits the appropriate row. New rows go at the TOP of their section
(reverse-chronological). For training-done, the existing training-start row
matching --tag is edited in place.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIARY_DIR = REPO_ROOT / "docs" / "diary"
TEMPLATE = DIARY_DIR / "TEMPLATE.md"

# Asia/Seoul = UTC+9 (no DST)
KST = timezone(timedelta(hours=9))


def resolve_session(arg: str | None, *, full: bool = False) -> str:
    """Resolve the session column value.

    - If --session was passed explicitly: use it verbatim (no truncation).
    - Else: derive from $CLAUDE_CODE_SESSION_ID env var.
      - full=True (used by session-start): return the full UUID, e.g.
        'f3ab7f37-218c-463b-ba24-e555d496dec1' — copy-paste-ready for
        `claude --resume <UUID>`.
      - full=False (everywhere else): return first 8 hex chars, e.g.
        'f3ab7f37' — compact for tables with many rows.
    - Else: 'unknown'.

    Sub-agents are expected to pass --session "<parent-prefix>/<role>" explicitly
    (e.g. "f3ab7f37/developer") so lineage is preserved in the diary row.

    Convention: the Sessions table carries the full UUID (one row per session,
    canonical anchor for `claude --resume`); Events and Training runs carry the
    prefix (many rows, compactness matters; cross-reference back to the Sessions
    row to recover the full UUID).
    """
    if arg:
        return arg.strip()
    env = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    if env:
        return env if full else env.split("-")[0][:8]
    return "unknown"


_HEX = re.compile(r"^[0-9a-f]{7,40}$")
_PATH_HINT = re.compile(r"[/\\]|\.(md|py|yaml|yml|json|toml|sh|txt|html|csv|tsv)$")

# Encoded project path Claude Code uses under ~/.claude/projects/<encoded>/<UUID>.jsonl.
# Same encoding for the synced copy under claude_data/.claude/projects/.
_PROJECT_ENCODED = "-media-nas01-projects-Interoceptive-AI-grid-world-pain"


def resolve_full_uuid(prefix: str) -> str | None:
    """Resolve an 8-char hex prefix to a full session UUID.

    Scans Claude Code's project directories — first ~/.claude/projects/<encoded>/
    (local, fresh sessions), then claude_data/.claude/projects/<encoded>/ (NAS-
    synced from peers). Returns the first 36-char UUID stem matching the prefix,
    or None if not found (e.g. for a brand-new session whose JSONL hasn't been
    flushed yet).
    """
    needle = prefix.lower()
    candidates = [
        Path.home() / ".claude" / "projects" / _PROJECT_ENCODED,
        REPO_ROOT / "claude_data" / ".claude" / "projects" / _PROJECT_ENCODED,
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        for entry in d.iterdir():
            stem = entry.name[:-6] if entry.name.endswith(".jsonl") else entry.name
            if len(stem) >= 36 and stem.lower().startswith(needle):
                return stem
    return None


def ensure_session_row(text: str, time: str, session_token: str) -> str:
    """Lazy-backfill a Sessions row when an event arrives from an un-anchored session.

    `session_token` is what appears in the Events / Training runs row — either
    'f3ab7f37' (top-level) or 'f3ab7f37/training-runner' (sub-agent). The
    PARENT's full UUID is what we put in the Sessions table (one row per
    top-level session, regardless of how many sub-agents it spawns).

    Skips silently when:
      - token is missing, malformed, or 'unknown'
      - the prefix can't be resolved to a full UUID (no JSONL on disk yet)
      - a Sessions row containing this full UUID already exists
    """
    if not session_token or session_token == "unknown":
        return text
    prefix = session_token.split("/")[0]
    if len(prefix) < 8 or not all(c in "0123456789abcdef" for c in prefix.lower()):
        return text

    full_uuid = resolve_full_uuid(prefix)
    if not full_uuid:
        return text

    # Already anchored?
    try:
        body_start, body_end = section_bounds(text, "Sessions")
    except SystemExit:
        return text
    for ln in text.splitlines()[body_start:body_end]:
        if full_uuid in ln:
            return text

    row = (
        f"| {time} | (open) | {full_uuid} | "
        f"(auto — session-start was not called) | "
        f"Auto-created on first event. |  |"
    )
    return insert_row_at_top(text, "Sessions", row)


def format_link(s: str) -> str:
    """Render a single link token in markdown-friendly form.

    - Empty / None → empty.
    - Already a markdown link (starts with '[') or anchor (starts with '<') → leave as-is.
    - 7–40 hex chars → `commit \`<hash>\`` (disambiguates git commit IDs).
    - Path-shaped (contains slash, or ends in a known extension) → `[<stem>](<path>)`,
      where <path> is rewritten to be relative TO THE DIARY FILE's directory
      (`docs/diary/`) so the link resolves correctly when clicked in any markdown
      viewer. Repo-relative input (e.g. `docs/memory/foo.md`) becomes
      `../../docs/memory/foo.md`; `docs/develop/x.md` becomes `../develop/x.md`.
    - Anything else (e.g. '(pending)', '(none)', URL) → leave as-is.
    """
    if not s:
        return ""
    s = s.strip()
    if not s:
        return ""
    if s.startswith("[") or s.startswith("<") or s.startswith("http"):
        return s
    if _HEX.match(s):
        return f"commit `{s}`"
    if _PATH_HINT.search(s):
        from os.path import basename, splitext, relpath
        stem = splitext(basename(s))[0]
        # Rewrite the path to be relative to docs/diary/ (where the diary file lives).
        # Input is expected to be repo-relative (e.g. "docs/memory/foo.md"); we
        # resolve it against REPO_ROOT, then make it relative to DIARY_DIR.
        try:
            target_abs = (REPO_ROOT / s).resolve()
            link = relpath(target_abs, start=DIARY_DIR)
        except Exception:
            link = s  # fall back to the raw input
        return f"[{stem}]({link})"
    return s


def format_links(s: str) -> str:
    """Format a whitespace-separated list of link tokens (used for --commits)."""
    if not s:
        return ""
    return " ".join(format_link(tok) for tok in s.split())


def today_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def now_hhmm() -> str:
    return datetime.now(KST).strftime("%H:%M")


def daily_path(date: str) -> Path:
    return DIARY_DIR / f"{date}.md"


def lock_path(date: str) -> Path:
    return Path(tempfile.gettempdir()) / f"diary-{date}.lock"


def ensure_daily_file(date: str) -> Path:
    p = daily_path(date)
    if p.exists():
        return p
    if not TEMPLATE.is_file():
        sys.exit(f"ERROR: template missing at {TEMPLATE}")
    text = TEMPLATE.read_text().replace("YYYY-MM-DD", date, 1)
    p.write_text(text)
    return p


def section_bounds(text: str, header: str) -> tuple[int, int]:
    """Return (start_line, end_line) of the given section's table body, exclusive
    of header and separator rows. The section ends at the next ## header or EOF.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == f"## {header}":
            start = i
            break
    if start is None:
        sys.exit(f"ERROR: section '{header}' not found in daily file")
    # Find table header (first | row after section start)
    header_row = None
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("|"):
            header_row = i
            break
        if lines[i].startswith("## "):
            break
    if header_row is None:
        sys.exit(f"ERROR: no table found under section '{header}'")
    sep_row = header_row + 1
    body_start = header_row + 2
    # Body ends at next ## or EOF
    body_end = len(lines)
    for i in range(body_start, len(lines)):
        if lines[i].startswith("## "):
            body_end = i
            break
    return body_start, body_end


def insert_row_at_top(text: str, header: str, row: str) -> str:
    """Insert a new row at the top of a section's table body (reverse-chrono).

    If the section currently shows the placeholder '_(no ... yet today)_' row,
    REPLACE that placeholder with the new row instead of stacking on top of it.
    """
    body_start, body_end = section_bounds(text, header)
    lines = text.splitlines(keepends=False)
    # Drop trailing blank lines inside the section body
    body = lines[body_start:body_end]
    # Strip trailing blanks
    while body and body[-1].strip() == "":
        body.pop()
    # Replace placeholder if present
    if body and "_(no " in body[0] and "yet today)_" in body[0]:
        body = [row] + body[1:]
    else:
        body = [row] + body
    # Reassemble
    new_lines = lines[:body_start] + body + [""] + lines[body_end:]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def edit_training_row(text: str, tag: str, ended: str, result: str, analysis_doc: str | None) -> str:
    """Find the training-runs row whose Tag column equals `tag` and update its
    Ended / Status / Result / Doc fields. Status becomes 'done HH:MM'.
    """
    body_start, body_end = section_bounds(text, "Training runs")
    lines = text.splitlines(keepends=False)
    body = lines[body_start:body_end]
    found = False
    for i, ln in enumerate(body):
        if not ln.lstrip().startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip("|").split("|")]
        # Layout: Started | Ended | Session | Tag | Node:GPU | Status | Cell | WandB | Result | Doc
        if len(cols) < 10:
            continue
        if cols[3] == tag:
            cols[1] = ended
            cols[5] = f"done {ended}"
            cols[8] = result
            if analysis_doc:
                cols[9] = analysis_doc
            body[i] = "| " + " | ".join(cols) + " |"
            found = True
            break
    if not found:
        sys.exit(f"ERROR: no open training-runs row with tag={tag!r} in {daily_path(today_str())}")
    new_lines = lines[:body_start] + body + lines[body_end:]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def edit_session_end_row(text: str, label: str, ended: str, commits: str | None) -> str:
    """Find the open session row whose Label equals `label` (Ended is empty or
    placeholder), update its Ended field and append commits to the Links column.
    """
    body_start, body_end = section_bounds(text, "Sessions")
    lines = text.splitlines(keepends=False)
    body = lines[body_start:body_end]
    found = False
    for i, ln in enumerate(body):
        if not ln.lstrip().startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip("|").split("|")]
        # Layout: Started | Ended | Session | Label | Summary | Links
        if len(cols) < 6:
            continue
        if cols[3] == label and (cols[1] == "" or cols[1] == "(open)"):
            cols[1] = ended
            if commits:
                cols[5] = (cols[5] + " " + commits).strip()
            body[i] = "| " + " | ".join(cols) + " |"
            found = True
            break
    if not found:
        sys.exit(f"ERROR: no open session row with label={label!r}")
    new_lines = lines[:body_start] + body + lines[body_end:]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def section_freeform_bounds(text: str, header: str) -> tuple[int, int]:
    """Return (start_line, end_line) of a freeform (non-table) section's body.

    body_start = line immediately after `## <header>`.
    body_end   = first line at next `## ` heading, or EOF.

    Unlike `section_bounds` (which targets table bodies), this helper makes no
    assumption about the section's content shape, so it is suitable for the
    `## Progress reports` section (which holds `### Session …` blocks separated
    by `---` rules rather than a markdown table).
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == f"## {header}":
            start = i
            break
    if start is None:
        sys.exit(f"ERROR: section '{header}' not found in daily file")
    body_start = start + 1
    body_end = len(lines)
    for i in range(body_start, len(lines)):
        if lines[i].startswith("## "):
            body_end = i
            break
    return body_start, body_end


def append_progress_report(
    text: str,
    session_token: str,
    title: str,
    what_this_did: str,
    headline: str,
    whats_next: str,
    sources: str,
) -> str:
    """Append or update a per-session Progress report under `## Progress reports`.

    Layout of a single entry:

        ### Session `<prefix>` — <title>

        **Full session UUID** (for `claude --resume`): `<full UUID or prefix>`

        **What this session did** (in plain words):

        <what_this_did>

        **Headline finding**:

        <headline>

        **What's next**:

        <whats_next>

        **Detailed sources**:

        <sources>

    **One progress report per session.** If an entry for the calling session's
    prefix already exists in today's `## Progress reports` section, it is
    REPLACED in place (preserving position). This keeps the diary compact —
    multiple `progress-report` calls within the same session do not stack.

    For entries from *different* sessions, the section grows oldest-first,
    entries separated by a horizontal rule (`---`). The first entry on a fresh
    day replaces the placeholder `_(no progress reports yet today)_`.
    """
    body_start, body_end = section_freeform_bounds(text, "Progress reports")
    lines = text.splitlines(keepends=False)
    body = lines[body_start:body_end]

    prefix = (session_token or "unknown").split("/")[0]
    full_uuid = resolve_full_uuid(prefix) if prefix and prefix != "unknown" else None
    uuid_display = full_uuid or (prefix if prefix != "unknown" else "(unresolved)")

    entry_lines = [
        f"### Session `{prefix}` — {title}",
        "",
        f"**Full session UUID** (for `claude --resume`): `{uuid_display}`",
        "",
        "**What this session did** (in plain words):",
        "",
        what_this_did.rstrip(),
        "",
        "**Headline finding**:",
        "",
        headline.rstrip(),
        "",
        "**What's next**:",
        "",
        whats_next.rstrip(),
        "",
        "**Detailed sources**:",
        "",
        sources.rstrip(),
    ]

    # Look for an existing entry for THIS session's prefix — if found, replace
    # in place rather than appending a new section.
    existing_header = f"### Session `{prefix}` —"
    existing_start = None
    for i, ln in enumerate(body):
        if ln.startswith(existing_header):
            existing_start = i
            break

    if existing_start is not None:
        # Find the end of this entry: the next `### Session` header (which
        # belongs to a different session), then walk back through trailing
        # blank lines and the `---` separator that precedes the next entry.
        existing_end = len(body)
        for i in range(existing_start + 1, len(body)):
            if body[i].startswith("### Session "):
                end = i
                while end > existing_start + 1 and body[end - 1].strip() in ("", "---"):
                    end -= 1
                existing_end = end
                break
        else:
            # Last entry in section — trim trailing blank lines only.
            while existing_end > existing_start + 1 and body[existing_end - 1].strip() == "":
                existing_end -= 1
        new_body = body[:existing_start] + entry_lines + body[existing_end:]
    else:
        has_entries = any(ln.startswith("### Session") for ln in body)
        if not has_entries:
            # First report: drop placeholder, write the entry at the bottom of
            # the quote block.
            kept = [ln for ln in body if "_(no progress reports yet today)_" not in ln]
            while kept and kept[-1].strip() in ("", "---"):
                kept.pop()
            new_body = kept + ["", *entry_lines, ""]
        else:
            kept = list(body)
            while kept and kept[-1].strip() in ("", "---"):
                kept.pop()
            new_body = kept + ["", "---", "", *entry_lines, ""]

    new_lines = lines[:body_start] + new_body + lines[body_end:]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def append_note(text: str, note: str) -> str:
    """Append a free-form bullet to the Notes section."""
    lines = text.splitlines(keepends=False)
    notes_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "## Notes":
            notes_idx = i
            break
    if notes_idx is None:
        sys.exit("ERROR: ## Notes section missing")
    # Find first non-blank, non-italic-placeholder line after header
    insert_at = notes_idx + 1
    while insert_at < len(lines) and (lines[insert_at].strip() == "" or lines[insert_at].strip().startswith("_(")):
        if lines[insert_at].strip().startswith("_("):
            # Drop placeholder
            lines.pop(insert_at)
            continue
        insert_at += 1
    # Insert bullet after the header (newest at top)
    lines.insert(notes_idx + 1, f"- {now_hhmm()} — {note}")
    if not (notes_idx + 2 < len(lines) and lines[notes_idx + 2].strip() == ""):
        lines.insert(notes_idx + 2, "")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def with_lock(date: str, action):
    """Acquire flock on the per-date lock file, run action, release."""
    lock_p = lock_path(date)
    with open(lock_p, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            action()
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def cmd_session_start(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        link = format_link(args.link or "")
        # Sessions row uses the FULL UUID (copy-paste-ready for `claude --resume`).
        sess = resolve_session(args.session, full=True)
        row = f"| {time} | (open) | {sess} | {args.label} | {args.summary} | {link} |"
        text = insert_row_at_top(text, "Sessions", row)
        write_atomic(path, text)
        print(f"session-start logged at {time} (session={sess}) into {path.name}")
    with_lock(args.date, go)


def cmd_session_end(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        commits_md = format_links(args.commits) if args.commits else None
        text = edit_session_end_row(text, args.label, time, commits_md)
        write_atomic(path, text)
        print(f"session-end logged at {time} into {path.name}")
    with_lock(args.date, go)


def cmd_event(args, type_label):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        link = format_link(args.link)
        sess = resolve_session(args.session)
        text = ensure_session_row(text, time, sess)
        row = f"| {time} | {sess} | {type_label} | {args.subject} | {link} |"
        text = insert_row_at_top(text, "Events (chronological, newest first)", row)
        write_atomic(path, text)
        print(f"{type_label} logged at {time} (session={sess}) into {path.name}")
    with_lock(args.date, go)


def cmd_training_start(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        doc_md = format_link(args.doc)
        sess = resolve_session(args.session)
        text = ensure_session_row(text, time, sess)
        row = (
            f"| {time} |  | {sess} | {args.tag} | {args.node}:{args.gpu} | running | "
            f"{args.cell} | {args.wandb} | — | {doc_md} |"
        )
        text = insert_row_at_top(text, "Training runs", row)
        write_atomic(path, text)
        print(f"training-start logged at {time} (session={sess}, tag={args.tag}) into {path.name}")
    with_lock(args.date, go)


def cmd_training_done(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        analysis_md = format_link(args.analysis) if args.analysis else None
        text = edit_training_row(text, args.tag, time, args.result, analysis_md)
        write_atomic(path, text)
        print(f"training-done logged at {time} (tag={args.tag}) into {path.name}")
    with_lock(args.date, go)


def cmd_progress_report(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        sess = resolve_session(args.session)
        text = ensure_session_row(text, time, sess)
        text = append_progress_report(
            text,
            session_token=sess,
            title=args.title,
            what_this_did=args.what_this_did,
            headline=args.headline,
            whats_next=args.whats_next,
            sources=args.sources,
        )
        write_atomic(path, text)
        prefix = (sess or "unknown").split("/")[0]
        print(f"progress-report logged at {time} (session={prefix}) into {path.name}")
    with_lock(args.date, go)


def cmd_note(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        text = append_note(text, args.text)
        write_atomic(path, text)
        print(f"note appended to {path.name}")
    with_lock(args.date, go)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", default=today_str(), help="YYYY-MM-DD; default today (KST)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("session-start"); s.add_argument("--label", required=True)
    s.add_argument("--summary", required=True); s.add_argument("--link", default="")
    s.add_argument("--time"); s.add_argument("--session"); s.set_defaults(func=cmd_session_start)

    s = sub.add_parser("session-end"); s.add_argument("--label", required=True)
    s.add_argument("--commits"); s.add_argument("--time"); s.set_defaults(func=cmd_session_end)

    for ev in ("implemented", "verified", "insight"):
        s = sub.add_parser(ev); s.add_argument("--subject", required=True)
        s.add_argument("--link", required=True); s.add_argument("--time")
        s.add_argument("--session")
        s.set_defaults(func=lambda a, e=ev: cmd_event(a, e))

    s = sub.add_parser("training-start")
    for f in ("tag", "node", "gpu", "cell", "wandb", "doc"):
        s.add_argument(f"--{f}", required=True)
    s.add_argument("--time"); s.add_argument("--session"); s.set_defaults(func=cmd_training_start)

    s = sub.add_parser("training-done"); s.add_argument("--tag", required=True)
    s.add_argument("--result", required=True); s.add_argument("--analysis", default=None)
    s.add_argument("--time"); s.set_defaults(func=cmd_training_done)

    s = sub.add_parser("progress-report")
    s.add_argument("--title", required=True)
    s.add_argument("--what-this-did", required=True)
    s.add_argument("--headline", required=True)
    s.add_argument("--whats-next", required=True)
    s.add_argument("--sources", required=True)
    s.add_argument("--time"); s.add_argument("--session")
    s.set_defaults(func=cmd_progress_report)

    s = sub.add_parser("note"); s.add_argument("--text", required=True)
    s.set_defaults(func=cmd_note)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
