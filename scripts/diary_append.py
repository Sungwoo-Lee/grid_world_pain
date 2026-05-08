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


_HEX = re.compile(r"^[0-9a-f]{7,40}$")
_PATH_HINT = re.compile(r"[/\\]|\.(md|py|yaml|yml|json|toml|sh|txt|html|csv|tsv)$")


def format_link(s: str) -> str:
    """Render a single link token in markdown-friendly form.

    - Empty / None → empty.
    - Already a markdown link (starts with '[') or anchor (starts with '<') → leave as-is.
    - 7–40 hex chars → `commit \`<hash>\`` (disambiguates git commit IDs).
    - Path-shaped (contains slash, or ends in a known extension) → `[<stem>](<path>)`.
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
        from os.path import basename, splitext
        stem = splitext(basename(s))[0]
        return f"[{stem}]({s})"
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
        # Layout: Started | Ended | Tag | Node:GPU | Status | Cell | WandB | Result | Doc
        if len(cols) < 9:
            continue
        if cols[2] == tag:
            cols[1] = ended
            cols[4] = f"done {ended}"
            cols[7] = result
            if analysis_doc:
                cols[8] = analysis_doc
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
        # Layout: Started | Ended | Label | Summary | Links
        if len(cols) < 5:
            continue
        if cols[2] == label and (cols[1] == "" or cols[1] == "(open)"):
            cols[1] = ended
            if commits:
                cols[4] = (cols[4] + " " + commits).strip()
            body[i] = "| " + " | ".join(cols) + " |"
            found = True
            break
    if not found:
        sys.exit(f"ERROR: no open session row with label={label!r}")
    new_lines = lines[:body_start] + body + lines[body_end:]
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
        row = f"| {time} | (open) | {args.label} | {args.summary} | {link} |"
        text = insert_row_at_top(text, "Sessions", row)
        write_atomic(path, text)
        print(f"session-start logged at {time} into {path.name}")
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
        row = f"| {time} | {type_label} | {args.subject} | {link} |"
        text = insert_row_at_top(text, "Events (chronological, newest first)", row)
        write_atomic(path, text)
        print(f"{type_label} logged at {time} into {path.name}")
    with_lock(args.date, go)


def cmd_training_start(args):
    def go():
        path = ensure_daily_file(args.date)
        text = path.read_text()
        time = args.time or now_hhmm()
        doc_md = format_link(args.doc)
        row = (
            f"| {time} |  | {args.tag} | {args.node}:{args.gpu} | running | "
            f"{args.cell} | {args.wandb} | — | {doc_md} |"
        )
        text = insert_row_at_top(text, "Training runs", row)
        write_atomic(path, text)
        print(f"training-start logged at {time} (tag={args.tag}) into {path.name}")
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
    s.add_argument("--time"); s.set_defaults(func=cmd_session_start)

    s = sub.add_parser("session-end"); s.add_argument("--label", required=True)
    s.add_argument("--commits"); s.add_argument("--time"); s.set_defaults(func=cmd_session_end)

    for ev in ("implemented", "verified", "insight"):
        s = sub.add_parser(ev); s.add_argument("--subject", required=True)
        s.add_argument("--link", required=True); s.add_argument("--time")
        s.set_defaults(func=lambda a, e=ev: cmd_event(a, e))

    s = sub.add_parser("training-start")
    for f in ("tag", "node", "gpu", "cell", "wandb", "doc"):
        s.add_argument(f"--{f}", required=True)
    s.add_argument("--time"); s.set_defaults(func=cmd_training_start)

    s = sub.add_parser("training-done"); s.add_argument("--tag", required=True)
    s.add_argument("--result", required=True); s.add_argument("--analysis", default=None)
    s.add_argument("--time"); s.set_defaults(func=cmd_training_done)

    s = sub.add_parser("note"); s.add_argument("--text", required=True)
    s.set_defaults(func=cmd_note)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
