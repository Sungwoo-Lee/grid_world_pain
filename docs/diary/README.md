# Diary — short-timeline event log

> One file per date: `docs/diary/YYYY-MM-DD.md`. Updated automatically by parallel Claude sessions whenever notable events fire. Append-only with `flock` to handle concurrent writes.

## Purpose

A compact, scannable record of what's happening across all parallel Claude sessions on this project. Where the in-repo memory layer at `.claude-memory/` carries multi-section session insights, the diary carries the **short timeline**: when sessions started and ended, when implementations and verifications landed, when memory captures fired, and when training runs launched + finished.

The diary is intentionally compact. For details, each row links to the authoritative document — the plan in `docs/develop/`, the experiment doc in `docs/experiments/`, the insight in `.claude-memory/`, the commit hash in git, the WandB run in the dashboard.

## Daily file structure

Each daily file (`docs/diary/YYYY-MM-DD.md`) has five sections in this order:

1. **`## Sessions`** — start/end of top-level Claude sessions, one row each.
2. **`## Events (chronological)`** — one row per implementation, verification, or insight capture, reverse-chronological at the top.
3. **`## Training runs`** — one row per training, with start time, status, end time, result, and links. Rows are edited in place (status changes from `running` → `done` when analysis completes).
4. **`## Progress reports`** — one concise per-session wrap-up in plain language, separate from the row tables above. Fired on `session-end` of multi-step sessions. **One entry per session prefix** — a second `progress-report` call from the same session REPLACES that session's entry in place (keeps the section scannable); a call from a different session appends below, separated by a `---` rule (oldest-first).
5. **`## Notes`** — free-form bullets for anything that doesn't fit the above.

See `TEMPLATE.md` for the canonical layout.

## How updates happen

Three mechanisms (any of them is valid):

- **Skill (`/diary`)** at `.claude/skills/diary/SKILL.md` — triggered by Claude when an event fires. The skill calls the helper script under the hood.
- **Helper script** `scripts/diary_append.py` — direct CLI for scripts/agents that don't go through the skill. Same locking semantics.
- **Manual edit** — fine for one-off notes. Use the Notes section to avoid colliding with the structured tables.

The helper script holds an exclusive `flock` on `/tmp/diary-<date>.lock` while it does the read-modify-write, so concurrent updates from parallel sessions queue rather than clobber.

## What's not here

- Multi-section insights with rationale and rejected alternatives → `.claude-memory/memories/<topic>/<id>.md` (use `/memorize`).
- Implementation plans → `docs/develop/active/<topic>/<plan>.md`.
- Experimental designs and analyses → `docs/experiments/active/<topic>/<doc>.md`.
- Code, configs, scripts → the rest of the repo.

The diary is the **index** to those, not a duplicate.

## Conventions

- **Date** is local date at write time (Asia/Seoul, the project's working timezone).
- **Time** is `HH:MM` 24-hour, no seconds.
- **Reverse-chronological at every level** (newest at the top of each section).
- All content English; no emojis unless explicitly requested.
- Links use repo-relative paths (e.g., `docs/develop/active/meta/foo.md`) so the diary is meaningful on any clone.
- Link cells are auto-formatted by `scripts/diary_append.py`: 7–40 hex chars become `` commit `<hash>` `` (so a git commit ID is unambiguous), and path-shaped strings become `[<filename-stem>](<path>)` (so the link is clickable in any markdown viewer). Pass raw values to the script — do not pre-wrap them yourself.
