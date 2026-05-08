---
name: diary
description: "Append an event to the project's daily diary at docs/diary/YYYY-MM-DD.md. ALWAYS use this skill when one of these moments fires, even if the user does not ask for it: (a) a top-level Claude session is starting or wrapping up, (b) the developer agent finishes implementing a plan or the senior-developer finishes verifying one, (c) /memorize captures one or more insights, (d) the training-runner launches a training run on a lab node, (e) the experiment-analyzer finishes analyzing a training run. Also trigger on the explicit slash command /diary or natural-language phrases like 'log this in the diary', 'add to today's diary', 'note this'. The skill records a single short row per event and links out to the authoritative document — it does NOT duplicate plans, insights, or analyses. For training runs the same row is edited from 'running' to 'done' when the analysis completes, so the diary is a single-glance status board across parallel sessions. The skill calls scripts/diary_append.py, which uses an exclusive flock to handle concurrent writes from parallel Claude sessions."
---

# Diary — log short-timeline events to `docs/diary/YYYY-MM-DD.md`

This skill appends one row to today's diary file for a notable event. The diary is the project's at-a-glance status board across parallel Claude sessions: who started what, what landed, what insights were captured, what's training, what finished training.

The skill is a thin wrapper over `scripts/diary_append.py`, which holds an exclusive `flock` while doing the read-modify-write so concurrent updates queue rather than clobber.

**Authoritative reference**: `docs/diary/README.md` (folder purpose + section schema). Treat it as ground truth for the row layouts. This SKILL.md describes WHEN to invoke and WHICH subcommand to use; the README + the script's `--help` carry the contract.

## When to use

ALWAYS invoke at these moments, even without an explicit user request:

| Moment | Subcommand |
|---|---|
| Top-level Claude session is starting an obviously-multi-step task | `session-start` |
| Same session is wrapping up (commit boundary, "we're done", explicit `/wrap`) | `session-end` |
| `developer` agent reports an implementation complete | `implemented` |
| `senior-developer` reports a verification complete | `verified` |
| `/memorize` writes 1+ insight files | `insight` (one call per insight) |
| `training-runner` launches a training | `training-start` |
| `experiment-analyzer` finishes analyzing a training | `training-done` |
| User says "log this in the diary", "note this", `/diary`, or asks for a one-off entry | the matching subcommand, or `note` for free-form |

Do NOT use for:

- Multi-section session insights with rationale → `/memorize` (which itself triggers a `diary insight` call).
- Plans → `docs/develop/active/<topic>/<plan>.md`.
- Experimental designs / analyses → `docs/experiments/active/<topic>/<doc>.md`.

The diary is a pointer index, not a duplicate of any of these.

## How to invoke

The skill calls `scripts/diary_append.py` via Bash with the conda Python interpreter. One subcommand per call. Default date = today (Asia/Seoul); override with `--date YYYY-MM-DD`. Default time = now (HH:MM); override with `--time HH:MM`.

Use `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/diary_append.py <subcommand> <args>`.

### `session-start` — when a multi-step session begins

```bash
.../python scripts/diary_append.py session-start \
  --label "<short-label, e.g., 'memory + memorize ship'>" \
  --summary "<one-line goal of this session>" \
  --link "<optional plan or doc path>"
```

Inserts a row into `## Sessions` with `Ended` blank (`(open)`).

### `session-end` — when the session wraps up

```bash
.../python scripts/diary_append.py session-end \
  --label "<must match the session-start label>" \
  --commits "<space-separated commit hashes from this session>"
```

Edits the matching open `Sessions` row in place: sets `Ended = HH:MM`, appends commits to `Links`. Errors if no open row with that label exists.

### `implemented` / `verified` / `insight` — single-event rows

```bash
.../python scripts/diary_append.py implemented \
  --subject "<one-line, e.g., '/memorize skill'>" \
  --link   "<commit hash, plan path, or insight ID>"

.../python scripts/diary_append.py verified \
  --subject "<one-line, e.g., 'Memory system seed'>" \
  --link   "<plan doc path>"

.../python scripts/diary_append.py insight \
  --subject "<one-line, copy from the insight's summary frontmatter>" \
  --link   "<.claude-memory/memories/<topic>/<id>.md>"
```

Inserts a row at the top of `## Events (chronological, newest first)`.

For `insight`: emit one call per insight written. If `/memorize` writes 2 insights, call this twice.

### `training-start` — when a training launches

```bash
.../python scripts/diary_append.py training-start \
  --tag "<TAG from the launch manifest>" \
  --node 113 --gpu 1 \
  --cell "<cell letter, or '-' if not part of a cell battery>" \
  --wandb "<wandb run name>" \
  --doc "<docs/experiments/active/<topic>/<design-doc>.md>"
```

Inserts a row into `## Training runs` with `Status = running` and `Ended` blank.

### `training-done` — when the analysis completes

```bash
.../python scripts/diary_append.py training-done \
  --tag "<must match the training-start TAG>" \
  --result "<one-line result, e.g., 'survival 23 ± 2 steps'>" \
  --analysis "<docs/experiments/active/<topic>/<analysis-doc>.md>"
```

Edits the matching `Training runs` row in place: sets `Ended = HH:MM`, `Status = done HH:MM`, fills `Result`, replaces `Doc` with the analysis doc.

### `note` — free-form

```bash
.../python scripts/diary_append.py note --text "<bullet content>"
```

Prepends a bullet to `## Notes`. Use sparingly; structured rows are preferred.

## Session column (who wrote each row)

Every row carries a `Session` column so you can tell which Claude session wrote it — important when multiple sessions update the diary in parallel.

- **Top-level Claude session**: pass nothing. The script defaults to the first 8 hex chars of `$CLAUDE_CODE_SESSION_ID` (e.g. `f3ab7f37`). This env var is set by Claude Code in every shell the harness spawns.
- **Sub-agent (`developer`, `senior-developer`, `training-runner`, `experiment-analyzer`)**: pass `--session "${CLAUDE_CODE_SESSION_ID:0:8}/<role>"` explicitly so the row reads `f3ab7f37/developer`. The slash separator makes lineage visible: parent session prefix on the left, sub-agent role on the right. Each agent's profile under `.claude/agents/` carries the exact `--session` value to use.
- **Manual / scripted call from outside Claude Code**: pass any short label, e.g. `--session "manual"` or `--session "cron"`. If unset and `$CLAUDE_CODE_SESSION_ID` is empty, the script writes `unknown` and continues.

The Session column is informational — it does not affect row matching for `session-end`, `training-done`, etc. (those still match by `--label` and `--tag`).

## Link auto-formatting

The script auto-formats whatever you pass to `--link` / `--doc` / `--analysis` / `--commits`:

- **Git commit hash** (7–40 hex chars, e.g. `a16a6c9` or `dfa3c68f4e1`) → rendered as `` commit `<hash>` `` so the row makes clear it is a commit ID, not a generic identifier.
- **Repo-relative path** (contains `/` or ends in `.md` / `.py` / `.yaml` / etc.) → rendered as `[<filename-stem>](<path>)` so it is clickable in any markdown viewer. Use repo-relative paths so the link is valid on a fresh clone.
- **Already a markdown link** (`[text](url)`), an HTML anchor (`<a …>`), or a URL (`http…`) → left as-is.
- **Anything else** (`(pending)`, `(none)`, free text) → left as-is.

So: always pass the **raw value** — the commit hash by itself, the path by itself — and let the script wrap it. Do not pre-format markdown links yourself; that is the script's job.

For `session-end --commits "a16a6c9 f878873"` (whitespace-separated list), each token is formatted independently.

## Hard rules

- **Always pass repo-relative paths in `--link`, `--doc`, `--analysis`.** Never absolute paths from `$HOME` or system roots — links must be valid on a fresh clone.
- **One subcommand per invocation.** If you have 2 insights to log, call the script twice. Each call acquires the flock independently, so concurrent calls from parallel sessions are safe.
- **Conda Python**: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. Never `python3`.
- **Tags must match** between `training-start` and `training-done` (the script edits in place by tag).
- **Session labels must match** between `session-start` and `session-end` (same lookup).
- **Subject and result text**: keep to one line, no newlines (would break the markdown table). The diary row is meant to be scannable.
- All content English; no emojis unless the user explicitly asks.
- **Do not commit from this skill.** The diary file is updated, but commit decisions belong to the user. The skill ends at the script call.
- If the script errors (e.g., training-done with no matching tag, session-end with no matching label), surface the error to the user — do not silently retry or fabricate the missing row.

## Why a script and not Edit

Parallel Claude sessions writing to the same daily file would race if each one read-then-Edit'd it. The script holds an exclusive `flock` on `/tmp/diary-<date>.lock` for the duration of read-modify-write, so concurrent calls queue instead of clobbering. Using Edit/Write directly from the skill bypasses this and risks lost rows.

## References

- `docs/diary/README.md` — folder purpose, section schema, conventions.
- `docs/diary/TEMPLATE.md` — daily file template.
- `scripts/diary_append.py` — the helper script (see `--help` for arg details).
- Companion skills: `.claude/skills/memorize/SKILL.md` (capture), `.claude/skills/recall/SKILL.md` (read).
