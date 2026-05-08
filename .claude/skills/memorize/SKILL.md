---
name: memorize
description: "Capture the current conversation as one or more session-insight files in this project's in-repo memory system at .claude-memory/. Use whenever the user says 'remember this', 'save this', 'memorize this', '/memorize', 'save to memory', or session-end phrases like 'wrap up', 'wrapping up', 'we're done', 'good night', 'end of session', 'let's call it'. Also trigger when the user asks to capture a decision, finding, debugging conclusion, design rationale, rejected alternative, or 'what we learned' for future retrieval. The skill writes 14-field-frontmatter + 5-section insight files, updates the topic and global indexes under .claude-memory/, and optionally archives a raw conversation. Use proactively at end-of-session even if the user does not name memory explicitly. Do NOT use for short typed rules another agent must obey on every invocation — those go to the built-in Claude Code auto-memory at ~/.claude/.../memory/MEMORY.md, which is a different layer with a different mechanism."
---

# Memorize — capture conversation insights into `.claude-memory/`

This skill captures the current conversation into the in-repo memory layer at `.claude-memory/`. It produces one or more 5-section insight files, updates indexes, and optionally writes a raw-conversation archive.

**Authoritative contract**: `.claude-memory/CLAUDE.md`. Treat it as ground truth for the frontmatter schema, naming rules, fragmentation safeguards, and raw-archive policy. This file describes the *invocation flow*; the operating manual carries the *contract*. If they disagree, the operating manual wins.

## When to use

Trigger on:

- `/memorize`, "remember this", "save this", "memorize this", "save to memory".
- Session-end keywords: "wrap up", "wrapping up", "let's call it", "good night", "we're done", "end of session".
- User asks to capture a decision, finding, debugging conclusion, design rationale, or rejected alternative.

Do **not** use for:

- Short typed operational rules another agent must obey on every invocation → built-in `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` (harness-managed, different layer).
- Project-wide invariants → project root `CLAUDE.md`.
- Recall ("what did we decide?") → no skill needed; read `.claude-memory/` directly.

## Quick reference (read first; details on demand)

| Thing | Value | Notes |
|---|---|---|
| Operating manual | `.claude-memory/CLAUDE.md` | Read on every invocation. |
| Topic registry | `.claude-memory/ROOT_INDEX.md` | Read before classifying (safeguard layer 1). |
| Tag dictionary | `.claude-memory/memories/_global_tags.md` | Reuse existing tags. |
| Insight template | `.claude-memory/TEMPLATES/insight.md` | Copy as starting point. |
| Insight path | `.claude-memory/memories/<topic>/<id>.md` | `<topic>` is English snake_case ≤ ~3 words. |
| Insight filename | `YYYYMMDD_HHMM_<slug>.md` | `date +%Y%m%d_%H%M`; slug ≤ ~6 words English snake_case. |
| Frontmatter | 14 fields | id, date, time, folder, tags, summary, related, session_origin, session_label, importance, status, supersedes, raw_source, raw_completeness. |
| Body sections | 5 named, in order | `## Key conclusion` / `## Evidence, measurements, facts` / `## Decisions and actions` / `## Open questions and follow-ups` / `## References`. |
| Raw archive | `.claude-memory/_archive/raw_conversations/<id>.md` | Gitignored (local-only). Pointed at by insight `raw_source`. |
| Full-raw helper | `scripts/claude_jsonl_to_md.py <jsonl> <out>` | Use conda Python `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. |

## Step-by-step flow

### Step 1 — Read context

Read three files (one read per file, no duplicates):

1. `.claude-memory/CLAUDE.md` — the contract.
2. `.claude-memory/ROOT_INDEX.md` — every existing folder's 1-line definition.
3. `.claude-memory/memories/_global_tags.md` — active tags.

If any is missing, halt: "`.claude-memory/` is not seeded. Run the seed flow before invoking memorize." Do not create the seed yourself.

### Step 2 — Identify candidate insights

Walk the recent conversation. Identify each distinct unit at the insight-density bar — a decision, a finding, a rejected alternative, a debugging conclusion, a design rationale.

Anti-pattern: do **not** dump the whole session into one mega-insight. Split by distinct decision/finding so each insight is independently useful at retrieval time.

For each candidate, draft:

- One-line plain-English summary → `summary` frontmatter field.
- Slug — English snake_case, ≤ ~6 words.
- Importance: `high` / `medium` / `low`.

Show the candidate list and ask:

> I found N candidate insight(s):
>
> 1. <one-liner> — slug `<slug>`, importance `<level>`
> 2. ...
>
> Save which? (Y = all / N = none / numbers like "1,3" / edit to revise)

If non-interactive (subagent eval, batch run), default to Y on all candidates.

### Step 3 — Topic-folder routing (safeguard layer 1)

For each confirmed insight, match against `ROOT_INDEX.md` definitions:

- **Strong match** — use that folder.
- **Medium / partial** — ask the user; default to existing folder if non-interactive.
- **No match** — propose new folder (English snake_case, ≤ ~3 words, definition ≤ ~30 chars). Ask to confirm; default to the proposal if non-interactive.

For any new folder, the insight body must include a **"Why a new folder"** justification line in `## References` (safeguard layer 2). Name the closest existing folder and explain why the new insight does not fit there.

### Step 4 — Write each insight file

For each confirmed insight:

1. Compute `YYYYMMDD_HHMM` from `date +%Y%m%d_%H%M` (Bash). Use the same value in the filename, the `id` frontmatter field, and the `time` field consistently. If two insights share an `HHMM`, bump the second by one minute.

2. Write to `.claude-memory/memories/<topic>/<id>.md`. Start from `.claude-memory/TEMPLATES/insight.md`. Fill all 14 frontmatter fields:
   - `folder` must match the parent directory.
   - `tags` reuse from `_global_tags.md`; new tags allowed but require a row in step 6.
   - `session_origin: claude_code` for any Claude Code session.
   - `status: settled` if the conclusion holds firmly; `active` if still being shaped.
   - `raw_source: none` and `raw_completeness: none` for now (Step 7 may overwrite).

3. Fill all 5 body sections in plain English. `## Open questions and follow-ups` writes "None" if there are none. `## References` includes the "Why a new folder" line if applicable.

### Step 5 — Update the topic index

For each insight:

- New topic folder → create `.claude-memory/memories/<topic>/_topic_index.md` with the standard header (read an existing topic index for the format) and a single row.
- Existing folder → prepend a row (reverse-chronological) to the existing `_topic_index.md`. Match the existing row format.

### Step 6 — Update root index and tag dictionary

`.claude-memory/ROOT_INDEX.md`:

- Bump `Total insights`.
- Update each touched folder's row: `Insights` count, `Last update` date, `Top tags` list.
- New folder → append a row with `folder | definition | count=1 | last_update | tags=[…]` (this is the **definition lock**, safeguard layer 3 — future-Claude matches against this string verbatim).
- Append one line under `## Change history`: `YYYY-MM-DD: Captured N insight(s) into <topic>(s).`

`.claude-memory/memories/_global_tags.md`:

- Existing tag used → bump count; do not change first-use ID.
- New tag → append row with `tag | meaning | count=1 | first_use=<insight_id>`. English snake_case, singular, no slashes.

### Step 7 — Raw-archive prompt (optional)

After all insights are written, ask:

> Also archive the raw conversation? (full / approximate / skip)

- **full** — verbatim via `scripts/claude_jsonl_to_md.py`. `raw_completeness: full`.
- **approximate** — Claude writes a markdown summary of recent turns. Smaller, less complete. `raw_completeness: approximate`.
- **skip** — no archive; fields stay at `none`.

Non-interactive default: **skip**, unless the user's original message explicitly asked for a raw archive.

If **full**:

```bash
JSONL=$(ls -t ~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/*.jsonl | head -1)
ARCH=.claude-memory/_archive/raw_conversations/<chronologically-first-insight-id>.md
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/claude_jsonl_to_md.py "$JSONL" "$ARCH"
```

Then update each insight's `raw_source` and `raw_completeness`, and add a one-line note in `## References`: `raw_source link is local-only (archives are gitignored).`

If **approximate**: write a markdown summary of recent ~20 turns into `.claude-memory/_archive/raw_conversations/<id>.md`. Same field updates and local-only note.

If archive size > 50 KB, surface: `Archive ~XX KB: check for sensitive content before committing the insight.` (The archive is gitignored; only the insight file is committed.)

### Step 8 — Log each insight to the diary (mandatory)

After all insights are written and indexes updated, call the diary helper for **each** insight written. One call per insight; the script flock-protects concurrent invocations from parallel sessions.

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/diary_append.py insight \
  --subject "<copy from the insight's frontmatter `summary` field>" \
  --link   ".claude-memory/memories/<topic>/<id>.md"
```

This is a hard step, not optional — the diary is the project's cross-session status board, and a memory capture without a diary entry creates an invisible gap. If a diary call errors (e.g., `docs/diary/` is missing), surface the error to the user but do not roll back the insight writes; the insights are the durable artefact.

### Step 9 — Confirm and report

Report in plain English:

> Saved N insight(s) to `.claude-memory/` (and logged to today's diary):
>
> - `<topic>` — <one-liner>
> - `<topic>` — <one-liner>
>
> Indexes updated. Raw archive: <full / approximate / skip>.

Do **not** offer to commit. The skill ends at the file writes; the user holds standing auto-commit authorization.

## Hard rules

- **Repo-relative paths only.** Never write to `~/.claude/...`, `/home/`, or any absolute path outside the current repo. The skill writes only inside the project tree under `.claude-memory/` and `scripts/`. This matters in worktrees and sandboxes — absolute paths defeat isolation.
- **Never write to** `~/.claude/projects/.../memory/MEMORY.md` from this skill — that layer is harness-managed by Claude Code itself.
- All filenames English snake_case; all content English; no emojis unless the user explicitly asks.
- Conda Python: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. Never `python3`.
- "Delete" = move to `.claude-memory/.trash/`. Hard-delete only on explicit user instruction.
- Fragmentation safeguards (4 layers) are non-optional; skipping any of them is a regression.
- If `.claude-memory/CLAUDE.md` and this SKILL.md disagree, the operating manual wins — flag it and ask the user.

## References

- `.claude-memory/CLAUDE.md` — operating manual (sections 4, 5, 6, 7, 11).
- `.claude-memory/TEMPLATES/insight.md` — copy as starting point.
- `.claude-memory/ROOT_INDEX.md` — topic registry.
- `.claude-memory/memories/_global_tags.md` — tag dictionary.
- `scripts/claude_jsonl_to_md.py` — full-raw archive helper.
- `docs/develop/active/meta/claude_memory_system_design.md` — design rationale.
