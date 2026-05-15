---
name: memorize
description: "Capture the current conversation as one or more session-insight files in this project's in-repo memory system at docs/memory/. Use whenever the user says 'remember this', 'save this', 'memorize this', '/memorize', 'save to memory', or session-end phrases like 'wrap up', 'wrapping up', 'we're done', 'good night', 'end of session', 'let's call it'. Also trigger when the user asks to capture a decision, finding, debugging conclusion, design rationale, rejected alternative, or 'what we learned' for future retrieval. The skill writes 14-field-frontmatter + 5-section insight files, updates the topic and global indexes under docs/memory/, and optionally archives a raw conversation. Use proactively at end-of-session even if the user does not name memory explicitly. Do NOT use for short typed rules another agent must obey on every invocation — those go to the built-in Claude Code auto-memory at ~/.claude/.../memory/MEMORY.md, which is a different layer with a different mechanism."
---

# Memorize — capture conversation insights into `docs/memory/`

This skill captures the current conversation into the in-repo memory layer at `docs/memory/`. It produces one or more 5-section insight files, updates indexes, and optionally writes a raw-conversation archive.

**Authoritative contract**: `docs/memory/CLAUDE.md`. Treat it as ground truth for the frontmatter schema, naming rules, fragmentation safeguards, and raw-archive policy. This file describes the *invocation flow*; the operating manual carries the *contract*. If they disagree, the operating manual wins.

## When to use

Trigger on:

- `/memorize`, "remember this", "save this", "memorize this", "save to memory".
- Session-end keywords: "wrap up", "wrapping up", "let's call it", "good night", "we're done", "end of session".
- User asks to capture a decision, finding, debugging conclusion, design rationale, or rejected alternative.

Do **not** use for:

- Short typed operational rules another agent must obey on every invocation → built-in `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` (harness-managed, different layer).
- Project-wide invariants → project root `CLAUDE.md`.
- Recall ("what did we decide?") → no skill needed; read `docs/memory/` directly.

## Quick reference (read first; details on demand)

| Thing | Value | Notes |
|---|---|---|
| Operating manual | `docs/memory/CLAUDE.md` | Read on every invocation. |
| Topic registry | `docs/memory/ROOT_INDEX.md` | Read before classifying (safeguard layer 1). |
| Tag dictionary | `docs/memory/memories/_global_tags.md` | Reuse existing tags. |
| Insight template | `docs/memory/TEMPLATES/insight.md` | Copy as starting point. |
| Insight path | `docs/memory/memories/<topic>/<id>.md` | `<topic>` is English snake_case ≤ ~3 words. |
| Insight filename | `YYYYMMDD_HHMM_<slug>.md` | `date +%Y%m%d_%H%M`; slug ≤ ~6 words English snake_case. |
| Frontmatter | 14 fields | id, date, time, folder, tags, summary, related, session_origin, session_label, importance, status, supersedes, raw_source, raw_completeness. |
| Body sections | 5 named, in order | `## Key conclusion` / `## Evidence, measurements, facts` / `## Decisions and actions` / `## Open questions and follow-ups` / `## References`. |
| Raw archive | `docs/memory/_archive/raw_conversations/<id>.md` | Gitignored (local-only). Pointed at by insight `raw_source`. |
| Full-raw helper | `scripts/claude_jsonl_to_md.py <jsonl> <out>` | Use conda Python `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. |

## Step-by-step flow

### Step 1 — Read context

Read three files (one read per file, no duplicates):

1. `docs/memory/CLAUDE.md` — the contract.
2. `docs/memory/ROOT_INDEX.md` — every existing folder's 1-line definition.
3. `docs/memory/memories/_global_tags.md` — active tags.

If any is missing, halt: "`docs/memory/` is not seeded. Run the seed flow before invoking memorize." Do not create the seed yourself.

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

2. Write to `docs/memory/memories/<topic>/<id>.md`. Start from `docs/memory/TEMPLATES/insight.md`. Fill all 14 frontmatter fields:
   - `folder` must match the parent directory.
   - `tags` reuse from `_global_tags.md`; new tags allowed but require a row in step 6.
   - `session_origin: claude_code` for any Claude Code session.
   - `status: settled` if the conclusion holds firmly; `active` if still being shaped.
   - `raw_source: none` and `raw_completeness: none` for now (Step 7 may overwrite).

3. Fill all 5 body sections in plain English. `## Open questions and follow-ups` writes "None" if there are none. `## References` includes the "Why a new folder" line if applicable.
   - When referencing another insight, write `[[<insight_id>]]` inline in the body section (typically `## References` or `## Decisions and actions`). The `related:` frontmatter is auto-populated by `scripts/regen_memory_links.py` — do not hand-type it.

### Step 5 — Update the topic index

For each insight:

- New topic folder → create `docs/memory/memories/<topic>/_topic_index.md` with the standard header (read an existing topic index for the format) and a single row.
- Existing folder → prepend a row (reverse-chronological) to the existing `_topic_index.md`. Match the existing row format.

### Step 6 — Update root index and tag dictionary

`docs/memory/ROOT_INDEX.md`:

- Bump `Total insights`.
- Update each touched folder's row: `Insights` count, `Last update` date, `Top tags` list.
- New folder → append a row with `folder | definition | count=1 | last_update | tags=[…]` (this is the **definition lock**, safeguard layer 3 — future-Claude matches against this string verbatim).
- Append one line under `## Change history`: `YYYY-MM-DD: Captured N insight(s) into <topic>(s).`

`docs/memory/memories/_global_tags.md`:

- Existing tag used → bump count; do not change first-use ID.
- New tag → append row with `tag | meaning | count=1 | first_use=<insight_id>`. English snake_case, singular, no slashes.

### Step 7 — Set raw_source to the synced JSONL (automatic)

The raw conversation is the JSONL Claude Code writes per session. The user's `sync-agent-data.sh` script mirrors `~/.claude/` → `claude_data/` on the NAS, so the JSONL is reachable from any node that has pulled. There is no separate `.md` archive — the JSONL itself is the canonical raw record.

For each insight written, set the frontmatter fields:

- `raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/<UUID>.jsonl` — `<UUID>` is the full session UUID from `$CLAUDE_CODE_SESSION_ID`.
- `raw_completeness: full` — the JSONL is verbatim.

In each insight's `## References` section, add one line:

> Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume <UUID>` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

After all insights are written, surface a single one-line reminder to the user:

> Raw-conversation links point at `claude_data/.../<UUID>.jsonl`. If you have not pushed recently, run `./sync-agent-data.sh claude push` so the link resolves on other nodes.

(No prompt asking the user to choose — the link is set automatically. The user's existing sync habit handles the rest.)

### Step 8 — Log each insight to the diary (mandatory)

After all insights are written and indexes updated, call the diary helper for **each** insight written. One call per insight; the script flock-protects concurrent invocations from parallel sessions.

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/diary_append.py insight \
  --subject "<copy from the insight's frontmatter `summary` field>" \
  --link   "docs/memory/memories/<topic>/<id>.md"
```

This is a hard step, not optional — the diary is the project's cross-session status board, and a memory capture without a diary entry creates an invisible gap. If a diary call errors (e.g., `docs/diary/` is missing), surface the error to the user but do not roll back the insight writes; the insights are the durable artefact.

### Step 9 — Auto-commit the capture (mandatory)

After all writes succeed, bundle every file this skill touched into a single commit. The user has standing auto-commit authorization — do not ask. One coherent commit per `/memorize` run, regardless of how many insights or diary rows were produced.

Before staging, run both regenerators in order:

1. **Link regenerator** — populates `related:` from body `[[id]]` tokens and normalises any hand-typed values:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/regen_memory_links.py
```

2. **Graph regenerator** — rebuilds `GRAPH_REPORT.md` and injects per-insight Backlinks blocks (idempotent):

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/regen_memory_graph.py
```

Then stage by name only — include any insight files either regenerator touched (each prints which files it updated):

```bash
git add docs/memory/ROOT_INDEX.md \
        docs/memory/memories/_global_tags.md \
        docs/memory/memories/<topic>/_topic_index.md \
        docs/memory/memories/<topic>/<id>.md \
        docs/diary/<YYYY-MM-DD>.md \
        docs/memory/GRAPH_REPORT.md
# If regen_memory_links.py updated older insights, add those paths too.
# If regen_memory_graph.py updated Backlinks blocks in older insights, add those paths too.
# repeat the topic-index and insight paths for every (topic, id) pair written

git commit -m "$(cat <<'EOF'
docs(memory): 📚 capture N insight(s) from <one-line session theme>

- <topic>/<slug>: <one-liner>
- <topic>/<slug>: <one-liner>
EOF
)"
```

Match the existing project style: conventional-commit type `docs`, scope `memory`, gitmoji `📚`, subject ≤ ~70 chars (check `git log --oneline -- docs/memory/` for tone). The body lists each insight as a bullet with topic/slug and the summary line — this is what makes the commit greppable months later.

Hard rules for the commit:

- **Stage by name only.** Never `git add -A` or `git add .` — other unrelated working-tree changes must not slip in.
- **Skip if any prior step errored.** A failed insight write or a failed diary call means the capture is incomplete; do not commit a partial state. Surface the error and let the user resolve.
- **No `--no-verify`, no secrets, no push.** Pre-commit hooks must run; do not push.
- **Auto-commit the diary rows here, not in Step 8.** Step 8 calls `diary_append.py` without committing. This step bundles the diary file alongside the insights so the whole capture lands as one atomic commit. (Standalone `/diary` calls outside `/memorize` commit themselves — see `.claude/skills/diary/SKILL.md`.)

### Step 10 — Confirm and report

Report in plain English:

> Saved N insight(s) to `docs/memory/` (and logged to today's diary):
>
> - `<topic>` — <one-liner>
> - `<topic>` — <one-liner>
>
> Indexes updated. Committed as `<short-hash>`. Raw conversation linked to `claude_data/.../<UUID>.jsonl`. If you have not pushed recently, run `./sync-agent-data.sh claude push`.

## Hard rules

- **Repo-relative paths only.** Never write to `~/.claude/...`, `/home/`, or any absolute path outside the current repo. The skill writes only inside the project tree under `docs/memory/` and `scripts/`. This matters in worktrees and sandboxes — absolute paths defeat isolation.
- **Never write to** `~/.claude/projects/.../memory/MEMORY.md` from this skill — that layer is harness-managed by Claude Code itself.
- All filenames English snake_case; all content English; no emojis unless the user explicitly asks.
- Conda Python: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. Never `python3`.
- "Delete" = move to `docs/memory/.trash/`. Hard-delete only on explicit user instruction.
- Fragmentation safeguards (4 layers) are non-optional; skipping any of them is a regression.
- If `docs/memory/CLAUDE.md` and this SKILL.md disagree, the operating manual wins — flag it and ask the user.

## References

- `docs/memory/CLAUDE.md` — operating manual (sections 4, 5, 6, 7, 11).
- `docs/memory/TEMPLATES/insight.md` — copy as starting point.
- `docs/memory/ROOT_INDEX.md` — topic registry.
- `docs/memory/memories/_global_tags.md` — tag dictionary.
- `scripts/claude_jsonl_to_md.py` — full-raw archive helper.
- `docs/develop/active/meta/claude_memory_system_design.md` — design rationale.
