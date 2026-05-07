# CLAUDE.md — `.claude-memory/` Operating Manual

> Authoritative single source for this memory layer's policies, workflows, and templates.
> The project root `CLAUDE.md` routes future-Claude here. Do not duplicate this content elsewhere.

**Last updated**: 2026-05-08
**Related**: [project CLAUDE.md](../CLAUDE.md), [ROOT_INDEX.md](ROOT_INDEX.md), [design plan](../docs/develop/active/meta/claude_memory_system_design.md)

---

## 1. Purpose and scope

This layer — `.claude-memory/` — is an **in-repo, version-controlled** session memory system. It stores multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. It coexists with the Claude Code built-in auto-memory at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` and does **not** replace it.

This layer is designed for:

- Multi-section insights with key conclusion, evidence, decisions, open questions, and references.
- Traceable links from a polished insight back to the raw conversation that produced it.
- Topic-level grouping with fragmentation safeguards.
- Natural-language recall: "what did we decide last week?"

---

## 2. Coexistence with built-in `MEMORY.md`

The two layers split by **insight density**. Use the decision rule below; when in doubt, ask yourself whether the thing being saved is a short typed rule or a session insight.

| Layer | Owner | Density | When to write here |
|---|---|---|---|
| Built-in `~/.claude/.../memory/MEMORY.md` | Claude Code harness + user | One-line rule + sibling `feedback_*.md` | Short typed rule another agent must obey on every invocation; user-and-machine local. |
| In-repo `.claude-memory/` | This repo + Claude Code | Per-session insight (frontmatter + 5 sections) | Decision with rationale, rejected alternatives, debugging arc, or finding the team needs to retrace later. |

**If both fit**: write to `.claude-memory/` and add a one-line entry in `MEMORY.md` only if another agent needs the rule on every invocation.

Concrete routing examples:

| Scenario | Goes to | Why |
|---|---|---|
| "Training-runner must `pgrep -af '<TAG>'` after every launch." | Built-in `MEMORY.md` | Short typed rule; no rationale chain. |
| "We chose FiLM over gate-only for three reasons." | `.claude-memory/memories/<topic>/<id>.md` | Decision with reasons; future-Claude must retrace. |
| "Don't use `python3` — use the conda env interpreter." | Project root `CLAUDE.md` | Project-wide invariant; not a session insight. |

---

## 3. New-session flow

1. Read this file (`CLAUDE.md`) — policies and templates.
2. On any memory work (capture, recall, audit): read `ROOT_INDEX.md` and `memories/_global_tags.md` (one-time per session).
3. Narrow to a topic: read `memories/<topic>/_topic_index.md`.
4. Read a specific insight: read `memories/<topic>/<id>.md`.
5. Raw archive (L4): read `_archive/raw_conversations/<id>.md` **only after explicit user confirmation** and after the large-file warning protocol (see section 10).

---

## 4. Capture triggers (3-tier)

| Tier | Trigger phrases (English, exact) | Behavior |
|---|---|---|
| Explicit | "remember this" / "save this" / "/memorize" / "memorize this" / "save to memory" | Extract insight(s) immediately and write. No further confirmation unless user is ambiguous about scope. |
| Session-end keyword | "wrap up" / "wrapping up" / "let's call it" / "good night" / "we're done" / "end of session" | Identify N candidate insights; list them with one-line summaries: `Save? Y / N / partial / edit`. In the same prompt, ask: `Also archive raw conversation (~XX KB)? Y / N`. |
| Inline suggestion | (Claude initiates) On a clear decision, finding, or rejected alternative, drop a single line: _"Worth saving to memory?"_ At most once per turn; do not interrupt flow. |

Anti-patterns to avoid:

- Do not auto-save without one of the three triggers.
- Do not re-prompt for the same insight twice in a session if the user said no.
- Do not dump the entire session into a single mega-insight — split by distinct decision/finding.

---

## 5. Insight file template

**Filename**: `YYYYMMDD_HHMM_<english_snake_case_slug>.md` — `HHMM` disambiguates same-day insights; slug is ≤ ~6 words.

Full template (14 frontmatter fields + 5 sections):

```markdown
---
id: YYYYMMDD_HHMM_<slug>
date: YYYY-MM-DD
time: HH:MM
folder: <topic>
tags: [<tag1>, <tag2>]
summary: "1-2 sentence plain-English summary."
related: []
session_origin: claude_code | claude_web
session_label: "<free-form session identifier>"
importance: high | medium | low
status: active | settled | superseded
supersedes: []
raw_source: _archive/raw_conversations/<id>.md   # or "none"
raw_completeness: full | approximate | none
---

# <one-line title>

## Key conclusion
1-3 sentences. The decision or finding in plain prose.

## Evidence, measurements, facts
- Bulleted: numbers, citations, file paths, run IDs.

## Decisions and actions
- What was decided. What action was taken or queued.

## Open questions and follow-ups
- Loose ends. Write "None" if there are none.

## References
- Related insight IDs, doc paths, run links, external URLs.
- If this insight created a new folder: include a one-line "Why a new folder" justification here.
```

Field semantics:

- `id` matches the filename stem and must be unique across the whole memory tree.
- `folder` must equal the folder the file lives under — required for self-description if the file is extracted.
- `status: settled` means the conclusion holds and is not under active revision; `active` means still being shaped; `superseded` mirrors the develop-doc lifecycle.
- `supersedes` and the symmetric `superseded_by` (added when superseding occurs) work like the develop-doc contract — the older insight keeps its file but flips `status` to `superseded`.
- `raw_completeness: full` when the archive was produced by `scripts/claude_jsonl_to_md.py`; `approximate` when Claude wrote a summary inline at session-end; `none` when no archive was kept.
- When `raw_source` points at a local-only archive: note in the `## References` section that this link is local-only (will be a broken link on a fresh clone).

---

## 6. Fragmentation safeguards (4-layer)

All four layers are mandatory — skipping any of them is a regression.

1. **Pre-classification**: before writing an insight, read `ROOT_INDEX.md`. Match against existing folder definitions. Strong match → use that folder. Medium match → ask the user. No match → proceed to layer 2.
2. **New-folder justification**: any new folder requires a one-line "Why a new folder" justification in the insight's `## References` section, naming the closest existing folder and explaining why the new insight does not fit there.
3. **Definition lock in `ROOT_INDEX.md`**: when creating a new folder, append a row with `folder | 1-line definition (≤ ~30 chars) | count=1 | last_update=YYYY-MM-DD | tags=[…]`. Future-Claude must match against this definition verbatim — if it needs to change, that is an audit-level event.
4. **Audit trigger**: when active-folder count ≥ 10 OR last audit older than 30 days, surface a brief audit prompt listing folders with low counts and similar tag overlap, suggesting merges. Log audit in `ROOT_INDEX.md` "Change history".

---

## 7. Raw-archive policy

- Archive lives at `.claude-memory/_archive/raw_conversations/<insight-id>.md`. Filename matches the primary insight ID. If one conversation produced multiple insights, name the archive after the chronologically first one; each insight's `raw_source` points at it.
- **Archives are local-only (gitignored).** The `.gitignore` ignores `_archive/raw_conversations/*.md` but tracks `.gitkeep`. The insight's `raw_source` link is meaningful only on the originating machine; cloners see a broken link by design. Note this in the insight's `## References` section.
- **Capture path (pick one)**:
  - Default (session-end keyword + user confirms): Claude writes a markdown summary of recent turns into the archive file. Mark `raw_completeness: approximate`.
  - On request ("save full raw"): user supplies the path to the Claude Code transcript JSONL under `~/.claude/projects/.../` and Claude runs `scripts/claude_jsonl_to_md.py <jsonl> <out_md>`. Mark `raw_completeness: full`.
- **Loading**: L4 only (see section 9), behind the large-file warning protocol (see section 10).
- When archive size exceeds 50 KB, surface a one-line reminder when the archive write trigger fires: `"Archive ~XX KB: check for sensitive content before committing the insight."` (Note: the archive itself is not committed — only the insight file is.)

---

## 8. Natural-language recall

### Mode selection

| Trigger style | Mode | Output |
|---|---|---|
| "what did we decide", "what did we work on", "remind me about", "last week", "yesterday", "memory of" | Natural-language recall | Time-grouped, reverse-chronological, plain English. |
| "show ROOT_INDEX", "list folders", "audit", "fragmentation", "lazy load", "L1 / L2 / L3", "tag dictionary" | Technical query | Raw indexes, folder names, counts, frontmatter fields. |

### Natural-language output format

Time headers (always show `Today` and `Yesterday` even if empty):

```
## Today
No new entries today.

## Yesterday
- YYYY-MM-DD HH:MM — <plain-English summary using the folder's 1-line definition, not the raw folder name>

## This week
- ...

## Earlier
- ...
```

Rules:
- Reverse-chronological at every level (newest at top within each group).
- Use the folder's 1-line definition from `ROOT_INDEX.md` as context, not the raw folder name.
- `Today` and `Yesterday` headers always appear; `This week` and `Earlier` only appear if non-empty.

**Do not expose** in natural-language mode: folder names verbatim, lazy-load level codes (L0–L4), `max_bytes`, `audit`, `fragmentation`, frontmatter field names, raw tag strings, or token costs. Size warnings (section 10) are an exception — they stay even in natural mode.

---

## 9. Lazy-load levels (L0–L4)

| Level | What | When |
|---|---|---|
| L0 | `.claude-memory/CLAUDE.md` (this file) | Whenever the user mentions memory/recall, or at the start of any session touching this layer. |
| L1 | `ROOT_INDEX.md` + `memories/_global_tags.md` | First read of any memory operation (capture, recall, audit). |
| L2 | `memories/<topic>/_topic_index.md` | When the user's question narrows to one topic. |
| L3 | `memories/<topic>/<id>.md` | When the user wants details on a specific insight, or when an L2 entry's one-line summary is insufficient. |
| L4 | `_archive/raw_conversations/<id>.md` | **Only after explicit user confirmation.** |

---

## 10. Large-file warning protocol

| Size | Action |
|---|---|
| < 50 KB | Read silently. |
| 50–300 KB | Announce in one line before reading: `Loading <path>, ~XX KB` |
| > 300 KB | Do not read until user confirms. Show the warning block below. |

Warning block for files > 300 KB:

```
Large file warning
  Target: <path>
  Size: ~XXX KB
  Context impact: ~X% of context
  Alternatives: summary, grep, partial range
  Proceed? Y / N / specific section
```

---

## 11. Soft-delete (`.trash/`) and filename rules

- **Soft-delete**: "delete" = move to `.trash/`. Never hard-delete without explicit user instruction. `.trash/*` is gitignored; `.trash/.gitkeep` is tracked.
- **Filename format**: `YYYYMMDD_HHMM_<english_snake_case_slug>.md`. English only, snake_case, no hyphens, no camelCase.
- **Folder naming**: English snake_case, ≤ ~3 words, no hyphens. 1-line definition ≤ ~30 characters in `ROOT_INDEX.md`.
- **Policy changes**: edit this file directly. Do not put policy in the built-in `MEMORY.md` — it is a pointer only.
