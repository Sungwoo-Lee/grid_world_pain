# CLAUDE.md — `docs/memory/` Operating Manual

> Authoritative single source for this memory layer's policies, workflows, and templates.
> The project root `CLAUDE.md` routes future-Claude here. Do not duplicate this content elsewhere.

**Last updated**: 2026-05-16
**Related**: [project CLAUDE.md](../CLAUDE.md), [ROOT_INDEX.md](ROOT_INDEX.md), [design plan](../docs/develop/active/meta/claude_memory_system_design.md)

---

## 1. Purpose and scope

This layer — `docs/memory/` — is an **in-repo, version-controlled** session memory system. It stores multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. It coexists with the Claude Code built-in auto-memory at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` and does **not** replace it.

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
| In-repo `docs/memory/` | This repo + Claude Code | Per-session insight (frontmatter + 5 sections) | Decision with rationale, rejected alternatives, debugging arc, or finding the team needs to retrace later. |

**If both fit**: write to `docs/memory/` and add a one-line entry in `MEMORY.md` only if another agent needs the rule on every invocation.

Concrete routing examples:

| Scenario | Goes to | Why |
|---|---|---|
| "Training-runner must `pgrep -af '<TAG>'` after every launch." | Built-in `MEMORY.md` | Short typed rule; no rationale chain. |
| "We chose FiLM over gate-only for three reasons." | `docs/memory/memories/<topic>/<id>.md` | Decision with reasons; future-Claude must retrace. |
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
4. **Audit trigger**: when active folder count ≥ 10 OR last audit > 30 days, run `scripts/lint_memory.py` to produce the full punch list (broken refs, orphans, tag-dictionary drift, near-duplicate folder definitions, `raw_source` resolution). Review the output with the user and log the audit in `ROOT_INDEX.md` "Change history".

---

## 7. Raw-archive policy

The raw conversation is the JSONL Claude Code writes per session — there is no separate `.md` archive maintained by `/memorize`. The JSONL is the canonical raw record. The user's `sync-agent-data.sh` script mirrors `~/.claude/` → `claude_data/` on the NAS, so the JSONL is reachable from any node that has pulled.

- **Where it lives**: `claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/<UUID>.jsonl` (synced from each node's `~/.claude/.../<UUID>.jsonl`). Repo-relative; `claude_data/` is `.gitignore`'d so the JSONL travels via the sync script, not git.
- **Insight pointer**: `/memorize` Step 7 sets every insight's `raw_source` to the JSONL path computed from `$CLAUDE_CODE_SESSION_ID`, with `raw_completeness: full`. No prompt — the link is set automatically.
- **Sync responsibility**: `/memorize` does not auto-push. It surfaces a one-line reminder at the end of capture: "If you have not pushed recently, run `./sync-agent-data.sh claude push`." The user runs the sync.
- **Reading the raw conversation** (L4 — only on explicit user confirmation):
  - Best: `claude --resume <UUID>` re-enters the session in Claude Code (full UI, navigation, search).
  - Ad-hoc: `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` produces a one-shot markdown view. Apply the section 10 large-file warning protocol before reading the resulting file in context.
- **Cross-node**: on a fresh node, `./sync-agent-data.sh claude pull` brings down all JSONLs; `raw_source` links resolve afterwards.

The legacy `_archive/raw_conversations/` directory is no longer used. New insights point at the synced JSONL directly. Existing insights with `_archive/raw_conversations/...md` raw_source values were backfilled to point at the JSONL when this design changed.

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
| L0 | `docs/memory/CLAUDE.md` (this file) | Whenever the user mentions memory/recall, or at the start of any session touching this layer. |
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

---

## 12. Wikilinks and graph regeneration

### Writing wikilinks

The body of an insight MAY contain `[[<id>]]` inline tokens that link to other insights. `<id>` is the filename stem: `YYYYMMDD_HHMM_<slug>`. Example:

```
See also [[20260508_0315_claude_memory_system_genesis]] for the genesis rationale.
```

Place wikilinks wherever they are naturally meaningful — in `## Decisions and actions`, `## References`, or inline in prose. Prefer `[[id]]` over hand-typing the ID into `related:`.

Obsidian renders `[[<id>]]` as a clickable link in the `docs/.obsidian` vault; the regenerator script keeps the `related:` frontmatter in sync with those tokens.

### `related:` frontmatter

`related:` is **auto-populated** from body `[[id]]` tokens by `scripts/regen_memory_links.py`. Do not hand-type it. The canonical form is:

```yaml
related: ["id1", "id2"]    # sorted, double-quoted, single-line
related: []                # when no links
```

The regenerator is additive: it merges any `[[id]]` tokens it finds with whatever is already in `related:`, never removes existing IDs. Running it twice in a row is always idempotent.

### `scripts/regen_memory_links.py`

| Mode | Command | Effect |
|---|---|---|
| Default | `python scripts/regen_memory_links.py` | Walk all insights; add body `[[id]]` tokens to `related:` in canonical form. |
| One-time migration | `python scripts/regen_memory_links.py --normalise-existing` | Same algorithm; explicitly documented as the migration pass for pre-existing hand-typed `related:` values. |
| Pre-commit check | `python scripts/regen_memory_links.py --check` | Read-only; exits 0 if no files would change, 1 if any would. |
| Custom root | `python scripts/regen_memory_links.py --root <path>` | Override the default `docs/memory` root. |

The script is stdlib-only (no PyYAML); it parses frontmatter with regex and rewrites only the `related:` line, preserving every other byte of the file.

### `/memorize` Step 9 contract

Step 9 (auto-commit) **must** run `scripts/regen_memory_links.py` before staging, so any `[[id]]` tokens written in the body during Step 4 are reflected in `related:` before the commit lands. See `.claude/skills/memorize/SKILL.md` Step 9 for the exact command.

### `[[id|alias]]` form

If you encounter `[[id|alias]]` (Obsidian alias syntax), the regenerator captures `id` and logs the alias. Alias support is not used in this project yet; use plain `[[id]]` only.

---

## 13. Contradiction handling at ingest

When `/memorize` writes a new insight, it MUST first check for likely contradictions against existing `status: settled` insights. Karpathy-style "flag, don't resolve" — surface possible conflicts to the user, do not silently merge.

### The check (Step 4.5 of `/memorize`)

1. **Scope the candidate set.** Read every `status: settled` insight in folders that share ≥ 1 tag with the new insight's `tags:`. Skip if the candidate set is empty (no overlapping settled insights).
2. **Compare conclusions.** For each candidate, compare the new insight's `## Key conclusion` paragraph against the candidate's `## Key conclusion`. Use the same conversational context (no separate API call). Return a single best-match decision: `{"contradicts": <id|null>, "rationale": "<one sentence>"}`.
3. **Surface or proceed.**
   - If `contradicts: null` → proceed to Step 5 (topic-folder routing).
   - If non-null → halt the write and ask the user:

     > Possible contradiction with `<prior_id>` (<prior_summary>).
     > Prior says: "<prior_key_conclusion_excerpt>"
     > New says: "<new_key_conclusion_excerpt>"
     > Rationale: <one-sentence reason from the check>
     > Choose: Y = save as a new insight (both stay live) / S = supersede the prior / N = skip this capture entirely

4. **Resolve.**
   - **Y**: write the new insight normally; both remain `status: settled` (or whatever the user picked); user has accepted the divergence consciously.
   - **S**: write the new insight with `supersedes: ["<prior_id>"]`; flip the prior insight to `status: superseded` with `superseded_by: ["<new_id>"]` added to its frontmatter. This re-uses the existing supersession mechanism.
   - **N**: abort the capture for this insight only (other candidates in the same `/memorize` batch can still proceed). Surface a one-line note.

5. **Non-interactive default.** In subagent / eval / batch runs (no human user), default to Y — flag in the Implementation Report or wherever appropriate, but do not block. False positives are tolerable.

### Tag-overlap scope rules

- "Shares ≥ 1 tag" is the default scope. If a candidate insight matches > 5 candidates, that's likely too noisy — increase to ≥ 2 tag overlap and re-scope. Print a one-line note if scope was tightened.
- Folders are NOT used to scope (cross-folder contradictions are real and the most important to catch).
- `status: active` insights are NOT in scope (they're still being shaped; conflict is expected).
- `status: superseded` insights are NOT in scope (already known to be replaced).

### Failure modes

- **False positive**: the user types N or Y to override. Cost: one extra prompt per `/memorize`. Acceptable.
- **False negative**: matches v1 baseline (no check at all). Acceptable; this is a "best effort" surface, not a guarantee.

### Why no separate script

The contradiction check is reasoning-shaped, not algorithm-shaped. A Python script would have to embed an LLM call to do the comparison, which the `/memorize` flow already has access to natively. Keeping the check inside the skill flow avoids an extra round-trip and keeps the contract auditable in one place.
