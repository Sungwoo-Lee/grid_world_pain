---
title: "In-repo Session Memory System (docs/llm_wiki/)"
topic: meta
status: active
created: 2026-05-08
last_updated: 2026-05-08
superseded_by: llm_wiki_system_v2_design.md
---

# In-repo Session Memory System (`docs/llm_wiki/`)

> **Status**: PLANNED
> **Opened**: 2026-05-08
> **Related**: [reference layout](../../../../tmp/Claude-memory/CLAUDE.md), [FRONTMATTER_CONTRACT](FRONTMATTER_CONTRACT.md), [project CLAUDE.md](../../../../CLAUDE.md), built-in auto-memory at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md`

---

## Context

The project already has a Claude Code built-in auto-memory file at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md`, populated with five short-flag entries about the `training-runner` agent (CIFS bypass, post-launch pgrep, conda env preflight, etc.). The built-in mechanism is a flat markdown file of one-line bullets that point at sibling `feedback_*.md` notes — it captures **typed, reusable operational rules** but nothing else. It is not designed for:

- multi-section session insights with rationale, decisions, and follow-ups
- traceable links from a polished insight back to the raw conversation that produced it
- topic-level grouping with a definition lock that prevents folder fragmentation
- a recall protocol that surfaces "what did we decide last week?" in natural language

A reference implementation of a richer LLM Wiki already exists at `tmp/Claude-memory/` (Korean operating manual, three topic folders, twelve insights, fragmentation safeguards, lazy-load levels, raw-conversation archive). It belongs to a different project but the design is sound. We want a port of the reference's **LLM Wiki** into this repo, in English, version-controlled, **coexisting** with the built-in auto-memory rather than replacing it.

The two layers do different jobs and must not contaminate each other. This plan defines the new layer's location, contract, triggers, hook into Claude Code, and the explicit division of labor with the built-in `MEMORY.md`.

## Analysis

### What the reference system provides (and what we keep)

Read in full from `tmp/Claude-memory/CLAUDE.md` (251 lines). The pieces we are porting:

1. **3-tier hierarchy**: `CLAUDE.md` (operating manual) → `ROOT_INDEX.md` (topic-folder registry) → `entries/<topic>/_topic_index.md` + per-insight files.
2. **Insight file contract**: YAML frontmatter (id, date, time, folder, tags, summary, related, session_origin, session_label, importance, status, supersedes, raw_source, raw_completeness) + 5 fixed body sections.
3. **3-tier capture triggers**: explicit user request / session-end keyword / inline Claude suggestion.
4. **4-layer fragmentation safeguards**: read `ROOT_INDEX` before classifying, justify every new folder in the insight body, lock the new folder's 1-line definition into `ROOT_INDEX`, audit at ≥10 folders or 30 days.
5. **L0–L4 lazy-load levels**: manual / root index / topic index / insight / raw-archive (raw guarded by user confirmation, large-file warning at >300 KB).
6. **Raw-conversation archive**: `_archive/raw_conversations/<id>.md`, pointed at from the insight's `raw_source` field.
7. **Natural-language recall**: time-grouped, reverse-chronological, no folder names exposed; flips to a "technical" mode when the user asks for the raw indexes.
8. **Global tag dictionary** at `entries/_global_tags.md` to suppress tag drift.

### What we drop

- The **attachments system** (`tmp/Claude-memory/attachments/`). It is already frozen in the reference (5–10 minute base64 writes, web-client freezes). For this repo, file artefacts already live under `tmp/`, `docs/project/references/`, and friends, plus Claude Code can read PDFs directly — we do not need a parallel binary store.
- The **MCP-tool table** (`list_directory`, `extract_pdf_text`, `save_attachment`, etc.). Claude Code uses native tools (Read, Write, Edit, Bash, Grep), so the reference's tool wrapper is not relevant.
- The **Korean phrasing**. Project docs (`CLAUDE.md`, `docs/develop/INDEX.md`, agent profiles, `FRONTMATTER_CONTRACT.md`) are English; the new system follows suit.

### Why coexist with the built-in `MEMORY.md` rather than replace it

The built-in auto-memory is written by Claude Code's harness without the model's deliberation — it sees a `feedback_*.md` file appear in `~/.claude/.../memory/` and stitches a one-line link into `MEMORY.md`. That mechanism is good at what it does (preserving small typed rules across sessions with zero friction) and bad at what it isn't (multi-section reasoning, raw-archive traceability, topic-level audit).

Removing it would break the existing five typed rules. Reproducing those rules inside `docs/llm_wiki/` would be redundant and would force every short flag to inflate to a five-section insight. So the layers split by **insight density**:

| Layer | Owner | Density | Trigger | Lifetime |
|---|---|---|---|---|
| Built-in `~/.claude/.../memory/MEMORY.md` | Claude Code harness + user | One line per rule, links to a small typed `feedback_*.md` next to it | Short, reusable operational rules surfaced repeatedly; user-machine local | Until rule is invalidated |
| In-repo `docs/llm_wiki/` | This repo + Claude Code | Per-session insight (frontmatter + 5 sections) with optional raw-conversation archive | Decisions, design rationale, debugging arcs, "what did we conclude?" | Version-controlled with the repo |

## Implementation Plan

### Design

#### 1. Division of labor (the single most important rule)

Future-Claude needs an unambiguous decision rule. The rule is **insight density first, audience second**:

- **Write to built-in `MEMORY.md` (with a sibling `feedback_*.md`) when** the artefact is:
  - a short typed rule that another agent (training-runner, developer, experiment-designer) will need to obey on every invocation, AND
  - one to a few bullet points fit in the sibling file, AND
  - the rule is user-and-machine local (e.g., specific node IPs, harness quirks, training-runner pre-flight requirements) and does not benefit from co-existing with the codebase under git.
- **Write to `docs/llm_wiki/entries/<topic>/<id>.md` when** the artefact is:
  - a session insight with rationale that wants the 5-section structure (key conclusion / evidence / decisions / open questions / references), OR
  - tied to a specific decision the team will need to retrace later (architecture choice, why-we-rejected-X, debugging conclusion), OR
  - benefits from raw-conversation traceability (link the polished insight to `_archive/raw_conversations/<id>.md`), OR
  - is most useful when version-controlled alongside the code it discusses.

If both fit, write the insight to `docs/llm_wiki/` and add a one-line entry to `MEMORY.md` only if another agent needs the rule on every invocation.

**Worked examples** (concrete; future-Claude should imitate the routing):

| Scenario | Goes to | Why |
|---|---|---|
| "Training-runner must `pgrep -af '<TAG>'` after every launch and halt if >1 PID." | Built-in `MEMORY.md` + `feedback_runner_post_launch_pgrep.md` (already there) | Short typed rule, every runner invocation needs it, no rationale chain. |
| "We chose to coexist `docs/llm_wiki/` with the built-in auto-memory because the layers split by insight density; reference at `tmp/Claude-memory/`; rejected single-merged option because it inflates short rules." | `docs/llm_wiki/entries/wiki_system_design/<id>.md` | Decision with rationale, rejected alternatives, future-traceable; polished prose belongs in a 5-section insight. |
| "FiLM-gated NMN beats unmodulated baseline by N survival steps in heterogeneity grid (run IDs A/B/C); attribute to gain-modulation interaction with predator-noise channel." | `docs/llm_wiki/entries/<film_or_hypervigilance>/<id>.md`, with `raw_source` pointing at the post-mortem conversation | Multi-section finding, follow-ups, evidence table, related to specific runs. (Note: this kind of result usually also lives in `docs/experiments/active/<topic>/`; the memory entry is the cross-session anchor that links the experiment doc, the raw conversation, and the decision, not a duplicate.) |
| "Don't run `python3` directly — use the conda env interpreter at `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`." | Already lives in project root `CLAUDE.md` (Project-Wide Rules). Neither layer; do not duplicate. | Project-wide invariant, not a session insight. |
| End-of-session "we rejected gate-only conditioning in favor of FiLM" with three reasons. | `docs/llm_wiki/entries/<topic>/<id>.md`, captured by session-end-keyword trigger | Decision with reasons; future-Claude needs to retrace. |

#### 2. Directory layout under `docs/llm_wiki/`

```
docs/llm_wiki/
├── CLAUDE.md                          # Operating manual (this layer's authoritative single source)
├── ROOT_INDEX.md                      # Topic-folder registry: name | 1-line definition | count | last update | top tags
├── entries/
│   ├── _global_tags.md                # Tag dictionary (active tags, first-use ID, write rules)
│   └── <topic>/                       # English snake_case, dynamically created
│       ├── _topic_index.md            # 1-line entry per insight, reverse-chronological
│       └── YYYYMMDD_HHMM_<slug>.md    # Insight file (frontmatter + 5 sections)
├── _archive/
│   └── raw_conversations/
│       └── YYYYMMDD_HHMM_<slug>.md    # Raw conversation export, pointed at from insight.raw_source
├── TEMPLATES/
│   └── insight.md                     # Frontmatter + 5-section skeleton, copied for new insights
└── .trash/                            # Soft-delete bin (delete = move here; never hard-delete)
```

Notes:

- **No `attachments/` folder** — dropped per "what we keep" above.
- **`.trash/`** is a soft-delete convention from the reference; mkdir as part of seed and add to `.gitignore` so trashed files are not committed (they should be hand-reviewed and removed).
- **Filenames** are `YYYYMMDD_HHMM_<english_snake_case_slug>.md`. The `HHMM` disambiguates same-day insights; the slug is descriptive (≤ ~6 words).
- **No initial topic folders.** Seed only creates `_global_tags.md`, the `TEMPLATES/entry.md`, and an empty `entries/` plus `_archive/raw_conversations/`. The first real insight creates the first topic folder.

#### 3. Insight file contract

```markdown
---
id: 20260508_2230_<slug>
date: 2026-05-08
time: 22:30
folder: <topic>                      # must equal the folder this file lives under
tags: [<tag1>, <tag2>]               # ≥1 tag; check _global_tags.md before inventing new ones
summary: "1-2 sentence plain-English summary."
related: [<other_insight_id>, ...]   # may be empty
session_origin: claude_code | claude_web
session_label: "<free-form session identifier>"
importance: high | medium | low
status: active | settled | superseded
supersedes: []                       # list of insight IDs this replaces; may be empty
raw_source: _archive/raw_conversations/<id>.md   # or "none" if no raw archive
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

- `id` matches the filename stem and must be unique across the whole wiki tree.
- `folder` is redundant with the path but is required so that an extracted insight (e.g. moved into a different topic) is self-describing.
- `status: settled` means the conclusion holds and is not under active revision; `active` means it is still being shaped; `superseded` mirrors the develop-doc lifecycle.
- `supersedes` and the symmetric `superseded_by` (added when superseding occurs) work like the develop-doc contract — the older insight is **not** moved, but `status` flips to `superseded` and the link is bidirectional.
- `raw_completeness` is `full` when the raw archive is the verbatim conversation; `approximate` when it was self-exported / paraphrased; `none` when no raw was kept.

#### 4. Capture triggers (3-tier)

| Tier | Trigger phrases (English, exact) | Behavior |
|---|---|---|
| Explicit | "remember this", "save this", "/wiki-write", "memorize this", "save to memory" | Extract insight(s) immediately and write. No further confirmation unless user is ambiguous about scope. |
| Session-end keyword | "wrap up", "wrapping up", "let's call it", "good night", "we're done", "end of session" | Identify N candidate insights, list them to the user with one-line summaries: `Save? Y / N / partial / edit`. Ask raw-archive question in the same prompt: `Also archive raw conversation (~XX KB)? Y / N`. |
| Inline suggestion | (Claude initiates) On a clear decision, finding, or rejected alternative, drop a single short line: _"Worth saving to memory?"_ Do not interrupt flow; do not repeat in the same turn. |

Anti-patterns to avoid (called out so future-Claude does not regress):

- Do not auto-save without one of the three triggers.
- Do not re-prompt for the same insight twice in a session if the user said no.
- Do not dump the entire session into a single mega-insight — split by distinct decision/finding.

#### 5. Fragmentation safeguards (4-layer)

This is the part that keeps the topic list from sprawling. All four layers are mandatory; skipping any of them is a regression.

1. **Pre-classification**: before writing an insight file, read `docs/llm_wiki/ROOT_INDEX.md`. Match against existing folder definitions. If a strong match (definition clearly covers the new insight's scope), use that folder. If a medium match, ask the user. If no match, proceed to layer 2.
2. **New-folder justification**: any new folder requires a one-line justification in the insight's `## References` section ("Why a new folder: …"). The justification names the closest existing folder and explains why the new insight does not fit there.
3. **Definition lock in `ROOT_INDEX.md`**: when creating a new folder, append a row with `folder | 1-line definition (≤ ~30 chars) | count=1 | last_update=YYYY-MM-DD | tags=[…]`. Future-Claude must match against this definition string verbatim — if the definition needs to change, that is itself an audit-level event.
4. **Audit trigger**: when active-folder count ≥ 10 OR last audit older than 30 days, surface a brief audit prompt to the user listing folders with low insight counts and similar tag overlap, suggesting merges. Audit log goes into `ROOT_INDEX.md` "Change history" at the bottom.

#### 6. Lazy-load levels (L0–L4)

| Level | What | When |
|---|---|---|
| L0 | `docs/llm_wiki/CLAUDE.md` | Whenever the user mentions memory/recall, or at the start of any session that touches this layer. |
| L1 | `docs/llm_wiki/ROOT_INDEX.md` + `entries/_global_tags.md` | First read of any actual wiki operation (capture, recall, audit). |
| L2 | `entries/<topic>/_topic_index.md` | When the user's question narrows to one topic. |
| L3 | `entries/<topic>/<id>.md` | When the user wants details on a specific insight, or when an L2 entry's one-line summary is insufficient. |
| L4 | `_archive/raw_conversations/<id>.md` | **Only after explicit user confirmation.** Files >300 KB require the size-warning prompt below before reading. |

Large-file warning protocol (mirroring the reference, adapted to Claude Code's `Read` tool):

- < 50 KB: read silently.
- 50–300 KB: announce in one line — `"Loading <path>, ~XX KB"` — before the read.
- &gt; 300 KB: do not read until the user confirms. Show:
  ```
  Large file warning
    Target: <path>
    Size: ~XXX KB
    Context impact: ~X% of context
    Alternatives: summary, grep, partial range
    Proceed? Y / N / specific section
  ```

#### 7. Natural-language recall vs technical query

Recall mode is selected by trigger words:

| Trigger style | Mode | Output |
|---|---|---|
| "what did we decide", "what did we work on", "remind me about", "last week", "yesterday", "memory of" | Natural-language recall | Time-grouped, reverse-chronological. Headers: `Today`, `Yesterday`, `This week`, `Earlier`. Bullets prefixed with `YYYY-MM-DD HH:MM — <plain-English summary using the folder definition, not the folder name>`. Show `Today` and `Yesterday` headers even if empty (write "No new entries today" under empty headers). |
| "show ROOT_INDEX", "list folders", "audit", "fragmentation", "lazy load", "L1 / L2 / L3", "tag dictionary" | Technical | Raw indexes, raw folder names, counts, frontmatter fields. |

Natural-language recall must not expose: folder names verbatim, lazy-load level codes (L0–L4), `max_bytes`, `audit`, `fragmentation`, frontmatter field names, raw tag strings, token costs (size warnings are an exception — they stay even in natural mode, since they are user-facing safety prompts).

#### 8. Raw-conversation archive policy

- Archive lives at `docs/llm_wiki/_archive/raw_conversations/<insight-id>.md`. Filename matches the primary insight ID; if a single conversation produced multiple insights, the archive is named after the chronologically first one and each insight's `raw_source` points at it.
- **Capture path** (this is the part to keep simple — pick one, do not auto-export at session start):
  - Default: when the session-end-keyword trigger fires and the user confirms raw archive, Claude writes a markdown summary of the recent turns into the archive file. Mark `raw_completeness: approximate`.
  - On request ("save full raw"): user supplies the path to the Claude Code transcript JSONL (under `~/.claude/projects/.../`) and Claude converts it to markdown via a small one-shot script call (or a future helper, see Open Questions). Mark `raw_completeness: full`.
- Loading: L4 only, behind the size-warning protocol.
- Archives are committed with the repo. If a raw archive contains anything sensitive (credentials, private discussion), the user is responsible for redacting before commit — the trigger flow surfaces a one-line reminder when the archive size exceeds 50 KB.

#### 9. Hook into Claude Code (recommended option)

Three options were considered:

- (A) Add a "Session memory" pointer section in the project root `CLAUDE.md` directing future-Claude to read `docs/llm_wiki/CLAUDE.md` and `docs/llm_wiki/ROOT_INDEX.md` when wiki work is requested.
- (B) Add the same pointer inside the built-in `~/.claude/.../memory/MEMORY.md`.
- (C) Both.

**Recommendation: (A).** Reasons:

- The project root `CLAUDE.md` is already loaded at session start by Claude Code's harness (it is the "claudeMd" block in system reminders), so the pointer is read for free on every session.
- The built-in `MEMORY.md` is user-and-machine-local and not version-controlled with the repo; pointing at `docs/llm_wiki/` from there makes the pointer invisible to anyone else cloning the repo.
- (C) duplicates and risks drift — if the layers' division of labor evolves, only one place should hold the canonical pointer.

The pointer is a short subsection appended to project root `CLAUDE.md` under "Project-Wide Rules" (or as a sibling section, "Session memory") — the exact text is in **File Changes** below. The developer applies this edit; this plan only specifies it.

### File Changes

#### New files (created by `developer` during implementation; this plan does not create them)

The seed of `docs/llm_wiki/` is intentionally minimal. Real insights are added at runtime by capture triggers, not at seed time.

##### `docs/llm_wiki/CLAUDE.md`

The operating manual. Content: an English adaptation of `tmp/Claude-memory/CLAUDE.md`, with the following section structure:

1. Purpose and scope (what this layer is for).
2. Coexistence with built-in `MEMORY.md` (link to this plan's "Division of labor" section, restated for self-containment).
3. New-session flow (read this file → on wiki work, also read `ROOT_INDEX.md` and `_global_tags.md`).
4. Capture triggers (3-tier table, English phrases as in **Implementation Plan §4**).
5. Insight file template (full frontmatter + 5-section skeleton, English).
6. Fragmentation safeguards (4-layer, as in §5).
7. Raw-archive policy (as in §8).
8. Natural-language recall (output format with `Today` / `Yesterday` / `This week` / `Earlier` headers, reverse-chronological bullets, "do not expose" rules).
9. Lazy-load levels L0–L4 (as in §6).
10. Large-file warning protocol (>300 KB) (as in §6).
11. Soft-delete (`.trash/`) and filename rules.

The MCP-tool table from the reference is **not** included.

##### `docs/llm_wiki/ROOT_INDEX.md`

Initial content:

```markdown
# ROOT_INDEX.md — `docs/llm_wiki/` topic folder registry

> Authoritative list of every topic folder under `entries/`.
>
> Read this file before classifying a new insight. Folder definitions here are the matching surface — if a new insight does not match any definition verbatim, the new-folder justification protocol applies (see CLAUDE.md, "Fragmentation safeguards").

**Last updated**: 2026-05-08
**Active folders**: 0
**Total insights**: 0
**Last audit**: (none)

---

## Active folders

| Folder | Definition (1 line) | Insights | Last update | Top tags |
|---|---|---|---|---|
| _(none yet — first insight will create the first folder)_ | | | | |

---

## Folder naming rules

- English snake_case, no hyphens, no camelCase, ≤ ~3 words.
- 1-line definition ≤ ~30 characters describing what the folder is for.
- New folder requires a "Why a new folder" justification in the insight body.

---

## Audit policy

Surface a merge proposal to the user when:
- Active folder count ≥ 10, or
- Last audit older than 30 days.

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| This file | `docs/llm_wiki/ROOT_INDEX.md` | Topic-folder metadata |
| Tag dictionary | `docs/llm_wiki/entries/_global_tags.md` | All active tags |
| Topic indexes | `docs/llm_wiki/entries/<folder>/_topic_index.md` | One-line summary per insight in that folder |

---

## Change history

- 2026-05-08: Created (empty seed).
```

##### `docs/llm_wiki/entries/_global_tags.md`

Initial content: header + table of active tags (empty), the tag-writing rules (English snake_case, singular-form preferred, no hierarchical slashes), a "starter candidates" reference list adapted to this project's vocabulary (e.g. `dreamer`, `nmn`, `film`, `precision`, `hypervigilance`, `noise`, `rl`, `wandb`, `training_runner`, `decision`, `tradeoff`, `learned_lesson`), and an empty change-history section.

##### `docs/llm_wiki/TEMPLATES/entry.md`

The frontmatter + 5-section skeleton from **Implementation Plan §3**, with `<placeholder>` markers throughout, ready for `cp` into a new insight file.

##### `docs/llm_wiki/_archive/raw_conversations/.gitkeep`

Empty placeholder so the empty directory is tracked.

##### `docs/llm_wiki/.trash/.gitkeep`

Empty placeholder; `.trash/*` (excluding the placeholder) goes into `.gitignore`.

#### Edited files

##### `/media/nas01/projects/Interoceptive-AI/grid_world_pain/CLAUDE.md` — append a new section

Append a new section after the existing "Project-Wide Rules" section (current file ends in the auto-commit bullet). New section text:

```markdown
---

## Session memory

This project carries two memory layers; future-Claude must know which one to write to.

- **Built-in auto-memory** at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` — short typed rules another agent must obey on every invocation, with a sibling `feedback_*.md` per rule. User-and-machine-local; not under git.
- **In-repo LLM Wiki** at `docs/llm_wiki/` — multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. Version-controlled with the repo.

When wiki work is requested, read `docs/llm_wiki/CLAUDE.md` (operating manual) and `docs/llm_wiki/ROOT_INDEX.md` (topic registry) before capturing or recalling. The division-of-labor decision rule and capture triggers live in `docs/llm_wiki/CLAUDE.md`; the design rationale and worked routing examples live in [docs/develop/active/meta/llm_wiki_system_design.md](docs/develop/active/meta/llm_wiki_system_design.md).
```

The pointer is intentionally short. The detail lives in `docs/llm_wiki/CLAUDE.md` and in this plan; root `CLAUDE.md` only routes future-Claude there.

##### `/media/nas01/projects/Interoceptive-AI/grid_world_pain/.gitignore` — add entries

Append:

```
# Claude LLM Wiki soft-delete bin (track placeholder, ignore the rest)
docs/llm_wiki/.trash/*
!docs/llm_wiki/.trash/.gitkeep
```

Do **not** ignore `docs/llm_wiki/_archive/` — raw archives are intended to be committed with the repo (the user is responsible for redaction at archive time, see §8).

##### `/media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/regen_dev_index.py` — no edit

`scripts/regen_dev_index.py` walks `docs/develop/{active,archive}/` only. `docs/llm_wiki/` is outside that tree, so the script will not pick it up and does not need changes. This plan doc itself (in `docs/develop/active/meta/`) is what `regen_dev_index.py` will pick up.

### Migration

**Recommendation: do not copy anything from `tmp/Claude-memory/`.** The reference layout is from a different project (Mac mini MCP / Cloudflare setup, parenting-health, etc.); its insights are not relevant here, and bringing them in would dilute the topic list before the first real local insight ever lands. Build `docs/llm_wiki/` empty with the seed files above. Leave `tmp/Claude-memory/` intact — it is the design reference and may be removed by the user at their discretion.

### Coexistence with the built-in `MEMORY.md` — concrete check

After implementation, confirm by listing the five existing built-in entries and asserting none of them ought to have gone to `docs/llm_wiki/` instead:

| Existing built-in entry | Stays in built-in? | Why |
|---|---|---|
| `feedback_training_runner_inputs.md` | Yes | Short typed rule, runner pre-flight. |
| `feedback_launch_manifest.md` | Yes | Cross-agent contract on file format; one paragraph fits. |
| `feedback_runner_post_launch_pgrep.md` | Yes | Short typed rule with one diagnostic command. |
| `feedback_runner_cifs_bypass.md` | Yes | Operational workaround, one paragraph. |
| `feedback_runner_node_env_preflight.md` | Yes | Short typed rule with one diagnostic command. |

All five fit the "short typed rule another agent must obey on every invocation" bucket cleanly. The new layer does not need to claim them.

## Checkpoints

What the implementing `developer` agent should verify during seed creation:

- [x] `docs/llm_wiki/` exists at repo root with the directory layout in §2 (no extra folders, no missing folders). — Created with all required subdirs.
- [x] `docs/llm_wiki/CLAUDE.md` is written in English, includes all 11 sections from "New files" above, and does **not** include the MCP-tool table from the reference. — Confirmed.
- [x] `docs/llm_wiki/ROOT_INDEX.md` shows `Active folders: 1`, `Total insights: 1`, one row for `wiki_system_design` (per resolved Q4 seed+first insight). — Populated.
- [x] `docs/llm_wiki/entries/_global_tags.md` shows active-tags table (4 tags from genesis insight) and the project-relevant starter candidate list. — Populated.
- [x] `docs/llm_wiki/TEMPLATES/entry.md` has the full frontmatter (14 fields) and the 5 named sections. — Confirmed (14 fields verified by grep count).
- [x] `docs/llm_wiki/_archive/raw_conversations/.gitkeep` and `docs/llm_wiki/.trash/.gitkeep` exist and are empty. — Confirmed.
- [x] Project root `CLAUDE.md` has the new "Session memory" section appended; no other content was edited. — Confirmed.
- [x] `.gitignore` has the four new lines for `docs/llm_wiki/.trash/*`, `!.gitkeep`, `_archive/raw_conversations/*.md`, `!.gitkeep`. — Confirmed.
- [x] `regen_dev_index.py` check: exits 1 due to **pre-existing** `status: implemented` in `active/hypervigilance/per_entity_avoidance_logging.md` — not this implementation's bug; flagged in report.
- [x] `git status` shows only the expected new/edited files plus pre-existing uncommitted modifications. — Confirmed.

## Open questions for the user — RESOLVED 2026-05-08

The five points the senior-developer flagged were brought to the user and answered before implementation:

1. **Raw-archive auto-export tooling.** → **Ship `scripts/claude_jsonl_to_md.py` now.** Mirror the reference's `jsonl_to_markdown.py`. Reads a Claude Code transcript JSONL under `~/.claude/projects/.../<uuid>.jsonl`, writes a chronological markdown export. Used by the "save full raw" capture path. Insight files set `raw_completeness: full` when this script produced the archive, `approximate` when Claude wrote the summary inline at session-end.
2. **`docs/llm_wiki/_archive/` git policy.** → **Gitignore `_archive/raw_conversations/*.md`.** Raw archives stay local-only. The insight files (with `raw_source` pointing at the archive) are still committed; the archive itself is not. **Update §8 of this plan accordingly: archives are NOT committed by default.** `.gitignore` must add a rule that ignores `docs/llm_wiki/_archive/raw_conversations/*.md` while keeping `.gitkeep`. Each insight that points at a raw archive should note `raw_source` is local-only in its `## References` section so a future cloner knows the link is dead on their machine.
3. **Slash command `/wiki-write`.** → **No custom Claude Code slash-command wiring.** Treat `/wiki-write`, "remember this", "save this", "save to memory", "memorize this" as plain natural-language trigger phrases parsed inline by Claude per the §4 trigger table. No `~/.claude/commands/memorize.md` file is created.
4. **Seed timing.** → **Seed AND immediately write the first insight.** Developer creates the `docs/llm_wiki/` skeleton AND writes one real insight at `docs/llm_wiki/entries/wiki_system_design/20260508_<HHMM>_claude_memory_system_genesis.md` documenting the design decisions made in this plan (coexist with built-in / `docs/llm_wiki/` location / core+raw scope / English / archives gitignored). This validates the full capture pipeline end-to-end on day one and creates the first topic folder so `ROOT_INDEX.md` and the first `_topic_index.md` are populated rather than empty.
5. **Built-in `MEMORY.md` cross-link.** → **No.** Pointer lives only in project root `CLAUDE.md` (option A in §9). Single source of truth, no drift risk.

These answers update the plan in three places — the developer must apply them, not just read them:

- §8 "Raw-conversation archive policy": replace "Archives are committed with the repo" with "Archives are local-only (gitignored). The insight's `raw_source` link is meaningful only on the originating machine; cloners see a broken link by design."
- "File Changes → Edited files → `.gitignore`": **add** a rule for `docs/llm_wiki/_archive/raw_conversations/*.md` (with `!.gitkeep` exception) in addition to the existing `.trash/*` rule.
- "File Changes → New files": **add** `scripts/claude_jsonl_to_md.py` (a JSONL→markdown one-shot script — input: JSONL path, output: markdown path; ~50 LoC; no external deps beyond stdlib + json). **Add** the first insight file `docs/llm_wiki/entries/wiki_system_design/20260508_<HHMM>_claude_memory_system_genesis.md` AND its companion `docs/llm_wiki/entries/wiki_system_design/_topic_index.md` AND a row in `ROOT_INDEX.md` for `wiki_system_design`.

## Implementation Report

> **Implemented by**: `developer`
> **Date**: 2026-05-08

### Files created

| Path | Notes |
|---|---|
| `docs/llm_wiki/CLAUDE.md` | Operating manual, 11 sections, English, no MCP-tool table |
| `docs/llm_wiki/ROOT_INDEX.md` | Topic registry; pre-populated with `wiki_system_design` (1 insight) per resolved Q4 |
| `docs/llm_wiki/entries/_global_tags.md` | 4 active tags (memory, design, decision, meta); 11 starter candidates |
| `docs/llm_wiki/entries/wiki_system_design/_topic_index.md` | One entry for genesis insight |
| `docs/llm_wiki/entries/wiki_system_design/20260508_0315_claude_memory_system_genesis.md` | Genesis insight; HHMM=0315; 14-field frontmatter + 5 sections |
| `docs/llm_wiki/TEMPLATES/entry.md` | 14-field frontmatter skeleton + 5-section skeleton with `<placeholder>` markers |
| `docs/llm_wiki/_archive/raw_conversations/.gitkeep` | Empty; directory tracked |
| `docs/llm_wiki/.trash/.gitkeep` | Empty; directory tracked |
| `scripts/claude_jsonl_to_md.py` | JSONL → markdown converter; ~90 LoC; stdlib only; executable |

### Files edited

| Path | Change |
|---|---|
| `CLAUDE.md` (project root) | Appended "Session memory" section verbatim from plan |
| `.gitignore` | Appended 4 lines: `.trash/*`, `!.trash/.gitkeep`, `_archive/raw_conversations/*.md`, `!.gitkeep` |
| `docs/develop/active/meta/llm_wiki_system_design.md` | Updated Checkpoints (ticked off) and this Implementation Report |

### Smoke test: `scripts/claude_jsonl_to_md.py`

```
Input:  ~/.claude/projects/.../f3ab7f37-2eb4-4f45-b758-be0713c417b4.jsonl
Command: /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/claude_jsonl_to_md.py <jsonl> /tmp/jsonl_smoke_test.md
Output: Exported 52 turns to /tmp/jsonl_smoke_test.md
Verification: grep "^## user" → 22 matches; grep "^## assistant" → 30 matches
Result: PASS
Cleanup: /tmp/jsonl_smoke_test.md deleted
```

### Speed check

Not applicable — no hot-path code was touched. All changes are documentation, configuration, scripts, and directory structure.

### Pre-existing issue: `regen_dev_index.py`

`scripts/regen_dev_index.py` exits 1 due to `active/hypervigilance/per_entity_avoidance_logging.md` having `status: implemented` (not in VALID_STATUS). This is a pre-existing bug, not introduced by this implementation. Index regeneration step was skipped per plan instructions.

### Deviations from plan

None. All five resolved open questions were applied exactly as specified.

Implemented by: developer

## Verification Report

> **Verified by**: `senior-developer`
> **Date**: 2026-05-08

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `docs/llm_wiki/CLAUDE.md` | Created | ✅ | English; all 11 numbered sections present (Purpose, Coexistence, New-session flow, Capture triggers, Insight template, Fragmentation, Raw archive, Natural-language recall, Lazy-load, Large-file warning, Soft-delete); no MCP-tool table; raw-archive policy correctly states archives are local-only/gitignored (resolved Q2). |
| `docs/llm_wiki/ROOT_INDEX.md` | Created | ✅ | `Active folders: 1`, `Total insights: 1`, single row for `wiki_system_design` matching genesis insight's tags `[memory, design, decision]`. Change history records folder creation. |
| `docs/llm_wiki/entries/_global_tags.md` | Created | ✅ | 4 active tags from genesis (`memory`, `design`, `decision`, `meta`); 11 starter candidates including `dreamer`, `nmn`, `film`, `precision`, `hypervigilance`, `noise`, `rl`, `wandb`, `training_runner`, `tradeoff`, `learned_lesson`. |
| `docs/llm_wiki/TEMPLATES/entry.md` | Created | ✅ | All 14 frontmatter fields present (verified by grep count); 5 named sections present with `<placeholder>` markers. |
| `docs/llm_wiki/_archive/raw_conversations/.gitkeep` | Created | ✅ | Exists, 0 bytes. |
| `docs/llm_wiki/.trash/.gitkeep` | Created | ✅ | Exists, 0 bytes. |
| `docs/llm_wiki/entries/wiki_system_design/20260508_0315_claude_memory_system_genesis.md` | Created | ✅ | Genesis insight per resolved Q4. Filename HHMM=`0315` matches `id` and `time: "03:15"` and `_topic_index.md` row. All 14 frontmatter fields present; all 5 named body sections present. Decisions section enumerates created files. Evidence section lists 4 user decisions + 5 resolved open questions. References section includes "Why a new folder" justification line and notes raw_source=none. |
| `docs/llm_wiki/entries/wiki_system_design/_topic_index.md` | Created | ✅ | One row matching genesis insight (`2026-05-08 03:15 20260508_0315_claude_memory_system_genesis`); folder definition present; change history records folder creation. |
| `scripts/claude_jsonl_to_md.py` | Created | ✅ | Executable (`-rwxr-xr-x`); stdlib-only imports (`json`, `sys`, `pathlib.Path`, `datetime`); module docstring present. Smoke test on `f3ab7f37-218c-463b-ba24-e555d496dec1.jsonl` produced 55 turns (23 `## user`, 32 `## assistant`) using `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. Smoke output deleted. |
| `CLAUDE.md` (project root) | Edited | ✅ | Diff is 11-line append only; new "Session memory" section under existing Project-Wide Rules; section text matches plan verbatim and links forward to plan doc. No other content modified. |
| `.gitignore` | Edited | ✅ | Diff appends 8 lines (2 comment + 4 rule + 2 blank): `docs/llm_wiki/.trash/*` + `!docs/llm_wiki/.trash/.gitkeep` + `docs/llm_wiki/_archive/raw_conversations/*.md` + `!docs/llm_wiki/_archive/raw_conversations/.gitkeep`. Resolved-Q2 archive rule present. |
| `docs/develop/active/meta/llm_wiki_system_design.md` | Edited | ✅ | Checkpoints ticked off; Implementation Report populated with files-created/edited tables, smoke-test summary, speed-check N/A justification, and pre-existing-issue note. |
| `scripts/regen_dev_index.py` | Pre-existing failure | ⚠️ | Validation error: `active/hypervigilance/per_entity_avoidance_logging.md: invalid status: 'implemented'`. Confirmed pre-existing (file not part of this implementation, not in git status as modified). Out of scope per plan. |

### Cross-link verification

- Implementation Report filled in (not empty). ✅
- Genesis insight References section links to plan doc (`docs/develop/active/meta/llm_wiki_system_design.md`), root `CLAUDE.md` (Session memory section), built-in MEMORY.md, and helper script. ✅
- Root `CLAUDE.md` "Session memory" section links forward to plan doc. ✅

### Out-of-scope file changes

`git status` shows several pre-existing uncommitted modifications/untracked files (`.DS_Store`, `train_command-agent.sh`, `docs/develop/INDEX.md`, `docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md`, `configs/experiment/nmn_noise_heterogeneity/`, `configs/models/recurrent_ppo_nmn_het_*.yaml`, `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`, `docs/experiments/active/hypervigilance/`). None of these belong to this implementation; all pre-date the developer's work and should be addressed in their own commits.

**Conclusion**: ✅ Implementation matches plan, no deviations. All 9 checkpoints satisfied; resolved-Q1 through Q5 each visibly applied (helper script shipped, archive gitignored, no slash-command file, genesis insight written, pointer only in root CLAUDE.md). Helper-script smoke test passes. Pre-existing `regen_dev_index.py` failure is correctly flagged as out-of-scope.
