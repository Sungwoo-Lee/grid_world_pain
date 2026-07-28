---
aliases: [llm_wiki_operating_manual]
---

# CLAUDE.md — `docs/llm_wiki/` Operating Manual

> Authoritative single source for this LLM Wiki's policies, workflows, and templates.
> The project root `CLAUDE.md` routes future-Claude here. Do not duplicate this content elsewhere.

**Last updated**: 2026-07-27  <!-- §3 consult gate + cost model added -->
**Related**: [project CLAUDE.md](../CLAUDE.md), [ROOT_INDEX.md](ROOT_INDEX.md), [design plan](../docs/develop/active/meta/llm_wiki_system_design.md)

---

## 1. Purpose and scope

This layer — `docs/llm_wiki/` — is an **in-repo, version-controlled** LLM Wiki. It stores multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. It coexists with the Claude Code built-in auto-memory at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` and does **not** replace it.

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
| In-repo `docs/llm_wiki/` | This repo + Claude Code | Per-session insight (frontmatter + 5 sections) | Decision with rationale, rejected alternatives, debugging arc, or finding the team needs to retrace later. |

**If both fit**: write to `docs/llm_wiki/` and add a one-line entry in `MEMORY.md` only if another agent needs the rule on every invocation.

Concrete routing examples:

| Scenario | Goes to | Why |
|---|---|---|
| "Training-runner must `pgrep -af '<TAG>'` after every launch." | Built-in `MEMORY.md` | Short typed rule; no rationale chain. |
| "We chose FiLM over gate-only for three reasons." | `docs/llm_wiki/entries/<topic>/<id>.md` | Decision with reasons; future-Claude must retrace. |
| "Don't use `python3` — use the conda env interpreter." | Project root `CLAUDE.md` | Project-wide invariant; not a session insight. |

---

## 3. When to consult, and how far to drill

### The pull gate

Nothing in this layer is auto-loaded. The root `CLAUDE.md` sets the trigger: **before any non-trivial task**, check the Active-folders table for a folder covering the area about to be touched — not just when wiki work is explicitly requested. That check is deliberately cheap, and the levels below are deliberately not.

| Read | Cost | When |
|---|---|---|
| `ROOT_INDEX.md` **Active folders table** (top ~30 lines) | ~2 KB | Every non-trivial task. This is the gate. |
| `ROOT_INDEX.md` **whole file** | ~52 KB | Audits only. The tail is a dated change log — never load it just to classify a topic. |
| `entries/<topic>/_topic_index.md` | 3–25 KB | Only when the gate matched that folder. |
| `entries/<topic>/<id>.md` | 3–10 KB | Only when a topic-index summary looks relevant to the task in hand. |

The gate answers *"does the wiki know anything about this area?"* — a yes/no that costs 2 KB. It does not answer *"what does it know?"*; that costs more and is paid only after a match. Reading entries speculatively defeats the lazy-load design and is the failure mode this table exists to prevent.

**On no match, proceed.** A miss is the expected outcome for most tasks and is not a reason to widen the search.

### New-session flow

1. Read this file (`CLAUDE.md`) — policies and templates.
2. On any wiki work (capture, recall, audit): read `ROOT_INDEX.md` and `entries/_global_tags.md` (one-time per session).
3. Narrow to a topic: read `entries/<topic>/_topic_index.md`.
4. Read a specific insight: read `entries/<topic>/<id>.md`.
5. Raw archive (L4): read `_archive/raw_conversations/<id>.md` **only after explicit user confirmation** and after the large-file warning protocol (see section 10).

---

## 4. Capture triggers (3-tier)

| Tier | Trigger phrases (English, exact) | Behavior |
|---|---|---|
| Explicit | "remember this" / "save this" / "/wiki-write" / "memorize this" / "save to memory" | Extract insight(s) immediately and write. No further confirmation unless user is ambiguous about scope. |
| Session-end keyword | "wrap up" / "wrapping up" / "let's call it" / "good night" / "we're done" / "end of session" | Identify N candidate insights; list them with one-line summaries: `Save? Y / N / partial / edit`. In the same prompt, ask: `Also archive raw conversation (~XX KB)? Y / N`. |
| Inline suggestion | (Claude initiates) On a clear decision, finding, or rejected alternative, drop a single line: _"Worth saving to the wiki?"_ At most once per turn; do not interrupt flow. |

Anti-patterns to avoid:

- Do not auto-save without one of the three triggers.
- Do not re-prompt for the same insight twice in a session if the user said no.
- Do not dump the entire session into a single mega-insight — split by distinct decision/finding.

---

## 5. Insight file template

**Filename**: `YYYYMMDD_HHMM_<english_snake_case_slug>.md` — `HHMM` disambiguates same-day insights; slug is ≤ ~6 words.

Full template (16 frontmatter fields + 5 sections):

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
valid_until: YYYY-MM-DD | null
confidence: high | medium | low | null
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

- `id` matches the filename stem and must be unique across the whole wiki tree.
- `date` is the capture date (`YYYY-MM-DD`); `time` is the capture time (`HH:MM`).
- `folder` must equal the folder the file lives under — required for self-description if the file is extracted.
- `tags` reuse from `_global_tags.md`; new tags require a row in that file.
- `summary` is the 1-2 sentence plain-English summary surfaced in recall listings.
- `related` is auto-populated from body `[[id]]` tokens by `scripts/claude/regen_wiki_links.py` — do not hand-type.
- `session_origin` is `claude_code` for Claude Code sessions; `claude_web` for Claude.ai web sessions.
- `session_label` is a free-form string identifying the session (e.g., a branch name or task theme).
- `importance` is `high`, `medium`, or `low` — how important this insight is for future recall.
- `status: settled` means the conclusion holds and is not under active revision; `active` means still being shaped; `superseded` mirrors the develop-doc lifecycle.
- `valid_until: YYYY-MM-DD | null` — the date on or after which this insight's claim should NOT be relied on without re-verification. `null` means "indefinite / no known invalidation date." Use this for time-sensitive findings (e.g., a verdict tied to a code version that's likely to be revisited). Distinct from `status: superseded` — a `valid_until` date passing does not automatically flip the status; it just signals "re-check before acting." A later capture can supersede the insight, in which case `superseded_by:` should also be set.
- `confidence: high | medium | low | null` — distinct from `status`. `status` is lifecycle (`active` / `settled` / `superseded`); `confidence` is "how strong is the evidence?" A `settled` + `high` insight is rock-solid; a `settled` + `low` insight is "we decided to act on this but the evidence base is thin." `null` means "not assessed at capture time" — acceptable but discouraged for new insights. Existing pre-E insights have `null` confidence by default.
- `supersedes` and the symmetric `superseded_by` (added when superseding occurs) work like the develop-doc contract — the older insight keeps its file but flips `status` to `superseded`.
- `raw_source` points at the JSONL path for the raw conversation (or `none` if no archive).
- `raw_completeness: full` when the archive was produced by `scripts/claude_jsonl_to_md.py`; `approximate` when Claude wrote a summary inline at session-end; `none` when no archive was kept.
- When `raw_source` points at a local-only archive: note in the `## References` section that this link is local-only (will be a broken link on a fresh clone).

**Optional fields added in v3** — all have defined defaults, so every pre-v3 entry stays valid:

- `headline: "<= 120 chars"` — the scan-level one-liner rendered into `_topic_index.md`. When absent, the generator truncates `summary` at its first sentence (140-char cap). Set it by hand when the truncation reads badly; there is no backfill obligation.
- `relations: ["<type>:<id>", ...]` — **auto-populated** from typed body wikilinks by `scripts/claude/regen_wiki_links.py`; do not hand-type. See §12.
- `use_count: <int>` / `last_used: <YYYY-MM-DD>` — reinforcement counters bumped by `scripts/claude/wiki_touch.py` when `/wiki-read` surfaces the entry at L3. **Read-count only, never decay** (a refuted hypothesis stays refuted; age is not evidence of irrelevance), and it counts *skill-mediated reads only* — an ad-hoc grep is invisible. Treat it as a lower bound on usefulness: fine for breaking ties in recall order or spotting folders nothing reads, never grounds for deleting an entry.

---

## 6. Fragmentation safeguards (4-layer)

All four layers are mandatory — skipping any of them is a regression.

1. **Pre-classification**: before writing an insight, read `ROOT_INDEX.md`. Match against existing folder definitions. Strong match → use that folder. Medium match → ask the user. No match → proceed to layer 2.
2. **New-folder justification**: any new folder requires a one-line "Why a new folder" justification in the insight's `## References` section, naming the closest existing folder and explaining why the new insight does not fit there.
3. **Definition lock in `ROOT_INDEX.md`**: when creating a new folder, append a row with `folder | 1-line definition (≤ ~30 chars) | count=1 | last_update=YYYY-MM-DD | tags=[…]`. Future-Claude must match against this definition verbatim — if it needs to change, that is an audit-level event.
4. **Audit trigger**: when active folder count ≥ 10 OR last audit > 30 days, run `scripts/claude/lint_wiki.py` to produce the full punch list (broken refs, orphans, tag-dictionary drift, near-duplicate folder definitions, `raw_source` resolution). Review the output with the user and log the audit in `ROOT_INDEX.md` "Change history".

---

## 7. Raw-archive policy

The raw conversation is the JSONL Claude Code writes per session — there is no separate `.md` archive maintained by `/wiki-write`. The JSONL is the canonical raw record. The user's `sync-agent-data.sh` script mirrors `~/.claude/` → `claude_data/` on the NAS, so the JSONL is reachable from any node that has pulled.

- **Where it lives**: `claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/<UUID>.jsonl` (synced from each node's `~/.claude/.../<UUID>.jsonl`). Repo-relative; `claude_data/` is `.gitignore`'d so the JSONL travels via the sync script, not git.
- **Insight pointer**: `/wiki-write` Step 7 sets every insight's `raw_source` to the JSONL path computed from `$CLAUDE_CODE_SESSION_ID`, with `raw_completeness: full`. No prompt — the link is set automatically.
- **Sync responsibility**: `/wiki-write` Step 7.5 (mandatory, automatic) runs `./sync-agent-data.sh claude push` after writing the insights and before the diary log + commit, so the JSONL is mirrored to NAS by the time the new insight's `raw_source` link goes live. Non-gating — if the push errors (NAS unreachable, rsync failure), the capture still completes and a one-line warning is surfaced at Step 10. The user no longer needs to remember to push manually.
- **Reading the raw conversation** (L4 — only on explicit user confirmation):
  - Best: `claude --resume <UUID>` re-enters the session in Claude Code (full UI, navigation, search).
  - Ad-hoc: `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` produces a one-shot markdown view. Apply the section 10 large-file warning protocol before reading the resulting file in context.
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

**Do not expose** in natural-language mode: folder names verbatim, lazy-load level codes (L0–L4), `max_bytes`, `audit`, `fragmentation`, frontmatter field names, raw tag strings, or token costs. Size warnings (section 10) are an exception — they stay even in natural mode. `valid_until` expiry warnings are also an exception: when an insight's `valid_until` date is in the past, append a one-line warning to its rendered output: `⚠️  This insight's valid_until passed on YYYY-MM-DD — re-verify before acting.`

---

## 9. Lazy-load levels (L0–L4)

Level codes for the drill-down described in §3; the *when to start at all* rule lives there.

| Level | What | When |
|---|---|---|
| L0 | `docs/llm_wiki/CLAUDE.md` (this file) | Whenever the user mentions the wiki / recall, or at the start of any session touching this layer. |
| L1a | `ROOT_INDEX.md` **Active-folders table only** (~2 KB) | The §3 pull gate — before any non-trivial task, wiki-related or not. |
| L1b | `ROOT_INDEX.md` full file + `entries/_global_tags.md` | An explicit wiki operation (capture, recall, audit). Not for topic classification alone. |
| L2 | `entries/<topic>/_topic_index.md` | When the user's question narrows to one topic. |
| L3 | `entries/<topic>/<id>.md` | When the user wants details on a specific insight, or when an L2 entry's one-line summary is insufficient. |
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

`related:` is **auto-populated** from body `[[id]]` tokens by `scripts/claude/regen_wiki_links.py`. Do not hand-type it. The canonical form is:

```yaml
related: ["id1", "id2"]    # sorted, double-quoted, single-line
related: []                # when no links
```

The regenerator is additive: it merges any `[[id]]` tokens it finds with whatever is already in `related:`, never removes existing IDs. Running it twice in a row is always idempotent.

### `scripts/claude/regen_wiki_links.py`

| Mode | Command | Effect |
|---|---|---|
| Default | `python scripts/claude/regen_wiki_links.py` | Walk all insights; add body `[[id]]` tokens to `related:` in canonical form. |
| One-time migration | `python scripts/claude/regen_wiki_links.py --normalise-existing` | Same algorithm; explicitly documented as the migration pass for pre-existing hand-typed `related:` values. |
| Pre-commit check | `python scripts/claude/regen_wiki_links.py --check` | Read-only; exits 0 if no files would change, 1 if any would. |
| Custom root | `python scripts/claude/regen_wiki_links.py --root <path>` | Override the default `docs/llm_wiki` root. |

The script is stdlib-only (no PyYAML); it parses frontmatter with regex and rewrites only the `related:` and `relations:` lines, preserving every other byte of the file.

### `/wiki-write` Step 9 contract

Step 9 (auto-commit) **must** run `scripts/claude/regen_wiki_links.py` before staging, so any `[[id]]` tokens written in the body during Step 4 are reflected in `related:` before the commit lands. See `.claude/skills/wiki-write/SKILL.md` Step 9 for the exact command.

### Typed links — `[[id|type]]`

A bare `[[id]]` says two entries are related but not *how*. In a research log the most valuable edge is "entry B refutes entry A", and that information was previously unrecordable. Use the alias slot to carry a relation type:

```
Round 3 overturned this under matched smells — see [[20260512_1428_sameprop_class_discriminating_defence_event_level|refutes]].
```

**Closed vocabulary.** An unrecognised type is treated as a typo, not a new relation: the regenerator warns and demotes it to `see_also`, so nobody silently invents a one-off edge kind that no query can find.

| Type | Use for |
|---|---|
| `refutes` | This entry's evidence overturns the linked claim |
| `supersedes` | Replaces the linked entry wholesale (pair with the `supersedes:` frontmatter field) |
| `extends` | Refines or builds on the linked entry without contradicting it |
| `caused` | The linked entry's decision or bug produced what this entry describes |
| `fixed` | This entry records the fix for a problem the linked entry recorded |
| `depends_on` | This conclusion is only valid while the linked one holds |
| `contradicts` | Tension surfaced but NOT resolved — deliberately weaker than `refutes` |
| `see_also` | Plain relatedness; the default meaning of an untyped `[[id]]` |

Both forms coexist. `related:` keeps holding **bare IDs** for every link, typed or not, so every existing consumer keeps working; `relations:` holds the typed form. The ~326 pre-v3 untyped edges remain valid and mean `see_also` — there is no backfill obligation. Type new links when the relation is genuinely one of the above; leave it untyped when it is just "related".

---

## 13. Contradiction handling at ingest

When `/wiki-write` writes a new insight, it MUST first check for likely contradictions against existing `status: settled` insights. Karpathy-style "flag, don't resolve" — surface possible conflicts to the user, do not silently merge.

### The check (Step 4.5 of `/wiki-write`)

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
   - **N**: abort the capture for this insight only (other candidates in the same `/wiki-write` batch can still proceed). Surface a one-line note.

5. **Non-interactive default.** In subagent / eval / batch runs (no human user), default to Y — flag in the Implementation Report or wherever appropriate, but do not block. False positives are tolerable.

### Tag-overlap scope rules

- "Shares ≥ 1 tag" is the default scope. If a candidate insight matches > 5 candidates, that's likely too noisy — increase to ≥ 2 tag overlap and re-scope. Print a one-line note if scope was tightened.
- Folders are NOT used to scope (cross-folder contradictions are real and the most important to catch).
- `status: active` insights are NOT in scope (they're still being shaped; conflict is expected).
- `status: superseded` insights are NOT in scope (already known to be replaced).

### Failure modes

- **False positive**: the user types N or Y to override. Cost: one extra prompt per `/wiki-write`. Acceptable.
- **False negative**: matches v1 baseline (no check at all). Acceptable; this is a "best effort" surface, not a guarantee.

### Why no separate script

The contradiction check is reasoning-shaped, not algorithm-shaped. A Python script would have to embed an LLM call to do the comparison, which the `/wiki-write` flow already has access to natively. Keeping the check inside the skill flow avoids an extra round-trip and keeps the contract auditable in one place.

---

## 14. Code-graph snapshots

`docs/llm_wiki/code_snapshots/` holds point-in-time records of `src/`'s structural state — derived from graphifyy's tree-sitter pass, captured at architecturally significant moments. Distinct from the *live* code graph at `src/graphify-out/` (gitignored, regenerable on demand).

**Capture**: `python scripts/snapshot_code_graph.py <label>` — see `docs/llm_wiki/code_snapshots/README.md`.

**When to snapshot**: major refactors, ship milestones, deprecation events, "this is what we threw away when we pivoted." Not every commit. The `/wiki-write` skill optionally surfaces a snapshot candidate when the session involves ≥ 5 changed files in `src/`.

**Lifetime**: immutable. Snapshots are dated records, not living documents. They never get edited, only superseded by newer snapshots.

**Recall**: `/wiki-read` surfaces snapshots when the user asks about past code state ("how did the model look at v1.4 ship"). Otherwise the live graph in `src/graphify-out/GRAPH_REPORT.md` is the right surface for "what does the code look like now."

---

## 15. Folder synthesis pages (`_state.md`) — the semantic tier

An entry records **what happened in one session**. Nothing in v1/v2 recorded **what we currently believe about a topic**. With 28 entries in `dreamer_diagnosis`, a reader asking "so what's wrong with Dreamer?" had to read all 28 and synthesize — every time, from scratch. That is the gap the 2026 literature calls the step from *Reflection* to *Experience* (cross-trajectory abstraction).

Each topic folder may carry a `_state.md`: the folder's current belief state, with every claim citing the entries behind it. Start from `docs/llm_wiki/TEMPLATES/state.md`.

**Rules that keep it honest:**

- **Rewritten in place, never appended.** It is a belief state, not a log. The log is the entries.
- **Every claim cites entries.** A claim in `_state.md` with no `[[id]]` behind it is an unsourced assertion and should be deleted or turned into an entry.
- **Contested things stay contested.** Where entries disagree, `_state.md` names both sides and says why it is unresolved. It must not silently pick a winner — an open disagreement is real information, and resolving one is what the §13 contradiction protocol is for.
- **Staleness is visible.** The frontmatter carries `synthesized_from` (entry count), `synthesized_on`, and `last_entry_included`, so a reader can see at a glance how far behind the page is.
- **Refresh trigger**: when a folder has gained ≥ 5 entries since `last_entry_included`, or on demand. Refreshing is a `/wiki-write`-adjacent act, not something to do casually mid-task.

**Read position**: L2, alongside `_topic_index.md` — and read it *first*. For the question "what do we know about X", `_state.md` is strictly cheaper and better than scanning the index and opening three entries. `_topic_index.md` links to it automatically when it exists.

## 16. Promotion — the procedural tier

A lesson that keeps recurring should stop being something to *re-read* and become something that is *followed*. The wiki is the episodic and semantic record; it is not where enforcement lives.

| Signal | Promote to | Why there |
|---|---|---|
| The same lesson recurs in ≥ 3 entries | a rule in the owning agent / skill doc | it has become procedure, not history |
| A one-line rule an agent must obey on **every** invocation | built-in auto-memory `MEMORY.md` | already the §2 split — machine-local typed rules |
| A repeatable multi-step procedure | a skill under `.claude/skills/` | skills are the executable tier |
| A project-wide invariant | root `CLAUDE.md` | already the §2 split |

**Promotion does not delete the entry.** The entry keeps the rationale — why the rule exists, what was tried, what was rejected — which is exactly what a one-line rule cannot carry. The promoted artifact carries the *instruction* and links back to the entry for the *reasoning*. Deleting the entry on promotion is how a project ends up with rules nobody can question because nobody remembers why they exist.

**The three tiers, named:** episodic (`entries/<topic>/<id>.md`) → semantic (`entries/<topic>/_state.md`) → procedural (skills, agent docs, `CLAUDE.md`, auto-memory). Each is a different lifetime and a different read cost; none replaces the one below it.
