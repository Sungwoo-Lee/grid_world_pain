---
title: "LLM Wiki v3 — closing the gaps the 2026 agent-memory literature identifies"
topic: meta
status: active
created: 2026-07-28
last_updated: 2026-07-28
---

# LLM Wiki v3 — closing the gaps the 2026 agent-memory literature identifies

## Context

This project keeps a version-controlled record of what past sessions decided and why, at
[`docs/llm_wiki/`](../../../llm_wiki/CLAUDE.md) — 193 entries across 10 topic folders. It is an
implementation of the "LLM wiki" pattern Andrej Karpathy published in April 2026 (an LLM-maintained,
interlinked set of markdown notes, governed by a schema file, sitting on top of immutable raw
sources). A web survey of the 2026 literature — three arXiv surveys plus the practitioner critiques
of Karpathy's original pattern — found this implementation already conforms to the canonical pattern
and independently reinvented one of its pieces (our daily diary is Karpathy's `log.md`).

The survey also found six real gaps. This plan closes them. In plain terms, the wiki today is a
**pile of well-written episodes with no summary and no lessons drawn** — it records what happened in
each session but never says "and therefore, here is what we now believe about Dreamer", and never
turns a lesson repeated across five entries into a rule anybody follows. Three of the six fixes are
policy (write a new kind of page, follow a new promotion rule); three are code.

Deliberately **not** adopted: hybrid vector/BM25/graph retrieval (the benchmark it comes from shows
orders-of-magnitude cost for no proportional gain at our scale), event-driven automation hooks
(a five-implementation comparison found unguarded automation produced "confident chaos"), and
time-based confidence decay (a refuted hypothesis stays refuted; decay is wrong for a research log).

## Problem

| # | Gap | Evidence | Consequence today |
|---|---|---|---|
| 1 | No cross-trajectory abstraction | *From Storage to Experience* survey: Storage → Reflection → **Experience** | 28 `dreamer_diagnosis` entries; nowhere states current belief |
| 2 | Single flat index fails past ~200 pages | LLM Wiki v2 critique | We are at **193** entries; topic-index rows average ~600 B vs ~120 B needed |
| 3 | No episodic → procedural pipeline | SKILL-DISCO; Agent Skills survey | Recurring lessons stay prose; nothing becomes a rule or skill |
| 4 | No lifecycle signal at scan time | Agent-native paper: systems without lifecycle "return stale facts" | `superseded` / expired `valid_until` invisible in the index |
| 5 | Untyped links | LLM Wiki v2 typed relations | `[[a]]` cannot express "refutes" — the most valuable edge in a research log |
| 6 | One memory tier | LLM Wiki v2 consolidation tiers | Episodic only; no semantic or procedural layer |

## Approach

Six changes, three of them code.

### A. Topic indexes become generated artifacts (gaps 2, 4)

New `scripts/claude/regen_wiki_indexes.py` rebuilds every `_topic_index.md` from entry frontmatter.

- Row summary = the `headline:` field when present, else the **first sentence** of `summary`, capped
  at 140 characters. No backfill needed: 193 existing entries keep working via truncation, and any
  entry can opt into a hand-written `headline:` later.
- Lifecycle markers are rendered inline: `[superseded]` and `[expired YYYY-MM-DD]` prefixes, so
  staleness is visible while scanning instead of only after opening the entry.
- Counts and `Last updated` are computed, not hand-maintained.

This also removes a whole class of bug we hit on 2026-07-28: two parallel sessions hand-editing the
same `_topic_index.md` left it with rows missing and a stale count. A generated index is race-proof —
whoever runs it last is correct.

**Contract change**: `/wiki-write` Step 5 stops prepending a row by hand and runs the regenerator.

### B. Typed wikilinks (gap 5)

Adopt the `[[id|type]]` form already parsed-and-logged by `regen_wiki_links.py`. Closed vocabulary,
chosen to match what this project actually records:

`refutes` · `supersedes` · `extends` · `caused` · `fixed` · `depends_on` · `contradicts` · `see_also`

- `related:` frontmatter keeps holding bare IDs (no format break, no backfill).
- A new `relations:` frontmatter line records the typed form, e.g. `relations: ["refutes:2026...id"]`.
- Existing 326 untyped edges stay valid and mean `see_also`. Typing is opt-in going forward.
- `GRAPH_REPORT.md` gains a "Typed edges" section so `refutes` chains are greppable.

### C. Folder synthesis pages — the semantic tier (gaps 1, 6)

Each topic folder gains `_state.md`: **what we currently believe about this topic**, with each claim
citing the entries backing it. Rewritten in place, never appended — it is a belief state, not a log.

- Trigger: written or refreshed when a folder gains ≥ 5 entries since its last refresh, or on demand.
- Read position: L2, alongside `_topic_index.md`. It is the cheapest useful read for "what do we know
  about X" and should be read *before* individual entries.
- Explicitly may contradict no entry: where entries disagree, `_state.md` says so and names both.

### D. Promotion rule — the procedural tier (gaps 3, 6)

A documented rule for when a lesson stops being an entry and becomes something enforced:

| Signal | Promote to | Why |
|---|---|---|
| Same lesson recurs in ≥ 3 entries | a rule in the owning agent/skill doc | it is procedure, not history |
| A one-line rule an agent must obey every invocation | built-in auto-memory `MEMORY.md` | already the documented split |
| A repeatable multi-step procedure | a skill under `.claude/skills/` | skills are the executable tier |
| A project-wide invariant | root `CLAUDE.md` | already the documented split |

The wiki entry is **not** deleted on promotion — it keeps the rationale; the promoted artifact carries
the instruction and links back.

### E. Use-count reinforcement, no decay (gap 4, half)

Add `use_count:` and `last_used:` frontmatter. `/wiki-read` increments them when it surfaces an entry
at L3 (full read), via `scripts/claude/wiki_touch.py`.

**Honest coverage limit**: this counts skill-mediated reads only. An ad-hoc `grep` or a direct file
read is invisible. The number is therefore a *lower bound* on usefulness and must never be used to
delete entries — only to rank ties and to spot never-read folders. Documented as such.

## File changes

| File | Change |
|---|---|
| `scripts/claude/regen_wiki_indexes.py` | **new** — generate all `_topic_index.md` |
| `scripts/claude/wiki_touch.py` | **new** — increment `use_count` / set `last_used` |
| `scripts/claude/regen_wiki_links.py` | typed-link parsing; emit `relations:` |
| `scripts/claude/regen_wiki_graph.py` | typed-edge section in `GRAPH_REPORT.md` |
| `docs/llm_wiki/CLAUDE.md` | §5 frontmatter (`headline`, `relations`, `use_count`, `last_used`); §12 typed links; new §15 `_state.md`; new §16 promotion rule |
| `docs/llm_wiki/TEMPLATES/state.md` | **new** |
| `.claude/skills/wiki-write/SKILL.md` | Step 5 → run regenerator; Step 9 adds index regen |
| `.claude/skills/wiki-read/SKILL.md` | read `_state.md` at L2; call `wiki_touch.py` at L3 |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | two new scripts + changed callers (Maintenance Contract) |

New frontmatter keys are all **optional with defined defaults** — absent `headline` falls back to
`summary`, absent `relations` means untyped, absent `use_count` means 0. No existing entry breaks,
so this does not violate the no-fallback-defaults rule (that rule governs training configs, where a
missing key must fail loudly; here the schema is explicitly versioned and additive).

## Verification

| Step | Check | Failure looks like |
|---|---|---|
| A | `regen_wiki_indexes.py` then `git diff` — every folder's row count equals its file count | a folder's count differs from `ls | wc -l` |
| A | Re-run twice; second run reports no changes | non-idempotent generator |
| B | Add one `[[id|refutes]]` link; `relations:` shows `refutes:id`; `related:` still holds the bare id | typed link lost, or `related:` corrupted |
| B | `regen_wiki_links.py --check` exits 0 after a full run | non-idempotent |
| C | `_state.md` for one folder cites only IDs that exist | broken citation |
| E | `wiki_touch.py` on one entry bumps `use_count` 0→1 and sets `last_used` | frontmatter mangled |
| all | `lint_wiki.py` error count does not increase from its pre-change baseline (2 errors, 28 warnings) | new breakage |

Baseline recorded before any change: **2 errors, 28 warnings** — 1 pre-existing broken wikilink
(counted twice) and 28 orphan warnings.

## Risks

- **Generated indexes overwrite hand-written summaries.** Existing `_topic_index.md` rows were
  hand-written and some are better than a truncated `summary`. Mitigation: the generator prefers an
  explicit `headline:`; anything worth keeping can be promoted into that field. Accepted loss:
  first-run rows get shorter and blunter, which is the point of gap 2.
- **`_state.md` drifts from its entries.** A stale belief page is worse than none. Mitigation: it
  carries `synthesized_from` (entry count + date) so staleness is visible, and refresh is part of the
  ≥5-entry trigger.
- **Two parallel sessions running the index generator** produce the same output, so a race is benign —
  unlike today's hand-editing.

## Links

- Survey and decision that produced this plan: [[20260728_1645_wiki_search_subagent_rejected_on_economics]] (why no search agent), [[20260728_1644_wiki_pull_gate_widened_before_any_task]] (the consult gate this builds on).
- Operating manual: [`docs/llm_wiki/CLAUDE.md`](../../../llm_wiki/CLAUDE.md).
