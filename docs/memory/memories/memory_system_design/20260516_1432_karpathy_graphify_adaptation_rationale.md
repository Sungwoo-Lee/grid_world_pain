---
id: 20260516_1432_karpathy_graphify_adaptation_rationale
date: 2026-05-16
time: "14:32"
folder: memory_system_design
tags: [memory, design, decision, meta]
summary: "Memory v2 borrowed wikilinks + backlinks + lint + contradiction-flag from Karpathy's LLM Wiki and god-nodes + surprising-connections + edge-confidence from Graphify, then extended with bitemporal valid_until + confidence fields and a first-class conversation-records-as-graph-nodes layer. Explicitly skipped: deterministic regex wikilinks (auto-generated instead), full code-graph merge into memory (kept as separate gitignored layer). The adaptation rationale is what makes v2 distinct, and what future memory v3 needs to retrace before re-thinking."
related: ["20260516_1431_v2_three_role_architecture"]
session_origin: claude_code
session_label: "memory v2 build complete + graphify integration + bridge"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/a06843e3-ec5e-4850-9b54-75f95633989b.jsonl
raw_completeness: full
---

# Karpathy LLM Wiki + Graphify adaptation rationale (what borrowed, extended, skipped)

## Key conclusion

Memory v2 adopted four primitives directly from Karpathy's "LLM Wiki" pattern and four primitives conceptually from Graphify, then made two deliberate extensions and one deliberate non-adoption. The result is structurally similar to both reference designs but operates over markdown session insights rather than over a code corpus, and bridges the code corpus as a sibling layer rather than as the primary content. Knowing what was borrowed vs extended vs skipped — and why — is the meta-insight a future v3 designer needs to consult before deciding "let's just use Graphify directly" or "let's drop the contradiction flag because it false-positives too much."

## Evidence, measurements, facts

**Borrowed from Karpathy LLM Wiki:**

| Karpathy primitive | v2 equivalent | Status |
|---|---|---|
| `[[wikilink]]` deterministic syntax | `[[<insight_id>]]` parsed by `scripts/regen_memory_links.py` | ✅ adopted |
| `related:` frontmatter mirror auto-generated from body links | Same; idempotent --check mode | ✅ adopted |
| Per-page backlink index | `<!-- BACKLINKS -->` block injected by `scripts/regen_memory_graph.py` | ✅ adopted |
| Lint pass (broken refs, orphans, contradictions) | `scripts/lint_memory.py` 11 checks; CLAUDE.md §6 layer-4 audit trigger | ✅ adopted |
| "Flag, don't resolve" contradiction rule | CLAUDE.md §13 + memorize Step 4.5; uses existing `supersedes`/`superseded_by` mechanism on `S` choice | ✅ adopted |
| `raw/` immutable sources separate from `wiki/` curated | `claude_data/<UUID>.jsonl` immutable raw + `docs/memory/memories/` curated | ✅ adopted (predated v2) |

**Borrowed from Graphify (conceptually — custom Python, not Graphify):**

| Graphify primitive | v2 equivalent (in `docs/memory/GRAPH_REPORT.md`) | Status |
|---|---|---|
| God nodes (most-connected concepts) | `## God nodes (top-10 by inbound link count)` section | ✅ adopted |
| Surprising connections (cross-file ranked edges) | `## Surprising connections (cross-folder edges)` section | ✅ adopted |
| `GRAPH_REPORT.md` output format | `docs/memory/GRAPH_REPORT.md` six-section format | ✅ adopted (format only) |
| Confidence tags (EXTRACTED / INFERRED / AMBIGUOUS) | `confidence: high | medium | low` frontmatter field | ✅ adopted (coarser bucket) |

**Extended (v2 additions not in either reference):**

- **Bitemporal `valid_until`** — date on or after which the insight should not be relied on without re-verification. Distinct from `status: superseded`. `/recall` surfaces a past-date warning. No equivalent in Karpathy or Graphify.
- **Conversation records as first-class graph nodes** — session UUID extracted from `raw_source`, grouped into `## Conversation provenance` section in GRAPH_REPORT, plus `scripts/open_conversation.py` helper that prints restore commands. Not in either reference (Karpathy treats raw as opaque; Graphify is code-only).

**Deliberately not adopted:**

- **Full code-graph merge into the memory layer.** Considered; rejected. Live `src/graphify-out/` stays gitignored. Reasons: (a) different epistemic kinds — memory is curated decisions, code-graph is derivable structural facts; (b) auto-regen churn would pollute commit history; (c) stale-vs-current confusion. Bridged instead via `code_snapshots/` (point-in-time records, manually triggered) + `/recall` god-node cross-reference hint.

## Decisions and actions

- The two extensions are load-bearing: `valid_until` enables time-sensitivity for verdicts tied to code versions; conversation-as-node enables session-level recall (e.g., "show me all insights derived from session X").
- The non-adoption is also load-bearing: keeping code-graph as a sibling rather than primary content preserves memory's epistemic purity (decisions vs facts).
- Future v3 should consult this insight before reopening the "should we just commit `src/graphify-out/` into `docs/memory/`?" question. The answer was no for principled reasons, not for implementation reasons.

## Open questions and follow-ups

- Will the `valid_until` field actually get populated by future captures, or will most insights leave it null because authors don't know when to set a date? If null is the default, the field's value depends on whether `/recall` surfaces enough past-date warnings to make authors retroactively set valid_until.
- Should the contradiction-flag scope rule in §13 be tightened? Phase C eval showed 1,244 candidate pairs at ≥ 1 tag overlap (always exceeds the > 5 trigger), so ≥ 2 is the effective default. The cost per capture is ~30 conclusion comparisons in-session, which is fine — but if false-positive rate climbs as the corpus grows, ≥ 3 might be needed.

## References

- Karpathy LLM Wiki gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Graphify GitHub: https://github.com/safishamsi/graphify
- LLM Wiki v2 extension: https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2
- State of AI Agent Memory 2026 (mem0): https://mem0.ai/blog/state-of-ai-agent-memory-2026
- v2 design doc: [docs/develop/active/meta/claude_memory_system_v2_design.md](../../../../docs/develop/active/meta/claude_memory_system_v2_design.md) — Analysis section comparison table
- Companion insight: [[20260516_1431_v2_three_role_architecture]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume a06843e3-4` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1433_graphifyy_integration_cheatsheet]] (cluster_ops, 2026-05-16) — To use Graphify in any project conda env: install `graphifyy` (double-y; single-
<!-- END BACKLINKS -->
