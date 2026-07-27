---
id: 20260516_1431_v2_three_role_architecture
date: 2026-05-16
time: "14:31"
folder: wiki_system_design
tags: [wiki, design, decision, meta]
summary: "Memory System v2 separates code/memory knowledge into three surfaces — curated session insights (docs/llm_wiki/entries/), live regenerable code-graph (src/graphify-out/, gitignored), dated immutable code-snapshots (docs/llm_wiki/code_snapshots/) — bridged by /wiki-write's optional snapshot prompt at Step 2 and /wiki-read's god-node cross-reference hint. Different epistemic kinds get different homes; agents route to the right surface for the question type."
related: ["20260508_0429_memorize_skill_design_and_ship", "20260509_1619_summarize_study_skill_design_and_ship", "20260509_1620_documentation_framing_policy", "20260513_2310_orphan_memory_branch_rewrite"]
session_origin: claude_code
session_label: "memory v2 build complete + graphify integration + bridge"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/a06843e3-ec5e-4850-9b54-75f95633989b.jsonl
raw_completeness: full
---

# Memory System v2 — three-role architecture + memory/code-graph bridge

## Key conclusion

Memory v2 separates knowledge into three surfaces with distinct lifetimes and update mechanics: (1) **curated session insights** at `docs/llm_wiki/entries/<topic>/<id>.md` — version-controlled, written by `/wiki-write`, never auto-regenerated, the "why did we decide" surface; (2) **live regenerable code-graph** at `src/graphify-out/` — gitignored, refreshed on demand via `python scripts/regen_code_graph.py`, the "what does the code look like RIGHT NOW" surface; (3) **dated immutable code-snapshots** at `docs/llm_wiki/code_snapshots/<id>_<label>.md` — committed, written by `scripts/snapshot_code_graph.py <label>`, the "what did the code look like AT past moment X" surface. The bridge between surfaces is two-way: `/wiki-write` Step 2 offers an opt-in snapshot candidate when the session touches ≥ 5 files in `src/`, and `/wiki-read` appends a `graphify explain <symbol>` hint when a recalled insight body mentions a god-node from the live code graph.

## Evidence, measurements, facts

- v2 build landed in 36 commits (Phases 0/A/B/C/D/E/F + bridge) on top of `v1.4` at `07887e4`. Tip at the close of this session: `4d95411` (latest merge commit).
- 56 insights across 6 topic folders. 22 inter-insight wikilink edges (after Phase A normalisation). 13 conversation-provenance sessions in `docs/llm_wiki/GRAPH_REPORT.md`.
- Live code graph: 735 nodes, 1009 edges, 59 communities (`src/graphify-out/GRAPH_REPORT.md`, 22 KB, regen ~2 s tree-sitter only).
- Dated snapshots: 1 captured so far — `docs/llm_wiki/code_snapshots/20260516_1322_v2_memory_build_complete.md` at commit `0dab341`.
- 11/11 lint checks pass on the corpus (`scripts/lint_wiki.py`).
- Auto-sync wired into `/wiki-write` Step 7.5: `./sync-agent-data.sh claude push` runs after raw_source is set, before diary + commit, so the JSONL the `raw_source` field points at is mirrored to NAS before the link goes live.
- The bridge proved real on existing data: at least two settled insights mention `RSSM` (god-node #10) and `DreamerNeuromodulatorRNN` (god-node #3) in their bodies — `/wiki-read` will fire the cross-reference hint on these.
- Frontmatter went from 14 fields to 16 (added `valid_until`, `confidence`). Backward-compatible — existing 56 insights have absent (= null) values.

## Decisions and actions

- **Adopted the three-role split** — committed knowledge (curated insights) ≠ derivable knowledge (code graph) ≠ snapshot of derivable knowledge at a moment (code snapshot). Each has a different home, different lifetime, different update trigger.
- **Bridge by routing, not by merging** — `/wiki-read` knows all three surfaces and surfaces the right one for the question type. `/wiki-write` Step 2 offers a snapshot when structurally significant. The wikilink namespaces stay separate (graphify uses `[[main()]]` / `[[_COMMUNITY_*]]`; memory uses `[[YYYYMMDD_HHMM_slug]]`) — no Obsidian namespace collision.
- **No auto-snapshot on every `/wiki-write`** — snapshots are manually triggered when something architecturally worth remembering happens (refactor, ship milestone, pivot). The opt-in candidate is offered only when the heuristic fires.
- **The conversation-record link is now a first-class graph node**, not just a buried frontmatter field. `GRAPH_REPORT.md` Conversation provenance section groups insights by session UUID + lists restore commands. `scripts/open_conversation.py <id>` prints `claude --resume <UUID>` + JSONL→MD command without executing.
- The full architecture lives in [v2 design doc](../../../../docs/develop/active/meta/llm_wiki_system_v2_design.md). This insight is the one-page summary; consult the design doc for per-phase Implementation + Verification reports.

## Open questions and follow-ups

- How often will users actually trigger snapshots in practice? The heuristic-driven `/wiki-write` Step 2 prompt is opt-in; if nobody ever picks it, the snapshot surface stays empty and the bridge is half-built.
- Will the god-node cross-reference hint fire on enough insights to be useful? Today 2 of 56 insights mention god-nodes. As the corpus grows alongside the code, this should increase, but the ratio is the right metric to watch.
- Should `valid_until` + `confidence` be backfilled on existing insights? Decided no during Phase E (backfill would be guessing). But if recall surfaces enough `null`-confidence insights without context, a one-off backfill pass may become worthwhile.

## References

- v2 design doc: [docs/develop/active/meta/llm_wiki_system_v2_design.md](../../../../docs/develop/active/meta/llm_wiki_system_v2_design.md) — full phase-by-phase plan + reports
- Operating manual: [docs/llm_wiki/CLAUDE.md](../../../CLAUDE.md) — §13 contradiction handling, §14 code-graph snapshots
- Bridge skill wiring: [memorize SKILL.md](../../../../.claude/skills/wiki-write/SKILL.md) Step 2 + Step 7.5; [recall SKILL.md](../../../../.claude/skills/wiki-read/SKILL.md) snapshot recall + god-node cross-reference
- Prior memory-system insights: [[20260513_2310_orphan_memory_branch_rewrite]], [[20260509_1619_summarize_study_skill_design_and_ship]], [[20260509_1620_documentation_framing_policy]], [[20260508_0429_memorize_skill_design_and_ship]]
- First code-graph snapshot: `docs/llm_wiki/code_snapshots/20260516_1322_v2_memory_build_complete.md`
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume a06843e3-4` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260516_1432_karpathy_graphify_adaptation_rationale]] (wiki_system_design, 2026-05-16) — Memory v2 borrowed wikilinks + backlinks + lint + contradiction-flag from Karpat
- [[20260516_1433_graphifyy_integration_cheatsheet]] (cluster_ops, 2026-05-16) — To use Graphify in any project conda env: install `graphifyy` (double-y; single-
- [[20260516_1435_worktree_baseref_and_propagation]] (subagent_engineering, 2026-05-16) — EnterWorktree defaults to branching from origin/<default-branch> (worktree.baseR
<!-- END BACKLINKS -->
