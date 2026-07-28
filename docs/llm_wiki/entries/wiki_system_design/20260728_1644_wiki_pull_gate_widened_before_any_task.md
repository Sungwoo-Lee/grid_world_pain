---
id: 20260728_1644_wiki_pull_gate_widened_before_any_task
date: 2026-07-28
time: "16:44"
folder: wiki_system_design
tags: [wiki, design, decision, meta]
summary: "The wiki is pull-only and was gated on 'when wiki work is requested', so it was never consulted during ordinary work (bug fixes, refactors, config changes) even when a relevant entry existed — while Claude Code's built-in auto-memory is pushed into context every turn. Widened the trigger to 'before any non-trivial task' while preserving lazy-load: read the ROOT_INDEX Active-folders TABLE (~2 KB), not the file (52 KB, mostly change log). Drill to a topic index only on a match. Commit 76d29d5."
related: ["20260508_0315_claude_memory_system_genesis", "20260516_1431_v2_three_role_architecture", "20260516_1434_cross_phase_generator_backlinks_strip", "20260728_1645_wiki_search_subagent_rejected_on_economics"]
relations: ["extends:20260508_0315_claude_memory_system_genesis", "extends:20260516_1431_v2_three_role_architecture"]
session_origin: claude_code
session_label: "LLM Wiki rename + consult gate"
importance: high
status: settled
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/dec28250-837d-46e8-b4c9-5ce845ae7698.jsonl
raw_completeness: full
---

# The wiki was never read during ordinary work — the consult gate was too narrow

## Key conclusion

Nothing in the wiki loads automatically. The only instruction telling anyone to read it said *"when wiki work is requested"* — meaning it was consulted when the user asked about the wiki, and essentially never during the work the wiki exists to inform. Meanwhile Claude Code's built-in auto-memory is injected into context on every single turn. The layer holding 184 entries of hard-won rationale was the one that never got read; the layer of one-line rules was the one that could not be missed. The trigger is now widened to **before any non-trivial task** — bug fix, feature, refactor, experiment launch, config change — but deliberately made cheap enough that widening it does not cost context.

## Evidence, measurements, facts

- Demonstrated live in the session that produced this entry: the LLM Wiki rename was executed while consulting **zero** entries, even though [[20260516_1434_cross_phase_generator_backlinks_strip]] (about the exact regenerator script being renamed and re-run) and `20260513_2310_orphan_memory_branch_rewrite` (about index conflicts in that exact directory) were both sitting in the wiki.
- Cost measurements that shaped the rule:
  - `ROOT_INDEX.md` whole file: **52,669 bytes** — of which the Active-folders table (lines 1–32) is only **2,062 bytes**. The remaining ~50 KB is dated change log.
  - Topic indexes: 3–24 KB (`cluster_ops` 24,489 B for 40 rows).
  - Mean entry: **12,514 bytes** across 184 entries.
- The initial draft of this rule described `ROOT_INDEX.md` as "~10 rows, near-free" — wrong by 25x at the file level. Measuring before writing the rule is what caught it; the rule now names the *table*, not the file.

## Decisions and actions

- Root `CLAUDE.md` gains a **"Consult before acting (the pull gate)"** paragraph: read the Active-folders table (~2 KB, stop before the change-log tail); drill into a folder's `_topic_index.md` only on a match; open an entry only when its one-line summary looks relevant; on no match, proceed.
- The framing that keeps it cheap: **the gate answers "does the wiki know anything about this area?" for 2 KB. It does not answer "what does it know?"** — that is paid only after a match. Reading entries speculatively is the failure mode the cost table exists to prevent.
- `docs/llm_wiki/CLAUDE.md` gains §3 "When to consult, and how far to drill", carrying the per-level cost table. Mechanism detail lives there, not in root `CLAUDE.md`, which is loaded every turn and must stay minimal.
- §9 (lazy-load levels) had to be reconciled: its old L1 row said "first read of any wiki **operation**", which directly contradicted the widened trigger. L1 is now split into **L1a** (Active-folders table only — the gate, before any non-trivial task) and **L1b** (full `ROOT_INDEX` + tag dictionary — explicit wiki operations only).
- Because sub-agents inherit the project `CLAUDE.md`, this applies to `developer`, `experiment-designer`, the reviewers, and the rest — not only top-level Claude.
- Refines the lazy-load design from [[20260508_0315_claude_memory_system_genesis|extends]] and [[20260516_1431_v2_three_role_architecture|extends]] rather than superseding it: the L0–L4 levels are unchanged, only the entry trigger moved.

## Open questions and follow-ups

- `confidence: medium` because the cost numbers are measured but the *behavioural* effect is not: whether future-Claude actually performs the 2 KB check before ordinary work is unvalidated. Worth re-checking in a few sessions by asking whether any task cited a wiki entry it would previously have missed.
- A `SessionStart` hook would enforce it harder, at the cost of paying the read every session whether relevant or not. Not adopted.
- The cheapest remaining win is upstream: topic-index rows average ~600 bytes when a scannable index wants ~120. See [[20260728_1645_wiki_search_subagent_rejected_on_economics]].

## References

- Commit `76d29d5`.
- Root `CLAUDE.md` "LLM Wiki" section; `docs/llm_wiki/CLAUDE.md` §3 and §9.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume dec28250-837d-46e8-b4c9-5ce845ae7698` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260728_1642_llm_wiki_rename_ends_memory_collision]] (wiki_system_design, 2026-07-28) — The project's in-repo session-insight layer and Claude Code's built-in auto-memo
- [[20260728_1645_wiki_search_subagent_rejected_on_economics]] (subagent_engineering, 2026-07-28) — Considered and rejected a dedicated wiki-search sub-agent that would answer topi
<!-- END BACKLINKS -->
