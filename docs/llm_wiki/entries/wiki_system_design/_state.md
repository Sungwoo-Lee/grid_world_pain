---
folder: wiki_system_design
synthesized_from: 14 entries
synthesized_on: 2026-07-28
last_entry_included: 20260728_1644_wiki_pull_gate_widened_before_any_task
---

# LLM Wiki design — current belief

## What we believe now

- **Two recall layers, split by insight density, deliberately named apart.** Claude Code's built-in auto-memory holds one-line rules an agent must obey every invocation and is pushed into context every turn; the in-repo LLM Wiki holds decisions with rationale and is pulled on demand. The split rule has held unchanged since genesis; only the naming was fixed, in July 2026, because calling both "memory" forced a disambiguation in every conversation. [[20260508_0315_claude_memory_system_genesis]], [[20260728_1642_llm_wiki_rename_ends_memory_collision]]

- **The wiki is three surfaces with different lifetimes, not one store.** Curated session entries (durable, hand-written), a live regenerable code graph (disposable), and dated code snapshots (immutable). The code graph was deliberately kept OUT of the wiki as a gitignored sibling. [[20260516_1431_v2_three_role_architecture]]

- **Consulting is a cheap gate, not a full read.** Nothing auto-loads. Before any non-trivial task, read the `ROOT_INDEX` Active-folders table (~2 KB) to learn *whether* the wiki knows something; drill further only on a match. Reading entries speculatively defeats the design. [[20260728_1644_wiki_pull_gate_widened_before_any_task]]

- **The filename stem IS the identifier, so entry IDs are immutable.** `[[filename]]` wikilinks resolve at runtime and survive file moves; `aliases:` are reserved for ~10–20 load-bearing docs. This is why the July 2026 rename left every entry ID carrying the old vocabulary rather than rewriting 326 edges. [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]], [[20260728_1642_llm_wiki_rename_ends_memory_collision]]

- **Borrowed deliberately, extended deliberately.** Wikilinks, backlinks, lint, and contradiction-flagging came from the Karpathy LLM Wiki pattern; god-nodes and the graph-report format from Graphify. The project's own additions are the bitemporal `valid_until` + `confidence` fields and conversations-as-graph-nodes. [[20260516_1432_karpathy_graphify_adaptation_rationale]]

- **Capture and recall are a shipped skill pair, not a manual habit.** `/wiki-write` and `/wiki-read`, each validated against evals at ship time. [[20260508_0429_memorize_skill_design_and_ship]], [[20260508_0447_recall_skill_design_and_ship]]

- **Documentation framing is project-wide, and was promoted out of this folder.** Every doc leads with a plain-language entry point. It started as one skill's rule and became a `CLAUDE.md` rule for all doc-producing surfaces — the project's first worked example of the §16 promotion path. [[20260509_1620_documentation_framing_policy]], [[20260509_1619_summarize_study_skill_design_and_ship]]

## What is contested

- **Nothing currently contested in this folder.** The 14 entries are additive: each extends or refines its predecessors rather than overturning them. If that stays true it is worth being suspicious of — a design folder with zero refutations may mean the design has never been stress-tested against a case it fails.

## What was refuted

- **A dedicated wiki-search sub-agent** was evaluated and rejected on measured economics (~36 KB agent boot floor against a 12.5 KB mean entry, giving a ~3-entry crossover) plus reachability (sub-agents hold no `Agent` tool). Recorded in the sibling folder: [[20260728_1645_wiki_search_subagent_rejected_on_economics]]

## What is still open

- Whether the widened consult gate is actually honoured in practice. The cost numbers are measured; the behavioural effect is not. [[20260728_1644_wiki_pull_gate_widened_before_any_task]]
- Whether generated topic indexes plus `_state.md` are enough retrieval structure past ~200 entries, or whether the v2 critique's hybrid search eventually becomes unavoidable.
- No automated guard exists against the component-path class of rename bug. [[20260728_1643_bulk_rename_component_path_and_domain_term_traps]]

## How to use this page

For "why is the wiki built this way", read [[20260508_0315_claude_memory_system_genesis]] then [[20260516_1431_v2_three_role_architecture]]. For "how do I touch it without breaking it", read [[20260728_1643_bulk_rename_component_path_and_domain_term_traps]].
