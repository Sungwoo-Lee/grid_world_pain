# _topic_index.md — `memory_system_design` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `memory_system_design` topic.

**Folder definition**: Claude memory system's own design decisions
**Insights**: 11
**Last updated**: 2026-06-09

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-09 | 17:25 | [20260609_1725_env_docs_tutorial_primer_pattern](20260609_1725_env_docs_tutorial_primer_pattern.md) | Pattern for tutorial-grade reference docs: re-sync to code-as-truth first, then one shared API-primer with stable HTML anchors taught once + per-doc verbatim embeds with primer-linked callouts + a hub learning path, instead of re-teaching each API everywhere. |
| 2026-05-28 | 02:16 | `20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases` | Project-wide doc-linking convention adopted: `[[filename]]` wikilinks default for new cross-doc references (resolved by Foam at runtime; survives file moves); `aliases: [<id>]` opt-in only for ~10–20 god-node docs (survives renames). Old `[text](path.md)` links kept; no bulk migration. Filename stem IS the id — no separate `id:` field. 6 load-bearing docs aliased (project_plan, NEUROMODULATION_ALGORITHM, ENVIRONMENT_SUMMARY, FRONTMATTER_CONTRACT, memory/CLAUDE.md, AGENT_PLAYBOOK). Convention doc at `docs/develop/active/meta/doc_linking_convention.md`; CLAUDE.md bullet under Project-Wide Rules. Same pattern shape as [[20260509_1620_documentation_framing_policy]]: policy promoted from one layer (memory tree) to all doc-producing surfaces. |
| 2026-05-16 | 14:32 | `20260516_1432_karpathy_graphify_adaptation_rationale` | Memory v2 adopted four Karpathy LLM Wiki primitives (wikilinks + backlinks + lint + contradiction-flag) and four Graphify primitives conceptually (god-nodes + surprising-connections + GRAPH_REPORT format + edge confidence), then extended with bitemporal `valid_until` + `confidence` fields and a first-class conversation-records-as-graph-nodes layer. Deliberately did NOT merge the live code-graph into the memory layer (kept as gitignored sibling). The adaptation rationale is what makes v2 distinct and what future v3 needs to retrace. |
| 2026-05-16 | 14:31 | `20260516_1431_v2_three_role_architecture` | Memory v2 separates code/memory knowledge into three surfaces with distinct lifetimes: curated session insights (docs/memory/memories/), live regenerable code-graph (src/graphify-out/, gitignored), and dated immutable code-snapshots (docs/memory/code_snapshots/). Bridged by /memorize's opt-in snapshot prompt at Step 2 and /recall's god-node cross-reference hint. Different epistemic kinds get different homes; agents route to the right surface for the question type. |
| 2026-05-13 | 23:10 | `20260513_2310_orphan_memory_branch_rewrite` | When a worktree branch carrying a single .claude-memory insight commit falls behind its base by N commits and another session edits the same docs/memory/ index files in the meantime, the merge produces synthetic index-only conflicts — the insight body itself is unique. Cleaner pattern: rewrite the insight body fresh on the current base via `git show <orphan-branch>:<path>`, regenerate indexes from CURRENT counts, drop the orphan branch. ~5 min; linear history; orphan commit survives 30 days in reflog as safety net. |
| 2026-05-09 | 16:20 | `20260509_1620_documentation_framing_policy` | Established a project-wide Documentation framing policy in CLAUDE.md: every plan/design/analysis/summary/review/direction doc must lead with a plain-language entry-point section a reader without prior context can follow. 16 files updated in commit 62f96b9 (CLAUDE.md + 2 templates + 13 doc-producing agent profiles); first retro-application to v2 synthesis (commit 5c5adf3) added a 350-word TL;DR. The /summarize-study skill is the worked example. |
| 2026-05-09 | 16:19 | `20260509_1619_summarize_study_skill_design_and_ship` | Added a third documentation layer: docs/experiments/summaries/ for study-level reader-facing summaries. First executed manually (commit d075fdc, NMN comparison study summary), then promoted to a /summarize-study skill (commit 2b52c08) that automates the flow on demand. Same append-only versioning + timestamped-filename + auto-commit pattern as /memorize. |
| 2026-05-09 | 03:11 | `20260509_0311_diary_auto_session_backfill` | `scripts/diary_append.py` lazy-creates a Sessions row when an event arrives from a session that never called `session-start`. Resolves the parent UUID by scanning `~/.claude/projects/<encoded>/<UUID>.jsonl` (local) and `claude_data/.../<UUID>.jsonl` (synced). Convention preserved: full UUID only in Sessions table; 8-char prefix in Events / Training runs. |
| 2026-05-08 | 04:47 | `20260508_0447_recall_skill_design_and_ship` | Designed and shipped the `/recall` Claude Code skill, completing the `/memorize` ↔ `/recall` capture-recall pair; default = 10-most-recent flat list (overrides operating manual §8 time-grouped default). |
| 2026-05-08 | 04:29 | `20260508_0429_memorize_skill_design_and_ship` | Designed and shipped the `/memorize` Claude Code skill that captures conversations into `docs/memory/`; validated end-to-end with a 3-eval benchmark (with-vs-without skill: +33pt). |
| 2026-05-08 | 03:15 | `20260508_0315_claude_memory_system_genesis` | Design decisions for in-repo `docs/memory/` layer: coexists with built-in MEMORY.md; insight density splits the two layers; archives are local-only/gitignored. |

---

## Change history

- 2026-05-16: Added 2 insights from the memory v2 build + graphify integration session: `20260516_1431_v2_three_role_architecture` (the 3-role architecture + memory/code-graph bridge as the load-bearing design decision of v2) and `20260516_1432_karpathy_graphify_adaptation_rationale` (what was borrowed from Karpathy LLM Wiki + Graphify, what was extended, what was skipped — the meta-insight a future v3 needs to consult). No new tags (all reused: memory, design, decision, meta).
- 2026-05-13: Added 1 insight from the dreamer-srl v3 CP1 closure session: `20260513_2310_orphan_memory_branch_rewrite` (when a memory-layer worktree branch falls behind its base and the only conflict surface is synthetic index counts, rewriting the insight on the current base + regenerating indexes is cleaner than merge-with-conflict-resolution). No new tags (all reused: memory, design, learned_lesson, decision, meta).
- 2026-05-09: Added insights `20260509_1619_summarize_study_skill_design_and_ship` (third documentation layer at `docs/experiments/summaries/` + the `/summarize-study` skill that automates it) and `20260509_1620_documentation_framing_policy` (project-wide plain-language entry-point rule promoted from one skill to all doc-producing surfaces).
- 2026-05-09: Added insight `20260509_0311_diary_auto_session_backfill` (`scripts/diary_append.py` self-heals the Sessions table for sessions that miss `session-start`).
- 2026-05-08: Added insight `20260508_0447_recall_skill_design_and_ship` (`/recall` skill ship).
- 2026-05-08: Added insight `20260508_0429_memorize_skill_design_and_ship` (`/memorize` skill ship).
- 2026-05-08: Folder created. Added genesis insight `20260508_0315_claude_memory_system_genesis`.
