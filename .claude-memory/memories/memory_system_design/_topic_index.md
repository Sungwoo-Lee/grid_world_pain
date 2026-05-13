# _topic_index.md — `memory_system_design` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `memory_system_design` topic.

**Folder definition**: Claude memory system's own design decisions
**Insights**: 7
**Last updated**: 2026-05-13

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-13 | 23:10 | `20260513_2310_orphan_memory_branch_rewrite` | When a worktree branch carrying a single .claude-memory insight commit falls behind its base by N commits and another session edits the same .claude-memory/ index files in the meantime, the merge produces synthetic index-only conflicts — the insight body itself is unique. Cleaner pattern: rewrite the insight body fresh on the current base via `git show <orphan-branch>:<path>`, regenerate indexes from CURRENT counts, drop the orphan branch. ~5 min; linear history; orphan commit survives 30 days in reflog as safety net. |
| 2026-05-09 | 16:20 | `20260509_1620_documentation_framing_policy` | Established a project-wide Documentation framing policy in CLAUDE.md: every plan/design/analysis/summary/review/direction doc must lead with a plain-language entry-point section a reader without prior context can follow. 16 files updated in commit 62f96b9 (CLAUDE.md + 2 templates + 13 doc-producing agent profiles); first retro-application to v2 synthesis (commit 5c5adf3) added a 350-word TL;DR. The /summarize-study skill is the worked example. |
| 2026-05-09 | 16:19 | `20260509_1619_summarize_study_skill_design_and_ship` | Added a third documentation layer: docs/experiments/summaries/ for study-level reader-facing summaries. First executed manually (commit d075fdc, NMN comparison study summary), then promoted to a /summarize-study skill (commit 2b52c08) that automates the flow on demand. Same append-only versioning + timestamped-filename + auto-commit pattern as /memorize. |
| 2026-05-09 | 03:11 | `20260509_0311_diary_auto_session_backfill` | `scripts/diary_append.py` lazy-creates a Sessions row when an event arrives from a session that never called `session-start`. Resolves the parent UUID by scanning `~/.claude/projects/<encoded>/<UUID>.jsonl` (local) and `claude_data/.../<UUID>.jsonl` (synced). Convention preserved: full UUID only in Sessions table; 8-char prefix in Events / Training runs. |
| 2026-05-08 | 04:47 | `20260508_0447_recall_skill_design_and_ship` | Designed and shipped the `/recall` Claude Code skill, completing the `/memorize` ↔ `/recall` capture-recall pair; default = 10-most-recent flat list (overrides operating manual §8 time-grouped default). |
| 2026-05-08 | 04:29 | `20260508_0429_memorize_skill_design_and_ship` | Designed and shipped the `/memorize` Claude Code skill that captures conversations into `.claude-memory/`; validated end-to-end with a 3-eval benchmark (with-vs-without skill: +33pt). |
| 2026-05-08 | 03:15 | `20260508_0315_claude_memory_system_genesis` | Design decisions for in-repo `.claude-memory/` layer: coexists with built-in MEMORY.md; insight density splits the two layers; archives are local-only/gitignored. |

---

## Change history

- 2026-05-13: Added 1 insight from the dreamer-srl v3 CP1 closure session: `20260513_2310_orphan_memory_branch_rewrite` (when a memory-layer worktree branch falls behind its base and the only conflict surface is synthetic index counts, rewriting the insight on the current base + regenerating indexes is cleaner than merge-with-conflict-resolution). No new tags (all reused: memory, design, learned_lesson, decision, meta).
- 2026-05-09: Added insights `20260509_1619_summarize_study_skill_design_and_ship` (third documentation layer at `docs/experiments/summaries/` + the `/summarize-study` skill that automates it) and `20260509_1620_documentation_framing_policy` (project-wide plain-language entry-point rule promoted from one skill to all doc-producing surfaces).
- 2026-05-09: Added insight `20260509_0311_diary_auto_session_backfill` (`scripts/diary_append.py` self-heals the Sessions table for sessions that miss `session-start`).
- 2026-05-08: Added insight `20260508_0447_recall_skill_design_and_ship` (`/recall` skill ship).
- 2026-05-08: Added insight `20260508_0429_memorize_skill_design_and_ship` (`/memorize` skill ship).
- 2026-05-08: Folder created. Added genesis insight `20260508_0315_claude_memory_system_genesis`.
