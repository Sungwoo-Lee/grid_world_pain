# _topic_index.md — `memory_system_design` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `memory_system_design` topic.

**Folder definition**: Claude memory system's own design decisions
**Insights**: 3
**Last updated**: 2026-05-08

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-08 | 04:47 | `20260508_0447_recall_skill_design_and_ship` | Designed and shipped the `/recall` Claude Code skill, completing the `/memorize` ↔ `/recall` capture-recall pair; default = 10-most-recent flat list (overrides operating manual §8 time-grouped default). |
| 2026-05-08 | 04:29 | `20260508_0429_memorize_skill_design_and_ship` | Designed and shipped the `/memorize` Claude Code skill that captures conversations into `.claude-memory/`; validated end-to-end with a 3-eval benchmark (with-vs-without skill: +33pt). |
| 2026-05-08 | 03:15 | `20260508_0315_claude_memory_system_genesis` | Design decisions for in-repo `.claude-memory/` layer: coexists with built-in MEMORY.md; insight density splits the two layers; archives are local-only/gitignored. |

---

## Change history

- 2026-05-08: Added insight `20260508_0447_recall_skill_design_and_ship` (`/recall` skill ship).
- 2026-05-08: Added insight `20260508_0429_memorize_skill_design_and_ship` (`/memorize` skill ship).
- 2026-05-08: Folder created. Added genesis insight `20260508_0315_claude_memory_system_genesis`.
