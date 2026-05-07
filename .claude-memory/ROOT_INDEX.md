# ROOT_INDEX.md — `.claude-memory/` topic folder registry

> Authoritative list of every topic folder under `memories/`.
>
> Read this file before classifying a new insight. Folder definitions here are the matching surface — if a new insight does not match any definition verbatim, the new-folder justification protocol applies (see CLAUDE.md, "Fragmentation safeguards").

**Last updated**: 2026-05-08
**Active folders**: 2
**Total insights**: 3
**Last audit**: (none)

---

## Active folders

| Folder | Definition (1 line) | Insights | Last update | Top tags |
|---|---|---|---|---|
| `memory_system_design` | Claude memory system's own design decisions | 2 | 2026-05-08 | [memory, design, decision, skill] |
| `subagent_engineering` | Subagent + worktree usage gotchas | 1 | 2026-05-08 | [meta, learned_lesson, worktree, subagent] |

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
| This file | `.claude-memory/ROOT_INDEX.md` | Topic-folder metadata |
| Tag dictionary | `.claude-memory/memories/_global_tags.md` | All active tags |
| Topic indexes | `.claude-memory/memories/<folder>/_topic_index.md` | One-line summary per insight in that folder |

---

## Change history

- 2026-05-08: Captured 2 insight(s): `20260508_0429_memorize_skill_design_and_ship` into `memory_system_design`, `20260508_0430_worktree_isolation_path_safety` into the new `subagent_engineering` folder.
- 2026-05-08: Created. Added `memory_system_design` folder (genesis insight: `20260508_0315_claude_memory_system_genesis`).
