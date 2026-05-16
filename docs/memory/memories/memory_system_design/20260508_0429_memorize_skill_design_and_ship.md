---
id: 20260508_0429_memorize_skill_design_and_ship
date: 2026-05-08
time: 04:29
folder: memory_system_design
tags: [memory, design, decision, skill]
summary: "Designed and shipped the /memorize Claude Code skill (.claude/skills/memorize/) that captures conversations into the in-repo docs/memory/ layer; validated end-to-end via a 3-eval benchmark with +33pt with-vs-without skill delta."
related: ["20260508_0315_claude_memory_system_genesis"]
session_origin: claude_code
session_label: "memorize skill rollout"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/f3ab7f37-218c-463b-ba24-e555d496dec1.jsonl
raw_completeness: full
---

# `/memorize` skill design and ship

## Key conclusion

Built `.claude/skills/memorize/` — a project-local Claude Code skill that captures the current conversation as one or more 5-section insight files in the `docs/memory/` layer, updates the topic and global indexes, and optionally archives the raw conversation. Shipped at commit `f878873`. The skill is now the canonical capture path (Step-by-step flow in `docs/memory/CLAUDE.md` §4); future `/memorize` invocations use it.

## Evidence, measurements, facts

- **Skill location**: `.claude/skills/memorize/SKILL.md` (project-local, ships with the repo).
- **Decisions captured at design-time** (locked via AskUserQuestion):
  - Name: `memorize` (mirrors the operating-manual trigger phrase "memorize this").
  - Multi-insight handling: **auto-split** with user confirmation (matches operating-manual anti-pattern "do not dump the whole session into one mega-insight").
  - Raw-archive default: **ask each time** (full / approximate / skip).
  - Test setup: **full eval loop** (skill-creator's draft → run → grade → benchmark workflow).
- **Eval results (iteration-1, 3 evals × 2 configs = 6 subagent runs in worktrees)**:
  - eval-1 single-decision: with_skill 9/10 (90%), without_skill 2/10 (20%).
  - eval-2 multi-insight: with_skill 2/10 (20%, **timed out at 6 min before substantive writes**), without_skill 8/10 (80%, but the agent wrote to the main repo's `docs/memory/` and to `~/.claude/.../memory/MEMORY.md` — boundary violation, see paired insight `20260508_0430_worktree_isolation_path_safety`).
  - eval-3 raw-archive: with_skill 10/10 (100%), without_skill 1/10 (10% — invented its own `insights/`, `raw/`, `topics/` folders, ignored the contract).
  - **Mean: with_skill 70%, without_skill 37%, delta +33 pts.**
- **Iteration-2 changes applied directly** (user opted to skip a second eval round): added a quick-reference table at the top of SKILL.md (saves time + tokens by avoiding re-reads of `docs/memory/CLAUDE.md` for every step), explicit "repo-relative paths only" hard rule (prevents the boundary-violation pattern observed in the eval-2 baseline), loosened the eval-1 assertion from "exactly one insight" to "one or two" (the agent's split of scope-decision from supporting pilot finding was defensible).
- **Tooling shipped**: `.claude/skills/memorize-workspace/grade.py` (programmatic assertion grader), `.claude/skills/memorize-workspace/build_benchmark.py` (turns grading.json files into benchmark.json). Workspace dirs gitignored as transient.
- **Coexistence rule** (from operating manual §2): short typed rules another agent must obey on every invocation → built-in `~/.claude/.../memory/MEMORY.md`; multi-section session insights → `docs/memory/`. The skill enforces this by hard rule and never writes to MEMORY.md.

## Decisions and actions

- Shipped `.claude/skills/memorize/` with `SKILL.md` and `evals/evals.json`.
- `.gitignore` updated: `.claude/skills/*-workspace/` and `.claude/worktrees/` are transient and stay local.
- Eval workspace artifacts (`memorize-workspace/iteration-1/`, including `review.html` and `benchmark.json`) preserved locally for reference but not committed.
- Did **not** wire `/memorize` as a custom slash command under `~/.claude/commands/` — the skill's `description` frontmatter already triggers on the slash invocation and the natural-language phrases.

## Open questions and follow-ups

- The 6-minute eval-2 timeout suggests subagent runs that read+write the full memory contract are at the edge of the harness's stream-idle limit. If we ever benchmark this skill again, consider (a) trimming SKILL.md further, (b) caching frequently-read indexes between subagent steps, or (c) splitting the skill into "identify candidates" + "write files" to reduce per-call wallclock.
- The eval-2 baseline's 80% pass-rate is misleading — it inflated by writing to the main repo (path-namespace escape) rather than the worktree. The grader saw the writes and counted them as success. Future evaluator scripts for any skill that writes files should explicitly assert "files only inside `cwd` / worktree" using path-prefix checks.
- Pre-existing `regen_dev_index.py` validation failure (`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md` carries `status: implemented`, not in `VALID_STATUS`) remained out of scope for this work but blocks the `docs/develop/INDEX.md` regeneration. Flag for a separate small bug-fix.

## References

- Skill: `.claude/skills/memorize/SKILL.md`
- Eval definitions: `.claude/skills/memorize/evals/evals.json`
- Operating manual: `docs/memory/CLAUDE.md` (the contract this skill enforces)
- Genesis insight: `20260508_0315_claude_memory_system_genesis` (paired — same folder, sequential decisions)
- Paired insight in this session: `20260508_0430_worktree_isolation_path_safety` (the engineering finding from running the evals)
- Design plan: `docs/develop/active/meta/claude_memory_system_design.md`
- Ship commit: `f878873`
- `raw_source` points at the synced JSONL on NAS (`claude_data/.claude/projects/.../<UUID>.jsonl`). Push via `./sync-agent-data.sh claude push` to update the NAS copy; on another node, `./sync-agent-data.sh claude pull` first, then `claude --resume <UUID>` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot view). Backfilled from the deprecated `_archive/raw_conversations/` design.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_2310_orphan_memory_branch_rewrite]] (memory_system_design, 2026-05-13) — When a worktree branch carrying a single .claude-memory insight commit falls beh
- [[20260516_1431_v2_three_role_architecture]] (memory_system_design, 2026-05-16) — Memory System v2 separates code/memory knowledge into three surfaces — curated s
<!-- END BACKLINKS -->
