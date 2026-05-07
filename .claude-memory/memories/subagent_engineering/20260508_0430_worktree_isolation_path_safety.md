---
id: 20260508_0430_worktree_isolation_path_safety
date: 2026-05-08
time: 04:30
folder: subagent_engineering
tags: [meta, learned_lesson, worktree, subagent]
summary: "Agent-tool worktree isolation is filesystem isolation only — NOT path-namespace isolation. Subagents using absolute paths escape the sandbox. Future subagent prompts using worktree isolation must explicitly enforce repo-relative paths."
related: [20260508_0429_memorize_skill_design_and_ship]
session_origin: claude_code
session_label: "memorize skill rollout"
importance: high
status: settled
supersedes: []
raw_source: _archive/raw_conversations/20260508_0429_memorize_skill_design_and_ship.md
raw_completeness: full
---

# Worktree isolation is filesystem-only, not path-namespace

## Key conclusion

The `Agent` tool's `isolation: "worktree"` mode creates a separate git worktree for the subagent's working directory, but does **not** sandbox absolute filesystem paths. A subagent that writes to `/home/...`, `~/.claude/...`, or any absolute path under the parent repo (e.g., `/media/nas01/projects/Interoceptive-AI/grid_world_pain/.claude-memory/`) will write to the **real** filesystem location, defeating isolation. Subagent prompts that rely on worktree isolation must explicitly tell the subagent to use only relative paths or paths rooted at the worktree's `cwd`.

## Evidence, measurements, facts

- **Concrete failure 1**: During the `/memorize` skill iteration-1 eval, the eval-2 *baseline* (without_skill) subagent ran in a worktree at `.claude/worktrees/agent-a55aaa3dd14bdf829/`. Its agent report listed file paths like `/media/nas01/projects/Interoceptive-AI/grid_world_pain/.claude-memory/memories/wandb_logging/...` — the **main repo path**, not the worktree. It also edited `/home/vncuser/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` (the user's actual built-in auto-memory). Both writes succeeded because the harness only isolated the working tree, not the path namespace.
- **Concrete failure 2**: The eval-3 *with_skill* subagent (worktree `agent-aa8d1cdf666e4bcb6`) reported "the seed-copy from parent also brought in a `wandb_logging` folder with an existing insight that was apparently created concurrently in the parent (at 03:36)". Cross-contamination happened because the eval-3 agent copied the seeded `.claude-memory/` from the main repo at a moment when the main repo had **already** been polluted by the eval-2 baseline's escape. Worktrees that "copy from parent" inherit any pollution that crossed the boundary.
- **Pollution observed in `git status`**: `M .claude-memory/ROOT_INDEX.md`, `M .claude-memory/memories/_global_tags.md`, `?? .claude-memory/memories/wandb_logging/` — all from a subagent that should have been confined to its worktree. Cleanup required `git checkout` on the modified files plus `rm -rf` on the new untracked folder.
- **System reminder during the session**: Claude Code's harness even surfaced `~/.claude/.../memory/MEMORY.md was modified, either by the user or by a linter` — the modification was actually made by an escaping subagent; the harness had no way to attribute it correctly because the write went through the same filesystem APIs.
- **Confirmed mechanism**: `git worktree add` creates a separate working directory (with its own `.git` link) but Linux file APIs operate on absolute paths regardless of cwd. The subagent's shell tool can `cd` to anywhere or just write to absolute paths and bypass the worktree entirely.

## Decisions and actions

- **Engineering rule**: every subagent prompt that uses `isolation: "worktree"` and that involves writes must include an explicit instruction: *"Write your changes inside this worktree (the cwd) only. Do not touch any path outside the worktree, including `~/.claude/...`, `/home/...`, or absolute paths into the parent repo."* The `/memorize` skill's iteration-2 SKILL.md adopted this rule under "Hard rules → Repo-relative paths only".
- **Grading rule**: any future eval grader for a writes-files skill should explicitly assert "all created/modified files are inside the worktree path" using a path-prefix check (e.g., `realpath(modified) startswith realpath(worktree)`). The iteration-1 `grade.py` only checked `.claude-memory/` location, not absolute-path containment, which let the eval-2 baseline pass 80% despite its escape.
- **Cleanup playbook recorded** (for future occurrences): `git checkout <polluted-tracked-files>; rm -rf <polluted-untracked-dirs>; git worktree remove -f -f <worktree-path>; git branch -D <worktree-branch>`. Double-`-f` is required when the worktree is locked.

## Open questions and follow-ups

- Is there a safer agent-isolation mode in Claude Code (e.g., chroot, container, fakeroot) that enforces path-namespace isolation? The current `"worktree"` is the only option I know of. If not, worth a feature request.
- Should the harness emit a warning when a subagent's tool calls write outside its worktree's `cwd`? Even a soft warning ("file <abs-path> is outside worktree <wt-path>; proceeding") would have flagged the boundary violation in real time.
- The pre-existing `regen_dev_index.py` validation failure (`per_entity_avoidance_logging.md status: implemented`) is still pending a separate fix and unrelated to this insight.

## References

- Paired insight: `20260508_0429_memorize_skill_design_and_ship` (the eval that exposed this gotcha).
- Iteration-1 evidence: `.claude/skills/memorize-workspace/iteration-1/` (gitignored; lives locally).
- Iteration-2 SKILL.md hard-rule fix: `.claude/skills/memorize/SKILL.md` → "Hard rules" section, line about "Repo-relative paths only".
- Closest existing folder: `memory_system_design`. **Why a new folder**: `memory_system_design` is for decisions about the `.claude-memory/` layer's *content*; this insight is about Claude Code subagent tooling and worktree mechanics, which is upstream of any specific layer. A future insight about an agent-profile change or a subagent-spawning quirk would fit `subagent_engineering` cleanly but would feel orphaned in `memory_system_design`.
- `raw_source` link is local-only (archives are gitignored; cloners see a broken link by design).
