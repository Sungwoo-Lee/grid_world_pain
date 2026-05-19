---
id: 20260516_1435_worktree_baseref_and_propagation
date: 2026-05-16
time: "14:35"
folder: subagent_engineering
tags: [meta, worktree, learned_lesson, decision]
summary: "EnterWorktree defaults to branching from origin/<default-branch> (worktree.baseRef: fresh), which may be hundreds of commits behind the user's working branch. Reset the new worktree to the user's branch tip via `git reset --hard <branch>` — untracked files survive. To propagate work back, push refuses on a checked-out branch; receive.denyCurrentBranch=updateInstead requires a clean working tree (usually too restrictive). Cleanest pattern: from the user's main checkout, run `git -C <user_checkout> merge <worktree-branch> --ff-only`. Worked even with the user's branch repeatedly moving forward concurrently."
related: ["20260508_0430_worktree_isolation_path_safety", "20260516_1431_v2_three_role_architecture"]
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

# Worktree base-ref + checked-out-branch propagation pattern

## Key conclusion

Two related git-worktree gotchas surfaced repeatedly during the v2 build. (1) When a Claude background session calls `EnterWorktree`, the new worktree branches from `origin/<default-branch>` per the `worktree.baseRef: fresh` setting — which on this repo means `origin/main`, 467 commits behind the user's active `v1.4` branch at the time. The empty `docs/memory/`, missing scripts, missing diary entries pull the rug out from under any work that assumes "this worktree has the project's current state." Fix: immediately after EnterWorktree, `git reset --hard v1.4` (or whatever the user's branch is) — untracked files (e.g., a design doc you just authored) survive the reset. (2) When the work is done in the worktree, you can't `git push . HEAD:v1.4` to land it back on the user's branch — git refuses pushes to checked-out branches by default, and `receive.denyCurrentBranch=updateInstead` requires a clean working tree which is usually not the case (the user is working live). Fix: from the user's main checkout, run `git -C /path/to/user merge worktree-<branch-name> --ff-only` — this updates the user's branch ref + working tree atomically, and works even when the user has uncommitted changes in files the merge doesn't touch.

## Evidence, measurements, facts

- This session's worktree was created at `.claude/worktrees/memory-v2-design/` via `EnterWorktree(name="memory-v2-design")`. Initial state: branched from `origin/main` at commit `ba3fa5c`. `ls .claude-memory/` returned "No such file or directory" even though the user's main checkout had a 56-insight `.claude-memory/` directory — confirmed `git log --oneline origin/main..v1.4 | wc -l` returned 467 (the gap).
- `git reset --hard v1.4` brought the worktree branch tip to `07887e4` (the v1.4 tip at the start). The previously-untracked v2 design doc at `docs/develop/active/meta/claude_memory_system_v2_design.md` survived the reset (only tracked files are touched by `--hard`).
- Push attempts (`git push . HEAD:v1.4`) failed with `remote rejected … branch is currently checked out`. Setting `receive.denyCurrentBranch=updateInstead` on the user's checkout caused the next push to fail with "Working directory has unstaged changes" because the user had `docs/diary/2026-05-15.md` modified.
- Working pattern (used ~6 times during the build): `git -C /media/nas01/projects/Interoceptive-AI/grid_world_pain merge worktree-memory-v2-design --ff-only`. Succeeded every time, including when v1.4 had moved forward concurrently (the worktree branch first merged v1.4 in, then the ff from user-side completed).
- The user's branch moved forward 3 times during the build (commits adding film-lit reviews + dreamer-srl + hypervigilance launches). Each time, the worktree had to first `git merge v1.4 --no-edit` in the worktree, THEN ff-push from user side. The "ort" merge strategy handled all overlaps cleanly (the only overlap was `docs/diary/2026-05-16.md`, auto-merged by interleaving table rows by timestamp).

## Decisions and actions

- **Pattern adopted for any future Claude session needing to land work back on the user's branch**:
  1. `EnterWorktree` (or use existing).
  2. `git reset --hard <user-branch>` to bring the current project state in.
  3. Work, commit normally inside the worktree.
  4. To propagate: `git -C <user-checkout-path> merge <worktree-branch-name> --ff-only`.
  5. If v1.4 moved during the work, first `git merge <user-branch> --no-edit` in the worktree, then re-attempt the ff from user side.
- **`receive.denyCurrentBranch=updateInstead` rejected as default** — too brittle (requires clean working tree). The user-side `merge --ff-only` is more forgiving (only blocks if the merge would overwrite uncommitted changes to specific files, not if the working tree has any changes anywhere).
- **`worktree.baseRef: head` setting was considered** (would default to local HEAD instead of origin/<default-branch>) but NOT adopted — would mask legitimate cases where the user wants the upstream main as the base, and on this repo origin/main is meaningfully different from v1.4 (different state of the project). Reset-to-user-branch is the explicit form.
- The v2 build's 36+ commits + 6+ rounds of propagation validated the pattern. Zero history loss.

## Open questions and follow-ups

- Could the project's `.claude/settings.json` enforce `worktree.baseRef: head` automatically for new worktrees? That would make `EnterWorktree` default to the user's local HEAD, removing the need for the manual `git reset --hard`. Tradeoff: locks in a project-specific convention that may surprise an agent expecting Claude Code defaults.
- Is there a way to make the merge automatic on a Claude hook? E.g., a post-commit hook in the worktree that triggers `git -C <user> merge --ff-only` when safe? Probably more trouble than worth it — the manual merge is one bash line and makes the propagation explicit.

## References

- Prior worktree insight: [[20260508_0430_worktree_isolation_path_safety]] (about repo-relative path discipline inside worktrees — orthogonal concern)
- EnterWorktree tool docs: shown in Claude Code's deferred-tool schema; the `worktree.baseRef` setting is the underlying knob
- Companion insight (architecture): [[20260516_1431_v2_three_role_architecture]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume a06843e3-4` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1510_worktree_misses_post_branch_main_assets]] (subagent_engineering, 2026-05-16) — Background-session worktrees branched from origin/main miss lit-review assets th
- [[20260519_1810_bg_isolation_blocks_edit_not_bash]] (subagent_engineering, 2026-05-19) — In bg (background-job) sessions, the worktree-isolation guard blocks the Edit an
<!-- END BACKLINKS -->
