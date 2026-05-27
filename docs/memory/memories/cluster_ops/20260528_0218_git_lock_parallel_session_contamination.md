---
id: 20260528_0218_git_lock_parallel_session_contamination
date: 2026-05-28
time: "02:18"
folder: cluster_ops
tags: [meta, learned_lesson, decision]
summary: "When .git/index.lock from a parallel Claude session clears and your `git add <file> && git commit` runs, the commit consumes EVERYTHING currently staged in the index — including files staged by the queued parallel session. Recovery: git reset --soft + restore --staged for contamination; git commit -C <hash> to re-create accidentally-undone parallel-session commits."
related: ["20260516_1435_worktree_baseref_and_propagation"]
session_origin: claude_code
session_label: "EPISODE project_plan rewrite + Foam wikilink convention + VSCode startup fix"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/e386e7cb-3b2e-4604-abc2-921d1b6c973a.jsonl
raw_completeness: full
---

# Git lock + parallel Claude sessions — `git add <file>` does not guarantee atomicity

## Key conclusion

When multiple Claude Code sessions touch the same repo, `.git/index.lock` becomes a contention point. The default mental model — "I only `git add docs/diary/X.md && git commit -m ...`, so only that file lands in my commit" — is wrong. While your session is queued behind a stale or held lock, a parallel session can stage its own files into the same index. When the lock clears and your commit fires, it commits **everything currently staged**, not just what you added. The commit-by-name-only convention does not protect against this. Recovery requires care: undoing the bad commit risks accidentally erasing a SUBSEQUENT parallel-session commit that landed on top.

## Evidence, measurements, facts

- Concrete incident in this session: I ran `git add docs/diary/2026-05-26.md && git commit -m "docs(diary): 📚 note: project_plan.md rewritten..."`. The resulting commit (`bb46df6`) contained THREE files:
  - `docs/diary/2026-05-26.md` (mine — intended)
  - `.gitignore` (parallel session — added `docs/.obsidian/` ignore line)
  - `train_command-agent.sh` (parallel session — switched the script from a hypervigilance Round 2.6 launch to a dreamer_srl log50k cell C launch)
- The parallel session had staged those two files but was blocked behind a stale `.git/index.lock` (0 bytes, no process holder, created at 19:07). When I removed the stale lock and committed, the index already had the parallel-session changes pre-staged. My single-file `git add` ran on top of that pre-staged state.
- Recovery attempt and second failure: I ran `git reset --soft HEAD~1` to undo the contaminated commit. But the reflog showed that BETWEEN my contaminated commit (`bb46df6`) and my reset, the parallel session had cherry-picked another commit (`7f4ba4b` — "fix(dreamer-srl): --log-interval CLI") on top of mine. My `--soft HEAD~1` therefore moved HEAD from `7f4ba4b` back to `bb46df6`, leaving `7f4ba4b`'s changes staged but undone in the chain.
- Final recovery: `git commit -C 7f4ba4b` re-created the parallel-session commit (new SHA `0fb2795`, identical content + message + co-author). The contaminated `bb46df6` was left in place (its 3 files are individually valid; the message just mismatches the content).
- The diary skill's documented rule "Stage only by name; never `git add -A` or `.`" addresses a different failure mode (accidentally including other working-tree changes). It does NOT prevent index-state contamination from concurrent sessions.

## Decisions and actions

- Before any commit when parallel Claude sessions may be active: run `git diff --cached --stat` to inspect what is staged BEFORE committing. If the output includes files you did not add, do `git restore --staged <file>` for the unrelated ones to release them back to the parallel session, then commit only your intended files.
- If a contaminated commit lands and a subsequent parallel-session commit is on top of it: NEVER blindly `git reset --soft HEAD~1`. First check `git reflog` and `git log -5 --oneline` to see what's actually on top. If a parallel commit is on top, use `git commit -C <hash>` to re-create it after the reset, OR avoid the reset entirely and accept the message-vs-content mismatch (the changes themselves are valid).
- Stale `.git/index.lock` (0 bytes, no process holder) can be safely removed, but only after confirming via `ps -ef | grep -E "(git commit|git add|git checkout|git merge|git mv|git worktree)" | grep -v grep` that no actual git process holds it. The 0-byte size is the diagnostic — a holder writes its PID into the lock.
- The project's auto-commit standing-authorisation rule remains "stage specific files by name". This insight adds a second rule: also verify `git diff --cached --stat` matches expectations before committing when parallel sessions are in play.

## Open questions and follow-ups

- Whether a wrapper script (`safe_git_commit <file...> -m <msg>`) that does `git diff --cached --stat`, checks against the file list, and aborts on mismatch would be worth writing. Pros: prevents the failure mode. Cons: would have to be threaded through every skill that commits (diary, memorize, snapshot_code_graph, etc.). Probably not worth it; the recovery path is straightforward once the mechanism is understood.
- The parallel session's commits (`55d28e0` analysis doc, the cherry-picked `7f4ba4b` log-interval fix, `72186aa` Obsidian config cleanup) were all from other Claude sessions running on the same v1.4/v2.0 branch on the same NAS-mounted repo. This is the same root cause as the existing [[20260516_1435_worktree_baseref_and_propagation]] friction — multiple checkouts of the same branch interact via the shared `.git/`.

## References

- Reflog from incident (in conversation): `7f4ba4b → reset --soft → bb46df6` (contaminated); recovered via `git commit -C 7f4ba4b → 0fb2795`.
- Final clean v1.4/v2.0 chain: `72186aa` → `bb46df6` (contaminated but inert) → `0fb2795` (recreated cherry-pick) → ... → today's commits.
- Related cross-session friction insight: [[20260516_1435_worktree_baseref_and_propagation]] (worktree → user-branch propagation has analogous contamination via `GRAPH_REPORT.md` regeneration).
- Project root `CLAUDE.md` auto-commit rule: "Stage specific files by name (never `git add -A` / `.`)". This insight refines that rule with the parallel-session caveat.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume e386e7cb-3b2e-4604-abc2-921d1b6c973a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
