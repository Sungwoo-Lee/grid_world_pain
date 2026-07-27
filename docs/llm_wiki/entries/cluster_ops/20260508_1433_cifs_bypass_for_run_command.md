---
id: 20260508_1433_cifs_bypass_for_run_command
date: 2026-05-08
time: "14:33"
folder: cluster_ops
tags: [meta, training_runner, learned_lesson]
summary: "Node 114 CIFS client caches `train_command-agent.sh` inode content; fresh NAS edits can be invisible to remote bash → silent duplicate launches. Workaround: edit train_command-agent.sh on NAS for audit, AND ALSO write the same content to /tmp/train_cmd_<unique>.sh on the node directly via SSH; then run_command.py executes the /tmp path."
related: ["20260508_1428_node_env_recovery_recipe", "20260508_1434_terminate_command_key_auth_refactor"]
session_origin: claude_code
session_label: "dreamer_diagnostic_battery_2026-05-07"
importance: medium
status: settled
supersedes: []
raw_source: _archive/raw_conversations/20260508_1431_diagnostic_battery_refutes_four_fixes.md
raw_completeness: full
---

# CIFS bypass: write launch script to /tmp/<unique>.sh on the node, not the NAS

## Key conclusion
The lab nodes' CIFS client (verified on node 114; assume true for the rest) caches the inode content of files mounted from the NAS, and the cache can lag a NAS-side `Edit` by minutes. When `run_command.py 114 grid_world_pain "bash train_command-agent.sh"` runs after a fresh NAS edit, node 114 may bash the **previously cached** `train_command-agent.sh` content — silently launching the wrong (prior) experiment. This caused 2 unwanted duplicate cell-A launches on 2026-05-07 when trying to launch cell B. Workaround that always works: (1) Edit `train_command-agent.sh` on the NAS for audit-trail/diff history; (2) ALSO write the same content to `/tmp/train_cmd_$(date +%s)_${RANDOM}.sh` on the target node directly via SSH; (3) call `run_command.py 114 grid_world_pain "bash <that-/tmp-path>"`. `/tmp/` is node-local, never CIFS-cached, and the unique random suffix prevents reuse of any cached path.

## Evidence, measurements, facts
- Incident date: 2026-05-07. Tried to launch cell B of the diagnostic battery; cell A was instead launched 3 times total (PIDs 1910637 at 21:34, 1930400 at 21:54, 1932641 at 22:00) — all with `--tag dreamer_ablation_lever_only_s0`, even though the runner had Edited `train_command-agent.sh` to cell B's content before the second and third invocations.
- Diagnosis path: the runner's post-launch `pgrep -af '<tag>'` check (also added on 2026-05-07) caught the duplicates; SSH'd to node 114 to confirm the running command-line was cell A's, not B's; compared `train_command-agent.sh` content on NAS (cell B) vs what node 114 was actually running (cell A). Mismatch identified as CIFS staleness.
- Recovery: SIGINT to PIDs 1930400 and 1932641 cleaned up; cell A's original PID 1910637 kept running.
- Workaround validated end-to-end: cells B/C/D on 2026-05-07 (diagnostic battery) and cells E1/E2/E3/E4 on 2026-05-08 (probe battery) all launched cleanly via `/tmp/train_cmd_<cell>_$(date +%s)_${RANDOM}.sh` — 8/8 launches successful, no further duplicate-PID incidents.
- Memory audit at the time: `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` already contains a one-line entry pointing at the runner profile (`feedback_runner_cifs_bypass.md`) — this insight here is the rationale chain that the one-liner does not carry.

## Decisions and actions
- Updated `.claude/agents/training-runner.md` §4 (Launch — use the CIFS-bypass pattern) on 2026-05-07 (commit `6c2aa34`). Mandatory pattern for every launch.
- Added matching entry in built-in `MEMORY.md` (`feedback_runner_cifs_bypass.md`) so all runner spawns inherit the rule on every invocation.
- Added the post-launch `pgrep -af '<tag>'` mandatory check (§4b in the runner profile) at the same time — second-line defense even if CIFS-bypass fails.
- All subsequent launches on node 114 (8 cells across two batteries) used the workaround successfully.

## Open questions and follow-ups
- Is the staleness window finite (e.g., CIFS attribute cache TTL ~30 s)? If so, a `sleep 60` between the NAS edit and the SSH-launch might also resolve it without the `/tmp/` redirect — but the redirect is more deterministic and the cost is trivial.
- Are nodes 101–113 also affected? Untested. The bypass pattern is safe to apply universally; no need to verify per-node.
- /tmp/ files accumulate on each node. Lab-node `/tmp/` is auto-cleaned on reboot, but if a node runs continuously for weeks, manual cleanup may be useful. Outside the runner's scope.

## References
- Built-in MEMORY.md entry: `feedback_runner_cifs_bypass.md` (one-liner, harness-managed).
- Runner profile: `.claude/agents/training-runner.md` §4 (Launch — use the CIFS-bypass pattern), §4b (Post-launch pgrep), §4c (terminate_command.py for cleanup).
- Sibling cluster_ops insight: `20260508_1428_node_env_recovery_recipe` (parallel-session capture about node-101 conda recovery).
- Sibling cluster_ops insight: `20260508_1434_terminate_command_key_auth_refactor` (same-session refactor of the cleanup tool).
- Commits: `6c2aa34` (CIFS-bypass pattern in profile), `fcdf1cc` (direct terminate_command.py invocation), `40f0f4d` (terminate_command.py refactor).
- raw_source link is local-only (archives are gitignored — broken link on a fresh clone). The archive is shared with the 3 sibling insights captured in the same session.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
