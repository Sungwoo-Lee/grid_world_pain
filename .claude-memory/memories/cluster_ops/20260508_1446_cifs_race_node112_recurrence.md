---
id: 20260508_1446_cifs_race_node112_recurrence
date: 2026-05-08
time: "14:46"
folder: cluster_ops
tags: [meta, training_runner, learned_lesson]
summary: "CIFS attribute-cache race recurred on node 112 (not just node 114), and the training-runner agent did not auto-apply the canonical /tmp/<unique>.sh bypass on the first launch of either round — duplicate phantom processes spawned twice (Round 1 `gh1cz01q`, Round 2 Cell C `f5d933ro`). Confirms the bypass is universal across nodes and must be applied upfront, not as a fallback after `pgrep` catches the duplicate."
related: ["20260508_1433_cifs_bypass_for_run_command", "20260508_1434_terminate_command_key_auth_refactor"]
session_origin: claude_code
session_label: "hypervigilance_sameprop_2026-05-08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl
raw_completeness: full
---

# CIFS race recurrence on node 112 — bypass must be applied upfront, not as a fallback

## Key conclusion

The CIFS attribute-cache race documented in `20260508_1433_cifs_bypass_for_run_command` (originally on node 114) recurred on **node 112** during this session's Round 1 and Round 2 launches — confirming the question "Are nodes 101–113 also affected?" with a positive cross-node data point. More importantly, the runner agent did **not** auto-apply the canonical `/tmp/train_cmd_<unique>.sh` bypass on the first launch of either round; in both rounds it caught the duplicate via the post-launch `pgrep -af '<tag>'` check (good) and then SIGKILL'd the duplicate (good), but only switched to an inline-command form (not the canonical `/tmp` bypass) for the *second* launch of the pair. This is a process-compliance gap, not a new failure mode: the canonical bypass works when used, but the runner is using `bash train_command-agent.sh` upfront despite its profile (`.claude/agents/training-runner.md` §4) requiring the bypass on every launch. Future-Claude orchestrating training launches must enforce the bypass upfront on every launch of every round, on every node — `pgrep` is a backstop, not a substitute.

## Evidence, measurements, facts

- **Round 1 incident** (2026-05-07, ~22:31): launching `hypervigilance-sameprop-relog-seed42_n112_gpu0` on node 112 cuda:0 spawned a duplicate phantom process `gh1cz01q` with the same tag/seed. Caught by post-launch pgrep; SIGKILL'd within ~2 minutes before any checkpoints were written. Runner switched to inline-command form for the seed43/cuda:1 launch and that succeeded.
- **Round 2 incident** (2026-05-08, ~14:17): launching `hypervigilance-round2-C-seed42_n112_gpu0` on node 112 cuda:0 spawned a duplicate phantom process `f5d933ro`. Identified by start-time (the authoritative `0u266oj5` started at 14:17:54, the duplicate at 14:19:08, both with identical command-line). SIGTERM at 14:30; the JAX process took ~90 seconds to respond and end (consistent with JAX's signal-handling lag — a re-check loop over `pgrep` is necessary; do not assume a SIGTERM has succeeded after 5 seconds). cuda:0 dropped from 10968 MiB (two procs) to 5510 MiB (single proc) after the kill. Cell A1 cuda:1 launch used the inline-command form and did not duplicate.
- Node 112 hardware: RTX 3090 (24 GB VRAM). Steady-state per-process VRAM ≈ 5.5 GB. Two concurrent same-config processes on cuda:0 = ~11 GB — not OOM, but compete for SM time and corrupt the experiment if both write to the same WandB run name. (In both incidents the runner-issued `--wandb-name` was identical for the duplicate, so WandB sometimes assigned a different ID and the two procs ended up writing to two different runs — visible cleanup needed in WandB UI.)
- Canonical bypass (from `20260508_1433_cifs_bypass_for_run_command` and runner profile §4): on every launch, write the command-script content to `/tmp/train_cmd_$(date +%s)_${RANDOM}.sh` on the target node directly via SSH, then call `run_command.py 112 grid_world_pain "bash /tmp/train_cmd_<that>.sh"`. `/tmp/` is node-local, never CIFS-cached, and the random suffix prevents reuse of any cached path.
- Runner's deviation today: ran `bash train_command-agent.sh` (the NAS-mounted path) on the first launch of each round. The runner *did* edit `train_command-agent.sh` on the NAS for each cell, but the node read a stale-cached version. The post-launch `pgrep` check then caught the duplicate, and the runner switched to inline-command form (also a valid bypass — it avoids reading any script file at all on the node — but not the canonical `/tmp/<unique>.sh` form documented in the runner profile).
- Inline-command form ≠ `/tmp/<unique>.sh` form: both bypass the CIFS cache, but the canonical form preserves the script-file audit trail and works with longer command strings. Inline-command can hit shell-quoting issues with very long commands. The canonical form should remain the default.
- The runner's report flagged `f5d933ro` for manual WandB-side junk-marking. As of this insight write, the user has not yet marked it junk in the WandB UI.

## Decisions and actions

- **Reinforcement, not new rule**: the existing canonical bypass from `20260508_1433` and the corresponding `feedback_runner_cifs_bypass.md` in built-in `MEMORY.md` are correct as-is. No code or config change is needed; only consistent agent compliance.
- **Compliance gap to surface**: future orchestrators (top-level Claude or agent-manager) spawning the training-runner should explicitly remind the runner in the prompt to use the canonical bypass for *every* launch, including the first launch of a parallel pair. The runner's own profile already requires this, so the prompt-time reminder is belt-and-braces.
- **Universal applicability confirmed**: the bypass is now empirically required on at least nodes 112 and 114. No reason to expect any cluster node to be safe; treat the bypass as universal.
- **WandB cleanup pending (manual)**: mark `gh1cz01q` and `f5d933ro` as junk in the WandB UI. Both wrote ≤ 2 minutes of data and would clutter analysis if not flagged.
- **JAX SIGTERM timing**: budget at least 30–90 seconds for a JAX training process to respond to SIGTERM. Do not assume failure if the process is still alive 5 seconds in; it is mid-step. Use a re-check loop (e.g. `until ! pgrep -f <tag>; do sleep 5; done`) rather than escalating to SIGKILL prematurely. Round 2's duplicate kill took ~90 s; Round 1's took ~120 s.

## Open questions and follow-ups

- Is the inline-command bypass ever preferable to `/tmp/<unique>.sh`? Possibly when the command string is short, the cell is one-off, and editing `train_command-agent.sh` is not desired (no audit need). For all production-experiment cells, default to `/tmp/<unique>.sh`.
- Does CIFS attribute-cache TTL ever expire, or is a positive-cache invalidation needed? If TTL is finite (e.g. 30 s), a `sleep 30` between NAS edit and SSH-launch might also resolve it without the `/tmp` redirect. Untested. The redirect is more deterministic; cost is trivial.
- Should the runner profile be updated to **require** a SIGTERM-then-wait protocol (not SIGTERM-then-recheck-after-5s) for cleanup? Worth a small profile patch if this recurs on a future cleanup.
- WandB cleanup of `gh1cz01q` and `f5d933ro` still pending — outside Claude's scope, user action.

## References

- Sibling insight: `20260508_1433_cifs_bypass_for_run_command` (canonical bypass recipe; this insight is a recurrence + cross-node confirmation, not a replacement).
- Sibling insight: `20260508_1434_terminate_command_key_auth_refactor` (the cleanup-tool refactor used to SIGKILL duplicates).
- Sibling insight: `20260508_1428_node_env_recovery_recipe` (cross-folder context: lab-node ops).
- Runner profile: `.claude/agents/training-runner.md` §4 (Launch — use the CIFS-bypass pattern), §4b (Post-launch `pgrep` check), §4c (`terminate_command.py` for cleanup).
- Built-in memory pointer: `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/feedback_runner_cifs_bypass.md` (one-line operational rule).
- WandB IDs to mark junk: `gh1cz01q` (Round 1 duplicate, 2026-05-07), `f5d933ro` (Round 2 Cell C duplicate, 2026-05-08).
- Authoritative runs (DO NOT junk): `rg5nl1ov`, `6ks4bjbq` (Round 1), `0u266oj5`, `27svrmhv` (Round 2).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b12b1fea-94dc-491e-998b-735c656560cb` or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl /tmp/20260508_1446.md`.
