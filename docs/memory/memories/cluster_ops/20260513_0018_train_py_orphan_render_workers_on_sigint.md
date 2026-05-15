---
id: 20260513_0018_train_py_orphan_render_workers_on_sigint
date: 2026-05-13
time: "00:18"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "When `train.py` is terminated mid-run via SIGINT (e.g., via `terminate_command.py`), the parent process exits cleanly and flushes WandB, but the ~16-20 `render_recordings.py` worker subprocesses it spawned to render gameplay videos do NOT receive the signal automatically. They keep running as orphans on the node, all targeting the same recordings directory (with `--skip-existing` they do no useful work). A second `terminate_command.py … 'render_recordings.py'` call is needed to clean them up — or send the SIGINT to the process group rather than the parent."
related: ["20260508_1434_terminate_command_key_auth_refactor"]
session_origin: claude_code
session_label: "NMN R2 continual + 6-specialist analyzer verdict — first positive FiLM finding"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# `train.py` SIGINT leaves render_recordings.py workers orphaned — second kill pattern needed

## Key conclusion

When the user terminated the `passive_matched` specialist (PID 11855 on n104) via `terminate_command.py 104 'rppo_nmn_meta_spec_passive_matched_s0' -y` (which sends SIGINT for a graceful shutdown), the parent `train.py` process exited cleanly and flushed its WandB summary — the intended behaviour. But a follow-up `ps -ef` revealed **19 orphaned `render_recordings.py` worker subprocesses** still running on the node, all targeting the same `recordings/9720013` directory with identical args. These are children that `train.py` had spawned during its periodic checkpoint-rendering step but had not reaped before SIGINT killed it. With `--skip-existing` in their args, they do no useful work, but they consume CPU and clutter the node's process table. The fix in this session was a second `terminate_command.py … 'render_recordings.py' -y` call — the workers had already self-exited under SIGINT propagation by the time the second scan ran, but the pattern (kill main, then sweep up render orphans) is the safe default.

## Evidence, measurements, facts

- **Concrete instance**: 2026-05-12 23:25 KST, n104 cuda:1.
  - Main process: PID 11855 = `python train.py --config configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml --agent_config configs/models/recurrent_ppo_nmn_het_unmod.yaml --num-envs 128 --episodes 10000000 --seed 0 --device cuda:1 --tag rppo_nmn_meta_spec_passive_matched_s0`.
  - After SIGINT: main process exited, WandB summary captured at 9,670,022 episodes (96.7% of 10M target).
  - Then `pgrep -af 'rppo_nmn_meta_spec_passive_matched'` returned 19 entries — all identical:
    `python /media/.../scripts/render_recordings.py results/JAX_RecurrentPPO/20260509-184214_.../recordings/9720013 --concat --skip-existing --cleanup-per-episode --fps 5`
- **Why 19 identical commands**: `train.py` spawns rendering as a multiprocessing pool when a checkpoint completes; the pool's workers do not get reaped if the parent dies between spawn and pool-close.
- **Self-exit on second kill**: by the time the second `terminate_command.py 104 'render_recordings.py' -y` scan ran (~3 minutes later), the workers had already exited on their own — likely under shell session-end / process-group cleanup, or because `--skip-existing` finished their work. So in this specific case the second kill was redundant, but the pattern is not safe to rely on.
- **Safer alternative for future**: send SIGINT to the **process group** (`kill -INT -- -<PGID>`) rather than the parent PID. This propagates to all children automatically.
- **Where this matters**: any time `terminate_command.py` is used on `train.py` mid-checkpoint. A clean training run that finishes normally reaps its own children, so this gotcha is termination-specific.

## Decisions and actions

- This session's workaround (two-call pattern) is workable but not ergonomic. Logged as a known operational quirk; not yet a fix.
- **Suggested fix to `terminate_command.py`** (not yet implemented): after killing the main pattern, automatically scan for and sweep `render_recordings.py` on the same node(s). One flag, one extra call.
- **Alternative fix to `train.py`** (more invasive): wrap the multiprocessing pool in a signal handler that closes the pool on SIGINT before the main process exits, so the workers terminate with their parent.
- Either fix is a `cluster_ops` `feature-workflow` candidate when prioritized. Low urgency — known workaround works.

## Open questions and follow-ups

- Does the same orphan pattern occur with other auxiliary tools `train.py` spawns (e.g., WandB sync workers)? Quick `pgrep` after a known termination on n104 would confirm.
- Does `train.py` actually call `pool.close()` + `pool.join()` in its checkpoint-render path? If yes, the orphans imply the pool was spawned but not joined before SIGINT. If no, the fix is straightforward.
- Should `terminate_command.py` grow a `--sweep-children` flag that knows the common child-process patterns for this project's training scripts?

## References

- Related: [[20260508_1434_terminate_command_key_auth_refactor]] (the SSH-key-auth refactor of `terminate_command.py` — this insight extends that tool's known-issues list).
- Script: [`terminate_command.py`](../../../terminate_command.py)
- Trigger script: [`scripts/render_recordings.py`](../../../scripts/render_recordings.py)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260513_0018_train_py_orphan_render_workers_on_sigint.md`.
