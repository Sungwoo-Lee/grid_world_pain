---
id: 20260819_1944_run_command_parallel_race_and_remote_pkill_self_kill
date: 2026-08-19
time: "19:44"
folder: cluster_ops
tags: [learned_lesson, training_runner, meta]
summary: "run_command.py is NOT safe to invoke concurrently: five parallel calls to five different nodes all returned node 112's two PIDs (532510/532722), because the calls race through a shared SSH control socket. Identical PIDs across distinct machines is the tell. Run node fan-out SEQUENTIALLY. Separately, a remote `pkill -f <pattern>` kills its own `bash -c` wrapper (the pattern appears in the wrapper's own command line), so any verification chained after it in the same remote command never runs — verify out-of-band with nvidia-smi/gpu_status."
related: ["20260630_1631_training_runner_double_launch_relay_auth", "20260726_0417_dwell_sweep_rerun_silent_skip_traps", "20260806_0306_gpu_claim_and_nas_git_lock_protocol"]
session_origin: claude_code
session_label: "no-hiding-predator verdict + cluster teardown"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# `run_command.py` cross-contaminates when run in parallel; remote `pkill -f` kills its own wrapper

## Key conclusion
Two independent traps hit while terminating 10 training runs across 6 nodes.

**(1) `run_command.py` must be called ONE NODE AT A TIME.** Backgrounding five calls to five different nodes and `wait`ing returned *byte-identical output for all five*, namely node 112's process list. The calls race through a shared SSH control socket / multiplexing path, and one node's response is served to all callers. This is silent — every node "answers", the answers just aren't theirs. The only reason it was caught is that **the same two PIDs (532510, 532722) cannot exist on five separate machines**; had the PIDs been plausible, the follow-up action (killing by PID) would have targeted the wrong processes on four nodes.

**(2) A remote `pkill -f '<pattern>'` kills the shell running it.** `run_command.py <node> "pgrep -af PAT; pkill -f PAT; sleep 8; pgrep -af PAT"` sends the whole string to a `bash -c`, whose *own* command line contains PAT. `pkill` therefore matches and kills the wrapper, and everything chained after it — the survivor check — never executes. The kill of the real targets still happens; only the verification is silently swallowed.

## Evidence, measurements, facts
- Parallel attempt: 5 background `run_command.py` calls (nodes 107/108/110/111/112) writing to 5 separate files. All 5 files contained `532510` and `532722`. The later sequential pass proved those PIDs were node 112's; nodes 107/110 actually held 1204293/1204507 and 2838529/2838740, node 108 a single 3161807, node 111 1800371.
- Sequential re-run (same command, one node at a time, `timeout 300` each) returned distinct, plausible PIDs per node — confirming the race, not a script bug in the remote command.
- `pkill` self-kill: every node's log ends at `--- sending SIGTERM ---`; the `--- survivors ---` and `(end)` markers never printed on any of the 6 nodes. Wrapper PIDs visible in the same logs (450993, 58644, 660718, 1708834, 3383075, 3656372) each carried the full `pgrep -af 'basic_bushrefuge_restpremium_nohide' ...` string, hence matched the pattern.
- Out-of-band verification worked: `scripts/lab/gpu_status.py` afterwards showed all 10 target GPUs `FREE`, 0% util, memory back to 1-196 MiB. `gpu_status.py` itself fans out fine — it produced correct per-node-distinct output on 6 and on 13 nodes, so the parallel hazard is specific to `run_command.py`, not to SSH fan-out in general.

## Decisions and actions
- Standing rule: **fan `run_command.py` out sequentially**, even when it costs minutes. For read-only cluster state prefer `scripts/lab/gpu_status.py`, which is parallel-safe.
- Sanity-check any multi-node result set for *impossible agreement* (identical PIDs / identical timestamps) before acting on it. Treat cross-node identity as corruption, not coincidence.
- When a remote command must both kill and verify, split into two `run_command.py` calls, or verify with a tool that does not share the killed pattern. Kill by a pattern that is specific to the target (here the config path `basic_bushrefuge_restpremium_nohide`, unique to one experiment) so a parallel session's runs cannot be caught.
- The same-shell `pkill` self-match was already noted for the LOCAL case in [[20260726_0417_dwell_sweep_rerun_silent_skip_traps]]; this entry records the REMOTE variant, where the consequence is a silently skipped verification rather than a dead worker.

## Open questions and follow-ups
- Root cause inside `run_command.py` not yet read — likely a fixed ControlPath shared across invocations. A fix (per-invocation ControlPath, or an internal lock) would make parallel fan-out safe and is worth scoping if multi-node batches become routine.

## References
- Code: `run_command.py` (repo root); `scripts/lab/gpu_status.py` (parallel-safe alternative).
- Related: [[20260726_0417_dwell_sweep_rerun_silent_skip_traps]] (local `pkill` self-match), [[20260806_0306_gpu_claim_and_nas_git_lock_protocol]] (why nvidia-smi alone is not proof of occupancy).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
