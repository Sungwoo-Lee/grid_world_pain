---
id: 20260806_0306_gpu_claim_and_nas_git_lock_protocol
date: 2026-08-06
time: "03:06"
folder: cluster_ops
tags: [training_runner, learned_lesson, decision, meta]
summary: "Three cluster-ops rules from the Dreamer Gate-2 arc: a free-looking GPU can already be claimed by a run still in its CPU phase (check processes and the diary, first diary row wins), a slow NAS git commit must be waited out rather than kill-retried, and concurrent deterministic runs must be compared by matched iteration, not wall-clock."
related: ["20260728_1646_git_commit_pathspec_prevents_cross_session_sweep", "20260806_0305_equivalence_testing_noise_floor_and_det_trio"]
relations: ["extends:20260728_1646_git_commit_pathspec_prevents_cross_session_sweep"]
session_origin: claude_code
session_label: "dreamer-integration Gate 2 — determinism audit + fix + deterministic leg"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/87d7d010-fa7d-466e-9113-1a7618f94d95.jsonl
raw_completeness: full
---

# Cluster-ops rules from the Gate-2 arc: claiming a GPU, waiting out a NAS git commit, comparing concurrent runs

## Key conclusion

Three operational lessons came out of running seven long GPU jobs across three nodes with parallel Claude sessions active. First, `nvidia-smi` reports GPU *residency*, not *intent* — a run launched 30-60 minutes earlier can still be in its CPU setup phase and invisible on the GPU while fully owning it, so a claim check needs the process list and today's diary as well. Second, git operations on this NAS can legitimately take five to nine minutes under parallel sessions; killing and retrying leaves orphaned zero-byte lock files and makes things worse. Third, when two deterministic runs execute concurrently on different GPUs, compare them at matched *iteration numbers*, never at matched wall-clock times.

## Evidence, measurements, facts

- **GPU double-booking incident (2026-08-04, node 102).** A Gate-2 A/B pair was launched onto GPUs 102:0 and 102:1 that a parallel session's two rPPO bush-refuge runs had claimed roughly an hour earlier. `nvidia-smi` showed no compute apps because those runs were still in config load / JAX compile / buffer init. Both Gate-2 runs (`raeuhlnt`, `eybhlb4k`) were aborted at ~6 minutes and relocated to node 106; the diary training rows carry the aborted pair with the relocation reason.
- The three-way check adopted: (1) `nvidia-smi` free, **and** (2) `pgrep -af "train.py|dreamer_srl_main"` on the node shows no unaccounted trainer, **and** (3) today's and yesterday's `docs/diary/` Training-runs table has no running row for that node:GPU. Conflict resolution rule: **the first diary row wins** — the later claimant relocates. Promoted to auto-memory as `feedback_gpu_preflight_diary_check`.
- **NAS git under parallel sessions.** Commits were observed hanging for 5-9 minutes; kill-and-retry loops left orphaned zero-byte `.git/index.lock` files, which then blocked subsequent operations for reasons unrelated to the original slowness. The protocol: make **one patient attempt** with a generous timeout (~10 minutes) and stderr visible; if a lock persists, prove it is orphaned via `lsof` on the lock file and `/proc/<pid>/fd` for any candidate holder **before** removing it. Never pre-check `[ ! -f .git/index.lock ]` and then act — that is a race that loses on a slow index. (This extends the project's standing git-lock rule with the orphan-verification step.)
- **Comparing tqdm-heavy logs.** The wrapper logs use carriage returns for progress bars; run `tr '\r' '\n'` before any line-wise comparison or the whole progress stream reads as a single line.
- **Matched-iteration, not wall-clock, comparison.** The three deterministic runs shared node 114 concurrently and ran at 3.4-4.2 s/iteration across nominally identical RTX 6000 Ada cards — a ~20% spread purely from slot contention. Wall-clock-aligned comparison would have manufactured differences; matched-iteration comparison showed 482/482 iteration tuples identical. The same effect appeared on node 106, where slot 106:1 ran ~5% slower than 106:0 and produced a −4.6% throughput reading that had nothing to do with the code under test.

## Decisions and actions

- The GPU three-way claim check and the first-diary-row-wins rule were written into auto-memory (`feedback_gpu_preflight_diary_check`) so the `training-runner` applies them on every launch.
- A binding config freeze was declared in the diary while the deterministic runs were live, because the pre-refactor side resolved shared config layers through the hardcoded main-checkout path — a config edit mid-run would have silently changed one side of the comparison.
- Log comparison tooling for this project standardises on carriage-return normalisation plus iteration-keyed alignment.

## Open questions and follow-ups

- Nothing enforces the diary claim check automatically; it is a discipline, and a parallel session that skips the diary row can still double-book. A pre-launch script that reads the diary Training-runs table would close it.
- Per-slot speed variation on nominally identical cards (3.4-4.2 s/it on one node) is unexplained — thermal or contention. Harmless for correctness, but it means throughput must always be compared slot-matched.

## References

- Diary: `docs/diary/2026-08-04.md` (aborted node-102 pair + relocation rows), `docs/diary/2026-08-05.md` (config-freeze note), `docs/diary/2026-08-06.md` (session progress report).
- Auto-memory rule: `feedback_gpu_preflight_diary_check` (machine-local; not in git).
- Throughput/slot evidence: `docs/experiments/active/dreamer_integration/gate2_ab_analysis.md` §4.5, Finding 3.
- Methodology context: [[20260806_0305_equivalence_testing_noise_floor_and_det_trio]].
- Extends the existing git-lock lesson [[20260728_1646_git_commit_pathspec_prevents_cross_session_sweep|extends]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 87d7d010-fa7d-466e-9113-1a7618f94d95` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/20260806_0306_gpu_claim_and_nas_git_lock_protocol.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260818_1621_wandb_log_code_walks_whole_repo]] (cluster_ops, 2026-08-18) — wandb.run.log_code('.') walked the ENTIRE repo before every training run: wandb'
- [[20260819_1944_run_command_parallel_race_and_remote_pkill_self_kill]] (cluster_ops, 2026-08-19) — run_command.py is NOT safe to invoke concurrently: five parallel calls to five d
<!-- END BACKLINKS -->
