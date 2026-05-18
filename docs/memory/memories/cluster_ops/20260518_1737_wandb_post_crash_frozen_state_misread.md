---
id: 20260518_1737_wandb_post_crash_frozen_state_misread
date: 2026-05-18
time: "17:37"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "When a WandB run crashes, its `run.summary` keys stay frozen at the last logged values and the run object's _runtime stops incrementing. Querying summary keys via `api.run(...).summary` does NOT signal liveness — a post-crash query returns the same numbers the run logged at the moment of death, indefinitely. The 2026-05-13 14:20 re-summary read R2.6's frozen post-crash summary keys (the runs had died at 23:38 on 2026-05-12) and described them as 'in-flight progress at 29% elapsed', a misframing caught only when the user pointed out 'There is no running training' on 2026-05-14. The fix is to always check `run.state` (running / finished / failed / crashed) and ideally also the wall-clock delta against `run.summary._runtime` before interpreting `run.summary` keys as live."
related: ["20260518_1516_wandb_log_dict_timesteps_key", "20260518_1735_sameprop_a1_seed45_corner_camping_refuted", "20260518_1736_sameprop_c_seed44_directional_replication"]
session_origin: claude_code
session_label: "hypervigilance Round 2.6 re-launch + check"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl
raw_completeness: full
---

# WandB `run.summary` keys persist after crash — always check `run.state` before interpreting them as live

## Key conclusion

The Python WandB API's `api.run(<id>).summary` dictionary holds the **last logged values** of each summary key, and those values **do not change after the run terminates** (crashed, failed, killed, or finished). The `run` object's `_runtime` field also stops incrementing at the moment of termination. Neither of these signals leak the run's liveness state. The only canonical liveness signal is `run.state` (values: `running`, `finished`, `failed`, `crashed`, etc.). On 2026-05-13 at 14:20 KST I queried R2.6's two WandB runs (`s3k03eua` Cell C and `k08v38af` Cell A1) via `summary` keys to "check progress" and wrote up the numbers in the re-summary §2 row 4 as "in flight at 29% elapsed time, bush-dive gap +28.5 pp and still rising." Both runs had actually crashed ~14.5 h earlier on 2026-05-12 at 23:38 — the 29% elapsed-time number was the frozen `_runtime/3600 / 22 ≈ 0.30` from the moment of death, and the "still rising" was a comparison across already-frozen WandB time-series windows. The user caught it on 2026-05-14 ("There is no running training. Check them"), and `state=failed` was the conclusive signal. The fix is one extra line in any WandB-progress-check script: log `run.state` first, refuse to treat summary keys as "current progress" unless `state == 'running'` AND the most recent log entry is within an expected freshness window (e.g., `time.time() - last_log_ts < 600 s`).

## Evidence, measurements, facts

- **The bug pattern** (paraphrased): I ran a Python snippet that did `r = api.run(f'.../{rid}')`, then `s = r.summary`, then printed `s.get('Episode/BushDiveRate_predator')` and similar. The values returned were the last-logged values from before the crash. No exception, no warning, no liveness flag in the result. The script wasn't wrong; my interpretation was wrong — I treated the returned dict as "the current state of training" when it's really "the last state at which the run wrote to wandb." For a crashed run, that's the moment of death.
- **The corroborating numbers** (post-incident verification, 2026-05-14): same `api.run(...)` query later returned `state=failed`, `_runtime/3600 = 6.62 h` for Cell C and 6.57 h for Cell A1. Both `_step` values (1.18 M and 0.87 M episodes) matched what we'd expect for a 6.6 h run on a 3090 — confirming the crash time of 2026-05-12 23:38 KST (launch at 17:01 + 6.6 h).
- **WandB time-series queries have the same blind spot.** When I ran the 6-window `timeseries` query at 2026-05-13 14:20 to look at the bush-dive gap evolution, the data I got back was the full crash-truncated time-series. The fact that the last window ended at the crash time (not at "now") was not surfaced in the output unless I cross-checked timestamps against wall-clock.
- **The corrected check pattern**, used at this session's 2026-05-18 progress query:

  ```python
  r = api.run(f'sungwoolee/grid_world_pain/{rid}')
  runtime_h = r.summary.get('_runtime', 0) / 3600
  print(f'state={r.state}  runtime={runtime_h:.2f}h  step={r.summary.get("_step")}')
  ```

  The `state` field was logged at the top of the report and used to gate any "in flight" framing. The 2026-05-18 query correctly identified Cell C as `running` (49.49 h, still climbing) and Cell A1 as `finished` (47.75 h, 10 M ep).

- **The 2026-05-13 14:20 re-summary** was corrected post-hoc by a fresh dated re-summary at 2026-05-14 23:32 (`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`). The README index row for the 2026-05-13 file carries an inline caveat. The corrected re-summary's §2 row 4 now reads "Round 2.6 CRASHED at ~6.6 h on n106" with the partial-data framing properly bounded.

## Decisions and actions

- **Add to the team's WandB-progress-check pattern**: every script that queries `run.summary` for "is this run going well?" purposes must print `run.state` first and refuse to interpret summary keys as live unless `state == 'running'`. For `state in ('failed', 'crashed', 'killed')`, the script should label the values as "frozen post-crash snapshot" and surface the freeze time (`run.summary._runtime / 3600` h from start, compared against the actual launch time).
- **A future polish would log freshness** — i.e., the wall-clock age of the most recent log entry. WandB's run object has `lastHistoryStep` and timestamps; an "age" check (`time.time() - last_log_ts < 600 s`) catches a run that's in `state=running` but has stalled (e.g., hung on a GPU OOM that didn't propagate to wandb-side crash detection within the heartbeat window). Not a P0; nice-to-have.
- **The post-mortem framing in the corrected re-summary** is now the canonical example of this lesson — readers who follow the link chain `summaries/README.md → 20260514_2332 → §2 row 4 "Round 2.6 CRASHED" → this insight` will see the lesson in context.
- **Don't blame the WandB API.** The behaviour is correct (summary keys are persistent by design — that's the whole point of `summary` vs `history`); my reading was wrong. The lesson is for Claude (and any human teammate) doing live-progress checks, not for the WandB library.

## Open questions and follow-ups

- **Should `wandb-analysis` skill bake this in?** The `wandb-analysis` skill at `.claude/skills/wandb-analysis/SKILL.md` is the canonical surface for "I want to look at training results." If it doesn't already require a `state` check before quoting summary keys, this insight is the trigger to add it. (Not checked in this session — flag for a future audit pass.)
- **Are there other "looks live but isn't" surfaces in WandB?** The web UI shows a "stopped at <date>" badge for finished runs, but the API does not always make this prominent. A future audit could enumerate: `summary`, `history`, `lastHistoryStep`, `_runtime`, `state`, and `last_updated_at` — and document which fields are reliable liveness signals.

## References

- The misframed re-summary (correct on the science, wrong on the in-flight status): [`docs/experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md`](../../../experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md) — see §2 row 4 (now annotated with a post-hoc caveat via the README's index row).
- The corrected reader-friendly re-summary: [`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`](../../../experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md) — §2 row 4 carries the canonical "CRASHED at ~6.6 h on n106" framing.
- Sibling cluster_ops WandB lesson: [[20260518_1516_wandb_log_dict_timesteps_key]] — also about WandB-API gotchas, different axis (step-metric vs log_dict).
- Companion insights from this session — Cell A1 corner-camping refuted: [[20260518_1735_sameprop_a1_seed45_corner_camping_refuted]]; Cell C directional replication still running: [[20260518_1736_sameprop_c_seed44_directional_replication]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059` or `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl /tmp/20260518_1737.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260518_1735_sameprop_a1_seed45_corner_camping_refuted]] (hypervigilance, 2026-05-18) — R2.6 Cell A1 seed 45 finished 10M ep on n102 with Episode/Steps=98/500 — H₁(A1-s
- [[20260518_1736_sameprop_c_seed44_directional_replication]] (hypervigilance, 2026-05-18) — R2.6 Cell C seed 44 at 62.5% of 10M-episode budget on n101 (still running) repro
<!-- END BACKLINKS -->
