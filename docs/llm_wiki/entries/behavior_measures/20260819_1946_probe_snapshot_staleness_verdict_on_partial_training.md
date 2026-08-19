---
id: 20260819_1946_probe_snapshot_staleness_verdict_on_partial_training
date: 2026-08-19
time: "19:46"
folder: behavior_measures
tags: [learned_lesson, meta, decision]
summary: "A watermark-incremental probe only covers checkpoints that existed WHEN IT LAST RAN. The rest-premium verdict was computed on runs probed to ~35.7-40.3M episodes that had since trained to the full 100M budget — 60% of training never analysed. The diary compounded it: 154 rows still read `running`, so 'this run is still early' was indistinguishable from 'this run finished long ago'. Before believing any probe-derived verdict, diff PROBED_TO against CKPT_AVAIL per run."
related: ["20260726_0417_dwell_sweep_rerun_silent_skip_traps", "20260818_1620_rest_premium_sweep_refuted", "20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect"]
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

# A probe snapshot silently ages: the rest-premium verdict rests on 40% of training

## Key conclusion
`run_sweep.py` is incremental against a high-water mark: it evaluates only checkpoints newer than the max step already in the CSV. That is correct and cheap, but it means **the probe covers the training that existed at the moment the sweep last ran, not the training that exists now**. If the runs keep going and nobody re-probes, every downstream figure, statistic and verdict quietly describes a prefix of the run.

That is what happened here. The 10 parent rest-premium arms were probed to ~35.7-40.3M episodes. They then continued to their **full 100M-episode budget** and finished. No probe ever touched the last ~60M. The refutation verdicts in [[20260818_1620_rest_premium_sweep_refuted]] and [[20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect]] therefore rest on the first ~40% of training.

**This is distinct from the re-run traps in [[20260726_0417_dwell_sweep_rerun_silent_skip_traps]].** Those are about a re-run doing nothing when you asked it to. This is about **never asking** — the pipeline has no notion of "the run has moved on since I last looked", and nothing surfaces the shortfall.

## Evidence, measurements, facts
- Per-arm coverage gap (parent rest-premium arms), probe CSV max step vs newest checkpoint on disk:

  | arm | probed to | available |
  |---|---|---|
  | a01 | 39.4M | 100.0M |
  | a06 | 35.7M | 100.0M |
  | a09 | 39.1M | 100.0M |

  All 10 arms share the shape: probed 35.7-40.3M, available 100.0M.
- Completion is verifiable from disk without any live process: `models/` holds exactly **1000 numbered checkpoint dirs** (`checkpoint_frequency: 100000`) with the last at ~`100000039`, against the run config's `episodes: 100000000`. 1000/1000 = finished naturally.
- Same gap on the no-hide arms in the other direction: probed to ~38.2-40.3M, killed at ~57.6-60.4M.
- **The diary made this invisible.** 154 training rows across the diary still read `Status = running`, oldest from 2026-05-09 — `training-done` is rarely fired at run end. So "a run marked running" carried no information about whether it was 5% or 100% done. In this session the diary listed 24 runs as running; only 10 actually were, and the other 14 had completed their entire budget days earlier.
- Cross-check that closed it: a full-cluster `gpu_status.py` sweep showed every GPU on nodes 101-113 idle, contradicting the diary's 24 "running" rows.

## Decisions and actions
- **Check coverage before believing a verdict.** For each run, compare the probe CSV's max `step` against `ls <run>/models | sort -n | tail -1`. Cheap, and it is the difference between "flat for the whole run" and "flat for the part we looked at".
- **Read completion from disk, not from the diary.** Checkpoint count vs the config's `episodes` budget is ground truth; a `running` row is not evidence of anything.
- Closed the 24 stale August rows this session (commit `8ca2410`) with results distinguishing TERMINATED from COMPLETED. Deliberately did NOT bulk-rewrite the ~130 older stale rows — the result strings would have been fabricated; if cleaned, it should be scripted from on-disk checkpoint state.
- Re-probing the unanalysed 60M costs GPU time only, no retraining, and all 10 GPUs were freed by the teardown. Offered to the user; not yet run.

## Open questions and follow-ups
- Worth a `--check-coverage` mode on `run_sweep.py` that prints probed-vs-available per run and warns when a run has advanced past its probe watermark.
- Whether the flat premium result survives to 100M is genuinely open. Flat at 20M and flat at 40M makes flat at 100M the likely continuation, but "likely" is not "checked".

## References
- Code: `scripts/eval/dwell_sweep/run_sweep.py` (watermark logic); run configs under `results/JAX_RecurrentPPO/<run>/models/config.yaml`.
- Commit: `8ca2410` (closed 24 diary rows: 10 terminated, 14 completed-full-budget).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect]] (behavior_measures, 2026-08-19) — REFUTED: ambush risk does NOT explain why injury suppresses cover use. Re-traini
<!-- END BACKLINKS -->
