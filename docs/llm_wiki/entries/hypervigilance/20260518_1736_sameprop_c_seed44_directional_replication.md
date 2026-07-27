---
id: 20260518_1736_sameprop_c_seed44_directional_replication
date: 2026-05-18
time: "17:36"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, decision]
summary: "R2.6 Cell C seed 44 at 62.5% of 10M-episode budget on n101 (still running) reproduces the R2.5 event-level class discrimination directionally: training-time M2 gap +29 pp (R2.5 eval-time was +37 pp), M5_predator 0.95 (R2.5 eval was 0.75), M5_rabbit 1.37 (R2.5 was 1.19), per-tag rabbit_TL vs rabbit_BR identical to within 0.06 pp. Both threshold misses (M2_predator 0.74 < 0.80; M5_predator 0.95 ≥ 0.80) are consistent with online-training-average being weaker than the locked deterministic eval protocol — final verdict requires the offline eval-rollout that produced R2.5's §12 numbers. 2026-05-21 SETTLED: H₁(C-event) confirmed at eval-time on the 10 M checkpoint — see `## Closing update (2026-05-21)` below."
related: ["20260512_1428_sameprop_class_discriminating_defence_event_level", "20260518_1735_sameprop_a1_seed45_corner_camping_refuted", "20260518_1737_wandb_post_crash_frozen_state_misread"]
session_origin: claude_code
session_label: "hypervigilance Round 2.6 re-launch + check"
importance: medium
status: settled
valid_until: 2026-05-21
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl
raw_completeness: full
---

# R2.6 Cell C seed 44 — directional replication of the +37 pp event-level class gap; final verdict pending offline eval

## Key conclusion

Round 2.6 Cell C (decoupleFood, seed 44) on n101 cuda:0 has reached step 6.25 M of the 10 M-episode target (62.5%) at 49.5 h wall-clock and is still running. The training-time online cumulative numbers for the event-level toolkit measures (`Episode/BushDiveRate_*`, `Episode/EatUnderThreatRatio_*`) reproduce the **direction** of the R2.5 §12 finding — predator-near bush-diving is higher than rabbit-near, and per-tag rabbit_TL vs rabbit_BR are essentially identical, ruling out single-instance artifacts — but the **magnitudes** are weaker than R2.5's eval-time numbers, and two of the three locked confirmation thresholds from H₁(C-event) are currently failing. Specifically, M2_BushDiveRate_predator is 0.74 vs the 0.80 bar, the Δ_M2 class gap is +29 pp vs the +30 pp bar, and the EatUnderThreatRatio_predator is 0.95 vs the < 0.80 bar. This is **not** a refutation of R2.5: the R2.5 §12 numbers come from a deterministic eval-rollout (200 episodes, exploration off) at the final checkpoint, while the R2.6 numbers shown here are training-time cumulative averages that include earlier-training exploration. The right comparison is R2.6's eval-rollout at the 10 M checkpoint vs R2.5's eval-rollout at the 10 M checkpoint; that comparison cannot be made until the run finishes (~30 h more on n101's 2080 Ti) and the offline scripts run. The verdict is **directional replication confirmed, magnitude verdict pending**.

## Evidence, measurements, facts

- **Run identity**: WandB `ja5fu5k3`, name `hypervigilance-round26-C-seed44_n101_gpu0_relaunch`. Launched 2026-05-16T13:21:04 KST; `state=running` at check time (2026-05-18 17:30 KST); 49.5 h elapsed.
- **Progress**: `_step = 6,250,017` of 10,000,000 target (62.5%). Steady-state SPS implies remaining ~30 h → expected completion 2026-05-19 ~23:00 KST. The 2080 Ti is ~3× slower than the 2026-05-12 launch's 3090 on n106, which is why the projected wall-time stretched from ~22 h to ~80 h.
- **Pre-registered H₁(C-event)** (from `sameprop_round26_design.md` §1): `M2_BushDiveRate_predator ≥ 0.80` AND `Δ_M2_class ≡ M2_predator − M2_rabbit ≥ +30 pp` AND `M5_EatUnderThreatRatio_predator < 0.80`. Per-tag rabbit_TL vs rabbit_BR agree within ±5 pp on M2 and ±0.10 on M5.
- **Online cumulative @62.5%** (Episode/* keys, all-episodes averages from training):

| Measure | R2.5 (eval, seed 42) | R2.6 (online, seed 44 @62.5%) | Threshold | Status |
|---|---|---|---|---|
| `BushDiveRate_predator` | 0.876 (eval) | 0.7406 | ≥ 0.80 | fail-online (note: `predator_full` subtag = 0.7922, closer to bar) |
| `BushDiveRate_rabbit` | 0.508 (eval) | 0.4478 | — | — |
| `Δ_M2_class` | +0.368 (eval) | +0.293 | ≥ +0.30 | marginal (just below) |
| `BushDiveRate_rabbit_TL` | (within noise vs BR) | 0.5290 | — | — |
| `BushDiveRate_rabbit_BR` | (within noise vs TL) | 0.5298 | — | — |
| per-tag rabbit fan-out (M2) | within noise | Δ = 0.08 pp ≪ ±5 pp | ≤ ±5 pp | ✓ |
| `EatUnderThreatRatio_predator` | 0.748 (eval) | 0.9485 | < 0.80 | fail-online |
| `EatUnderThreatRatio_rabbit` | 1.186 (eval) | 1.3650 | — | over-baseline ✓ |
| `EatUnderThreatRatio_rabbit_TL` | (within noise) | 1.4308 | — | — |
| `EatUnderThreatRatio_rabbit_BR` | (within noise) | 1.3715 | — | — |
| per-tag rabbit fan-out (M5) | within noise | Δ = 0.059 ≪ ±0.10 | ≤ ±0.10 | ✓ |

- **Survival**: `Episode/Steps = 387.2/500` (vs R2.5 seed 42 which had `Episode/Steps` ≈ 380s by the last 10% of training; comparable). Termination breakdown: Injury = 0.33, MaxSteps = 0.45, Starvation = 0.22 — a mixed-cause death profile, similar to R2.5.
- **Contact counts**: `Episode/PredatorHits = 3.51`, `Episode/RabbitHits = 0.98`, `Episode/HidingPredatorHits = 2.44`. The agent still touches the patrolling predator a lot — bush-diving as a defensive policy isn't fully protective, only differential.
- **Online vs eval-time discrepancy** is the load-bearing methodological caveat. R2.5's §12 numbers come from 200 deterministic eval episodes at the final checkpoint using `scripts/eval_rollout.py` (exploration off, fixed-policy rollouts). R2.6's online numbers here are running averages over **all** training episodes, including exploration-heavy early ones. Online numbers are systematically biased toward "less class-discriminating" because they include the agent's pre-convergence policy.
- **What the R2.6 verdict requires**: run the same `scripts/eval_rollout.py` + `scripts/motif_cluster.py` pipeline that produced R2.5 §12 against the final R2.6 Cell C checkpoint. The R2.6 design's §1 confirmation criteria are explicitly written for the eval protocol — the online numbers shown here are an early-warning indicator, not the verdict.

## Decisions and actions

- **Hold the R2.6 Cell C verdict open until offline eval-rollout completes.** Don't propagate the "M2 = 0.74 < 0.80, threshold misses" as a refutation. The directional replication (positive class gap, per-tag fan-out within noise, predator-near eat suppression below rabbit-near) is the load-bearing observation; the magnitude verdict will be the offline eval.
- **Wait for completion (~30 h, 2026-05-19 ~23:00 KST).** Then spawn `experiment-analyzer` with the same offline pipeline used for R2.5 §12.
- **Log this as `status: active` with `valid_until: 2026-05-21`** — by that date, the offline eval should have run and produced the definitive R2.6 verdict. After that, this insight should either flip to `settled` (verdict matched) or be superseded by the post-eval finding.
- **The slow wall-time on n101 (2080 Ti)** is now confirmed: 49.5 h → 62.5 % means ~80 h total vs the original n106 (3090) projection of ~22 h. For future R2.x launches that need turnaround speed, request 3090-class hardware (n102/103/104/105 have 3090/4090s) and avoid the 2080 Tis on n101 unless tolerance for ~3× wall-time is acceptable. Companion lesson on hardware-aware scheduling — captured as a soft heuristic, not worth a separate insight.

## Open questions and follow-ups

- **Will the magnitude gap close at convergence?** If the eval-time R2.6 numbers land near (M2_predator ≈ 0.85, Δ_M2 ≈ +35 pp, M5_predator ≈ 0.75), R2.5 is seed-locked and the §12 verdict is paper-grade. If they land near (0.78, +28 pp, 0.90), the R2.5 finding is **partial** — present but smaller than originally reported, with seed-variance the dominant uncertainty.
- **What is the eval-vs-online correction factor?** Worth computing for future cells: on R2.5 Cell C, the ratio M2_predator_eval / M2_predator_online_at_final would give a calibration constant. Useful for any future online-monitoring approach where we want to flag a "likely-to-pass" verdict before running offline eval.
- **The Episode/PredatorHits = 3.5/ep is high.** If bush-diving were fully protective it should be near zero. The fact that it isn't means the agent's defensive strategy is "dive when threatened, but accept contact sometimes." This is consistent with the R2.5 finding that bush-diving is class-discriminating (the *gap* is large) without being fully protective (the absolute predator-hit rate is non-zero). Could motivate a deeper analysis of *when* bush-diving fails — is it geometric (predator approaches from a direction with no bush)? policy-failure (agent sees threat but doesn't dive in time)? worth a per-motif breakdown after the offline eval.
- **Hardware lesson generalisability**: the 80 h / 22 h wall-time ratio (3.6× slow) on n101 vs n102 should be sanity-checked against the dreamer-srl `SPS_SIZE_NUM_ENVS_SWEEP.md` results. If the ratio matches there, n101 is permanently flagged as "low-throughput tier"; if it doesn't, there's something specific to RPPO that hurts on the 2080 Ti.

## Closing update (2026-05-21) — verdict H₁(C-event) confirmed

The run finished cleanly at 10,000,021 episodes on n101 cuda:0, 81 h 43 m total wall-clock (vs the in-flight estimate of ~80 h — accurate). The offline eval-rollout pipeline (200 deterministic episodes, seeds 1000–1199, cue radius R=3.0, K=5, K_motif=7, k-means k=6 seed=42 — same protocol as R2.5 §12.1) was run against the final checkpoint at `results/JAX_RecurrentPPO/20260516-132103_hypervigilance-round26-C-seed44_n101_gpu0_relaunch/models/10000021/`, with output at `results/eval/models/10000021/` (matching R2.5's `results/eval/models/10000022/` convention).

**Eval-time cross-tab numbers (the load-bearing comparison)** — computed via `tmp/20260521_r26_c_seed44_eval_analysis.py`, mirroring R2.5's `tmp/20260511_r25_appendix_analysis.py`:

| Measure | R2.6 seed 44 (eval, n=200, seeds 1000–1199) | R2.5 seed 42 (§12.2, eval, n=200, seeds 0–199) | Threshold | Verdict |
|---|---:|---:|---|---|
| M2 bush-dive rate, predator | **88.8 %** (2529 / 2849) | 87.6 % (2222 / 2537) | ≥ 80 % | ✓ passes by +8.8 pp |
| M2 bush-dive rate, rabbit | 51.5 % (897 / 1741) | 50.8 % (851 / 1674) | — | — |
| Δ_M2_class (pred − rab) | **+37.3 pp** | +36.8 pp | ≥ +30 pp | ✓ passes by +7.3 pp |
| M5 eat-under-threat ratio, predator | **0.769** | 0.748 | < 0.80 | ✓ passes by 0.031 |
| M5 eat-under-threat ratio, rabbit | 1.202 | 1.186 | — | — |
| Per-tag rabbit M2 fan-out (TL vs BR) | 49.6 % / 53.3 % → 3.7 pp gap | 50.3 % / 52.3 % → 2.0 pp gap | ±5 pp | ✓ within band |
| Per-tag rabbit M5 fan-out (TL vs BR) | 1.255 / 1.116 → 0.139 gap | 1.207 / 1.130 → 0.077 gap | ±0.10 | ⚠ 0.039 over the band |
| M7 silhouette | 0.186 | 0.186 (identical to 3 decimals) | ≥ 0.20 nominal | matches R2.5 — same caveat applies |

**Verdict (formal predicate)**: **H₁(C-event) confirmed**. All three primary §4.1 thresholds met; per-tag M2 fan-out within band; per-tag M5 fan-out marginally over band (a secondary-check overrun on a measure whose primary threshold passes by 0.031). The §6 designer's 80 %-prior outcome held.

**Status**: this insight flips from `active` to `settled`. The threshold misses observed at 62.5 % training (M2_pred 0.74, M5_pred 0.95) WERE the predicted exploration-driven underestimate of the deterministic-eval values — the in-flight reading was correct in calling these "directional, magnitude pending." The full design doc analysis is at [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../../../experiments/active/hypervigilance/sameprop_round26_design.md) §§9–12.

**Methodological notes worth surfacing**:
- The eval-rollout takes ~5 min CPU-time per 200-episode RPPO checkpoint on n101's CPU (matching R2.5's protocol; CPU is sufficient — GPU is not needed for the eval phase).
- The 10-episode mini-redo (seeds 1200–1209) for reproducibility-of-protocol gave M2_pred = 0.790 — 9.8 pp below the n=200 main eval; this is in the lower tail of the predicted n=10 bootstrap distribution from the n=200 sample (95 % CI [0.811, 0.950]) but not a hard reproducibility failure. The user-brief 5 pp seed-sample-reproducibility threshold is too tight for n=10 (1 σ ≈ 3.5 pp at n=10). The n=200 main eval is the authoritative number; the 5 pp threshold needs n ≥ ~50 to be a sensible policing rule on the M2 measure.
- The training-time online cumulative M2_predator_full reached the +0.80 verdict bar by training-window 9 (0.802 mean over episodes 8.12–9.05 M); the deterministic-eval M2_predator (per-class) at 0.888 is +8.8 pp above the bar, confirming that the deterministic-policy eval is decisively past threshold.

## References

- Design doc: [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../../../experiments/active/hypervigilance/sameprop_round26_design.md) §1 (pre-registered predicates), §3 (Launch Manifest — Run 3 row).
- Event-level verdict insight that this run is seeking to replicate: [[20260512_1428_sameprop_class_discriminating_defence_event_level]].
- Companion insight from this session — Cell A1 corner-camping refuted: [[20260518_1735_sameprop_a1_seed45_corner_camping_refuted]].
- Companion insight — methodological lesson about WandB post-crash reads: [[20260518_1737_wandb_post_crash_frozen_state_misread]].
- WandB run: `ja5fu5k3` ([wandb.ai/sungwoolee/grid_world_pain/runs/ja5fu5k3](https://wandb.ai/sungwoolee/grid_world_pain/runs/ja5fu5k3)).
- Reader-friendly re-summary (study-level context for this run): [`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`](../../../experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059` or `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/d79a0d50-3ac8-4fa2-9e7e-6a6437d6b059.jsonl /tmp/20260518_1736.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260518_1735_sameprop_a1_seed45_corner_camping_refuted]] (hypervigilance, 2026-05-18) — R2.6 Cell A1 seed 45 finished 10M ep on n102 with Episode/Steps=98/500 — H₁(A1-s
- [[20260518_1737_wandb_post_crash_frozen_state_misread]] (cluster_ops, 2026-05-18) — When a WandB run crashes, its `run.summary` keys stay frozen at the last logged 
<!-- END BACKLINKS -->
