# _topic_index.md — `behavior_measures` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `behavior_measures` topic.

**Folder definition**: Behavior-measure platform & probes
**Insights**: 9
**Last updated**: 2026-07-10

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-07-10 | 16:35 | `20260710_1635_bush_hiding_metastable_dwell_measure` | Bush-hiding on the hard-predator task is metastable/intermittent (policy flickers run<->hide across checkpoints), NOT a stable phase; coarse 1M sampling ALIASES it into a false spike. Use dwell% (time-in-bush), not bush_use% (binary entry overstates via pass-throughs). Config-system era does NOT explain behavior: predator difficulty (easy->stable hide, hard->transient) + training phase do; at matched difficulty+depth old/new agents match. Extended 128-env training raises dwell 13->19.5% post-10M but stays bistable. Refines 20260630_1718. |
| 2026-07-10 | 16:34 | `20260710_1634_behavior_probe_eval_speed_parallel_batched_fdsafe` | Avoidance-probe eval sped ~8.8x (measured A/B/C): PARALLELISM is the dominant win (6.1x, per-process JAX startup ~14s dominates not rollout); batched single-process --batched rollout adds ~1.45x (parity 90/90 bit-exact; nnx.jit read-after-nnx.update trap = eager model() reads stale restored weights, scan body must be nnx.jit-wrapped). Parallel matplotlib render needs RLIMIT_NOFILE raised (Errno 24). Commits fa9ada0, 7ddacb7. |
| 2026-07-04 | 20:14 | [20260704_2014_deterministic_probe_significance_inflates](20260704_2014_deterministic_probe_significance_inflates.md) | A near-deterministic probe (tiny within-model variance) inflates p-values AND Cohen's d, so any trivial difference is "significant" — judge model comparisons by effect size + absolute magnitude + overlap, not p. Spatial spread (radius of gyration) is significant but a WEAK run-vs-hide discriminator vs the categorical bush-use signal. Rendered-math stats tutorial written. |
| 2026-07-04 | 20:13 | [20260704_2013_probe_rerun_stale_checkpoint_contamination](20260704_2013_probe_rerun_stale_checkpoint_contamination.md) | Re-running a probe sweep on an ADVANCED checkpoint writes new recordings alongside the old ones (separate models/<step>/ subdirs), and the auto-aggregating heatmap silently mixes checkpoints (720/360 recs). Fix: rm -rf the output-root before each re-eval (clean overwrite). |
| 2026-07-04 | 20:12 | [20260704_2012_noise_matched_frozen_probe](20260704_2012_noise_matched_frozen_probe.md) | A noise-trained agent must be probed with a config carrying its EXACT training perceptual_noise block (+ entity smell-std) and eval_obs_noise: training, else the clean probe mis-reads its off-regime behavior. Verified the eval injects the noise (agent obs != noise-free true_obs). Extends the sensory-match principle to perceptual noise. |
| 2026-06-30 | 17:18 | [20260630_1718_cover_use_late_emerging_run_vs_hide](20260630_1718_cover_use_late_emerging_run_vs_hide.md) | An EARLY model (randpred, 1.5M) RUNS/kites around the grid perimeter instead of diving into the bush — cover-seeking is a LATE-emerging skill, absent at 1.5M even as the reward curve flattens. Bush-use metrics discriminate run-vs-hide; movement metrics (time-moving, longest-chase) stay near-flat because the mature model hides intermittently; spatial spread (R_g) modestly higher for the runner. |
| 2026-06-30 | 17:17 | [20260630_1717_avoidance_reflex_needs_motion_and_olfaction](20260630_1717_avoidance_reflex_needs_motion_and_olfaction.md) | The flee-to-cover reflex requires BOTH an approaching/chasing animal AND a recognizable olfactory signature — zeroing the predator-identifiable smell channel abolishes fleeing even under chase (corrects an earlier 'motion-only' read). Threat discrimination is sustained-phase & damage-driven. Injury does NOT make avoidance earlier/preemptive. Injury heals over time; high injury alone never triggers hiding. |
| 2026-06-30 | 17:16 | [20260630_1716_foraging_hunger_timing_fixed_opening](20260630_1716_foraging_hunger_timing_fixed_opening.md) | Hunger changes the TIMING of foraging, not the path — fuller agents dither longer before going to food (departure-delay/satiety measure). Death floor at nutrition 15-20. The 'up-first' opening is a FIXED action prior (not an adaptive gradient probe) + gradient-following, and foraging works in all 4 directions. |
| 2026-06-30 | 17:15 | [20260630_1715_behavior_measure_study_method_and_tooling](20260630_1715_behavior_measure_study_method_and_tooling.md) | Interoceptive behavior-measure study: discover MEASURES of foraging/avoidance vs nutrition/injury by trajectory-first→metric-second on a frozen in-distribution agent; configs split core/explore; reusable scripts/behavior_measures/ heatmap tooling; deterministic probe → small initial-state jitter for statistics. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
