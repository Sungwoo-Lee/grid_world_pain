---
id: 20260513_0017_mod_h_logging_gap_blocks_cka_precheck
date: 2026-05-13
time: "00:17"
folder: nmn_diagnosis
tags: [nmn, learned_lesson, meta, training_runner]
summary: "The R2 continual probe's modulator-engagement Mahalanobis check is unevaluable because the project never logs the raw modulator hidden vector — only summary statistics (mean, std). Three new WandB metrics requested: `modulator/mod_h_norm`, `Episode/Term_Predator`, `Episode/Occupancy_*`. Without `eval/h_mod_samples` hook + `mod_h_norm`, no CKA pre-check or representation-similarity test is possible on existing or future NMN runs."
related: ["20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin", "20260509_1410_nmn_temp_head_natural_target_3_to_5"]
session_origin: claude_code
session_label: "NMN R2 continual + 6-specialist analyzer verdict — first positive FiLM finding"
importance: medium
status: active
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# Modulator hidden-vector logging gap blocks the Mahalanobis / CKA pre-check on NMN runs

## Key conclusion

When the experiment-analyzer tried to run the design doc's formal modulator-engagement check on the R2 continual sister pair (a Mahalanobis test on the modulator's hidden representation across stages), it found the metric was unevaluable because **the project's training loop has never logged the raw modulator hidden vector** — only summary statistics (`modulator/temperature_mean`, `modulator/beta_uni_std`, `modulator/gamma_multi_mean`, etc.). The same gap blocks any CKA-style representation-similarity comparison between modulated and unmodulated representations. This is a recurring class of methodological-vs-instrumentation mismatch: the experimental design specifies a representation-level test, but the training infrastructure logs only scalar summaries. Three new WandB metrics requested by the analyzer to close this and related gaps; the primary one is a new `eval/h_mod_samples` hook that periodically dumps the raw modulator state.

## Evidence, measurements, facts

- **What the design doc asked for**: Mahalanobis distance between the modulator hidden state `h_mod` at stage-1 active and stage-3/5 active (after re-entering the active condition). If the modulator reuses the same subnetwork on each return, the Mahalanobis distance should be small relative to within-stage variance. The test is the formal version of the H₁c reusable-subnetwork prediction.
- **What's actually in the WandB logs**: scalar summaries only. The full list of currently-logged modulator fields (verified from R2 run summaries): `modulator/temperature_mean`, `modulator/temperature_max`, `modulator/beta_uni_std`, `modulator/beta_multi_std`, `modulator/beta_multi_mean`, `modulator/gamma_multi_std`, `modulator/gamma_multi_mean`, `modulator/gamma_uni_std`, `modulator/z_memory_mean`, `modulator/z_memory_std`, `modulator/grad_norm`. None of these is the raw hidden vector.
- **Three metrics requested** (added to `NMN_META_2x3_MIXTURE_PROBE.md` §8.2 / §8.3 by the analyzer):
  1. `modulator/mod_h_norm` — scalar L2-norm of the modulator's hidden state per logging interval (cheap pre-check; not a full CKA but a first-cut sanity signal).
  2. `Episode/Term_Predator` — per-episode counter for predator-caused termination (currently aggregated into `Episode/Term_Injury`; can't separate predator from danger-zone deaths).
  3. `Episode/Occupancy_*` — quadrant-level occupancy histograms, for explicit corner-camping detection (currently inferred only from `MeanDist*` averages).
- **What's also needed (not yet listed as a separate metric)**: a periodic `eval/h_mod_samples` hook that dumps the raw modulator hidden vector at fixed eval points, written as a WandB artifact (parquet). Without this, full CKA across stages is impossible regardless of the three scalar metrics above.
- **Scope**: this gap affects all NMN-modulated runs to date — R2 pair, the temp_clip rerun (P3/P4/P5), the heterogeneity sweep, and the future meta head-to-head.

## Decisions and actions

- The R2 verdict notes "Mahalanobis test is unevaluable" rather than "passed/failed" — see [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]] mechanism-check section.
- The 3 scalar metrics + 1 artifact hook are documented in `docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md` §8.2 / §8.3 as "Metrics Requested" follow-ups. Not blocking on the R2 verdict (which has H₁b + H₁c confirmed by survival-step metrics independently), but blocks the formal representation-level confirmation.
- This work routes through `feature-workflow` (developer + code-reviewer) when prioritized — the user decides timing. Suggested order: `mod_h_norm` first (cheapest, 1-line code change), then `Term_Predator` + `Occupancy_*` (env-side accounting), then the `eval/h_mod_samples` hook (needs an eval-time forward pass with logging).
- Cross-reference with sibling insight on H₁a predicate flaw: the project has now flagged **two** methodology-vs-implementation mismatches on the same probe (H₁a schedule asymmetry, H₁c logging gap). Worth adding a pre-launch checklist item to experiment-designer: "for every pre-registered predicate, is the required metric currently being logged?"

## Open questions and follow-ups

- Cost of full `eval/h_mod_samples` artifact dump: how often + how much storage. Order-of-magnitude estimate before scoping.
- Is the modulator's hidden state always a fixed-dim vector, or does it depend on the FiLM variant in use? Affects whether one schema works for all NMN configs.
- Should the existing scalar summaries be augmented with **per-stage** stratification (modulator/temperature_mean | stage=01_active vs stage=05_active), as a cheaper precursor to the artifact dump?
- Does sheeprl's PyTorch DreamerV3 (the bridge from session `268d07a3`) have the same logging gap, or does it expose richer modulator state by default?

## References

- Design doc: [`docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md`](../../../docs/experiments/active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md) §8.2 / §8.3 (Metrics Requested)
- Sibling insights: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]] (R2 verdict where this gap was discovered), [[20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric]] (sister methodology gap)
- Related: [[20260509_1410_nmn_temp_head_natural_target_3_to_5]] (prior modulator-internal observation — used summary stats, found the temp_clip[0.5, 5.0] target; benefited from the same logging that this insight finds insufficient for representation-level tests)
- Commit: `46dc0b1` (analysis docs)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md`.
