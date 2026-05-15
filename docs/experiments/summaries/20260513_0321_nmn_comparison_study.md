---
title: "NMN Comparison Study — Week of 2026-05-07 → 2026-05-13 (heterogeneity sweep + temp-clip rerun + vitality probe)"
study: nmn_comparison
generated: 2026-05-13T03:21
window: "2026-05-07 → 2026-05-13"
status: snapshot
---

# NMN Comparison Study — Re-summary as of 2026-05-13 03:21 KST

> **Headline.** The prior verdict from this study (2026-05-09 summary: "the modulator partially closes the gap but never beats the baseline") was a verdict about **steady-state environments only**. This week added a **schedule-changing environment** — the agent moves through five worlds in sequence, twice returning to a world it has trained on before — and on that benchmark **the modulated agent clearly beats the baseline** for the first time in the project's history. Per-stage survival on the two return-to-active stages is +107 and +132 steps over the unmodulated baseline, roughly 25× the seed-noise floor. Two specific hypotheses about *why* the modulator should help with schedule changes — that it remembers prior worlds better, and that it reuses a "core subnetwork" across worlds rather than rewriting itself each time — are both confirmed. A third hypothesis (that the modulator dampens the survival drop at world transitions) was inconclusive because its formal definition turned out to be malformed for the chosen schedule shape.
>
> This summary replaces (does not edit) the 2026-05-09 summary as the current view of the study; the older summary remains as a snapshot of what we believed then.

> **This is a snapshot.** Re-summaries should be written as new dated files, not by editing this one.

---

## 1. Study question

Both agents see the same world. The **unmodulated agent** is a standard recurrent policy (LayerNorm + GRU + actor/critic head). The **modulated (FiLM) agent** has the same backbone plus an extra "modulator head" — a small subnetwork whose only job is to up- or down-weight different sensory channels depending on context (for example, trust vision less when visual noise is high). The architectural bet is that this extra head should improve survival in environments where some sensory channels carry information and others don't, or where the world changes in a way that demands re-weighting.

Through 2026-05-09 the study answered **only one half** of this bet: under steady-state perception (one world the agent stays in for the entire training run), the modulator either does not help or marginally hurts. The reframing this week comes from asking the **other half**: what if the world itself changes during training, so that the modulator's job is not just to re-weight sensory inputs but to **preserve and restore prior policy behaviour when a previously-seen world returns**?

The puzzle in plain English: can the modulator *forget less catastrophically* than the baseline when the agent cycles through a sequence of worlds — and does it do this by reusing the same core subnetwork across the worlds, rather than building separate policies from scratch each time?

Performance is measured in **survival steps** (project rule), never cumulative reward.

---

## 2. Experiments completed

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0** *(prior anchors — already summarised on 2026-05-09; included here for context only)* | **Heterogeneity sweep + temperature-ceiling rerun** *(2026-05-07 → 09)* | Under steady-state noisy perception, can the modulator outperform the baseline at any noise heterogeneity level? Did a binding configuration knob hide an effect? | 5 noise-spread profiles × 2 architectures; then a follow-up where the modulator's temperature-output cap was loosened from 3.0 to 10.0. | Modulator **never beats** the baseline under steady-state perception. With the higher cap it closes most of the gap on the hardest profiles but still trails. Bottleneck is downstream of the temperature head. | Closed the "modulator just needs heterogeneous noise" hypothesis and the "modulator just needs a wider temperature range" hypothesis. Left open: does the modulator have a *different* job — one that only matters in non-steady-state environments? |
| **1** | **Vitality Round 1 — five-stage continual schedule** *(2026-05-09 launch; BUGGED)* | Does the modulator forget less when the agent cycles through five sequential worlds (active predator → passive predator → active → passive → active)? | Two agents (modulated, unmodulated), same hardware, on the schedule above. **Intended 5.1M episodes** per run. | **Bugged: the schedule's episode budget was 1000× too small.** Each run finished in ~10 minutes at 5,100 episodes total — far below the learning threshold. No learning, no result. Detected and killed on 2026-05-11. | Surfaced the bug; motivated the relaunch (Experiment 2). No verdict carried forward. |
| **2** | **Vitality Round 2 — continual sister pair** *(2026-05-11 → 12)* | Same question as Experiment 1, with the fixed schedule budget. | Two agents on the 5-stage `active → passive → active → passive → active` schedule for the full 5.1M episodes (modulator config and baseline config differ only by the presence of the modulator). | **First clearly-positive architecture-vitality finding in the project's history.** When the agent returns to the active world for the first time (stage 3), the modulated agent survives ~237 steps per episode versus the baseline's ~131 — a +107-step gap. When it returns a second time (stage 5), the gap widens to +132 (243 vs 111). For comparison: the project's seed-noise floor in this regime is ±4.4 steps, so this gap is roughly 25× the noise floor. The modulator's "2nd return ≥ 1st return" with no decay; the baseline shows monotone decline (−19.8 steps between returns). Two specific predictions — that the modulator forgets less (~65–72% less forgetting), and that it reuses a core subnetwork across worlds (clean sign-flip on the return-vs-return comparison) — are both confirmed. The third prediction (that the modulator dampens the survival drop at world transitions by ≥50%) is inconclusive because its formal definition was malformed for this specific schedule shape: in an active-then-passive cycle, half the transitions are *easier* than the prior stage and survival *rises* rather than dips, so a "half-the-dip" predicate is undefined on those transitions. The modulator's actual contribution lives in **recovery speed** at the harder transitions, not in dip-depth attenuation. Mechanism check: the modulator's temperature output swings sharply at three of four stage boundaries (sign-consistent: sharper on active stages, softer on passive), confirming it is mechanistically engaged. The deepest mechanism check — comparing the modulator's hidden representation across the two active-stage returns — was unevaluable because the project never logged the raw modulator hidden vector. | **Reframes the study's headline.** The 2026-05-09 verdict ("modulator never beats baseline") was a verdict about **steady-state environments only**. Under schedule changes, the same architecture clearly wins — and wins along exactly the dimensions predicted by the design (forgetting resistance + reusable-subnetwork). This is the strongest positive evidence for the modulator the project has produced. Caveat: single seed each; replication recommended. |
| **3** | **Vitality "specialist" battery — six single-world ceilings** *(2026-05-09 → 12)* | What survival can the *unmodulated* baseline reach on each of six individual worlds (active predator × matched / distinct / swapped olfactory mapping; passive predator × same three mappings)? These ceilings are the reference points for a future meta head-to-head. | Six independent unmodulated runs, one per world, 10M episodes each. | All six trained to convergence; 4 finished cleanly, 1 crashed at 99.7% (Python `rmdir` exception, terminal survival captured), 1 was user-terminated at 96.7% with a clean WandB flush. Ceiling table: passive_matched 491 steps > passive_swapped 488 > passive_distinct 457 > **active_swapped 402 > active_distinct 382 > active_matched 338**. **The unmod baseline finds the `active_swapped` world EASIER than the `active_matched` world** by 64 steps. Side finding: all three passive cells show a synchronised mid-training collapse around 5–7M episodes that recovers — possible systemic exploration/exploitation rebalance crisis. | **Refutes** the design-time intuition that the swapped-olfactory condition (predator smells like rabbit) is the hardest single-world condition. Swap is not load-bearing in isolation; it is the *combination* of swap and matched-context that the future meta head-to-head should test. Reframes the upcoming meta head-to-head as a **factorisation test** (can the modulator support two simultaneous policy modes whose individual ceilings differ?) rather than a "swap is the hardest condition" test. |

---

## 3. Where this leaves the study

- **The headline finding has flipped under a different benchmark.** Under steady-state perception the modulator does not beat the baseline (2026-05-09 verdict, unchanged). Under a schedule-changing environment the modulator clearly beats the baseline (~25× the seed-noise floor on return-to-active stages). The architecture has a job; that job is not steady-state denoising, it is **schedule-aware preservation and recovery of prior policy behaviour**.
- **Two of the three formal predictions about *why* the modulator should help with schedule changes are confirmed.** Catastrophic-forgetting resistance: modulator forgets 65–72% less than baseline on the two return-to-active stages. Reusable subnetwork: the modulator's second return matches or exceeds its first, with a clean sign-flip relative to the baseline's monotone decay. The third prediction (dampened transition dips) is inconclusive — the predicate was malformed for this schedule shape, not refuted by the data.
- **The mechanism evidence is partial.** The modulator's temperature output swings sharply at three of four stage boundaries with the expected sign pattern (sharper for active stages, softer for passive), which is consistent with the modulator being mechanistically engaged at boundaries. But the load-bearing representation-level test — comparing the modulator's hidden state across the two active-stage returns — is unevaluable because the project never logged the raw modulator hidden vector. This affects every NMN run to date, not just this week's.
- **The specialist ceiling table is not the table the design assumed.** The single-world specialist runs show `active_swapped` is *easier* than `active_matched` for the unmodulated baseline (by 64 steps). The future meta head-to-head was designed around "swap is the hardest condition", which is now refuted; the meta head-to-head needs its hypothesis section rewritten as a factorisation test before it can run.
- **Three logging / methodology gaps were surfaced on the same probe.** (a) Raw modulator hidden vector never logged → blocks Mahalanobis / CKA pre-checks across all NMN runs. (b) Predator-caused episode termination aggregated into a generic `Term_Injury` bucket → blocks predator-specific death analysis. (c) Quadrant occupancy never logged as a histogram → corner-camping detection has to be inferred indirectly from mean-distance stats. Three new metrics requested by the analyzer; one of them (`modulator/mod_h_norm`) is a 1-line change that should ship first.
- **A practical config fix from the prior summary still stands**: the canonical FiLM agent config should raise its temperature-clip ceiling from `[0.5, 3.0]` to `[0.5, 5.0]` (the ceiling was found to be binding on harder steady-state noise profiles, even though it is not the bottleneck).

---

## 4. What's next (still pending decision)

1. **3-seed replication of the Round-2 continual sister pair** *(`experiment-designer` to author)* — single-seed positive is strong but standard practice is to lock multi-seed before treating as load-bearing for a paper. Same hardware, same schedule, two more seeds each side.
2. **Add `modulator/mod_h_norm` + `eval/h_mod_samples` artifact hook to training** *(`senior-developer` to plan → `developer` to implement)* — the cheap scalar lets future runs do a Mahalanobis pre-check; the artifact hook lets future runs do a full CKA comparison across stages. Without these the strongest formal version of the reusable-subnetwork prediction stays unevaluable. Suggested order: scalar first (1-line code change), then the artifact (needs an eval-time forward pass with logging).
3. **Re-formalise the "half-the-dip stability gap" predicate** *(`experiment-designer` to author)* — current predicate is undefined on active→passive transitions because survival rises rather than dips. Candidate replacement: **recovery half-life** — episodes after a passive→active transition until survival recovers to 90% of the pre-passive steady state. Modulator should have shorter recovery half-life than baseline.
4. **Rewrite the meta head-to-head hypothesis section** *(`experiment-designer` to author, blocked on item 5)* — the original "swap is the hardest condition; modulator wins by handling it" framing is refuted by the specialist ceiling table. New framing: factorisation test ("can the modulator support two simultaneous policy modes when each has a different ceiling?").
5. **Build the `--mixture-mode` developer touch** *(`senior-developer` to plan → `developer` to implement)* — still blocks the meta head-to-head entirely. Once item 4 is rewritten and this lands, the meta head-to-head can launch.
6. **Update the canonical FiLM agent config** *(`developer`, low priority)* — temperature-clip ceiling from `[0.5, 3.0]` to `[0.5, 5.0]`, carried forward from the prior summary as still-unshipped.

---

## 5. Links

### Design docs (the experiments themselves)

- [NMN_NOISE_HETEROGENEITY_SWEEP](../active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md) — Anchor experiment 1 (heterogeneity sweep); full design + Results / Analysis / Conclusions; closed.
- [NMN_TEMP_CLIP_CEILING_RERUN](../active/hypervigilance/NMN_TEMP_CLIP_CEILING_RERUN.md) — Anchor experiment 2 (temp-clip rerun); full design + Results / Analysis / Conclusions; closed.
- [NMN_CONTINUAL_DOUBLE_RETURN_PROBE](../active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md) — Experiment 2 (continual sister pair); full design + §5.5 Results + §6 Conclusions; status `ANALYZED`.
- [NMN_META_2x3_MIXTURE_PROBE](../active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md) — Experiment 3 (specialist battery + future meta head-to-head); specialist-arm Results filled in §5.5, meta head-to-head still blocked on `--mixture-mode`.

### Anchor diagnosis (the prior work this study extends)

- [NMN_PERFORMANCE_DIAGNOSIS_v8](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — the v8 diagnosis that posed the original puzzle.

### Prior summary (re-summarised by this one)

- [20260509_1421_nmn_comparison_study](20260509_1421_nmn_comparison_study.md) — view as of 2026-05-09: covered heterogeneity sweep + temp-clip rerun; verdict was "modulator partially closes the gap but never beats the baseline (under steady-state perception)". This re-summary preserves that verdict for the steady-state regime and adds the schedule-changing-regime verdict.

### Memory insights (per-finding rationale, rejected alternatives, replication caveats)

From this re-summary's new window (2026-05-09 → 2026-05-13):

- [20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin](../../../docs/memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md) — R2 continual verdict: catastrophic-forgetting resistance + reusable-subnetwork hypotheses confirmed at ~25× seed-noise floor; first clearly-positive finding.
- [20260513_0015_active_swapped_geq_matched_reframes_meta](../../../docs/memory/memories/hypervigilance/20260513_0015_active_swapped_geq_matched_reframes_meta.md) — Specialist ceiling table refutes "swap is load-bearing"; reframes the meta head-to-head as a factorisation test.
- [20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric](../../../docs/memory/memories/nmn_diagnosis/20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric.md) — The "half-the-dip" predicate is malformed for this schedule shape; modulator's contribution lives in recovery speed, not dip-depth.
- [20260513_0017_mod_h_logging_gap_blocks_cka_precheck](../../../docs/memory/memories/nmn_diagnosis/20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md) — Raw modulator hidden vector never logged; blocks Mahalanobis / CKA tests across all NMN runs; 3 metrics + 1 artifact hook requested.

From the prior summary's window (2026-05-07 → 2026-05-09), still relevant:

- [20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse](../../../docs/memory/memories/nmn_diagnosis/20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse.md) — Heterogeneity sweep verdict; the steady-state framing.
- [20260508_2004_profile_dependent_temp_saturation_mc_film](../../../docs/memory/memories/nmn_diagnosis/20260508_2004_profile_dependent_temp_saturation_mc_film.md) — Temperature-saturation mechanism finding that motivated the temp-clip rerun.
- [20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4](../../../docs/memory/memories/nmn_diagnosis/20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4.md) — Temp-clip rerun verdict; the steady-state "partial binder" reframing.
- [20260509_1410_nmn_temp_head_natural_target_3_to_5](../../../docs/memory/memories/nmn_diagnosis/20260509_1410_nmn_temp_head_natural_target_3_to_5.md) — Temperature head's natural target sits in `[3.0, 5.0)`; basis for the canonical-config fix.

### Working files (raw analyzer extractions, intermediate notes)

- `tmp/20260512_nmn_analysis_summary.md` — Analyzer chain's working synthesis for the R2 + specialist verdict.
- Per-run parquet/csv histories under `tmp/` for the 2 continual + 6 specialist runs.

### Diary days covering this study

- [2026-05-08](../../diary/2026-05-08.md) — Heterogeneity sweep launches + analysis (prior window).
- [2026-05-09](../../diary/2026-05-09.md) — Temp-clip rerun analysis + vitality probe design + 10-cell battery launches.
- [2026-05-10](../../diary/2026-05-10.md) — Specialists in progress.
- [2026-05-11](../../diary/2026-05-11.md) — Round 1 bug detected; R2 sister pair relaunched on n106.
- [2026-05-12](../../diary/2026-05-12.md) — R2 sister pair finishes overnight; analyzer launched; orphan render-worker cleanup; 5 capture insights.
- [2026-05-13](../../diary/2026-05-13.md) — Capture commit + this re-summary.

### Implementation commits relevant to this study

- `0ed4942` — Heterogeneity sweep configs (prior window).
- `fd9278a` — Temp-clip rerun memory insights (prior window).
- `0684eda` — 8 training-done diary rows for the R2 + 6 specialist runs (this window).
- `46dc0b1` — Analysis docs (`NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md` §5.5 + `NMN_META_2x3_MIXTURE_PROBE.md` §5.5 + specialist ceiling table) filled by the experiment-analyzer agent.
- `e0d0704` — 5 memory insights captured for this window's findings.

---

## 6. Reading order if you have 10 minutes

1. **This document** (5 min) — the reframed headline + open follow-ups.
2. The R2 continual design doc's §5.5–§6 ([NMN_CONTINUAL_DOUBLE_RETURN_PROBE](../active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md)) — concrete per-stage survival numbers + the locked predicate verdicts.
3. Memory insight [20260513_0014](../../../docs/memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md) — the most-up-to-date verdict on the study's central question.

If you have 30 minutes, also read the meta-probe design doc's §5.5 (specialist ceiling table) and the two methodology insights — [20260513_0016 (predicate malformation)](../../../docs/memory/memories/nmn_diagnosis/20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric.md) and [20260513_0017 (logging gap)](../../../docs/memory/memories/nmn_diagnosis/20260513_0017_mod_h_logging_gap_blocks_cka_precheck.md) — these contain the methodological arguments that motivate follow-ups 2 and 3 in §4 above.
