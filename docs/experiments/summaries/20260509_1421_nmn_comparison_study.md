---
title: "NMN Comparison Study — Week of 2026-05-07 → 2026-05-09 (heterogeneity sweep + temp-clip rerun)"
study: nmn_comparison
generated: 2026-05-09T14:21
window: "2026-05-07 → 2026-05-09"
status: snapshot
---

# NMN Comparison Study — Summary as of 2026-05-09 14:21 KST

> **One-paragraph summary.** This week's work tested whether a **modulated agent** (a "neuromodulator network", or NMN, with a FiLM gating head) can outperform a simpler **unmodulated baseline** at survival under noisy perception. Two follow-up experiments to the prior `v8` diagnosis were run. **Both produced negative results for the modulated agent** — it is not faster than the baseline at any noise profile tested. However, the second experiment overturned part of the first's interpretation: a configuration ceiling on the modulator's "temperature" output had been throttling it on the harder profiles. Removing the ceiling allowed the modulator to *partially* close the gap, but **not** to actually beat the baseline. The study leaves us with: (a) the simpler agent is genuinely hard to beat on this task at this scale; (b) the bottleneck is downstream of the modulator's temperature head; (c) a hidden 4-5 step seed-to-seed noise was discovered, meaning future single-seed comparisons in this regime are unreliable.

> **This is a snapshot.** Re-summaries should be written as new dated files in this folder, not by editing this one.

---

## 1. Study question

Both agents see the same noisy world. The **unmodulated** agent is a standard recurrent policy (LayerNorm + GRU + actor/critic head). The **modulated (FiLM)** agent has the same backbone plus an **extra "modulator head"** that — in principle — should learn to up- or down-weight different sensory channels depending on context (e.g., trust vision less when visual noise is high). The expected payoff is *better survival* under heterogeneous noise, where some sensory channels carry information and others don't.

**The puzzle**: a prior diagnosis (`v8`) found the modulated agent **does not** outperform the baseline. This week's two experiments tested two competing explanations for that null:

1. **The noise was wrong.** v8 used roughly uniform noise across all sensory channels — nothing for a *differential* gating mechanism to gate on. → **Experiment 1**: introduce heterogeneity.
2. **A configuration knob was binding.** The modulator's temperature output was clamped to a tight range that might have prevented it from doing its job. → **Experiment 2**: open the knob.

Performance is measured in **survival steps** (project rule), never cumulative reward.

---

## 2. Experiments completed this week

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0** *(prior anchor — not run this week)* | **v8 NMN diagnosis** | When sensory noise is added uniformly across all channels, does the modulated agent beat the unmodulated baseline? | One noise profile, modulated vs unmodulated, several gating-group sizes. | Modulated did **not** beat baseline (a clean null). Suggested either (a) the noise profile was too uniform to give the modulator a job, or (b) the architecture itself is the problem. | Defined the central puzzle. Two follow-ups proposed and run this week. |
| **1** | **Noise-heterogeneity sweep** *(2026-05-07 → 08, 10 cells)* | Does the modulated agent start beating the baseline when **some channels are very noisy and others are clean**, instead of v8's uniform noise? | 5 noise profiles ranging from "all channels equally noisy" → "one channel drowns in noise while others are pristine". Each profile run with both architectures. **Total noise across profiles was held constant — only the *spread* changed.** | **Modulated never wins.** It is consistently 5-13 survival steps **worse** than the baseline at every heterogeneity level. The gap shrinks as heterogeneity grows but never crosses zero. Side finding: the modulator's "temperature" output **saturated at its allowed ceiling (3.0)** on the high-heterogeneity profiles — i.e. the modulator was *trying* to engage further but the configuration was throttling it. | Closed the "v8 noise was just too uniform" hypothesis. Surfaced a **new candidate cause**: the v8 null might be a hyperparameter cap, not a structural failure. |
| **2** | **Temperature-ceiling rerun** *(2026-05-08 → 09, 5 cells)* | Was the v8 + Experiment-1 null caused by the throttling ceiling? **Raise the ceiling from 3.0 → 10.0 and see whether the modulated agent now beats (or matches) the baseline.** | A single config change: temperature ceiling 3.0 → 10.0. Re-ran the saturated profiles, with **3 seeds at one profile** to test whether the prior single-seed numbers were stable. | **Mixed: ceiling was a partial binder, not the whole story.** With the higher ceiling: the temperature output climbs to ~4.5–4.7 on the harder profiles (so the cap WAS binding before), and the modulated agent **closes most of the survival gap on two profiles (+6.4 and +8.0 steps over the v8/Experiment-1 baseline)** — it is no longer dramatically worse. **But the modulated agent still does not beat the baseline on any profile.** Also: the 3-seed test showed survival varies by **±4.4 steps** seed-to-seed, meaning **single-seed results in this regime are unreliable**. | Re-frames the v8 null as *partly* a hyperparameter mistake (ceiling too tight), but **not entirely** — even with the temperature head fully unleashed, the modulator can't beat a baseline that doesn't have one. The bottleneck is **downstream** of the temperature head. |

---

## 3. Where this leaves the study

- **The simpler unmodulated agent is, on this task and at this scale, hard to beat.** Two attempts at "rescuing" the modulated agent (give it a better-shaped problem; give it more output range) both failed to flip the sign.
- The remaining open question is **structural**: even when the modulator's temperature output is operating freely, the survival benefit is at most parity. The temperature head is **not** the bottleneck. Something further along the modulator's pipeline (the per-channel gates that turn the temperature into actual gating, or how the gated signal feeds the policy) is the next thing to redesign.
- A **hidden cost was discovered**: in this regime, single-seed results carry **±4-5 steps** of noise. Future runs at this scale should default to ≥3 seeds for any survival comparison smaller than ~10 steps. The first two experiments above were single-seed by design — at least two of their cells (the parent's "P3 = +0.08" anomaly and Experiment 2's "P5 = +3.28" near-miss) cannot be safely interpreted without replication.
- A **practical config fix** falls out of the work: the project's canonical FiLM agent config should raise its temperature ceiling from `[0.5, 3.0]` to `[0.5, 5.0]` going forward (the ceiling was found to be binding on harder noise profiles).

---

## 4. What's next (still pending decision)

1. **Replicate the +6.4 / +8.0 partial improvements with 3 seeds each** *(`experiment-designer` to author)* — before letting them anchor any redesign. The 3-seed P3 result already showed seed-noise can flip a single-seed sign.
2. **Architectural redesign of the modulator's downstream stages** *(`senior-developer` to plan, blocked on item 1)* — with the explicit constraint that the partial improvements seen in Experiment 2 must not regress, and with the temperature head explicitly **de-prioritised** in the redesign (it is operating freely and is not the bottleneck).
3. **Update the `v8` NMN performance-diagnosis doc** to reflect the partial-binder reframing, so the canonical diagnosis no longer reads "modulated agent is structurally degenerate at high heterogeneity".

---

## 5. Links to authoritative documents

### Design docs (the experiments themselves)

- [NMN_NOISE_HETEROGENEITY_SWEEP](../active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md) — Experiment 1, full design + Results / Analysis / Conclusions.
- [NMN_TEMP_CLIP_CEILING_RERUN](../active/hypervigilance/NMN_TEMP_CLIP_CEILING_RERUN.md) — Experiment 2, full design + Results / Analysis / Conclusions.

### Anchor diagnosis (the prior work this study extends)

- [NMN_PERFORMANCE_DIAGNOSIS_v8](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — the v8 diagnosis that posed the central puzzle; §6.3 P3 + P4 are the two follow-ups this week's work executed.

### Memory insights (per-finding rationale, rejected alternatives, replication caveats)

Each insight carries the full reasoning behind its conclusion and explicitly names what it supersedes / refines:

- [20260508_1426_v8_noise_bug_refuted](../../../.claude-memory/memories/nmn_diagnosis/20260508_1426_v8_noise_bug_refuted.md) — pre-experiment-1 audit closed a v8-flagged config bug as not present in current code.
- [20260508_1427_nmn_heterogeneity_sweep_design](../../../.claude-memory/memories/nmn_diagnosis/20260508_1427_nmn_heterogeneity_sweep_design.md) — Experiment 1 design rationale (matched-mean σ, R-gradient).
- [20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse](../../../.claude-memory/memories/nmn_diagnosis/20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse.md) — Experiment 1 verdict.
- [20260508_2004_profile_dependent_temp_saturation_mc_film](../../../.claude-memory/memories/nmn_diagnosis/20260508_2004_profile_dependent_temp_saturation_mc_film.md) — Experiment 1 mechanism finding (the ceiling-saturation that motivated Experiment 2).
- [20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4](../../../.claude-memory/memories/nmn_diagnosis/20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4.md) — Experiment 2 verdict, including the partial-binder reframing of Experiment 1.
- [20260509_1410_nmn_temp_head_natural_target_3_to_5](../../../.claude-memory/memories/nmn_diagnosis/20260509_1410_nmn_temp_head_natural_target_3_to_5.md) — Experiment 2 mechanism finding (the temperature head's natural target sits in `[3.0, 5.0)`; basis for the canonical-config fix).

### Working files (raw analyzer extractions, intermediate notes)

- [20260508_nmn_het_synthesis](../../../tmp/20260508_nmn_het_synthesis.md) + `tmp/20260508_nmn_het_extraction.json` — Experiment 1 raw data.
- [20260509_124500_nmn_tempceil10_analysis](../../../tmp/20260509_124500_nmn_tempceil10_analysis.md) + `tmp/20260509_nmn_tempceil10_extraction.json` — Experiment 2 raw data.

### Diary days covering this study

- [2026-05-08](../../diary/2026-05-08.md) — Experiment 1 launches (08:21–10:35 KST), analysis (19:54), Experiment 2 launches (20:34–20:45).
- [2026-05-09](../../diary/2026-05-09.md) — Experiment 2 analysis (12:45) + insight captures.

### Implementation commits relevant to this study

- `0ed4942` — Experiment 1 configs (5 env + 2 agent) + 10-row launch manifest.
- `fd9278a` — Experiment 2's two memory insights + indexes.
- (Experiment 2's design doc + new agent config are still uncommitted at the time of writing.)

---

## 6. Reading order if you have 10 minutes

1. **This document** (5 min) — gets you the headline finding and the open follow-ups.
2. The Experiment 1 design doc's §11–§12 (Conclusions) and the Experiment 2 design doc's §10–§11 — concrete numbers + the locked predicate verdicts.
3. Memory insight `20260509_1409` — the most-up-to-date verdict on the study's central question.

If you have 30 minutes, also read the v8 anchor diagnosis (§6.3) and the two mechanism insights (`20260508_2004` and `20260509_1410`) — these contain the architectural argument that motivates the redesign in §4 above.
