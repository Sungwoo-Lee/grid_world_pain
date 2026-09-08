---
title: "Training-health and engineering-correctness audit of the modulation-site grid and its estimator-swapped twin"
topic: nmn_input_site_grid
status: active
created: 2026-09-07
last_updated: 2026-09-08
wandb_group: "nmn_input_site_grid, nmn_input_site_grid_gaenorm"
wandb_tag: "rppo_nmnsite_*_s42, rppo_nmngaenorm_*_s42"
develop_link: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
---

# Training-health and engineering-correctness audit of the modulation-site grid and its estimator-swapped twin

## 0. What this document is, and what it found

**The question.** Two batches of sixteen training runs each are being audited here. Both batches run
the same experiment: an agent that can carry a small second network — the "neuromodulator" — which
reads the agent's senses and continuously re-tunes the main policy network. The sixteen runs in each
batch cross **where** that re-tuning is applied (the sensory front-end, the memory cell, the
action-choosing head, the value-estimating head, or all four at once) with **what** the modulator is
allowed to read (all 27 sensory numbers, the 2 internal-body numbers only, or the 19
outside-world numbers only), plus one run with no modulator at all as a reference. The two batches
differ in exactly one setting: the recipe the trainer uses to turn a rollout of experience into the
numbers the agent learns from. The first batch uses the plain add-up-the-actual-rewards recipe; the
second blends the value head's own predictions into the estimate, which is less noisy. Everything
else — environment, architecture, learning rates, budget, random seed — is identical.

This document asks **only whether the machinery works**: is every run learning, is every switch
wired to the thing it claims to switch, is any number diverging, is anything unaccountably slow. The
scientific comparison between the sixteen arrangements, and between the two batches, is separate
work and is deliberately not attempted here.

**Where each batch stands.** **Both batches have now finished.** All thirty-two runs completed their
full ten million episodes — the first batch on 7 September, the second on 8 September. Every number
reported below is a genuine end-of-training value; nothing in this version of the document is a
snapshot of a run still in progress, and every figure the previous version marked provisional has
been recomputed on the full histories.

**The verdict: the machinery is working, in both batches.** Across all thirty-two completed runs
there is not a single not-a-number or infinite value in any logged quantity. Every run's survival
time rises from about 18 steps at the start to the mid-160s (first batch) or the low 170s (second
batch), still very slightly rising when the episode budget expires. Each of the thirty-two
configurations logs exactly the signals its own saved configuration says it should, and none that it
should not. In all thirty modulated runs the modulator received real training signal from beginning
to end — its gradient never once fell to the "disconnected" level, the smallest value seen anywhere
across either batch's complete history being about **seven times** the threshold the project's own
metrics guide calls vanishing (0.0071 in the first batch, 0.0076 in the second, and in both cases the
same cell: modulation written into the action-choosing head, driven by the two internal-body signals
only). **No run needs to be discarded, in either batch.**

**Both reference runs pass their pre-registered gates.** The first batch's unmodulated control
finished at **164.46** mean survival steps over its final million episodes, inside the 162.4–166.9
band its design registered in advance from five earlier seeds. The second batch's control finished at
**170.62**, inside the **169.9–171.2** band its own design registered — a materially narrower gate,
because the five reference seeds for the second recipe agreed with each other far more closely than
the first recipe's did. Both recomputed independently here from the runs' own logs; see §4.1 and
§5.1. Both grids therefore have a valid reference point, and the pre-registered precondition for
comparing them to each other is met.

**Five things worth the user's attention:**

1. **The odd, non-monotone shape of the re-tuning drift found in the first batch mostly reproduces in
   the second — six of seven site families match, one does not.** In both batches the sensory
   front-end's multiplier falls, bottoms out around the middle of training, and then climbs back; the
   memory-cell and value-head multipliers reach a plateau; the action-head multiplier keeps creeping
   down to the last logged step; the front-end and memory offsets flatten; and the additive offset at
   the action head is still sliding downward at the budget's end, fastest in the arm that feeds the
   modulator all 27 senses. The **exception** is the additive offset at the value head: in the first
   batch that signal was still sliding in the all-senses arm (−0.124 over the final tenth), whereas in
   the second batch the all-senses arm is essentially flat (−0.017) and the arm still sliding is the
   outside-world one (−0.089) instead. Because both grids share the same random seed, a reproduced
   shape is **weaker** evidence than a second seed would be — but it does show the shape survives
   changing the learning-signal recipe, which the single-batch data could not show. §5.4.
2. **The twin comparison changes when it is done at full length rather than early, in three ways —
   and all three are consequences of the two batches diverging slowly rather than of anything
   breaking.** The modulator-gradient agreement band was 0.77×–1.18× over an early matched window and
   is **0.70×–0.94× over the complete histories**, with the second batch below the first in all
   fifteen pairs rather than straddling. The reason is visible in the data: the first batch's
   modulator gradient **grows** over training in fourteen of fifteen arms, while the second batch's
   stays flat. Agreement between the two batches' re-tuning signals also loosens with length — 58 of
   60 signals agreed to within 0.14 early, but only 41 of 60 do at the end, and the six largest gaps
   are all offsets that the first batch drove much further negative (largest gap 0.72). And the one
   systematic clipping difference reported earlier — more windows running hot under the new recipe —
   **washes out and then reverses**: it is confined to roughly the first fifth of training, after
   which the second batch is the quieter of the two everywhere. §5.5–§5.7.
3. **The second batch is markedly quieter in the gradient tail than the first.** The first batch
   recorded 22 post-warm-up gradient excursions above 3.0 spread over thirteen of its sixteen runs,
   including a single 1030-magnitude event; the second recorded **6 events over five of sixteen runs,
   the largest being 8.4**. Nothing remotely like the 1030 appeared in the twin. That is consistent
   with the first batch's noisier learning-signal recipe and is not a defect in either. §5.3.
4. **The provenance-recording weakness is now final and confirmed at its worst.** The launcher's
   check for "were there uncommitted edits at launch time" recorded "unknown" in **all sixteen** runs
   of the second batch, against fifteen of sixteen in the first. The gate both experiment designs
   registered — every row must record "no uncommitted edits" — is **unsatisfiable from the recorded
   evidence in either batch**. §6.2 explains why this is very probably harmless here and what was
   checked instead; it is a reproduced defect, not a one-off.
5. **The estimator's effect on survival is uniform in sign across every one of the sixteen
   configurations, which is a health-relevant observation about the pair.** Reading each cell's mean
   survival over its final million episodes, the second batch finishes above the first in **16 of 16
   cells**, by +1.63 to +6.79 steps (mean +4.31). The spread of that effect across the sixteen cells
   (standard deviation 1.59 steps) is smaller than the seed-to-seed spread of the same measurement
   in the reference five-seed study (4.5 steps). Nothing in the data therefore requires the modulator
   machinery to be interacting with the learning-signal recipe in some cell-specific way — which is
   the precondition for treating the two grids as a matched pair. **This is an observation about 16
   paired draws at one seed, not a tested hypothesis**, and it is not a claim about which
   configuration is better. §5.8.

**Three standing flags carried forward, unchanged in substance.** (a) The project's metrics reference
still describes the front-end gain as a "pre-sigmoid" quantity needing to be squashed, which is not
true under the code these thirty-two runs used, and a reader who follows it will misread every
front-end number in both batches — §6.1. (b) The launcher's clean-working-tree flag is now
demonstrably unsatisfiable in both batches — §6.2. (c) The claim that switching a re-tuning site on
is a no-op at step 0 holds for the layer's average but not for individual units, which start at a
multiplier of 1.0 ± ~0.3 — §8.1, now confirmed a second time on the twin's complete data.

---

## 1. Scope

**In scope.** Numerical health (finite values, divergence, stalls), wiring correctness (does the
metric set match the configuration), gradient health of the modulator paths, gradient clipping
regime, throughput anomalies, and anything that looks like a code defect.

**Out of scope, deliberately.** Which write site or which input slice produces better survival or
more context-dependent behaviour; and whether the second batch's learning-signal recipe is better
than the first's. Those are the pre-registered analyses in [[NMN_INPUT_SITE_GRID]] and
[[NMN_INPUT_SITE_GRID_GAENORM]] and belong to a different session. Survival numbers appear here only
(a) as a liveness check that a run is learning at all, and (b) to confirm the control's reference
band.

**Discipline about what the numbers can support.**

- **There is one random seed per cell in both batches**, so no comparison between two arms in this
  document is a claim about which arm is better. A 5-seed study of this same configuration measured
  a seed-to-seed spread of about 4.5 survival steps, and most gaps in the tables below are inside
  that.
- **Both batches are now complete**, so nothing in this document is confounded by unequal training
  progress any more. The card each run landed on still confounds **throughput** comparisons (§7) and
  nothing else. The twin comparison in §5.5–§5.7 is now run over the **complete** histories; where a
  number differs from the earlier draft's early-window value, both are shown and the difference is
  explained by which stretch of training the window covered — a full-history mean and a
  first-eighth-of-training mean are different measurements, not a contradiction.
- Differences between the two batches are **engineering observations about how the learning-signal
  recipe interacts with the modulator**, not behavioural or scientific claims.

**Project rule observed.** Performance is read in survival steps, never cumulative reward.

---

## 2. What was audited

### 2.1 Batch A — the MC grid (finished)

Sixteen runs, WandB group `nmn_input_site_grid`, job type `pilot`, all seed 42, all on the
10×10 jump-attack environment config, all 10,000,000 episodes, all at code commit `a71f4471`.
One unmodulated control plus fifteen modulated arms crossing five write targets (sensory front-end /
memory cell / action head / value head / all four) with three input slices (all 27 sensory numbers
"ALL", the 2 interoceptive numbers "I", the 19 exteroceptive numbers "X").

The design document's Launch Manifest (§3 of [[NMN_INPUT_SITE_GRID]]) still shows every row as
`planned` with empty run-ID and node columns. The table below is what the runs' own metadata say,
recovered from `wandb/run-*/files/wandb-metadata.json` and each run's `models/provenance.json`.
**It is offered as evidence for whoever owns the manifest; this audit does not write to the
manifest.**

| Cell | Tag | WandB run ID | Local log dir | Node:GPU | Card | Episodes | % |
|---|---|---|---|---|---|---|---|
| T1_none (control) | `rppo_nmnsite_t1none_s42` | `anpfno02` | `wandb/run-20260907_045542-anpfno02` | 106:0 | RTX 3090 | 10,000,000 | 100% |
| T2_enc_ALL | `rppo_nmnsite_t2enc_ALL_s42` | `pom30693` | `wandb/run-20260907_050232-pom30693` | 106:1 | RTX 3090 | 10,000,000 | 100% |
| T2_enc_I | `rppo_nmnsite_t2enc_I_s42` | `9f8wx4b9` | `wandb/run-20260907_050233-9f8wx4b9` | 107:0 | RTX 3090 | 10,000,000 | 100% |
| T2_enc_X | `rppo_nmnsite_t2enc_X_s42` | `kir9fubn` | `wandb/run-20260907_050234-kir9fubn` | 107:1 | RTX 3090 | 10,000,000 | 100% |
| T3_rnn_ALL | `rppo_nmnsite_t3rnn_ALL_s42` | `3ed43b2i` | `wandb/run-20260907_050235-3ed43b2i` | 108:0 | RTX 3090 | 10,000,000 | 100% |
| T3_rnn_I | `rppo_nmnsite_t3rnn_I_s42` | `84964jnn` | `wandb/run-20260907_050235-84964jnn` | 108:1 | RTX 3090 | 10,000,000 | 100% |
| T3_rnn_X | `rppo_nmnsite_t3rnn_X_s42` | `ryt9z2lo` | `wandb/run-20260907_050237-ryt9z2lo` | 109:0 | RTX 3090 | 10,000,000 | 100% |
| T4_act_ALL | `rppo_nmnsite_t4act_ALL_s42` | `p6yc4akb` | `wandb/run-20260907_050238-p6yc4akb` | 109:1 | RTX 3090 | 10,000,000 | 100% |
| T4_act_I | `rppo_nmnsite_t4act_I_s42` | `qeesiize` | `wandb/run-20260907_050239-qeesiize` | 110:0 | RTX 3090 | 10,000,000 | 100% |
| T4_act_X | `rppo_nmnsite_t4act_X_s42` | `skmnxb2h` | `wandb/run-20260907_050239-skmnxb2h` | 110:1 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_ALL | `rppo_nmnsite_t5crt_ALL_s42` | `qug0fubt` | `wandb/run-20260907_050240-qug0fubt` | 111:0 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_I | `rppo_nmnsite_t5crt_I_s42` | `hi2ly2sh` | `wandb/run-20260907_050241-hi2ly2sh` | 111:1 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_X | `rppo_nmnsite_t5crt_X_s42` | `f5qlrnql` | `wandb/run-20260907_050242-f5qlrnql` | 112:0 | RTX 3090 | 10,000,000 | 100% |
| T16_quad_ALL | `rppo_nmnsite_t16quad_ALL_s42` | `b2cen70a` | `wandb/run-20260907_050243-b2cen70a` | 112:1 | RTX 3090 | 10,000,000 | 100% |
| T16_quad_I | `rppo_nmnsite_t16quad_I_s42` | `ldsttwlo` | `wandb/run-20260907_050243-ldsttwlo` | 113:0 | RTX 4090 | 10,000,000 | 100% |
| T16_quad_X | `rppo_nmnsite_t16quad_X_s42` | `hxey8k3h` | `wandb/run-20260907_050243-hxey8k3h` | 113:1 | RTX 4090 | 10,000,000 | 100% |

**Change since the previous version of this audit.** Ten of these sixteen were between 84% and 99%
complete when the first draft was written. All sixteen are now at exactly 10,000,000 episodes. Every
figure below is therefore end-of-training, and where a number moved as a run finished, the new value
is what is reported.

### 2.2 Batch B — the GAE_NORM twin (finished)

Sixteen runs, WandB group `nmn_input_site_grid_gaenorm`, job type `pilot`, all seed 42, same
environment config, same 10,000,000-episode budget, all at code commit `6695aa29`. Cell names match
Batch A exactly, so a cell name identifies a **pair**. **All sixteen have now completed the full
budget**; the percentages in the earlier draft (12.8–20.6%) are superseded.

The twin's Launch Manifest (§3 of [[NMN_INPUT_SITE_GRID_GAENORM]]) is also still showing every row
as `planned`. Reconstructed from the runs' own metadata:

| Cell | Tag | WandB run ID | Local log dir | Node:GPU | Card | Episodes | % |
|---|---|---|---|---|---|---|---|
| T1_none (control) | `rppo_nmngaenorm_t1none_s42` | `xvw6uzrs` | `wandb/run-20260907_155906-xvw6uzrs` | 101:0 | RTX 2080 Ti | 10,000,000 | 100% |
| T2_enc_ALL | `rppo_nmngaenorm_t2enc_ALL_s42` | `dtai9q6q` | `wandb/run-20260907_155906-dtai9q6q` | 101:1 | RTX 2080 Ti | 10,000,000 | 100% |
| T2_enc_I | `rppo_nmngaenorm_t2enc_I_s42` | `a0qrdefg` | `wandb/run-20260907_155914-a0qrdefg` | 102:0 | RTX 4090 | 10,000,000 | 100% |
| T2_enc_X | `rppo_nmngaenorm_t2enc_X_s42` | `9ovumjg0` | `wandb/run-20260907_155914-9ovumjg0` | 102:1 | RTX 4090 | 10,000,000 | 100% |
| T3_rnn_ALL | `rppo_nmngaenorm_t3rnn_ALL_s42` | `k3nkwafh` | `wandb/run-20260907_155909-k3nkwafh` | 103:0 | RTX 2080 Ti | 10,000,000 | 100% |
| T3_rnn_I | `rppo_nmngaenorm_t3rnn_I_s42` | `3om5z4jv` | `wandb/run-20260907_155910-3om5z4jv` | 103:1 | RTX 2080 Ti | 10,000,000 | 100% |
| T3_rnn_X | `rppo_nmngaenorm_t3rnn_X_s42` | `igrblp5d` | `wandb/run-20260907_155912-igrblp5d` | 104:0 | RTX 2080 Ti | 10,000,000 | 100% |
| T4_act_ALL | `rppo_nmngaenorm_t4act_ALL_s42` | `bo6t6y4m` | `wandb/run-20260907_155912-bo6t6y4m` | 104:1 | RTX 2080 Ti | 10,000,000 | 100% |
| T4_act_I | `rppo_nmngaenorm_t4act_I_s42` | `3whh6e86` | `wandb/run-20260907_155914-3whh6e86` | 105:0 | RTX 2080 Ti | 10,000,000 | 100% |
| T4_act_X | `rppo_nmngaenorm_t4act_X_s42` | `3jqmnf3a` | `wandb/run-20260907_155914-3jqmnf3a` | 105:1 | RTX 2080 Ti | 10,000,000 | 100% |
| T5_crt_ALL | `rppo_nmngaenorm_t5crt_ALL_s42` | `rinfx023` | `wandb/run-20260907_155915-rinfx023` | 106:0 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_I | `rppo_nmngaenorm_t5crt_I_s42` | `e1ty78id` | `wandb/run-20260907_155916-e1ty78id` | 106:1 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_X | `rppo_nmngaenorm_t5crt_X_s42` | `76krj6ri` | `wandb/run-20260907_155917-76krj6ri` | 114:0 | RTX 6000 Ada | 10,000,000 | 100% |
| T16_quad_ALL | `rppo_nmngaenorm_t16quad_ALL_s42` | `u69q0auc` | `wandb/run-20260907_155918-u69q0auc` | 114:1 | RTX 6000 Ada | 10,000,000 | 100% |
| T16_quad_I | `rppo_nmngaenorm_t16quad_I_s42` | `khcs2pxc` | `wandb/run-20260907_155920-khcs2pxc` | 114:2 | RTX 6000 Ada | 10,000,000 | 100% |
| T16_quad_X | `rppo_nmngaenorm_t16quad_X_s42` | `d4okzt4v` | `wandb/run-20260907_155922-d4okzt4v` | 114:3 | RTX 6000 Ada | 10,000,000 | 100% |

Every one of the sixteen reached 10,000,000 episodes; the final checkpoint line in each run's
captured output reads `Saving model at episode 10000004`–`10000097` (the small overshoot is the
trainer finishing the iteration it was in).

**Note the hardware difference from Batch A when reading throughput only.** Batch A ran entirely on
RTX 3090s and two RTX 4090s; Batch B is spread across 2080 Ti, 3090, 4090 and RTX 6000 Ada cards.
That made the twin's runs finish at very different wall-clock times, and it is why §7.2's speed table
cannot be compared cell-for-cell with §7.1's. It has no bearing on any gradient, loss or survival
figure, all of which are measured per episode.

### 2.3 The two batches differ in one setting — verified on the trainer's own saved configs

The user's claim is that the twin differs from the parent in exactly one config key,
`agent.return_mode`. **Independently re-verified here on three twin pairs** (`T16_quad_X`,
`T5_crt_ALL`, `T2_enc_I`) by a key-by-key diff of the **trainer-written resolved configuration** —
the config the trainer actually built the model from, not a fresh reload of the source YAML. All
three pairs give the same answer, with **264 keys on each side and exactly 5 differing**:

| Key | MC grid | GAE_NORM twin | Expected? |
|---|---|---|---|
| `agent.return_mode` | `MC` | `GAE_NORM` | ✅ the intended variable |
| `experiment.log_measures` | `[bush_dwell, survival_steps]` | `[bush_hiding, survival_steps]` | ✅ the commit-delta rename |
| `tag` | `rppo_nmnsite_…` | `rppo_nmngaenorm_…` | ✅ identity |
| `wandb.group` | `nmn_input_site_grid` | `nmn_input_site_grid_gaenorm` | ✅ identity |
| `wandb.name` | `rppo_nmnsite_…` | `rppo_nmngaenorm_…` | ✅ identity |

Nothing else differs. **No finding.**

**Extended to all sixteen twin runs at completion.** The three-pair, 264-key diff above is a
whole-config check on a sample. It is now supplemented by a targeted check on **every one of the
sixteen** twin runs' trainer-written configs: all sixteen record `return_mode: GAE_NORM`; all fifteen
modulated ones record a write-site block that matches their cell name exactly (encoder-only,
memory-only, action-head-only, value-head-only, or all four); all fifteen record
`temperature.enabled: false` and `rnn_mechanism: activation`; and the modulator's input sensor list is
`all` in the six ALL arms, `[Satiation, Interoceptive Nociception]` in the five narrow-input arms and
`[Extero Nociception, Olfaction, Collision, Visual]` in the five outside-world arms. The control's
config carries no modulation block at all. **Sixteen for sixteen, no exception.**

**The commit delta is confirmed harmless to training.** `a71f4471` is an ancestor of `6695aa29`, and
`git diff a71f4471 6695aa29 -- src/ train.py` is **empty** — the training code is byte-identical
between the two batches. The delta touches only evaluation-sweep configs, the evaluation default
config, docs, analysis scripts and a launcher script. The single change inside
`configs/evaluation/default.yaml` is the `bush_dwell` → `bush_hiding` rename of a measure name, and
that file's `during_training.enabled` is `false`, so it is not read on the training path at all.

### 2.4 Method

Everything below is read from **local files only** — no WandB web API was queried, per project
convention. Three independent sources were cross-checked, which matters because two of them share a
code path and the third does not:

1. Each run's WandB transaction log (`wandb/run-*/run-*.wandb`), parsed directly with the `wandb`
   package's on-disk datastore reader. This gives the complete metric history, not a downsampled web
   view. Note that different metric families are emitted in different history records, so every
   series is aligned to an episode count by forward-filling from the nearest preceding
   `Episode/Number` record rather than by position.
2. Each run's **trainer-written resolved configuration**
   (`results/JAX_RecurrentPPO/<run>/models/config.yaml`).
3. Each run's **runtime banner**, printed by the training process at startup and captured in
   `wandb/run-*/files/output.log`, which states the return mode, modulation type, input sensors and
   enabled sites as the constructed model reports them.

Working extractions: `tmp/nmnsite2/` (Batch A, and Batch B's superseded partial pass) and
`tmp/nmnsite3/` (Batch B re-extracted at completion, plus every cross-batch comparison in §5),
consolidated in `tmp/20260907_172348_nmnsite_audit_refresh.md` and
`tmp/20260908_gaenorm_final_audit.md`.

---

## 3. Wiring correctness — does each run log exactly what its configuration says?

**Result: exact match, thirty-two for thirty-two.** This is the strongest single piece of evidence
that the site-selection code is wired correctly, because a site that is enabled emits its own pair of
signals and a site that is disabled emits nothing at all — so the *set* of series present is a
direct readout of which sites the model actually built.

| Cell | Sites declared in the run's own saved config | Modulator series actually present | MC | GAE_NORM |
|---|---|---|---|---|
| T1_none | (no modulation block) | none at all | ✅ | ✅ |
| T2_enc_{ALL,I,X} | encoder only | `gamma_uni_*`, `beta_uni_*`, `gamma_multi_*`, `beta_multi_*`, `grad_norm` | ✅ | ✅ |
| T3_rnn_{ALL,I,X} | rnn only | `gamma_rnn_*`, `beta_rnn_*`, `grad_norm` | ✅ | ✅ |
| T4_act_{ALL,I,X} | actor only | `gamma_actor_*`, `beta_actor_*`, `grad_norm` | ✅ | ✅ |
| T5_crt_{ALL,I,X} | critic only | `gamma_critic_*`, `beta_critic_*`, `grad_norm` | ✅ | ✅ |
| T16_quad_{ALL,I,X} | encoder + rnn + actor + critic | all ten pairs above + `grad_norm` | ✅ | ✅ |

No arm in either batch logs a series for a site it did not enable. Every enabled site emits **both**
its mean and its per-unit standard deviation. `modulator/grad_norm` is present in all thirty
modulated runs and absent from both controls.

**The two absence checks pass in both batches.** `modulator/temperature_*` is absent from all
thirty-two runs (every config sets `temperature.enabled: false`), and `modulator/z_memory_*` is
absent from all thirty-two (that series belongs to the legacy gate-bias mechanism; every run here
uses `rnn_mechanism: activation`). Both are expected absences per the metrics reference §2.8.

**Non-modulator metric sets are identical across all thirty-two runs** — the same 59 keys in every
run of both batches, with no arm missing an episode or loss metric its siblings have and no arm
carrying an extra one. Re-verified on the twin's complete logs: sixteen runs, 59 non-modulator keys
each, empty extra-set and empty missing-set against the reference run in every case.

**Input slices and return mode verified from the runtime, not from the source YAML.** The startup
banner in each of the sixteen twin runs reports `Return Mode: GAE_NORM` and the sensor list the
constructed modulator was given: `[all]` for the six ALL arms,
`[Satiation, Interoceptive Nociception]` for the five I arms, and
`[Extero Nociception, Olfaction, Collision, Visual]` for the five X arms — matching the design's arm
table exactly, and matching the parent batch's banners cell for cell.

**Code prerequisites verified.** For the MC batch: `e1aab726` and `83b8140b` are both ancestors of
the launch commit `a71f4471`, and the diff of `src/` between `e1aab726` and `a71f4471` is empty. For
the twin: `a71f4471` is an ancestor of `6695aa29` and `src/` + `train.py` are unchanged between them
(§2.3). The one requirement that is *not* satisfied by the evidence, in either batch, is the
clean-working-tree flag — see §6.2.

---

## 4. Batch A (MC grid) — final numbers

### 4.1 Is every run actually learning? — Yes, all sixteen, to completion

No run flatlined, stalled, or diverged. Every run's survival time rises through the same shape: a
fast climb through the first million episodes, then a long slow improvement that is still very
slightly positive at the end.

Survival steps (the rolling 5,000-episode mean the trainer logs), averaged within episode bands:

| Cell | 0–1M | 1–2M | 2–5M | 5–9M | **final 1M** | s.d. within final 1M | final 200k |
|---|---|---|---|---|---|---|---|
| T1_none (control) | 81.5 | 146.1 | 158.4 | 163.1 | **164.46** | 3.9 | 162.90 |
| T2_enc_ALL | 66.0 | 147.5 | 161.6 | 167.3 | 168.44 | 3.6 | 166.95 |
| T2_enc_I | 88.0 | 146.4 | 155.2 | 160.7 | 163.56 | 3.5 | 164.46 |
| T2_enc_X | 92.7 | 152.5 | 162.1 | 166.4 | 167.61 | 3.8 | 168.34 |
| T3_rnn_ALL | 86.9 | 155.2 | 160.9 | 163.9 | 164.87 | 3.7 | 164.74 |
| T3_rnn_I | 71.7 | 146.5 | 157.3 | 163.7 | 166.93 | 3.1 | 166.49 |
| T3_rnn_X | 88.1 | 148.2 | 158.7 | 164.3 | 165.62 | 3.3 | 167.19 |
| T4_act_ALL | 83.4 | 152.5 | 162.1 | 166.0 | 166.87 | 3.4 | 166.76 |
| T4_act_I | 67.6 | 146.0 | 157.6 | 163.4 | 165.83 | 3.1 | 166.12 |
| T4_act_X | 84.1 | 146.9 | 156.4 | 162.3 | 165.17 | 3.4 | 165.92 |
| T5_crt_ALL | 83.1 | 145.3 | 157.5 | 163.5 | 165.43 | 3.3 | 167.55 |
| T5_crt_I | 86.7 | 143.6 | 155.0 | 161.6 | 163.53 | 3.2 | 164.60 |
| T5_crt_X | 83.5 | 146.9 | 158.9 | 164.4 | 165.74 | 3.1 | 166.23 |
| T16_quad_ALL | 82.5 | 152.6 | 161.9 | 167.4 | 169.76 | 3.2 | 170.30 |
| T16_quad_I | 77.2 | 146.0 | 154.6 | 160.8 | 163.04 | 3.4 | 162.34 |
| T16_quad_X | 75.5 | 151.0 | 163.8 | 168.4 | **170.58** | 3.2 | 170.61 |

**How to read the "final 1M" column, and how not to.** It is the mean of the rolling-window survival
series over each run's last million episodes; the s.d. column is the scatter of that rolling series,
**not** a seed-to-seed uncertainty. With one seed per cell there is no seed-to-seed uncertainty
available at all. The whole 16-arm spread is 163.0 to 170.6, i.e. 7.6 steps, against a
previously-measured 5-seed spread of about 4.5 steps for this configuration. Read that as "the arms
are all in the same performance neighbourhood", not as a ranking.

**Control gate confirmed (independently recomputed).** The control's final-million mean of
**164.46** falls inside the pre-registered **162.4–166.9** reference band from five earlier seeds of
the identical unmodulated configuration. The grid's reference point is sound.

**Losses.** Total loss, value loss, policy loss and entropy are finite and stable in every run. Total
loss sits at 0.104–0.114 in every arm at the end (control 0.108); value loss at 0.224–0.243 (control
0.233); entropy at −0.60 to −0.70 (control −0.66) with no sign of premature collapse toward zero.
Nothing separates the modulated arms from the control on any loss channel by more than the
run-to-run scatter.

**Zero non-finite values.** A full scan of every numeric value in every history record of all sixteen
transaction logs found **no NaN and no infinity**, in any metric, at any step.

### 4.2 Does the modulator receive gradient in every modulated arm? — Yes, in all fifteen, to the end

This was the brief's most-likely-defect. It is not present.

`modulator/grad_norm` is the L2 norm of the gradient of the loss with respect to the modulator's own
parameters, computed before clipping (`src/models/recurrent_ppo_trainer.py:385`). The metrics
reference calls anything below 0.001 vanishing.

| Cell | min ever | mean | max | first decile | last decile | share of points below 0.001 |
|---|---|---|---|---|---|---|
| T2_enc_ALL | 0.0362 | 0.1737 | 0.674 | 0.154 | 0.181 | 0% |
| T2_enc_I | 0.0327 | 0.1767 | 0.692 | 0.116 | 0.202 | 0% |
| T2_enc_X | 0.0313 | 0.1730 | 2.251 | 0.132 | 0.195 | 0% |
| T3_rnn_ALL | 0.0358 | 0.0863 | 0.165 | 0.073 | 0.093 | 0% |
| T3_rnn_I | 0.0233 | 0.1274 | 0.315 | 0.083 | 0.152 | 0% |
| T3_rnn_X | 0.0302 | 0.0582 | 0.113 | 0.053 | 0.062 | 0% |
| T4_act_ALL | 0.0173 | 0.0264 | 0.075 | 0.029 | 0.025 | 0% |
| T4_act_I | **0.0071** | 0.0600 | 0.203 | 0.044 | 0.060 | 0% |
| T4_act_X | 0.0102 | 0.0221 | 0.048 | 0.024 | 0.024 | 0% |
| T5_crt_ALL | 0.0138 | 0.0378 | 0.108 | 0.035 | 0.036 | 0% |
| T5_crt_I | 0.0114 | 0.0352 | 0.111 | 0.036 | 0.037 | 0% |
| T5_crt_X | 0.0088 | 0.0266 | 0.095 | 0.024 | 0.026 | 0% |
| T16_quad_ALL | 0.0567 | 0.1901 | 0.540 | 0.159 | 0.195 | 0% |
| T16_quad_I | 0.0328 | 0.1567 | 0.423 | 0.129 | 0.171 | 0% |
| T16_quad_X | 0.0368 | 0.1969 | 1.355 | 0.125 | 0.196 | 0% |

**The smallest value observed anywhere across all fifteen modulated runs and their now-complete
histories is 0.0071** — unchanged from the earlier draft, about **seven times** the vanishing
threshold. No site is disconnected, and no arm's modulator gradient decayed toward zero as training
went on: the last-decile mean is equal to or larger than the first-decile mean in **fourteen of
fifteen** arms, and in the fifteenth (the all-senses action-head arm) it is within 13% of it
(0.0292 → 0.0254).

**The two never-before-run sites are both live.** The action-head site (`gamma_actor_*`) and the
value-head site (`gamma_critic_*`) both carry gradient in every arm that enables them, and both move
their signals substantially over training (§4.3) — an independent confirmation that they are in the
computational graph, since a disconnected site's signals would stay pinned at their initialisation.

The action-head site carries the *smallest* modulator gradient of the four (mean 0.022–0.026 in the
ALL and X arms) — roughly an order of magnitude below the front-end site. That is a plausible
consequence of where it sits (a single hidden layer close to the output, downstream of everything
else), not a defect, and it is still 20× above the vanishing threshold.

### 4.3 Where the gain and offset signals ended up, and whether they were still moving

**What the numbers mean.** Each enabled site multiplies its layer by a per-unit gain (γ) and then
adds a per-unit offset (β). Under the FiLM code these logged values are the gains and offsets **as
applied** — there is no sigmoid squashing in between (verified in
`src/models/recurrent_ppo_network.py:218-263, 548, 553, 564`). A gain of 1.0 with an offset of 0.0
leaves the layer untouched.

**At the first logged point** — iteration 50, the earliest sample WandB holds — every site in every
arm has a mean gain between **0.94 and 1.07** and a mean offset between **−0.07 and +0.04**. That is
the identity arrangement, still essentially unmoved after 200 parameter updates. The caveat about the
per-unit spread is §8.1 and it is important.

#### 4.3.1 End-of-training values — these are the numbers a published figure should draw

Values are the mean of the last five logged points of each completed run.

**Mean gain (γ) at end of training:**

| Site | ALL | I | X | single-site range | four-site arm (ALL / I / X) | full range incl. four-site |
|---|---|---|---|---|---|---|
| encoder, stage 1 (`gamma_uni_mean`) | 0.713 | **0.375** | **0.869** | 0.375 – 0.869 | 0.596 / 0.613 / 0.690 | 0.375 – 0.869 |
| encoder, stage 2 (`gamma_multi_mean`) | 0.783 | 0.726 | 0.909 | 0.726 – 0.909 | 0.552 / 0.827 / 0.727 | 0.552 – 0.909 |
| memory cell (`gamma_rnn_mean`) | 0.459 | 0.420 | 0.662 | 0.420 – 0.662 | 0.717 / 0.629 / 0.745 | 0.420 – 0.745 |
| action head (`gamma_actor_mean`) | 0.508 | 0.667 | 0.683 | 0.508 – 0.683 | 0.500 / 0.717 / 0.691 | 0.500 – 0.717 |
| value head (`gamma_critic_mean`) | 0.316 | **0.203** | 0.619 | 0.203 – 0.619 | 0.350 / 0.261 / 0.391 | 0.203 – 0.619 |

**Mean offset (β) at end of training:**

| Site | ALL | I | X | four-site arm (ALL / I / X) | full range |
|---|---|---|---|---|---|
| encoder, stage 1 (`beta_uni_mean`) | −0.735 | **−1.083** | −0.639 | −0.651 / −0.608 / −0.649 | −1.083 … −0.608 |
| encoder, stage 2 (`beta_multi_mean`) | −0.224 | −0.354 | −0.164 | −0.222 / −0.202 / −0.152 | −0.354 … −0.152 |
| memory cell (`beta_rnn_mean`) | +0.007 | −0.002 | +0.004 | −0.076 / −0.030 / −0.182 | −0.182 … +0.007 |
| action head (`beta_actor_mean`) | **−1.827** | −0.529 | −0.793 | −1.048 / −0.787 / −0.805 | −1.827 … −0.529 |
| value head (`beta_critic_mean`) | **−1.817** | −0.825 | −1.562 | −1.071 / −1.009 / −0.834 | −1.817 … −0.825 |

**How these moved relative to the earlier draft.** The site ranges shifted only modestly as the ten
unfinished runs completed — encoder 0.39–0.89 → **0.375–0.909**, memory 0.43–0.74 → **0.420–0.745**,
action head 0.52–0.72 → **0.500–0.717**, value head 0.20–0.62 → **0.203–0.619**. The two extreme
offsets both grew slightly more negative: action head −1.63 → **−1.827**, value head −1.80 →
**−1.817**.

#### 4.3.2 Did the drift continue to the end, or flatten? — Mostly flatten; two signals did not

This is the question the earlier draft could not answer, and the answer is **site-dependent**. It
matters because a plateau means the modulator settled on a configuration, whereas a continuing slide
means the run's episode budget, not the optimisation, decided where the signal stopped.

The table gives, for each signal, the smoothed extremum, how far into the run it occurred, how much
the signal recovered afterwards, and the change over the final tenth of training.

| Site family | Behaviour at the end | Where the extremum sits | Change over final tenth |
|---|---|---|---|
| **encoder gains** (`gamma_uni`, `gamma_multi`) | **Bottomed out and partly reversed** — rising at the last logged point in every arm | 25–60% of the run in the single-site arms; 81–96% in the four-site arms | **+0.003 to +0.054** (upward) |
| **memory-cell gain** (`gamma_rnn`) | **Plateau** | 89–99% | −0.007 to +0.015 (flat) |
| **value-head gain** (`gamma_critic`) | **Plateau** | 27–100% | −0.014 to +0.012 (flat) |
| **action-head gain** (`gamma_actor`) | **Still creeping down**, slowly | 89–100% | −0.003 to −0.060 |
| **encoder / memory offsets** | Plateau | 59–100% | −0.032 to +0.019 (flat) |
| **action-head offset** (`beta_actor`) | **Still sliding** in the all-senses arm | 98–100% in every arm | **−0.197** (ALL), −0.024 to −0.069 elsewhere |
| **value-head offset** (`beta_critic`) | **Still sliding** in the all-senses arm | 86–100% | **−0.124** (ALL), −0.006 to −0.043 elsewhere |

Worked examples of the two ends of that spectrum:

- **Front-end gain, exteroceptive input** (`T2_enc_X`, `gamma_uni_mean`): fell to a minimum of
  **0.678 at 25%** of training, then climbed back to **0.850** at the end — a recovery of +0.172,
  and still rising at +0.030 per final decile. The "monotone downward drift" reported in the earlier
  draft was, for this family, the middle of a U-shape.
- **Action-head offset, all-senses input** (`T4_act_ALL`, `beta_actor_mean`): reached its minimum at
  **99%** of training and changed by **−0.197 over the final decile alone**, i.e. about **11% of its
  own final magnitude of −1.83 was added in the last tenth of the run**. This signal had **not**
  converged when the budget ran out.
- **Value-head offset, all-senses input** (`T5_crt_ALL`, `beta_critic_mean`): minimum at **100%**,
  **−0.124 over the final decile**, about 7% of its final magnitude of −1.81. Also not converged.

**Interpretation, kept narrow.** A FiLM gain is not identifiable on its own — if the modulator halves
a layer's pre-activation, the layer's own weights can double and the network is unchanged — so a
gain that ends below 1.0 is **not** by itself evidence that a layer is being silenced. The project's
metrics guide publishes no healthy band for a *linear* FiLM gain (the 1.0–3.0 band it does publish is
for the *old* pre-sigmoid quantity and does not apply here), so nothing here crosses a threshold.
Everything downstream looks normal: the value-head arms, where the gain drops furthest, have value
losses indistinguishable from the control's (0.230–0.240 vs 0.233) and survival inside the pack.

What still deserves a targeted check is the **combination** of a gain near 0.2–0.4 with an offset
near −1.0 to −1.8 at a layer that is then passed through a rectifier: if the layer's own
pre-activation is O(1), that combination pushes a large fraction of its units permanently negative,
i.e. permanently off. That would be a real sparsification of the action and value heads, and it would
be invisible in every metric currently logged. §9 requests the two cheap scalars that would settle
it. The fact that these two offsets were **still moving at the end** makes that request more, not
less, worth acting on before the next wave.

**Spread rises with drift.** The per-unit standard deviation of the gain roughly doubles-to-triples
over training at every site (e.g. front-end stage 1: 0.32–0.37 at the start → 0.73–1.29 at the end),
and the offset's spread rises further still — the value-head offset in the exteroceptive arm reaches
a spread of **2.35**. Rising spread alongside a falling mean is the signature of the modulator
learning to treat different units differently, which is what it is for. It is also, per the metrics
guide's own caveat, exactly the situation in which a mean alone is misleading.

### 4.4 Do the narrow-input arms behave differently? — Yes, consistently, and they are not degenerate

The five `I` arms give the modulator only two numbers to read: how full the agent is, and a smoothed
internal ache. The brief's worry was that two dimensions might not be enough to learn from, leaving a
modulator that emits a near-constant signal and receives no gradient.

**That worry is not borne out.** The narrow-input modulators are, if anything, *more* strongly driven
than the wide-input ones. Their own gradient is comparable or larger at every site (front-end 0.177
vs 0.174 ALL / 0.173 X; memory cell 0.127 vs 0.086 / 0.058; action head 0.060 vs 0.026 / 0.022; value
head 0.035 vs 0.038 / 0.027). Their gains move at least as far from identity as the wide-input arms'
— the two largest gain excursions in the whole batch, front-end down to 0.375 and value head down to
0.203, are both `I` arms. Their per-unit spread is not collapsed: it grows over training exactly as
the other arms' does, and in the front-end arms it is the largest in the batch (1.19–1.29).

**But there is a clean and consistent difference in the other direction.** At four of the five site
families, the `I` arm produces a markedly *quieter* total gradient than its ALL and X siblings. Final
numbers, over all post-warm-up logging windows (after the first 10% of updates):

**Percentage of post-warm-up logging windows whose in-window peak gradient exceeded the 0.5 clip ceiling:**

| Site family | ALL | I | X |
|---|---|---|---|
| encoder | 90.8% | **36.5%** | 79.3% |
| memory cell | 58.7% | **6.0%** | 76.3% |
| action head | 82.9% | **4.5%** | 80.1% |
| value head | 73.8% | 56.6% | 59.8% |
| all four sites | 35.3% | **4.5%** | 81.9% |
| *(control, for reference)* | *79.9%* | | |

Four of five families show the same sign and a large margin. That is a real, repeated engineering
observation, not a coin flip — but note that all five share the same random seed, so it is not five
*independent* confirmations either. The mechanistic reading is straightforward and testable: a
modulator driven by two slowly-varying body signals emits a smoother, lower-variance modulation than
one driven by nineteen fast-changing world signals, and a smoother modulation puts less
high-frequency energy into the gradient. **The same ordering reproduces in the GAE_NORM twin over
its complete runs, with the same single exception** — the narrow-input arm is quietest at the sensory
front-end, the memory cell, the action head and the all-four-sites configuration, and the family that
breaks the pattern is the value head in both batches (§5.3). That is a second,
estimator-independent showing of the pattern. It is still one seed per arm in both grids, so it is a
repeated observation, not two independent confirmations.

**Bottom line for the engineering question:** a two-dimensional modulator input is *not* degenerate
in this codebase. The modulator learns, receives gradient, and moves its outputs further from
identity than the wide-input arms do. What it produces is a lower-variance signal, which is a
property to be aware of, not a fault. Whether that lower variance is good or bad for the science is
out of scope here.

**The check this analysis still could not perform.** All of the above rests on aggregate statistics,
and an aggregate can hide a conditional behaviour: a modulator whose output varies strongly across
the 128 units but barely at all across time would produce exactly these numbers while being
functionally a fixed re-parameterisation of the layer. The currently-logged spread mixes the
across-unit and across-time components and cannot separate them. §9 requests the one extra scalar per
site that would.

### 4.5 Does modulation push the total gradient into the clip? — No; every arm sits well below the ceiling

The trainer clips the global gradient norm at **0.5** (`train.py:1176-1183`, with
`agent.max_grad_norm: 0.5` confirmed in all sixteen runs' own saved configs). The logged
`loss/grad_norm` is measured **before** clipping.

The reference point is the completed return-mode study ([[return_mode_cmp_10M]]), where the arms that
failed to learn sat 150–300× above this ceiling and were clipped on essentially every update.
**Nothing resembling that happens here.**

Post-warm-up (after the first 10% of updates), full completed runs:

| Cell | mean grad norm | 99th pct | max windowed mean | % of windows with mean above 0.5 | vs. ceiling |
|---|---|---|---|---|---|
| T1_none (control) | 0.264 | 0.372 | 0.500 | 0.00% | 0.53× |
| T2_enc_ALL | 0.306 | 0.436 | 0.519 | 0.12% | 0.61× |
| T2_enc_I | 0.250 | 0.283 | 0.335 | 0.00% | 0.50× |
| T2_enc_X | 0.291 | 0.362 | 0.464 | 0.00% | 0.58× |
| T3_rnn_ALL | 0.254 | 0.331 | 0.592 | 0.24% | 0.51× |
| T3_rnn_I | 0.207 | 0.236 | 0.256 | 0.00% | 0.41× |
| T3_rnn_X | 0.271 | 0.394 | 0.473 | 0.00% | 0.54× |
| T4_act_ALL | 0.272 | 0.401 | 0.467 | 0.00% | 0.54× |
| T4_act_I | 0.196 | 0.259 | 0.512 | 0.12% | 0.39× |
| T4_act_X | 0.265 | 0.376 | 0.496 | 0.00% | 0.53× |
| T5_crt_ALL | 0.259 | 0.435 | 0.534 | 0.12% | 0.52× |
| T5_crt_I | 0.239 | 0.372 | 0.441 | 0.00% | 0.48× |
| T5_crt_X | 0.250 | 0.351 | 10.526 | 0.12% | 0.50× |
| T16_quad_ALL | 0.249 | 0.304 | 0.358 | 0.00% | 0.50× |
| T16_quad_I | 0.211 | 0.235 | 0.251 | 0.00% | 0.42× |
| T16_quad_X | 0.299 | 0.375 | 0.444 | 0.00% | 0.60× |

Every arm's typical gradient is roughly **half** the ceiling; every arm's 99th percentile is below
it; and the fraction of logging windows whose *average* exceeds the ceiling is at most 0.24%
anywhere, including the control. All sixteen sit inside the metrics reference's healthy band of
0.01–1.0, and no arm comes near the 5.0 warning level. (The single 10.5 windowed mean in `T5_crt_X`
is the 1030 spike of §4.8 propagating into that window's average.)

**Adding a modulator does change the clipping regime, but only at the margin.** The widest-input arms
run about 10–16% hotter than the control (front-end ALL 0.306 and four-site X 0.299, against the
control's 0.264) and the narrow-input arms run 5–26% cooler. That is a shift within the same regime,
not a change of regime.

**Warm-up clipping is heavy and universal, including in the control.** In the first 10% of updates
every run, control included, records windowed mean gradient norms of 0.27–0.42 with in-window peaks
of 9.5–32.7. That is the normal opening transient of this trainer and it resolves in every arm.

### 4.6 Is the all-four-sites arm unstable? — No

| Measure | T16_quad_ALL | T16_quad_I | T16_quad_X | single-site range | control |
|---|---|---|---|---|---|
| post-warm-up mean grad norm | 0.249 | 0.211 | 0.299 | 0.196 – 0.306 | 0.264 |
| distinct post-warm-up gradient excursions above 3.0 | **0** | 1 | 4 | 0 – 3 | 1 |
| non-finite values | 0 | 0 | 0 | 0 | 0 |
| modulator gradient (mean) | 0.190 | 0.157 | 0.197 | 0.022 – 0.177 | — |
| gains at end of training | all in 0.35–0.72 | all in 0.26–0.83 | all in 0.39–0.75 | 0.203 – 0.909 | — |

The four-site arm under the all-senses input has the **fewest** post-warm-up gradient excursions of
any arm in the batch (zero) and the lowest fraction of windows touching the clip ceiling among the
wide-input arms (35.3% vs 59–91% for the single-site ALL arms). Its modulator carries the largest
gradient in the batch, which is what four active sites should look like. The two mild elevations are
the four-site X arm's mean gradient (0.299, the second-highest in the batch, still 0.60× the ceiling)
and its four post-warm-up excursions, the most of any arm in raw count — though its largest, 18.6, is
smaller than the largest excursion in two single-site arms.

**A caveat the numbers cannot resolve.** Two of the three four-site arms are also the two arms on
different hardware (node 113, RTX 4090). Hardware does not affect gradient magnitudes, so the
stability conclusion stands, but it does affect §7.1.

### 4.7 Nothing else looks like an engineering problem

**No recompilation, no host-device stall.** Each run prints `JIT compiling train_iteration…` exactly
**once** and never again. The wall-clock interval between consecutive logged rows has a median of
7.0–9.5 s in every run, a 99th percentile of 13.4–18.7 s, and a maximum of 21.8–34.9 s (the periodic
checkpoint-and-render step). **No run has a single gap exceeding 4.3× its own median.** There is no
stall signature anywhere.

**No suspicious exact-zero or exact-constant series.** A scan of every logged numeric series in all
sixteen runs (excluding the counters `_step`, `iteration`, `_runtime`) found **zero** series whose
minimum equals its maximum. No metric is pinned at its initialisation value, which is the signature a
disconnected component would leave.

**No warnings or errors** in any run's captured output.

### 4.8 The 1030 gradient spike — it stayed a one-off, and only one new comparable event appeared

**Status of the outlier: unchanged through completion.** The value-head arm on the exteroceptive
input (`T5_crt_X`) recorded a single in-window peak gradient of **1030.3** at episode ≈9,716,000
(97% of its run) — roughly 4,000× its own typical value of 0.25. It appears in exactly **two
consecutive logged rows** because the logging window is a rolling buffer, so it is **one iteration**,
not a period. It did not recur in the remaining 3% of the run. Nothing else moved: the value loss,
total loss and modulator gradient are all at their normal values immediately before, during and
after. The clip did its job — an update of norm 1030 against a 0.5 ceiling is scaled by a factor of
0.0005, i.e. it was effectively a null update — and this run finished all 10,000,000 episodes at
165.74 mean survival steps, mid-pack.

**Did anything comparable appear in the ten Batch A runs that were still training when the previous
version of this audit was written?** **No.** Exactly one new
post-warm-up excursion above 3.0 appeared anywhere as those runs completed: `T2_enc_I` recorded a
peak of **6.6** at episode ≈9,996,000, in the final 0.04% of its run. That is an ordinary member of
this trainer's heavy tail. **Nothing above 20 appeared, and nothing remotely like 1030.**

**Complete final census of post-warm-up excursions above 3.0**, with location:

| Cell | events | peaks and where |
|---|---|---|
| T16_quad_X | 4 | 3.2 @ 47%; 11.0 @ 69%; 3.8 @ 76%; **18.6 @ 98%** |
| T5_crt_X | 3 | 9.3 @ 91%; **1030.3 @ 97%**; 4.0 @ 100% |
| T4_act_I | 3 | 4.8 @ 27%; **32.6 @ 60%**; 3.5 @ 92% |
| T3_rnn_ALL | 2 | **33.1 @ 44%**; 25.5 @ 79% |
| T4_act_ALL | 2 | 4.2 @ 46%; 3.9 @ 80% |
| T2_enc_ALL | 1 | 6.1 @ 78% |
| T2_enc_I | 1 | 6.6 @ 100% *(new since the earlier draft)* |
| T3_rnn_X | 1 | 3.0 @ 91% |
| T4_act_X | 1 | 6.1 @ 88% |
| T5_crt_ALL | 1 | 3.1 @ 51% |
| T5_crt_I | 1 | 5.4 @ 17% |
| T16_quad_I | 1 | 4.0 @ 79% |
| T1_none (control) | 1 | 4.9 @ 82% |
| T2_enc_X, T3_rnn_I, T16_quad_ALL | 0 | — |

**A second, stronger reason to think the 1030 was not a defect, available only now.** The fifteen
modulated architectures and the control were re-run for ten million episodes each under a different
return mode, from the same seed and the same initialisation. That replication produced **six**
post-warm-up excursions above 3.0 in total, the largest being **8.4** (§5.3). Nothing within two
orders of magnitude of the 1030 recurred in a sixteen-run, 160-million-episode replication of the
same modulation code. If the 1030 were caused by the FiLM machinery it had every opportunity to
reappear.

**Is the 1030 a defect?** Probably not, and here is the discipline: excursions above 3.0 after
warm-up occur in **thirteen of the sixteen runs, including the unmodulated control**, so this trainer
has a heavy-tailed gradient distribution as a background property. The 1030 is an extreme draw from a
distribution that already produces 33s elsewhere, in an arm whose value head is being FiLM-modulated
with a gain that has fallen to 0.62 and an offset that has fallen to −1.56. It is a plausible tail
event, not a demonstrated bug. **What would settle it**: per-update gradient norms (currently only
the per-iteration mean and the window peak are kept), so one could see whether the excursion was one
minibatch or the whole iteration; and the value-target magnitudes for that iteration, to see whether
an outlier return drove it. Both are §9 requests. **Suggestive but not conclusive**: the GAE_NORM
twin, whose value targets are variance-reduced by construction, has produced **zero** excursions
above 3.0 in its first ~1.3–2.0M episodes across all sixteen runs (§5.2) — but the MC batch produced
only one excursion in that same early window too, so the comparison is not yet informative.

---

## 5. Batch B (GAE_NORM twin) — final numbers

**What this section can and cannot say.** All sixteen twin runs completed 10,000,000 episodes, so
every figure here is end-of-training. It remains a health and correctness section: which arm is
better is not asked, and the one place survival appears as more than a liveness check is §5.1's
control gate and §5.8's paired estimator observation, which is framed as an engineering property of
the *pair*, not as a result about either grid.

### 5.1 Is every run learning, with no NaN or infinity? — Yes, all sixteen, to completion

**Zero non-finite values** across all sixteen complete transaction logs, in any metric, at any step.
Every run prints `JIT compiling train_iteration…` exactly **once** and never again, has **zero**
warning, error or traceback lines in its captured output, and has **zero** exact-constant numeric
series (a series pinned at its initialisation value is the signature a disconnected component would
leave; there are none).

Survival steps (the rolling 5,000-episode mean the trainer logs), averaged within episode bands:

| Cell | 0–1M | 1–2M | 2–5M | 5–9M | **final 1M** | s.d. within final 1M | final 200k | late slope (steps per M, 5M→10M) |
|---|---|---|---|---|---|---|---|---|
| T1_none (control) | 87.8 | 152.2 | 162.9 | 168.6 | **170.62** | 3.2 | 170.66 | +0.82 |
| T2_enc_ALL | 87.5 | 154.9 | 164.9 | 170.1 | 172.65 | 3.3 | 174.45 | +0.96 |
| T2_enc_I | 95.9 | 154.3 | 163.4 | 168.1 | 170.35 | 3.2 | 169.64 | +0.89 |
| T2_enc_X | 87.7 | 156.3 | 167.0 | 171.7 | **173.16** | 3.3 | 173.51 | +0.69 |
| T3_rnn_ALL | 97.6 | 155.3 | 162.4 | 166.1 | **167.59** | 3.4 | 168.17 | +0.44 |
| T3_rnn_I | 88.2 | 153.4 | 162.6 | 167.4 | 170.22 | 3.3 | 171.31 | +1.08 |
| T3_rnn_X | 100.5 | 155.1 | 164.1 | 168.9 | 170.77 | 3.4 | 171.50 | +0.77 |
| T4_act_ALL | 95.8 | 155.2 | 163.8 | 167.8 | 168.90 | 3.2 | 168.71 | +0.62 |
| T4_act_I | 79.4 | 150.8 | 161.4 | 166.6 | 169.22 | 3.2 | 169.70 | +0.99 |
| T4_act_X | 84.3 | 152.1 | 162.3 | 168.1 | 170.32 | 3.2 | 170.80 | +0.85 |
| T5_crt_ALL | 84.0 | 149.3 | 162.1 | 168.3 | 169.80 | 3.3 | 170.35 | +0.77 |
| T5_crt_I | 83.1 | 148.4 | 161.4 | 168.3 | 169.38 | 3.5 | 170.26 | +0.64 |
| T5_crt_X | 81.5 | 151.3 | 162.7 | 167.9 | 169.97 | 3.3 | 170.12 | +1.02 |
| T16_quad_ALL | 97.0 | 156.2 | 164.8 | 169.7 | 172.18 | 3.2 | 173.17 | +1.06 |
| T16_quad_I | 91.6 | 152.1 | 162.5 | 167.7 | 169.02 | 3.1 | 169.17 | +0.66 |
| T16_quad_X | 73.5 | 153.7 | 165.6 | 171.2 | 172.22 | 3.5 | 173.19 | +0.67 |

Every run follows the same shape as Batch A's: a fast climb through the first million episodes, then
a long slow improvement whose slope over the second half of training is **+0.44 to +1.08 survival
steps per million episodes** and is positive in all sixteen. Nothing flatlined, stalled, diverged or
turned over.

**Control gate — PASSED, recomputed independently here.** The unmodulated control's mean survival
over its final million episodes is **170.62 steps**, inside the pre-registered **169.9 – 171.2** band
that [[NMN_INPUT_SITE_GRID_GAENORM]] §4.1 fixed in advance from five earlier seeds of the identical
unmodulated `GAE_NORM` configuration. The gate is not sensitive to the averaging window: the final
500,000 episodes give **171.03** and the final 200,000 give **170.66**, both also inside. For
comparison the parent grid's control gate also passes on both windows (164.46 and 162.90, band
162.4–166.9). **Both grids have a valid reference point, so the design's precondition for attempting
the cross-grid analysis at all is satisfied.**

This gate deserves the emphasis the design gave it. The twin's reference band is **1.3 steps wide**
against the parent's 4.5, because the five `GAE_NORM` reference seeds agreed with each other far more
tightly than the five `MC` ones did. A control landing at 168.5 would have failed here and passed
there. It landed at 170.62, 0.26 above the five-seed reference mean of 170.36 — comfortably inside,
not scraping the edge.

**How to read the rest of the column, and how not to.** The whole 16-arm spread of the final-million
means is 167.6 to 173.2, i.e. **5.6 steps**. Batch A's caveat was that its spread (7.6 steps) sat
against a 4.5-step five-seed noise band, so the arms were "all in the same neighbourhood". Here the
arithmetic is different and whoever does the outcome analysis should notice it: the five-seed noise
band for this recipe is only 1.3 steps wide, so a 5.6-step spread is **not** obviously swallowed by
seed noise. That is a live question for the pre-registered analysis and this audit does not answer
it — it is flagged only so nobody carries Batch A's "all in the same neighbourhood" sentence across
to Batch B by habit. With one seed per cell, this document offers no ranking either way.

**Losses.** Total loss 0.094–0.099 (control 0.099), value loss 0.202–0.209 (control 0.209), policy
loss −0.0009 to −0.0012, entropy −0.476 to −0.619 (control −0.476), all finite and stable, with the
control inside the pack on every channel. The entropy term flattens by the third decile of training
in every arm and stays flat to the end — no run shows a runaway drive toward zero entropy. Batch B's
entropy sits nearer zero than Batch A's (−0.48 vs −0.67 in the two controls), which is a batch-level
property of the estimator, is stable rather than progressive, and is not a health finding.

### 5.2 Does the modulator receive gradient in all fifteen modulated arms? — Yes, to the end, and it does not fade to nothing

| Cell | min ever | mean | max | first decile | last decile | last/first | share of points below 0.001 |
|---|---|---|---|---|---|---|---|
| T2_enc_ALL | 0.0272 | 0.1353 | 0.315 | 0.1401 | 0.1330 | 0.95 | 0% |
| T2_enc_I | 0.0260 | 0.1565 | 0.689 | 0.1357 | 0.1476 | 1.09 | 0% |
| T2_enc_X | 0.0298 | 0.1390 | 0.388 | 0.1378 | 0.1354 | 0.98 | 0% |
| T3_rnn_ALL | 0.0323 | 0.0604 | 0.123 | 0.0582 | 0.0640 | 1.10 | 0% |
| T3_rnn_I | 0.0212 | 0.1126 | 0.358 | 0.0889 | 0.1300 | 1.46 | 0% |
| T3_rnn_X | 0.0234 | 0.0487 | 0.116 | 0.0480 | 0.0486 | 1.01 | 0% |
| T4_act_ALL | 0.0100 | 0.0223 | 0.062 | 0.0294 | 0.0213 | **0.73** | 0% |
| T4_act_I | **0.0076** | 0.0478 | 0.170 | 0.0428 | 0.0484 | 1.13 | 0% |
| T4_act_X | 0.0116 | 0.0183 | 0.063 | 0.0229 | 0.0172 | **0.75** | 0% |
| T5_crt_ALL | 0.0097 | 0.0331 | 0.102 | 0.0291 | 0.0326 | 1.12 | 0% |
| T5_crt_I | 0.0079 | 0.0329 | 0.121 | 0.0304 | 0.0310 | 1.02 | 0% |
| T5_crt_X | 0.0082 | 0.0194 | 0.142 | 0.0225 | 0.0192 | 0.85 | 0% |
| T16_quad_ALL | 0.0534 | 0.1450 | 0.371 | 0.1552 | 0.1463 | 0.94 | 0% |
| T16_quad_I | 0.0336 | 0.1338 | 0.446 | 0.1270 | 0.1500 | 1.18 | 0% |
| T16_quad_X | 0.0330 | 0.1574 | 0.521 | 0.1243 | 0.1806 | 1.45 | 0% |

**The smallest value observed anywhere across all fifteen modulated runs' complete histories is
0.0076** — unchanged from the early-life draft, about **seven and a half times** the 0.001 the metrics
reference calls vanishing, and in the same cell (`T4_act_I`) that holds the parent batch's minimum of
0.0071. No site is disconnected in any arm, including the two that had never been run before this
week.

**Does it fade? — Not to any level that matters, but the two batches differ in direction.** In Batch
A the modulator gradient *grew* over training: the last-decile mean was equal to or above the
first-decile mean in fourteen of fifteen arms. In Batch B it is **flat overall** — nine of fifteen
arms up, six down, median ratio 1.02. The six that decline are led by the two action-head arms
(0.73× and 0.75×), and the parent's single declining arm was also an action-head arm (0.87×), so the
action-head site is where the modulator's training signal weakens in **both** batches. Even there
nothing approaches disconnection: the smallest last-decile mean anywhere in the twin is **0.0172**,
seventeen times the vanishing threshold, with 0% of points below it in every arm.

### 5.3 Gradient against the clip ceiling, and the excursion census

The trainer clips the global gradient norm at **0.5**, and the logged `loss/grad_norm` is measured
before clipping. Post-warm-up here means after the first 10% of each run's logged rows.

| Cell | mean grad norm | 99th pct | max windowed mean | % of windows with mean above 0.5 | % of windows whose in-window **peak** exceeds 0.5 | largest peak | excursions above 3.0 |
|---|---|---|---|---|---|---|---|
| T1_none (control) | 0.192 | 0.298 | 0.330 | 0.00% | 24.5% | 1.3 | 0 |
| T2_enc_ALL | 0.246 | 0.311 | 0.355 | 0.00% | 70.2% | 1.3 | 0 |
| T2_enc_I | 0.218 | 0.271 | 0.295 | 0.00% | 32.0% | 1.6 | 0 |
| T2_enc_X | 0.240 | 0.387 | 0.516 | 0.06% | 44.4% | **8.4** | 1 |
| T3_rnn_ALL | 0.200 | 0.385 | 0.469 | 0.00% | 27.1% | 1.5 | 0 |
| T3_rnn_I | 0.178 | 0.213 | 0.228 | 0.00% | **1.7%** | 0.8 | 0 |
| T3_rnn_X | 0.189 | 0.297 | 0.334 | 0.00% | 16.5% | 1.0 | 0 |
| T4_act_ALL | 0.191 | 0.323 | 0.391 | 0.00% | 18.4% | 4.8 | 2 |
| T4_act_I | 0.162 | 0.229 | 0.256 | 0.00% | **2.5%** | 3.2 | 1 |
| T4_act_X | 0.237 | 0.424 | 0.512 | 0.06% | 60.7% | 1.7 | 0 |
| T5_crt_ALL | 0.220 | 0.393 | 0.534 | 0.06% | 50.5% | 1.5 | 0 |
| T5_crt_I | 0.219 | 0.375 | 0.469 | 0.00% | 51.9% | **6.4** | 1 |
| T5_crt_X | 0.176 | 0.265 | 0.297 | 0.00% | 9.7% | 1.3 | 0 |
| T16_quad_ALL | 0.205 | 0.276 | 0.353 | 0.00% | 23.6% | 1.6 | 0 |
| T16_quad_I | 0.180 | 0.223 | 0.247 | 0.00% | **4.7%** | 2.6 | 0 |
| T16_quad_X | 0.231 | 0.350 | 0.399 | 0.00% | 25.8% | **7.6** | 1 |

Every arm's typical gradient is between 0.32× and 0.49× the ceiling; every arm's 99th percentile is
below it; and the fraction of logging windows whose *average* exceeds the ceiling is at most 0.06%
anywhere, including the control. All sixteen sit inside the metrics reference's healthy band of
0.01–1.0, and nothing comes near the 5.0 warning level as a windowed mean.

**The narrow-input ordering of §4.4 reproduces, with the same single exception.** The arm fed only
the two internal-body signals produces the quietest gradient of its three siblings at the sensory
front-end (32.0% of windows touching the ceiling, against 70.2% and 44.4%), the memory cell (1.7% vs
27.1% and 16.5%), the action head (2.5% vs 18.4% and 60.7%) and the all-four-sites configuration
(4.7% vs 23.6% and 25.8%) — four of five families, the same four as in Batch A, and the family that
breaks the pattern is the value head in **both** batches. That is the mechanism §4.4 proposed
(a modulator driven by two slowly-varying body signals emits a smoother modulation, which puts less
high-frequency energy into the gradient) surviving a change of learning-signal recipe. It remains one
seed per arm in both batches, so it is a repeated observation rather than five independent
confirmations.

**Excursion census — the twin is markedly quieter in the tail.** Post-warm-up peaks above 3.0, with
location:

| Cell | events | peaks and where |
|---|---|---|
| T2_enc_X | 1 | 8.4 @ 95% |
| T4_act_ALL | 2 | 4.8 @ 35%; 3.4 @ 86% |
| T4_act_I | 1 | 3.2 @ 30% |
| T5_crt_I | 1 | 6.4 @ 98% |
| T16_quad_X | 1 | 7.6 @ 48% |
| the other eleven runs, control included | 0 | — |

**Six events across five of sixteen runs, largest 8.4.** Batch A recorded **22 events across thirteen
of sixteen runs**, with a largest of **1030.3**. So the heavy gradient tail this trainer shows under
the first learning-signal recipe is substantially thinner under the second, and **nothing remotely
like the 1030 event appeared anywhere in the twin**. Warm-up peaks are the same in both batches
(9.1–32.7), so this is a property of the post-warm-up regime, not of initialisation. Read together
with §4.8 this makes the 1030 look even more like an extreme draw from a heavy-tailed distribution
that belongs to the `MC` estimator, rather than a defect in the modulation code — the same fifteen
modulated architectures, differing only in return mode, produced no comparable event in ten million
episodes each.

### 5.4 Where the gain and offset signals ended up, and whether the parent's drift shape reproduced

#### 5.4.1 End-of-training values

Each enabled site multiplies its layer by a per-unit gain (γ) and then adds a per-unit offset (β);
under the FiLM code these are the values **as applied**, with no sigmoid in between. A gain of 1.0
with an offset of 0.0 leaves the layer untouched. Values below are the mean of the last five logged
points of each completed run.

**Mean gain (γ) at end of training:**

| Site | ALL | I | X | single-site range | four-site arm (ALL / I / X) | full range incl. four-site |
|---|---|---|---|---|---|---|
| encoder, stage 1 (`gamma_uni_mean`) | 0.654 | 0.491 | 0.641 | 0.491 – 0.654 | 0.677 / 0.575 / 0.568 | 0.491 – 0.677 |
| encoder, stage 2 (`gamma_multi_mean`) | 0.594 | 0.542 | 0.734 | 0.542 – 0.734 | 0.514 / 0.656 / 0.635 | 0.514 – 0.734 |
| memory cell (`gamma_rnn_mean`) | 0.516 | 0.458 | 0.576 | 0.458 – 0.576 | 0.684 / 0.603 / 0.681 | 0.458 – 0.684 |
| action head (`gamma_actor_mean`) | 0.684 | 0.612 | **0.771** | 0.612 – 0.771 | 0.537 / 0.733 / 0.674 | 0.537 – 0.771 |
| value head (`gamma_critic_mean`) | 0.340 | 0.211 | 0.624 | 0.211 – 0.624 | 0.340 / **0.160** / 0.411 | **0.160** – 0.624 |

**Mean offset (β) at end of training:**

| Site | ALL | I | X | four-site arm (ALL / I / X) | full range |
|---|---|---|---|---|---|
| encoder, stage 1 (`beta_uni_mean`) | −0.572 | −0.751 | −0.834 | −0.582 / −0.591 / −0.627 | −0.834 … −0.572 |
| encoder, stage 2 (`beta_multi_mean`) | −0.210 | −0.307 | −0.180 | −0.231 / −0.213 / −0.202 | −0.307 … −0.180 |
| memory cell (`beta_rnn_mean`) | −0.011 | −0.026 | +0.013 | +0.048 / +0.097 / +0.042 | −0.026 … +0.097 |
| action head (`beta_actor_mean`) | **−1.108** | −0.731 | −0.654 | −0.709 / −0.576 / −0.607 | −1.108 … −0.576 |
| value head (`beta_critic_mean`) | **−1.118** | −0.919 | −1.003 | −0.889 / −1.050 / −0.570 | −1.118 … −0.570 |

**Comparison with Batch A, at the level of range.** The gains land in a similar neighbourhood
(Batch A 0.203–0.909, Batch B 0.160–0.771), though the twin's are compressed at the top and reach
slightly lower at the bottom. The offsets are the channel that differs: Batch A's most negative
offset is **−1.827** and Batch B's is **−1.118**, and every one of the six largest twin-pair
disagreements in §5.6 is an offset that Batch A drove further negative. Whatever the first recipe's
noisier learning signal does, one of its consequences is that the modulator's additive channel
travels roughly 50% further from its starting point.

**Per-unit spread.** At the first logged point the per-unit standard deviation of the gain is
**0.29–0.39** and of the offset **0.28–0.35** in every twin arm — matching Batch A and
reproducing the §8.1 caveat exactly. By the end the gain's spread has grown to 0.33–1.14 and the
offset's to 0.32–1.70, the largest being the value-head offset in the outside-world arm. Batch A
reached 2.35 on that same signal. Rising spread alongside a falling mean is the modulator learning to
treat units differently, which is what it is for — and it is also the situation in which a mean alone
misleads.

#### 5.4.2 Does the parent's drift shape reproduce? — In six of seven site families, yes; in one, no

This is the check the user most wanted, and it is worth being precise about what it can prove. **The
two grids share seed 42**, so a matched pair of cells starts from bit-identical weights. Reproducing
a shape across the pair therefore shows the shape survives changing the learning-signal recipe; it
does **not** show the shape survives changing the seed, which is the stronger and still-untested
claim. A *failure* to reproduce would have been the more informative outcome, and there is one.

| Site family | Batch A (MC) behaviour | Batch B (GAE_NORM) behaviour | Reproduced? |
|---|---|---|---|
| **encoder stage-1 gain** (`gamma_uni`) | falls, bottoms out at 25–60% (single-site) / 81–96% (four-site), then climbs back; final tenth **+0.003 to +0.054** | falls, bottoms out at **43–62%** in five of six arms, then climbs back by +0.043 to +0.078; final tenth **+0.006 to +0.017** in those five (the sixth, the four-site exteroceptive arm, is flat at −0.002) | ✅ **yes**, and in the twin the reversal reaches two of three four-site arms too |
| **encoder stage-2 gain** (`gamma_multi`) | same U-shape, milder | minima at 84–100%, recovery ≤ 0.023, final tenth −0.019 to +0.016 | ◐ **partly** — the twin's stage-2 gain plateaus rather than reversing |
| **memory-cell gain** (`gamma_rnn`) | plateau; extremum at 89–99%; final tenth −0.007 to +0.015 | plateau; extremum at **85–100%**; final tenth **−0.021 to +0.007** | ✅ yes |
| **value-head gain** (`gamma_critic`) | plateau; extremum at 27–100%; final tenth −0.014 to +0.012 | plateau; extremum at **41–100%**; final tenth **−0.013 to +0.012** | ✅ yes, near-identical bands |
| **action-head gain** (`gamma_actor`) | still creeping down; final tenth −0.003 to −0.060 | still creeping down; final tenth **−0.010 to −0.035** | ✅ yes |
| **encoder / memory offsets** | plateau; final tenth −0.032 to +0.019 | plateau; final tenth **−0.031 to +0.007** | ✅ yes |
| **action-head offset** (`beta_actor`) | **still sliding**, worst in the all-senses arm: **−0.197**, others −0.024 to −0.069 | **still sliding**, worst in the all-senses arm: **−0.118**, others −0.002 to −0.044 | ✅ yes — same arm, ~60% of the magnitude |
| **value-head offset** (`beta_critic`) | **still sliding**, worst in the all-senses arm: **−0.124**, others −0.006 to −0.043 | still sliding, but **worst in the outside-world arm: −0.089**; the all-senses arm is nearly flat at **−0.017**, and one four-site arm is moving *upward* (+0.037) | ❌ **no** — the "which arm has not converged" answer does not carry across |

**Worked examples, the same two the parent used, plus the one that broke.**

- **Front-end gain, exteroceptive input** (`T2_enc_X`, `gamma_uni_mean`): falls to a minimum of
  **0.564 at 43%** of training, then climbs back to **0.642** at the end — a recovery of +0.078,
  still rising at +0.011 per final decile. Batch A's counterpart bottomed at 0.678 at 25% and
  recovered +0.172. **Same shape, same direction, roughly half the amplitude.**
- **Action-head offset, all-senses input** (`T4_act_ALL`, `beta_actor_mean`): minimum at **96%**,
  changed by **−0.118 over the final decile**, about **11% of its own final magnitude of −1.108**.
  Batch A's counterpart moved −0.197, also 11% of *its* final magnitude of −1.827. The percentage is
  the same to the digit; the absolute is smaller only because the twin's offset travelled less far.
  **This signal had not converged in either batch when the budget ran out.**
- **Value-head offset, all-senses input** (`T5_crt_ALL`, `beta_critic_mean`): minimum at **100%** but
  changed by only **−0.017 over the final decile**, 1.5% of its final magnitude of −1.118 — against
  Batch A's −0.124 and 7%. The signal that had *not* settled in Batch A had essentially settled here,
  while the twin's outside-world arm (`T5_crt_X`) slid **−0.089**, 8.9% of its final −1.003, which
  Batch A's outside-world arm did not do (−0.043).

**What to take from the one failure.** With one seed per cell in each grid, a difference of this size
between two arms of the same family sits inside what a single seed could produce — the parent grid's
own five value-head-offset signals spanned −0.006 to −0.124 with no apparent logic to the ordering.
The honest reading is therefore: **the family-level shapes (which sites plateau, which reverse, which
are still moving at the budget's end) reproduce; the arm-level attribution of "this exact arm has not
converged" does not, and should not have been leaned on.** What survives both batches is the
conclusion that matters operationally — **the offsets at the action and value heads are the signals
that have not converged at 10,000,000 episodes**, which is the §9 item-2 request's justification and
is unaffected by which input slice is worst.

### 5.5 Twin comparison at full length — modulator gradient

Both batches are now complete, so the comparison can use each pair's **entire** history rather than a
matched early window. Both readings are shown, because they answer different questions and they do
not agree.

| Cell | MC mean (full) | GN mean (full) | ratio | MC first→last decile | GN first→last decile |
|---|---|---|---|---|---|
| T2_enc_ALL | 0.1737 | 0.1353 | 0.78 | 0.154 → 0.181 | 0.140 → 0.133 |
| T2_enc_I | 0.1767 | 0.1565 | 0.89 | 0.116 → 0.202 | 0.136 → 0.148 |
| T2_enc_X | 0.1730 | 0.1390 | 0.80 | 0.132 → 0.195 | 0.138 → 0.135 |
| T3_rnn_ALL | 0.0863 | 0.0604 | **0.70** | 0.073 → 0.093 | 0.058 → 0.064 |
| T3_rnn_I | 0.1274 | 0.1126 | 0.88 | 0.083 → 0.152 | 0.089 → 0.130 |
| T3_rnn_X | 0.0582 | 0.0487 | 0.84 | 0.053 → 0.062 | 0.048 → 0.049 |
| T4_act_ALL | 0.0264 | 0.0223 | 0.85 | 0.029 → 0.025 | 0.029 → 0.021 |
| T4_act_I | 0.0600 | 0.0478 | 0.80 | 0.044 → 0.060 | 0.043 → 0.048 |
| T4_act_X | 0.0221 | 0.0183 | 0.83 | 0.024 → 0.024 | 0.023 → 0.017 |
| T5_crt_ALL | 0.0378 | 0.0331 | 0.88 | 0.035 → 0.036 | 0.029 → 0.033 |
| T5_crt_I | 0.0352 | 0.0329 | **0.94** | 0.036 → 0.037 | 0.030 → 0.031 |
| T5_crt_X | 0.0266 | 0.0194 | 0.73 | 0.024 → 0.026 | 0.023 → 0.019 |
| T16_quad_ALL | 0.1901 | 0.1450 | 0.76 | 0.159 → 0.195 | 0.155 → 0.146 |
| T16_quad_I | 0.1567 | 0.1338 | 0.85 | 0.129 → 0.171 | 0.127 → 0.150 |
| T16_quad_X | 0.1969 | 0.1574 | 0.80 | 0.125 → 0.196 | 0.124 → 0.181 |

**The band did not hold, and the reason is legible.** Over the early matched window (episodes
1.00M–1.25M) the earlier draft measured 0.77×–1.18× with five pairs above 1.0 and nine below. Over
the **complete** histories the band is **0.70×–0.94×, median 0.83×, with the twin below the parent in
all fifteen pairs**. Those two statements are both correct and describe different stretches of
training. The mechanism is in the last two columns: the parent's modulator gradient **grows** through
training (last decile above first in fourteen of fifteen arms), while the twin's is **flat** (median
last/first 1.02). The two start together — they must, they share an initialisation — and then the
parent's climbs away.

**Is that a health problem? No.** Every twin arm remains between 17× and 180× the vanishing
threshold; no arm's share of points below 0.001 is anything other than 0%; and a 0.83× median ratio
between two different learning-signal recipes is a small difference by the standards of this trainer,
whose per-arm modulator gradients already span an order of magnitude within a single batch (0.018 to
0.157). It is worth recording because it is systematic — fifteen of fifteen in the same direction is
not a coin flip — and because it makes any future statement of the form "the modulator is equally
well-driven under either estimator" need the qualifier "for the first fifth of training".

Over the same complete histories, the grid-mean post-warm-up **total** gradient is **0.255 (MC) vs
0.205 (GAE_NORM)**, i.e. the twin runs about 20% cooler overall. The earlier draft's "3% difference"
was measured over episodes 200k–1.25M, where the two are indeed nearly identical (0.351 vs 0.362).

### 5.6 Twin comparison at full length — the re-tuning signals

Sixty signals are comparable (fifteen modulated arms × their enabled gain/offset pairs). Comparing
each pair's **end-of-training** value (mean of the last five logged points):

| Agreement threshold | signals within it |
|---|---|
| |GN − MC| ≤ 0.05 | 24 / 60 |
| ≤ 0.10 | 35 / 60 |
| **≤ 0.14** | **41 / 60** |
| ≤ 0.20 | 50 / 60 |
| ≤ 0.30 | 55 / 60 |
| ≤ 0.50 | 57 / 60 |

The earlier draft, comparing at episodes 1.00M–1.25M, found 58 of 60 within 0.14. At the end of
training only **41 of 60** are. **Agreement loosens with training length**, which is what two
optimisation trajectories that start identical and are driven by different advantage estimates should
be expected to do.

The twelve largest disagreements are dominated by one channel and one direction:

| Cell | signal | MC | GAE_NORM | difference |
|---|---|---|---|---|
| T4_act_ALL | action-head offset | −1.827 | −1.108 | **+0.719** |
| T5_crt_ALL | value-head offset | −1.817 | −1.118 | **+0.698** |
| T5_crt_X | value-head offset | −1.562 | −1.003 | +0.558 |
| T16_quad_ALL | action-head offset | −1.048 | −0.709 | +0.339 |
| T2_enc_I | encoder stage-1 offset | −1.083 | −0.751 | +0.333 |
| T16_quad_X | value-head offset | −0.834 | −0.570 | +0.264 |
| T2_enc_X | encoder stage-1 gain | 0.869 | 0.641 | −0.228 |
| T16_quad_X | memory-cell offset | −0.182 | +0.042 | +0.225 |
| T16_quad_I | action-head offset | −0.787 | −0.576 | +0.211 |
| T4_act_I | action-head offset | −0.529 | −0.731 | −0.202 |
| T16_quad_X | action-head offset | −0.805 | −0.607 | +0.198 |
| T2_enc_X | encoder stage-1 offset | −0.639 | −0.834 | −0.195 |

Nine of the twelve are **offsets**, and in eight of those nine the parent's offset is the more
negative. The single largest *gain* disagreement is 0.228. **The two batches agree much better on how
far the modulator scales a layer than on how far it shifts it**, and the parent's noisier learning
signal is associated with a consistently larger additive excursion. That is an engineering
observation about the estimator, not a claim about behaviour.

### 5.7 Twin comparison at full length — the clipping difference washed out, then reversed

The earlier draft's one systematic difference was that the twin's outside-world-input and all-senses
arms spent 10–27% of logging windows with a **mean** gradient above the 0.5 ceiling against the
parent's 0–4%, **and that the effect was present in the unmodulated control too** (10.4% vs 0.0%),
which is what attributed it to the estimator rather than to modulation. That measurement was taken
over episodes 200k–1.25M. It reproduces exactly on the completed data — the window is unchanged, so
it must.

**What the full run adds is that the effect is confined to roughly the first tenth of training.**
Tracking the same quantity per decile:

| Decile of training | 1 | 2 | 3 | 4 | 5–10 |
|---|---|---|---|---|---|
| control — MC | 2.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| control — GAE_NORM | **9.9%** | 0.0% | 0.0% | 0.0% | 0.0% |
| encoder / exteroceptive — MC | 4.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| encoder / exteroceptive — GAE_NORM | **21.9%** | 0.5% | 0.0% | 0.0% | 0.0% |
| action head / exteroceptive — GAE_NORM | **19.9%** | 0.5% | 0.0% | 0.0% | 0.0% |
| four-site / exteroceptive — GAE_NORM | **13.5%** | 0.0% | 0.0% | 0.0% | 0.0% |

By the second decile the difference is gone in every one of the sixteen pairs, and from the third
decile onward **no arm in either batch has a single logging window whose mean gradient exceeds the
ceiling**, with two isolated exceptions in the parent batch (one window each in `T3_rnn_ALL` and
`T5_crt_X`).

**And on the softer measure — how often a window's *peak* touched the ceiling — the ordering
reverses.** That statistic falls over training in both batches but much faster in the twin:

| Cell | decile 1 | decile 3 | decile 5 | decile 8 | decile 10 |
|---|---|---|---|---|---|
| control — MC | 99% | 98% | 79% | 72% | **71%** |
| control — GAE_NORM | 98% | 55% | 26% | 0% | **0%** |
| action head / all-senses — MC | 100% | 99% | 92% | 74% | **52%** |
| action head / all-senses — GAE_NORM | 98% | 44% | 10% | 1% | **1%** |
| four-site / exteroceptive — MC | 98% | 97% | 91% | 77% | **43%** |
| four-site / exteroceptive — GAE_NORM | 95% | 52% | 24% | 8% | **0%** |

Over the whole post-warm-up run this makes the twin the **quieter** batch on every arm: its
grid-mean gradient is 0.205 against 0.255, its worst arm's peak-touch fraction is 70.2% against the
parent's 90.8%, and its tail has six excursions above 3.0 against twenty-two (§5.3).

**How to read this.** The correct summary is now: *under the second learning-signal recipe the
opening transient is briefly hotter and the rest of training is uniformly cooler.* The earlier
draft's sentence — "the second batch's outside-world-input arms spend more of their time in the upper
part of the gradient range" — was true of the data it had and is **not** true of the completed runs,
and it is superseded. The effect remains present in the unmodulated control at both ends of the
reversal, so it remains attributable to the estimator rather than to the modulator. Nothing in either
batch leaves the healthy regime at any point.

### 5.8 The estimator's survival effect is uniform in sign across all sixteen cells

**Why a survival number appears in a health audit.** The two grids are intended to be read as a
matched pair. That reading needs the modulation machinery not to interact with the learning-signal
recipe in some cell-specific way — if, say, the estimator helped four arms and hurt three, the pair
would not be matched and no cross-grid statement could be made. Checking that is an engineering
precondition, so it belongs here. **It is not a claim about which configuration performs better, and
it is not an outcome analysis.**

Mean survival over each run's final million episodes, per cell:

| Cell | MC | GAE_NORM | difference |
|---|---|---|---|
| T1_none (control) | 164.46 | 170.62 | **+6.16** |
| T2_enc_ALL | 168.44 | 172.65 | +4.21 |
| T2_enc_I | 163.56 | 170.35 | +6.79 |
| T2_enc_X | 167.61 | 173.16 | +5.56 |
| T3_rnn_ALL | 164.87 | 167.59 | +2.72 |
| T3_rnn_I | 166.93 | 170.22 | +3.29 |
| T3_rnn_X | 165.62 | 170.77 | +5.14 |
| T4_act_ALL | 166.87 | 168.90 | +2.03 |
| T4_act_I | 165.83 | 169.22 | +3.39 |
| T4_act_X | 165.17 | 170.32 | +5.15 |
| T5_crt_ALL | 165.43 | 169.80 | +4.37 |
| T5_crt_I | 163.53 | 169.38 | +5.85 |
| T5_crt_X | 165.74 | 169.97 | +4.23 |
| T16_quad_ALL | 169.76 | 172.18 | +2.42 |
| T16_quad_I | 163.04 | 169.02 | +5.98 |
| T16_quad_X | 170.58 | 172.22 | +1.63 |
| **all sixteen** | | | **mean +4.31, s.d. 1.59, range +1.63 to +6.79, positive in 16/16** |

Read on the final 200,000 episodes instead, the same picture: **mean +4.58, range +1.95 to +7.76,
positive in 16/16**.

**The precondition holds, with three qualifications the user should carry.**

1. **Uniform in sign, and the dispersion is small relative to known noise.** The sixteen differences
   scatter with a standard deviation of **1.59 steps**, against a **4.5-step** spread between the
   five reference seeds of the unmodulated `MC` configuration measured the same way. A single
   common estimator effect plus one-seed noise reproduces the sixteen numbers without needing any
   cell-specific interaction term. Grouping them, the write-site means run +3.35 to +5.52 and the
   input-slice means +3.15 to +5.06 — ranges of about two steps, well inside single-seed noise.
2. **"Uniform" is an observation about 16 paired draws at one seed, not a tested hypothesis.** The
   sixteen differences are not sixteen independent experiments: every cell in both grids uses seed
   42, so a lucky or unlucky draw of that seed is common to all of them. A sign test on 16/16
   would be misleading for exactly that reason and is not offered.
3. **The apparent uniformity is partly a ceiling effect and should not be over-read.** The
   correlation between a cell's parent-batch level and its difference is **−0.72**: the three cells
   where the parent scored highest (`T16_quad_X` 170.58, `T16_quad_ALL` 169.76, `T2_enc_ALL` 168.44)
   are the three with the smallest differences (+1.63, +2.42, +4.21). Both batches' cells sit within
   a few steps of each other near the top of this environment's range, so the differences compress
   where the parent already did well. That is a regression-toward-the-ceiling pattern, and it is the
   most likely explanation for why the twin's arm-to-arm spread (5.6 steps) is smaller than the
   parent's (7.6).

**On the control specifically, and on the earlier `+4.47` figure.** The control's difference here is
**+6.16** steps (final million) or **+7.76** (final 200,000). The reference for that comparison is
**not** the +4.47 in [[return_mode_cmp_10M]] §4.9 — that number is a **greedy-evaluation** result
(2,000 evaluation episodes per seed, 170.04 vs 165.57), a different measurement from the
training-time rolling mean used throughout this document. The right training-time comparator is the
same study's five-seed end-of-budget means: **170.36 (`GAE_NORM`) − 165.20 (`MC`) = +5.16 steps**.
Against that, this grid's single-seed control difference of +6.16 is one step high, decomposing as
its `MC` control landing 0.74 below the five-seed `MC` mean and its `GAE_NORM` control landing 0.26
above the five-seed `GAE_NORM` mean — both ordinary single-seed draws given the `MC` reference's
1.9-step standard deviation. **The control reproduces the known estimator effect within single-seed
noise, and the +4.47 and +5.16 figures should not be conflated.**

---

## 6. Documentation / provenance defects

### 6.1 The metrics reference is still misleading about the front-end gain — unchanged, and now affects thirty-two runs

**What it says.** [[NMN_METRICS_REFERENCE]] §5.1 states that `gamma_uni_mean` and `gamma_multi_mean`
are "raw pre-sigmoid values, not the actual gains applied", and instructs the reader to apply a
sigmoid to recover the gain. Its §3 table lists their initialisation value as "~2.0", and its §4
healthy-range table gives them a healthy band of 1.0–3.0.

**What is true for these thirty-two runs.** They use `modulation.type: FiLM`, and under FiLM the
front-end applies the gain **linearly**, with no sigmoid
(`src/models/recurrent_ppo_network.py:240-242, 258-260`), and initialises the gain bias to 1.0 rather
than 2.0 (`src/models/neuromodulator.py:142-146`). The data agrees: the first logged `gamma_uni_mean`
in every encoder arm of both batches is 0.96–1.00, not ~2.0.

**Why it matters.** A reader who follows §5.1's instruction on these batches will compute
`sigmoid(0.375) = 0.593` and conclude the front-end gain is 0.59, when it is 0.375. And §4's healthy
band of 1.0–3.0 would mark every one of these perfectly healthy runs as sub-healthy. The reference's
own §2.8 already gets this right for the three *new* sites — the gap is that the encoder's series are
now *also* linear whenever the type is FiLM, and §5.1, §3 and §4 have not been updated to say
"depends on `modulation.type`".

**This is a documentation fix, not a code fix.** The code is doing the right thing. Flagged in §10;
this audit does not edit the reference.

### 6.2 The provenance dirty-flag failure reproduced, and got worse

**What the design required.** §3 of [[NMN_INPUT_SITE_GRID]]: "All 16 rows must show the **same** SHA
with `git_dirty: false`; a row that does not is a run that cannot be reconstructed and is excluded
from every comparison." The twin's design inherits that gate.

**What was recorded.**

| Batch | Same SHA in all 16? | `git_dirty: false` | `git_dirty: "unknown"` | `git_dirty: true` |
|---|---|---|---|---|
| MC (`a71f4471`) | yes | 0 | **15** | 1 |
| GAE_NORM (`6695aa29`) | yes | 0 | **16** | 0 |

Taken literally, **the design's gate is not met by any of the thirty-two rows**. The twin is
*worse*: not one of its sixteen runs succeeded in reading the flag. **Re-checked on the twin's
provenance files now that all sixteen have finished — still `"unknown"` in all sixteen.** The flag is
written once at launch, so this was expected; it is stated because the earlier reading was taken
mid-run and a reader should not have to wonder whether it changed.

**Why it happened.** `src/utils/provenance.py:99-102` determines dirtiness by running
`git status --porcelain --untracked-files=no` with a **10-second timeout**, and returns the string
`"unknown"` on timeout or failure — deliberately, so that "we could not tell" is distinguishable from
"it was clean". Sixteen training processes started within seventeen seconds of each other, each running
`git status` against the same repository on a NAS filesystem; `git status` takes the index lock and
is documented in this project as slow on that mount. The cheap `git rev-parse` calls succeeded in all
thirty-two, which is consistent with lock contention rather than broken git. **The twin's 16/16
failure rate against the parent's 15/16 confirms this is a systematic property of simultaneous
multi-run launch, not bad luck.**

**Why it is very probably harmless here, and what was checked instead.** Since the recorded flag
cannot answer the question, four substitute checks were run, and all four pass:

- The MC launch commit **contains** both prerequisite commits (`e1aab726`, `83b8140b`) as ancestors,
  and `git diff e1aab726 a71f4471 -- src/` is **empty**.
- The twin's launch commit `6695aa29` has `a71f4471` as an ancestor, and
  `git diff a71f4471 6695aa29 -- src/ train.py` is **empty** — so both batches ran byte-identical
  training code.
- The **trainer-written resolved configs** of three twin pairs differ in exactly the five expected
  keys out of 264 (§2.3), which is a direct readout of what the trainer actually built.
- The **model that was actually built** in each of the thirty-two runs reports — in its own startup
  banner and in the trainer's own saved config — exactly the return mode, sites and input sensors the
  designs specify. That is a stronger check than a dirty flag, because it inspects what the system
  produced rather than what the source ought to produce.

**The engineering issue that remains** is that this project's provenance record becomes unreliable
precisely when it is needed most — a simultaneous multi-run launch — and it now has two independent
demonstrations. A one-line fix exists (retry, or raise the timeout, or take the dirtiness reading once
in the launcher and pass it down) but it is a code change and is therefore flagged in §10 rather than
made here.

---

## 7. Throughput and cost

### 7.1 MC batch — no arm is anomalously slow; the ordering is dominated by hardware placement

Environment steps per second (cumulative average over each run's life):

| Cell | Node:GPU | Card | steps/s | vs. control |
|---|---|---|---|---|
| T1_none (control) | 106:0 | 3090 | 49,655 | — |
| T5_crt_X | 112:0 | 3090 | 43,988 | −11% |
| T5_crt_ALL | 111:0 | 3090 | 43,844 | −12% |
| T4_act_I | 110:0 | 3090 | 43,380 | −13% |
| T3_rnn_X | 109:0 | 3090 | 43,078 | −13% |
| T2_enc_I | 107:0 | 3090 | 42,868 | −14% |
| T4_act_X | 110:1 | 3090 | 42,168 | −15% |
| T2_enc_ALL | 106:1 | 3090 | 42,061 | −15% |
| T5_crt_I | 111:1 | 3090 | 41,764 | −16% |
| T3_rnn_ALL | 108:0 | 3090 | 41,536 | −16% |
| T2_enc_X | 107:1 | 3090 | 40,614 | −18% |
| T4_act_ALL | 109:1 | 3090 | 40,062 | −19% |
| T16_quad_ALL | 112:1 | 3090 | 37,314 | −25% |
| T3_rnn_I | 108:1 | 3090 | 36,661 | −26% |
| T16_quad_I | 113:0 | **4090** | 48,937 | n/a (faster card) |
| T16_quad_X | 113:1 | **4090** | 49,951 | n/a (faster card) |

**The honest reading.** On identical hardware, adding a modulator costs 11–26% throughput, and the
four-site configuration costs about 25%. That is a compute cost, not a defect. But two confounds make
any finer comparison meaningless:

- **Card.** The two four-site arms on the 4090 node run at the control's speed on a faster card, so
  their apparent parity with the control is hardware, not efficiency.
- **GPU slot.** On **all seven** RTX 3090 nodes (106–112) the run on GPU 0 is faster than the run on
  GPU 1 of the same node, by 2–13%; the only node where the ordering reverses is the 4090 node. Seven
  of seven is not chance. That is a systematic placement effect (thermal, PCIe, or host contention
  between the two co-resident processes), and it is large enough to reorder the middle of this table
  on its own.

### 7.2 GAE_NORM batch — speed is set by the card, and cannot be compared with §7.1

Environment steps per second (cumulative average over each completed run):

| Card | Node | Cells | steps/s (range) |
|---|---|---|---|
| RTX 2080 Ti | 101, 103, 104, 105 | control, enc_ALL, rnn_{ALL,I,X}, act_{ALL,I,X} | 29,955 – 37,601 |
| RTX 3090 | 106 | crt_ALL, crt_I | 42,667 – 43,493 |
| RTX 4090 | 102 | enc_I, enc_X | 50,635 – 50,722 |
| RTX 6000 Ada | 114 | crt_X, quad_{ALL,I,X} | 46,128 – 53,679 |

Card separates the speed groups without a single inversion, and it also determined the order in which
these runs finished (the 2080 Ti arms completed roughly six hours after the Ada arms). **§7.1's
modulator-overhead figures cannot be carried across.** Batch A ran fourteen of sixteen cells on
identical RTX 3090s, which is what made its 11–26% overhead estimate meaningful; Batch B spreads four
card generations across sixteen cells, so its speed table measures hardware, not architecture. The
one within-card comparison available here — the control (37,601) against `T2_enc_ALL` (29,955), both
on node 101's two 2080 Ti GPUs — gives a 20% cost for a single front-end modulator, consistent with
§7.1's range but a single pair and subject to the same GPU-slot effect §7.1 documents.

Wall-clock gaps between consecutive logged rows are 6.7–11.9 s median, 12.7–23.3 s at the 99th
percentile, maximum 19.4–38.6 s. **The largest gap in any twin run is 5.5× its own median**, on the
fast 4090 node where the fixed-cost checkpoint step is a larger multiple of a smaller median. **No
stall signature anywhere**, and no run recompiled.

---

## 8. Where a metric looks unusual but the code is probably fine

### 8.1 "A newly enabled site is a no-op at step 0" is true on average, not per unit — confirmed twice

The design documents lean on the property that switching a re-tuning site on cannot hurt at the start
of training, because the site begins as an identity operation. The code
(`src/models/neuromodulator.py:142-146, 178-186`) sets the **bias** of the gain head to exactly 1.0
and the **bias** of the offset head to exactly 0.0 — but leaves the head's **weight matrix** at the
framework's default random initialisation, and adds no zero-gate in front of it. So at the very first
forward pass the gain is `1.0 + W·h`, where `h` is the modulator's internal state: one on average,
but with a per-unit random deviation.

**The size of that deviation is measurable, is not small, and reproduces in the twin.** At the first
logged point the per-unit standard deviation of the gain is **0.29–0.37** in every MC arm and
**0.29–0.39** in every GAE_NORM arm; of the offset, **0.27–0.35** and **0.28–0.35** respectively —
re-measured on the twin's complete logs, unchanged from the early-life reading because these are
first-logged-point values. So on day one a typical unit is being multiplied by something in the
region of 0.7 to 1.3 and shifted by ±0.3, rather than left alone.

**Why this is a note and not a bug report.** The code comment at
`src/models/neuromodulator.py:178-181` explicitly claims "a newly-enabled site is a no-op at step 0",
and taken per-unit that claim is not exact. But nothing observed suggests it caused harm — every one
of the thirty completed modulated runs learned normally from the start, and the arms differ from
their controls in their first-million survival by amounts that are not ordered by how many sites they
enable (in the twin, the four-site all-senses arm has the *highest* first-million survival at 97.0
and the four-site exteroceptive arm the *lowest* at 73.5). The correct
response is to **state the property accurately** in the designs' reasoning rather than to change the
initialisation mid-experiment.

**What would confirm the alternative** (that the random per-unit deviation does matter): re-run one
cell with the gain and offset heads' weight matrices initialised to zero, making the site an exact
identity at step 0, and compare the first-million-episode survival curve. That is a one-run control,
not a code change to either batch.

### 8.2 The gain drift, restated with both batches finished

Covered in §4.3 and §5.4. Restating the discipline: a falling FiLM gain is **not** evidence of a
layer being silenced, because the layer's own weights are free to grow and absorb the change. The
observation across both batches is "the mean gain fell to 0.160–0.909 depending on site, arm and
batch, then **plateaued or partly recovered at every site except the action head**, while the offsets
at the action and value heads were **still sliding when the budget ran out in both batches**". The
inference "the site is silencing its layer" is **not** supported by anything currently logged.

Two things the completed twin adds. First, the combination §4.3 flagged as worth a targeted check —
a gain in the 0.2–0.4 region together with an offset near −1.0 at a layer that is then passed through
a rectifier — is **not** milder in the twin despite its offsets travelling less far: the twin's
four-site narrow-input arm ends with a value-head gain of **0.160** and a value-head offset of
**−1.050**, the most extreme such pairing anywhere in either batch. Second, the fact that the
non-convergence at the budget's end reproduces at the *family* level in both batches (§5.4.2) removes
the possibility that it was one run's quirk. Both raise, not lower, the priority of §9 item 2.

---

## 9. Metrics Requested

Four metrics whose absence bounded this audit. All are cheap. None is required for either batch's
pre-registered analysis to proceed.

**Priority changed now that both batches are complete.** Item 2 (post-FiLM active fraction) was
already ranked highest and the twin's completion **raises** it further; item 1 (splitting the
modulator's spread into its across-unit and across-time parts) **rises from third to second**; item 3
(per-update clipping statistics) **falls**, because the completed data answered most of what it was
wanted for. Reasons are given per row.

| | Metric | Why now | Where it'd live | Cost |
|---|---|---|---|---|
| **1 (was 2) — highest** | `network/<site>_postfilm_active_fraction` — the fraction of the modulated layer's units whose post-FiLM pre-activation is positive, i.e. surviving the rectifier (dimensionless, 0–1) | **Raised in priority by the completed twin.** The concern is a gain in the 0.2–0.4 region combined with an offset near −1.0, at a layer feeding a rectifier: if the layer's own pre-activation is O(1), that combination pushes a large fraction of units permanently off, which would be a real sparsification invisible in every currently-logged metric. Three things now argue for it that did not before. (a) The twin's offsets travel *less* far than the parent's (§5.6), yet the twin still produces the most extreme pairing in either batch — a value-head gain of 0.160 with an offset of −1.050 (§8.2). (b) The non-convergence of the action- and value-head offsets at the budget's end reproduces across both batches at the family level (§5.4.2), so it is a property of the setup, not one run's quirk. (c) Any cross-grid conclusion now rests on the two batches' modulators doing comparable things, and the largest cross-batch disagreements are precisely in this channel (§5.6) — this scalar is what would say whether a −1.83 offset and a −1.11 offset are functionally the same or not | `src/models/recurrent_ppo_network.py`, immediately after the FiLM applications at lines 553 and 564 | cheap — one comparison and one mean per site per iteration |
| **2 (was 1)** | `modulator/gamma_<site>_temporal_std` and `modulator/gamma_<site>_unit_std` — the standard deviation of the site's gain **across time/batch after averaging over units**, and **across units after averaging over time** (dimensionless multiplier) | The single currently-logged spread mixes these two components, so it cannot distinguish a modulator that genuinely re-tunes the network moment-to-moment from one that has learned a fixed per-unit re-parameterisation and is functionally inert. The narrow-input (`I`) arms raise exactly this question, and it is now **doubly live**: the same quiet-gradient ordering reproduced at full length under a second estimator (§5.3) without becoming any easier to interpret, and the per-unit spread grew 3–4× over training in both batches (§4.3, §5.4.1) with no way to say which component grew. It is also the discriminator that keeps a conditional behaviour from being averaged away | `train.py` around the `_mod_spread` helper at line ~1848, where the existing mean/std over `mod_info` are already computed | cheap — two extra reductions over an array that is already resident, at the existing logging cadence |
| **3 (was 3) — lowered** | `loss/grad_norm_p99` and `loss/grad_norm_clipped_fraction` — the 99th percentile of the **per-update** pre-clip gradient norm within a logging window, and the fraction of individual updates whose norm exceeded `max_grad_norm` | **Lowered, because the completed data answered most of the question.** The reason for asking was that the twin's clipping regime appeared to differ from the parent's in the upper tail, which the current logging measures worst. At full length that difference turned out to be an opening transient that reverses (§5.7), and the twin's tail is unambiguously *thinner* (six excursions above 3.0 against twenty-two, §5.3) — a conclusion the window peak was adequate to reach. What the metric would still buy is localising §4.8's 1030 excursion to a single minibatch, which is a curiosity rather than a blocker now that a full sixteen-run replication under a second estimator produced nothing above 8.4 | `src/models/recurrent_ppo_trainer.py` around line 375, where `grad_norm` is already computed per update, plus the windowing in `train.py` | cheap — the per-update norms already exist; this is an extra reduction over them |
| **4 (unchanged)** | `modulator/param_count` and `network/param_count` at startup (integers, logged once) | §7 attributes a throughput cost to the modulator, but the audit cannot state the parameter cost of each site configuration because it is not recorded. This got slightly worse rather than better: Batch B is spread across four card generations (§7.2), so the only clean overhead measurement left is Batch A's, and a future reader comparing the two grids' cost has to reconstruct parameter counts from the architecture | `train.py`, in the startup banner block that already prints the modulation configuration | cheap — one scalar each, logged once |

---

## 10. Related Issues

Items that need a code or documentation change and are therefore **not** made by this audit. The user
decides whether to route them.

1. **Metrics reference is stale for FiLM-type runs** (§6.1).
   `docs/develop/active/neuromodulation/NMN_METRICS_REFERENCE.md` §5.1, §3 (init values for
   `gamma_uni` / `gamma_multi`) and §4 (healthy band 1.0–3.0) describe the pre-sigmoid behaviour and
   do not say it is conditional on `modulation.type`. Under `type: FiLM` the encoder's gains are
   linear and initialise at 1.0. Suggested route: documentation maintenance. **A reader who follows
   §5.1 on these thirty-two runs will misread every front-end number.**
2. **Provenance dirty-flag is unreliable under simultaneous multi-run launch — now reproduced and
   final** (§6.2). `src/utils/provenance.py:99-102` runs `git status` with a 10 s timeout per training
   process; sixteen concurrent launches produced `"unknown"` in 15/16 on the first batch and
   **16/16** on the second, re-confirmed after both batches finished. Suggested route: a bug-fix plan under `docs/develop/active/`. Options
   worth weighing: retry on timeout, raise the timeout for this call specifically, or have the
   launcher read the flag once and pass it to every child process. The current behaviour is *safe*
   (it never claims clean when it does not know) — the defect is that it makes an experiment-design
   gate unsatisfiable, twice.

Also worth surfacing to whoever owns the design documents, though neither is a defect:

3. **Neither Launch Manifest was filled in, and both batches are now finished.** All sixteen rows of
   [[NMN_INPUT_SITE_GRID]] §3 and all sixteen of [[NMN_INPUT_SITE_GRID_GAENORM]] §3 still read
   `planned` with empty node, GPU, run-ID and log-path columns, although every run has completed.
   §2.1 and §2.2 above reconstruct the actual values from the runs' own metadata and are offered as
   evidence for whoever owns the manifests. Per the manifests' stated ownership, only
   `training-runner` writes those columns; this audit has not touched them.
4. **The designs' step-0 identity argument should be restated** (§8.1). "Turning a site on cannot
   hurt at step 0" holds for the layer mean but not per unit, where the gain starts at 1.0 ± ~0.3.
   Confirmed independently in both completed batches.
5. **One claim in the previous version of this document is withdrawn** (§5.7). It reported that the
   twin's wide-input arms spend more of their training with the gradient running hot against the clip
   than the parent's. That was true of the first fifth of training, which was all the data then
   available; over the complete runs the ordering **reverses** and the twin is the quieter batch
   throughout the remaining four-fifths. Anyone who took the earlier sentence into a figure or a
   summary should update it. The attribution is unchanged — the effect is present in the unmodulated
   control at both ends, so it belongs to the estimator, not to the modulator.

---

## 11. Bottom line

**Both batches are finished and both are healthy.** All thirty-two runs completed 10,000,000
episodes. No run needs to be discarded. The wiring is correct in all thirty-two, verified three
independent ways, and the twin's verification is now run-by-run rather than on a sample: every one of
its sixteen trainer-written configs records the right learning-signal recipe, the right write sites
for its cell, the right sensor list for its input slice, the temperature mechanism off and the
memory-cell mechanism set to activation — and every one of its sixteen logs exactly the corresponding
metric set, with the temperature and legacy gate-bias series absent as expected and the same 59
non-modulator keys in every run of both batches. Zero non-finite values anywhere. Zero recompiles.
Zero constant series. Zero warnings or errors. No stall signature.

**Both controls pass their pre-registered gates**, independently recomputed here: 164.46 survival
steps for the first batch against a 162.4–166.9 band, and **170.62 for the second against a
169.9–171.2 band** that is three and a half times narrower. Both grids therefore have a valid
reference point, and the designs' precondition for attempting the cross-grid analysis is met.

**The modulator was driven throughout, in all thirty modulated runs.** The smallest gradient seen
anywhere across either batch's complete history is 0.0071 (first batch) and 0.0076 (second), about
seven times the vanishing threshold, both in the same cell. No arm has a single logged point below
the threshold. The first batch's modulator gradient *grows* over training in fourteen of fifteen
arms; the second's is flat; the only site where it weakens in both is the action head, and even there
the smallest last-decile mean is seventeen times the threshold.

**The drift shape reproduced across the pair in six of seven site families.** Front-end gains fall,
bottom out near the middle of training and climb back; memory-cell and value-head gains plateau; the
action-head gain keeps creeping down; front-end and memory offsets flatten; and the action-head
offset is still sliding at the budget's end, worst in the all-senses arm, at 11% of its own final
magnitude per final tenth **in both batches to the digit**. The one family that did not reproduce is
the value-head offset, where the arm that had not converged differs between the batches. Because the
two grids share seed 42, a reproduced shape is **weaker** evidence than a second seed would be; what
it shows is that the shape survives changing the learning-signal recipe. The arm-level claim "this
exact arm has not converged" does not survive and should not be leaned on. The family-level claim —
**the action-head and value-head offsets have not converged at 10,000,000 episodes** — survives both
batches and is the justification for the highest-priority requested metric.

**The twin comparison, kept to engineering, and it changed with length.** Three of the earlier
draft's numbers move when measured over the complete runs rather than an early window, and in each
case both readings are correct measurements of different stretches of training. The modulator's
gradient agreement band goes from 0.77×–1.18× (early) to **0.70×–0.94× with the twin below the parent
in all fifteen pairs** (full), because the parent's grows and the twin's does not. Agreement between
the two batches' re-tuning signals loosens from 58-of-60 within 0.14 (early) to **41 of 60** (end of
training), with nine of the twelve largest gaps being offsets that the parent drove further negative.
And the one systematic clipping difference — **more windows running hot under the new recipe** —
turns out to be confined to the opening tenth of training and then to **reverse**: from the third
decile onward the twin is the quieter batch on every arm, with a grid-mean gradient of 0.205 against
0.255 and six post-warm-up excursions above 3.0 against twenty-two. That earlier sentence is
**withdrawn and superseded** (§10 item 5). Its attribution is not: the effect appears in the
unmodulated control at both ends of the reversal, so it belongs to the estimator rather than to the
modulator, and nothing leaves the healthy regime at any point.

**The 1030 spike looks weaker as a defect candidate than it did.** Fifteen modulated architectures
plus a control, re-run for ten million episodes each under a different return mode, produced nothing
above 8.4. That is consistent with the event being an extreme draw from a heavy tail belonging to the
first learning-signal recipe, and inconsistent with it being a fault in the modulation code.

**One health-relevant property of the pair.** The second batch finishes above the first in **16 of 16
cells**, mean +4.31 survival steps, spread 1.59 — smaller than the 4.5-step seed-to-seed spread of
the same measurement. Nothing in the data requires the modulator machinery to interact with the
learning-signal recipe cell-specifically, which is the precondition for treating the two grids as a
matched pair. **This is 16 paired draws at a single shared seed, not a tested hypothesis**, it is
partly a ceiling effect (the correlation between a cell's parent-batch level and its gain is −0.72),
and it is not a claim about which configuration is better. The control's own difference of +6.16
reproduces the known unmodulated estimator effect of +5.16 within single-seed noise — and should not
be compared against the +4.47 figure from the return-mode study, which is a greedy-evaluation number
rather than a training-time one.

**What is still wrong is documentation and provenance, not training.** The metrics reference would
make a reader misread every front-end number in both batches; the launcher's clean-working-tree flag
recorded "unknown" in 31 of 32 runs, making an experiment-design gate unsatisfiable twice over; and
neither Launch Manifest has been filled in although every run has now finished. All three are in §10
for the user to route.

---

## Appendix

### A. Working files

Raw extractions, retained for traceability (gitignored):

- `tmp/nmnsite2/extract.py` — the local transaction-log reader used here (no WandB web API). Stores
  every series as `(row index, value)` pairs so that metrics emitted in different history records can
  be aligned to an episode count by forward fill.
- `tmp/nmnsite2/lib.py` — shared cell↔run-ID maps and the episode-aligned column accessor.
- `tmp/nmnsite2/mc_<runid>.json` — Batch A per-run complete series, key inventory and non-finite
  scan. Unchanged; Batch A was already complete when these were written.
- `tmp/nmnsite2/gn_<runid>.json` — Batch B at 12.8–20.6%. **Superseded**, retained only so the
  earlier draft's numbers can be reproduced.
- `tmp/nmnsite3/gn_<runid>.json` — **Batch B re-extracted at completion**; the source of every Batch B
  number in this version. `tmp/nmnsite3/lib.py` reads Batch A from `nmnsite2` and Batch B from
  `nmnsite3`, so every cross-batch comparison in §5.5–§5.8 uses the final data on both sides.
- `tmp/nmnsite2/r1.py` … `r9.py` and `tmp/nmnsite3/{r1,r2,r3,r5,r6,r7,r8b,t1,t2,t3}.py` — the
  analyses this document is built from (progress and survival; modulator gradient, fade and clipping;
  FiLM temporal tables; metric-set audit; excursion census and throughput; constant-series and
  wall-clock-gap scan; drift turning points for both batches; full-history twin comparison;
  per-decile clipping evolution; delta dispersion, control gate and temporal evolution).
- `tmp/nmnsite2/cfgdiff.py` — the flattened key-by-key config differ used in §2.3.
- `tmp/20260907_172348_nmnsite_audit_refresh.md` — consolidated output of the first refresh.
- `tmp/20260908_gaenorm_final_audit.md` — consolidated output of this pass.

### B. Provenance of every number here

Every figure in this document comes from one of: a run's own WandB transaction log, a run's own
trainer-written `models/config.yaml`, a run's own `models/provenance.json`, a run's own startup
banner, a `git` query against this repository, or a named line of source in `src/` / `train.py`. No
number is taken from the WandB web interface, and none is quoted from another agent without
independent recomputation — including the control's reference-band result in §4.1 and the one-key
twin-difference claim in §2.3.

### C. Changelog

| Date | Change |
|---|---|
| 2026-09-07 | Initial audit. MC grid: six runs complete, ten at 84–99%. |
| 2026-09-08 | **Final.** GAE_NORM twin now 16/16 complete at 10,000,000 episodes — every provisional twin figure recomputed on full histories and §5 rewritten from "early-life health check" to "final numbers". Twin control gate independently confirmed **PASS** at 170.62 against the pre-registered 169.9–171.2 (§5.1). Drift-shape reproduction tested family by family: six of seven reproduce, the value-head offset does not (§5.4.2). Twin comparison redone at full length: gradient ratio band moves to 0.70×–0.94× with the twin below in 15/15 (§5.5), FiLM end-value agreement loosens to 41/60 within 0.14 (§5.6), and **the hot-window clipping difference is shown to be an opening transient that then reverses — the earlier claim is withdrawn** (§5.7, §10 item 5). Excursion census: 6 events above 3.0 across 5 of 16 runs, max 8.4, against Batch A's 22 across 13 of 16 and max 1030 (§5.3). Added the paired estimator-effect observation, framed as a matched-pair precondition (§5.8). Config verification extended from 3 pairs to all 16 twin runs (§2.3). Metrics Requested re-prioritised (§9). Batch B throughput rewritten as card-bound and non-comparable with Batch A's (§7.2). |
| 2026-09-07 | **Refresh.** MC grid now 16/16 complete — every reported figure recomputed on full histories (§4). Added the drift turning-point analysis that the partial data could not support (§4.3.2): drift mostly plateaued or reversed; two offsets had not converged. Confirmed the 1030 spike stayed a one-off and censused the one new excursion (§4.8). Added Batch B, the GAE_NORM twin, at 12.8–20.6%: manifest (§2.2), one-key config verification on three twin pairs (§2.3), full early-life health check (§5), matched-episode twin comparison (§5.4), throughput (§7.2). Provenance defect reproduced at 16/16 (§6.2). |
