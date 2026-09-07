---
title: "Training-health and engineering-correctness audit of the modulation-site grid and its estimator-swapped twin"
topic: nmn_input_site_grid
status: active
created: 2026-09-07
last_updated: 2026-09-07
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

**Where each batch stands.** The first batch (the "MC grid") has **finished — all sixteen runs
completed their full ten million episodes**, so every number reported for it below is a genuine
end-of-training value rather than a snapshot of a run still in progress. The second batch (the
"GAE_NORM twin") is **12.8% to 20.6% of the way through**, so it gets an early-life health check
only. Nothing in this document compares the two batches on how well they perform.

**The verdict: the machinery is working, in both batches.** Across all thirty-two runs there is not a
single not-a-number or infinite value in any logged quantity. Every run's survival time rises from
about 18 steps at the start toward the mid-160s (finished batch) or mid-150s and still climbing
(running batch). Each of the thirty-two configurations logs exactly the signals its own saved
configuration says it should, and none that it should not. In all thirty modulated runs the
modulator is receiving real training signal — its gradient never once fell to the "disconnected"
level, the smallest value seen anywhere in either batch being about **seven times** the threshold the
project's own metrics guide calls vanishing. Nothing here justifies stopping, restarting or
discarding a run.

**Four things worth the user's attention:**

1. **The finished batch answers a question the earlier draft of this audit could not: the downward
   drift of the re-tuning signals did *not* continue to the end — it mostly flattened, and at the
   sensory front-end it partly reversed.** The multiplier the modulator applies at the front-end
   bottomed out around 25–60% of the way through training and then climbed back up by 0.03 to 0.17;
   the memory-cell and value-head multipliers reached a plateau and stayed there. **Two signals are
   the exception and had *not* converged when the episode budget ran out**: the additive offset at
   the action head and at the value head, in the arms that feed the modulator all 27 senses. Those
   two were still sliding downward at the last logged point, at roughly 11% and 7% of their own
   final magnitude per final tenth of training. Detail and the full turning-point table in §4.3.
2. **The twin cells behave alike on the health metrics, at matched training progress.** Because both
   batches use the same random seed, matched cells start from bit-identical weights. Compared over
   the same window of episodes, the modulator's gradient in the second batch is 0.79× to 1.16× the
   first batch's in all fifteen modulated pairs, the typical total gradient differs by 3% across the
   grid, and the re-tuning signals agree to within about 0.1 in nearly every case. The one systematic
   difference is that the second batch's outside-world-input arms spend more of their time in the
   upper part of the gradient range — 10–27% of logging windows against 0–4% — without ever
   approaching a pathological level. §5.
3. **The one large gradient spike found in the earlier draft stayed a one-off.** The value-head
   run on outside-world input recorded a single spike to 1030 at 97% of its run; it did not recur,
   nothing else moved when it happened, the gradient clip absorbed it, and the run finished mid-pack.
   Exactly one new spike above 3.0 appeared anywhere as the ten then-unfinished runs completed
   (peak 6.6, in the front-end narrow-input arm, in its final 0.04% of training). §4.8.
4. **The provenance-recording weakness reported earlier got worse, not better.** The launcher's
   check for "were there uncommitted edits at launch time" recorded "unknown" in **all sixteen** runs
   of the second batch (it was fifteen of sixteen in the first). The experiment design's gate
   requiring every row to record "no uncommitted edits" is unsatisfiable from the recorded evidence
   in both batches. §6.2 explains why this is very probably harmless here and what was checked
   instead. This is now a reproduced defect, not a one-off.

**The reference run passes its gate.** The unmodulated control of the finished batch ended at
**164.46 mean survival steps** over its final million episodes, inside the 162.4–166.9 band the
design registered in advance from five previously-run seeds of the same configuration. Independently
recomputed here; see §4.1.

**One correction carried forward unchanged.** The project's metrics reference is still out of date in
a way that would make a future reader misread every front-end number in both batches — it says the
front-end gain is "pre-sigmoid" and must be squashed, which is no longer true under this code. §6.1.

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
- **The second batch's sixteen runs are at *different* progress points** (12.8%–20.6%), and that
  spread is explained entirely by which graphics card each run landed on (§7.2). Any cross-arm
  comparison *within* that batch is therefore confounded by how far each run has trained, and none is
  offered. Where the two batches are compared at all (§5), it is only over a window of episodes that
  **every one of the thirty-two runs has completed**.
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

### 2.2 Batch B — the GAE_NORM twin (running)

Sixteen runs, WandB group `nmn_input_site_grid_gaenorm`, job type `pilot`, all seed 42, same
environment config, same 10,000,000-episode budget, all at code commit `6695aa29`. Cell names match
Batch A exactly, so a cell name identifies a **pair**.

The twin's Launch Manifest (§3 of [[NMN_INPUT_SITE_GRID_GAENORM]]) is also still showing every row
as `planned`. Reconstructed from the runs' own metadata:

| Cell | Tag | WandB run ID | Local log dir | Node:GPU | Card | Episodes | % |
|---|---|---|---|---|---|---|---|
| T1_none (control) | `rppo_nmngaenorm_t1none_s42` | `xvw6uzrs` | `wandb/run-20260907_155906-xvw6uzrs` | 101:0 | RTX 2080 Ti | 1,548,000 | 15.5% |
| T2_enc_ALL | `rppo_nmngaenorm_t2enc_ALL_s42` | `dtai9q6q` | `wandb/run-20260907_155906-dtai9q6q` | 101:1 | RTX 2080 Ti | 1,320,000 | 13.2% |
| T2_enc_I | `rppo_nmngaenorm_t2enc_I_s42` | `a0qrdefg` | `wandb/run-20260907_155914-a0qrdefg` | 102:0 | RTX 4090 | 1,836,000 | 18.4% |
| T2_enc_X | `rppo_nmngaenorm_t2enc_X_s42` | `9ovumjg0` | `wandb/run-20260907_155914-9ovumjg0` | 102:1 | RTX 4090 | 1,876,000 | 18.8% |
| T3_rnn_ALL | `rppo_nmngaenorm_t3rnn_ALL_s42` | `k3nkwafh` | `wandb/run-20260907_155909-k3nkwafh` | 103:0 | RTX 2080 Ti | 1,324,000 | 13.2% |
| T3_rnn_I | `rppo_nmngaenorm_t3rnn_I_s42` | `3om5z4jv` | `wandb/run-20260907_155910-3om5z4jv` | 103:1 | RTX 2080 Ti | 1,372,000 | 13.7% |
| T3_rnn_X | `rppo_nmngaenorm_t3rnn_X_s42` | `igrblp5d` | `wandb/run-20260907_155912-igrblp5d` | 104:0 | RTX 2080 Ti | 1,284,000 | 12.8% |
| T4_act_ALL | `rppo_nmngaenorm_t4act_ALL_s42` | `bo6t6y4m` | `wandb/run-20260907_155912-bo6t6y4m` | 104:1 | RTX 2080 Ti | 1,332,000 | 13.3% |
| T4_act_I | `rppo_nmngaenorm_t4act_I_s42` | `3whh6e86` | `wandb/run-20260907_155914-3whh6e86` | 105:0 | RTX 2080 Ti | 1,480,000 | 14.8% |
| T4_act_X | `rppo_nmngaenorm_t4act_X_s42` | `3jqmnf3a` | `wandb/run-20260907_155914-3jqmnf3a` | 105:1 | RTX 2080 Ti | 1,420,000 | 14.2% |
| T5_crt_ALL | `rppo_nmngaenorm_t5crt_ALL_s42` | `rinfx023` | `wandb/run-20260907_155915-rinfx023` | 106:0 | RTX 3090 | 1,748,000 | 17.5% |
| T5_crt_I | `rppo_nmngaenorm_t5crt_I_s42` | `e1ty78id` | `wandb/run-20260907_155916-e1ty78id` | 106:1 | RTX 3090 | 1,732,000 | 17.3% |
| T5_crt_X | `rppo_nmngaenorm_t5crt_X_s42` | `76krj6ri` | `wandb/run-20260907_155917-76krj6ri` | 114:0 | RTX 6000 Ada | 2,060,000 | 20.6% |
| T16_quad_ALL | `rppo_nmngaenorm_t16quad_ALL_s42` | `u69q0auc` | `wandb/run-20260907_155918-u69q0auc` | 114:1 | RTX 6000 Ada | 1,712,000 | 17.1% |
| T16_quad_I | `rppo_nmngaenorm_t16quad_I_s42` | `khcs2pxc` | `wandb/run-20260907_155920-khcs2pxc` | 114:2 | RTX 6000 Ada | 1,788,000 | 17.9% |
| T16_quad_X | `rppo_nmngaenorm_t16quad_X_s42` | `d4okzt4v` | `wandb/run-20260907_155922-d4okzt4v` | 114:3 | RTX 6000 Ada | 1,928,000 | 19.3% |

Episode counts are as of the last record flushed to each run's local transaction log at the time of
this extraction; all sixteen are still advancing.

**The progress spread is hardware, not health.** The eight runs on the 2080 Ti nodes (101, 103, 104,
105) are at 12.8–15.5%; the two on the 3090 node (106) at 17.3–17.5%; the two on the 4090 node (102)
at 18.4–18.8%; the four on the RTX 6000 Ada node (114) at 17.1–20.6%. Card explains the ordering
essentially perfectly (§7.2), which is why §1 rules out cross-arm comparison within this batch.

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

Working extractions: `tmp/nmnsite2/`, consolidated in
`tmp/20260907_172348_nmnsite_audit_refresh.md`.

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
run of both batches. No arm is missing an episode or loss metric its siblings have, and no arm has an
extra one.

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
high-frequency energy into the gradient. **The same ordering reproduces in the GAE_NORM twin**
(§5.3), which is a second, estimator-independent showing of the pattern.

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

**Did anything comparable appear in the ten runs that were still training?** **No.** Exactly one new
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

## 5. Batch B (GAE_NORM twin) — early-life health check

**What this section can and cannot say.** The sixteen twin runs are at 12.8–20.6% of their budget.
Every statement below is an early-life liveness and correctness check. Nothing here says anything
about how the twin will finish, and — because the sixteen runs are at different progress points,
determined by which card each landed on — nothing here ranks the twin's arms against each other.

### 5.1 Is every run learning, with no NaN or infinity? — Yes, all sixteen

**Zero non-finite values** across all sixteen transaction logs, in any metric, at any step. Every run
prints `JIT compiling train_iteration…` exactly once, has zero warning or error lines in its captured
output, and has zero exact-constant numeric series.

Survival steps by episode band, plus a short trailing window that is not contaminated by the opening
climb:

| Cell | 0–1M | 1–2M | last 200k episodes | s.d. | episodes done |
|---|---|---|---|---|---|
| T1_none (control) | 87.8 | 149.5 | 152.89 | 3.4 | 1,548,000 |
| T2_enc_ALL | 87.5 | 149.7 | 150.61 | 3.8 | 1,320,000 |
| T2_enc_I | 95.9 | 153.9 | 156.37 | 3.2 | 1,836,000 |
| T2_enc_X | 87.7 | 155.9 | 159.86 | 2.9 | 1,876,000 |
| T3_rnn_ALL | 97.6 | 152.5 | 153.41 | 3.8 | 1,324,000 |
| T3_rnn_I | 88.2 | 149.3 | 151.10 | 2.9 | 1,372,000 |
| T3_rnn_X | 100.5 | 151.6 | 151.88 | 3.0 | 1,284,000 |
| T4_act_ALL | 95.8 | 151.6 | 153.05 | 2.9 | 1,332,000 |
| T4_act_I | 79.4 | 147.6 | 149.81 | 3.5 | 1,480,000 |
| T4_act_X | 84.3 | 147.6 | 148.98 | 3.1 | 1,420,000 |
| T5_crt_ALL | 84.0 | 147.6 | 153.05 | 3.3 | 1,748,000 |
| T5_crt_I | 83.1 | 146.6 | 150.45 | 4.0 | 1,732,000 |
| T5_crt_X | 81.5 | 151.3 | 156.97 | 2.9 | 2,060,000 |
| T16_quad_ALL | 97.0 | 155.0 | 157.88 | 3.0 | 1,712,000 |
| T16_quad_I | 91.6 | 151.3 | 155.05 | 3.0 | 1,788,000 |
| T16_quad_X | 73.5 | 153.2 | 157.92 | 3.8 | 1,928,000 |

**Read this as a liveness check only.** The "last 200k" column is **not** comparable across rows —
each row's window sits at a different point in that run's training (the 2080 Ti arms are ~600k
episodes behind the Ada arms), and a run that is 20% through has simply had more time to climb than
one that is 13% through. Every run is climbing, none has flatlined or collapsed. That is the whole
finding.

**Losses.** Total loss 0.094–0.100, value loss 0.204–0.216, entropy −0.47 to −0.63 in every arm, all
finite and stable, with the control inside the pack on every channel.

### 5.2 Does the modulator receive gradient in all fifteen modulated arms? — Yes

| Cell | min ever | mean | max | first decile | last decile | share below 0.001 |
|---|---|---|---|---|---|---|
| T2_enc_ALL | 0.0272 | 0.1391 | 0.315 | 0.054 | 0.165 | 0% |
| T2_enc_I | 0.0260 | 0.1490 | 0.473 | 0.052 | 0.158 | 0% |
| T2_enc_X | 0.0298 | 0.1463 | 0.363 | 0.047 | 0.158 | 0% |
| T3_rnn_ALL | 0.0323 | 0.0578 | 0.102 | 0.050 | 0.062 | 0% |
| T3_rnn_I | 0.0212 | 0.0872 | 0.241 | 0.046 | 0.098 | 0% |
| T3_rnn_X | 0.0234 | 0.0481 | 0.116 | 0.048 | 0.044 | 0% |
| T4_act_ALL | 0.0100 | 0.0295 | 0.062 | 0.026 | 0.026 | 0% |
| T4_act_I | **0.0076** | 0.0425 | 0.157 | 0.013 | 0.059 | 0% |
| T4_act_X | 0.0117 | 0.0231 | 0.063 | 0.023 | 0.021 | 0% |
| T5_crt_ALL | 0.0097 | 0.0306 | 0.076 | 0.022 | 0.037 | 0% |
| T5_crt_I | 0.0079 | 0.0313 | 0.103 | 0.026 | 0.033 | 0% |
| T5_crt_X | 0.0082 | 0.0213 | 0.142 | 0.027 | 0.019 | 0% |
| T16_quad_ALL | 0.0534 | 0.1529 | 0.358 | 0.099 | 0.145 | 0% |
| T16_quad_I | 0.0336 | 0.1277 | 0.446 | 0.074 | 0.127 | 0% |
| T16_quad_X | 0.0330 | 0.1289 | 0.277 | 0.070 | 0.149 | 0% |

**The smallest value observed anywhere in the twin so far is 0.0076**, in the same cell that holds the
parent batch's minimum (`T4_act_I`, 0.0071) — about **seven and a half times** the vanishing
threshold. No site is disconnected in any twin arm, including the two sites that had never been run
before this week.

**Gradient norm against the ceiling.** Over each twin run's post-warm-up rows so far, the mean total
gradient is 0.24–0.44, the 99th percentile 0.28–0.60, the largest windowed mean 0.30–0.64, and the
largest in-window peak anywhere is **1.5**. **There is not a single excursion above 3.0 anywhere in
the twin**, against thirteen of sixteen runs having at least one in the parent batch — but the parent
batch's early window is comparably quiet (one event across sixteen runs before episode 2M), so this
is not yet a difference. Warm-up peaks are the same 9.1–32.7 as the parent's.

### 5.3 Do the per-site signals start at identity, and where are they heading?

**They start at identity, exactly as the parent batch does.** At the first logged point every enabled
site in every twin arm has a mean gain between **0.94 and 1.08** and a mean offset between **−0.06
and +0.05**, with a per-unit standard deviation of **0.28–0.39** for both — matching the parent's
0.29–0.37 and reproducing the §8.1 caveat exactly (identity holds for the layer mean, not per unit).

**Where they are heading so far** (mean of last five logged points; remember each run is at a
different progress point, so these are not comparable across rows):

| Site | ALL | I | X | four-site arm (ALL / I / X) | direction |
|---|---|---|---|---|---|
| encoder stage 1 gain | 0.733 | 0.537 | 0.633 | 0.714 / 0.585 / 0.667 | down from 1.0 |
| encoder stage 2 gain | 0.865 | 0.838 | 0.921 | 0.829 / 0.903 / 0.905 | down, mildly |
| memory-cell gain | 0.838 | 0.704 | 1.001 | 0.848 / 0.813 / 1.006 | down, mildly |
| **action-head gain** | **1.039** | **1.398** | **1.368** | 1.176 / 1.298 / 1.222 | **up from 1.0** |
| value-head gain | 0.391 | 0.371 | 0.635 | 0.517 / 0.271 / 0.562 | down, strongly |
| encoder stage 1 offset | −0.323 | −0.575 | −0.439 | −0.357 / −0.524 / −0.389 | down |
| memory-cell offset | −0.039 | −0.010 | +0.035 | +0.047 / +0.017 / +0.022 | ≈ 0 |
| action-head offset | −0.105 | −0.179 | −0.075 | −0.278 / −0.244 / −0.257 | down, mildly |
| value-head offset | −0.519 | −0.533 | −0.343 | −0.480 / −0.856 / −0.502 | down |

**The action-head gain rising above 1.0 is not an estimator effect.** It looks like the opposite of
the parent batch, where that gain ends at 0.50–0.72. But the parent batch's action-head gain was
**also above 1.0 at this stage** (1.05–1.33 over episodes 1.0–1.25M) and only fell below 1.0 later.
Compared at matched episodes the two batches agree (§5.4). This is a progress artifact, not a
difference — and it is a good illustration of why the twin comparison has to be done on matched
windows.

Every twin signal is still moving (all last-decile changes non-zero), as expected 13–20% into
training.

### 5.4 Do the twin cells behave alike, at matched training progress?

This is the check that only became possible once a second batch existed. Both batches use seed 42, so
a matched pair of cells starts from **bit-identical weights**; a large early divergence in modulator
gradient or gain trajectory would therefore be attributable to the return mode. All comparisons below
use the episode window **1,000,000 – 1,250,000**, which **every one of the thirty-two runs has
completed** (63 logged rows on each side of every pair).

**Answer: yes, they behave alike.** No health metric diverges materially.

| Cell | modulator gradient MC | GAE_NORM | ratio | total gradient MC | GAE_NORM |
|---|---|---|---|---|---|
| T2_enc_ALL | 0.1542 | 0.1620 | 1.05 | 0.402 | 0.347 |
| T2_enc_I | 0.1435 | 0.1696 | 1.18 | 0.248 | 0.271 |
| T2_enc_X | 0.1575 | 0.1580 | 1.00 | 0.354 | 0.411 |
| T3_rnn_ALL | 0.0782 | 0.0610 | 0.78 | 0.321 | 0.398 |
| T3_rnn_I | 0.0923 | 0.1065 | 1.15 | 0.252 | 0.227 |
| T3_rnn_X | 0.0539 | 0.0461 | 0.86 | 0.376 | 0.299 |
| T4_act_ALL | 0.0289 | 0.0298 | 1.03 | 0.384 | 0.385 |
| T4_act_I | 0.0582 | 0.0549 | 0.94 | 0.295 | 0.277 |
| T4_act_X | 0.0208 | 0.0219 | 1.05 | 0.368 | 0.451 |
| T5_crt_ALL | 0.0406 | 0.0322 | 0.79 | 0.389 | 0.409 |
| T5_crt_I | 0.0372 | 0.0288 | 0.77 | 0.383 | 0.407 |
| T5_crt_X | 0.0244 | 0.0220 | 0.90 | 0.377 | 0.331 |
| T16_quad_ALL | 0.1705 | 0.1617 | 0.95 | 0.329 | 0.298 |
| T16_quad_I | 0.1595 | 0.1426 | 0.89 | 0.265 | 0.239 |
| T16_quad_X | 0.1347 | 0.1339 | 0.99 | 0.389 | 0.436 |

**Modulator gradient agrees to within 0.77×–1.18× in all fifteen pairs**, with only a slight tilt:
five pairs are higher under the new estimator, nine lower, one identical (median ratio 0.95). Over a wider common window (episodes
200k–1.25M), the grid-mean total gradient is **0.351 (MC) vs 0.362 (GAE_NORM)** — a 3% difference
across thirty-two runs.

**The re-tuning signals also agree.** Over the same window, the largest disagreement between any twin
pair on any FiLM signal is **+0.255** (action-head gain, narrow-input arm: 1.249 under MC, 1.505
under GAE_NORM); the next largest is +0.176 (same signal, exteroceptive arm); every other one of the
60 compared signals agrees to within **0.14**, and most to within 0.05. The consistent direction of
those two largest gaps — the new estimator drives the action-head gain slightly further above 1.0 —
is worth noting for whoever analyses the twin scientifically, but it is a 20% relative difference on
a signal that both batches move in the same direction.

**The one systematic clipping difference.** Over the common post-warm-up window (episodes
200k–1.25M), the fraction of logging windows whose *mean* gradient exceeded the 0.5 ceiling is higher
under the new estimator in the outside-world-input and all-senses arms:

| Cell | MC | GAE_NORM |
|---|---|---|
| T1_none (control) | 0.0% | 10.4% |
| T2_enc_X | 0.7% | 26.9% |
| T4_act_X | 0.0% | 22.3% |
| T16_quad_X | 0.0% | 16.5% |
| T4_act_ALL | 1.4% | 11.0% |
| T3_rnn_X | 0.0% | 6.2% |
| all five `I` (narrow-input) arms | 0.0% | 0.0% |

**How to read this, and how not to.** The effect is present in the **control** as well as the
modulated arms, so it is a property of the estimator, not of the modulator — the new recipe's
advantage signal simply carries slightly more energy at this stage of training. It stays inside the
healthy regime throughout: the 99th percentile never exceeds 0.60, the largest in-window peak
anywhere in the twin is 1.5, and there are no excursions above 3.0 at all. The narrow-input arms are
unaffected in both batches, which reproduces the §4.4 ordering under a second estimator. **It is an
engineering observation about how the estimator interacts with the trainer's clip, and it is not a
behavioural claim.**

**Survival at matched episodes**, for completeness and as a liveness cross-check only: the twin cell
is ahead of its parent in all sixteen pairs, by 0.9 to 10.4 steps (mean +5.2). That is consistent
with the direction the five-seed return-mode study measured on unmodulated agents, but **it is one
seed per cell at 12% of training and must not be reported as a result**; it is included here solely
because a twin that was *behind* everywhere would have been a health signal worth chasing.

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
*worse*: not one of its sixteen runs succeeded in reading the flag.

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

### 7.2 GAE_NORM batch — the progress spread is entirely the card

| Card | Node | Cells | steps/s (range) | progress (range) |
|---|---|---|---|---|
| RTX 2080 Ti | 101, 103, 104, 105 | control, enc_ALL, rnn_{ALL,I,X}, act_{ALL,I,X} | 29,401 – 36,896 | 12.8 – 15.5% |
| RTX 3090 | 106 | crt_ALL, crt_I | 41,571 – 42,266 | 17.3 – 17.5% |
| RTX 4090 | 102 | enc_I, enc_X | 49,262 – 49,377 | 18.4 – 18.8% |
| RTX 6000 Ada | 114 | crt_X, quad_{ALL,I,X} | 45,231 – 52,736 | 17.1 – 20.6% |

Card separates the progress groups without a single inversion. **This is why §1 forbids cross-arm
comparison within this batch** — an arm that looks ahead of a sibling is simply on a better card.
Wall-clock gaps between consecutive logged rows are 5.0–8.2 s median, 11.9–20.2 s at the 99th
percentile, maximum 19.4–38.6 s; the largest gap in any twin run is 7.5× its own median (on the fast
4090 node, where the fixed-cost checkpoint step is a larger multiple of a smaller median). **No
stall signature.**

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
**0.29–0.39** in every GAE_NORM arm; of the offset, **0.27–0.35** and **0.28–0.35** respectively. So
on day one a typical unit is being multiplied by something in the region of 0.7 to 1.3 and shifted by
±0.3, rather than left alone.

**Why this is a note and not a bug report.** The code comment at
`src/models/neuromodulator.py:178-181` explicitly claims "a newly-enabled site is a no-op at step 0",
and taken per-unit that claim is not exact. But nothing observed suggests it caused harm — every arm
in both batches learned normally from the start, and the arms differ from their controls in their
first-million survival by amounts that are not ordered by how many sites they enable. The correct
response is to **state the property accurately** in the designs' reasoning rather than to change the
initialisation mid-experiment.

**What would confirm the alternative** (that the random per-unit deviation does matter): re-run one
cell with the gain and offset heads' weight matrices initialised to zero, making the site an exact
identity at step 0, and compare the first-million-episode survival curve. That is a one-run control,
not a code change to either batch.

### 8.2 The gain drift, restated with the finished data

Covered in §4.3. Restating the discipline: a falling FiLM gain is **not** evidence of a layer being
silenced, because the layer's own weights are free to grow and absorb the change. The observation is
"the mean gain fell to 0.203–0.909 depending on site and arm, then **plateaued or partly recovered at
every site except the action head**, while the offsets at the action and value heads under all-senses
input were **still sliding when the budget ran out**". The inference "the site is silencing its
layer" is **not** supported by anything currently logged. §9 asks for the two scalars that would let a
future reader tell these apart — and the fact that two signals had not converged makes that request
more urgent for the next wave.

---

## 9. Metrics Requested

Four metrics whose absence bounded this audit. All are cheap. None is required for either batch's
pre-registered analysis to proceed.

| | Metric | Why now | Where it'd live | Cost |
|---|---|---|---|---|
| 1 | `modulator/gamma_<site>_temporal_std` and `modulator/gamma_<site>_unit_std` — the standard deviation of the site's gain **across time/batch after averaging over units**, and **across units after averaging over time**, in the same units as the existing gain (dimensionless multiplier) | The single currently-logged spread mixes these two components, so it cannot distinguish a modulator that genuinely re-tunes the network moment-to-moment from one that has learned a fixed per-unit re-parameterisation and is functionally inert. This is exactly the question the narrow-input (`I`) arms raise in §4.4 — and that question is now doubly live, because the same ordering reproduced under a second estimator (§5.4) without becoming any easier to interpret. It is also the discriminator that keeps a conditional behaviour from being averaged away | `train.py` around the `_mod_spread` helper at line ~1848, where the existing mean/std over `mod_info` are already computed | cheap — two extra reductions over an array that is already resident, at the existing logging cadence |
| 2 | `network/<site>_postfilm_active_fraction` — the fraction of the modulated layer's units whose post-FiLM pre-activation is positive, i.e. surviving the rectifier (dimensionless, 0–1) | §4.3 now shows the completed picture: gains near 0.2–0.4 combined with offsets near −1.0 to −1.8 at the action and value heads, **with those offsets still moving downward at the last logged step**. Whether that has silenced a large part of those layers, or is a harmless scale re-partition absorbed by the layer's own weights, is currently unanswerable — and the two possibilities have opposite implications for reading both batches' results. This is the highest-value of the four | `src/models/recurrent_ppo_network.py`, immediately after the FiLM applications at lines 553 and 564 | cheap — one comparison and one mean per site per iteration |
| 3 | `loss/grad_norm_p99` and `loss/grad_norm_clipped_fraction` — the 99th percentile of the **per-update** pre-clip gradient norm within a logging window, and the fraction of individual updates whose norm exceeded `max_grad_norm` (L2 norm; fraction 0–1) | Only the per-iteration mean and the window peak are kept, so §4.5's and §5.4's clipping statements are bounded above and below rather than measured, and §4.8's 1030 excursion cannot be localised to a single minibatch. The return-mode study established clipping fraction as the decisive variable separating arms that learn from arms that crawl — and this audit now has a *second* return mode in flight whose clipping regime differs from the first's precisely in the upper tail (§5.4), which is the part the current logging measures worst | `src/models/recurrent_ppo_trainer.py` around line 375, where `grad_norm` is already computed per update, plus the windowing in `train.py` | cheap — the per-update norms already exist; this is an extra reduction over them |
| 4 | `modulator/param_count` and `network/param_count` at startup (integers, logged once) | §7 attributes an 11–26% throughput cost to the modulator, but the audit cannot state the parameter cost of each site configuration because it is not recorded; a future reader comparing throughput across site configurations has to reconstruct it from the architecture | `train.py`, in the startup banner block that already prints the modulation configuration | cheap — one scalar each, logged once |

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
2. **Provenance dirty-flag is unreliable under simultaneous multi-run launch — now reproduced**
   (§6.2). `src/utils/provenance.py:99-102` runs `git status` with a 10 s timeout per training
   process; sixteen concurrent launches produced `"unknown"` in 15/16 on the first batch and
   **16/16** on the second. Suggested route: a bug-fix plan under `docs/develop/active/`. Options
   worth weighing: retry on timeout, raise the timeout for this call specifically, or have the
   launcher read the flag once and pass it to every child process. The current behaviour is *safe*
   (it never claims clean when it does not know) — the defect is that it makes an experiment-design
   gate unsatisfiable, twice.

Also worth surfacing to whoever owns the design documents, though neither is a defect:

3. **Neither Launch Manifest was filled in.** All sixteen rows of [[NMN_INPUT_SITE_GRID]] §3 and all
   sixteen of [[NMN_INPUT_SITE_GRID_GAENORM]] §3 still read `planned` with empty node, GPU, run-ID
   and log-path columns. §2.1 and §2.2 above reconstruct the actual values from the runs' own
   metadata. Per the manifests' stated ownership, only `training-runner` writes those columns; this
   audit has not touched them.
4. **The designs' step-0 identity argument should be restated** (§8.1). "Turning a site on cannot
   hurt at step 0" holds for the layer mean but not per unit, where the gain starts at 1.0 ± ~0.3.
   Confirmed independently in both batches.

---

## 11. Bottom line

**Batch A, the MC grid, is finished and healthy.** All sixteen runs completed 10,000,000 episodes.
No run needs to be discarded. The wiring is correct in all sixteen, verified three independent ways.
The modulator received gradient throughout in all fifteen modulated arms — the smallest value seen
anywhere across the complete histories is 0.0071, seven times the vanishing threshold. Nothing
diverged, nothing stalled, nothing was crushed by the gradient clip, and the four-simultaneous-sites
configuration is if anything the calmest arm. The control lands inside its pre-registered reference
band at 164.46 survival steps, so the batch has a valid reference point. The single 1030 gradient
spike stayed a one-off, and only one further excursion above 3.0 (peak 6.6) appeared as the last ten
runs finished.

**The refined finding the completed data allowed.** The downward drift of the re-tuning signals was
not monotone to the end: the front-end gains bottomed out between 25% and 60% of training and
recovered by up to +0.17, the memory-cell and value-head gains plateaued, and only the action-head
gain is still creeping down. Two signals had **not** converged when the episode budget expired — the
additive offset at the action head and at the value head, in the all-senses arms, still moving at
roughly 11% and 7% of their final magnitude per final tenth of training. That is the strongest reason
to add the post-FiLM active-fraction scalar (§9, item 2) before the next wave.

**Batch B, the GAE_NORM twin, is healthy at 12.8–20.6%.** All sixteen learning, zero non-finite
values, zero recompiles, zero constant series, correct metric set in every arm, modulator gradient
alive in all fifteen modulated arms with a minimum of 0.0076, and every site starting at identity
exactly as the parent did. Its progress spread is fully explained by card assignment, so nothing
within it may be compared arm-to-arm yet.

**The twin comparison, kept to engineering.** At matched episodes the two batches' health metrics
agree closely — modulator gradient within 0.77×–1.18× in all fifteen pairs, grid-mean total gradient
differing by 3%, and 58 of 60 re-tuning signals agreeing to within 0.14. The one systematic
difference is that under the new estimator the outside-world-input and all-senses arms spend
10–27% of their logging windows with a mean gradient above the 0.5 clip ceiling, against 0–4% under
the old one — an effect that is **also present in the unmodulated control**, so it belongs to the
estimator rather than to the modulator, and one that stays comfortably inside the healthy regime.

---

## Appendix

### A. Working files

Raw extractions, retained for traceability (gitignored):

- `tmp/nmnsite2/extract.py` — the local transaction-log reader used here (no WandB web API). Stores
  every series as `(row index, value)` pairs so that metrics emitted in different history records can
  be aligned to an episode count by forward fill.
- `tmp/nmnsite2/lib.py` — shared cell↔run-ID maps and the episode-aligned column accessor.
- `tmp/nmnsite2/{mc,gn}_<runid>.json` — per-run complete series, key inventory and non-finite scan.
- `tmp/nmnsite2/r1.py` … `r9.py` — the nine analyses this document is built from (progress and
  survival; modulator gradient and clipping; FiLM temporal tables; matched-band twin comparison;
  metric-set audit; excursion census and throughput; constant-series and wall-clock-gap scan;
  turning-point analysis; common-window clipping comparison).
- `tmp/nmnsite2/cfgdiff.py` — the flattened key-by-key config differ used in §2.3.
- `tmp/20260907_172348_nmnsite_audit_refresh.md` — consolidated output of all nine.

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
| 2026-09-07 | **Refresh.** MC grid now 16/16 complete — every reported figure recomputed on full histories (§4). Added the drift turning-point analysis that the partial data could not support (§4.3.2): drift mostly plateaued or reversed; two offsets had not converged. Confirmed the 1030 spike stayed a one-off and censused the one new excursion (§4.8). Added Batch B, the GAE_NORM twin, at 12.8–20.6%: manifest (§2.2), one-key config verification on three twin pairs (§2.3), full early-life health check (§5), matched-episode twin comparison (§5.4), throughput (§7.2). Provenance defect reproduced at 16/16 (§6.2). |
