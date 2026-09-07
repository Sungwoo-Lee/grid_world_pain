---
title: "Training-health and engineering-correctness audit of the 16-run modulation-site grid"
topic: nmn_input_site_grid
status: active
created: 2026-09-07
last_updated: 2026-09-07
wandb_group: nmn_input_site_grid
wandb_tag: "rppo_nmnsite_*_s42"
develop_link: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
---

# Training-health and engineering-correctness audit of the 16-run modulation-site grid

## 0. What this document is, and what it found

**The question.** Sixteen training runs were launched this morning on brand-new code — a rewrite
that lets the agent's small "neuromodulator" network be pointed at four different places inside the
main policy network, and lets it be fed a chosen subset of the agent's senses. Because the code is
one day old, the first thing worth knowing is not *which arrangement works best* but **whether the
machinery is working at all**: is every run learning, is every switch actually wired to the thing it
claims to switch, is any number diverging, is anything running unaccountably slowly. That is this
document. The scientific comparison between the sixteen arrangements is a separate piece of work and
is deliberately not attempted here.

**The verdict: the machinery is working.** Across all sixteen runs there is not a single
not-a-number or infinite value anywhere in any logged quantity; every run's survival time rose from
about 18 steps at the start to roughly 163–171 steps by the end; the sixteen configurations each
logged exactly the signals their configuration says they should and none that they should not; and
in every one of the fifteen runs that has a modulator, the modulator is receiving real training
signal — its gradient never once fell to the "disconnected" level, with the smallest value observed
anywhere being about seven times the threshold the project's own metrics guide calls vanishing.
Nothing here justifies stopping, restarting or discarding a run.

**The single most reassuring result, because it was the most likely defect.** When a brand-new
wiring change is switched on, the classic silent failure is that a component is nominally enabled but
is not actually connected to the thing being optimised — it sits in the network doing nothing, and
the survival numbers look perfectly normal because the rest of the network compensates. That failure
mode is ruled out: the modulator's gradient is healthily non-zero in all fifteen modulated runs, at
all four of the new write sites, including the two sites (the action-choosing layer and the
value-estimating layer) that had never been run before today.

**Three things are worth the user's attention, none of which is a broken run:**

1. **The claim that switching a site on is harmless at the start of training is true on average
   but not exactly true.** The new re-tuning sites start with a gain of one and an offset of zero
   *as an average across the layer's 128 units*, which is what makes the arrangement a no-op. But
   the layer of the modulator that produces those numbers has randomly-initialised weights, not zero
   weights, so at the very first update each individual unit's gain is one **plus a random deviation
   of about ±0.3**, and each unit's offset is zero plus a similar random deviation. The design
   document's reasoning that "turning a site on cannot hurt at step 0" therefore holds in
   expectation, not literally. This does not invalidate anything, but it should be stated rather
   than assumed. Detail in §5.1.
2. **The project's own metrics guide is now out of date in a way that would make a future reader
   misinterpret these runs.** It says the gain numbers logged for the sensory front-end are
   "pre-sigmoid" and must be squashed through a sigmoid function before being read as an actual
   gain. Under the new code that is no longer true for these runs — the front-end now uses the same
   plain multiply-and-add as the other three sites, so the logged number *is* the gain. Anyone who
   applies the documented correction will get the wrong answer. Detail in §6.1.
3. **The record of exactly what code state each run was launched from is incomplete.** Every run
   agrees on the code version, and the code at that version is confirmed identical to the two
   commits the design named as its prerequisites. But the launcher's check for "were there
   uncommitted edits at launch time" timed out in fifteen of the sixteen runs and recorded the
   answer as "unknown"; the one run whose check succeeded says there *were* uncommitted edits. The
   experiment design set a gate requiring all sixteen to record "no uncommitted edits", and that
   gate cannot be met from the recorded evidence. §6.2 explains why this is very probably harmless
   here, and what was checked instead.

**Progress at the time of the audit.** Six runs have completed their full ten million episodes; the
other ten are between 84% and 99% complete. Every number in this document uses each run's complete
recorded history, and every partial run is labelled with its percentage.

**The control run passes the reference gate.** The unmodulated control finished at **164.5 mean
survival steps** over its final million episodes, inside the 162.4–166.9 band the design registered
in advance from five previously-run seeds of the same configuration. Independently confirmed here;
see §4.1.

---

## 1. Scope

**In scope.** Numerical health (finite values, divergence, stalls), wiring correctness (does the
metric set match the configuration), gradient health of the new modulator paths, gradient clipping
regime, throughput anomalies, and anything that looks like a code defect.

**Out of scope, deliberately.** Which write site or which input slice produces better survival or
more context-dependent behaviour. That is the pre-registered analysis in
[[NMN_INPUT_SITE_GRID]] and belongs to a different session. Survival numbers appear here only
(a) as a liveness check that a run is learning at all, and (b) to confirm the control's reference
band. **There is one random seed per cell**, so no comparison between two arms in this document is a
claim about which arm is better — a 5-seed study of this same configuration measured a
seed-to-seed spread of about 4.5 survival steps, and most gaps in the table below are inside that.

**Project rule observed.** Performance is read in survival steps, never cumulative reward.

---

## 2. What was audited

Sixteen runs, WandB group `nmn_input_site_grid`, job type `pilot`, all seed 42, all on the
10x10 jump-attack environment config, all 10,000,000 episodes, all at code commit `a71f4471`.
One unmodulated control plus fifteen modulated arms crossing five write targets (sensory front-end
/ memory cell / action head / value head / all four) with three input slices (all 27 sensory
numbers "ALL", the 2 interoceptive numbers "I", the 19 exteroceptive numbers "X").

### 2.1 Run manifest as observed

The design document's Launch Manifest (§3 of [[NMN_INPUT_SITE_GRID]]) is still showing every row as
`planned` with empty run-ID and node columns — the launching agent did not fill it in. The table
below is what the runs' own metadata say, recovered from `wandb/run-*/files/wandb-metadata.json`
and each run's `models/provenance.json`. **It is offered as evidence for whoever owns the manifest;
this audit does not write to the manifest.**

| Cell | Tag | WandB run ID | Local log dir | Node:GPU | Card | Episodes done | % |
|---|---|---|---|---|---|---|---|
| T1_none (control) | `rppo_nmnsite_t1none_s42` | `anpfno02` | `wandb/run-20260907_045542-anpfno02` | 106:0 | RTX 3090 | 10,000,000 | 100% |
| T2_enc_ALL | `rppo_nmnsite_t2enc_ALL_s42` | `pom30693` | `wandb/run-20260907_050232-pom30693` | 106:1 | RTX 3090 | 9,624,000 | 96% |
| T2_enc_I | `rppo_nmnsite_t2enc_I_s42` | `9f8wx4b9` | `wandb/run-20260907_050233-9f8wx4b9` | 107:0 | RTX 3090 | 9,968,000 | 99.7% |
| T2_enc_X | `rppo_nmnsite_t2enc_X_s42` | `kir9fubn` | `wandb/run-20260907_050234-kir9fubn` | 107:1 | RTX 3090 | 9,120,000 | 91% |
| T3_rnn_ALL | `rppo_nmnsite_t3rnn_ALL_s42` | `3ed43b2i` | `wandb/run-20260907_050235-3ed43b2i` | 108:0 | RTX 3090 | 9,440,000 | 94% |
| T3_rnn_I | `rppo_nmnsite_t3rnn_I_s42` | `84964jnn` | `wandb/run-20260907_050235-84964jnn` | 108:1 | RTX 3090 | 8,472,000 | 85% |
| T3_rnn_X | `rppo_nmnsite_t3rnn_X_s42` | `ryt9z2lo` | `wandb/run-20260907_050237-ryt9z2lo` | 109:0 | RTX 3090 | 9,844,000 | 98% |
| T4_act_ALL | `rppo_nmnsite_t4act_ALL_s42` | `p6yc4akb` | `wandb/run-20260907_050238-p6yc4akb` | 109:1 | RTX 3090 | 9,040,000 | 90% |
| T4_act_I | `rppo_nmnsite_t4act_I_s42` | `qeesiize` | `wandb/run-20260907_050239-qeesiize` | 110:0 | RTX 3090 | 10,000,000 | 100% |
| T4_act_X | `rppo_nmnsite_t4act_X_s42` | `skmnxb2h` | `wandb/run-20260907_050239-skmnxb2h` | 110:1 | RTX 3090 | 9,772,000 | 98% |
| T5_crt_ALL | `rppo_nmnsite_t5crt_ALL_s42` | `qug0fubt` | `wandb/run-20260907_050240-qug0fubt` | 111:0 | RTX 3090 | 10,000,000 | 100% |
| T5_crt_I | `rppo_nmnsite_t5crt_I_s42` | `hi2ly2sh` | `wandb/run-20260907_050241-hi2ly2sh` | 111:1 | RTX 3090 | 9,728,000 | 97% |
| T5_crt_X | `rppo_nmnsite_t5crt_X_s42` | `f5qlrnql` | `wandb/run-20260907_050242-f5qlrnql` | 112:0 | RTX 3090 | 10,000,000 | 100% |
| T16_quad_ALL | `rppo_nmnsite_t16quad_ALL_s42` | `b2cen70a` | `wandb/run-20260907_050243-b2cen70a` | 112:1 | RTX 3090 | 8,496,000 | 85% |
| T16_quad_I | `rppo_nmnsite_t16quad_I_s42` | `ldsttwlo` | `wandb/run-20260907_050243-ldsttwlo` | 113:0 | RTX 4090 | 10,000,000 | 100% |
| T16_quad_X | `rppo_nmnsite_t16quad_X_s42` | `hxey8k3h` | `wandb/run-20260907_050243-hxey8k3h` | 113:1 | RTX 4090 | 10,000,000 | 100% |

Episode counts are as of the last record flushed to each run's local transaction log at the time of
this audit; the ten unfinished runs are still advancing.

**Hardware note that matters for §4.7.** The two runs on node 113 are on RTX 4090 cards; the other
fourteen are on RTX 3090s. Throughput cannot be compared across that boundary.

### 2.2 Method

Everything below is read from **local files only** — no WandB web API was queried, per project
convention. Three independent sources were cross-checked, which matters because two of them share a
code path and the third does not:

1. Each run's WandB transaction log (`wandb/run-*/run-*.wandb`), parsed directly with the
   `wandb` package's on-disk datastore reader. This gives the complete metric history, not a
   downsampled web view.
2. Each run's **trainer-written resolved configuration** (`results/JAX_RecurrentPPO/<run>/models/config.yaml`)
   — the config the trainer actually built the model from, not a fresh reload of the source YAML.
3. Each run's **runtime banner**, printed by the training process at startup and captured in
   `wandb/run-*/files/output.log`, which states the modulation type, input sensors and enabled
   sites as the constructed model reports them.

Working extractions: `tmp/nmnsite/`.

---

## 3. Wiring correctness — does each run log exactly what its configuration says?

**Result: exact match, sixteen for sixteen.** This is the strongest single piece of evidence that
the new site-selection code is wired correctly, because a site that is enabled emits its own pair of
signals and a site that is disabled emits nothing at all — so the *set* of series present is a
direct readout of which sites the model actually built.

| Cell | Sites declared in the run's own saved config | Modulator series actually present | Match |
|---|---|---|---|
| T1_none | (no modulation block) | none at all | ✅ |
| T2_enc_{ALL,I,X} | encoder only | `gamma_uni_*`, `beta_uni_*`, `gamma_multi_*`, `beta_multi_*`, `grad_norm` | ✅ |
| T3_rnn_{ALL,I,X} | rnn only | `gamma_rnn_*`, `beta_rnn_*`, `grad_norm` | ✅ |
| T4_act_{ALL,I,X} | actor only | `gamma_actor_*`, `beta_actor_*`, `grad_norm` | ✅ |
| T5_crt_{ALL,I,X} | critic only | `gamma_critic_*`, `beta_critic_*`, `grad_norm` | ✅ |
| T16_quad_{ALL,I,X} | encoder + rnn + actor + critic | all ten pairs above + `grad_norm` | ✅ |

No arm logs a series for a site it did not enable. In particular no `gamma_actor_*` appears in an
encoder-only arm, which was the specific defect the brief asked to rule out.

**The two absence checks also pass.** `modulator/temperature_*` is absent from all sixteen runs
(every config sets `temperature.enabled: false`), and `modulator/z_memory_*` is absent from all
sixteen (that series belongs to the legacy gate-bias mechanism; every run here uses
`rnn_mechanism: activation`). Both are expected absences per the metrics reference §2.8.

**Non-modulator metric sets are byte-identical across all sixteen runs** — the same 59 keys in
every run. No arm is missing an episode or loss metric its siblings have.

**Input slices verified from the runtime, not from the source YAML.** The startup banner in each
run reports the sensor list the constructed modulator was given:
`[all]` for the six ALL arms, `[Satiation, Interoceptive Nociception]` for the five I arms, and
`[Extero Nociception, Olfaction, Collision, Visual]` for the five X arms. All match the design's
arm table exactly.

**Code prerequisites verified.** The design required the launch commit to have the modulation-site
refactor's Part B as an ancestor and to differ from it nowhere under `src/`. Both hold:
`e1aab726` and `83b8140b` are both ancestors of the launch commit `a71f4471`, the diff of `src/`
between `e1aab726` and `a71f4471` is empty, and `src/` has not changed since. The one requirement
that is *not* satisfied by the evidence is the clean-working-tree flag — see §6.2.

---

## 4. Findings, question by question

### 4.1 Is every run actually learning? — Yes, all sixteen

No run has flatlined, stalled, or diverged. Every run's survival time rises through the same shape:
a fast climb through the first million episodes, then a long slow improvement that is still very
slightly positive at the end.

Survival steps (the rolling 5,000-episode mean the trainer logs), averaged within episode bands:

| Cell | 0–1M | 1–2M | 2–5M | 5–9M | final 1M | s.d. within final 1M | completion |
|---|---|---|---|---|---|---|---|
| T1_none (control) | 81.5 | 146.1 | 158.4 | 163.1 | **164.46** | 3.9 | 100% |
| T2_enc_ALL | 66.0 | 147.5 | 161.6 | 167.3 | 169.21 | 3.5 | 96% |
| T2_enc_I | 88.0 | 146.4 | 155.2 | 160.7 | 163.38 | 3.5 | 99.7% |
| T2_enc_X | 92.7 | 152.5 | 162.1 | 166.4 | 166.78 | 4.1 | 91% |
| T3_rnn_ALL | 86.9 | 155.2 | 160.9 | 163.9 | 164.91 | 3.6 | 94% |
| T3_rnn_I | 71.7 | 146.5 | 157.3 | 163.4 | 165.00 | 3.5 | 85% |
| T3_rnn_X | 88.1 | 148.2 | 158.7 | 164.3 | 165.36 | 3.2 | 98% |
| T4_act_ALL | 83.4 | 152.5 | 162.1 | 166.0 | 165.51 | 3.6 | 90% |
| T4_act_I | 67.6 | 146.0 | 157.6 | 163.4 | 165.83 | 3.1 | 100% |
| T4_act_X | 84.1 | 146.9 | 156.4 | 162.3 | 165.03 | 3.4 | 98% |
| T5_crt_ALL | 83.1 | 145.3 | 157.5 | 163.5 | 165.43 | 3.3 | 100% |
| T5_crt_I | 86.8 | 143.6 | 155.0 | 161.6 | 163.03 | 3.3 | 97% |
| T5_crt_X | 83.5 | 146.9 | 158.9 | 164.4 | 165.74 | 3.1 | 100% |
| T16_quad_ALL | 82.5 | 152.6 | 161.9 | 167.3 | 168.61 | 3.2 | 85% |
| T16_quad_I | 77.2 | 146.0 | 154.6 | 160.8 | 163.04 | 3.4 | 100% |
| T16_quad_X | 75.5 | 151.0 | 163.8 | 168.4 | 170.58 | 3.2 | 100% |

**How to read the "final 1M" column, and how not to.** It is the mean of the rolling-window survival
series over each run's last million episodes; the s.d. column is the scatter of that rolling series,
**not** a seed-to-seed uncertainty. With one seed per cell there is no seed-to-seed uncertainty
available at all. The whole 16-arm spread is 163.0 to 170.6, i.e. 7.6 steps, against a
previously-measured 5-seed spread of about 4.5 steps for this configuration. Read that as "the arms
are all in the same performance neighbourhood", not as a ranking.

**Control gate confirmed (independently of the coordinator's read).** The control's final-million
mean of **164.46** falls inside the pre-registered 162.4–166.9 reference band from five earlier
seeds of the identical unmodulated configuration. The grid's reference point is sound.

**Losses.** Total loss, value loss, policy loss and entropy are finite and stable in every run. Total
loss sits at 0.104–0.114 in every arm at the end (control 0.108); value loss at 0.225–0.245 (control
0.234); entropy at −0.60 to −0.70 (control −0.66) with no sign of premature collapse toward zero.
Nothing separates the modulated arms from the control on any loss channel by more than the
run-to-run scatter.

**Zero non-finite values.** A full scan of every numeric value in every history record of all
sixteen transaction logs found **no NaN and no infinity**, in any metric, at any step.

**Largest survival drawdown.** In every run, including the control, the largest fall from a running
maximum after the 2-millionth episode is 20–28 steps (control: 24.8). These are rolling-window
excursions on a series whose per-episode standard deviation is around 200 steps; nothing here is a
collapse.

### 4.2 Does the modulator receive gradient in every modulated arm? — Yes, in all fifteen

This was the brief's most-likely-defect. It is not present.

`modulator/grad_norm` is the L2 norm of the gradient of the loss with respect to the modulator's own
parameters, computed before clipping (`src/models/recurrent_ppo_trainer.py:385`). The metrics
reference calls anything below 0.001 vanishing.

| Cell | min ever | mean | max | share of logged points below 0.001 |
|---|---|---|---|---|
| T2_enc_ALL | 0.0362 | 0.173 | 0.674 | 0% |
| T2_enc_I | 0.0327 | 0.177 | 0.692 | 0% |
| T2_enc_X | 0.0313 | 0.171 | 2.251 | 0% |
| T3_rnn_ALL | 0.0358 | 0.086 | 0.165 | 0% |
| T3_rnn_I | 0.0233 | 0.123 | 0.313 | 0% |
| T3_rnn_X | 0.0302 | 0.058 | 0.113 | 0% |
| T4_act_ALL | 0.0173 | 0.027 | 0.075 | 0% |
| T4_act_I | **0.0071** | 0.060 | 0.203 | 0% |
| T4_act_X | 0.0102 | 0.022 | 0.048 | 0% |
| T5_crt_ALL | 0.0138 | 0.038 | 0.108 | 0% |
| T5_crt_I | 0.0114 | 0.035 | 0.111 | 0% |
| T5_crt_X | 0.0088 | 0.027 | 0.095 | 0% |
| T16_quad_ALL | 0.0567 | 0.190 | 0.540 | 0% |
| T16_quad_I | 0.0328 | 0.157 | 0.423 | 0% |
| T16_quad_X | 0.0368 | 0.197 | 1.355 | 0% |

The **smallest value observed anywhere across all fifteen modulated runs and their full histories is
0.0071**, about seven times the vanishing threshold. No site is disconnected.

**The two never-before-run sites are both live.** The action-head site (`gamma_actor_*`) and the
value-head site (`gamma_critic_*`) both carry gradient in every arm that enables them, and both
show their signals moving substantially over training (§4.3) — which is a second, independent
confirmation that they are in the computational graph, since a disconnected site's signals would
stay pinned at their initialisation.

The action-head site carries the *smallest* modulator gradient of the four (mean 0.022–0.026 in the
ALL and X arms) — roughly an order of magnitude below the front-end site. That is a plausible
consequence of where it sits (a single hidden layer close to the output, downstream of everything
else), not a defect, and it is still 20× above the vanishing threshold.

### 4.3 Are the gain and offset signals staying in a sane range? — Yes, but every site drifts in the same direction, and the drift is large

**What the numbers mean.** Each enabled site multiplies its layer by a per-unit gain (γ) and then
adds a per-unit offset (β). Under the new FiLM code these logged values are the gains and offsets
**as applied** — there is no sigmoid squashing in between (verified in
`src/models/recurrent_ppo_network.py:218-263, 548, 553, 564`). A gain of 1.0 with an offset of 0.0
leaves the layer untouched.

**At the first logged point** — iteration 50, the earliest sample WandB holds — every site in every
arm has a mean gain between **0.95 and 1.05** and a mean offset between **−0.05 and +0.03**. That is
the identity arrangement, still essentially unmoved after 200 parameter updates. Combined with the
code (`src/models/neuromodulator.py:142-146, 178-186`, which sets the gain head's bias to exactly
1.0 and the offset head's bias to exactly 0.0 under FiLM), the "starts as a no-op" property holds
**for the layer mean**. The caveat about the per-unit spread is §5.1 and it is important.

**End-of-training mean gain, by site.** Every site in every arm drifts downward, monotonically, and
none reverses:

| Site | ALL | I | X | single-site range | in the four-site arm |
|---|---|---|---|---|---|
| encoder, stage 1 (`gamma_uni_mean`) | 0.72 | 0.39 | 0.83 | 0.39 – 0.83 | 0.59 / 0.61 / 0.68 |
| encoder, stage 2 (`gamma_multi_mean`) | 0.76 | 0.73 | 0.89 | 0.73 – 0.89 | 0.54 / 0.82 / 0.73 |
| memory cell (`gamma_rnn_mean`) | 0.48 | 0.43 | 0.67 | 0.43 – 0.67 | 0.72 / 0.63 / 0.74 |
| action head (`gamma_actor_mean`) | 0.55 | 0.67 | 0.68 | 0.55 – 0.68 | 0.52 / 0.72 / 0.68 |
| value head (`gamma_critic_mean`) | 0.32 | **0.20** | 0.62 | 0.20 – 0.62 | 0.37 / 0.26 / 0.40 |

(the three numbers in the last column are the four-site arm under the ALL / I / X input slices)

**End-of-training mean offset**, same layout — all negative, largest at the two new sites:

| Site | ALL | I | X |
|---|---|---|---|
| encoder, stage 1 (`beta_uni_mean`) | −0.69 | −1.07 | −0.65 |
| encoder, stage 2 (`beta_multi_mean`) | −0.22 | −0.33 | −0.18 |
| memory cell (`beta_rnn_mean`) | +0.03 | −0.004 | +0.003 |
| action head (`beta_actor_mean`) | **−1.63** | −0.52 | −0.79 |
| value head (`beta_critic_mean`) | **−1.80** | −0.81 | −1.57 |

**Is this collapse?** Not by the evidence available, and probably not at all — but it is the pattern
most worth a targeted check. Three reasons for the "probably not":

- The gains are all still comfortably positive. Nothing is near zero, nothing has gone negative,
  nothing has exploded. There is no runaway.
- A FiLM gain is not identifiable on its own. If the modulator halves a layer's pre-activation, the
  layer's own weights can double and the network is unchanged. **A falling mean gain is therefore
  not by itself evidence that the layer is being silenced** — it is equally consistent with a scale
  re-partition between the two. The project's metrics guide has no pre-registered healthy band for
  a linear FiLM gain (the band it does publish, 1.0–3.0, is for the *old* pre-sigmoid quantity and
  does not apply here), so there is no threshold this crosses.
- Everything downstream looks normal. The value-head arms, where the gain drops furthest, have value
  losses indistinguishable from the control's (0.230–0.242 vs 0.234) and survival inside the pack.

The reason it still deserves a check is the **combination** of a gain near 0.2–0.4 with an offset
near −1.0 to −1.8 at a layer that is then passed through a rectifier: if the layer's own
pre-activation is O(1), that combination pushes a large fraction of its units permanently negative,
i.e. permanently off. That would be a real sparsification of the value head, and it would be
invisible in every metric currently logged. §7 requests the two cheap scalars that would settle it.

**Spread rises with drift.** The per-unit standard deviation of the gain roughly doubles over
training at every site (e.g. front-end stage 1: 0.30–0.37 at the start → 0.72–1.28 at the end), and
the offset's spread rises further still — the value-head offset in one arm reaches a spread of
2.47. Rising spread alongside a falling mean is the signature of the modulator learning to treat
different units differently, which is what it is for. It is also, per the metrics guide's own
caveat, exactly the situation in which a mean alone is misleading.

**Nothing crosses a published pathological threshold.** The only per-site band the metrics reference
publishes that applies to these runs is the one for the gradient norm, and §4.5 shows every arm
inside it.

### 4.4 Do the narrow-input arms behave differently? — Yes, consistently, but they are not degenerate

The five `I` arms give the modulator only two numbers to read: how full the agent is, and a smoothed
internal ache. The brief's worry was that two dimensions might not be enough to learn from, leaving
a modulator that emits a near-constant signal and receives no gradient.

**That worry is not borne out.** The narrow-input modulators are, if anything, *more* strongly
driven than the wide-input ones:

- Their own gradient is comparable or larger at every site: front-end 0.183 (I) vs 0.176 (ALL) /
  0.175 (X); memory cell 0.128 vs 0.087 / 0.059; action head 0.062 vs 0.026 / 0.022; value head
  0.035 vs 0.038 / 0.027.
- Their gains move at least as far from identity as the wide-input arms (the two largest gain
  excursions in the whole grid — front-end down to 0.39 and value head down to 0.20 — are both `I`
  arms).
- Their per-unit spread is not collapsed: it grows over training exactly as the other arms' does,
  and in the front-end arms it is the largest in the grid.

**But there is a clean and consistent difference in the other direction.** At four of the five site
families, the `I` arm produces a markedly *quieter* total gradient than its ALL and X siblings —
smaller mean, much shorter tail:

| Site family | fraction of post-warm-up logging windows whose peak exceeded the 0.5 clip ceiling: ALL / I / X |
|---|---|
| encoder | 91.0% / **36.5%** / 80.9% |
| memory cell | 62.0% / **4.2%** / 77.3% |
| action head | 86.9% / **4.5%** / 80.9% |
| value head | 73.8% / 58.2% / 59.8% |
| all four sites | 40.2% / **4.5%** / 81.9% |
| *(control, for reference)* | *79.9%* |

Four of five families show the same sign and a large margin. That is a real, repeated engineering
observation, not a coin flip — but note that all five share the same random seed, so it is not five
*independent* confirmations either. The mechanistic reading is straightforward and testable: a
modulator driven by two slowly-varying body signals emits a smoother, lower-variance modulation than
one driven by nineteen fast-changing world signals, and a smoother modulation puts less high-frequency
energy into the gradient.

**Bottom line for the engineering question:** a two-dimensional modulator input is *not* degenerate
in this codebase. The modulator learns, receives gradient, and moves its outputs further from
identity than the wide-input arms do. What it produces is a lower-variance signal, which is a
property to be aware of, not a fault. Whether that lower variance is *good or bad for the science*
is out of scope here.

**The check this analysis could not perform, and should.** All of the above rests on aggregate
statistics, and an aggregate can hide a conditional behaviour: a modulator whose output varies
strongly across the 128 units but barely at all across time would produce exactly these numbers
while being functionally a fixed re-parameterisation of the layer. The currently-logged spread mixes
the across-unit and across-time components and cannot separate them. §7 requests the one extra
scalar per site that would.

### 4.5 Does modulation push the total gradient into the clip? — No; every arm sits well below the ceiling, and the control is in the same regime

The trainer clips the global gradient norm at **0.5** (`train.py:1176-1183`, with
`agent.max_grad_norm: 0.5` confirmed in all sixteen runs' own saved configs). The logged
`loss/grad_norm` is measured **before** clipping.

The reference point is the recently completed return-mode study
([[return_mode_cmp_10M]]), where the arms that failed to learn sat 150–300× above this
ceiling and were clipped on essentially every update. **Nothing resembling that happens here.**

Post-warm-up (after the first 10% of updates):

| Cell | mean grad norm | 99th percentile | mean above 0.5 | vs. ceiling |
|---|---|---|---|---|
| T1_none (control) | 0.264 | 0.373 | 0.00% | 0.53× |
| T2_enc_ALL | 0.308 | 0.442 | 0.19% | 0.62× |
| T2_enc_I | 0.250 | 0.283 | 0.00% | 0.50× |
| T2_enc_X | 0.292 | 0.371 | 0.00% | 0.58× |
| T3_rnn_ALL | 0.256 | 0.338 | 0.25% | 0.51× |
| T3_rnn_I | 0.204 | 0.231 | 0.00% | 0.41× |
| T3_rnn_X | 0.272 | 0.394 | 0.00% | 0.54× |
| T4_act_ALL | 0.278 | 0.409 | 0.00% | 0.56× |
| T4_act_I | 0.196 | 0.260 | 0.12% | 0.39× |
| T4_act_X | 0.266 | 0.379 | 0.00% | 0.53× |
| T5_crt_ALL | 0.259 | 0.438 | 0.12% | 0.52× |
| T5_crt_I | 0.240 | 0.372 | 0.00% | 0.48× |
| T5_crt_X | 0.250 | 0.353 | 0.12% | 0.50× |
| T16_quad_ALL | 0.251 | 0.320 | 0.00% | 0.50× |
| T16_quad_I | 0.211 | 0.235 | 0.00% | 0.42× |
| T16_quad_X | 0.299 | 0.377 | 0.00% | 0.60× |

Every arm's typical gradient is roughly **half** the ceiling; every arm's 99th percentile is still
below it; and the fraction of logging windows whose *average* exceeds the ceiling is at most 0.25%
anywhere, including the control. All sixteen sit inside the metrics reference's healthy band of
0.01–1.0, and no arm comes near the 5.0 warning level.

**Adding a modulator does change the clipping regime, but only at the margin.** The widest-input
arms run about 10–17% hotter than the control (front-end ALL 0.308 and four-site X 0.299, against
the control's 0.264) and the narrow-input arms run 5–26% cooler. That is a shift in the same
regime, not a change of regime. The decisive variable in the return-mode study is not in play here.

**Warm-up clipping is heavy and universal, including in the control.** In the first 10% of updates
every run, control included, records gradient norms of 1.2–1.6 with in-window peaks of 9.5–33. That
is the normal opening transient of this trainer and it resolves in every arm.

### 4.6 Is the all-four-sites arm unstable? — No

The four-simultaneous-sites configuration had never been run. On every stability axis measured it is
indistinguishable from, or better than, the single-site arms:

| Measure | T16_quad_ALL | T16_quad_I | T16_quad_X | single-site range | control |
|---|---|---|---|---|---|
| post-warm-up mean grad norm | 0.251 | 0.211 | 0.299 | 0.196 – 0.308 | 0.264 |
| gradient excursions above 3.0 after warm-up | **0** | 1 | 4 | 0 – 3 | 1 |
| largest survival drawdown after 2M episodes | 22.6 | 20.7 | 22.9 | 20.1 – 28.5 | 24.8 |
| non-finite values | 0 | 0 | 0 | 0 | 0 |
| modulator gradient | 0.190 | 0.157 | 0.197 | 0.022 – 0.183 | — |
| gains at end of training | all in 0.37–0.72 | all in 0.26–0.72 | all in 0.40–0.75 | 0.20 – 0.89 | — |

The four-site arm under the all-senses input has the **fewest** post-warm-up gradient excursions of
any arm in the grid (zero) and the lowest fraction of windows touching the clip ceiling among the
wide-input arms (40.2% vs 62–91% for the single-site ALL arms). Its modulator carries the largest
gradient in the grid, which is what four active sites should look like. The two mild elevations are
the four-site X arm's mean gradient (0.299, the second-highest in the grid, still 0.60x the
ceiling) and its four post-warm-up gradient excursions, the most of any arm bar none in raw count
— though its largest, 18.6, is smaller than the largest excursion in two single-site arms.

**A caveat the numbers cannot resolve.** Two of the three four-site arms are also the two arms on
different hardware (node 113, RTX 4090). Hardware does not affect gradient magnitudes, so the
stability conclusion stands, but it does affect §4.7.

### 4.7 Throughput and cost — no arm is anomalously slow; the ordering is dominated by hardware placement

Environment steps per second (cumulative average over each run's life):

| Cell | Node:GPU | Card | steps/s | vs. control |
|---|---|---|---|---|
| T1_none (control) | 106:0 | 3090 | 49,655 | — |
| T5_crt_X | 112:0 | 3090 | 43,988 | −11% |
| T5_crt_ALL | 111:0 | 3090 | 43,844 | −12% |
| T4_act_I | 110:0 | 3090 | 43,380 | −13% |
| T3_rnn_X | 109:0 | 3090 | 43,075 | −13% |
| T2_enc_I | 107:0 | 3090 | 42,866 | −14% |
| T4_act_X | 110:1 | 3090 | 42,142 | −15% |
| T2_enc_ALL | 106:1 | 3090 | 42,050 | −15% |
| T5_crt_I | 111:1 | 3090 | 41,722 | −16% |
| T3_rnn_ALL | 108:0 | 3090 | 41,501 | −16% |
| T2_enc_X | 107:1 | 3090 | 40,464 | −19% |
| T4_act_ALL | 109:1 | 3090 | 39,788 | −20% |
| T16_quad_ALL | 112:1 | 3090 | 37,293 | −25% |
| T3_rnn_I | 108:1 | 3090 | 35,881 | −28% |
| T16_quad_I | 113:0 | **4090** | 48,937 | n/a (faster card) |
| T16_quad_X | 113:1 | **4090** | 49,951 | n/a (faster card) |

**The honest reading.** On identical hardware, adding a modulator costs 11–28% throughput, and the
four-site configuration costs about 25%. That is a compute cost, not a defect. But two confounds
make any finer comparison meaningless:

- **Card.** The two four-site arms on the 4090 node run at the control's speed on a faster card, so
  their apparent parity with the control is hardware, not efficiency.
- **GPU slot.** On **all seven** RTX 3090 nodes (106–112) the run on GPU 0 is faster than the run on
  GPU 1 of the same node, by 2–13%; the only node where the ordering reverses is the 4090 node.
  Seven of seven is not chance. That is a systematic placement effect (thermal, PCIe, or host
  contention between the two co-resident processes), and it is large enough to reorder the middle
  of this table on its own. The slowest arm in the grid (memory-cell site, narrow input, 35,881
  steps/s) sits on a GPU-1 slot next to a sibling arm at 41,501 — a 14% gap that the slot effect
  alone plausibly explains.

**No recompilation, no host-device stall.** Each run prints `JIT compiling train_iteration...`
exactly **once**, and never again — a mid-run recompile would print again and show as a wall-clock
gap. The wall-clock interval between consecutive logged rows has a median of 7–9.6 s in every run,
a 99th percentile of 13–19 s, and a maximum of 22–35 s (the periodic checkpoint-and-render step).
**No run has a single gap exceeding five times its own median.** There is no stall signature
anywhere.

### 4.8 Anything else that looks like an engineering problem

**One genuine outlier, absorbed by the clip.** The value-head arm on the exteroceptive input
(`T5_crt_X`) recorded a single gradient excursion to **1030** at iteration 90,550 — about 97% of the
way through its run, and roughly 4,000× its own typical value of 0.25. Context around it:

| iteration | mean grad norm | peak in window | value loss | total loss | modulator grad norm |
|---|---|---|---|---|---|
| 90,450 | 0.206 | 0.38 | 0.237 | 0.111 | 0.015 |
| 90,500 | 0.205 | 0.33 | 0.240 | 0.112 | 0.033 |
| **90,550** | **10.51** | **1030** | 0.241 | 0.112 | 0.032 |
| **90,600** | **10.53** | **1030** | 0.243 | 0.114 | 0.023 |
| 90,650 | 0.221 | 0.79 | 0.242 | 0.113 | 0.045 |
| 90,700 | 0.232 | 0.65 | 0.240 | 0.112 | 0.015 |

It appears in two consecutive rows because the logging window is a rolling buffer, so one bad
iteration is visible twice. **Nothing else moved**: the value loss, total loss and modulator
gradient are all at their normal values immediately before, during and after. The clip did its job —
an update of norm 1030 against a 0.5 ceiling is scaled by a factor of 0.0005, i.e. it was
effectively a null update — and this run went on to finish all 10,000,000 episodes at 165.7 mean
survival steps, mid-pack.

**Is this a defect?** Probably not, and here is the discipline: excursions above 3.0 after warm-up
occur in **thirteen of the sixteen runs, including the unmodulated control** (which peaks at 4.9),
so this trainer has a heavy-tailed gradient distribution as a background property. The 1030 is an
extreme draw from a distribution that already produces 33s elsewhere, in an arm whose value head is
being FiLM-modulated with a gain that has fallen to 0.62 and an offset that has fallen to −1.57. It
is a plausible tail event, not a demonstrated bug. **What would settle it**: per-update gradient
norms (currently only the per-iteration mean and the window peak are kept), so one could see whether
the excursion was one minibatch or the whole iteration; and the value-target magnitudes for that
iteration, to see whether an outlier return drove it. Both are §7 requests.

**Other post-warm-up excursions above 3.0**, for calibration — none of these needs action:

| Cell | distinct events | largest |
|---|---|---|
| T16_quad_X | 4 | 18.6 |
| T5_crt_X | 3 | **1030** |
| T4_act_I | 3 | 32.6 |
| T3_rnn_ALL | 2 | 33.1 |
| T4_act_ALL | 2 | 4.2 |
| T1_none (control) | 1 | 4.9 |
| T2_enc_ALL, T3_rnn_X, T4_act_X, T5_crt_ALL, T5_crt_I, T16_quad_I | 1 each | 3.0 – 6.1 |
| T2_enc_I, T2_enc_X, T3_rnn_I, T16_quad_ALL | 0 | — |

Each event appears in two consecutive rolling windows, so the raw row counts are double these;
the counts above collapse them. Twelve of the sixteen runs have at least one such event.

**No suspicious exact-zero or exact-constant series.** Every logged series in every run varies. No
metric is pinned at its initialisation value, which is the signature a disconnected component would
leave.

**No warnings or errors** in any run's captured output.

---

## 5. Where a metric looks unusual but the code is probably fine

### 5.1 "A newly enabled site is a no-op at step 0" is true on average, not per unit

The design document's reasoning leans on the property that switching a re-tuning site on cannot
hurt at the start of training, because the site begins as an identity operation. The code
(`src/models/neuromodulator.py:142-146, 178-186`) sets the **bias** of the gain head to exactly 1.0
and the **bias** of the offset head to exactly 0.0 — but leaves the head's **weight matrix** at the
framework's default random initialisation, and adds no zero-gate in front of it. So at the very
first forward pass the gain is `1.0 + W·h`, where `h` is the modulator's internal state: one on
average, but with a per-unit random deviation.

**The size of that deviation is measurable and is not small.** At the first logged point the
per-unit standard deviation of the gain is **0.28–0.37** in every arm, and of the offset
**0.27–0.35**. So on day one a typical unit is being multiplied by something in the region of 0.7 to
1.3 and shifted by ±0.3, rather than left alone.

**Why this is a note and not a bug report.** The code comment at
`src/models/neuromodulator.py:178-181` explicitly claims "a newly-enabled site is a no-op at step
0", and taken per-unit that claim is not exact. But nothing observed suggests it caused harm — every
arm learned normally from the start, and the arms differ from the control in their first-million
survival by amounts (66 to 93 steps, control 82) that are not ordered by how many sites they enable.
The correct response is to **state the property accurately** in the design's reasoning rather than
to change the initialisation mid-grid.

**What would confirm the alternative** (that the random per-unit deviation does matter): re-run one
cell with the gain and offset heads' weight matrices initialised to zero, making the site an exact
identity at step 0, and compare the first-million-episode survival curve. That is a one-run control,
not a code change to this grid.

### 5.2 The downward gain drift, again

Covered in §4.3. Restating the discipline: a falling FiLM gain is **not** evidence of a layer being
silenced, because the layer's own weights are free to grow and absorb the change. The observation is
"the mean gain fell to 0.20–0.89 depending on site and arm, monotonically, at every site, in every
arm". The inference "the site is silencing its layer" is **not** supported by anything currently
logged. §7 asks for the two scalars that would let a future reader tell these apart.

---

## 6. Two documentation / provenance defects found

### 6.1 The metrics reference is now misleading about the front-end gain

**What it says.** [[NMN_METRICS_REFERENCE]] §5.1 states that `gamma_uni_mean` and
`gamma_multi_mean` are "raw pre-sigmoid values, not the actual gains applied", and instructs the
reader to apply a sigmoid to recover the gain. Its §3 table lists their initialisation value as
"~2.0", and its §4 healthy-range table gives them a healthy band of 1.0–3.0.

**What is true for these sixteen runs.** They use `modulation.type: FiLM`, and under FiLM the
front-end applies the gain **linearly**, with no sigmoid
(`src/models/recurrent_ppo_network.py:240-242, 258-260`), and initialises the gain bias to 1.0 rather
than 2.0 (`src/models/neuromodulator.py:142-146`). The data agrees: the first logged
`gamma_uni_mean` in every encoder arm is 0.965–1.001, not ~2.0.

**Why it matters.** A reader who follows §5.1's instruction on this grid's numbers will compute
`sigmoid(0.39) = 0.60` and conclude the front-end gain is 0.60, when it is 0.39. And §4's healthy
band of 1.0–3.0 would mark every one of these perfectly healthy runs as sub-healthy. The reference's
own §2.8 already gets this right for the three *new* sites ("All six are FiLM signals in **linear**
space (unlike `gamma_uni_*` / `gamma_multi_*` … which are pre-sigmoid)") — the gap is that the
encoder's series are now *also* linear whenever the type is FiLM, and §5.1, §3 and §4 have not been
updated to say "depends on `modulation.type`".

**This is a documentation fix, not a code fix.** The code is doing the right thing. Flagged in §8
for a maintenance update to the reference; this audit does not edit it.

### 6.2 Fifteen of sixteen runs could not record whether the working tree was clean

**What the design required.** §3 of [[NMN_INPUT_SITE_GRID]]: "All 16 rows must show the **same** SHA
with `git_dirty: false`; a row that does not is a run that cannot be reconstructed and is excluded
from every comparison."

**What was recorded.** All sixteen runs record the same commit, `a71f4471`, on branch `v3.0`. But
the `git_dirty` field reads the string `"unknown"` in fifteen of them and `true` in one
(the four-site exteroceptive arm). Taken literally, **the design's gate is not met by any of the
sixteen rows**, and one row positively fails it.

**Why it happened.** `src/utils/provenance.py:99-102` determines dirtiness by running
`git status --porcelain --untracked-files=no` with a **10-second timeout**, and returns the string
`"unknown"` on timeout or failure — deliberately, so that "we could not tell" is distinguishable
from "it was clean". Sixteen training processes started within eleven seconds of each other, each
running `git status` against the same repository on a NAS filesystem; `git status` takes the index
lock and is documented in this project as slow on that mount. The cheap `git rev-parse` calls
succeeded in all sixteen, which is consistent with lock contention rather than a broken git.

**Why it is very probably harmless here, and what was checked instead.** Since the recorded flag
cannot answer the question, three substitute checks were run, and all three pass:

- The launch commit **contains** both prerequisite commits (`e1aab726`, `83b8140b`) as ancestors.
- `git diff e1aab726 a71f4471 -- src/` is **empty**, and `src/` has had no commits since — so the
  training code at the launch commit is exactly the reviewed code.
- The working tree is **clean right now** under `src/`, `train.py` and the grid's config folder, and
  the uncommitted work in this repository today lives in an unrelated environment-config folder and
  a diary file, neither of which the trainer reads.
- Most importantly, the **model that was actually built** in each of the sixteen runs reports —
  in its own startup banner and in the trainer's own saved config — exactly the sites and input
  sensors the design specifies. That is a stronger check than a dirty flag, because it inspects what
  the system produced rather than what the source ought to produce.

**The engineering issue that remains** is that this project's provenance record becomes unreliable
precisely when it is needed most — a simultaneous multi-run launch. A one-line fix exists (retry, or
raise the timeout, or take the dirtiness reading once in the launcher and pass it down) but it is a
code change and is therefore flagged in §8 rather than made here.

---

## 7. Metrics Requested

Four metrics whose absence bounded this audit. All are cheap. None is required for the grid's
pre-registered analysis to proceed.

| | Metric | Why now | Where it'd live | Cost |
|---|---|---|---|---|
| 1 | `modulator/gamma_<site>_temporal_std` and `modulator/gamma_<site>_unit_std` — the standard deviation of the site's gain **across time/batch after averaging over units**, and **across units after averaging over time**, in the same units as the existing gain (dimensionless multiplier) | The single currently-logged spread mixes these two components, so it cannot distinguish a modulator that genuinely re-tunes the network moment-to-moment from one that has learned a fixed per-unit re-parameterisation and is functionally inert. This is exactly the question the narrow-input (`I`) arms raise in §4.4, and it cannot be answered from what is logged. It is also the discriminator that keeps a conditional behaviour from being averaged away | `train.py` around the `_mod_spread` helper at line ~1848, where the existing mean/std over `mod_info` are already computed | cheap — two extra reductions over an array that is already resident, at the existing logging cadence |
| 2 | `network/<site>_postfilm_active_fraction` — the fraction of the modulated layer's units whose post-FiLM pre-activation is positive, i.e. surviving the rectifier (dimensionless, 0–1) | §4.3 finds gains near 0.2–0.4 combined with offsets near −1.0 to −1.8 at the action and value heads. Whether that has silenced a large part of those layers, or is a harmless scale re-partition absorbed by the layer's own weights, is currently unanswerable — and the two possibilities have opposite implications for reading the grid's results | `src/models/recurrent_ppo_network.py`, immediately after the FiLM applications at lines 553 and 564 | cheap — one comparison and one mean per site per iteration |
| 3 | `loss/grad_norm_p99` and a `loss/grad_norm_clipped_fraction` — the 99th percentile of the **per-update** pre-clip gradient norm within a logging window, and the fraction of individual updates whose norm exceeded `max_grad_norm` (L2 norm; fraction 0–1) | Only the per-iteration mean and the window peak are kept, so §4.5's clipping statement is bounded above and below rather than measured, and §4.8's 1030 excursion cannot be localised to a single minibatch. The return-mode study established clipping fraction as the decisive variable separating arms that learn from arms that crawl, so it deserves to be measured directly rather than inferred | `src/models/recurrent_ppo_trainer.py` around line 375, where `grad_norm` is already computed per update, plus the windowing in `train.py` | cheap — the per-update norms already exist; this is an extra reduction over them |
| 4 | `modulator/param_count` and `network/param_count` at startup (integers, logged once) | §4.7 attributes an 11–28% throughput cost to the modulator, but the audit cannot state the parameter cost of each site configuration because it is not recorded; a future reader comparing throughput across site configurations has to reconstruct it from the architecture | `train.py`, in the startup banner block that already prints the modulation configuration | cheap — one scalar each, logged once |

---

## 8. Related Issues

Two items that need a code or documentation change and are therefore **not** made by this audit.
The user decides whether to route them.

1. **Metrics reference is stale for FiLM-type runs** (§6.1). `docs/develop/active/neuromodulation/NMN_METRICS_REFERENCE.md`
   §5.1, §3 (init values for `gamma_uni` / `gamma_multi`) and §4 (healthy band 1.0–3.0) describe the
   pre-sigmoid behaviour and do not say it is conditional on `modulation.type`. Under `type: FiLM`
   the encoder's gains are linear and initialise at 1.0. Suggested route: documentation maintenance
   (the reference carries the modulation-site refactor as its plan link, so the refactor's own doc
   is the natural place to record the amendment). **A reader who follows §5.1 on these sixteen runs
   will misread every front-end number.**
2. **Provenance dirty-flag is unreliable under simultaneous multi-run launch** (§6.2).
   `src/utils/provenance.py:99-102` runs `git status` with a 10 s timeout per training process;
   sixteen concurrent launches on the NAS produced `"unknown"` in fifteen of sixteen. Suggested
   route: a bug-fix plan under `docs/develop/active/`. Options worth weighing: retry on timeout,
   raise the timeout for this call specifically, or have the launcher read the flag once and pass it
   to every child process. Note that the current behaviour is *safe* (it never claims clean when it
   does not know) — the defect is that it makes an experiment-design gate unsatisfiable.

Also worth surfacing to whoever owns the design document, though it is not a defect:

3. **The Launch Manifest in [[NMN_INPUT_SITE_GRID]] §3 was never filled in.** All sixteen rows still
   read `planned` with empty node, GPU, run-ID and log-path columns. §2.1 above reconstructs the
   actual values from the runs' own metadata. Per the manifest's stated ownership, only
   `training-runner` writes those columns; this audit has not touched them.

4. **The design's own step-0 identity argument should be restated** (§5.1). "Turning a site on
   cannot hurt at step 0" holds for the layer mean but not per unit, where the gain starts at
   1.0 ± ~0.3.

---

## 9. Bottom line

Sixteen runs, brand-new code, first wave ever. **No run needs to be stopped, restarted, or
discarded.** The wiring is correct in all sixteen, verified three independent ways. The modulator is
alive and receiving gradient in all fifteen modulated arms, including at the two sites that had
never been exercised before today. Nothing is diverging, nothing is stalling, nothing is being
crushed by the gradient clip, and the four-simultaneous-sites configuration — the most aggressive
one — is if anything the calmest arm in the grid. The control lands inside its pre-registered
reference band, so the grid has a valid reference point.

The three things to carry forward are a documentation correction that would otherwise cause a future
reader to misinterpret every front-end number, a provenance-recording weakness that makes one of the
design's own gates unsatisfiable, and one metric pattern — gains falling to 0.2–0.4 alongside
offsets falling to −1.8 at the two new sites — that is not currently distinguishable from a benign
scale re-partition and would be, with two cheap extra scalars.

---

## Appendix

### A. Working files

Raw extractions, retained for traceability (gitignored):

- `tmp/nmnsite/parse_wandb.py`, `tmp/nmnsite/parse_full.py` — the local transaction-log readers used
  here (no WandB web API).
- `tmp/nmnsite/<runid>.json` — per-run downsampled series, all metrics.
- `tmp/nmnsite/full_<runid>.json` — per-run complete series for the health-critical metrics, plus
  the non-finite-value scan.
- `tmp/nmnsite/summary_core.txt`, `summary_mod.txt`, `summary_clip.txt`, `summary_slice.txt` — the
  aggregated tables this document is built from.

### B. Provenance of every number here

Every figure in this document comes from one of: a run's own WandB transaction log, a run's own
trainer-written `models/config.yaml`, a run's own `models/provenance.json`, a run's own startup
banner, or a named line of source in `src/` / `train.py`. No number is taken from the WandB web
interface, and none is quoted from another agent without independent recomputation — including the
control's reference-band result in §4.1.

### C. Changelog

| Date | Change |
|---|---|
| 2026-09-07 | Initial audit. Six runs complete, ten at 84–99%. |
