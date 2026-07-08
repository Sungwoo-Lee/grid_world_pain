---
title: "Parity audit 03 — actor-critic, imagination + returns (dreamer_srl vs vendored sheeprl)"
topic: dreamer
status: active
created: 2026-07-06
last_updated: 2026-07-08
---

# Parity audit 03 — Actor-critic, imagination + returns

## Question (plain-language entry point)

This is area 3 of a 5-part, from-scratch numerical-equivalence audit of the project's
JAX re-implementation of DreamerV3 ("dreamer_srl", under `src/algorithms/dreamer_srl/`)
against the vendored PyTorch reference implementation it was ported from (sheeprl,
pinned at commit `33b6366` under `vendor/sheeprl/`). The question for this area:
**does the behaviour-learning half of the algorithm — the imagined rollout the agent
dreams forward from its world model, the λ-return targets computed over that dream,
the return normalizer, the policy (actor) loss, the value (critic) loss, and the
slow-moving target-value network — compute the same numbers the reference computes?**

Every check below was re-derived from the two source trees, not from prior review
documents; prior docs were used only to classify a difference as already-declared or
new. The audit baseline includes two landed fixes: the terminal-row buffer zeroing
(H5, commit `b1dd90a`) and the v1 fix that carries the rollout-time action through
the policy loss instead of re-drawing a fresh one (the v1 root-cause bug).

**Verdict in one line:** the behaviour-learning math is a faithful port — every
structural item checked lands PARITY or provably-equivalent, confirmed by empirical
probes that reproduce the reference's numbers to 0–1 float32 ULP — **except one new
undeclared finding: the discount factor gamma in every live config is
`0.996840347`, not the reference's `0.996996996996997`**, a ~5% shorter effective
planning horizon (316.5 vs 333 steps) shipped under a comment that mis-cites the
reference file. Two impactful deviations in this area were already declared
elsewhere (no gradient clipping on any optimizer; the epsilon inside the entropy
formula) and are re-confirmed, not re-counted as new.

## Classification rubric

| Label | Meaning |
|---|---|
| **PARITY** | Same formula, same constants, same slicing/ordering; where probed, numerically identical (≤ 1 ULP float32). |
| **EQUIVALENT-BY-DESIGN** | Different mechanism (JAX-functional vs torch-mutating / graph-structural), with a proof in the details section that the produced numbers and gradients are the same. |
| **DECLARED** | A real difference already recorded in `DEVIATION_LOG.md`, `KNOWN_BUGS.md`, or a prior review; cited. |
| **UNDECLARED** | A real difference not recorded anywhere; severity assigned. |

## Complete findings table

Ours = `src/algorithms/dreamer_srl/`; Ref = `vendor/sheeprl/sheeprl/algos/dreamer_v3/`.

| # | Item | Ours (file:line) | Reference (file:line) | Classification |
|---|---|---|---|---|
| 1 | Imagination seeded from **all T×B posteriors** of the replay batch, flattened row-major `[T,B] → [T·B]` | `train.py:803-813` | `dreamer_v3.py:203-205` | **PARITY** (same flatten order; terminated flag flattened identically, so rows pair correctly) |
| 2 | Posteriors + recurrent states **stop-gradient'd** before seeding the rollout | `train.py:803-804` | `dreamer_v3.py:203-204` (`.detach()`) | **PARITY** |
| 3 | Actor input **detached at every rollout step** | rollout runs outside any `value_and_grad` closure; loss re-forwards on `sg(latents)` (`train.py:888, 905`) | `actor(imagined_latent_state.detach())` `dreamer_v3.py:219,240` | **EQUIVALENT-BY-DESIGN** (proof §D3) |
| 4 | Horizon = 15; rollout stores **H+1 latents and H+1 actions** (action at last latent computed but unused) | `agent.py:1785-1820`; config `01_food_only.yaml:43` | `dreamer_v3.py:206-241`; `configs/algo/dreamer_v3.yaml` `horizon: 15` | **PARITY** |
| 5 | One-step imagination = RecurrentModel(cat(prior, action)) → GRU → transition prior (sampled, unimix'd); **no posterior correction** | `agent.py:1795-1804` | `agent.py:482-498` (rssm.imagination), `:467-480` | **PARITY** (structure; float32 matmul drift = declared D-007/D-008) |
| 6 | Prior/actor **sampling RNG**: gumbel-max straight-through vs torch `rsample()` | `agent.py:915-923, 1539-1545` | `agent.py:480` via `compute_stochastic_state`; `agent.py:834` | **DECLARED** (D-009 cross-PRNG class; distribution-identical, streams differ) |
| 7 | **Last imagined step excluded from the actor loss** (`[:-1]` on log-probs, entropy, discount; λ-targets length H) | `train.py:919, 594, 600` | `dreamer_v3.py:286, 295, 297` | **PARITY** |
| 8 | **λ-return recursion**: `interm = r + c·V·(1−λ)`; backward loop `ret[t] = interm[t] + c[t]·λ·ret[t+1]`; bootstrap = `values[-1]`; inputs `rewards[1:], values[1:], continues[1:]·γ` | `utils.py:157-166`, `train.py:483-488` | `utils.py:66-77`, `dreamer_v3.py:251-256` | **PARITY** (probe: max_abs_diff **0.0**, §P1) |
| 9 | **§S5 true-continue splice**: first continue = `1 − terminated` (real env flag; truncation does NOT count as termination), imagined continues thereafter | `train.py:472-479, 856` | `dreamer_v3.py:247-248` | **PARITY** at this consumer (production of `terminated` has declared Finding 8 caveat — see §D9) |
| 10 | First step's continue enters **only the discount cumprod**, not the λ recursion (`continues[1:]` to λ) | `train.py:486, 491` | `dreamer_v3.py:254, 260` | **PARITY** |
| 11 | **Discount weights**: `cumprod(continues·γ)/γ`, no-grad, `[:-1]` slice applied to both losses; critic squeezes trailing dim | `train.py:215-219, 318, 600` | `dreamer_v3.py:259-260, 297, 316` | **PARITY** (probe: 1 ULP, §P1) |
| 12 | Continue-head **mode** = `sigmoid(logits) > 0.5` (BernoulliSafeMode) | `loss.py:304-316`, `train.py:850-853` | `dreamer_v3.py:246`; `sheeprl/utils/distribution.py:409-416` | **PARITY** |
| 13 | **Return normalizer (Moments)**: 5th/95th percentile (linear-interp quantile), EMA decay 0.99, `offset = low`, `invscale = max(1/max, high − low)` with config `max: 1.0` → **max(1, range)** floor; updated-then-used once per gradient step | `utils.py:217-263`, `train.py:870-877`; config `01_food_only.yaml:95-100` | `utils.py:40-63`, `dreamer_v3.py:276`; `configs/algo/dreamer_v3.yaml` `moments.max: 1.0` | **PARITY** (probe over a 5-update chain: diff **0.0**, §P2; in-place→functional state = declared D-001) |
| 14 | **Advantage**: `(λ − offset)/invscale − (baseline − offset)/invscale`, baseline = live-critic `predicted_values[:-1]`, per-term normalization kept | `train.py:574-585` | `dreamer_v3.py:275-279` | **PARITY** (probe: diff **0.0**, §P3) |
| 15 | **Actor loss (discrete)**: pure REINFORCE — `log π(sg(a_rollout)) · sg(advantage)`; no dynamics-backprop term; loss `−mean(sg(discount)[:-1] · (objective + ent_coef·entropy[:-1]))`; `ent_coef = 3e-4` | `train.py:500-604, 891-953`; config `:94` | `dreamer_v3.py:283-297`; config `ent_coef: 3e-4` | **PARITY** (end-to-end probe: policy-loss diff **0.0**, §P3) |
| 16 | **Actor log-prob mechanism**: fresh forward on detached latents at loss time, log-prob of the *rollout* action | `train.py:905-919` (`forward_logits` + one-hot dot with `sg(imagined_actions)`) | `dreamer_v3.py:273, 286` (fresh `actor(traj.detach())`, `log_prob(imgnd_act.detach())`) | **PARITY** (both sides recompute the distribution at loss time; v1 resampling bug confirmed absent) |
| 17 | **Entropy formula**: `−Σ p·log(p + 1e-8)` vs torch `Categorical.entropy()` | `train.py:912-913`, `agent.py:1558-1559` | `dreamer_v3.py:294` | **DECLARED** (CP3 review finding A4; probe: ≤ 2.4e-7 abs diff, bias ≤ 4.1e-8 even at the unimix floor — negligible, §P4) |
| 18 | **Actor unimix 0.01**: probability-space mix then `log(probs)` (matches `probs_to_logits`); applied before both sampling and log-prob/entropy | `agent.py:1472-1482, 819-856` | `agent.py:839-845, 437-449` | **PARITY** (tiny-clamp inactivity declared in audit memo §8) |
| 19 | **Straight-through estimator**: `hard − sg(soft) + soft` | `agent.py:1545, 923` | torch `OneHotCategoricalStraightThrough.rsample()` | **EQUIVALENT-BY-DESIGN** (identical gradient algebra; irrelevant to the loss anyway — see §D3) |
| 20 | **Critic loss**: two-hot symlog cross-entropy, **two terms** — `−log q(sg(λ)) − log q(sg(target_critic_mean))` — discount-weighted mean; critic re-forwarded inside the grad closure on `sg(latents)[:-1]` | `train.py:227-324, 960-982` | `dreamer_v3.py:307-316` | **PARITY** |
| 21 | **Two-hot distribution**: bins `linspace(−20, 20, 255)` in symlog space, symlog on targets, symexp at mean, cross-weight interpolation, `logits − logsumexp` | `loss.py:94-245` | `sheeprl/utils/distribution.py:225-276` | **PARITY** (linspace midpoint ULP = declared D-006) |
| 22 | Critic regresses on **raw** λ-values (not Moments-normed) | `train.py:310` | `dreamer_v3.py:314` | **PARITY** |
| 23 | **Target-critic update**: fires **before** each train step, every `per_rank_target_network_update_freq=1` gradient steps, `tau = 1.0` hard copy at cumulative step 0 then `tau = 0.02`; freq honored in both driver paths | `dreamer_srl_main.py:990-1011` (scan path, `step_idx % _tuf`), `:1610-1620` (legacy path) | `dreamer_v3.py:674-680`; config `tau: 0.02`, freq 1 | **PARITY** (in-place→functional = declared D-011) |
| 24 | **Target-critic init**: fresh random module instead of `deepcopy(critic)` | `agent.py:2113-2121` | `agent.py:1217` | **EQUIVALENT-BY-DESIGN** (proof §D24) |
| 25 | **Update ordering within a step**: WM optimizer step → imagination (post-update WM weights) → reward/value/continue heads → moments → actor step → critic step (pre-update critic used for `predicted_values`) | `train.py:793-982` | `dreamer_v3.py:200-327` | **PARITY** |
| 26 | **Optimizer hyperparams** feeding these losses: Adam, WM lr 1e-4/eps 1e-8, actor+critic lr 8e-5/eps 1e-5 | `dreamer_srl_main.py:658-660`; config `:86-113` | config optimizer blocks | **PARITY** (optax-vs-torch eps placement declared in audit memo §8) |
| 27 | **Gradient clipping**: reference clips WM at 1000, actor at 100, critic at 100 before every optimizer step; ours has **no clipping on any of the three** | absent — `train.py:797, 954, 982` | `dreamer_v3.py:191-199, 300-304, 320-326` | **DECLARED** (Fable-5 diagnosis Finding 3; open KNOWN_BUGS row, priority raised after the 1e29 WM-loss spike) |
| 28 | **Discount factor γ**: `0.996840347` in all 19 live dreamer_srl configs vs reference `0.996996996996997` | `configs/models/dreamer_srl/*.yaml` (e.g. `01_food_only.yaml:41`) | `configs/algo/dreamer_v3.yaml` (`gamma: 0.996996996996997`) | **🔴 UNDECLARED — Medium** (§D28) |
| 29 | `imagine()` returns `imagined_log_probs` / `imagined_entropies` that no training-path consumer reads | `agent.py:1754-1826` | n/a | 🟢 nit (dead output + wasted rollout compute; harmless) |
| 30 | `compute_lambda_values` docstring claims `values: [T+1, B]`; actual contract (and sheeprl's) is length-T with `values[-1]` as bootstrap | `utils.py:134` | `utils.py:66-77` | 🟢 nit (doc only; code correct) |

## Details for non-PARITY rows

### §D3 — Detach-at-rollout is structurally different but gradient-identical (rows 3, 19)

In sheeprl the imagination rollout lives inside one autograd graph, so it must
explicitly `.detach()` the latent fed to the actor at every step
(`dreamer_v3.py:219,240`), and the actor loss must detach the trajectory
(`:273`), the action (`:286`), and the advantage (`:291`). In our port the
rollout (`train.py:816`) executes **outside** any `nnx.value_and_grad` closure —
in JAX, gradients exist only with respect to the arguments of the differentiated
function, so nothing computed in the rollout can leak gradient regardless of
stop-gradient placement. The actor loss then re-forwards the actor inside
`actor_loss_fn` on `jax.lax.stop_gradient(imagined_latents)` (`train.py:888`)
and pairs it with `stop_gradient(imagined_actions)` (`:889`), reproducing
sheeprl's gradient surface exactly: gradient reaches **only** the actor MLP
parameters, via log-prob and entropy of the recomputed distribution. The
remaining sheeprl gradient path one might worry about — straight-through action
gradients flowing into `imagined_trajectories` and thence into
`lambda_values`/`predicted_values` — is severed on the reference side too, by
`advantage.detach()` (actor loss), `lambda_values.detach()` (critic loss), the
internal `.detach()` in `Moments.forward`, and `torch.no_grad()` around the
discount. Net effective gradient: identical sets of parameters, identical
formulas. The end-to-end probe (§P3) confirms the forward values match to 0.0.

### §D9 — Terminated-vs-truncated at the splice (row 9)

The consumer is parity: both sides splice `1 − terminated` (not `1 − done`) as
the first continue, so a time-limit truncation correctly leaves the first
imagined step fully weighted. The **caveat** is upstream of this area: the
driver derives `terminated` as `termination_reason >= 2`, and the Fable-5
diagnosis (Finding 8, `docs/develop/active/issues/diag_fable5_20260704/04_dreamer_srl.md`)
already flagged that this reason code can fire without `done` (over-eating
quirk). Declared; owned by the driver/buffer audit area, not re-counted here.

### §D24 — Target critic starts random, not a copy (row 24)

Sheeprl builds `target_critic = copy.deepcopy(critic)` (`agent.py:1217`). Ours
builds a second, independently-initialized `FullMLPHead` (`agent.py:2114-2121`).
Proof of equivalence: (a) the only consumer of the target critic is
`one_train_step` (`train.py:962-965`); (b) every driver path executes the Polyak
update **before** the first `train_step` call, with `tau = 1.0` exactly when the
cumulative gradient-step counter is 0 (`dreamer_srl_main.py:992-995` scan path,
`:1612-1613` legacy path), which overwrites the target with a byte-exact copy of
the online critic before any read. So the random init is never observable.
(Residual nuance: both critics zero-init their output layer — cascade fix #27 —
so even a hypothetical pre-copy read would return the same 0-valued mean.)

### §D28 — 🔴 UNDECLARED (Medium): gamma is 0.996840347, reference is 0.996996996996997 (row 28)

**What.** Every live dreamer_srl agent config (all 19 `configs/models/dreamer_srl/*.yaml`
variants) sets `algo.gamma: 0.996840347` with the comment `# sheeprl
dreamer_v3.yaml:L22`. The cited reference line actually reads
`gamma: 0.996996996996997` (= 1 − 1/333, Hafner's 333-step discount horizon).
The value is not a float32 artifact of the reference value
(`float32(0.996996…) = 0.996997`), and it appears nowhere else in the repo or in
the vendored tree — it enters the codebase in the CP9 plan's config listing
(`docs/develop/active/dreamer_srl_v1/CP9_PLAN.md:441`, commit `bd8ff91`) with no
derivation, and there is no `DEVIATION_LOG.md` row for it.

**Impact.** Effective discount horizon `1/(1−γ)` is **316.5 steps vs the
reference's 333.0** (~5% shorter). Gamma enters the λ-return recursion
(`continues[1:]·γ`), the discount cumprod that weights both the actor and critic
losses, and thereby every value target, every advantage, and the Moments
normalizer's input scale. Training likely still works — 0.9968 is a reasonable
discount — but (a) any run-level parity comparison against a sheeprl baseline is
comparing two different recipes on the value-learning side, and (b) with
500-step episodes the truncated tail is weighted ~e^(500·Δ(1−γ))≈8% differently
by the horizon change. Severity **Medium**: does not invalidate the port's
correctness, does silently invalidate strict recipe parity and mis-documents
itself as reference-identical.

**Suggested fix** (for `developer`; no code changed by this audit): set
`algo.gamma: 0.996996996996997` in all dreamer_srl configs (or, if 0.996840347
was intentional, add the missing DEVIATION_LOG row and correct the mis-citing
comments). Note comparability: runs trained before/after the change use
different discounts.

### Row 27 — gradient clipping (declared, re-confirmed)

Re-verified independently: `train.py` calls `wm_opt.update` / `actor_opt.update`
/ `critic_opt.update` on raw gradients; no `optax.clip_by_global_norm` anywhere
in the module. The reference clips all three (WM 1000.0, actor 100.0, critic
100.0 — `dreamer_v3.py:191-199, 300-304, 320-326`; config `clip_gradients`
keys). Already open in KNOWN_BUGS ("world-model loss can spike to ~1e29–1e31 …
no gradient clipping", priority raised 2026-07-06). The actor and critic are
equally unprotected — worth stating because the open row's title mentions only
the world model; the diagnosis Finding 3 itself covers all three optimizers.

### Row 17 — entropy epsilon (declared, measured)

Ours computes `−Σ p·log(p + 1e-8)` where torch's `Categorical.entropy()` uses
normalized logits directly. Declared as concern A4 in
`docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md`. Probe §P4: with
unimix 0.01 over 5 actions the smallest possible probability is 0.002, so the
epsilon shifts entropy by ≤ 4.1e-8 nats (measured worst case at the floor) and
the full formula agrees with torch to ≤ 2.4e-7. Negligible; optional cleanup is
to reuse the `log_softmax` already computed one line above (`train.py:906`).

## Empirical probes (§P)

Method: fixtures generated on the torch side with **verbatim copies of the
vendored formulas** (`compute_lambda_values` utils.py:66-77; `Moments`
utils.py:40-63 with `fabric.all_gather` stripped — a no-op single-process, per
declared D-001; the actor-objective algebra of dreamer_v3.py:275-297), executed
with `/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python`; the JAX side ran
the **actual production functions** (`compute_imagined_returns`,
`compute_lambda_values`, `moments_update`, `compute_actor_objective`,
`compute_discount`) with `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`.
Scripts + fixture: `tmp/20260708_probe_torch_side.py`,
`tmp/20260708_probe_jax_side.py`, `tmp/20260708_probe_fixture.npz`.
Shapes H=15, B=6, γ = reference value, λ=0.95, ent_coef=3e-4, moments max=1.0.

| Probe | Quantity | max abs diff |
|---|---|---|
| P1 | λ-returns over a 15-step imagined batch with terminations (incl. §S5 splice path) | **0.0** |
| P1 | discount cumprod/γ | 1.19e-7 (1 ULP) |
| P2 | Moments offset + invscale after a 5-update EMA chain (drifting input distribution; floor active on early updates) | **0.0 / 0.0** |
| P3 | advantage (per-term normalization) | **0.0** |
| P3 | final actor policy loss (objective + entropy + discount weighting, end-to-end) | **0.0** (−1.6330012 both sides) |
| P4 | entropy formula vs `torch.distributions.Categorical.entropy()` on post-unimix logits | 2.38e-7 |

## Verdict

**Counts:** 22 PARITY · 3 EQUIVALENT-BY-DESIGN (proven §D3, §D24) · 4 DECLARED
(D-009 sampling RNG; Moments/Polyak functional-state D-001/D-011 mechanism, folded
into their rows; entropy epsilon CP3-A4; missing gradient clipping Finding 3 —
plus the Finding 8 terminated-production caveat owned by another area) ·
**1 UNDECLARED (Medium): the gamma config drift (§D28)** · 2 nits (dead rollout
outputs; a wrong docstring shape).

The imagined-rollout / returns / actor / critic / target-critic pipeline is a
faithful port of sheeprl's behaviour learning: identical seeding, slicing,
stop-gradient surface, normalization, loss formulas, and update ordering, with
empirical agreement to float32 exactness on every probed quantity. The one new
action item is the discount-factor config drift; the one previously-known item
that this area re-affirms as worth fixing is the absent gradient clipping on all
three optimizers.

## Unverifiable list

- **Bit-level sampling equivalence** of the gumbel-max sampler vs torch
  `rsample()` (rollout actions, prior states): cross-platform PRNG streams are
  incomparable by construction (declared D-002/D-009 class). Distributional
  equivalence is argued, not bit-tested.
- **Full actor/critic gradient parity on matched weights** (∂loss/∂θ against a
  torch run with identical parameters): not re-run in this audit; the CP2–CP8
  Lever-A fixture suite and the probes above cover forward values only. The §D3
  argument for gradient-surface equality is structural, not empirical.
- **Aggregate float32 drift over a full training run** (D-003/D-006/D-007/D-008
  compounding): out of reach for a static audit; only a policy-learning gate can
  measure it.
- **Whether `batch["terminated"]` is terminated-only in every environment
  configuration** at the §S5 splice: depends on the driver's
  `termination_reason >= 2` derivation (declared Finding 8), audited in the
  driver/buffer area, not here.

---
Reviewed by: `code-reviewer` (fresh line-by-line re-derivation, 2026-07-08; no
source, config, or vendor files modified).
