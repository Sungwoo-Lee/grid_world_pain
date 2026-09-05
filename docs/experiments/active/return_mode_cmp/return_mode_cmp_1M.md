---
title: "Return-mode comparison at 1M episodes — the project's z-scored critic target beats the textbook convention 3.3x"
topic: comparison
status: active
created: 2026-09-04
last_updated: 2026-09-05
wandb_group: return_mode_cmp
wandb_tag: "rppo_cmp_{mc,mcfixed,gae}_s{42..46}"
develop_link: docs/develop/active/issues/FIX_MC_RETURN_UNITS_AND_LEARNING_RATES.md
---

# Return-mode comparison at 1M episodes

> **Status**: COMPLETE (1M-episode set). **The 10M-episode replication has now landed and
> corrects two of the readings below — see [[return_mode_cmp_10M]].** In one sentence: the
> textbook-convention arms were *delayed*, not stalled; given ten times the budget every one of
> them learned to forage (MC_FIXED 40 → 95 survival steps, GAE 42 → 127), so §6.4's "early
> evidence for the stall reading" was wrong and the gap narrowed from 3.3x to 1.75x.
> **Separately, Finding 3's shared-trunk value-dominance *mechanism* should be treated as
> untested, and the loss-share evidence cited for it in Finding 3 does not bear on it.** The
> mechanism is a claim about the ratio of the two terms' *gradients* in the shared parameters,
> and that ratio has never been logged — here or at 10M. The loss-share numbers Finding 3 quotes
> cannot substitute for it: PPO's policy surrogate is `mean(ratio × advantage)` with advantages
> rescaled to average zero, so the policy term's *value* is ≈ 0 in expectation however large its
> *gradient* is. What the 10M runs do establish is that the strongest form of the claim is wrong
> — agents learned substantially while the loss balance never moved — but a version in which
> value dominance merely *slows* learning is untouched, and is in fact what a roughly ten-fold
> slowdown would look like. Cite the mechanism as an open question, not as established and not
> as refuted. Everything measured here at 1M still reproduces at 1M.
> **Date**: 2026-09-04
> **Author**: `experiment-analyzer` (post-hoc / unplanned mode — no pre-registered design doc)
> **Related**: [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] · [[ppo_return_normalization_survey]]

---

## 1. Research question

**Plain language.** A reinforcement-learning agent that learns by "policy gradient" needs two numbers
out of every batch of experience. The first is a **critic target**: the total future reward the
agent's value-predictor *should* have predicted at each moment. The second is an **advantage**: how
much better or worse an action turned out than the critic expected. Before either number is used,
the code has to decide **what scale to put them on** — leave them in raw reward units, or rescale
them to roughly mean 0 and spread 1.

This project has always rescaled the *critic target* and left the *advantage* alone. That is
backwards relative to the textbook: a survey of ten widely used implementations found that nine of
nine that state a convention do the opposite — raw critic target, rescaled advantage — and an
RL-theory review of this project endorsed switching. A new code path (`MC_FIXED`) was written to do
exactly that, and this experiment ran the old way, the new way, and the conventional
generalised-advantage-estimation ("GAE") baseline head to head.

**The question**: on this project's 10×10 predator-and-food survival environment, does adopting the
textbook convention improve how long the agent survives?

**The answer is no, and not by a small margin.** The project's existing scheme kept the agent alive
for **138 steps on average**; both textbook-convention arms kept it alive for **40–42 steps**, a
**3.3× gap** with no overlap between any pair of the fifteen runs. This document establishes that the
gap is real, characterises what the losing agents actually do instead (they learn to sit still and
starve rather than to forage), and identifies the most likely cause — which is **not** the choice of
estimator that the theory review was about, but a **scale mismatch between the critic's learning
signal and the policy's**, made lethal by a gradient-clipping threshold shared between them.

**Post-hoc caveat.** No design document was written before these runs, so no predicted outcomes were
registered in advance. The hypotheses below were written *after* seeing the survival numbers, and
should be treated as explanations to be tested rather than as confirmed results. Section 6.3 names
the experiment that would test them — and one of its four configs already exists in the repository.

### Formal hypotheses (written post-hoc)

> **H₀** (*the scaling convention does not matter*): survival steps are the same across the three
> `return_mode` settings.
> **H₁** (*the convention matters, textbook is better*): the two textbook-convention arms survive longer.
> **H₂** (*the convention matters, the project's scheme is better*): the project's existing scheme survives longer.

**Verdict: H₀ and H₁ are both refuted. H₂ is supported**, at an effect size (Cohen's *d* ≈ 38)
far beyond anything seed noise could produce.

---

## 2. Experimental design

### 2.1 Independent variable

Exactly one configuration line, `agent.return_mode`, takes three values. All three build the value
targets and advantages from the same rollout; they differ only in what is rescaled.

| Arm | Critic target | Advantage | Estimator | Scale relationship |
|---|---|---|---|---|
| **MC** (project's historical mode) | Monte-Carlo return, **rescaled** to mean 0 / spread 1 per batch | rescaled-return − critic prediction, **left alone** | Monte-Carlo | **matched** — critic and policy see the same units |
| **MC_FIXED** (new this session) | the **same** Monte-Carlo return, **raw** | return − critic prediction, **rescaled** | Monte-Carlo | **split** — critic sees units ~24× larger than the policy |
| **GAE** (unchanged baseline) | GAE advantage + critic prediction, **raw** | GAE advantage, **rescaled** | GAE(λ=0.95) | **split** |

Source: `src/models/recurrent_ppo_trainer.py`, `train_iteration`, the three branches on
`return_mode`. `MC_FIXED` reuses the `MC` branch's return computation, window-edge bootstrap and
real-death mask byte for byte; only the two normalisation lines differ.

The 2×2 that this three-arm set half-covers is worth naming, because it is the key to §5:
**estimator** (Monte-Carlo vs GAE) × **scale** (matched vs split). `MC` is the only matched-scale cell
present. The missing cell — GAE estimator with matched scale — is the decisive control (§6.3).

### 2.2 Controlled variables

Environment `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`: a 10×10 grid,
500-step episode cap, 4 regenerating food resources, 12 static damaging hazards, 2 hunting predators,
2 harmless "rabbits", 42 bushes/rocks. The agent pays 1 nutrition per step, starts each episode at a
**uniformly random nutrition level between 0 and 100**, and each unit of food restores 6. It has six
actions: four moves, Rest, and Eat. Plain recurrent PPO (GRU, 128 hidden), neuromodulation off,
γ = 0.95, GAE λ = 0.95, clip ε = 0.1, entropy coefficient 0.01, value coefficient 0.5,
`max_grad_norm` 0.5, learning rate 5×10⁻⁴, 4 epochs per update, 128 parallel environments,
128-step rollouts, 1,000,000 episodes, 5 seeds (42–46) per arm.

**Verified, not assumed.** The check was run against the configuration files the *trainer itself
wrote* into each run directory (`results/JAX_RecurrentPPO/*/models/config.yaml`), not against a fresh
reload of the source YAMLs. Flattening all 15 saved configs gives **559 keys**, of which exactly
**four** differ across runs: `agent.return_mode`, `seed`, `tag`, and `wandb.name`. The last two are
labels. Extraction: `tmp/cfgdiff2.py`; output in Appendix B.

### 2.3 Confounds and limitations

| # | Confound | Affected | Severity | Assessment |
|---|---|---|---|---|
| C1 | **Budget is 1M *episodes*, not 1M env steps.** Because MC's episodes are 3.3× longer, MC consumed **6.9–9.8×10⁷** environment steps and ~4,200 gradient iterations, while MC_FIXED/GAE consumed **3.5–3.7×10⁷** steps and ~2,150 iterations. The winning arm received ~2.3× the data and ~2× the updates. | all | **High a priori** | **Does not explain the result.** Re-plotting survival against *environment steps* (§4.4, Q1c) shows MC ahead at every matched budget: at 35M env steps — the largest budget all three reached — MC is at **119.4 ± 4.8** vs MC_FIXED **40.2 ± 1.6** and GAE **41.7 ± 0.7**. MC is already ahead at 2M steps. |
| C2 | Post-hoc analysis: no pre-registered predictions. | all | Medium | Stated openly. The mechanism in §5 is a hypothesis, not a tested result. |
| C3 | Only one hyperparameter set was tried. The losing arms might be recoverable by re-tuning (learning rate, value coefficient, gradient-clip threshold), which was not attempted. | MC_FIXED, GAE | **High for generality** | This is the single biggest limitation. The result says *"at this project's current hyperparameters"*, not *"as a general fact about PPO"*. See §6.2. |
| C4 | The environment's rewards are never normalised — no reward wrapper, no running return scaler, no PopArt. The measured return spread is ~24 and the mean return ≈ −68. | all | **High for external validity** | This is the likely reason the survey's "nine of nine" convention does not transfer (§5, Finding 4). |
| C5 | Only one seed per arm (s42) was re-evaluated for the behavioural rollouts in §4.3. | behavioural section | Low | The rollout conclusions match the WandB behavioural columns across all five seeds, which agree tightly (§4.2). |
| C6 | The trainer logs no explained-variance, advantage-scale, KL or clip-fraction metric, so parts of §5 are inferred from value-loss magnitudes rather than measured directly. | §5 | Medium | Raised in §7 Metrics Requested. |

## 3. Runs analysed

All 15 runs completed. Metrics were read from the **local** WandB datastores
(`wandb/run-*/run-*.wandb`) with a protobuf record scanner (`tmp/parse_wandb_local.py`) — no web API
call was made. Each logged `Episode/Steps` point is itself a rolling mean over 5,000 episodes
(`Episode/_window_n = 5000`), so single points are already heavily smoothed.

| Arm | Seeds | WandB run dirs (`wandb/run-20260903_…`) | Result dirs (`results/JAX_RecurrentPPO/20260903-…`) |
|---|---|---|---|
| MC | 42–46 | `163046-rk3huayr`, `163229-x2jxkpq8`, `163331-4dnvtpqf`, `163432-9kjjj5t8`, `163534-mx8orfd1` | `163034_rppo_cmp_mc_s42` … `163522_…_s46` |
| MC_FIXED | 42–46 | `163635-k7tu4ch0`, `163737-tifzlhcq`, `163838-0t83gfxv`, `163939-x6me4z6a`, `164041-kqjxkgwu` | `163623_rppo_cmp_mcfixed_s42` … `164030_…_s46` |
| GAE | 42–46 | `164144-g20s999t`, `164245-x36zqm96`, `164347-s1lpg7o8`, `164448-pwn54tw7`, `164549-0ghnqaly` | `164133_rppo_cmp_gae_s42` … `164538_…_s46` |

Related in-flight set: WandB group `return_mode_cmp_10m`, tags `rppo_cmp10m_*`, launched
2026-09-04 17:54 on nodes 101–109, 10M episodes, same seeds — see §6.4.

## 4. Results

### 4.1 Primary metric — survival steps

Each seed's value is the mean of its final 10 logged points (≈ the last 4% of training); the
arm value is the mean over its 5 seeds.

| Arm | Survival steps, mean ± sd (n = 5 seeds) | 95% CI | Per-seed values |
|---|---|---|---|
| **MC** | **138.18 ± 3.52** | [133.81, 142.55] | 132.3, 137.6, 139.7, 141.3, 140.0 |
| **MC_FIXED** | **40.27 ± 0.80** | [39.28, 41.26] | 41.3, 40.3, 39.7, 40.8, 39.3 |
| **GAE** | **41.93 ± 0.70** | [41.06, 42.80] | 41.4, 41.8, 41.4, 42.0, 43.1 |

| Contrast | Difference | Ratio | Welch *t* | *p* | Cohen's *d* |
|---|---|---|---|---|---|
| MC − GAE | **+96.3 steps** | 3.30× | 59.9 | 1.8×10⁻⁷ | 37.9 |
| MC − MC_FIXED | **+97.9 steps** | 3.43× | 60.7 | 1.3×10⁻⁷ | 38.4 |
| GAE − MC_FIXED | +1.7 steps | 1.04× | 3.5 | 8.3×10⁻³ | 2.2 |

**The user-supplied summary table reproduces.** Independent re-derivation from the raw local
datastores gives 138.2 / 40.3 / 41.9 against the reported 138.8 / 40.0 / 41.9 — the small
differences are the smoothing window (last-10-points vs the single final point) and are immaterial.

**Not a single-checkpoint artefact.** Three windows were compared per seed: the single last logged
point, the last 10 points, and the last 20% of training. They agree to within ±3 steps in every
arm, and the within-seed standard deviation over the final 20% is only 3.7–6.3 steps for MC and
0.7–1.3 steps for the other two (Appendix A.1). Every one of the 5 MC seeds finishes above 132; every
one of the 10 non-MC seeds finishes below 44. The distributions do not touch.

**Seed dispersion is small at the endpoint but large mid-training.** MC's across-seed spread peaks at
±27 steps around episode 500k — that is *takeoff-time* dispersion, not outcome dispersion: all five
MC seeds take off, at somewhat different times, and re-converge to ±3.5 by 1M.

> **Verdict**: the project's existing `MC` mode wins decisively. The result is not a fluke of
> smoothing, of a lucky seed, or of the unequal data budget.

### 4.2 Secondary metrics — what the agents actually do

End-of-training values (mean of the final 10% of logged points, averaged over 5 seeds).
The right-hand column divides each count by that arm's survival steps, which removes the
"MC just lives longer so it does more of everything" effect.

| Metric | MC | MC_FIXED | GAE | Per-step: MC | MC_FIXED | GAE |
|---|---|---|---|---|---|---|
| Survival steps | **136.1** | 40.1 | 41.8 | — | — | — |
| Food eaten per episode | **27.3** | 1.7 | 2.2 | **0.200** | 0.043 | 0.052 |
| Rest actions per episode | 50.3 | 22.7 | 24.2 | **0.369** | 0.565 | 0.578 |
| Episodes ending by starvation | **27.9%** | 52.6% | 54.7% | — | — | — |
| Episodes ending by injury | 54.9% | 47.4% | 45.3% | — | — | — |
| Episodes reaching the 500-step cap | **17.2%** | **0.000%** | **0.000%** | — | — | — |
| Mean distance to nearest food | **3.40** | 3.97 | 4.01 | — | — | — |
| Predator hits per episode | 0.75 | 0.73 | 0.71 | **0.0055** | 0.0182 | 0.0171 |
| Total damage per episode | 135.1 | 70.9 | 70.4 | 0.993 | 1.766 | 1.685 |
| Episode reward (secondary only) | **−105.8** | −136.6 | −134.8 | −0.778 | −3.405 | −3.224 |
| Longest episode observed | **500** | 428 | 463 | — | — | — |
| Spread of episode lengths (sd) | **187.9** | 43.9 | 47.9 | — | — | — |

Reading the table: **steps needed to eat one unit of food** is 5.0 for MC, 23.6 for MC_FIXED, 19.4
for GAE. MC forages roughly **four times faster per unit of time**, so its longer life is not merely
a consequence of living longer. It also takes predator hits at **one third** the per-step rate, so it
is not buying food with recklessness. Meanwhile MC_FIXED and GAE spend **57–58% of every step
resting** against MC's 37%, and their average distance to food *increases* over training while MC's
decreases.

**Reward agrees with survival here** (MC −105.8 vs −136.6 / −134.8), so the project's survival-only
rule is not doing any work in this particular verdict. Reward remains a secondary diagnostic.

### 4.3 Trajectory-level check — the losing agents learn to sit still

Aggregate statistics can average away a behaviour that only fires in a sub-regime, so the final
checkpoints of one seed per arm (seed 42) were re-run through
`scripts/eval/eval_rollout.py --record` with **identical evaluation seeds across arms**, giving 30
paired episodes per arm — same world layout, same starting nutrition and injury, same predator
placement. Action counts were read straight out of the recordings
(`tmp/traj_actions.py`, `tmp/traj_paired.py`).

| Action share, pooled over all recorded steps | MC | MC_FIXED | GAE |
|---|---|---|---|
| Move (Up/Right/Down/Left) | **0.308** | 0.163 | 0.187 |
| **Rest** | 0.415 | **0.692** | **0.733** |
| **Eat** | **0.271** | 0.124 | 0.057 |
| Longest unbroken run of Rest, mean per episode | 10.2 steps | **24.9** | **26.5** |
| Longest unbroken run of Rest, worst episode | 35 steps | **83** | **87** |
| Episodes reaching the 500-step cap | 6/30 | 0/30 | 0/30 |
| Mean survival in this eval | 156.3 | 46.5 | 44.1 |

Two things only the paired view reveals:

1. **The arms are identical on doomed episodes.** In 11 of the 30 paired episodes all three agents
   died at exactly the same step (2, 2, 6, 14, 15, 20, 32 …). Those are episodes where a low starting
   nutrition or an early predator made survival impossible. MC's *median* episode length (40) is
   essentially the same as MC_FIXED's (38) and GAE's (38). **MC's entire advantage lives in the upper
   tail.** Restricting to the 15 episodes in which all three arms survived at least 30 steps, mean
   length is MC **290.5** vs MC_FIXED 73.1 vs GAE 68.6, and the Eat-action share is MC **0.263** vs
   0.101 vs 0.053. MC converts survivable episodes into 250–500-step survivals; the others cap out
   around 90–150.
2. **GAE almost never eats.** Its Eat-action share is exactly 0.000 in 25 of 30 episodes; MC_FIXED's
   is 0.000 in 22 of 30. Both have also nearly abandoned one or more movement directions (GAE picks
   "Right" on 1.0% of steps, MC_FIXED on 0.1%).

**Why 40 steps is the specific number the losers land on.** The environment charges 1 nutrition per
step and starts each episode at a uniformly random nutrition between 0 and 100, i.e. ~50 on average.
An agent that never eats therefore dies of starvation after roughly 50 steps, less whatever fraction
is killed earlier by a predator. MC_FIXED eats 1.7 food × 6 nutrition ≈ 10 extra steps of life.
**40 steps is approximately the do-nothing ceiling of this environment.** The losing arms have not
learned a worse foraging policy; they have learned a *non-foraging* policy whose lifespan is set by
the nutrition it happened to spawn with.

> **Answer to "worse policy or no foraging at all"**: **no foraging at all.** They converged on a
> Rest-dominated, damage-minimising local optimum — resting cuts hazard damage by 4× and obstacle
> damage by 4× (§4.2) — and then starve.

### 4.4 Learning dynamics

**Survival against episodes** (arm mean ± sd over 5 seeds; full table in Appendix A.2):

| Episodes | MC | MC_FIXED | GAE |
|---|---|---|---|
| 50k | 33.8 ± 1.5 | 29.2 ± 1.2 | 29.8 ± 1.1 |
| 200k | 41.7 ± 3.8 | 33.1 ± 0.8 | 34.1 ± 0.6 |
| 400k | 60.9 ± 23.8 | 34.9 ± 0.6 | 37.3 ± 1.1 |
| 600k | 104.0 ± 31.5 | 37.8 ± 0.9 | 40.3 ± 0.8 |
| 800k | 129.4 ± 6.2 | 39.4 ± 1.3 | 41.1 ± 0.6 |
| 1M | **138.8 ± 3.7** | 40.0 ± 1.1 | 41.9 ± 1.3 |

**Survival against environment steps** — the confound-corrected view (C1):

| Env steps | MC | MC_FIXED | GAE |
|---|---|---|---|
| 2M | 36.4 ± 1.9 | 31.4 ± 0.6 | 31.0 ± 1.7 |
| 10M | 43.2 ± 4.0 | 34.3 ± 0.2 | 35.6 ± 1.4 |
| 20M | 70.5 ± 19.2 | 37.7 ± 0.6 | 40.1 ± 1.0 |
| **35M** (largest budget all arms reached) | **119.4 ± 4.8** | 40.2 ± 1.6 | 41.7 ± 0.7 |

**Are they still separating at 1M, or plateaued?** Fitting a line to the last 25% of each seed's
curve:

| Arm | Slope, survival steps gained per 100k episodes | Per-seed |
|---|---|---|
| MC | **+3.79** | +6.55, +4.21, +1.97, +3.24, +2.99 (5/5 positive) |
| MC_FIXED | +0.45 | +0.78, +0.44, +0.24, +0.40, +0.40 (5/5 positive) |
| GAE | +0.32 | −0.06, +0.07, +0.49, +0.23, +0.88 (4/5 positive) |

**The arms are still separating.** MC is gaining ~8× faster than the others and has not plateaued —
its cap-reaching rate is also still climbing (+0.010 per 100k episodes, from 0 at 300k to 0.179 at
1M). MC_FIXED and GAE are close to flat but not exactly zero. Nothing in the 1M data suggests the
gap is about to close on its own.

**Phase structure.** All three arms take the *same* first path: rest fraction climbs from 0.17 to
~0.6 by 100–200k episodes and starvation deaths climb from 25% to ~50%. That is the rest-and-starve
local optimum, and every arm enters it. MC's starvation rate peaks at **58.5% around episode 250–300k**,
then **falls back to 26.9%** as its food-per-step rate takes off from 0.056 (300k) to 0.200 (1M).
MC_FIXED's and GAE's starvation rates instead climb *monotonically* to 53.8% and 55.0%, and are still
climbing at 1M (+0.020 and +0.006 per 100k episodes).

> The three arms are not on different curves. They are on the **same** curve, and **only MC escapes
> the rest-and-starve local optimum.**

## 5. Analysis

### Finding 1 — the headline is solid, and the data-budget confound does not explain it

**What**: 3.3× gap, *d* ≈ 38, non-overlapping across all 15 runs, stable across three end-of-training
windows, and present at every matched environment-step budget from 2M steps upward.
**Evidence**: §4.1, §4.4.
**Confidence**: **High.** The most dangerous confound (C1 — MC earned 2.3× more data by surviving
longer) was tested directly and rejected: at a common 35M-step budget MC is at 119 and the others at
40–42.

### Finding 2 — the losing arms failed to learn foraging, they did not merely learn it worse

**What**: MC_FIXED and GAE eat 4× less food **per step**, rest 57–58% of steps against MC's 37%,
sit still for runs of 25–87 consecutive steps, drift *away* from food over training, and never once
reach the 500-step cap in five seeds × 1M episodes. Their 40-step lifespan is approximately what an
agent that never eats gets for free from its random starting nutrition.
**Why**: a Rest-heavy policy is a genuine local optimum here — it cuts hazard damage 4× and obstacle
damage 4× and is reachable in a few thousand updates, whereas foraging requires a longer behavioural
chain (navigate to food, then Eat) before it pays.
**Evidence**: §4.2, §4.3, §4.4.
**Confidence**: **High** — the WandB behavioural columns (all 5 seeds) and the paired action-level
rollouts (30 paired episodes) agree.

### Finding 3 — the training-health signature is unambiguous, and it is about scale

Value-loss *magnitudes* are not comparable across arms (MC's target is rescaled to spread 1; the
others' is raw with spread ~24). The comparisons below are therefore all **unit-free**: shares of
the loss budget, trends, and gradient norms against the arm-independent clipping threshold of 0.5.

| Diagnostic (end of training) | MC | MC_FIXED | GAE |
|---|---|---|---|
| Value term's share of the loss budget | 92.9% | **99.9923%** | **99.9875%** |
| **Policy term's share of the loss budget** | **1.94%** | **0.0024%** | **0.0043%** |
| Gradient norm before clipping | **0.399** | 132.1 | 78.2 |
| Multiple of the 0.5 clipping threshold | **0.8× (below it)** | **264×** | **156×** |
| Fraction of the gradient that survives clipping | **1.000 (never clipped)** | **0.0038** | **0.0064** |

("Loss budget" = |policy| + |0.5 × value| + |0.01 × entropy|, i.e. each term's contribution to the
total loss in absolute terms.)

**What**: in the raw-target arms the policy objective is **~500–800× smaller a fraction of the total
loss** than it is in MC, and the whole gradient is scaled down by 150–700× by `clip_by_global_norm`
on **every single update, throughout training**. MC's gradient starts at 1.44 (clipped, factor 0.35),
falls below the 0.5 threshold by ~150k episodes, and is **never clipped again** — so MC's effective
learning is *released* over training while the others' stays pinned.

**Why this matters even though Adam is scale-invariant.** A constant rescale of the gradient is
largely absorbed by Adam's per-parameter normalisation, so "clipping shrinks the learning rate" is
not by itself a valid argument, and this analysis does not make it. The argument is about
**direction**. The actor and critic in `ActorCriticRNN` share an encoder and the GRU; only the two
output heads are separate. For every shared parameter, the gradient Adam sees is the *sum* of a value
term and a policy term, and Adam normalises the sum elementwise. When the value term is ~40,000×
larger, the shared recurrent representation is optimised essentially **entirely for value
regression**, and the policy head is left to read out of features shaped for something else. In MC
the two terms are within a factor of ~50 of each other, so the trunk carries policy-relevant
structure.

**And the raw-target critic never actually fits.** Converting the value loss to a residual
root-mean-square error (√(2 × value loss)) gives a plateau of **0.687** for MC against a target whose
spread is 1.0 *by construction* — i.e. the critic explains about **53%** of the return variance. For
MC_FIXED the plateau is **22.9** raw return units, and for GAE **18.2**. Against the measured return
spread of ~24, MC_FIXED's critic explains roughly **9%**. Worse, neither raw-target arm improves
monotonically: MC_FIXED's value loss falls to 244 by 200k episodes, then **rises to 298 by 600k**
before settling at 262; GAE's goes 162 → 191 → 166. **The arms that spend 99.99% of their loss budget
on the critic end up with a critic that fits five times worse.** Rescaling the target per batch
strips out the batch-level mean and scale — nuisance quantities the advantage does not use anyway —
leaving the critic only the state-dependent structure that the policy actually needs.

**Confidence**: **High** for the measurements (shares, gradient norms, residual RMSE trends);
**Medium** for the shared-trunk causal story, which is a mechanism hypothesis and is exactly what the
control experiment in §6.3 is for. The 9%-explained-variance figure additionally depends on the
externally supplied return spread of ~24 — see §7.

### Finding 4 — the theory review was not wrong about PPO; it was incomplete about *this* codebase

**What**: the survey found nine of nine mainstream implementations use raw critic targets with
rescaled advantages. That is true and this experiment does not contradict it. What those
implementations also do — and what this project does not — is **normalise the reward or return stream
upstream** (`VecNormalize` / `NormalizeReward`-style running scalers, or PopArt-style value
normalisation), or operate in environments whose rewards are already order-1. This project's returns
have spread ~24 and mean ≈ −68, and there is no normaliser anywhere in the pipeline.

So `MC_FIXED` imported the mainstream **advantage** convention without the mainstream **scale**
convention, and landed in a configuration that essentially no surveyed library actually runs: a
critic trained on order-24 targets sharing a global gradient-norm clip of 0.5 and a shared trunk with
a policy trained on order-1 advantages. Read this way, `MC`'s per-batch z-scoring of the return is
*doing the job of the missing return normaliser*, and paying for it with the units mismatch between
the rescaled return and the un-rescaled bootstrap value that the original bug report identified.

**Why this is the important reading**: it means the finding is most likely **about scale, not about
the estimator**. `MC` and `MC_FIXED` use *identical* Monte-Carlo returns from *identical* code; only
the rescaling differs. And `MC_FIXED` (Monte-Carlo, split scale) performs the same as `GAE` (GAE,
split scale) to within 1.7 steps. Across the three cells present, **scale predicts the outcome and
estimator does not.**
**Confidence**: **Medium-High** — the 2×2 argument is strong but the fourth cell has not been run.

### Finding 5 — entropy is not the explanation (the proposed exploration mechanism is refuted)

The hypothesis under test: because MC's advantage is not rescaled, its magnitude tracks the critic's
error and shrinks as the critic fits, while the entropy coefficient stays fixed at 0.01 — so MC
should end up with a systematically *higher*-entropy, more exploratory policy.

| Episodes | MC | MC_FIXED | GAE |
|---|---|---|---|
| 0 | 1.515 ± 0.082 | 1.405 ± 0.159 | 1.336 ± 0.129 |
| 100k | 0.819 ± 0.135 | 0.724 ± 0.155 | 0.727 ± 0.084 |
| 300k | 0.776 ± 0.070 | 0.603 ± 0.043 | 0.621 ± 0.073 |
| 500k | 0.718 ± 0.094 | 0.657 ± 0.041 | 0.657 ± 0.052 |
| 700k | 0.640 ± 0.058 | 0.665 ± 0.049 | 0.689 ± 0.032 |
| **1M** | **0.658 ± 0.066** | **0.694 ± 0.066** | **0.678 ± 0.017** |

(Policy entropy in nats; the maximum for this 6-action environment is ln 6 = 1.792.)

**What**: all three arms converge to the **same** entropy, ~0.66–0.69 nats, well inside each other's
seed spread. At the endpoint MC is in fact the **lowest** of the three — the opposite of the
prediction. There is a modest transient difference between 200k and 500k episodes, where MC holds
0.72–0.78 while the others sit at 0.60–0.66; that window does overlap MC's takeoff, so a weak
contribution cannot be excluded. But the effect is ~0.1–0.17 nats against a seed spread of
0.04–0.09, and it reverses by 700k.

There is a quantitative reason to expect this. MC's advantage spread is exactly its critic residual
in rescaled units, i.e. **≈ 0.69**, against exactly **1.0** for the arms that rescale advantages. A
factor of 1.4 in advantage scale, with clip ε = 0.1, cannot produce a 3.3× survival difference. **The
un-rescaled advantage is not where the action is; the raw critic target is.** This is a useful
negative result: it removes the most obvious alternative explanation and points the mechanism
squarely at Finding 3/4.
**Confidence**: **High** for the refutation; the transient 200–500k difference is **Low** confidence
either way.

### Finding 6 — nothing else differs between the arms

559 configuration keys compared across the 15 trainer-written configs; four differ, of which two are
labels (§2.2). Same seeds, same environment, same node class, same code revision. All three arms
start at the same place — 19.3–19.6 survival steps and identical termination mix over the first
logged points. **This comparison is fair.** The one genuine asymmetry is C1 (unequal data budget from
the episode-based stopping rule), and it favours the winner but does not create the win.
**Confidence**: **High.**

## 6. Conclusions

### 6.1 Summary

- **The headline is real.** MC survives **138.2 ± 3.5** steps, MC_FIXED **40.3 ± 0.8**, GAE
  **41.9 ± 0.7** — a **3.3× gap**, 5/5 seeds per arm, no distributional overlap, robust to the
  smoothing window and to matching on environment steps rather than episodes. **The theory review's
  endorsement of the textbook convention is refuted on this environment at these hyperparameters.**
- **What the losers do**: they learn to Rest, not to eat. 57–58% of steps resting, unbroken Rest runs
  of 25–87 steps, 4× lower food intake per step, ~0 Eat actions in most episodes, and a 40-step
  lifespan that is approximately the free lifespan granted by their random starting nutrition. They
  are **non-foragers**, not weak foragers.
- **All three arms enter the same rest-and-starve local optimum around 100–300k episodes. Only MC
  escapes it** — its starvation rate peaks at 58.5% then falls to 26.9% while the others' climb
  monotonically to 54–55%.
- **The training-health signature is a scale collapse.** In the raw-target arms the policy objective
  is 0.002–0.004% of the loss budget (MC: 1.9%) and the gradient is clipped down by a factor of
  150–700 on every update (MC is never clipped after 150k episodes). Despite spending 99.99% of the
  loss budget on the critic, those arms end with a critic explaining ~9% of return variance against
  MC's ~53% — and their value loss *rises* through the middle of training.
- **The inference, stated as an inference**: the decisive variable is most likely the **critic
  target's scale**, not the choice of estimator. MC and MC_FIXED use identical returns from identical
  code and differ by 98 steps; MC_FIXED and GAE use different estimators with the same scale
  convention and differ by 1.7 steps. The nine surveyed libraries pair the raw-target convention with
  an upstream reward/return normaliser that this project does not have.
- **Entropy is not the explanation** — all three arms converge to ~0.67 nats and MC ends lowest.

### 6.2 What this does and does not license

**It does license**: keeping `return_mode: MC` as the project default for recurrent PPO on this
environment; treating any past result obtained under `MC` as *not* invalidated by the units-mismatch
bug report; and treating "switch to the textbook convention" as a change that must be paired with a
scale fix before it is adopted.

**It does not license**: the claim that z-scoring the critic target is generally superior to the
mainstream convention. Three things are unresolved. (a) The losing arms were never re-tuned — a lower
value coefficient, a much larger `max_grad_norm`, or a separate critic optimiser might recover them
entirely, and if so the finding is about an *interaction* with this project's fixed hyperparameters
rather than about the convention. (b) The mechanism in Finding 3 is a hypothesis; no experiment here
manipulates it. (c) One environment, one network, one learning rate.

**Honest caveats a sceptical reader deserves**: MC's *median* episode is no longer than the others'
(40 vs 38 in the paired evaluation) — its entire gain is in the upper tail of survivable episodes, so
"3.3× better" is a statement about the mean, which is the project's chosen metric. And MC pays for
its survival with 1.9× more total damage per episode; it is a higher-risk, higher-return policy, not
a uniformly safer one.

### 6.3 Recommended next experiments

| Priority | Experiment | Rationale | Effort |
|---|---|---|---|
| **1** | **Run the missing 2×2 cell: GAE estimator with a matched (rescaled) critic target.** The config **already exists** at `configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml` (`return_mode: GAE_NORM`), and as of 2026-09-04 a parallel session has the matching trainer branch and a unit test in its working tree (uncommitted). Reviewed here: that branch z-scores the GAE target and takes the advantage as target − critic, which mirrors `MC`'s scheme faithfully — the right control. If it behaves like MC, the cause is **scale** and Finding 4 stands; if it behaves like GAE, the cause is the **estimator** and Finding 4 is wrong. | This is the single decisive control. Nothing else here separates the two explanations. | 5 seeds × 1M ≈ 35 min/run |
| **2** | **MC_FIXED with the gradient clip effectively disabled** (`max_grad_norm` at 10⁶) and, separately, **MC_FIXED with the value coefficient scaled down ~500×** to restore the loss balance. | Directly manipulates the two quantities Finding 3 names. If either recovers MC_FIXED, the result is an interaction with this project's clip/coefficient settings rather than a fact about the convention — which is a materially different (and more publishable) claim. | 2 × 5 seeds × 1M |
| **3** | **Add an upstream return normaliser** (running mean/std over returns, or PopArt on the value head) and re-run MC_FIXED and GAE. | This is what the nine surveyed libraries actually do. If it recovers them, the theory review is vindicated and the project simply had a missing component. | needs a small code change first |
| 4 | **Log explained variance, advantage scale, KL and clip fraction** (see §7), then re-read these same runs' successors. | Several claims above are inferred from loss magnitudes rather than measured. | cheap |
| 5 | **Re-evaluate the remaining four seeds per arm** with paired evaluation seeds, not just seed 42. | The behavioural conclusion currently rests on one seed per arm plus the (concordant) five-seed WandB columns. | ~15 min |

Experiments 1–3 together turn a post-hoc observation into a pre-registered causal claim. **Priority 1
should be designed by `experiment-designer` with predicted outcomes written before launch**, since
the two possible results have opposite implications for the paper.

### 6.4 Will these conclusions hold at 10M episodes?

The identical comparison is running now as WandB group `return_mode_cmp_10m`.

**Prediction: the ordering will hold; the size of the gap is genuinely uncertain and could shrink
substantially.**

*Why the ordering should hold.* At every matched environment-step budget MC is ahead, and at 1M it is
the only arm still improving materially (+3.79 steps per 100k episodes versus +0.45 and +0.32). For
the ordering to reverse, MC would have to stall while a stalled arm accelerates past it — nothing in
1M episodes of data points that way.

*Why the gap might shrink.* All three arms are on the same trajectory and MC merely escapes the
rest-and-starve optimum first. Measuring the delay directly: MC first reaches 35 survival steps at
40k episodes, GAE at 230k (5.8× later), MC_FIXED at 310k (7.8×); MC first reaches 40 steps at 120k,
GAE at 470k (3.9×), MC_FIXED at 720k (6.0×). If that 4–8× time dilation simply continues, MC's
takeoff at ~280–350k episodes maps to ~1.1–2.8M for the other two — comfortably inside a 10M budget —
and by 10M all three could be near a common ceiling with the gap largely closed.

*Why the gap might not shrink.* Time dilation and a hard stall look identical at 1M, and the live 10M
runs already discriminate between them a little. **As of this writing** MC_FIXED has reached
~1.6M episodes at **43.8 ± 1.0** survival steps and GAE ~1.4M episodes at **44.5 ± 2.1**, versus
40.0 and 41.9 at 1M. They are creeping up at roughly +0.6 steps per 100k episodes — and they are now
**past 5× the episode count at which MC had already taken off (280k)** without any sign of a takeoff.
That is early evidence for the stall reading rather than the dilation reading. MC's 10M runs are at
~840k episodes and 135.2 ± 2.1, reproducing the 1M result.

*The specific things to check when the 10M set lands.* (i) Does either losing arm's starvation rate
ever turn over and start falling — the signature of MC's escape? (ii) Does either ever record a
non-zero `Episode/Term_MaxSteps`? Five seeds × 1M episodes produced exactly zero; the first one is
the escape. (iii) Does MC saturate near the 500-step cap, which would compress the gap from above
regardless of what the others do? (iv) Does the value loss in the raw-target arms keep drifting
upward, which would be evidence of slow critic degradation rather than slow progress.

*Caveat*: the 10M set inherits the same episode-based budget, so at 10M episodes MC will again have
consumed ~2.5× the environment steps. Any 10M comparison must be re-plotted against environment
steps, as §4.4 does here.

## 7. Metrics requested

None of these can be added by this analysis (read-only on `src/`). Each would materially sharpen a
claim above. If accepted, route through `feature-workflow`.

| Metric | Why now | Where it'd live | Cost |
|---|---|---|---|
| `value/explained_variance` — 1 − Var(target − prediction)/Var(target), unitless | **Finding 3's central number is currently inferred** from √(2 × value loss) plus an externally supplied return spread of ~24. Explained variance is the correct unit-free critic-quality measure and is directly comparable across arms with different target scales. This is the highest-value request. | `src/models/recurrent_ppo_trainer.py`, `train_iteration`, alongside the existing `loss/value` logging | cheap (one scalar per iteration) |
| `returns/mean`, `returns/std`, `targets/std` | Would confirm the ~24 spread and ≈ −68 mean from the runs themselves instead of from an external measurement, and would show whether the target scale drifts as the policy changes — which is a plausible reason the raw-target value loss *rises* mid-training. | same place, right after the target is computed | cheap (2–3 scalars) |
| `advantages/std_preclip`, `advantages/abs_mean` | Finding 5 infers MC's advantage spread (≈0.69) from the critic residual. Measuring it directly would settle the un-rescaled-advantage hypothesis outright rather than by proxy. | same place, before/after the normalisation lines | cheap |
| `policy/approx_kl` and `policy/clip_fraction` | Standard PPO health metrics; neither is logged. Without them there is no way to tell whether the losing arms' policy updates are being clipped by PPO's own ratio clip (a separate mechanism from the gradient-norm clip) or are simply too small to matter. | `update_step` in the same file | cheap |
| `grad_norm/policy_component` and `grad_norm/value_component` (norms of the two loss terms' gradients, separately) | Finding 3's shared-trunk mechanism is currently argued from *loss* shares. Gradient-norm shares would test it directly, and would show whether the value gradient dominates the shared encoder specifically. | `update_step`, via two extra `jax.grad` calls or a split of the existing one | moderate (one extra backward pass, or a restructure) |

## 8. Related issues

- [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] — the bug report that motivated `MC_FIXED` ("H4
  bootstraps in the wrong units"). **This analysis does not overturn the bug**: the units mismatch
  between the rescaled return and the un-rescaled window-edge bootstrap value is still real. What it
  shows is that the *fix as implemented* costs 98 survival steps, because it removed the rescaling
  that was incidentally serving as this project's only return normaliser. The issue doc should record
  that the fix is **not** to be adopted as the default until experiment 1 or 3 in §6.3 resolves it.
- [[ppo_return_normalization_survey]] — the nine-of-nine survey. Worth extending with a column
  recording **whether each surveyed library also normalises rewards or returns upstream**; the
  survey's conclusion changes meaning entirely depending on that column.
- No code bug was found by this analysis. The `MC_FIXED` implementation does what its docstring says.

---

## Appendix

### A. Raw data tables

Full extraction log, including all per-seed values, the 20-window survival trajectories, the
matched-timestep table, every behavioural metric's temporal evolution, and the loss/gradient/entropy
tables reproduced in abridged form above: `tmp/20260904_return_mode_cmp.md`.

**A.1 — end-of-training window sensitivity (survival steps)**

| Seed | last point | last 10 points | last 20% | sd over last 20% |
|---|---|---|---|---|
| mc_s42 | 133.77 | 132.33 | 127.12 | 5.56 |
| mc_s43 | 135.92 | 137.61 | 135.12 | 4.25 |
| mc_s44 | 140.89 | 139.70 | 137.01 | 4.47 |
| mc_s45 | 142.35 | 141.27 | 139.15 | 3.72 |
| mc_s46 | 141.05 | 139.98 | 134.04 | 6.26 |
| mcfixed_s42 | 40.57 | 41.28 | 40.73 | 0.76 |
| mcfixed_s43 | 40.98 | 40.29 | 40.48 | 0.81 |
| mcfixed_s44 | 39.15 | 39.66 | 38.48 | 1.10 |
| mcfixed_s45 | 40.75 | 40.79 | 40.31 | 0.88 |
| mcfixed_s46 | 38.41 | 39.34 | 39.15 | 0.73 |
| gae_s42 | 41.44 | 41.40 | 41.73 | 1.00 |
| gae_s43 | 40.17 | 41.78 | 41.65 | 0.87 |
| gae_s44 | 41.46 | 41.40 | 41.01 | 0.71 |
| gae_s45 | 43.56 | 41.97 | 42.08 | 0.98 |
| gae_s46 | 42.91 | 43.11 | 41.45 | 1.27 |

**A.2 — paired evaluation, 30 episodes, identical evaluation seeds** (seed-42 checkpoints):
per-episode lengths and Eat-action shares are in `tmp/traj_paired.py` output, archived in
`tmp/20260904_return_mode_cmp.md`.

**Extraction scripts** (working files, gitignored): `tmp/parse_wandb_local.py` (local `.wandb`
datastore reader), `tmp/cfgdiff2.py` (559-key config diff), `tmp/agg2.py`–`tmp/agg7.py` (metric
aggregation), `tmp/traj_actions.py` / `tmp/traj_paired.py` (recording-level action analysis),
`tmp/tenm.py` (live 10M progress).

### B. Config diff

All 15 trainer-written configs (`results/JAX_RecurrentPPO/*/models/config.yaml`) flattened to 559
keys. Differing keys, complete:

| Key | MC seeds 42–46 | MC_FIXED seeds 42–46 | GAE seeds 42–46 |
|---|---|---|---|
| `agent.return_mode` | `MC` | `MC_FIXED` | `GAE` |
| `seed` | 42, 43, 44, 45, 46 | 42, 43, 44, 45, 46 | 42, 43, 44, 45, 46 |
| `tag` | `rppo_cmp_mc_s<seed>` | `rppo_cmp_mcfixed_s<seed>` | `rppo_cmp_gae_s<seed>` |
| `wandb.name` | same as `tag` | same as `tag` | same as `tag` |

No other key differs — including `gamma`, `gae_lambda`, `eps_clip`, `entropy_coef`, `vf_coef`,
`max_grad_norm`, `lr_actor`, `lr_critic`, `K_epochs`, `sequence_length`, `hidden_size`, `rnn_type`,
`encoding_mode`, `modulation.type`, and every environment key.

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-09-04 | Initial analysis of the 1M-episode set (15 runs, 5 seeds × 3 arms) | `experiment-analyzer` |
| 2026-09-05 | Status header updated to point at the 10M replication, which corrects the "stall" reading (§6.4) and the shared-trunk mechanism (Finding 3). No 1M number changed. | `experiment-analyzer` |
