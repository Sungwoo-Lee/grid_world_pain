---
title: "Return-mode comparison at 10M episodes — the textbook convention is roughly 10x slower, not broken; the shared-trunk mechanism this project asserted has never been measured"
topic: comparison
status: active
created: 2026-09-05
last_updated: 2026-09-05
wandb_group: return_mode_cmp_10m
wandb_tag: "rppo_cmp10m_{mc,mcfixed,gae,gaenorm,mcraw}_s{42..46}"
corrects: docs/experiments/active/return_mode_cmp/return_mode_cmp_1M.md
develop_link: docs/develop/active/issues/FIX_MC_RETURN_UNITS_AND_LEARNING_RATES.md
---

# Return-mode comparison at 10M episodes

> **Status**: COMPLETE — all 25 runs finished; **revised 2026-09-05 after adversarial review**
> ([[plan_return_mode_cmp_10M]]). Three claims in the first draft were corrected: the
> shared-trunk mechanism is **unmeasured**, not refuted; the best setting's advantage does
> **not** vanish under exploration-free evaluation (the first evaluation was simply too small to
> see it, and a ten-times-larger one confirms it); and the experience penalty is **not** a
> constant factor — it drifts upward. Appendix E has the point-by-point.
> **Date**: 2026-09-05
> **Author**: `experiment-analyzer`
> **Mode**: partly pre-registered. Two of the five arms had predictions written before they
> ran ([[gae_norm_prereg]], [[mc_raw_prereg]]); the other three did not.
> **Related**: [[return_mode_cmp_1M]] (the 1M-episode predecessor, several of whose
> conclusions this document overturns) · [[plan_return_mode_cmp_10M]] (adversarial review of
> this document) · [[ppo_implementation_details_lit_review]] ·
> [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]]

---

## 1. Research question — and the headline, in plain language

**What was varied.** A reinforcement-learning agent that learns by policy gradient computes two
numbers from every batch of experience. The **critic target** is what the value-predictor
*should* have said about the rest of the episode. The **advantage** is how much better an
action turned out than the critic expected. Before either is used for training, the code
chooses a scale for each: leave it in raw reward units, or rescale it to roughly mean 0 and
spread 1. Five settings of that one choice were run, five random seeds each, on the same
10x10 predator-and-food survival world, with everything else in the configuration identical.

Two settings keep the critic target and the advantage **in the same units, near spread 1**
(the project's historical scheme, and its generalised-advantage-estimation twin). Two settings
**split** them — raw critic target of spread about 24, advantage separately rescaled to spread
1 — which is what nine of nine surveyed mainstream implementations do. The fifth setting keeps
them in the same units but **at the large scale**: raw target *and* raw advantage, both around
20-24.

**Why anyone should care.** At one million episodes ([[return_mode_cmp_1M]]) the two
textbook-convention settings looked catastrophically broken: 40 survival steps against the
project's 138, with agents that had learned to sit still and starve rather than forage, and
that never once in five million episodes reached the environment's 500-step episode cap. That
document floated two readings — the losers are *permanently stalled*, or the losers are merely
*delayed* — and reported "early evidence for the stall reading". It also invoked a mechanism to
explain the stall: in a network whose body is shared between the part that chooses actions and
the part that predicts value, the value-prediction training signal dominates the summed gradient,
so the action-chooser is reduced to reading out features shaped only for value prediction. A
later literature review established that **no paper in the surveyed corpus measures that
mechanism**; it is one practitioner's blog conjecture, and the same corpus contains a
counterexample.

**The three headline results.**

1. **"Permanently stalled" was wrong: these settings are delayed, by about an order of
   magnitude.** Given ten times the budget, every textbook-convention run escaped the
   sit-still-and-starve trap, started eating, and started hitting the 500-step cap — so they are
   not broken. Whether they would ever reach the *same level* as the fast settings is a different
   question, and this experiment does not answer it. Measured as "how much environment experience does this setting
   need to reach a given survival level", the penalty is **on the order of ten-fold** — the
   measured range across the three slow settings and every survival level from 50 to 130 steps
   is **6x to 14x**. That factor is **not flat**: in the two best-measured settings it **drifts
   upward** as the survival level rises (the generalised-advantage one goes 6.4x at survival 50
   to 11.6x at survival 130; the Monte-Carlo one goes 5.9x to 14.1x), while the third, measured
   on only two seeds, drifts the other way. Whether that upward drift is the early signature of
   a genuinely lower ceiling or just seed noise at two-to-three seeds per point is
   **undetermined by this data**, and there is no data at all on whether the slow settings would
   ever match the fast ones: reaching the fast settings' final level would need **at least
   eleven to nineteen times more** experience again than these runs were given — a lower bound,
   because if the factor keeps drifting upward the true requirement is larger. The two
   textbook-convention settings are also **no longer equivalent to each other**: at matched
   experience the generalised-advantage one reaches 98 survival steps against the Monte-Carlo
   one's 69, and it reaches every survival level roughly **1.1x to 1.4x sooner**, where at one
   million episodes the two were 1.7 steps apart.
2. **The mechanism this project has been asserting is *unmeasured*, in both directions.** The
   claim is that in a network whose body is shared between the action-chooser and the
   value-predictor, the value-predictor's training signal swamps the action-chooser's in the
   *summed gradient*, so the shared body ends up shaped only for value prediction. **Nothing in
   these logs measures a gradient split** — only the two terms' *loss values* are recorded, and
   a loss value is not a usable stand-in for a gradient here: the action-chooser's loss term is
   arithmetically near zero whatever its gradient is, because the advantages it multiplies are
   rescaled to average zero. So the loss-share numbers that the one-million-episode analysis
   offered as evidence *for* the mechanism do not support it, and the same numbers read the
   other way do not refute it. Two things that *are* established: substantial learning happened
   while the loss shares never moved, which rules out the strongest version of the claim (that
   value dominance *prevents* learning outright) but not a version that merely slows it; and the
   arm built to raise the action-chooser's share of the summed loss twenty-fold did not
   measurably outperform the arm it was derived from — though that test had far too little
   statistical power to see a moderate benefit, and it never left the regime the mechanism is
   about (99.99% → 99.96% value share). **The honest state is: not established, not refuted, and
   not testable without logging per-term gradient norms** (§7). A separate and better-supported
   *hypothesis* emerged for why that arm in particular failed — its policy stopped exploring —
   but that too is one arm, five seeds, observational, and awaits the intervention named in
   §6.3.
3. **The generalised-advantage variant of the project's own scheme is the best setting, by a
   small but real margin.** It survives 170.4 steps during training against the project
   default's 165.2, with all five of its random seeds above all five of the other's. A first
   check that switched off the agent's random exploration and re-ran the finished agents on 200
   fresh worlds appeared to show the gap vanishing — but that check was too small to see a
   5-step gap at all (its margin of error was ±6.4 steps). **Re-run on 2,000 worlds per agent
   (evaluation only, no retraining), the gap is there and it matches: 170.0 against 165.6, a
   difference of +4.5 steps with a 95% confidence interval of [+0.8, +8.2].** So the two
   measurements never disagreed, and the earlier "it's just a cheaper exploration bill"
   explanation is wrong — the exploration is switched off in that evaluation and the gap
   survives. The same enlarged evaluation also dissolves a second artefact: the "+8.2 steps"
   by which the exploration-free agents appeared to beat their own training logs was a fluke of
   the 200 worlds everything was measured on, and against 2,000 worlds it is under half a step.
   Section 4.9 has the numbers.

**Formal hypotheses.** These are stated for the record; each is translated in place above.

> **H₀** (*scaling does not matter*): survival steps are the same across the five settings.
> **H₁** (*the textbook convention is better*): the split-scale settings survive longer.
> **H₂** (*the project's scheme is better*): the matched-at-spread-1 settings survive longer.
> **H₃** (*relational reading*): what matters is that target and advantage share **units**,
> whatever the magnitude — so the matched-at-large-scale setting should behave like the
> matched-at-1 settings.
> **H₄** (*absolute reading*): what matters is that the advantage lands near **spread 1** — so
> the matched-at-large-scale setting should behave like the split-scale settings.
> **H₅** (*shared-trunk / value-dominance mechanism*): the slow settings are slow because the
> value loss dominates the summed gradient in the shared body of the network; raising the
> policy term's share should therefore help.

**Verdict**: H₀ and H₁ refuted. **H₂ supported but heavily qualified** — the advantage is in
*speed*, on the order of 10x, not in a demonstrated difference of attainable performance.
**H₃ refuted in its strong form, H₄ supported**: every seed of the matched-at-large-scale
setting finished below the worst seed of the matched-at-spread-1 setting, at every budget, so
"units-matching alone is what matters" cannot be right; the weaker question of whether
matching units at scale 24 helps *at all* relative to the split convention is **not resolved**
(§5.8). **H₅ untested** — the gradient quantity it is about is not logged, and the loss-share
quantity that is logged does not bear on it in either direction (§5.6).

---

## 2. Experimental design

### 2.1 The one independent variable

A single configuration key, `agent.return_mode`, takes five values. All five build the value
targets and the advantages from the same rollout data; they differ only in what gets rescaled.
The three Monte-Carlo settings share one estimator, byte for byte; the two
generalised-advantage-estimation ("GAE") settings share the other. Estimator and scale are
therefore fully crossed except for one missing cell (GAE estimator matched at the large scale).

| Setting | Critic target | Advantage | Same units? | At what scale | Estimator |
|---|---|---|---|---|---|
| `MC` (project default) | Monte-Carlo return, z-scored | `target − V`, not rescaled | yes | ≈ 1 | Monte-Carlo |
| `GAE_NORM` (new) | λ-return, z-scored | `target − V`, not rescaled | yes | ≈ 1 | GAE(λ=0.95) |
| `MC_FIXED` | Monte-Carlo return, raw | `return − V`, separately z-scored | no | split, ≈24 vs ≈1 | Monte-Carlo |
| `GAE` (baseline) | λ-return, raw | GAE advantage, separately z-scored | no | split, ≈24 vs ≈1 | GAE(λ=0.95) |
| `MC_RAW` (new) | Monte-Carlo return, raw | raw residual `return − V` | yes | ≈ 20-24 | Monte-Carlo |

Source: `src/models/recurrent_ppo_trainer.py`, `train_iteration`, the five branches on
`return_mode`. `MC_RAW` is the `MC_FIXED` branch with exactly one line — the advantage
normalisation — deleted. `GAE_NORM` is the `GAE` branch with `MC`'s two normalisation lines
substituted for `GAE`'s.

### 2.2 Controlled variables — verified, not assumed

Environment `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`: a 10x10 grid,
500-step episode cap, 4 regenerating food resources, 12 static damaging hazards, 2 hunting
predators, 2 harmless "rabbits", 42 bushes/rocks. The agent pays 1 nutrition per step, starts
each episode at a **uniformly random nutrition between 0 and 100**, and each unit of food
restores 6. Six actions: four moves, Rest, Eat. Recurrent PPO (GRU, 128 hidden units, actor and
critic sharing the encoder and the recurrent body, separate output heads), neuromodulation off,
γ = 0.95, GAE λ = 0.95, PPO clip ε = 0.1, entropy coefficient 0.01, value coefficient 0.5,
`max_grad_norm` 0.5, learning rate 5×10⁻⁴, one Adam optimiser over all parameters preceded by
global-norm clipping, 4 epochs per update, 128 parallel environments, 128-step rollouts
(1 iteration = 16,384 environment steps), 10,000,000 episodes, seeds 42-46.

The check was run against the configuration files **the trainer itself wrote** into each run
directory (`results/JAX_RecurrentPPO/*/models/config.yaml`), not against a fresh reload of the
source YAMLs. Flattening all 25 saved configs gives **559 keys**, of which exactly **four**
differ across runs: `agent.return_mode`, `seed`, `tag`, `wandb.name`. The last two are labels.
Extraction: `tmp/cmp10m/cfgdiff.py`.

### 2.3 Confounds and limitations

| # | Confound | Severity | Assessment |
|---|---|---|---|
| **C1** | **The budget is 10M *episodes*, not environment steps, and better settings run longer episodes.** Total experience consumed differs by 3.3x: GAE_NORM 1,563M steps, MC 1,516M, GAE 806M, MC_FIXED 619M, MC_RAW 481M. **Every comparison at "10M episodes" is confounded.** | **Critical** | Handled throughout by re-plotting against environment steps, which the trainer logs directly as `timesteps` (§4.2). The confound is not small and it does not point the way one might guess — see §5.1. |
| **C2** | **Right-censoring.** Three of the five arms had not stopped improving when the budget ran out. Any statement about their *ceiling* is unsupported. | **Critical** | Stated explicitly wherever it bites (§5.4). The document deliberately declines to answer the ceiling question. |
| C3 | Three of five arms were never pre-registered; two were. | Medium | Flagged per claim. The `MC_RAW` prediction was recorded in advance and is scored honestly in §5.5. |
| C4 | Only one hyperparameter set. The slow settings might be recoverable by re-tuning (learning rate, value coefficient, clip threshold, entropy coefficient). | **High for generality** | Untested. The result reads *"at this project's current hyperparameters"*, not as a general fact about PPO. |
| C5 | The environment's rewards are never normalised — no reward wrapper, no running return scaler, no PopArt. | High for external validity | Unchanged from the 1M analysis; still the most likely reason the surveyed convention does not transfer. |
| C6 | **No per-term gradient norms, no explained variance for raw-target arms, no advantage-scale metric, no KL or clip fraction.** §5.6's mechanism question is *about gradients*; only *loss values* are logged. | **Critical for §5.6** | §7 Metrics Requested. This is not a shortfall in precision but a shortfall in *kind*: the policy loss term is `mean(ratio × advantage)` with advantages rescaled to mean zero, so its value is ≈ 0 in expectation however large its gradient is. The loss shares therefore cannot test the mechanism in either direction, and §5.6 reports it as **untested**, not refuted. |
| C7 | **`MC_RAW` started ~4 hours after the other arms**, on different nodes, and **at a different git commit**. The 20 runs of the other four arms carry commit `788e5983`; the five `MC_RAW` runs carry `a171ea35`, a descendant of it. | Low | **Checked, not assumed** (Appendix B). The commit range between the two touches only `docs/`, `scripts/analysis/`, `scripts/claude/` and `tests/analysis/` — **no file under `src/` or `configs/`**, i.e. nothing on the training path. Together with the clean 559-key config diff this remains a scheduling difference only, but the earlier wording ("same git commit in every run") was wrong and is corrected here. |
| C8 | The offline evaluation in §4.7 uses the **greedy (argmax) policy**, while the training-log survival series reflects the **stochastic** policy. They measure different objects and should not be differenced naively. | Medium | Both are reported side by side. Where they appear to disagree, §4.9 checks first whether the evaluation is even *able* to resolve the effect before any disagreement is interpreted — and in the one case where that was done properly, at 2,000 episodes per seed, the two objects turned out to be within **half a step** of each other for both arms tested, so the difference between them is smaller in this environment than the 200-episode evaluation suggested. |
| C9 | **The offline evaluations (§4.7, §4.9) loaded the *source* environment YAML** (`configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, resolved through its `extends:` chain), **not each run's saved copy** of that config. They also fell through the evaluation script's "no `behavior_measures` block → use defaults" branch, so the world seeds (0, 1, 2, …), the greedy policy mode and the training-time observation noise were **tooling defaults rather than a chosen list**. | Low — **checked, not assumed** | Stated explicitly rather than left implicit. Two checks. (i) Because the *same* source YAML and the *same* seed list were used for every run, all worlds are identical across arms and every **between-arm** comparison is paired and safe; only an **absolute** level could drift, and it would drift equally in all arms. (ii) The drift was checked directly: loading the source YAML through the eval loader and flattening it gives **133 environment keys, all 133 of which match a run's trainer-written `config.yaml` exactly** (the saved file has 118 further keys, all agent/training-side). No file in the `extends:` chain has been modified since the runs (`04` 2026-07-04, `03-random_init_10x10` 2026-07-22, `default.yaml` 2026-08-26; no YAML under `configs/environment/` has an mtime on or after 2026-09-04). **No config drift, therefore no drift in the absolute levels from this source.** (The absolute levels in §4.7 *are* biased upward, but for an unrelated reason — the 200-world sample, §4.9.) |

---

## 3. Runs analysed

All 25 runs completed (an `exit` record is present in every local datastore). Metrics were read
from the **local** WandB datastores (`wandb/run-*/run-*.wandb`) with a protobuf record scanner
(`tmp/parse_wandb_local.py`); no web-API call was made.

| Arm | Seeds 42 → 46 (WandB directory suffix) | Results directory prefix |
|---|---|---|
| `MC` | `xy7nic92`, `cmqugy51`, `quydmpd7`, `ymbhe3qp`, `uqnl1scm` | `results/JAX_RecurrentPPO/20260904-1738…1742_rppo_cmp10m_mc_s4*` |
| `MC_FIXED` | `se72bm9i`, `ukte3hbu`, `ef2pg37t`, `b09iehij`, `vpabnrrz` | `…20260904-1743…1747_rppo_cmp10m_mcfixed_s4*` |
| `GAE` | `4sqc0lsd`, `haw7hl3e`, `whdabu5w`, `4ug6okvu`, `96ej3ngm` | `…20260904-1748…1752_rppo_cmp10m_gae_s4*` |
| `GAE_NORM` | `26mwmoc9`, `ac522oud`, `0mtwmifc`, `87a1r9ld`, `1v91itww` | `…20260904-1857…1903_rppo_cmp10m_gaenorm_s4*` |
| `MC_RAW` | `j0vuwph7`, `ls7riwt1`, `yo02nvg9`, `37oqpt8k`, `o6yac86p` | `…20260904-2158…2159_rppo_cmp10m_mcraw_s4*` |

**What the survival series actually is.** `Episode/Steps` is the arithmetic mean episode length
over a rolling window of the last **5,000 completed training episodes** (`Episode/_window_n`),
emitted every 4,000 episodes, counting only episodes that terminated (including by the 500-step
cap). It is the *stochastic* training policy's episode length. Two consequences matter:

- **There is no evaluation survival metric in WandB to check it against.** The `eval/` namespace
  in these runs contains only `eval/video` and `eval/checkpoint_episode`
  (`src/utils/wandb_utils.py`); no survival number is logged from evaluation. The "138.2" figure
  in the 1M analysis was itself computed from `Episode/Steps`, not from an independent
  evaluation — the two numbers being compared there were two smoothing windows of the same
  series, not a proxy against a ground truth. **The proxy could not have been, and was not,
  validated against WandB.**
- An independent check therefore requires running the offline evaluation script against the
  final checkpoints, which §4.5 does. That script uses the **greedy** policy, so it measures a
  different object; the comparison is informative but is not a like-for-like validation.

---

## 4. Results

### 4.1 Primary metric — survival steps at the end of the budget

Each seed's value is the mean of the logged survival series over the **final 2% of its
episodes** (≈ 200,000 episodes, ≈ 50 logged points). The window choice is not load-bearing:
§4.4 shows the ranking and the numbers are stable across four windows from "single last point"
to "trailing 10%".

| Arm | Survival steps, mean ± sd (n = 5) | 95% CI | Per-seed, s42 → s46 | Env. steps consumed |
|---|---|---|---|---|
| **`GAE_NORM`** | **170.36 ± 0.52** | [169.90, 170.82] | 170.6, 170.1, 171.2, 169.9, 170.0 | 1,563 M |
| **`MC`** | **165.21 ± 1.89** | [163.55, 166.87] | 164.2, 166.8, 162.4, 165.7, 166.9 | 1,516 M |
| **`GAE`** | **126.77 ± 10.14** | [117.88, 135.66] | 135.2, 134.7, 112.2, 120.3, 131.5 | 806 M |
| **`MC_FIXED`** | **94.60 ± 25.70** | [72.07, 117.13] | 88.1, 61.9, 122.6, 118.7, 81.8 | 619 M |
| **`MC_RAW`** | **78.85 ± 50.73** | [34.38, 123.32] | 131.7, 136.7, 45.0, 35.5, 45.3 | 481 M |

**Read this table with C1 in mind.** The right-hand column is why: the top arm ran on 3.3x the
experience of the bottom one. §4.2 removes that.

**The 1M result replicates inside these runs.** The separate 1M-episode run set finished at
`MC` 138.8, `MC_FIXED` 40.0, `GAE` 41.9; these 10M runs, read at the moment they passed 1M
episodes, give **134.4 ± 3.9**, **40.5 ± 0.7** and **41.6 ± 1.2** — an independent replication
with fifteen fresh runs. On the episode axis `MC_FIXED` then goes 40 → 95 and `GAE` 42 → 127
over the following nine million episodes.

### 4.2 The confound removed — survival at matched environment experience

The trainer logs `timesteps` directly, so no inversion of the episode/iteration curve is needed.
The largest budget **all 25 runs** reach is **349 M environment steps** (set by `mcraw_s44`).
Values are the mean of the survival series over a trailing 2% window in step-space.

| Env. steps | `MC` | `GAE_NORM` | `GAE` | `MC_FIXED` | `MC_RAW` |
|---|---|---|---|---|---|
| 50 M | 123.6 ± 3.9 | 125.2 ± 5.0 | 43.8 ± 1.8 | 42.4 ± 1.0 | 34.7 ± 1.7 |
| 100 M | 139.9 ± 3.3 | 142.3 ± 1.7 | 47.3 ± 2.2 | 47.9 ± 3.5 | 35.8 ± 2.3 |
| 200 M | 148.8 ± 2.3 | 154.1 ± 1.9 | 61.9 ± 7.0 | 53.3 ± 2.2 | 38.3 ± 4.9 |
| 300 M | 154.2 ± 2.8 | 158.3 ± 1.6 | 88.6 ± 12.3 | 64.5 ± 11.4 | 57.0 ± 28.2 |
| **349 M** | **156.8 ± 4.1** | **158.9 ± 1.1** | **98.3 ± 14.0** | **69.3 ± 11.7** | **70.4 ± 39.4** |

Per-seed at 349 M environment steps (sorted within arm):

| Arm | s-values | mean ± sd |
|---|---|---|
| `GAE_NORM` | 157.2, 158.3, 159.2, 159.7, 159.9 | 158.85 ± 1.09 |
| `MC` | 151.6, 153.2, 158.9, 159.4, 161.0 | 156.82 ± 4.12 |
| `GAE` | 81.6, 86.1, 100.5, 109.6, 113.4 | 98.25 ± 14.04 |
| `MC_RAW` | 35.5, 44.7, 45.4, 107.7, 118.6 | 70.38 ± 39.42 |
| `MC_FIXED` | 55.0, 59.7, 70.7, 79.9, 81.1 | 69.27 ± 11.74 |

Pairwise contrasts at 349 M environment steps (Welch, n = 5 per arm):

| Contrast | Difference | Ratio | *p* | Cohen's *d* |
|---|---|---|---|---|
| `GAE_NORM` − `MC_FIXED` | +89.6 | 2.29x | 6.2×10⁻⁵ | 10.7 |
| `MC` − `MC_FIXED` | +87.6 | 2.26x | 2.0×10⁻⁵ | 9.9 |
| `GAE_NORM` − `MC_RAW` | +88.5 | 2.26x | 0.0074 | 3.2 |
| `MC` − `GAE` | +58.6 | 1.60x | 4.0×10⁻⁴ | 5.7 |
| `GAE` − `MC_FIXED` | +29.0 | 1.42x | 0.0080 | 2.2 |
| `GAE` − `MC_RAW` | +27.9 | 1.40x | 0.20 | 0.9 |
| `MC_FIXED` − `MC_RAW` | −1.1 | 0.98x | **0.95** | −0.04 |
| `GAE_NORM` − `MC` | +2.0 | 1.01x | **0.34** | 0.7 |

The last two rows are the two designed single-variable contrasts, and both are null at this
budget. `GAE_NORM` − `MC` becomes non-null at larger matched budgets (§4.3); `MC_FIXED` −
`MC_RAW` never does.

**The matched-experience gap is *larger*, not smaller, than the matched-episode gap.** On the
episode axis the ratio to `MC_FIXED` is 1.75x; on the step axis it is 2.26x. This is the
opposite of the intuitive direction, and the reason is mechanical: at 349 M steps the fast arms
are only 23% of the way through their own runs and nearly flat, while the slow arms are at 56%
of theirs and climbing steeply. Matching on steps therefore *penalises* the fast arms and they
still win by more.

### 4.3 `GAE_NORM` versus `MC`, at matched experience, over the full range

Both arms reach ≈ 1,500 M environment steps, so they can be compared at matched experience far
beyond the 349 M floor imposed by the slowest run.

| Env. steps | `MC` | `GAE_NORM` | Difference |
|---|---|---|---|
| 200 M | 148.8 ± 2.3 | 154.1 ± 1.9 | +5.3 |
| 400 M | 157.0 ± 1.7 | 160.8 ± 0.9 | +3.8 |
| 800 M | 160.7 ± 1.6 | 166.3 ± 0.3 | +5.6 |
| 1,200 M | 163.4 ± 1.2 | 168.7 ± 0.3 | +5.2 |
| **1,488 M** (both arms' common maximum) | **164.8 ± 1.6** | **169.5 ± 0.8** | **+4.7** |

At 1,488 M environment steps the per-seed values are `MC` 162.5, 164.5, 164.6, 166.0, 166.5 and
`GAE_NORM` 168.4, 168.8, 169.9, 170.0, 170.3 — **complete separation**, every `GAE_NORM` seed
above every `MC` seed. Exact Mann-Whitney two-sided *p* = 0.0079, which is the smallest value
attainable with five versus five. The same complete separation holds on the episode axis at 10M
episodes (`MC` 162.4-166.9, `GAE_NORM` 169.9-171.2), same exact *p*.

### 4.4 Robustness of the endpoint numbers to the smoothing window

Survival at the matched 349 M-step budget under four window definitions:

| Window | `MC` | `GAE_NORM` | `GAE` | `MC_FIXED` | `MC_RAW` |
|---|---|---|---|---|---|
| single interpolated point | 158.1 ± 5.3 | 156.4 ± 1.5 | 99.4 ± 13.7 | 66.0 ± 13.3 | 71.2 ± 40.5 |
| trailing 2% | 156.8 ± 4.1 | 158.9 ± 1.1 | 98.3 ± 14.0 | 69.3 ± 11.7 | 70.4 ± 39.4 |
| trailing 5% | 156.5 ± 2.3 | 159.6 ± 1.2 | 97.3 ± 15.2 | 67.3 ± 10.6 | 69.5 ± 39.6 |
| trailing 10% | 155.5 ± 2.0 | 159.4 ± 0.9 | 97.0 ± 14.1 | 65.0 ± 8.4 | 67.1 ± 37.5 |

The arm ordering and every effect size in §4.2 are unchanged across all four. Note that the
single-point view is genuinely noisy (each logged point is a 5,000-episode mean, and adjacent
points differ by 3-5 steps), which is why the trailing-2% window is used throughout. The one
place the window choice changes a *sign* is `MC` vs `GAE_NORM` at exactly 349 M steps, where the
two arms are within noise of each other in any case (§4.2, *p* = 0.34).

### 4.5 Learning dynamics — every arm escapes, at wildly different times

**Survival against episodes** (arm mean ± sd over 5 seeds; trailing 2% window):

| Episodes | `MC` | `GAE_NORM` | `GAE` | `MC_FIXED` | `MC_RAW` |
|---|---|---|---|---|---|
| 1 M | 134.4 ± 3.9 | 137.8 ± 5.0 | 41.6 ± 1.2 | 40.5 ± 0.7 | 34.1 ± 1.3 |
| 2 M | 151.0 ± 3.0 | 155.0 ± 1.2 | 45.8 ± 1.4 | 45.3 ± 0.9 | 35.1 ± 1.9 |
| 4 M | 158.9 ± 2.2 | 163.8 ± 0.8 | 57.3 ± 7.2 | 50.7 ± 2.5 | 35.9 ± 2.8 |
| 6 M | 161.6 ± 1.8 | 166.6 ± 0.4 | 94.2 ± 17.2 | 62.3 ± 8.7 | 41.2 ± 10.6 |
| 8 M | 163.4 ± 1.0 | 168.8 ± 0.3 | 113.5 ± 16.4 | 80.6 ± 22.9 | 65.9 ± 42.8 |
| 10 M | 164.8 ± 1.1 | 170.1 ± 0.3 | 125.7 ± 8.3 | 92.4 ± 27.0 | 77.2 ± 51.4 |

**The four specific checks the 1M analysis asked for**, all answered:

| 1M analysis asked | Answer at 10M |
|---|---|
| (i) Does either losing arm's **starvation rate** ever turn over and start falling — the signature of the winner's escape? | **Yes, all three.** `MC_FIXED` peaks at 55.2% around 4M episodes and falls to 46.3%; `GAE` peaks at 56.5% around 3M and falls to 37.3%; `MC_RAW` peaks at 62.3% around 4M and falls to 50.6%. (`MC`/`GAE_NORM` sit flat at 29%.) |
| (ii) Does either ever record a **non-zero rate of reaching the 500-step cap**? Five seeds × 1M episodes produced exactly zero. | **Yes, all three.** By 10M episodes: `GAE` 14.4% of episodes, `MC_FIXED` 7.6%, `MC_RAW` 6.4%. First non-zero values appear around 1-2M episodes for `MC_FIXED`/`GAE` and around 5-6M for `MC_RAW`. |
| (iii) Does `MC` **saturate near the cap**, compressing the gap from above? | **No.** `MC`'s cap-reaching rate plateaus at 23.5% and its survival plateaus at 165 of a possible 500. There is no ceiling effect at play. |
| (iv) Does the raw-target arms' **value loss keep drifting upward**, indicating slow critic degradation? | **No.** `MC_FIXED`'s value loss falls 267 → 205 between 1M and 10M episodes, `GAE`'s 165 → 103. `MC_RAW`'s is flat (188 → 198). The "slow degradation" reading is refuted. |

**Escape time.** *One* definition is used throughout this document from here on: **the first
episode at which the survival series reaches 50 steps and holds it for 10 consecutive logged
points (~40,000 episodes)**. 50 steps is chosen because ~40-45 steps is the "never eat anything"
ceiling of this environment (§4.6), so crossing 50 means foraging has started. (Two other
thresholds appear in the working scripts — a 60-step variant, and a "ramp-onset" definition that
dates the first sustained rise rather than a level crossing. All three give the same rank
ordering of seeds and arms, so no test in this document changes with the choice; the 50-step
hold-10 definition is the one every number quoted here uses.)

| Arm | Escape episode, per seed (M) | Escape env. steps, per seed (M) |
|---|---|---|
| `GAE_NORM` | 0.37, 0.38, 0.38, 0.44, 0.48 | 15, 16, 16, 18, 20 |
| `MC` | 0.30, 0.36, 0.43, 0.44, 0.58 | 11, 13, 17, 17, 23 |
| `GAE` | 2.34, 2.68, 3.17, 3.46, 3.69 | 98, 113, 138, 145, 158 |
| `MC_FIXED` | 2.14, 3.02, 3.04, 3.66, 4.10 | 88, 126, 128, 158, 176 |
| `MC_RAW` | 5.65, 6.36, **> 10**, **> 10**, **> 10** | 224, 227, — , — , — |

Values are sorted within each arm, not matched to seeds. The two matched-at-spread-1 arms escape
after **11-23 M environment steps**; the split-scale arms need **88-176 M**, and `MC_RAW`
**224-227 M** for the two seeds that escaped at all — a **6-10x** difference in the experience
required just to begin foraging.

### 4.6 What the losing agents do — behaviour at the end of training

End-of-training behaviour (final 5% of episodes; arm mean over 5 seeds). Contrast with the
1M-episode table, where `MC_FIXED` and `GAE` ate 1.7 and 2.2 food per episode and reached the
cap in 0.000% of episodes.

| Metric | `MC` | `GAE_NORM` | `GAE` | `MC_FIXED` | `MC_RAW` |
|---|---|---|---|---|---|
| Survival steps | 164.9 | 170.3 | 126.4 | 93.9 | 78.2 |
| Food eaten per episode | 33.6 | 37.6 | 24.2 | 14.4 | 10.7 |
| Food eaten **per step** | 0.204 | 0.221 | 0.191 | 0.154 | 0.137 |
| Rest **fraction of steps** | 0.388 | 0.404 | 0.464 | 0.446 | **0.646** |
| Episodes reaching the 500-step cap | 23.5% | 23.8% | 14.4% | 7.6% | 6.4% |
| Episodes ending in starvation | 29.1% | 29.1% | 37.3% | 46.3% | 50.6% |
| Longest episode observed | 500 | 500 | 500 | 500 | 364 |
| **Policy entropy (nats; max = ln 6 = 1.79)** | **0.672** | **0.571** | **0.401** | **0.544** | **0.095** |

**`MC_RAW` is the only arm with an entropy collapse**, and it is not marginal. Translating
entropy into the probability of taking a non-greedy action (assuming one dominant action and
five equally-weighted alternatives). All entropies here are **end-of-training** values (final 5%
of episodes), matching the table above; the arm means quoted at the matched 349 M-step budget in
§4.8 are slightly different numbers (`MC_RAW` 0.086 there) because they are a different window.

| Arm / seed | Entropy (nats) | P(non-greedy action) | Expected non-greedy actions in a 35-step episode |
|---|---|---|---|
| `MC` | 0.672 | 15.2% | 5.3 |
| `GAE_NORM` | 0.571 | 12.3% | 4.3 |
| `MC_FIXED`, weakest seed (s43) | 0.617 | 13.6% | 4.8 |
| `GAE` | 0.401 | 7.8% | 2.7 |
| `MC_RAW`, arm mean | 0.095 | 1.4% | 0.48 |
| **`MC_RAW` s45 (the stuck seed)** | **0.013** | **0.14%** | **0.048** |

The stuck `MC_RAW` seed deviates from its greedy action roughly **once every 21 episodes**. That
is about **eighty-five times less exploration** than the weakest `MC_FIXED` seed, which is
otherwise its behavioural twin.

Per-seed end-of-training detail for the three low-performing arms:

| Arm / seed | Survival | Food/ep | Rest/ep | Longest episode | Cap rate | Entropy |
|---|---|---|---|---|---|---|
| `mcfixed` s43 (weakest) | 59.7 | 5.6 | 26.8 | 500 | 1.1% | 0.617 |
| `mcfixed` s44 (best) | 122.5 | 22.1 | 56.5 | 500 | 13.5% | 0.456 |
| `gae` s44 (weakest) | 112.7 | 20.3 | 51.8 | 500 | 11.9% | 0.405 |
| `mcraw` s43 (best) | 137.1 | 25.9 | 78.1 | 500 | 17.1% | 0.178 |
| `mcraw` s44 | 41.9 | 1.46 | 30.0 | 338 | 0.001% | 0.057 |
| `mcraw` s46 | 44.9 | 2.15 | 35.7 | 380 | 0.0003% | 0.040 |
| **`mcraw` s45 (stuck)** | **35.4** | **0.00006** | **32.2** | **exactly 100** | **0%** | **0.013** |

**`Episode/Steps_Max` = exactly 100 is a perfect detector of a zero-food policy.** The agent
starts each episode with a uniformly random nutrition between 0 and 100 and pays 1 per step, so
an agent that never eats cannot live past step 100 in any episode. `mcraw_s45` has a longest
observed episode of exactly 100.0 for the final seven million episodes. No other run in the set
ends in that state.

### 4.7 Independent check — offline evaluation of the final checkpoints

Because WandB carries no evaluation survival metric (§3), the final checkpoint of **all 25 runs**
was re-run through `scripts/eval/eval_rollout.py` with **200 episodes and identical evaluation
seeds in every run** — same world layouts, same starting nutrition, same predator placements —
so every pair of arms is compared on the same 200 worlds. The script uses the **greedy (argmax)
policy**; the training series reflects the **stochastic** policy, so these measure different
objects (C8).

| Arm | Greedy-eval survival, mean ± sd (n = 5 seeds × 200 episodes) | Training-log survival | Difference | Median episode | 500-step cap rate |
|---|---|---|---|---|---|
| `MC` | 177.59 ± 3.31 | 165.21 | +12.4 | 46 | 26.5% |
| `GAE_NORM` | 177.55 ± 5.26 | 170.36 | +7.2 | 48 | 25.4% |
| `GAE` | 137.32 ± 10.84 | 126.77 | +10.6 | 32 | 17.4% |
| `MC_FIXED` | 97.28 ± 33.23 | 94.60 | +2.7 | 31 | 7.8% |
| `MC_RAW` | 86.94 ± 58.03 | 78.85 | +8.1 | 31 | 8.5% |

> **Read the absolute levels in this table with §4.9 in hand.** Every one of the 25 runs was
> evaluated on the **same** 200 worlds, and that particular draw turns out to have been an easy
> one — worth **+12.0 steps** on the `MC` arm and **+7.5** on `GAE_NORM` relative to a 2,000-world
> sample, against a standard error of ±13 steps for a 200-world mean. So the survival column is
> **too high**, and the "Difference" column, which reads that inflation as a greedy-versus-training
> bias, is largely an artefact: against 2,000 worlds it falls to **+0.4 for `MC` and −0.3 for
> `GAE_NORM`**. Because the offset is *common to all 25 runs*, **the arm ordering, the between-arm
> gaps and all the paired-episode comparisons below are unaffected**; only the absolute numbers
> and the "Difference" column move. The `MC` and `GAE_NORM` rows are superseded by §4.9; the other
> three arms were not re-run, so the size of their offset is unmeasured (smaller in absolute
> terms, since their episode-length spread is smaller).

**Three things follow.**

1. **The training-log survival series is a good proxy for *ranking arms*, and it is not
   validated for anything finer.** Across the 25 runs the two measures correlate at Pearson
   *r* = **0.992** (Spearman ρ = 0.945) — but that correlation is driven almost entirely by the
   between-arm range (34 to 185 steps), so it establishes that the proxy orders the arms
   correctly and nothing more. The *within-arm* validity of the proxy — the only thing that
   matters for a 5-step question — is untested. The greedy policy appears to survive **+8.2 steps
   on average** here, higher in 21 of 25 runs. **No *p*-value is quoted on that apparent bias**:
   all 25 evaluations share the same 200 worlds, so the world-sampling offset (per-episode
   sd ≈ 210 → about ±13-15 steps on a 200-episode arm mean for the high arms) is *common* to all
   of them and they are not 25 independent observations. **§4.9 then measured it directly on
   2,000 worlds and the bias collapsed to +0.36 and −0.32 steps in the two arms re-run — so the
   +8.2 was the world draw, not a greedy-versus-stochastic effect.** **Use the proxy for
   arm-level ranking; do not difference it against a greedy number at the ±10-step level.**
2. **The greedy evaluation at 200 episodes cannot resolve the `GAE_NORM` − `MC` gap, in either
   direction.** It reads 177.55 versus 177.59, but its 95% confidence interval on the difference
   is about **[−6.4, +6.4] steps**, which *contains* the +4.7 to +5.2-step training-time effect.
   The two measurements do not disagree; one of them is too coarse to see the effect. §4.9
   re-runs the evaluation at ten times the episode count to settle it, and §5.2 states the
   conclusion.
3. **Arm ordering is otherwise unchanged**, and the `MC_FIXED`/`MC_RAW` seed dispersion is
   reproduced independently (sd 33 and 58 steps on 200 fresh paired episodes).

**Paired-episode structure.** Restricting to the episodes in which both arms survived at least
30 steps — i.e. excluding the episodes that were unwinnable from the start — the gap is far
larger than the means suggest: `MC` 307 versus `MC_FIXED` 171 (107 shared episodes), `MC` 316
versus `GAE` 248 (106 episodes), `MC` 303 versus `MC_RAW` 149 (109 episodes), and `MC` 287
versus `GAE_NORM` 290 (119 episodes). As at 1M, most of the difference lives in the upper tail
of survivable episodes.

**What a stuck `MC_RAW` seed does, versus a slow `MC_FIXED` seed** (200 identical paired
episodes each):

| | `mcfixed_s43` (slowest `MC_FIXED`) | `mcraw_s45` (the stuck seed) |
|---|---|---|
| Greedy-eval survival | 51.9 steps | 33.9 steps |
| Longest episode in 200 | **500** (hits the cap) | **exactly 100** |
| Food eaten per episode | 3.88 | **0.000** |
| Episodes with **zero** Eat actions | 117 / 200 | **198 / 200** |
| Eat action share | **0.202** | **0.000** |
| Rest action share | 0.383 | **0.904** |
| Longest unbroken run of Rest | mean 10.8, worst 32 | mean 27.8, worst **100** (a whole episode) |
| Abandoned actions (share < 0.1%) | none | **"Down" (0.000)** |
| Policy entropy at end of training | 0.617 nats | 0.013 nats |

### 4.8 Training-health diagnostics

Loss-budget shares. "Loss budget" is the sum of the three terms' absolute contributions to the
quantity actually differentiated — `abs(policy term) + abs(0.5 x value term) + abs(0.01 x entropy term)`. Values are arm means at
the matched 349 M-environment-step budget; the gradient norm is a median (it has heavy spikes).

| Arm | policy % | value % | entropy % | policy-term / entropy-term | gradient norm (pre-clip) | multiple of the 0.5 clip | updates clipped |
|---|---|---|---|---|---|---|---|
| `MC` | 1.3296 | 93.459 | 5.212 | 0.26 | 0.275 | 0.6x | **0%** |
| `GAE_NORM` | 1.4230 | 93.610 | 4.967 | 0.29 | 0.247 | 0.5x | **0%** |
| `MC_FIXED` | 0.0017 | 99.9931 | 0.0052 | 0.34 | 63.1 | 126x | **100%** |
| `GAE` | 0.0045 | 99.9869 | 0.0086 | 0.52 | 41.7 | 83x | **100%** |
| `MC_RAW` | **0.0348** | 99.9644 | 0.0008 | **42.25** | 63.4 | 127x | **100%** |

Critic quality. For the two z-scored-target arms the target has spread 1 by construction, so
the residual root-mean-square error √(2 × value loss) converts directly into explained variance
(1 − RMSE²). For the raw-target arms the target's spread is not logged, so explained variance is
**not computable** and only the raw residual is reported.

| Arm | 0.1-0.3 M ep | 1 M ep | 5 M ep | 9.5-10 M ep |
|---|---|---|---|---|
| `MC` explained variance | 55.6% | 52.9% | 53.7% | **53.3% ± 0.44** |
| `GAE_NORM` explained variance | 65.9% | 58.4% | 57.9% | **58.3% ± 0.24** |
| `MC_FIXED` residual RMSE (raw return units) | — | 23.1 | — | 20.1 |
| `GAE` residual RMSE (raw return units) | — | 18.2 | — | 14.4 |
| `MC_RAW` residual RMSE (raw return units) | — | 19.4 | — | 19.8 |

**Caveat on the explained-variance comparison.** `GAE_NORM`'s target is a λ-return, which
*contains the critic's own predictions*, so fitting it is partly self-referential and
mechanically easier than fitting a pure Monte-Carlo return. The 5-point gap over `MC` is
consistent across seeds and across training, but it should not be read as "`GAE_NORM`'s critic
is 5 points better at the same task".

### 4.9 The evaluation re-run at ten times the sample size — and what it changes

**Why it was run.** The 200-episode evaluation in §4.7 read the `GAE_NORM`-versus-`MC` gap as
177.55 against 177.59 and an earlier draft of this document called that a "dead tie" that made
the training-time result disappear. That reading was wrong on its own terms: at five seeds ×
200 shared worlds the 95% confidence interval on the difference is about **[−6.4, +6.4] steps**,
which *contains* the +4.7 to +5.2-step training-time effect. A measurement that cannot resolve
an effect is not a measurement that contradicts it. So the evaluation was simply re-run larger.

**What was run.** Evaluation only — **no retraining**. The same final checkpoints of the `MC` and
`GAE_NORM` arms, all five seeds each, through the same `scripts/eval/eval_rollout.py` with the
same greedy (argmax) policy, the same source environment configuration, and the same shared world
seeds — now **2,000 episodes per run instead of 200** (20,000 evaluation episodes in total). The
new world set is a strict **superset** of the old one, which gives a free integrity check:
restricting each run's 2,000 episodes to its first 200 reproduces every number in §4.7 to two
decimal places.

| Arm | Greedy-eval survival, 2,000 episodes/seed | Per-seed, s42 → s46 | Training-log survival | Same runs, first 200 worlds only |
|---|---|---|---|---|
| **`GAE_NORM`** | **170.04 ± 2.55** | 170.2, 170.3, 169.9, 166.3, 173.5 | 170.36 | 177.55 |
| **`MC`** | **165.57 ± 2.55** | 164.4, 167.2, 161.6, 167.3, 167.3 | 165.21 | 177.59 |

**Result 1 — the gap is real, and it is the same gap the training logs showed.**

| Statistic | `GAE_NORM` − `MC` |
|---|---|
| Difference of arm means | **+4.47 steps** |
| 95% CI, seeds as the random unit (Welch, 5 v 5) | **[+0.75, +8.19]**, *p* = **0.024** |
| 95% CI, seed bootstrap | [+1.82, +7.37] |
| 95% CI, worlds as the random unit (2,000 paired worlds) | **[+2.08, +6.85]** |
| Minimal detectable difference at 80% power | **≈ 5.2 steps** (was ≈ 8 at 200 episodes) |
| Per-seed pairs favouring `GAE_NORM` | 22 of 25 |
| Training-time effect, for comparison | +4.7 to +5.2 steps |

The interval excludes zero on both the seed-level and the world-level analysis, and it is
centred within half a step of the training-time estimate. **`GAE_NORM` does survive longer than
`MC` under the greedy policy, by about 4.5 steps.** The 200-episode "tie" was a resolution
failure, not a contradiction. Note the honest limit: separation is no longer *complete* —
`gaenorm_s45` (166.31) falls below three `MC` seeds, so the exact Mann-Whitney *p* is 0.056
rather than 0.008. Four of five `GAE_NORM` seeds are above all five `MC` seeds.

**Result 2 — the "+8.2-step greedy bias" in §4.7 was mostly a world-sampling artefact, and the
training-log proxy is better than that section credited.** Against 2,000 worlds:

| Arm | Training-log survival | Greedy eval, 2,000 episodes | Difference |
|---|---|---|---|
| `MC` | 165.21 | 165.57 | **+0.36** |
| `GAE_NORM` | 170.36 | 170.04 | **−0.32** |

The bias is under half a step in each arm and the two disagree in sign. What produced the
apparent +8.2 is visible directly: the first 200 worlds were an easy draw, worth **+12.0 steps**
on the `MC` arm relative to the full 2,000-world mean — and the standard error of a 200-world
mean, computed from the 2,000-world spread, is **13.4 steps**. Because all 25 runs in §4.7 shared
those same 200 worlds, that offset was common to every one of them; it inflated the arms' absolute
levels together and cancelled out of every between-arm comparison. **§4.7's arm *ordering* and its
between-arm gaps stand unchanged. For the two arms re-run here its absolute levels are 7.5 and 12
steps too high, and the "greedy policy survives longer than the training policy" claim does not
survive.** The other three arms were not re-run, so the size of their offset is unmeasured; it
will be smaller in absolute terms because their episode-length spread is smaller, and the
direction should be the same because the worlds are shared.

**Scope, and what is still unresolved.** Only the `MC` and `GAE_NORM` arms were re-run, because
their ~5-step gap was the one the 200-episode evaluation could not resolve. The comparisons
involving `GAE`, `MC_FIXED` and `MC_RAW` against the matched-scale arms are 40-90 steps against
intervals of ±10-30, so the smaller sample already resolves them. **One pair remains unresolved
and a larger evaluation would not fix it**: `MC_FIXED` versus `MC_RAW` (bootstrap 95% interval on
the greedy difference [−61, +43]). Its noise comes from **seed dispersion** — `MC_RAW`'s two
escaped seeds against its three stuck ones — not from world sampling, so the remedy is more
seeds (§6.3 rows 1 and 5), not more episodes.

---

## 5. Analysis

### 5.1 Question 1 — does the matched-scale advantage survive matched experience? Yes, and it grows

**Answer: yes, and matching on experience makes the gap *larger*, not smaller.** At the matched
349 M-environment-step budget the two matched-at-spread-1 arms are at **156.8 ± 4.1** (`MC`) and
**158.9 ± 1.1** (`GAE_NORM`) against **98.3 ± 14.0** (`GAE`), **69.3 ± 11.7** (`MC_FIXED`) and
**70.4 ± 39.4** (`MC_RAW`). The ratio to `MC_FIXED` is **2.26x** (Welch *p* = 2×10⁻⁵,
*d* = 9.9), against 1.75x on the confounded episode axis.

**Effect sizes with seed dispersion, not means alone.** The distributions do not touch for the
`GAE_NORM`/`MC` versus `MC_FIXED` contrast: the worst matched-scale seed is at 151.6 and the
best `MC_FIXED` seed at 81.1. For `GAE` the gap is smaller but the distributions still do not
overlap (worst matched-scale 151.6, best `GAE` 113.4). For `MC_RAW` they **do** overlap: two of
its seeds are at 108-119 while three sit at 35-45, so the arm mean of 70.4 ± 39.4 describes no
individual seed and the *d* = 3.2 headline is an artefact of a bimodal sample. **`MC_RAW`'s
number should be reported as a split, never as a mean.**

**The gap shrinks with budget, but slowly.** On the step axis the ratio `MC` : `MC_FIXED` runs
2.9x (50 M steps) → 2.9x (100 M) → 2.8x (200 M) → 2.4x (300 M) → 2.3x (349 M). Extrapolating a
trend from five points on right-censored curves would be irresponsible; what can be said is that
it is closing, not that it closes.

### 5.2 Question 2 — is `GAE_NORM` genuinely above `MC`? Yes, on both measurements once the second one is large enough to see it

**On the training-log survival metric the difference is as clean as five seeds can make it.**
+5.2 steps at 10M episodes and +4.7 steps at the common 1,488 M-environment-step budget, with
**complete separation** — all five `GAE_NORM` seeds above all five `MC` seeds on both axes,
exact Mann-Whitney *p* = 0.0079, the smallest value attainable at 5 versus 5. It is not a
budget artefact: `GAE_NORM` leads at **every** matched environment-step budget checked, from
50 M to 1,488 M — though the lead is only +1.6 to +2.0 steps and not significant below ~350 M,
and reaches its full ~+5 by 800 M. It is reproducible, and it is small: **3.1% of the mean.**

**On the greedy-policy evaluation, at an adequate sample size, the same gap appears.** The
200-episode evaluation read 177.55 against 177.59 and an earlier draft of this document called
that a "dead tie" and said the training-time advantage "disappears entirely". **That was a
misreading and it is withdrawn.** At five seeds × 200 shared worlds the 95% interval on the
difference is about [−6.4, +6.4] steps — it *contains* the ~5-step effect, so the evaluation was
never in a position to confirm or deny it. Re-run at **2,000 episodes per seed** (§4.9, no
retraining), the greedy evaluation gives **`GAE_NORM` 170.04 against `MC` 165.57, a difference of
+4.47 steps, 95% CI [+0.75, +8.19] on seeds and [+2.08, +6.85] on the 2,000 paired worlds**.
That is the training-time effect, reproduced on a different measurement of the same checkpoints,
to within half a step.

**So the two measurements never disagreed.** There is no discrepancy to reconcile, and the
exploration-cost story an earlier draft offered — that `GAE_NORM` merely takes 3 percentage
points fewer random actions, so its edge should vanish under a greedy policy — **predicts the
wrong thing**. Under a greedy policy neither arm takes random actions, and the gap is still
there. Something about `GAE_NORM`'s learned policy, not its exploration bill, is worth ~4.5 steps.
(The exploration difference is real — `MC` 0.672 nats against `GAE_NORM` 0.571, i.e. 15.2% versus
12.3% non-greedy actions — it just does not account for the gap.)

**So the honest claim is now broader than the previous draft's, and still bounded.**
`GAE_NORM` survives about 4.5-5 steps longer than `MC`, and this holds both during training and
under greedy evaluation of the final checkpoints — about **2.7% of the mean**, from a single
hyperparameter setting on a single environment. Separation across seeds is strong but not perfect
on the greedy measurement (22 of 25 seed pairs favour `GAE_NORM`; one `GAE_NORM` seed falls below
three `MC` seeds), so the exact-rank *p* is 0.056 there against 0.008 during training. **Why**
`GAE_NORM` is better is not established: §5.8 raises a candidate (its λ-return is effectively a
lower-variance ~10-step target) that this data cannot check without the return-scale logging
requested in §7.

**Is the tightness itself a finding?** Partly. `GAE_NORM`'s five seeds finish within a 1.25-step
band (sd 0.52) against `MC`'s 4.5-step band (sd 1.89) — a variance ratio of 13.2. But a
variance-ratio F-test gives *p* = 0.028 while the more robust Levene test gives *p* = 0.10, and
with five seeds per arm neither is decisive; the F-test in particular is very sensitive to
normality at this sample size. The same pattern appears in escape times (`GAE_NORM` 0.37-0.48 M
episodes, `MC` 0.30-0.58 M) but is not significant there either (*p* = 0.13), and it **does not
appear in the greedy evaluation at all**: at 2,000 episodes per seed the two arms' between-seed
standard deviations are **identical to two decimal places** (2.55 and 2.55), and at 200 episodes
the ordering reversed (5.26 for `GAE_NORM` against 3.31 for `MC`). So the tightness is a property
of the *training-time* series only. **Verdict: suggestive, not established.** A mechanistic reason is available —
λ = 0.95 bootstrapping shortens the effective return horizon to about ten steps, which lowers
target variance, and `GAE_NORM`'s critic does explain 58.3% ± 0.24 of its target's variance
against `MC`'s 53.3% ± 0.44 — but the explained-variance comparison is itself partly
self-referential (§4.8) and the greedy-eval reversal argues against reading too much into it.

### 5.3 Question 3 — characterising `MC_RAW`: a bistable trap, staggered escapes, and no upstream warning detected

**First, a correction to the framing — with the two counts kept separate, because they are
different claims.** `MC_RAW` is not "two seeds escape, three permanently stuck", but neither is
it "four of five escaped". Two distinct facts:

- **Measured, by this document's own escape criterion (§4.5, cross 50 survival steps and hold
  it): 2 of 5 seeds escaped** — `s42` and `s43`. The other three finished at 35-45 steps, below
  the criterion.
- **Also measured: 4 of 5 seeds are no longer in the zero-food state.** `s44` and `s46` left it
  — in the 2,000-episode greedy evaluation they eat 3.6 and 2.3 food per episode and one of them
  reaches a 500-step episode, against literally zero food for a trapped seed. That is a
  qualitative change of state, and it is a fact, not an extrapolation.
- **Extrapolated, and labelled as such:** that `s44` and `s46` would have *completed* the ramp
  had the budget continued. This is an argument by analogy with `s42`'s ramp shape, not an
  observation. `s44`'s departure from the trap occupies only the last ~0.5-1 M of its 10 M
  episodes, and both seeds finish at ~45 survival steps. **The right summary is "2 of 5 crossed
  the escape threshold; 4 of 5 had left the zero-food state; whether the latter two would have
  finished is not known."**

The fifth seed, `s45`, moved the *other* way. The per-seed history:

| Seed | 0 → 3 M episodes | What happened next | State at 10 M |
|---|---|---|---|
| `s43` | **never entered the trap** — ate 0.42-1.11 food/episode from 200k episodes onward, longest episode 180-265 | steady growth from ~3 M, crossed 50 steps at 5.7 M | 137 steps, 25.9 food/ep |
| `s42` | **in the trap by 500k** — food exactly 0, Rest 31 of 33 steps, longest episode exactly 100 | escape begins ~4.75 M, food 0 → 0.83 → 2.1 → 6.6 → 24.1 over the next 5 M episodes | 132 steps, 24.0 food/ep |
| `s46` | in the trap by 500k, identical signature | longest episode first exceeds 100 at ~6.0 M; food 0.002 → 0.089 → 1.19 → 2.27 | 45 steps, 2.15 food/ep, **still climbing** |
| `s44` | in the trap by 500k, identical signature | longest episode stays at **exactly 100.0** until 8.5 M, then 110 → 235 → 430 | 42-46 steps, 1.46 food/ep, **still climbing steeply** |
| `s45` | **not** in the trap, but not foraging either — a **low-entropy wanderer** at 35-37 survival steps (the do-nothing level) that ate 0.43-0.70 food/ep incidentally, Rest 0.02 of 37 steps, longest episode 190-234 | **switched to resting at ~3-4 M and stopped eating**: food → 0, Rest → 32, longest episode → exactly 100, and stayed there for the remaining 6-7 M episodes | 35.4 steps, food 0.000 |

So the phenomenon has two parts, and only the second is a "stuck seed".

**(a) The trap is a discrete, absorbing-looking state with an exact signature.** Its members have
food eaten per episode of literally zero (< 10⁻⁵), Rest on ~92% of steps, a starvation rate of
63%, and a **longest observed episode of exactly 100.0 steps** — which is the arithmetic ceiling
for a non-eating agent in this environment, since starting nutrition is uniform on [0, 100] and
costs 1 per step. Entry is fast (by 500k episodes for three seeds) and, for `s45`, can happen
*late*: at ~3-4 M episodes, after the seed had spent three million episodes outside the trap. It
is worth being precise about what `s45` lost, because it is less than it first appears. At
2-3 M episodes it was surviving 35-37 steps — **the same level as a do-nothing agent** — while
eating 0.43-0.70 food per episode, with an entropy already down at 0.037 nats. That is a
**low-entropy wandering policy that ate incidentally**, not a forager: a real forager in this
environment eats 10-38 food per episode and survives 90-170 steps. So the honest description is
"a wanderer that switched to resting and stopped eating", **not** catastrophic forgetting of an
already-learned foraging behaviour. What makes it notable is the *direction* — the only seed in
25 that moved into the trap rather than out of it.

**(b) Escape is a slow, stochastic, drawn-out process, not a discrete event.** Following `s42`:
food per episode goes 0 (4.5 M) → 0.83 (5.0 M) → 1.26 → 2.10 → 3.60 → 6.57 → 12.5 → 16.8 →
21.6 → 24.1 (10 M). That is a compounding ramp over roughly 4 million episodes (≈ 150 M
environment steps), not a jump. `s44` and `s46` show the same shape starting 2-4 M episodes
later. **What varies across seeds is the waiting time before the ramp begins**, and it varies by
at least 4 M episodes.

**(c) Why the trap might be escapable but slow — a candidate account, not a demonstrated
mechanism.** The trapped policy's entropy is 0.012-0.019 nats, i.e. a **0.14% chance of a
non-greedy action**, i.e. about one exploratory action every 21 episodes. If escape requires an
exploratory Eat action to fire while the agent happens to be standing on food, then with
exploration that rare the waiting time would be enormous and heavy-tailed — which matches the
observed 4-M-episode spread in escape times. The comparison arm is consistent: `MC_FIXED`'s
weakest seed sits at 0.617 nats (13.6% non-greedy), **85x more exploration**, and never enters
this state. **This is a correlation across arms plus a plausible story, not a test.** The low
entropy could equally be a *consequence* of having settled into resting rather than its cause
(§5.6); the entropy-coefficient intervention in §6.3 row 1 is what would separate them.

**(d) The most valuable question — does any logged training signal distinguish escapers from
non-escapers before they diverge? None was detected, at a sample size that could only have
detected a strong one.** Three tests, all null:

- **Within `MC_RAW`, in a pre-divergence window (3.0-4.0 M episodes, when all four trapped
  seeds have food = 0 and longest episode = 100.0).** The window is not perfectly clean: `s45`
  *entered* the trap during it (its Rest count goes 0.002 → 29 → 32 across the 2 M / 3 M / 4 M
  marks), so for that seed the window straddles the very transition it is supposed to precede.
  Fourteen diagnostics were rank-correlated
  with escape time across the four seeds. Two produced a perfect rank ordering — pre-clip
  gradient norm (68.6, 64.5, 64.5, 56.8 against escape at 5.0, 6.6, 9.0, never) and episode
  reward. With *n* = 4 the smallest attainable exact permutation *p* is **0.083**, and testing
  fourteen diagnostics one expects **1.2 perfect orderings by chance**. Two were found. This is
  exactly the null expectation.
- **Out-of-sample replication in the other slow arms.** `MC_FIXED` and `GAE` also have staggered
  escape times across their five seeds, giving ten more independent (seed, escape time,
  pre-escape gradient norm) triples. The gradient-norm relationship **does not replicate**:
  `MC_FIXED` ρ = +0.10 (*p* = 0.95), `GAE` ρ = −0.60 (*p* = 0.35).
- **Pooled, within-arm z-scored, across all fifteen slow-arm seeds.** Optimisation-side
  diagnostics measured at 1-2 M episodes show no relation to escape time: gradient norm
  ρ = −0.12 (*p* = 0.68), entropy ρ = +0.29 (*p* = 0.30), value loss ρ = −0.38 (*p* = 0.16),
  policy loss ρ = −0.25 (*p* = 0.37). The only correlates are **behavioural and near-tautological**
  — food eaten ρ = −0.55 (*p* = 0.033), survival steps ρ = −0.59 (*p* = 0.021), starvation rate
  ρ = +0.52 (*p* = 0.045) — i.e. "an agent already foraging a bit at 1-2 M episodes escapes
  sooner". Even these do not survive multiplicity correction across the eleven diagnostics
  tested (smallest Benjamini-Hochberg *q* = 0.23).

**Statement of the negative result, with its power stated.** *In the metrics this trainer
currently logs — gradient norm, the three loss terms and their shares, and policy entropy — no
pre-divergence predictor of which seed escapes was **detected at n = 4 to 15 seeds**.* That is
a weaker statement than "no signal exists", and the gap matters: with four seeds the smallest
*p*-value any perfect rank ordering can attain is 0.083, so **no within-arm result could have
reached significance at all**; and with fifteen pooled seeds the power to detect a moderate
rank correlation (ρ ≈ 0.5) is about **50%** — a coin flip. Whether the null reflects the absence
of a signal, insufficient seeds, or the absence of the right metric from the logs, this dataset
cannot distinguish. §6.3 row 5 (ten more `MC_RAW` seeds) is the direct remedy for the sample
size; §7 names the three cheapest candidate metrics.

**One caveat that cuts against the negative result.** Within `MC_RAW` the two seeds that
*never* entered the trap (`s43`, `s45`) were plainly distinguishable from the three that did,
from 200k episodes onward, by any behavioural measure. Yet `s43` went on to the best outcome in
the arm and `s45` to the worst. **Even a signal that predicts trap entry does not predict the
final outcome.**

### 5.4 Question 4 — did the split-scale arms asymptote lower, or were they merely slower?

**Plainly: this data does not settle it, and it comes closer to "merely slower" than the 1M
analysis expected.**

**What the data does establish.** Express each arm's progress as the environment experience it
needs to reach a given survival level, and take the ratio against `MC`. To remove survivorship
bias, the ratio is computed on a **fixed seed subset** — the seeds that reach the arm's highest
level — measured at every level.

**The full table, no rows omitted** (an earlier draft started at level 60, which hid the two
lowest values and made the factor look flatter than it is):

| Survival level | `GAE` (3 seeds) | `MC_FIXED` (2 seeds) | `MC_RAW` (2 seeds) |
|---|---|---|---|
| 50 | **6.4x** | **5.9x** | 12.9x |
| 60 | 7.8x | 10.2x | 10.6x |
| 70 | 8.7x | 11.8x | 10.7x |
| 80 | 9.1x | 12.0x | 10.2x |
| 90 | 8.9x | 12.7x | 10.1x |
| 100 | 10.4x | 13.8x | 11.0x |
| 110 | 9.8x | 14.1x | 10.2x |
| 120 | 10.0x | 12.6x | 8.2x |
| 130 | **11.6x** | — | — |

**The dilation factor spans 6x to 14x, and it drifts upward with survival level in two of the
three arms.** In the best-powered arm — `GAE`, three seeds, and the only arm reaching level 130 —
it rises **monotonically from 6.4x to 11.6x**, an 80% increase across the measured range.
`MC_FIXED` rises **5.9x → 14.1x** before dropping back to 12.6x at its last point. Only
`MC_RAW`, on two seeds, drifts down (12.9x → 8.2x).

**What that drift does and does not mean.** A time-to-level ratio that grows with level is the
early signature of the slow arms approaching a *lower asymptote* than the fast ones — that is the
shape a genuine ceiling difference produces first. It is also entirely compatible with seed noise
at two-to-three seeds per point, and the third arm moves the other way. **This document therefore
does not claim the slowdown is a constant factor, and does not claim it is diverging. It reports
6-14x with an upward drift in two arms of three, and records the ceiling question as
undetermined.** (The previous draft's "a pure constant-factor slowdown fits; a diverging one does
not" was an over-reading of a trend on two-to-three seeds — the same failure mode, in the
opposite direction, that produced the 1M "stall" call.)

Under a *pure* time-shift the aligned curves would coincide, and they do not exactly —
aligning each seed on its own escape point, `MC` is at 143 steps 100 M steps after escaping while
`MC_FIXED` is at 57 and `GAE` at 74 — so the slow arms are also slower *after* escaping, which is
what the dilation factor is measuring. Delay and slow post-escape progress are the same
phenomenon here, not two.

**What the data cannot establish, and why.** The highest level any slow arm reached is 130
survival steps; `MC` and `GAE_NORM` finish at 165 and 170. `MC` needs **907 M environment steps**
to reach 165. Holding the dilation factor fixed at the 10-13x seen near the top of the measured
range, `MC_FIXED` and `GAE` would need roughly **9-12 billion environment steps** to get there.
They were given **619 M** and **806 M** — between **11x and 19x too little experience** to test
the question. **Both of those figures are lower bounds, not estimates**: the factor is drifting
upward, so extrapolating it as a constant is the most optimistic assumption available, and if the
drift continues past level 130 the true requirement is larger. If the drift instead reflects a
genuine ceiling, the requirement is infinite. Three of the five arms were still climbing at +5 to
+6 survival steps per million episodes when the budget ended, against +0.7 for `MC` and
`GAE_NORM`, which are flat.

**So the correct statement is**: *the split-scale settings are at least ~10x less
experience-efficient; whether they eventually reach the same level is untested, and testing it
would need at least an order of magnitude more compute than this experiment used.* The 1M
analysis's "stall" reading is refuted at the budget it was made about — those arms did learn to
forage — but the *ceiling* question the stall reading was ultimately about remains open, and the
upward drift of the dilation factor is weak evidence on its side, not against it.

### 5.5 Question 5 — behaviour: the "never eats" failure mode is now confined to one run

At 1M episodes the failure was categorical: `GAE`'s Eat-action share was exactly 0.000 in 25 of
30 evaluation episodes and `MC_FIXED`'s in 22 of 30; neither arm reached the 500-step cap even
once in five seeds × 1M episodes; both were described as *non-foragers*, not weak foragers.

**That description no longer fits any `MC_FIXED` or `GAE` run.** At 10M episodes every one of
those ten runs eats (5.6-26.8 food per episode), reaches the 500-step cap in 1.1-16.1% of
episodes, and has a starvation rate that peaked and then fell. Even the weakest, `mcfixed_s43`,
takes the Eat action on 20.2% of its steps in greedy evaluation and hits the cap. They are now
**slow, unreliable foragers** — which is a categorically different thing from what they were.

**Exactly one run in the set is still a non-forager**: `mcraw_s45`. Its Eat-action share over
200 paired evaluation episodes is **exactly 0.000**, it eats **zero** food, 198 of 200 episodes
contain no Eat action at all, and its longest episode in 200 is **exactly 100 steps**. It rests
on 90.4% of steps, with unbroken Rest runs averaging 27.8 steps and reaching 100 — an entire
episode of doing nothing — and it has abandoned one movement direction outright.

**What distinguishes a stuck `MC_RAW` seed from a slow `MC_FIXED` one is exploration, not
competence.** `mcfixed_s43` and `mcraw_s45` finish at similar training-log survival (60 and 35)
and similar greedy-eval survival (52 and 34), and both spend most of their time resting. But
`mcfixed_s43` retains a broad policy (0.617 nats, 13.6% non-greedy actions, longest Rest run 32)
and therefore keeps sampling the behaviours that pay; it is climbing. `mcraw_s45` has a
near-deterministic policy (0.013 nats, 0.14% non-greedy, longest Rest run 100) and has stopped
sampling anything; it is flat. **Descriptively, the difference between them is exploration, not
competence — neither has learned to forage, but one is still sampling and one is not.** Whether
the loss of sampling *caused* `mcraw_s45`'s outcome or followed from it is not settled here
(§5.6).

The two remaining low `MC_RAW` seeds sit between these: `s44` and `s46` eat 3.6 and 2.3 food per
episode in evaluation, reach 500 and 390 steps at best, and are visibly mid-escape.

### 5.6 Question 6 — the shared-trunk mechanism: unmeasured, and the loss-share evidence is uninformative either way

**The claim under test, stated as its author stated it.** In `ActorCriticRNN` the actor and
critic share the encoder and the GRU; only the two output heads are separate, and one Adam
optimiser sees the sum of both terms' gradients. The 1M analysis's Finding 3 is explicit that its
argument is **not** about loss magnitudes or effective learning rates but about **gradient
direction**: *"this analysis does not make [the learning-rate argument]. The argument is about
**direction**… For every shared parameter, the gradient Adam sees is the *sum* of a value term
and a policy term… When the value term is ~40,000x larger, the shared recurrent representation
is optimised essentially entirely for value regression"*
(`return_mode_cmp_1M.md:341-349`). The loss shares it quoted were offered as **evidence for**
that claim, not as the claim. An earlier draft of this section characterised the claim as being
about the summed *loss* and then declared it refuted; that was a mischaracterisation and is
withdrawn.

Independently, the literature review established that **no paper in the surveyed corpus measures
this**; it appears once as a one-sentence conjecture in a practitioner's blog post, and the same
corpus contains a counterexample (the reference PPO implementation deliberately shares a
convolutional trunk on Atari, and the only recurrent PPO in that corpus is built on it).

**Why nothing in this dataset can settle it — the load-bearing point.** The quantity the claim is
about is the ratio of the two terms' **gradient** norms in the shared parameters. That is not
logged (C6). The quantity that *is* logged is the two terms' **loss values**, and for the policy
term a loss value is not a stand-in for a gradient — it is close to arithmetically guaranteed to
be near zero. PPO's policy surrogate is `mean(ratio × advantage)`; the advantages are z-scored
per batch, so they average zero, and early in an epoch the probability ratio is ≈ 1, so the
*value* of the term is ≈ 0 in expectation **no matter how large its gradient is**. A policy share
of 0.0017% therefore measures approximately nothing about the gradient split. This cuts in both
directions, and the second direction is the one that matters for this project's history: **the
1M analysis's loss-share evidence never supported the mechanism either.** `MC_RAW`'s larger
policy-term *value* is likewise partly just a non-zero-mean raw residual (`return − V` against a
biased critic), not necessarily a larger gradient.

**So the two tests below are reported for what they do establish, which is less than an earlier
draft claimed.**

**Test A — within-run, across the escape. Null, and it rules out the strong form only.** Every `MC_FIXED` and `GAE` seed
improved substantially between 1-2 M and 9-10 M episodes. Over the same interval the loss balance
did not move:

| Run | survival, 1-2 M → 9-10 M | value share of the loss budget, 1-2 M → 9-10 M | policy share | fraction of updates clipped |
|---|---|---|---|---|
| `mcfixed_s42` | 43.9 → 85.1 | 99.9928% → 99.9926% | 0.0021% → 0.0018% | 100% → 100% |
| `mcfixed_s44` | 43.2 → 121.6 | 99.9925% → 99.9929% | 0.0023% → 0.0019% | 100% → 100% |
| `mcfixed_s45` | 44.2 → 119.0 | 99.9922% → 99.9926% | 0.0023% → 0.0019% | 100% → 100% |
| `gae_s42` | 45.1 → 133.6 | 99.9877% → 99.9891% | 0.0040% → 0.0032% | 100% → 100% |
| `gae_s46` | 45.7 → 130.2 | 99.9886% → 99.9885% | 0.0034% → 0.0035% | 100% → 100% |

(All ten slow-arm runs behave this way; five are shown.) Survival roughly **doubled to tripled**
while the value term's share of the loss stayed **unchanged to five significant figures** and the
policy term's share, if anything, **fell**. The gradient was clipped on **100% of updates
throughout**, by a median factor of 146-221x at 1-2 M episodes and 168-217x at 9-10 M — i.e. the
clipping got no gentler as the agents got better.

**What this does establish:** substantial policy learning happened while the loss balance never
moved, so the **strong** form of the mechanism — *value dominance prevents policy learning* — is
inconsistent with the data. **What it does not establish, and where the earlier draft went
wrong:** a *constant, never-relieved handicap that makes learning ~10x slower* predicts exactly
this picture — an unchanging balance alongside slow but real progress. Since a roughly ten-fold
slowdown is this document's own headline result, Test A is arguably **more consistent with a
constant handicap than with none**. It cannot tell the two apart, because both predict the same
observation. Calling it "decisive" was wrong.

**Test B — the single-variable contrast `MC_RAW` was built for. Null, and badly underpowered.**
`MC_RAW` is `MC_FIXED` with one line deleted. It multiplies the policy term's loss magnitude by
~17x and its share of the loss budget by ~20x (0.0017% → 0.0348%), leaving the critic target, the
estimator, and every hyperparameter identical. Survival at matched experience: **69.3 ± 11.7
versus 70.4 ± 39.4**, Welch *p* = 0.95, *d* = −0.04.

Three reasons this null carries much less weight than "bought nothing measurable" implies:

1. **It could not have measured a large benefit.** The two arms' standard deviations at the
   matched 349 M-step budget are 39.4 and 11.7, so the standard error of the difference is
   **~18 survival steps** and the **minimal detectable difference at 80% power is about 50 steps**
   (normal approximation; ~65 once the small-sample degrees of freedom are accounted for). Under
   greedy evaluation it is worse still — sd 58.0 and 33.2, minimal detectable difference ~85-100
   steps, with a bootstrap 95% interval on `MC_RAW` − `MC_FIXED` of **[−61, +43]**. A 40-step
   gain — a *large* effect on a 70-step baseline — would have been missed.
2. **It rests on a mean this document elsewhere says should never be quoted.** §5.1 states that
   `MC_RAW`'s bimodal distribution means "the arm mean describes no individual seed" and "should
   be reported as a split, never as a mean" — and then a mean-based Welch test carries two
   conclusions. Per seed at 349 M steps, `MC_RAW`'s two escaped seeds (107.7, 118.6) are **above
   every `MC_FIXED` seed** (max 81.1) and level with `GAE`, while its three stuck seeds are
   **below every `MC_FIXED` seed**. That is a high-variance split, not an identity.
3. **It never left the regime the mechanism is about, and it is confounded by the alternative
   account this section itself offers.** The value term's share went from 99.99% to **99.96%**.
   If the claim is "the shared body is shaped almost entirely by value regression", it predicts
   no benefit from that change. Worse, the entropy-collapse hypothesis discussed below — which
   this document takes seriously — would *itself* explain why `MC_RAW` did not improve, and if it
   is right then this contrast says nothing about gradients at all. One null result cannot be
   both evidence about gradients and evidence about exploration.

**The one piece of evidence that appears to support the mechanism, and why it should not be
believed either.** Across the five arms, policy share and survival rank-correlate at ρ = +0.90
(*p* = 0.037). But policy share is not an independent variable here — it is a **mechanical
consequence** of the scaling choice, so this correlation cannot separate "policy share matters"
from "scale matters". And `MC_RAW` sits awkwardly on it: 20x the policy share of `MC_FIXED` with
no detectable survival difference (subject to caveat 1 above), and 8x the policy share of `GAE`
with 28 fewer steps. With five non-independent arms this is not evidence either way.

**Verdict: unmeasured, in both directions.** The mechanism is a claim about the ratio of
**gradient** norms in the shared parameters. That ratio has never been logged, here or in the 1M
runs. The loss shares cannot substitute for it (see above), so they neither support the claim
— as the 1M analysis believed they did — nor refute it, as an earlier draft of this section
claimed. What the data *does* rule out is the strongest version, that value dominance blocks
policy learning outright; the version that it merely slows learning is untouched. **The project
should stop asserting this mechanism *as established*** — in either direction — until per-term
gradient norms are logged (§7, first row). That single measurement would turn a "not measured"
into a "true" or a "false".

**A separate hypothesis about what `MC_RAW` changed — labelled a hypothesis, not a result.** The
ratio that moved by two orders of magnitude was not policy-versus-value but
**policy-versus-entropy**. With the entropy coefficient fixed at 0.01, |policy term| / |entropy
term| is 0.26-0.52 in every other arm — the entropy bonus *dominates* the policy objective by a
factor of 2-4 — and **42.25** in `MC_RAW`, a shift of roughly 120x. Alongside it: entropy ends at
0.095 nats (0.013 in the stuck seed) against 0.40-0.67 everywhere else, and three of five seeds
sit in a non-foraging state that takes millions of episodes to leave.

**The status of that account, stated plainly. It is one arm, five seeds, observational, with no
intervention, and the direction of causation is not established.** The alternative direction is
live: a policy that has settled into resting *becomes* deterministic because resting is stable
under large advantages, in which case low entropy is a **consequence** of the trap rather than
its cause. The fine trace argues neither way cleanly — `mcraw_s45`'s entropy was already down at
0.037 nats by 2 M episodes, while it was still a wanderer eating 0.67 food per episode, and the
switch to resting came about a million episodes later. Consistent with the entropy account;
also consistent with a slow drift toward determinism that resting then locked in. **The
experiment that would settle it is already named** — §6.3 row 1, `MC_RAW` re-run with the entropy
coefficient raised ~20x, nothing else changed. Until that runs, "`MC_RAW` failed through entropy
collapse" is the leading hypothesis and not a finding.

A useful corollary for the 1M analysis's Finding 5, which reported that entropy is *not* the
explanation for the `MC` versus `MC_FIXED`/`GAE` gap: that remains true — those three arms all
sit at 0.40-0.67 nats. Whatever is going on in `MC_RAW` is specific to `MC_RAW`, and the two
statements are consistent because its un-normalised advantage is ~20x larger than any of theirs.

**One further observation that complicates the clipping story.** The optimiser is
`optax.chain(clip_by_global_norm(0.5), adam(5e-4))`. A global-norm clip is a *scalar* rescale of
the whole gradient, and Adam's per-parameter normalisation absorbs most of a scalar rescale, so
"the raw-target arms are clipped 150x on every update" is a weaker argument than it looks —
clipping changes the gradient's magnitude, not its direction or its composition. These arms were
clipped on 100% of updates for 10 M episodes — by a median factor that did not fall over training —
and still went from 40 to 130 survival steps.

### 5.7 The completed 2x2 — the estimator matters after all, but only under the split convention

The 1M analysis's clearest structural claim was that **scale predicts the outcome and the
estimator does not**: `MC_FIXED` (Monte-Carlo, split) and `GAE` (GAE, split) differed by 1.7
survival steps, while `MC` and `MC_FIXED` — identical returns, different scaling — differed by 98.
`GAE_NORM` completes the 2x2, and at 10M episodes the picture is no longer that simple.

Survival at the matched 349 M-environment-step budget, mean ± sd over 5 seeds:

| | Monte-Carlo estimator | GAE(λ=0.95) estimator | estimator effect |
|---|---|---|---|
| **matched scale, ≈1** | `MC` 156.8 ± 4.1 | `GAE_NORM` 158.9 ± 1.1 | **+2.0** (*p* = 0.34) |
| **split scale** | `MC_FIXED` 69.3 ± 11.7 | `GAE` 98.3 ± 14.0 | **+29.0** (*p* = 0.008) |
| **scale effect** | **+87.6** | **+60.6** | |
| **matched scale, ≈24** | `MC_RAW` 70.4 ± 39.4 | *cell not run* | — |

**The statistic to quote is the split-scale row's own contrast: Welch *p* = 0.008.** A two-way
analysis of variance over the four complete cells (20 runs) reports a large main effect of
**scale** (*F* = 311, *p* = 7×10⁻¹²), a smaller main effect of **estimator** (*F* = 13.6,
*p* = 0.002) and an interaction (*F* = 10.3, *p* = 0.005) — but **those *p*-values should not be
leaned on**: the four cells' standard deviations range from 1.1 to 14.0, a **variance ratio above
160x**, and an ANOVA at five runs per cell is not robust to that. The interaction is also
*p* = 0.044 on the episode axis, i.e. marginal. What survives without a homoscedasticity
assumption is the row-by-row picture: within the split-scale row `GAE` beats `MC_FIXED` (Welch
*p* = 0.008); within the matched-scale row `GAE_NORM` and `MC` are not separated at this budget
(*p* = 0.34).

**The +29 steps is a snapshot on the steep part of two rising curves — express it as a dilation
ratio instead.** The `GAE` − `MC_FIXED` gap depends heavily on where it is read: **−0.5 steps** at
100 M environment steps, **+3.5** at 150 M, **+9.7** at 200 M, **+16.3** at 250 M, **+25.9** at
300 M, **+29.0** at 349 M, and +32 at 10 M
episodes. Both arms are still climbing, so a step count at one budget is a statement about *when
you looked*. The scale-free version is stable: **`GAE` reaches each survival level about 1.1x to
1.4x sooner than `MC_FIXED`** — 56 M vs 61 M environment steps to reach survival 45, 111 M vs
125 M to reach 50, 185 M vs 239 M to reach 60, 193 M vs 278 M to reach 65 (medians over all five
seeds of each arm at every level where all five reached it). Beyond level 65 the `MC_FIXED`
median is computed on the seeds that got there, so it understates the gap.

**The effect is real and not one-seed-driven.** At 349 M steps the per-seed ranges do not overlap
(`GAE` 81.6-113.4 against `MC_FIXED` 55.0-81.1), and dropping any single seed from either arm
still leaves a gap of at least +21 steps.

**What that means in words.** Scale is still by far the dominant variable, and the 1M
conclusion — that adopting the textbook advantage convention without a matching change to the
critic target is what costs performance — stands. But the 1M claim that "the estimator does not
matter" was a statement made from inside the split-scale row only, at a budget where both cells
of that row were still stuck at the do-nothing ceiling of ~41 steps. Given ten times the budget,
the GAE estimator buys a **1.1-1.4x speed-up under the split convention** and **nothing
detectable under the matched convention**. The estimator's contribution is real, it is
conditional on the scaling, and it was invisible at 1M because neither split-scale cell had
started learning yet.

### 5.8 Scoring the two pre-registrations, and the 1M analysis's own predictions

**`GAE_NORM` pre-registration** ([[gae_norm_prereg]]) — written before the arm ran.

| Prediction | Outcome |
|---|---|
| "If `GAE_NORM` lands near 139 (say above ~100) → **scale-matching is the mechanism**." | **Correct.** 170.4 at 10M episodes, 158.9 at matched experience. Well above the threshold. |
| Gradient norm falls below the 0.5 clipping ceiling | **Correct.** Median 0.17-0.39, clipped on 0% of updates. |
| Policy term's share rises from ~0.002% toward ~2% | **Correct in direction, slightly optimistic in size.** Observed 1.0-1.4%. |
| Critic's explained variance rises from ~9% toward ~53% | **Correct, and then some.** 58.3% ± 0.24, above `MC`'s 53.3%. |

The confound recorded in advance — that `GAE_NORM` mirrors `MC` faithfully and therefore
*inherits `MC`'s units quirk* (a bootstrap value in z-scored units mixed with raw rewards),
leaving "matched scale" and "carries `MC`'s quirk" confounded — **stands, and is worse than the
pre-registration assumed.** Reading the code: in `MC` the mismatch enters only at rollout-window
edges, whereas `GAE_NORM`'s recursion applies `δ = r + γV' − V` at **every step**, with `r` raw
and `V` on the z-scored scale. So the best-performing arm in this experiment carries the *most
severe* version of the units bug that started the whole investigation. A plausible consequence,
offered as a hypothesis: because `V` is small relative to the raw rewards, the bootstrap term is
nearly inert and `GAE_NORM`'s target is effectively a **z-scored ~10-step discounted reward sum**
(effective horizon 1/(1 − γλ) ≈ 10.3) rather than a bootstrapped λ-return. That would be a
lower-variance target than `MC`'s full return, which is consistent with the observed
explained-variance gap (§4.8) — but it is untested and would need `returns/std` logging to check.

**`MC_RAW` pre-registration** ([[mc_raw_prereg]]) — written before the arm ran, decision metric
explicitly "**at matched environment steps**".

| Prediction | Outcome |
|---|---|
| "Above ~100 ⟹ the **relational** reading is right (sharing units is what matters)" | Not met at the arm level. |
| "Below ~60 ⟹ the relational reading is **wrong**; what matters is *absolute* conditioning near spread 1" | Not met at the arm level either. |
| "Between ~60 and ~100 ⟹ both contribute; this design cannot rank them" | **This is where the arm mean landed (70.4).** |
| **"My own prediction, recorded so it can be wrong: closer to `MC_FIXED` than to `MC`."** | **Correct.** 70.4 versus `MC_FIXED` 69.3 and `MC` 156.8 — a distance of 1.1 versus 86.4. |

**The pre-registered outcome, scored first and on its own terms: "cannot rank".** The rule
written before the runs was a three-way banding on arm-level survival at matched environment
steps — above ~100 ⟹ relational reading right; below ~60 ⟹ relational reading wrong; between
~60 and ~100 ⟹ both contribute, this design cannot rank them. `MC_RAW` landed at **70.4**, inside
the middle band. **By the pre-registered rule, this experiment does not rank the relational
against the absolute reading.** That is the outcome of record, and everything below it is
post hoc.

**Why the banding could not do its job: `MC_RAW` is bimodal.** Two seeds landed at 108-119 (the
"relational reading is right" band) and three at 35-45 (the "relational reading is wrong" band).
Nothing landed in the middle. The mean of 70.4 describes no seed. The pre-registered banding was
built for a unimodal result and cannot classify this one — a limitation of the rule, not a result
of the experiment.

**Post-hoc reasoning, labelled as such.** The strong form of the relational reading — *what
matters is that target and advantage share units, whatever the magnitude, so `MC_RAW` should
behave like `MC`* — **is refuted**, and refuted on a fact that needs no test: **every one of
`MC_RAW`'s five seeds finished below `MC`'s worst seed, at every budget checked** (at 349 M
steps, `MC_RAW` max 118.6 against `MC` min 151.6). Correspondingly the absolute reading (H₄) —
that the advantage needs to land near spread 1 — is supported: the field's convention of pinning
advantages to spread 1 is solving a problem this environment does have.

**What is *not* shown is the weaker relational claim.** "Matching the units at the large scale
bought nothing *relative to the split convention*" rests on the `MC_RAW`-versus-`MC_FIXED`
comparison, whose *p* = 0.95 is the underpowered, mean-based null dissected in §5.6 (minimal
detectable difference ~50 survival steps; the arm mean describes no seed; two `MC_RAW` seeds are
above every `MC_FIXED` seed and three below every `MC_FIXED` seed). **Whether matching units at
scale 24 helps a little, not at all, or hurts is not resolved by these five seeds.**

**And the prediction was right for the wrong reason.** The pre-registration reasoned that
`MC_RAW` would improve on `MC_FIXED` but fall short of `MC` because "the network is still shaped
overwhelmingly by value regression". Its arm mean did not improve on `MC_FIXED`; the loss-share
numbers quoted in its reasoning were off by ~170x in absolute terms (predicted ~0.3% → ~6%;
observed 0.0017% → 0.0348%) even though the ~20x multiplier was right; and the loss shares turn
out not to bear on the shared-trunk mechanism in the first place (§5.6). The *second* thing the
arm was designed to test — that mechanism — therefore came out **untestable**, not negative.

**The 1M analysis's own forward predictions** ([[return_mode_cmp_1M]] §6.4):

| Prediction | Outcome |
|---|---|
| "The ordering will hold" | **Correct.** `MC` > `GAE` > `MC_FIXED` at 10M episodes, as at 1M. |
| "The size of the gap is genuinely uncertain and could shrink substantially" | **Correct.** 3.3x → 1.75x on the episode axis. |
| "**Early evidence for the stall reading rather than the dilation reading**" | **Wrong at the budget it was made about.** It was dilation, by a factor on the order of 10x. The evidence available at the time — that `MC_FIXED` was "past 5x the episode count at which `MC` had already taken off, without any sign of a takeoff" — was a real observation but the inference from it was not safe: `MC_FIXED`'s takeoff began about 5x later still. Note that the deeper question the stall reading was about — *is there a lower ceiling?* — is **not** settled by this experiment either (§5.4), and the upward drift of the dilation factor is weak evidence on the stall reading's side. |
| "`MC` will again have consumed ~2.5x the environment steps" | **Correct in kind, larger in size** — up to 3.3x. |

---

## 6. Conclusions

### 6.1 Summary

- **The 1M analysis's central finding survives in direction and shrinks in size.** Keeping the
  critic target and the advantage on the same scale near spread 1 is worth an
  **order-of-magnitude reduction in the experience needed to reach any given survival level**
  (measured range **6x to 14x**), and a **2.3x survival advantage at matched experience** at the
  largest budget all 25 runs share. At 1M episodes the same comparison read 3.3x with the losers
  apparently not learning at all. The dilation factor is **not flat**: it drifts upward with
  survival level in two of the three slow arms (6.4x → 11.6x and 5.9x → 14.1x) and downward in
  the third, on two-to-three seeds per point. Whether that is the onset of a genuinely lower
  ceiling or seed noise is **undetermined**.
- **"They never learn to forage" was a statement about the budget, not about the setting.**
  Given ten times the episodes, every split-scale run escapes the rest-and-starve local optimum,
  starts eating, starts reaching the 500-step cap, and turns its starvation rate around — the
  four things the 1M analysis nominated in advance as the signature of escape. The
  **"stall" reading is refuted**; the **"delay" reading is what happened**.
- **Whether the split-scale settings eventually catch up is untested and not close to tested.**
  They would need **at least** 9-12 billion environment steps — a lower bound computed by holding
  the dilation factor constant, which is the most optimistic assumption available given that the
  factor is drifting upward. They were given 0.6-0.8 billion. Three of the five arms were still
  climbing when the budget ended.
- **`GAE_NORM` is the best setting, on both measurements.** +5 survival steps over `MC` with
  complete seed separation during training, and **+4.5 steps (95% CI [+0.8, +8.2]) under greedy
  evaluation** of the same checkpoints at 2,000 episodes per seed (§4.9). An earlier 200-episode
  evaluation read a tie, but its margin of error (±6.4 steps) was larger than the effect; the
  two measurements never disagreed. The "`GAE_NORM` just explores less" reconciliation is
  therefore **wrong**: the greedy evaluation takes no exploratory actions in either arm and the
  gap is still there. *Why* `GAE_NORM` is better remains unexplained.
- **The completed 2x2 revises the 1M claim that "the estimator does not matter".** Scale remains
  the dominant variable by a wide margin, but under the split convention the GAE estimator
  reaches every survival level **1.1x to 1.4x sooner** than the Monte-Carlo one (Welch *p* = 0.008
  on the split-scale row at the largest shared budget; non-overlapping per-seed ranges), while
  under the matched convention the two are **not separated** (*p* = 0.34). At 1M this was
  invisible because both split-scale cells were still stuck at the do-nothing ceiling. The
  two-way ANOVA's *p*-values are quoted in §5.7 but should not be leaned on — the four cells'
  variances differ by more than 160x.
- **`MC_RAW` refutes the strong "units-matching is all that matters" reading; the weaker version
  is unresolved.** Every one of its five seeds finished below `MC`'s worst seed at every budget,
  so matching target and advantage at scale ~24 does **not** reproduce matching them at scale 1
  (2.26x worse at the largest shared budget). Whether it helps *at all* relative to the split
  convention is **not resolved**: that comparison's *p* = 0.95 could not have detected a 40-step
  gain, and the arm is bimodal. By its own pre-registered decision rule the arm landed in the
  "cannot rank" band.
- **`MC_RAW`'s failure route is *hypothesised* to be entropy collapse.** Its un-normalised
  advantage overwhelms a fixed entropy coefficient by ~120x relative to every other arm; policy
  entropy falls to 0.095 nats (0.14% non-greedy actions in the worst seed), and three of five
  seeds sit in a zero-food state whose exact signature is a longest-episode of 100.0 steps. This
  is **one arm, five seeds, observational, with no intervention**, and reverse causation — a
  policy that has settled into resting becomes deterministic *because* resting is stable — is not
  excluded. §6.3 row 1 is the test. One seed moved *into* the trap at 3-4 M episodes, having been
  a low-entropy wanderer eating incidentally at the do-nothing survival level (not, as an earlier
  draft said, a working forager).
- **The shared-trunk value-dominance mechanism is *unmeasured*, in both directions.** The claim is
  about the ratio of the two terms' **gradient** norms in the shared body; only their **loss
  values** are logged, and the policy loss term is ≈ 0 by construction (mean-zero advantages)
  regardless of its gradient. So the loss shares neither support the mechanism — as the 1M
  analysis believed — nor refute it. What the data *does* rule out is the strong form, that value
  dominance blocks policy learning outright: slow runs doubled and tripled their survival while
  the balance never moved. A constant handicap producing a ~10x slowdown predicts exactly that
  observation, so it is not distinguished. **The project should stop asserting this mechanism *as
  established*, in either direction**, until per-term gradient norms are logged (§7).
- **No logged optimisation signal was found to predict which seed escapes a trap — at a sample
  size that could only have found a strong one.** Three tests — within-arm (n = 4, where the
  smallest attainable *p* is 0.083), out-of-sample, and pooled across fifteen seeds (~50% power
  for a moderate correlation) — all null. The only correlates are behavioural and
  near-tautological, and even those do not survive multiplicity correction. Read this as **"not
  detected at n = 4-15"**, not as "no signal exists".

### 6.2 What this does and does not license

**It does license**: keeping a matched-at-spread-1 return mode as the project default for
recurrent PPO on this environment; treating past results obtained under `MC` as not invalidated
by the units-mismatch bug report; and treating "switch to the textbook convention" as a change
that costs roughly an order of magnitude in sample efficiency at these hyperparameters.

**It does not license**: (a) the claim that the textbook convention is *broken* — it is slow, and
at 10M episodes it is producing competent foraging agents; (b) any claim about the settings'
**ceilings**, which this budget cannot reach — and note that the upward drift of the dilation
factor is the one piece of evidence that a ceiling difference might be real, so "merely slower"
must not be stated as settled either; (c) any *explanation* of why `GAE_NORM` beats `MC` — the
+4.5-step effect is established on two independent measurements, but its cause is not; (d) the
shared-trunk value-dominance mechanism **in either direction** — it is not established and it is
not refuted, because the gradient quantity it concerns has never been logged; (e) "`MC_RAW`
failed through entropy collapse" as a finding — it is a hypothesis with one arm behind it and a
named test outstanding; (f) "matching units at scale 24 buys nothing relative to the split
convention" — that null could not have detected a 40-step gain; (g) generalisation beyond one
environment, one network, one learning rate, and one untuned set of coefficients (C4).

**Honest caveats a sceptical reader deserves.** `MC`'s median evaluation episode (52 steps at
2,000 episodes) is barely longer than `MC_FIXED`'s (31) — as at 1M, almost the entire advantage
lives in the upper tail of survivable episodes, so every ratio here is a statement about means.
`MC_RAW`'s arm-level numbers describe no individual seed and should always be reported split —
including in the *p* = 0.95 null that two conclusions were previously rested on. The
`GAE_NORM`-vs-`MC` **variance** comparison (as opposed to the mean comparison, which holds) is
significant on one test and not on another and disappears entirely under greedy evaluation. The
absolute survival levels in §4.7 are ~7-12 steps too high because all 25 runs were evaluated on
the same 200 easy worlds (§4.9); the between-arm comparisons are unaffected. And three of the
five arms were never pre-registered.

### 6.3 Recommended next experiments

| Priority | Experiment | What it settles | Effort |
|---|---|---|---|
| **1** | **`MC_RAW` with the entropy coefficient scaled up ~20x** (0.01 → 0.2), restoring the policy-to-entropy loss ratio to the ~0.3 the other four arms have. Nothing else changed. **Run ten seeds, not five** — this doubles as the power fix in row 5. | Turns §5.6's entropy-collapse **hypothesis** into a result or kills it. It is the only intervention available that breaks the reverse-causation ambiguity (does low entropy cause the trap, or does the trap cause low entropy?). If `MC_RAW` recovers to `MC_FIXED` levels or better, the "advantage must be near spread 1" conclusion needs restating as "the advantage must be near spread 1 *given a fixed entropy coefficient*". Highest-value follow-up; a one-key change. | 10 seeds × 10M |
| **2** | **Log per-term gradient norms and re-run one slow arm and one fast arm.** The specific quantity: the global norms of the policy term's and the value term's gradients **restricted to the shared parameters** (encoder + GRU), logged separately, per update. | **The only measurement that can settle the shared-trunk mechanism**, which §5.6 records as untested in both directions. Everything the project has said about it — for and against — has been argued from loss values that are ≈ 0 by construction for the policy term. Promoted from row 4 because it is the outstanding question, not a nice-to-have. | code change (§7 row 1) + 2 arms × 3 seeds |
| **3** | **`MC_FIXED` and `GAE` with an upstream return normaliser** (running mean/std over returns, or PopArt on the value head) — what the nine surveyed libraries actually do. | If it removes the 10x dilation, the finding is "this project is missing a standard component", which is a materially different and more publishable claim than "the convention is worse". | needs a small code change |
| **4** | **Continue `MC_FIXED` and `GAE` from their 10M checkpoints** for another 30-50M episodes. | Promoted, because §5.4 now reports the dilation factor **drifting upward** (`GAE` 6.4x → 11.6x) rather than flat, and that drift is the early signature of a lower ceiling. This is the cheapest measurement that extends the factor past survival 130 and says whether the drift continues. Still far short of the ~10 billion steps a full ceiling answer needs. | resume from checkpoint |
| 5 | **`MC_RAW` seeds 47-56** (ten more seeds), to characterise the escape-time distribution properly and give the precursor analysis in §5.3(d) the power it lacks at *n* = 4. Subsumed by row 1 if that is run at ten seeds with a matched-coefficient control. | §5.3(d)'s negative result is "no signal detected at *n* = 4-15" — a sample size at which no within-arm test could have reached significance at all. It also re-powers the `MC_RAW`-vs-`MC_FIXED` contrast, whose current minimal detectable difference is ~50 survival steps. | 10 seeds × 10M |
| 6 | **`GAE_NORM` versus `MC` — what causes the +4.5 steps?** Log `returns/std` and `targets/std` (§7 row 2) and check whether `GAE_NORM`'s λ-return is behaving as the lower-variance ~10-step target §5.8 hypothesises. | The effect is now established on two independent measurements (§4.9) and unexplained. The exploration-cost explanation is dead. This is the cheapest first probe. | logging change, then re-read |

---

## 7. Metrics requested

This analysis is read-only on `src/`. Each item below would sharpen a specific claim above; the
first two would change what this document is able to conclude. If accepted, route through
`feature-workflow` (`senior-developer` plans, `developer` implements) — not directly.

| Metric | Why now | Where it'd live | Cost |
|---|---|---|---|
| **`grad_norm/policy_component` and `grad_norm/value_component`**, ideally **restricted to the shared parameters** (encoder + GRU) — the global norms of the two loss terms' gradients, logged separately | **This is the top request, and it is a gap in *kind*, not in precision.** The shared-trunk mechanism is a claim about the ratio of these two gradient norms; §5.6 has to record it as **untested in both directions** because the quantity has never been logged, here or in the 1M runs. The substitute the project has used — the two terms' shares of the summed *loss* — is not a proxy: PPO's policy surrogate is `mean(ratio × advantage)` with z-scored advantages, so its **value** is ≈ 0 in expectation however large its **gradient** is. Every claim the project has made about this mechanism, for and against, rests on that non-proxy. This measurement is the one that turns "not measured" into "true" or "false". | `update_step` in `src/models/recurrent_ppo_trainer.py`, via two extra `jax.grad` calls or a split of the existing one | moderate (one extra backward pass, or a restructure) |
| **`returns/mean`, `returns/std`, `targets/std`, `advantages/std_preclip`** | Three separate claims currently rest on an externally supplied "return spread ≈ 24" that was measured once, at 1M episodes, when episodes were 40 steps long. Episodes are now 165 steps long, so that number is certainly stale and possibly wrong by a large factor. Without it, **explained variance is not computable for the three raw-target arms** (§4.8 has to leave those cells blank), the "advantage ≈ 20" characterisation of `MC_RAW` is unverified, and §5.7's hypothesis about `GAE_NORM`'s effective horizon cannot be checked. | `train_iteration`, right after `targets` and `advantages` are computed, in all five branches | cheap (4 scalars per iteration) |
| **`value/explained_variance`** — 1 − Var(target − prediction)/Var(target) | The correct unit-free critic-quality measure, directly comparable across arms with different target scales. Currently derivable only for the two z-scored arms, and only by assuming the target's spread is exactly 1. | same place as `loss/value` | cheap |
| **`policy/approx_kl`, `policy/clip_fraction`, `policy/entropy_per_step`** | Standard PPO health metrics; none is logged. §5.3's entropy-collapse account would be much stronger with the PPO ratio-clip fraction alongside it — currently there is no way to tell whether `MC_RAW`'s enormous advantages are being absorbed by PPO's own ratio clip (ε = 0.1) or are passing through. | `update_step` | cheap |
| **An evaluation survival scalar** — e.g. `eval/survival_steps`, `eval/food_eaten`, logged at each checkpoint, **over at least ~1,000 episodes** | **The `eval/` namespace currently carries a video and nothing else** (§3), so there is no independent survival number anywhere in WandB and §4.7 had to re-run 25 offline evaluations to get one. Every future analysis of every run will need this. Two design notes learned the hard way here: log *both* the greedy and the stochastic number (C8); and **do not use a small fixed world set** — §4.9 shows that 200 shared worlds carry a ±13-step sampling offset, enough to fabricate an apparent "+8.2-step greedy bias" and to hide a real 5-step between-arm effect. | `src/utils/wandb_utils.py` plus the checkpoint-eval path in `train.py` | moderate (one extra 1,000-episode batched rollout per checkpoint; ~30 s on CPU at this environment size) |
| **`Episode/Steps_Max` is already logged and turns out to be a free failure detector** — no change requested, but worth documenting | An agent that never eats cannot survive past step 100 in this environment (starting nutrition is uniform on [0, 100], 1 per step). `Episode/Steps_Max == 100.0` is therefore an exact, zero-cost detector of a zero-food policy, and it fired on `mcraw_s45` for seven million episodes. Worth adding to whatever run-health dashboard exists. | — | none |

---

## 8. Related issues

- [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] — the bug report that motivated `MC_FIXED`. **This
  analysis does not overturn the bug**: mixing a z-scored return with an un-rescaled bootstrap
  value is still a units mismatch. What it adds to the 1M finding is that the fix as implemented
  costs roughly **a factor of ten in sample efficiency**, not the total failure it appeared to
  cost at 1M — and, more awkwardly, that the **best-performing arm in the experiment carries a
  more severe version of the same mismatch than `MC` does** (§5.7), because `GAE_NORM` applies it
  at every step rather than only at rollout-window edges. The issue doc should record that the
  fix is not to be adopted as a default, and that the units mismatch is now positively correlated
  with performance across the arms that have it — which is not a defence of the mismatch, but is
  a reason to stop treating it as an explanation for anything.
- [[ppo_implementation_details_lit_review]] — established that no paper in the corpus measures
  the shared-trunk value-dominance mechanism. **§5.6 adds a methodological result to that
  bibliographic one**: the measurement this project *has* been using in place of the missing one
  — the two loss terms' shares of the summed loss — cannot serve as a proxy, because the policy
  surrogate's value is ≈ 0 by construction for mean-zero advantages. The conjecture is therefore
  unmeasured here as well as unmeasured in the literature. The lit review's entry for it should
  be cross-linked here.
- **No code bug was found by this analysis.** The five `return_mode` branches do what their
  docstrings say; the 559-key config diff across 25 runs is clean; every run has an exit record.
  One provenance correction: the 25 runs do **not** all share a git commit (C7) — 20 are at
  `788e5983` and the five `MC_RAW` runs at a descendant, `a171ea35`, whose diff touches nothing
  under `src/` or `configs/`.
- A **plan-reviewer** pass has been run on this document's verdicts
  ([[plan_return_mode_cmp_10M]], 2026-09-05) and its three Critical and seven Moderate findings
  are applied throughout; see Appendix E for the point-by-point response and Appendix C for the
  changelog.

---

## Appendix

### A. Extraction provenance

All numbers were derived from the **local** WandB datastores; no web-API call was made. Working
files (gitignored):

| File | Contents |
|---|---|
| `tmp/20260905_return_mode_cmp_10M.md` | Full extraction log — every table above at full precision, plus per-seed temporal traces |
| `tmp/cmp10m/hist/*.pkl` | Parsed history frames for all 25 runs (cached; 2,900-4,400 rows each) |
| `tmp/parse_wandb_local.py` | Local `.wandb` datastore reader (protobuf record scanner) |
| `tmp/cmp10m/cfgdiff.py` | 559-key config diff across the 25 trainer-written configs |
| `tmp/cmp10m/agg_curves2.py`, `agg_perseed.py`, `wsens.py` | Survival curves, per-seed statistics, window-sensitivity |
| `tmp/cmp10m/escape.py`, `dilation.py`, `dilation2.py`, `postescape.py` | Escape times, dilation factors, escape-aligned curves |
| `tmp/cmp10m/mcraw_fine.py`, `precursor2.py`, `pooled_precursor.py` | `MC_RAW` per-seed traces and the three precursor tests |
| `tmp/cmp10m/mech.py`, `grad.py`, `shares.py` | Loss-budget shares, gradient norms, the two mechanism tests |
| `tmp/cmp10m/behav.py`, `run_evals.sh`, `eval_agg.py` | Behavioural metrics; the 25 offline evaluation rollouts (200 episodes each) and their aggregation |
| `tmp/cmp10m/eval/<run>/` | 200 per-episode archives per run from `scripts/eval/eval_rollout.py` |
| `tmp/cmp10m/run_evals_2k.sh`, `eval2k_agg.py` | §4.9: the 2,000-episode re-evaluation of the `MC` and `GAE_NORM` arms, and its aggregation |
| `tmp/cmp10m/eval2k/<run>/` | 2,000 per-episode archives per run (10 runs, 20,000 episodes) |
| `tmp/20260905_return_mode_cmp_10M_eval2k.md` | §4.9's full output, the first-200-worlds superset check, and the git/config provenance checks in Appendix B |

### B. Config diff, and the provenance checks

**Git commit — corrected.** Read from each run's `wandb/run-*/files/wandb-metadata.json`, the 25
runs carry **two** commits, not one: `788e598303d927a7c598587f4a452c7e0b294fd8` for the 20 runs of
`MC`, `MC_FIXED`, `GAE` and `GAE_NORM`, and `a171ea35105d361b609182d0f58ce257b4835aed` for the
five `MC_RAW` runs launched ~4 hours later. `a171ea35` is a **descendant** of `788e5983`, and the
seven commits between them touch only `docs/`, `scripts/analysis/`, `scripts/claude/` and
`tests/analysis/` — **no file under `src/` or `configs/`**, i.e. nothing on the training path.
C7's earlier claim of a single shared commit was wrong; the conclusion it supported (that the
`MC_RAW` launch delay is a scheduling difference only) is unchanged.

**Evaluation environment config — checked.** The offline evaluations loaded the source YAML
`configs/environment/experiment/basic/04-jump_attack_10x10.yaml` rather than each run's saved
copy (C9). Resolving that source through the evaluation loader and flattening it gives **133
environment keys, all 133 identical to a run's trainer-written `models/config.yaml`** (which
carries 118 further agent/training-side keys). No YAML under `configs/environment/` has been
modified on or after 2026-09-04; the `extends:` chain's modification times are 2026-07-04 (`04`),
2026-07-22 (`03-random_init_10x10`) and 2026-08-26 (`default.yaml`), all predating the runs.

**Agent config diff.**

All 25 trainer-written configs (`results/JAX_RecurrentPPO/20260904-*_rppo_cmp10m_*/models/config.yaml`)
flattened to **559 keys**. Differing keys, complete: `agent.return_mode` (the five settings),
`seed` (42-46), `tag`, `wandb.name`. The last two are labels. No other key differs — including
`gamma`, `gae_lambda`, `eps_clip`, `entropy_coef`, `vf_coef`, `max_grad_norm`, `lr_actor`,
`lr_critic`, `K_epochs`, `sequence_length`, `hidden_size`, `rnn_type`, `encoding_mode`,
`modulation.type`, and every environment key.

(`lr_critic` = 1×10⁻⁴ appears in every config but is **not read** by the recurrent-PPO path,
which builds a single Adam optimiser on `lr_actor` = 5×10⁻⁴ over all parameters. Noted so that a
future reader does not assume separate actor/critic learning rates were in play.)

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-09-05 | Initial analysis of the 10M-episode set (25 runs, 5 seeds × 5 arms), including 25 offline evaluation rollouts of 200 episodes each | `experiment-analyzer` |
| 2026-09-05 | **Revision after [[plan_return_mode_cmp_10M]].** Three headline claims corrected (mechanism, `GAE_NORM` greedy tie, constant dilation factor) and seven moderate findings applied; new §4.9 reports a 2,000-episode-per-seed re-evaluation of the `MC` and `GAE_NORM` arms (evaluation only, no retraining); provenance of the git commits and the evaluation environment config checked rather than assumed. Point-by-point in Appendix E. | `experiment-analyzer` |

### D. Git status note

Nothing in this analysis was committed. `.git/index.lock` has been held by another session since
2026-09-04 17:49 and several agents have already failed on it, so **no write-side git operation
was attempted**. (The commit provenance in Appendix B was read with lock-free read-only commands
— `git log`, `git merge-base`, `git diff --stat` between two named commits — none of which touch
the index.) This file and the working files under `tmp/` are on disk and unstaged. Note also
that `docs/diary/2026-09-05.md` carries other sessions' edits.

### E. Response to the plan-reviewer pass

Point-by-point against [[plan_return_mode_cmp_10M]] (2026-09-05). The reviewer's own summary is
appended below this appendix and is left unedited.

| Finding | Disposition | Where |
|---|---|---|
| **C1** — the mechanism refutation does not hold; the entropy replacement is weaker than what it replaced | **Accepted in full.** §5.6 retitled "unmeasured, and the loss-share evidence is uninformative either way". The claim is now quoted as its author stated it (gradient direction, not summed loss); the reason a loss value cannot proxy a gradient here is spelled out; the 1M analysis's loss-share evidence is stated as never having supported the mechanism either. Test A is downgraded from "decisive" to "rules out the strong form only" with the constant-handicap alternative named. Test B carries its ~50-step minimal detectable difference, its dependence on a mean the document elsewhere forbids, and the 99.99% → 99.96% point. Entropy collapse is labelled a **hypothesis** with reverse causation named. Propagated to §1 item 2, the H₅ verdict, §5.3(c), §5.5, §6.1, §6.2(d)-(e), C6, and the Outcome sections of both pre-registrations and the 1M document's header. | §5.6, §1, §5.3(c), §5.5, §6.1, §6.2, C6 |
| **C2** — "disappears entirely under greedy evaluation" is unsupported | **Accepted, and settled with new data.** The wording is withdrawn. The evaluation was re-run at **2,000 episodes per seed** for the `MC` and `GAE_NORM` arms (evaluation only): the gap is **+4.47 steps, 95% CI [+0.75, +8.19]**, matching the training-time effect. The five-seed training-time result stands and is not withdrawn. The re-run also showed the "+8.2-step greedy bias" of §4.7 to be a world-sampling artefact (M6). | new §4.9, §5.2, §4.7 item 2, §1 item 3, §6.1, §6.3 |
| **C3** — "constant ~10x, no lower ceiling" is contradicted by the working table | **Accepted in full.** The level-50 row is restored (and levels 70, 90, 110 with it — the table is now complete). The claim is now "6-14x, drifting upward in two arms of three, ceiling undetermined at n = 2-3"; the "a diverging one does not fit" sentence is deleted; the 9-12-billion-step figure is labelled a **lower bound**. Follow-up row 4 (continue the slow arms from checkpoint) was promoted on the strength of this. | §5.4, §1 item 1, §6.1, §6.2(b), §6.3 |
| **M1** — pre-registered banding scored post hoc | Accepted. §5.8 now records the pre-registered outcome (**"cannot rank"**) first; the post-hoc reasoning follows, labelled. H₃-strong is refuted on the all-seeds-below-`MC` fact, not on *p* = 0.95. | §5.8, §6.1 |
| **M2** — `s45` was not "a working foraging policy" | Accepted. Described as a low-entropy wanderer eating incidentally at the do-nothing survival level; "catastrophic forgetting" removed. | §5.3, §5.3(a), §6.1 |
| **M3** — both escape counts | Accepted. 2 of 5 by this document's own criterion; 4 of 5 left the zero-food state; the completion of the ramp is labelled an extrapolation. | §5.3 opening, §6.1 |
| **M4** — the §5.3(d) null needs its power stated | Accepted. "Not detected at n = 4-15", with the *p* ≥ 0.083 floor at n = 4 and ~50% power at n = 15 stated inline; the `s45` window caveat added. | §5.3(d), §6.1 |
| **M5** — the 2x2's +29 steps and its ANOVA | Accepted. The split-row Welch *p* = 0.008 is the quoted statistic; the ANOVA *p*-values are retained but flagged (variance ratio > 160x, and the episode-axis interaction is only *p* = 0.044); the estimator effect is expressed as a **1.1-1.4x dilation ratio**, with the +29 shown as one point on a trajectory (−0.5 at 100 M steps → +9.7 at 200 M → +25.9 at 300 M → +29.0 at 349 M). The reviewer's finding that the effect is not one-seed-driven is confirmed independently and recorded (non-overlapping per-seed ranges; ≥ +21.6 after dropping any one seed from either arm). | §5.7, §6.1 |
| **M6** — the "+8.2-step greedy bias" treats 25 non-independent evaluations as independent | Accepted, and **confirmed empirically**: the *p*-value is dropped, the proxy is stated as validated for arm-level ranking only, and §4.9 shows the bias falls to +0.36 / −0.32 steps against 2,000 worlds. | §4.7 item 1, §4.9 |
| **M7** — propagate to the 1M header and both pre-registrations | Accepted; all three edited. | external files |
| 🟢 entropy 0.095 vs 0.086 | Fixed; the two windows are now labelled. | §4.6 |
| 🟢 three escape-time definitions | Fixed; one definition (50 steps, hold 10 points) is named and used throughout. | §4.5 |
| 🟢 `gae_norm_prereg` names an "evaluation window" that did not exist | Noted in that file's Outcome section. | external file |
| ❓ same git commit in all 25 runs | **Checked — the claim was false.** 20 runs at `788e5983`, five `MC_RAW` runs at `a171ea35`; the diff between them touches no `src/` or `configs/` file. | C7, Appendix B |
| ❓ evaluation used the source env YAML | **Checked and stated.** All 133 environment keys the eval loader resolves match a run's saved config exactly; no config in the `extends:` chain modified since the runs. | C9, Appendix B |
| ❓ tooling-default eval seeds | Stated explicitly as a tooling default rather than a chosen list. | C9 |
| ❓ "return spread ≈ 24" is stale | Unchanged — still flagged as stale in §7; only the return-scale logging request can fix it. | §7 |
| ❓ `MC_RAW` ran on mixed GPU classes | Not checked. No systematic effect is expected (the computation is deterministic given seeds and configuration), but this remains unverified. | — |

**Two things the reviewer flagged that this revision does *not* fully resolve**, stated so they
are not mistaken for closed: the **shared-trunk mechanism** stays unmeasured until per-term
gradient norms are logged (§7 row 1, §6.3 row 2), and the **ceiling question** stays undetermined
until the slow arms are run past survival level 130 (§6.3 row 4).

---

## Feedback from plan-reviewer

*Appended 2026-09-05. Full review with numbers and exit conditions: [[plan_return_mode_cmp_10M]]
(`docs/reviews/plan_return_mode_cmp_10M.md`). Verdict: **supported with caveats** for the core
result (delay not stall; ~10x experience penalty; scale dominates the estimator); three headline
sentences are **not supported by the evidence shown** as written.*

- **§5.6 (mechanism).** Not a refutation — the mechanism is *unmeasured*. Test A cannot tell a
  constant handicap from no handicap (a constant ~10x slowdown is what a constant handicap
  predicts); Test B moved value dominance from 99.99% to 99.96% and is confounded by the entropy
  collapse the same section proposes; the *p* = 0.95 null has a minimal detectable difference of
  ~50 steps; and loss-*value* shares are ≈ 0 by construction for mean-zero advantages, so they are
  not a gradient proxy in either direction. C6's "the claim was about the summed loss" contradicts
  the 1M document's own wording ("the argument is about direction... the gradient Adam sees"). The
  entropy-collapse replacement is one arm, observational, no intervention — a hypothesis until
  §6.3 row 1 runs.
- **§5.2 / §4.7 / §1 item 3 (`GAE_NORM` vs `MC`).** The greedy evaluation's 95% interval on the
  difference is about [−6.4, +6.4] at 5 seeds × 200 episodes; minimal detectable difference ≈ 8
  steps; the effect is ≈ 5. "Disappears entirely / dead tie" is not supported — the measurement
  is too coarse to see it. The training-time result does not need withdrawing.
- **§1 item 1 / §5.4 (constant factor).** The working table
  (`tmp/20260905_return_mode_cmp_10M.md:222-258`) shows the dilation rising 6.4x → 11.6x in `GAE`
  (3 seeds) and 5.9x → 14.1x in `MC_FIXED` (2 seeds); the published table omits the level-50 row.
  "Does not grow with level / no sign of a lower ceiling" should become "drifts upward; ceiling
  undetermined at n = 2-3".
- Moderate: `MC_RAW` pre-registered banding says "cannot rank" — the substitute criterion is
  post hoc and should be labelled; `s45` was a wanderer eating 0.4-0.7 food/episode at 35-37
  steps, not "a working foraging policy"; the §5.3(d) null is "not detected at n = 4-15", not
  "none"; the §5.7 interaction is real but its +29-step size is a snapshot on the steep part of two
  rising curves and the ANOVA is heteroscedastic by >100x; the §4.7 "+8.2 greedy bias, *p* = 4×10⁻⁵"
  treats 25 evaluations sharing the same 200 worlds as independent.

Reviewed by: plan-reviewer
