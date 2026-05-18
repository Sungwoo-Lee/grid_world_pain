---
title: "Death-penalty ablation for dreamer-srl v2 on 10×10 hypervigilance"
topic: dreamer_srl_v2
status: active
created: 2026-05-18
last_updated: 2026-05-18
phase: 2c
wandb_tag: dreamer_srl_v2_dp_ablation
cross_links:
  - docs/reviews/dreamer_srl_reward_head_audit.md
  - docs/experiments/active/dreamer_srl_v2/EXTENSION_RESULTS.md
  - docs/develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md
---

# Death-penalty ablation for dreamer-srl v2 on 10×10 hypervigilance

> **Status**: PLANNED (configs ready; no runs launched)
> **Opened**: 2026-05-18
> **Related**: see frontmatter `cross_links` and §9 References

---

## 1. Context (plain-language entry point)

**The question, in one sentence.** When the agent dies, the environment gives it a large negative reward — the "death penalty." On the 10×10 hypervigilance survival task, that penalty is currently set to a value of 100. Does *lowering* that penalty toward zero make the agent survive *longer* with the JAX rebuild of the Dreamer model-based RL agent ("dreamer-srl v2"), or is the penalty either irrelevant or actively load-bearing for survival learning?

**Why we are asking now.** A retrospective comparison ([EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md) — the "Phase 1" anchor) showed that on this exact task and step budget, dreamer-srl v2 ends training at ~69 survival steps while the PyTorch reference (sheeprl) ends at ~106. Subsequent diagnostics localized the most suspicious sub-system: the world model's *reward predictor* under-fits the negative side of the reward distribution by a factor of 2.2× to 6.4× more error than on the positive side. A code-level audit ([Phase 2a audit](../../../reviews/dreamer_srl_reward_head_audit.md)) found no bug, but separated the pos/neg asymmetry into two ingredients: (i) an **intrinsic** piece baked into the symlog reward encoding the world model uses — large-magnitude rewards get exponentially stretched when decoded — and (ii) a **remediable** piece driven by class imbalance under maximum-likelihood training: large-negative rewards are rare in the replay buffer (they happen only at terminal/death events), so the head's posterior is dominated by the near-zero mode and pays only a small loss penalty for mis-fitting the heavy negative tail. **Shrinking the death-penalty magnitude is a direct attack on the second ingredient** — it shrinks the heaviest negatives in the reward stream, so the buffer's tail mass moves closer to the head's mean and the head no longer has to model an exponentially-stretched outlier to fit the data. If the gap shrinks when we shrink the penalty, the reward-head asymmetry is causally on the path between the world model and the survival shortfall. If it does not, the death-penalty magnitude is not the lever, and we look elsewhere.

**What is varied, what is held fixed.** This experiment varies *only* the `death_penalty` value on the otherwise-identical 10×10 hypervigilance environment, across four values: 100 (baseline), 50, 10, and 0. Three random seeds per cell, 12 runs total. The agent recipe — the production-winning configuration that reaches ep_len_avg ≈ 191 on the unmodified baseline — is held identical across all 12 runs. Headline metric is **survival steps** (`Game/ep_len_avg`), in line with the project-wide convention; *not* cumulative reward, which is structurally biased downward by the very penalty we are varying.

---

## 2. Background — the two-component asymmetry story

The reason `death_penalty` is the right lever has a sharp mechanistic story behind it, and that story comes out of the Phase 2a audit. Restating in plain English:

**Component 1 — intrinsic symlog encoding asymmetry.** The world model does not predict raw reward; it predicts a probability distribution over 255 bins laid out in `symlog` space on `[-20, +20]`. To turn that distribution back into a real-space reward number, the decoder takes a weighted average over bin values, then applies `symexp` — the inverse of `symlog`. Near zero, `symexp` is approximately linear, so a small misallocation of probability mass between adjacent bins produces a small real-space MAE. Far from zero, `symexp` is exponential, so the same small probability misallocation produces a much larger real-space MAE. The hypervigilance task's positive rewards are bounded near +1 (per-step homeostatic deltas), so they live in the linear regime; the death event delivers a -100 reward, which `symlog(-100) ≈ -5.30`, sitting near bin index ~93 on the grid — far enough that the local `symexp` slope is steep. **This part of the pos/neg MAE asymmetry is a fixed cost of the encoding** and reducing it requires either reshaping the bin grid (out of scope for this experiment) or shrinking the dynamic range of the reward signal (which is what this experiment tries).

**Component 2 — class imbalance × MLE.** The training loss for the reward head is a categorical cross-entropy over twohot targets, averaged uniformly across every `[T, B]` sample in the replay batch. Hypervigilance episodes are ~190 steps long and end in one death step. So out of every ~190 samples, ~1 is a large-negative-reward sample and ~189 are near-zero homeostatic / per-step samples. Under MLE the head's unconditional bias is dragged hard toward the dense near-zero mode, and the small per-sample loss penalty for under-fitting the rare large-negative outlier never overcomes that pull. The Phase 2a audit ranks this as the dominant *remediable* contribution.

**Why lowering `death_penalty` attacks both at once.** Reducing the death penalty from 100 → 50 → 10 → 0 simultaneously (a) shrinks the magnitude of the rare-but-large negative reward, moving it inward toward the linear region of `symexp` (mitigates component 1 for that one bin), and (b) shrinks the *loss-penalty gap* between getting the death step right and getting an idle step right (mitigates component 2 by reducing the asymmetry the MLE objective sees). At `death_penalty = 0`, the only remaining negatives are the per-step homeostatic gradient (small, dense, near-zero) and the per-injury smoothed-damage stream (moderate, sparse, mid-magnitude) — both of which live in the easier regime for the encoding and the loss. **The remediable component should fall the fastest; the intrinsic component should fall more slowly because the homeostatic per-step gradient still has some negatives**, but at `dp=0` the absolute floor on the per-sample reward magnitude is much smaller.

**The hypothesis-driven prediction.** If the reward-head asymmetry is on the causal path from world-model error to the dreamer-srl ↔ sheeprl survival gap, then shrinking the penalty should *monotonically* improve survival, with the biggest jump between the baseline (`dp=100`, the regime where component 1 is most painful) and the first lowered cell, and a saturating effect by `dp=10` or `dp=0` (because both components shrink together and the agent eventually runs out of "easy" reward-head wins). If shrinking the penalty makes survival *worse* at `dp=0`, the death signal is doing real work in shaping defensive policy and we have over-corrected. Both shapes are useful signals.

---

## 3. Hypotheses (pre-registered)

Numerical thresholds are pre-registered to make this experiment falsifiable. "Confirmation" and "refutation" are stated for both directions; outcomes that fall in between are read as "inconclusive at the current seed count" and are noted in §8 (Risks).

### H3a — Death-penalty magnitude monotonically increases survival

**Plain-language**: Reducing `death_penalty` from 100 toward 0 makes the agent survive longer, in a monotonic dose-response.

**Formal**: For seeds-averaged `ep_len_avg` measured in the last-20% window of training (env-steps 3.2M–4.0M), the four cells satisfy

> mean(ep_len_avg | dp=0) ≥ mean(ep_len_avg | dp=10) ≥ mean(ep_len_avg | dp=50) ≥ mean(ep_len_avg | dp=100)

with a margin of ≥ 1 seed-std between adjacent cells.

**Confirmation threshold**: `dp=0` cell beats `dp=100` baseline by **> 25 survival steps** (> 13% relative lift) **with non-overlapping seed-std bands** between the two extremes.

**Refutation threshold**: `dp=0` cell is **within ±10 steps** of `dp=100` baseline (i.e., effect indistinguishable from zero at the seed-noise floor).

### H3b — Negative/positive reward-MAE ratio drops with penalty

**Plain-language**: The world model's pos/neg asymmetry in single-step reward prediction shrinks when the death penalty shrinks.

**Formal**: Define `R(dp) = model_reward_mae_neg / model_reward_mae_pos` averaged over the last-20% window. Then

> R(dp=0) < R(dp=10) < R(dp=50) < R(dp=100)

**Confirmation threshold**: `R(dp=0) < 1.5×` (vs. the baseline `R(dp=100) ≈ 3.3×` reported in the Phase 2a audit context).

**Refutation threshold**: `R` stays **> 2.5× across all four cells** (the asymmetry is unmoved by the lever).

### H3c — The dose-response curve has diagnostic shape

**Plain-language**: The shape of the curve "survival vs death-penalty" tells us *which* of the two components (intrinsic encoding vs. learning capacity vs. defensive signal need) is the bottleneck.

**Three pre-specified shapes and what each implies**:

1. **Roughly linear lift from dp=100 → dp=0.** The penalty is a linear lever; both encoding and class-imbalance components are responding proportionally to the lever. Recommendation: pursue heavier-handed remediation (twohot reweighting, head zero-init, expand bin range) in parallel because the lever is real but unsaturated.

2. **Saturates by dp=50 (dp=10 ≈ dp=0 ≈ dp=50, with most lift already achieved by dp=50).** The head's *learning capacity* is the bottleneck, not the penalty magnitude per se. Lowering the penalty unlocks the head's existing capacity but lowering further does not help because the head is now fitting the easier signal saturation-well. Recommendation: head-side remediation (capacity, zero-init, bin redesign) before any further reward-side moves.

3. **Overshoots at dp=0 (dp=0 is WORSE than dp=10).** The death signal is needed for defensive-policy learning (the agent learns "do not step into the predator" partly *from* the size of the penalty). Going to zero removes the policy gradient that teaches avoidance, and survival drops despite the reward-head asymmetry shrinking. Recommendation: do not pursue `death_penalty` reduction as a remediation; the asymmetry must be attacked head-side (loss reweighting, twohot bin redesign) without changing the reward signal itself.

**No single numerical threshold** for H3c — it is a *shape* hypothesis. The §7 analysis plan specifies a one-figure plot (`ep_len_avg` vs `dp` with seed-std error bars) and a one-paragraph verdict mapping the observed shape onto one of the three pre-registered diagnoses.

---

## 4. Experimental design

### 4.1 Independent variable

| Variable | Values | Rationale |
|---|---|---|
| `body.death_penalty` | {100, 50, 10, 0} | 100 is baseline (production); 50 = halved; 10 = nearly-zero; 0 = floor. Four points span ~2 decades of the lever's range and give the curve in §7 enough shape to discriminate H3c's three diagnoses. |

### 4.2 Controlled variables (held identical across all 12 runs)

**Agent recipe** (the production-winning XS/16/4M cell — see [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]]):

- Algorithm: dreamer-srl v2 (JAX), commit `ccb9a7b` or later (post-parity).
- Size preset: **XS** (256-wide dense, 256-wide recurrent, mlp_layers=1).
- `num_envs`: **16**.
- Total env steps: **4,000,000**.
- Agent config: `configs/dreamer_srl/01_food_only.yaml` (XS template).
- `learning_starts`: as defined in the agent config (1024 prefill — verify at launch).
- All other dreamer-srl hyperparameters (horizon, gamma, kl-regularizer, replay-ratio, twohot bin grid, etc.) inherited verbatim from the agent config.

**Env knobs** (held identical except `death_penalty`):

- Base file: [`configs/experiment/hypervigilance/01-interoNocicept.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept.yaml).
- 10×10 grid, max_steps=500.
- 4 hidden-predator tags, full-grid patrolling predator, 4 food clusters, 2 rabbits.
- `max_injury=100`, `injury_smoothing_duration=3`, `metabolic_cost=1.0`, `food_nutrition_gain=6`.
- `use_homeostatic_reward=true` (Euclidean drive over `[satiation-setpoint, injury]`).
- Sensory: olfactory enabled, visual enabled (range=0), proprioception enabled, interoceptive nociception enabled with α-kernel `tau=3, length=12`.
- Hidden-state observability: `injury_observable=false`, `nutrition_observable=false`.
- Perceptual noise: `enabled=false` (no noise injected in this experiment — the question is reward-side, not perception-side).
- Behavior-measures block: `enabled=true` with the 14-key canonical eval protocol (200 eval episodes, fixed eval-seed list as in the baseline).

**Seeds**: 3 per cell. Cell × seed identifier: `s42`, `s43`, `s44`. Same three seeds across all four cells so seed-noise comparisons are paired.

### 4.3 Sample size + power-analysis sketch

12 runs × ~3.4 h wall-clock per run (XS/16/4M is the recipe's documented runtime) ≈ 41 GPU-hours total. On 4 nodes in parallel (3 runs each), ~10 wall-clock hours.

**Seed-noise prior**: The closest reference data is the XS/16/4M production recipe ([[20260518_1513_production_recipe_xs_16_4m_hypervigilance]]) reporting `ep_len_avg ≈ 184` for the winning cell at a *single seed* (`yxij4lrc`). The Phase 1 retrospective ([EXTENSION_RESULTS.md](./EXTENSION_RESULTS.md)) ran one seed per cell on the harder extension tasks. **The project does not yet have a multi-seed std for dreamer-srl v2 on hypervigilance.** Therefore the power calculation for "is 3 seeds enough" is necessarily a forward-looking estimate, not a back-tested one.

The expected effect — H3a's confirmation threshold of > 25 steps lift (~13% of baseline) — is large compared to typical RL seed noise on this scale of survival metric. Reference data points:
- The sheeprl 10×10 baseline (yt1uts22) shows ~10-step within-window per-sample std at its plateau; treating seed std as comparable order-of-magnitude.
- The food-only parity launch surfaced "seed 0 broke free at 71k vs ~14.7k for others" — i.e., trajectory-onset variance is large, but *steady-state* variance in the last-20% window is much smaller once all seeds have escaped the random floor.

**Verdict on seed sufficiency**: For the headline H3a confirmation (> 25 step lift, end-of-training mean), 3 seeds is *probably* sufficient with seed-std bands ~10–15 steps. For the **shape discrimination in H3c** — specifically distinguishing "saturated by dp=50" from "linear all the way to dp=0" — 3 seeds may be **marginal** if the per-cell std is > 10 steps. **Flag for the user**: if the first 3-seed pass returns a clearly-non-monotonic curve or seed bands that fully overlap between adjacent cells (e.g., `dp=10` and `dp=0` bands overlap entirely), recommend a follow-up 2-seed extension on the contested cells (s45, s46) rather than re-running all 12. The four-cell × 3-seed manifest below is the **initial** allocation.

### 4.4 Confounds and limitations

| Confound | Affected cells | Severity | Mitigation |
|---|---|---|---|
| Lowered `death_penalty` shifts the absolute scale of cumulative reward, making per-cell reward-curves not directly comparable. | All non-baseline cells | Low | Headline is `ep_len_avg`, not reward. The reward-MAE asymmetry metric (H3b) is *ratio*-based and scale-invariant by construction. |
| `injury_smoothing_duration=3` is still in play; injury-driven (non-death) negatives are unaffected by the lever. | All cells equally | Low (intentional) | Per the prompt's scope, only `death_penalty` is varied; the injury-stream negatives are part of the controlled-variable set. |
| `dp=0` removes a categorical "you died" signal from the reward stream; in homeostatic-drive runs, the agent still gets the drive-change reward, but the death event no longer has a distinguished penalty. | `dp=0` only | Medium | This is exactly what H3c's "overshoot" diagnosis tests for. If `dp=0` is worse than `dp=10`, that's the signal. Not a confound to mitigate; an experimental question. |
| Single-codebase comparison: no parallel sheeprl runs at varied `death_penalty`. | All cells | Low | Phase 2a audit shows the reward-head asymmetry is dreamer-srl-specific in degree, not algorithm-universal. The point of this experiment is to test whether the dreamer-srl-specific lever moves dreamer-srl, not to compare cross-algorithm. If H3a confirms, a follow-up experiment can cross with sheeprl. |
| The behavior-measures eval block runs at training-end with `eval_obs_noise: training` and a fixed eval-seed list. Eval episodes terminate on death — the metric `ep_len_avg` is the per-eval-episode mean, also dependent on the death penalty's role in the policy. | All cells | Low | Same eval protocol across cells; comparison is internally consistent. |

---

## 5. File Changes (configs to produce — already produced)

Four new env configs, each a verbatim copy of `configs/experiment/hypervigilance/01-interoNocicept.yaml` with **only line 261 (`death_penalty: <X>`) changed**. No new YAML keys, no schema changes — `body.death_penalty` is already a mandatory key (loaded via `config.get_mandatory('body.death_penalty')` at `src/environment/config_loader.py:520`).

| New file | Single-line diff vs base |
|---|---|
| [`configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp100.yaml`](../../../../configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp100.yaml) | (byte-identical to base — explicit pointer for ablation provenance) |
| [`configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp50.yaml`](../../../../configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp50.yaml) | `death_penalty: 100` → `death_penalty: 50` |
| [`configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp10.yaml`](../../../../configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp10.yaml) | `death_penalty: 100` → `death_penalty: 10` |
| [`configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp0.yaml`](../../../../configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp0.yaml) | `death_penalty: 100` → `death_penalty: 0` |

The `dp100` file is included as a **named baseline pointer** so the launch manifest, the analysis script, and downstream re-runs all reference the same physical file rather than the unmodified base — keeping cross-cell provenance symmetric and making it impossible to lose the "this is the dp=100 cell in this experiment" identity at analysis time.

---

## 6. Launch Manifest

Per [docs/TEMPLATES/training_analysis.md §3](../../../TEMPLATES/training_analysis.md). Designer fills Tag, wandb-name, wandb-group, wandb-job-type, Seed, env-config, agent-config. Runner fills Node, GPU, Launched at, WandB run ID, Log path at launch time.

**Tag = wandb-name convention**: `dreamer_srl_v2_dp<DP>_s<SEED>` — uniquely identifies the (cell, seed) pair and is parseable.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | dp100 | `dreamer_srl_v2_dp100_s42` | `dreamer_srl_v2_dp_ablation` | ablation | 42 | — | — | — | — | — |
| 2 | planned | dp100 | `dreamer_srl_v2_dp100_s43` | `dreamer_srl_v2_dp_ablation` | ablation | 43 | — | — | — | — | — |
| 3 | planned | dp100 | `dreamer_srl_v2_dp100_s44` | `dreamer_srl_v2_dp_ablation` | ablation | 44 | — | — | — | — | — |
| 4 | planned | dp50 | `dreamer_srl_v2_dp50_s42` | `dreamer_srl_v2_dp_ablation` | ablation | 42 | — | — | — | — | — |
| 5 | planned | dp50 | `dreamer_srl_v2_dp50_s43` | `dreamer_srl_v2_dp_ablation` | ablation | 43 | — | — | — | — | — |
| 6 | planned | dp50 | `dreamer_srl_v2_dp50_s44` | `dreamer_srl_v2_dp_ablation` | ablation | 44 | — | — | — | — | — |
| 7 | planned | dp10 | `dreamer_srl_v2_dp10_s42` | `dreamer_srl_v2_dp_ablation` | ablation | 42 | — | — | — | — | — |
| 8 | planned | dp10 | `dreamer_srl_v2_dp10_s43` | `dreamer_srl_v2_dp_ablation` | ablation | 43 | — | — | — | — | — |
| 9 | planned | dp10 | `dreamer_srl_v2_dp10_s44` | `dreamer_srl_v2_dp_ablation` | ablation | 44 | — | — | — | — | — |
| 10 | planned | dp0 | `dreamer_srl_v2_dp0_s42` | `dreamer_srl_v2_dp_ablation` | ablation | 42 | — | — | — | — | — |
| 11 | planned | dp0 | `dreamer_srl_v2_dp0_s43` | `dreamer_srl_v2_dp_ablation` | ablation | 43 | — | — | — | — | — |
| 12 | planned | dp0 | `dreamer_srl_v2_dp0_s44` | `dreamer_srl_v2_dp_ablation` | ablation | 44 | — | — | — | — | — |

### 6.1 Configs to Produce

All 12 runs share the same agent config; they vary only by env config and seed.

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp100.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 2 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp100.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 3 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp100.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 4 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp50.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 5 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp50.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 6 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp50.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 7 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp10.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 8 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp10.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 9 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp10.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 10 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp0.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 11 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp0.yaml` | `configs/dreamer_srl/01_food_only.yaml` |
| 12 | `configs/experiment/hypervigilance/death_penalty_ablation/01-interoNocicept_dp0.yaml` | `configs/dreamer_srl/01_food_only.yaml` |

### 6.2 Tag-collision check

All 12 Tags listed above are unique within this manifest. No tag in the project's WandB history matches the pattern `dreamer_srl_v2_dp<N>_s<N>` (the ablation is the first use of this scheme). Globally unique tags.

---

## 7. Analysis plan (pre-specified)

Per the pre-registration discipline, the plots, tables, and primary statistics are fixed before any data lands. The `experiment-analyzer` agent will produce all of the following on the same doc (extending sections 4–6 of this file):

### 7.1 Primary table (H3a verdict surface)

`ep_len_avg` per cell, mean ± seed-std over the last-20% window (env-steps 3.2M–4.0M):

| Cell | dp | Seed 42 | Seed 43 | Seed 44 | Mean ± std | Lift vs dp=100 |
|---|---|---|---|---|---|---|
| dp100 | 100 | TBD | TBD | TBD | TBD | (baseline) |
| dp50 | 50 | TBD | TBD | TBD | TBD | TBD |
| dp10 | 10 | TBD | TBD | TBD | TBD | TBD |
| dp0 | 0 | TBD | TBD | TBD | TBD | TBD |

**Verdict on H3a**: confirm / refute / inconclusive, with the seed-std overlap check.

### 7.2 Primary figure (H3c shape surface)

A single line plot:
- X-axis: `death_penalty ∈ {0, 10, 50, 100}` (log-ish ordering, but plotted on a linear axis with the four points marked).
- Y-axis: `ep_len_avg` (mean over 3 seeds, error bars = ±1 seed-std).
- Three reference lines: (i) the sheeprl 10×10 baseline at ep_len_avg=106 (Phase 1 anchor), (ii) the production XS/16/4M baseline at ep_len_avg≈184–191 (insight-cited mean), (iii) the y=baseline-mean horizontal.

**Verdict on H3c**: map the curve shape onto one of the three pre-registered diagnoses (linear / saturated / overshoot).

### 7.3 Reward-MAE asymmetry table (H3b verdict surface)

Per cell, the world-model reward-head asymmetry, from training-time logging:

| Cell | dp | `model_reward_mae_pos` (mean) | `model_reward_mae_neg` (mean) | Ratio `neg / pos` |
|---|---|---|---|---|
| dp100 | 100 | TBD | TBD | TBD (expected ≈ 3.3×) |
| dp50 | 50 | TBD | TBD | TBD |
| dp10 | 10 | TBD | TBD | TBD |
| dp0 | 0 | TBD | TBD | TBD |

All values averaged over the last-20% window across the 3 seeds per cell.

**Verdict on H3b**: confirm / refute / inconclusive vs the 1.5× and 2.5× thresholds.

### 7.4 Temporal-evolution check (mandatory per project convention)

Per cell, the trajectory of `ep_len_avg` over training (not just end-of-training):
- Plot all 4 cells on one panel, 3 seeds per cell as faint individual lines + thick cell-mean line.
- X-axis: env-steps, 0 to 4M.
- Visible inspection for divergence points: does the dp=0 cell pull ahead early and stay ahead, or does it converge late, or does it lead early and collapse late? Distinguishing these reveals *how* the lever acts (early-learning vs steady-state-policy).

### 7.5 Cross-correlation with reward-head loss

Per cell, scatter the steady-state `model_reward_mae_neg` against `ep_len_avg`. If H3b (ratio shrinks) holds **and** H3a (survival rises) holds, expect a negative slope on this scatter — confirming the reward-head asymmetry is on the causal path. A null slope means H3a and H3b are independent effects (e.g., dp=0 helps survival via a non-reward-head mechanism), which is itself diagnostic.

### 7.6 Comparator baselines (Phase 1 anchor)

Compare the best cell of this ablation against:
- The sheeprl 10×10 baseline (`yt1uts22`, 106 steps at 200k budget — note the budget mismatch: 4M vs 200k).
- The dreamer-srl 10×10 baseline at the original extension launch (`405f0555`, 69 steps at 200k budget).
- The XS/16/4M production recipe winner (`yxij4lrc`, ~184 steps at 4M budget — this is the closest matched-budget reference).

The H3a confirmation that dp=0 reaches > 25 steps over dp=100 (i.e., > 209 steps) would put dreamer-srl meaningfully above the sheeprl 10×10 baseline *and* above the production winner. The interpretation in that case is that the death-penalty was a binding constraint on the existing recipe.

---

## 8. Risks and open questions

### Risk 1 — Effect smaller than seed noise (3-seed insufficiency)

With only 3 seeds per cell and no measured prior on within-cell seed-std for this stack, the 12-run manifest could come back with overlapping bands across all four cells, leaving H3a inconclusive. **Mitigation**: if the first pass shows the *direction* is right (`dp=0 > dp=100` mean) but bands overlap, recommend a 2-seed extension on cells dp=0 and dp=100 only (4 extra runs, ~14 GPU-hours), rather than re-running the whole grid. This is escalation-path-only; the initial manifest is 12.

### Risk 2 — `dp=0` reveals death penalty is irrelevant to the asymmetry → H1 partially refuted

The Phase 1 hypothesis H1 (reward-head asymmetry drives the gap) requires *some* path from reward-asymmetry to survival. If `dp=0` flattens the asymmetry ratio (H3b confirms) but survival is *unchanged* (H3a refutes), the asymmetry is not on the causal path to survival. **Follow-up that this would trigger**: re-examine the rPPO comparison's *own* reward-MAE shape. rPPO is a value-based critic, not a twohot reward head — but if the project finds that rPPO's value head shows the same 3.3× pos/neg asymmetry without the survival shortfall, the asymmetry alone is exonerated as a gap driver. The Phase 2b offline-WM diagnostic ([`docs/develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md`](../../../develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md)) is the right next instrument; this ablation provides the survival-side counterevidence.

### Risk 3 — `dp=0` actively harms survival (H3c overshoot)

If `dp=0` underperforms `dp=10` *and* `dp=100`, the death signal is a load-bearing policy-shaping signal — the agent learns "avoid the predator" partly *because* it costs 100. Going to zero removes the gradient that teaches avoidance, and survival collapses. **Implication**: the `death_penalty` lever is unsafe as a remediation. Future reward-asymmetry mitigations must operate on the *encoding/loss side* (zero-init the head, reweight the twohot loss, expand the bin range) without changing the reward signal itself.

### Risk 4 — Confound with `injury_smoothing_duration`

Per the scope decision the prompt locks, `injury_smoothing_duration=3` is NOT varied. But injury-stream negatives (smoothed predator damage of 15–45 spread over 3 steps = ~5–15 per step) are still in the buffer, contributing their own mid-magnitude tail mass. If the head's MLE bias is dominated by injury-stream negatives rather than death-event negatives, varying `death_penalty` alone may leave the asymmetry largely unchanged. **Mitigation**: H3b's confirmation/refutation thresholds are explicit; if H3b refutes (ratio stays > 2.5× across all cells), the injury stream — not the death event — is the dominant tail-mass source, and a follow-up ablation on `injury_smoothing_duration` is the right next move. **The prompt locks `injury_smoothing_duration` out of *this* experiment**, but the result may motivate a separate one.

### Open question 1 — Should evaluation be done under `dp=100` for every cell?

Currently each cell evaluates under its own `death_penalty` (the eval env uses the same config as training). One could argue that the "true" survival metric should be evaluated under the production penalty so all cells are scored on the same scale. **Decision (pre-registered)**: evaluate each cell under its own training penalty. Rationale: (a) the headline metric is `ep_len_avg`, which is invariant to `death_penalty` (a step is a step regardless of what reward it pays); (b) introducing a held-out eval env would conflate training-policy effects with eval-env effects; (c) the existing eval protocol with 200 eval episodes per cell at a fixed seed list is already strong.

### Open question 2 — Should `H_pos`/`H_neg` bucket counts be logged?

The Phase 2a audit Q5 flagged that `model_reward_mae_total` is dominated by the near-zero bucket and is uninformative about the pos/neg ratio. For the H3b verdict to be trustworthy, the analyzer needs to confirm `n_pos` and `n_neg` (the bucket-count denominators) are non-degenerate at the lower `dp` cells. **Recommendation**: a separate Phase 2b script (the offline-WM diagnostic plan referenced above) should run on at least one checkpoint per cell to confirm the bucket counts; not blocking this ablation but should be done before drawing final conclusions on H3b.

---

## 9. References and cross-links

**Phase 1 anchor** (motivation for the experiment): [`docs/experiments/active/dreamer_srl_v2/EXTENSION_RESULTS.md`](./EXTENSION_RESULTS.md) — the dreamer-srl v2 vs sheeprl head-to-head on the extension tasks, identifying the 35% survival gap on 10×10 hypervigilance.

**Phase 2a audit** (mechanistic basis): [`docs/reviews/dreamer_srl_reward_head_audit.md`](../../../reviews/dreamer_srl_reward_head_audit.md) — confirms no code bug; isolates the intrinsic-symlog vs class-imbalance-MLE two-component story that motivates `death_penalty` as the lever.

**Phase 2b offline-WM diagnostic plan** (complementary instrument): [`docs/develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md`](../../../develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md) — per-horizon reward-MAE with pos/neg split; can run on checkpoints from this ablation to verify bucket counts and per-horizon shape.

**Settled memory insights cited**:
- [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]] — dreamer-srl v2 parity success; the baseline working stack this ablation runs on.
- [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]] — the XS/16/4M recipe held fixed across all 12 cells.
- [[20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry]] — first observation of pos/neg asymmetry, partial fix via zero-init (cited as the related-lever-from-the-head-side counterpart).
- [[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]] — paper-canonical bins as another head-side lever attempt.
- [[20260509_1534_wm_reward_head_localized_failure_a1]] — offline-WM diagnostic localizing the failure to the reward head; original Dreamer codebase precedent for the per-horizon analysis.

**Base env config**: [`configs/experiment/hypervigilance/01-interoNocicept.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept.yaml).
**Base agent config**: [`configs/dreamer_srl/01_food_only.yaml`](../../../../configs/dreamer_srl/01_food_only.yaml) (XS template).
**New env configs**: under [`configs/experiment/hypervigilance/death_penalty_ablation/`](../../../../configs/experiment/hypervigilance/death_penalty_ablation/).

---

## 10. Hand-off

**Pre-launch (next session)**:
1. `env-config-auditor` pre-flight on the 4 new env configs at `configs/experiment/hypervigilance/death_penalty_ablation/`.
2. User collects node + GPU upfront via `AskUserQuestion`.
3. `training-runner` launches per the §6 manifest. Runner fills the actual columns at launch time.

**Post-training**:
4. `experiment-analyzer` reads §7 analysis plan and fills §7.1 / §7.2 / §7.3 / §7.4 / §7.5 tables and figures in this doc.
5. The §3 hypotheses verdicts (H3a, H3b, H3c) are written into a new §11 Results section. The §8 risk-triggered follow-ups (whichever fire) are routed to `experiment-designer` (Risk 1 escalation) or to `senior-developer` (Risk 4 follow-up ablation, or Risk 2 cross-check against rPPO).

**Not in scope for this session**: launching any runs, collecting node/GPU info, modifying agent configs, modifying `src/` or `scripts/`.
