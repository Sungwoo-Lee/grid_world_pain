---
title: "NMN FiLM grouping_size screen — how many modulation groups regularise the neuromodulator?"
topic: basic_curriculum
status: active
created: 2026-06-27
last_updated: 2026-06-27
phase: 1
wandb_tag: "rppo_nmn_film_g*_screen_s*"
develop_link:
---

# NMN FiLM grouping_size screen

## 1. Research Question

### Plain-language entry point

We have a small "neuromodulator" bolted onto a recurrent reinforcement-learning
agent. The neuromodulator is a tiny side-network that watches the agent's internal
state and, on every step, **rescales and shifts the activity of the main policy
network's neurons** — a learned per-neuron gain (γ) and bias (β), in the style called
"FiLM". Performance is always measured in **survival steps** — how many steps the
agent lives before it starves or is killed — never reward.

In a prior head-to-head (see [[nmn_film_vs_plain_longL4]] — "did the modulator
help?"), the most flexible version of this modulator, which gives **one γ/β pair per
single neuron** (the setting called `grouping_size: 1`, i.e. 128 independent
modulation groups for a 128-neuron network), **led early but then crashed late**:
around 34 million episodes its survival fell from ~268 to ~180 steps, time-locked to
its internal "temperature" knob pinning at its maximum allowed value and a burst of
exploding gradients. Diagnosis: the per-neuron modulator is **too powerful and too
weakly regularised** — it over-drives the policy network and tips into instability.

The obvious regulariser is **coarsening the grouping**: instead of one γ/β per neuron,
share one γ/β across a *group* of neurons. Fewer, larger groups = a smoother, less
expressive, more stable modulator. This experiment is a **screen**: train the modulator
from scratch on the hardest single world (a 10×10 arena with a fast, far-seeing
predator) across a ladder of group counts — from 128 groups (per-neuron, the known-bad
corner) down to 1 group (a single global modulation signal) — and ask **which group
count gives the best, most stable survival**. It is a first-pass localisation, not a
final confirmation; the winning group count earns a longer, multi-seed follow-up.

One confound is removed up front. The prior crash coincided with the temperature knob
hitting its ceiling of 5.0, so a naive group-count sweep would mix up "how many groups"
with "how much temperature headroom". We therefore **raise the temperature ceiling to a
non-binding 10.0 for all ten runs**, so group count is the only modulation-capacity
knob that changes inside the sweep.

**Formal hypothesis.**

> **H₀** (null — *group count is inert*): survival steps on the far-sight world do not
> depend on `grouping_size`; coarsening the modulation does not change end-of-training
> survival relative to the per-neuron setting.
>
> **H₁a** (interior optimum — *coarsening helps, up to a point*): an **intermediate**
> group count produces higher steady-state survival than both the per-neuron extreme
> (`grouping_size: 1`, 128 groups) and the global extreme (`grouping_size: 128`,
> 1 group), and beats the unmodulated from-scratch baseline of ~261 survival steps by a
> meaningful margin without a late crash.
>
> **H₁b** (monotone regularisation — *coarser is simply better*): survival increases
> monotonically as grouping coarsens (group count falls), with the global single-group
> setting best.

"Steady-state survival" = mean survival steps over the **last 1M episodes**. "A late
crash" is defined numerically in §5.

## 2. Experimental Design

### 2.0 Why this design (decisions and one override)

The user steered four design decisions; all four are adopted. One instruction is
**overridden with reason** (the "10 configs" count → 8 configs); see §3.1.

1. **Task / budget — from-scratch, single world, ~10M episodes.** Every run trains
   from scratch on the far-sight predator world (no curriculum) for 10M episodes at
   `num_envs=128`, matching the existing from-scratch baseline exactly (that baseline,
   WandB run `c8cd77ft` / name `rppo_basic04_farsight_n114`, plateaued at **~261
   survival steps**; 10M episodes, `num_envs=128`, seed 42, `max_steps=500`). This makes
   all ten runs (a) finish in one wall-clock pass on the ten GPUs and (b) directly
   comparable to the 261 anchor. It screens the **architecture**; it deliberately does
   **not** reproduce the continual-setting crash (see §2.3, the crash-horizon caveat).
2. **De-confound the temperature rail.** All ten runs hold `temp_clip = [0.5, 10.0]`
   (non-binding; the architecture's own default upper bound). `grouping_size` is then
   the only modulation-capacity knob varying inside the sweep. No slot is spent
   re-testing the old 5.0 ceiling — the prior analysis already characterised it as
   railed, and mixing ceilings back in would reintroduce the very confound we are
   removing. The comparison to the 5.0-ceiling regime is "external", via the prior doc.
3. **Grid — full single-seed curve over the group axis + two noise-check repeats.**
   `grouping_size ∈ {1, 2, 4, 8, 16, 32, 64, 128}` at seed 42 (8 runs) gives an
   eight-point curve over the whole capacity range — enough resolution to see the
   *shape* (interior optimum vs monotone vs flat). Two extra runs repeat the two
   most decision-critical points at a second seed (43): the per-neuron corner
   `grouping_size: 1` (the known-pathological setting — does it still destabilise from
   scratch under the de-confounded ceiling?) and a mid-coarse candidate
   `grouping_size: 16` (8 groups — a plausible "enough structure, coarse enough to be
   stable" sweet spot). This favours the grouping curve over seeds, as steered, while
   still buying a minimal seed-noise gauge at the two anchors.
4. **Keep grouping the dominant axis.** No second knob (e.g. `mod_hidden_size`) is
   swept; the two spare runs go to seed repeats, judged higher-value than a second axis
   for a first-pass screen whose explicit purpose is localising the group-count region.

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `modulation.grouping_size` (primary) | 1, 2, 4, 8, 16, 32, 64, 128 | Neurons per FiLM group. `num_groups = ceil(128 / grouping_size)` = 128, 64, 32, 16, 8, 4, 2, 1. Sets modulation capacity / regularisation strength. |
| Seed (secondary, anchors only) | 42 (all 8 curve points), 43 (repeats at `grouping_size` 1 and 16) | Minimal seed-noise gauge at the two decision-critical group counts. |

`num_groups` per value (load-bearing — this is what actually changes inside the model):

| `grouping_size` | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| `num_groups` (= ceil(128/g)) | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| interpretation | per-neuron (known-bad) | | | | mid candidate | | | global (1 signal) |

### 2.2 Controlled Variables

Held constant across all ten runs:

- **Env config** — `configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml`
  (far-sight predator, 10×10), unchanged. No curriculum, no `--configs-dir`.
- **Task budget** — `--episodes 10000000`, `--num-envs 128`, `--log-interval 10`,
  `--checkpoint-frequency 100000`, `environment.max_steps=500` (inherited).
- **Temperature ceiling** — `temp_clip = [0.5, 10.0]` for all ten (the de-confound).
- **All PPO / encoding / hierarchical hyperparameters** — byte-identical to the base
  FiLM agent config (`recurrent_ppo_nmn_film_g1_tempceil5.yaml`): `lr_actor 5e-4`,
  `lr_critic 1e-4`, `gamma 0.95`, `K_epochs 4`, `eps_clip 0.1`, `sequence_length 128`,
  `hidden_size 128`, `entropy_coef 0.01`, `gae_lambda 0.95`, `vf_coef 0.5`,
  `max_grad_norm 0.5`, `use_layer_norm true`, `rnn_type GRU`, `activation relu`,
  `return_mode MC`, `encoding_mode hierarchical`, and the rest of the `modulation:`
  block (`type FiLM`, `mod_hidden_size 16`, `percept_bias_init 3.0`,
  `memory_bias_init 0.0`, `memory_clip [-2.0, 2.0]`).
- **Comparison anchor** — the existing from-scratch far-sight baseline (~261 survival,
  run `c8cd77ft`). It is **not** re-run; all ten slots go to the modulated grid. The
  plain agent is the external anchor, matched on seed (42), budget, and env.

**Obs/action/sensor parity.** Only the agent config's `modulation` block changes; the
env config is the unmodified far-sight L4 file. Observation dimension, action dimension,
and the sensor fingerprint are fixed by the env + encoding, both unchanged, so they are
**identical across all ten runs and identical to the 261 baseline**. `grouping_size`
only changes the modulator's internal head output width (`num_groups`), never the
obs/action interface. No env-side parity risk is introduced.

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| **Crash-horizon mismatch** — the prior crash appeared at ~34M episodes in a *curriculum* run; this screen is from-scratch and stops at 10M episodes, so it likely **does not reach the regime where the late instability emerged**. | all | **High** | Stated loudly: this screens early/mid-training architecture quality + diagnostic health, not the late crash directly. The winning group count gets a long-horizon, multi-seed follow-up that *can* probe the crash. Pre-registered drawdown check (§5) still flags any in-window instability. |
| **Single seed on 6 of 8 curve points** | g ∈ {2,4,8,32,64,128} | Med | This is a screen, not a confirmation. Only `grouping_size` 1 and 16 get a 2nd seed. The 8-point curve reads *shape*; magnitudes at single-seed points are provisional. Winner routed to ≥3-seed follow-up before any effect is "confirmed". |
| **261 anchor is itself single-seed** (seed 42) | the baseline comparison | Med | Margin threshold (§5) set at +5% (~+14 steps) to exceed plausible seed noise; curve points share seed 42 with the anchor for maximal comparability. |
| **Temperature may rail at 10.0 anyway** | any | Low | If `temperature_mean` pins at 10.0 even with coarse grouping, temperature headroom is a separate binding axis → noted as a follow-up sweep, not a screen failure. |
| **Coarse vs fine is also a parameter-count change** | across grid | Low | Inherent to the knob under test — fewer groups = fewer head parameters. This is the regularisation mechanism, not a nuisance; reported as such. |

## 3. Launch Manifest

System-of-record for every run. Designer fills planned columns; `training-runner` fills
actual columns (`Node`, `GPU`, `Launched at`, `WandB run ID`, `Log path`, `Status→running/completed`).
Slot layout: nodes 106–110, two GPUs each (`:0`, `:1`) = 10 parallel slots, one run per GPU.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | g1 | `rppo_nmn_film_g1_screen_s42` | basic_curriculum | prod | 42 | 106 | 0 | — | — | — |
| 2 | planned | g2 | `rppo_nmn_film_g2_screen_s42` | basic_curriculum | prod | 42 | 106 | 1 | — | — | — |
| 3 | planned | g4 | `rppo_nmn_film_g4_screen_s42` | basic_curriculum | prod | 42 | 107 | 0 | — | — | — |
| 4 | planned | g8 | `rppo_nmn_film_g8_screen_s42` | basic_curriculum | prod | 42 | 107 | 1 | — | — | — |
| 5 | planned | g16 | `rppo_nmn_film_g16_screen_s42` | basic_curriculum | prod | 42 | 108 | 0 | — | — | — |
| 6 | planned | g32 | `rppo_nmn_film_g32_screen_s42` | basic_curriculum | prod | 42 | 108 | 1 | — | — | — |
| 7 | planned | g64 | `rppo_nmn_film_g64_screen_s42` | basic_curriculum | prod | 42 | 109 | 0 | — | — | — |
| 8 | planned | g128 | `rppo_nmn_film_g128_screen_s42` | basic_curriculum | prod | 42 | 109 | 1 | — | — | — |
| 9 | planned | g1 | `rppo_nmn_film_g1_screen_s43` | basic_curriculum | prod | 43 | 110 | 0 | — | — | — |
| 10 | planned | g16 | `rppo_nmn_film_g16_screen_s43` | basic_curriculum | prod | 43 | 110 | 1 | — | — | — |

Tag = wandb-name on every row (keeps `train.py`'s name fallback in sync). All tags are
unique. Scheme: `rppo_nmn_film_g<grouping_size>_screen_s<seed>`.

### 3.1 Configs to Produce (designer-only, pre-launch)

**Override of the "10 configs" instruction (stated with reason).** The user asked for
"10 agent YAML configs". Seed is **not** an agent-config key in this project — it is
supplied at launch via `--seed`. Runs 9 and 10 differ from runs 1 and 5 *only* in seed,
so they reuse the same agent config files. Writing two byte-identical duplicate configs
to represent a seed change would be an anti-pattern (it invites the false belief that the
files differ, and risks drift). **Therefore 8 distinct agent configs cover all 10 runs**;
the two repeats reuse configs g1 and g16 with `--seed 43`. The manifest, not a duplicate
file, is the system-of-record for the seed split.

| Run | Config (env) | Config (agent) | `--seed` |
|-----|--------------|----------------|----------|
| 1 | `configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml` | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml` | 42 |
| 2 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g2_screen.yaml` | 42 |
| 3 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g4_screen.yaml` | 42 |
| 4 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g8_screen.yaml` | 42 |
| 5 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml` | 42 |
| 6 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml` | 42 |
| 7 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g64_screen.yaml` | 42 |
| 8 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g128_screen.yaml` | 42 |
| 9 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml` (reused) | 43 |
| 10 | (same env) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml` (reused) | 43 |

All eight agent configs exist and are validated (parse cleanly; only the
`modulation` block differs from the base FiLM config).

### 3.2 Launch command template

Per run (the `training-runner` substitutes node, GPU, agent config, seed, tag):

```bash
./run_command.py <node> \
  "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
    --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
    --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g<G>_screen.yaml \
    --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
    --log-interval 10 --device cuda:<GPU> --seed <SEED> \
    --tag rppo_nmn_film_g<G>_screen_s<SEED>"
```

### 3.3 Compute estimate

10 runs × 10M episodes, each on one GPU, all in parallel across 10 GPUs → **one
wall-clock pass**. The existing far-sight from-scratch baseline (same env, budget,
num_envs) is the s/it reference; expect comparable per-run wall-clock. The FiLM modulator
adds a small fixed overhead (a 16-unit GRU + heads) that is independent of `grouping_size`
to first order, so all ten runs should finish within a similar window.

## 4. Predicted Outcomes

- **Most likely (H₁a, interior optimum):** survival rises as grouping coarsens from
  per-neuron (128 groups), peaks at an intermediate group count (somewhere around 4–16
  groups, i.e. `grouping_size` 8–32), then falls toward the global single-group setting
  as the modulator loses the per-channel flexibility that helped early. The per-neuron
  end shows the largest γ/β variance and the most gradient spikiness; the global end is
  stable but bland.
- **Possible (H₁b, monotone):** survival simply increases as group count falls — coarser
  is uniformly safer at this horizon — with the global setting best. Would suggest the
  early-transfer benefit of fine grouping needs longer horizons / curriculum to appear.
- **Null (H₀, flat):** all group counts land within noise of each other and of ~261.
  Would suggest 10M from-scratch L4 is too short for grouping to express itself (the
  crash-horizon caveat), or that grouping is not the binding constraint.

## 5. Analysis Plan (pre-specified)

**Primary statistic.** Steady-state survival = mean `Episode/Steps` over the **last 1M
episodes** of each run, per `grouping_size`. Plotted as survival vs `num_groups` (log x).
Where a group count has two seeds (g1, g16), report mean ± half-range as the seed gauge.

**Effect-size threshold ("a grouping config wins").** A `grouping_size` value **beats the
baseline** if its steady-state survival ≥ **261 + 14 ≈ 275 steps** (a +5% margin chosen to
exceed plausible single-seed noise around the 261 anchor).

**Stability threshold ("no late crash").** Over the **late half (5M–10M episodes)**, the
maximum sustained drawdown from a run's own running peak survival (measured on the
50-point rolling mean) must stay **< 30 steps**. (The prior pathology was a ~95-step
crash; <30 = "no crash-class instability at this horizon".)

**Temporal-evolution check (mandatory).** For every run, inspect the full survival
trajectory across 0→10M episodes, not just the endpoint: convergence episode, monotone vs
oscillating, any discrete drop. Diagnostic time series tracked per run (all already logged
by `src/`): `modulator/temperature_mean` (does it rail at 10.0 or settle in [3,5)?),
γ/β group means and **stds** (expected to shrink as grouping coarsens), `loss/grad_norm`
and `modulator/grad_norm` (spike frequency/magnitude), `loss/entropy` (collapse?),
`loss/value` (critic health).

**Cross-correlation (pre-specified).** If any run shows an in-window survival drawdown
≥ 30 steps, time-lock it against `temperature_mean` saturation and the γ-std trajectory
within a ±0.5M-episode window — the same signature reported in the prior analysis — to
test whether the per-neuron instability mechanism reproduces from scratch.

**Confirmation / refutation (pre-registered).**

- **Screen succeeds (motivates a follow-up):** at least one `grouping_size` value clears
  **both** thresholds — steady-state survival ≥ ~275 **and** late drawdown < 30 steps —
  AND it is a coarser setting than per-neuron (`grouping_size > 1`). Verdict: coarser
  grouping is a viable regulariser; route the best-clearing group count to a long-horizon,
  ≥3-seed confirmation.
- **Interior optimum confirmed (H₁a):** the winning group count beats *both* the
  per-neuron (g=1) and global (g=128) extremes by ≥ the +5% margin.
- **Refuted (grouping does not fix it):** **no** `grouping_size > 1` value clears both
  thresholds — i.e. either nothing beats ~275, or every config that beats ~275 also shows
  a ≥30-step late drawdown. Verdict: coarsening alone does not recover a stable modulated
  benefit on this task; the modulator's value is not regularisable via group count, and
  attention shifts to other levers (temperature schedule, mod_hidden_size, explicit γ/β
  penalties) or to the conclusion that the modulator does not help here.
- **Inert axis (H₀):** survival flat across the whole grid (max−min < the +5% margin).
  Verdict: grouping is not the binding constraint at this horizon; re-decide between a
  longer-horizon re-run and abandoning the grouping axis.

## 6. Failure-Mode Catalog (pre-decided)

- **NaN / value explosion in a run.** From-scratch, with all hyperparameters fixed,
  a NaN at a given `grouping_size` **counts against that group count** (the architecture
  is unstable at that capacity) — it refutes that point, not "the run". Record the episode
  and the γ-std / grad-norm at failure. A NaN at g=1 specifically would be a positive
  finding (per-neuron instability reproduces from scratch).
- **Temperature pins at the 10.0 ceiling for all group counts.** Not a screen failure —
  it means temperature headroom is a *separate* binding axis. Report and propose a
  temperature-schedule follow-up; the grouping verdict still stands on survival + drawdown.
- **Everything ≈ 261 (flat).** Read against the crash-horizon caveat (§2.3): most likely
  the 10M-episode from-scratch horizon is too short for grouping to separate. Null result
  for *this* horizon, **not** a design flaw; the recommendation becomes a longer-horizon
  re-run at a couple of group counts rather than declaring grouping inert.
- **Seed noise drowns the effect.** The two repeat points (g1, g16) gauge it: if the
  seed half-range at those points is comparable to the across-grouping spread, the screen
  is underpowered → escalate the winner (or the whole curve) to ≥3 seeds before any claim.
- **Saturation at the coarse end (g=128, 1 global signal underperforms).** Expected and
  informative, not a flaw — it bounds the useful end of the grouping axis.

## 7. Metrics Requested

The screen is fully evaluable with metrics already logged in `src/` (survival via
`Episode/Steps`; `modulator/temperature_{min,mean,max}`; γ/β group means and stds;
`loss/grad_norm`, `modulator/grad_norm`, `loss/entropy`, `loss/value`). No new metric is
*required* to run it. Two cheap scalars from the prior analysis ([[nmn_film_vs_plain_longL4]]
§6) would sharpen the temperature-rail and clipping diagnostics
(`modulator/temperature_frac_at_ceiling`, `modulator/update_clipped_frac`) but are **not
blocking** — if the user wants them, route via `feature-workflow` (`senior-developer` →
`developer`) before launch; otherwise the screen proceeds on existing metrics.

## 8. Related Docs

- [[nmn_film_vs_plain_longL4]] — the prior analysis that motivates this screen (per-neuron
  FiLM led early, crashed late, temperature railed at 5.0).
- [[basic_curriculum_convergence]] — source of the from-scratch far-sight baseline
  (~261 survival, run `c8cd77ft`) and per-stage budgets.
- [[basic_curriculum]] — study overview.

## 9. Results

_Left blank — filled after training by `experiment-designer` or `senior-developer`._

## 10. Conclusions

_Left blank — filled after training._
