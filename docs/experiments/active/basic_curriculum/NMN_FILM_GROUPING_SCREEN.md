---
title: "NMN FiLM grouping_size screen — how many modulation groups regularise the neuromodulator?"
topic: basic_curriculum
status: active
created: 2026-06-27
last_updated: 2026-06-27T21:10:00
phase: 1
wandb_tag: "rppo_nmn_film_g*_curric_longL4_s42"
develop_link:
---

# NMN FiLM grouping_size screen

> **Task changed (2026-06-27): from-scratch single-world → continual curriculum.**
> An earlier version of this screen trained the eight grouping variants **from scratch on
> one hard world** (the far-sight 10×10 predator). It was pre-registered and briefly
> launched (8 runs, WandB names `rppo_nmn_film_g*_screen_s42`), then **superseded before
> completion**: the payoff the neuromodulator is supposed to provide is in **continual
> learning** (carrying skill across a sequence of changing worlds), not in mastering a
> single fixed world, so a single-world screen tests the modulator on the wrong axis. The
> **live experiment** trains the *same* eight grouping configs on the **basic curriculum**
> — the five-stage continual schedule, long-L4 variant — instead. The superseded
> single-world manifest is preserved in §3.2 for the record; everything else in this doc
> describes the curriculum version.

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
on the **basic curriculum** — a fixed five-stage sequence of progressively harder worlds
(static predator → slow predator → fast predator → predator-plus-prey → far-sight
predator), run as one continual training pass under the "long-L4" schedule that lingers
on the final far-sight stage — across a ladder of group counts, from 128 groups
(per-neuron, the known-bad corner) down to 1 group (a single global modulation signal),
and ask **which group count gives the best, most stable survival on the final
far-sight stage**. The curriculum (rather than a single world) is the right setting
because a neuromodulator's hypothesised value is **carrying and re-using skill as the
world changes** — exactly what a continual schedule exercises. It is a first-pass
localisation, not a final confirmation; the winning group count earns a longer,
multi-seed follow-up.

One confound is removed up front. The prior crash coincided with the temperature knob
hitting its ceiling of 5.0, so a naive group-count sweep would mix up "how many groups"
with "how much temperature headroom". We therefore **raise the temperature ceiling to a
non-binding 10.0 for all eight runs**, so group count is the only modulation-capacity
knob that changes inside the sweep.

**Formal hypothesis.**

> **H₀** (null — *group count is inert*): survival steps on the final far-sight curriculum
> stage do not depend on `grouping_size`; coarsening the modulation does not change
> end-of-training survival relative to the per-neuron setting.
>
> **H₁a** (interior optimum — *coarsening helps, up to a point*): an **intermediate**
> group count produces higher steady-state survival than both the per-neuron extreme
> (`grouping_size: 1`, 128 groups) and the global extreme (`grouping_size: 128`,
> 1 group), and beats the unmodulated **continual** baseline (the plain curriculum run) by
> a meaningful margin without a late crash.
>
> **H₁b** (monotone regularisation — *coarser is simply better*): survival increases
> monotonically as grouping coarsens (group count falls), with the global single-group
> setting best.

"Steady-state survival" = mean survival steps over the **last 1M episodes**. "A late
crash" is defined numerically in §5.

## 2. Experimental Design

### 2.0 Why this design (decisions)

The screen varies one axis (`grouping_size`) over its full range at a single seed, on the
continual curriculum, against the two prior continual baselines.

1. **Task / budget — basic curriculum, long-L4 schedule, one continual pass.** Every run
   trains through the **five-stage basic curriculum** in order (static → slow → fast →
   predator+prey → far-sight), as one continual pass at `num_envs=128`, seed 42, driven by
   the long-L4 schedule (`configs/continual/basic_curriculum_schedule_longL4.yaml`). The
   schedule, **not** an `--episodes` flag, sets each stage's episode budget; stages 0–3 are
   short warm-ups and the final far-sight stage (L4) runs long (the "long-L4" lingering
   that gives the modulator time to express itself on the hardest world). This is the
   setting where the prior per-neuron crash actually appeared (~34M episodes into a
   curriculum run), so unlike a single-world screen it **can** reach the crash regime. The
   headline metric is survival steps on the **final far-sight stage**.
2. **De-confound the temperature rail.** All eight runs hold `temp_clip = [0.5, 10.0]`
   (non-binding; the architecture's own default upper bound). `grouping_size` is then
   the only modulation-capacity knob varying inside the sweep. No slot is spent
   re-testing the old 5.0 ceiling — the prior analysis already characterised it as
   railed, and mixing ceilings back in would reintroduce the very confound we are
   removing. The comparison to the 5.0-ceiling regime is "external", via the prior doc.
3. **Grid — full single-seed curve over the group axis.**
   `grouping_size ∈ {1, 2, 4, 8, 16, 32, 64, 128}` at seed 42 (8 runs) gives an
   eight-point curve over the whole capacity range — enough resolution to see the
   *shape* (interior optimum vs monotone vs flat). The eight runs fill the eight available
   GPU slots (nodes 106/108/109/110, two GPUs each). As a first-pass screen at one seed,
   magnitudes are provisional; the winning group count earns a ≥3-seed confirmation
   before any effect is called real.
4. **Keep grouping the dominant axis.** No second knob (e.g. `mod_hidden_size`) is
   swept; all eight slots go to the grouping curve, judged higher-value than a second axis
   for a first-pass screen whose explicit purpose is localising the group-count region.

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `modulation.grouping_size` (primary) | 1, 2, 4, 8, 16, 32, 64, 128 | Neurons per FiLM group. `num_groups = ceil(128 / grouping_size)` = 128, 64, 32, 16, 8, 4, 2, 1. Sets modulation capacity / regularisation strength. |
| Seed | 42 (all 8 curve points) | Single seed for this first-pass screen; the winning group count earns a ≥3-seed confirmation. |

`num_groups` per value (load-bearing — this is what actually changes inside the model):

| `grouping_size` | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| `num_groups` (= ceil(128/g)) | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| interpretation | per-neuron (known-bad) | | | | mid candidate | | | global (1 signal) |

### 2.2 Controlled Variables

Held constant across all eight runs:

- **Task** — the basic curriculum via
  `--configs-dir configs/environment/experiment/basic_curriculum` (the frozen 5-stage
  curriculum source: stages `00`–`04`, byte-identical copies of the `basic/` originals,
  deliberately excluding the standalone `05-random_init` stage) plus
  `--continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml`. Same
  schedule, same five stages, same order for every run.
- **Launch flags** — `--num-envs 128`, `--seed 42`, `--log-interval 10`. **No
  `--episodes`** — the continual schedule drives each stage's budget (and checkpoint
  frequencies). `environment.max_steps=500` (inherited from `environment/default`).
- **Temperature ceiling** — `temp_clip = [0.5, 10.0]` for all eight (the de-confound).
- **All PPO / encoding / hierarchical hyperparameters** — byte-identical to the base
  FiLM agent config (`recurrent_ppo_nmn_film_g1_tempceil5.yaml`): `lr_actor 5e-4`,
  `lr_critic 1e-4`, `gamma 0.95`, `K_epochs 4`, `eps_clip 0.1`, `sequence_length 128`,
  `hidden_size 128`, `entropy_coef 0.01`, `gae_lambda 0.95`, `vf_coef 0.5`,
  `max_grad_norm 0.5`, `use_layer_norm true`, `rnn_type GRU`, `activation relu`,
  `return_mode MC`, `encoding_mode hierarchical`, and the rest of the `modulation:`
  block (`type FiLM`, `mod_hidden_size 16`, `percept_bias_init 3.0`,
  `memory_bias_init 0.0`, `memory_clip [-2.0, 2.0]`).
- **Comparison baselines** — the two prior continual runs on the *same* long-L4
  curriculum: the **unmodulated (plain) recurrent agent**, `rppo_basic_curriculum_longL4`,
  and the **per-neuron FiLM agent** (`grouping_size: 1`), `rppo_nmn_film_curric_longL4_n114`.
  Neither is re-run; both are external anchors matched on schedule, num_envs, and seed.

**Obs/action/sensor parity.** Only the agent config's `modulation` block changes; the
task is the unmodified 5-stage curriculum. The five stage configs are byte-identical
copies of the `basic/` originals and all inherit the same `sensory`/`body` blocks from
`environment/default`, so observation dimension (**27**), action dimension (**6**), and
the sensor fingerprint are **identical across all eight runs, across all five stages, and
identical to the two continual baselines**. `grouping_size` only changes the modulator's
internal head output width (`num_groups`), never the obs/action interface. No env-side
parity risk is introduced.

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| **Single seed per curve point** | all 8 | **High** | This is a screen, not a confirmation. The 8-point curve reads *shape*; magnitudes at single-seed points are provisional. The winning group count is routed to a ≥3-seed follow-up before any effect is "confirmed". |
| **Continual confound — stage interactions** — survival on the final far-sight stage depends on what was carried from stages 0–3, not group count alone. | all | Med | All runs share the *identical* schedule and stage order, so the carried-skill pathway is held constant; only `grouping_size` varies. The plain and FiLM-g1 continual baselines bracket the modulated grid on the same schedule. |
| **Baselines are themselves single-seed** | the baseline comparison | Med | Margin threshold (§5) set to exceed plausible single-seed noise; curve points share seed 42 with the baselines for maximal comparability. |
| **Temperature may rail at 10.0 anyway** | any | Low | If `temperature_mean` pins at 10.0 even with coarse grouping, temperature headroom is a separate binding axis → noted as a follow-up sweep, not a screen failure. |
| **Coarse vs fine is also a parameter-count change** | across grid | Low | Inherent to the knob under test — fewer groups = fewer head parameters. This is the regularisation mechanism, not a nuisance; reported as such. |

## 3. Launch Manifest

System-of-record for every run. Designer fills planned columns; `training-runner` fills
actual columns (`Node`, `GPU`, `Launched at`, `WandB run ID`, `Log path`, `Status→running/completed`).
Slot layout: nodes 106, 108, 109, 110, two GPUs each (`:0`, `:1`) = 8 parallel slots, one run per GPU.

**Live (curriculum) manifest — 8 runs, all `planned`.** Tag = wandb-name on every row
(keeps `train.py`'s name fallback in sync). All tags unique. Scheme:
`rppo_nmn_film_g<grouping_size>_curric_longL4_s42`.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | g1 | `rppo_nmn_film_g1_curric_longL4_s42` | basic_curriculum | prod | 42 | 106 | cuda:0 | 2026-06-27T21:05:48 | srd59a4k | logs/20260627_210548_rppo_nmn_film_g1_curric_longL4_s42.log |
| 2 | running | g2 | `rppo_nmn_film_g2_curric_longL4_s42` | basic_curriculum | prod | 42 | 106 | cuda:1 | 2026-06-27T21:05:48 | gzinggvn | logs/20260627_210548_rppo_nmn_film_g2_curric_longL4_s42.log |
| 3 | running | g4 | `rppo_nmn_film_g4_curric_longL4_s42` | basic_curriculum | prod | 42 | 110 | cuda:0 | 2026-06-27T21:05:48 | zfrq2q4t | logs/20260627_210548_rppo_nmn_film_g4_curric_longL4_s42.log |
| 4 | running | g8 | `rppo_nmn_film_g8_curric_longL4_s42` | basic_curriculum | prod | 42 | 110 | cuda:1 | 2026-06-27T21:05:48 | 1zw6pyrw | logs/20260627_210548_rppo_nmn_film_g8_curric_longL4_s42.log |
| 5 | running | g16 | `rppo_nmn_film_g16_curric_longL4_s42` | basic_curriculum | prod | 42 | 108 | cuda:0 | 2026-06-27T21:05:48 | c5p3zv1s | logs/20260627_210548_rppo_nmn_film_g16_curric_longL4_s42.log |
| 6 | running | g32 | `rppo_nmn_film_g32_curric_longL4_s42` | basic_curriculum | prod | 42 | 108 | cuda:1 | 2026-06-27T21:05:48 | zv2qfe50 | logs/20260627_210548_rppo_nmn_film_g32_curric_longL4_s42.log |
| 7 | running | g64 | `rppo_nmn_film_g64_curric_longL4_s42` | basic_curriculum | prod | 42 | 109 | cuda:0 | 2026-06-27T21:05:48 | jf8gei6b | logs/20260627_210548_rppo_nmn_film_g64_curric_longL4_s42.log |
| 8 | running | g128 | `rppo_nmn_film_g128_curric_longL4_s42` | basic_curriculum | prod | 42 | 109 | cuda:1 | 2026-06-27T21:05:48 | vfqv04oq | logs/20260627_210548_rppo_nmn_film_g128_curric_longL4_s42.log |

### 3.1 Configs to Produce (designer-only, pre-launch)

Eight distinct agent configs (one per group count); a single shared task (the 5-stage
curriculum dir + long-L4 schedule). Seed is supplied at launch via `--seed 42`, not in any
config. The agent configs are unchanged by the task switch — only the launch invocation
(env dir + schedule, no `--episodes`) changed.

| Run | Task (env) | Continual schedule | Config (agent) | `--seed` |
|-----|-----------|--------------------|----------------|----------|
| 1 | `--configs-dir configs/environment/experiment/basic_curriculum` | `configs/continual/basic_curriculum_schedule_longL4.yaml` | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml` | 42 |
| 2 | (same) | (same) | `…/recurrent_ppo_nmn_film_g2_screen.yaml` | 42 |
| 3 | (same) | (same) | `…/recurrent_ppo_nmn_film_g4_screen.yaml` | 42 |
| 4 | (same) | (same) | `…/recurrent_ppo_nmn_film_g8_screen.yaml` | 42 |
| 5 | (same) | (same) | `…/recurrent_ppo_nmn_film_g16_screen.yaml` | 42 |
| 6 | (same) | (same) | `…/recurrent_ppo_nmn_film_g32_screen.yaml` | 42 |
| 7 | (same) | (same) | `…/recurrent_ppo_nmn_film_g64_screen.yaml` | 42 |
| 8 | (same) | (same) | `…/recurrent_ppo_nmn_film_g128_screen.yaml` | 42 |

All eight agent configs exist and are validated (parse cleanly; only the `modulation`
block — `grouping_size` — differs from the base FiLM config, temp ceiling 10.0). The
5-stage curriculum dir `configs/environment/experiment/basic_curriculum/` is the frozen
copy of `basic/` stages 00–04 (see that dir's `README.md`); it globs to exactly 5 stages,
matching the 5 `episode_boundaries` in the long-L4 schedule.

### 3.2 Superseded single-world manifest (preserved for record)

The earlier from-scratch single-world version of this screen (far-sight 10×10, no
curriculum, `--episodes 10000000`, temp ceiling 10.0). It was launched then superseded
before completion — these runs are **not** the live experiment; do not analyse them as
the grouping result. Scheme was `rppo_nmn_film_g<g>_screen_s<seed>`.

| Run | Status | Cell | Tag (= wandb-name) | Seed | Node | GPU | WandB run ID |
|-----|--------|------|--------------------|------|------|-----|--------------|
| 1 | superseded | g1 | `rppo_nmn_film_g1_screen_s42` | 42 | 106 | 0 | 7fj2z6wn |
| 2 | superseded | g2 | `rppo_nmn_film_g2_screen_s42` | 42 | 106 | 1 | ec0wlbor |
| 3 | superseded | g4 | `rppo_nmn_film_g4_screen_s42` | 42 | 110 | 0 | 6usynb6b |
| 4 | superseded | g8 | `rppo_nmn_film_g8_screen_s42` | 42 | 110 | 1 | rdpbbf0d |
| 5 | superseded | g16 | `rppo_nmn_film_g16_screen_s42` | 42 | 108 | 0 | el690g3l |
| 6 | superseded | g32 | `rppo_nmn_film_g32_screen_s42` | 42 | 108 | 1 | u7iuhb8u |
| 7 | superseded | g64 | `rppo_nmn_film_g64_screen_s42` | 42 | 109 | 0 | p0fc5tlw |
| 8 | superseded | g128 | `rppo_nmn_film_g128_screen_s42` | 42 | 109 | 1 | wwrskvg6 |

(The earlier plan also reserved two seed-43 repeats at g1 and g16; those were never
launched and are dropped in the curriculum version.)

### 3.3 Launch command template (curriculum)

Per run (the `training-runner` substitutes node, GPU, agent config, tag). Note:
`--configs-dir` + `--continual-schedule`, `--seed 42`, and **no `--episodes`** (the
schedule drives stage budgets and checkpoint frequencies):

```bash
./run_command.py <node> \
  "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
    --configs-dir configs/environment/experiment/basic_curriculum \
    --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
    --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g<G>_screen.yaml \
    --num-envs 128 --log-interval 10 --device cuda:<GPU> --seed 42 \
    --tag rppo_nmn_film_g<G>_curric_longL4_s42"
```

### 3.4 Compute estimate

8 runs, each one continual pass through the 5-stage long-L4 curriculum, on one GPU,
all in parallel across the 8 GPUs (nodes 106/108/109/110 × 2) → **one wall-clock pass**.
The two prior continual baselines (`rppo_basic_curriculum_longL4`, plain; and
`rppo_nmn_film_curric_longL4_n114`, FiLM-g1) on the *same* schedule are the s/it
reference; expect comparable per-run wall-clock. The FiLM modulator adds a small fixed
overhead (a 16-unit GRU + heads) that is independent of `grouping_size` to first order, so
all eight runs should finish within a similar window. The long-L4 schedule lingers on the
final far-sight stage, so total wall-clock is dominated by that stage.

## 4. Predicted Outcomes

All outcomes are read on **final-far-sight-stage survival**, against the two continual
baselines: the plain (unmodulated) curriculum run and the per-neuron FiLM-g1 curriculum
run.

- **Most likely (H₁a, interior optimum):** survival rises as grouping coarsens from
  per-neuron (128 groups), peaks at an intermediate group count (somewhere around 4–16
  groups, i.e. `grouping_size` 8–32), then falls toward the global single-group setting
  as the modulator loses the per-channel flexibility that helped early. The per-neuron end
  (g=1) reproduces the prior late crash; the global end is stable but bland.
- **Possible (H₁b, monotone):** survival simply increases as group count falls — coarser
  is uniformly safer — with the global setting best. Would suggest the early-transfer
  benefit of fine grouping does not survive the continual schedule.
- **Null (H₀, flat):** all group counts land within noise of each other and of the plain
  baseline. Would suggest grouping is not the binding constraint on this curriculum.

## 5. Analysis Plan (pre-specified)

**Primary statistic.** Steady-state survival = mean `Episode/Steps` over the **last 1M
episodes of the final far-sight stage** of each run, per `grouping_size`. Plotted as
survival vs `num_groups` (log x). Single seed (42) per point, so report point values; the
two continual baselines (plain `B_plain`, FiLM-g1 `B_g1`) overlay as reference lines.

**Effect-size threshold ("a grouping config wins").** A `grouping_size` value **beats the
plain baseline** if its steady-state far-sight survival exceeds the plain continual
baseline (`rppo_basic_curriculum_longL4`) by ≥ **5%** (a margin chosen to exceed plausible
single-seed noise). The per-neuron FiLM-g1 continual run
(`rppo_nmn_film_curric_longL4_n114`) is the within-architecture reference the coarser
settings must improve on.

**Stability threshold ("no late crash").** Over the **final far-sight stage**, the maximum
sustained drawdown from a run's own running peak survival (measured on the 50-point rolling
mean) must stay **< 30 steps**. (The prior per-neuron pathology was a ~95-step crash around
34M episodes; <30 = "no crash-class instability".)

**Temporal-evolution check (mandatory).** For every run, inspect the full survival
trajectory across the whole continual pass (all five stages, with attention to the
far-sight stage), not just the endpoint: per-stage convergence, monotone vs oscillating,
any discrete drop, and whether skill carries across stage boundaries. Diagnostic time
series tracked per run (all already logged by `src/`): `modulator/temperature_mean` (does
it rail at 10.0 or settle in [3,5)?), γ/β group means and **stds** (expected to shrink as
grouping coarsens), `loss/grad_norm` and `modulator/grad_norm` (spike frequency/magnitude),
`loss/entropy` (collapse?), `loss/value` (critic health).

**Cross-correlation (pre-specified).** If any run shows an in-window survival drawdown
≥ 30 steps, time-lock it against `temperature_mean` saturation and the γ-std trajectory
within a ±0.5M-episode window — the same signature reported in the prior analysis — to
test whether the per-neuron instability mechanism reproduces on this curriculum.

**Confirmation / refutation (pre-registered).**

- **Screen succeeds (motivates a follow-up):** at least one `grouping_size` value clears
  **both** thresholds — far-sight survival ≥ plain baseline +5% **and** late drawdown
  < 30 steps — AND it is a coarser setting than per-neuron (`grouping_size > 1`). Verdict:
  coarser grouping is a viable regulariser; route the best-clearing group count to a
  ≥3-seed confirmation.
- **Interior optimum confirmed (H₁a):** the winning group count beats *both* the
  per-neuron (g=1) and global (g=128) extremes by ≥ the +5% margin.
- **Refuted (grouping does not fix it):** **no** `grouping_size > 1` value clears both
  thresholds — i.e. either nothing beats plain +5%, or every config that does also shows a
  ≥30-step late drawdown. Verdict: coarsening alone does not recover a stable modulated
  benefit on this curriculum; attention shifts to other levers (temperature schedule,
  mod_hidden_size, explicit γ/β penalties) or to the conclusion that the modulator does not
  help here.
- **Inert axis (H₀):** survival flat across the whole grid (max−min < the +5% margin).
  Verdict: grouping is not the binding constraint on this curriculum; re-decide between a
  longer-horizon re-run and abandoning the grouping axis.

## 6. Failure-Mode Catalog (pre-decided)

- **NaN / value explosion in a run.** With all hyperparameters fixed and the same
  curriculum, a NaN at a given `grouping_size` **counts against that group count** (the
  architecture is unstable at that capacity) — it refutes that point, not "the run". Record
  the episode and the γ-std / grad-norm at failure. A NaN at g=1 specifically would be a
  positive finding (per-neuron instability reproduces on the curriculum).
- **Temperature pins at the 10.0 ceiling for all group counts.** Not a screen failure —
  it means temperature headroom is a *separate* binding axis. Report and propose a
  temperature-schedule follow-up; the grouping verdict still stands on survival + drawdown.
- **Everything ≈ plain baseline (flat).** Null result for *this* curriculum, **not** a
  design flaw; the recommendation becomes a longer-horizon re-run at a couple of group
  counts rather than declaring grouping inert.
- **Seed noise drowns the effect.** Single seed here is a known limitation (§2.3): the
  winner (or the whole curve) escalates to ≥3 seeds before any claim is called real.
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
  FiLM led early, crashed late, temperature railed at 5.0). Source of the two continual
  baselines: plain `rppo_basic_curriculum_longL4` and FiLM-g1 `rppo_nmn_film_curric_longL4_n114`.
- [[basic_curriculum_convergence]] — per-stage curriculum budgets and the from-scratch
  far-sight reference (~261 survival, run `c8cd77ft`, single-world — used only as an
  external sanity anchor now, not the live baseline).
- [[basic_curriculum]] — study overview.
- Task source: `configs/environment/experiment/basic_curriculum/` (frozen 5-stage
  curriculum dir — see its `README.md`) + `configs/continual/basic_curriculum_schedule_longL4.yaml`.

## 9. Results

_Left blank — filled after training by `experiment-designer` or `senior-developer`._

## 10. Conclusions

_Left blank — filled after training._
