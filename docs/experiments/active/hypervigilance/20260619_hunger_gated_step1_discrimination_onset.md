---
title: "Step 1 — discrimination-onset map (smell separation × per-episode noise)"
topic: hypervigilance
status: active
created: 2026-06-19
last_updated: 2026-06-19T02:55:00
phase: hunger_gated_avoidance
wandb_tag: hunger_gated
develop_link: "[[20260616_1557_hunger_gated_avoidance]]"
---

# Step 1 — discrimination-onset map (smell separation × per-episode noise)

> **Status**: COLLECTING (pre-registered — no runs launched yet)
> **Date**: 2026-06-19
> **Author**: experiment-designer
> **Related**: living plan [[20260616_1557_hunger_gated_avoidance]] (Step 1 + "Decisions locked 2026-06-19");
> base scene (archived) `configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`;
> config authoring [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md)

---

## 1. Research Question

**Plain-language framing.** We put one predator and one rabbit in the same little world. They
behave *identically* — same chasing, same speed, same retreat — so the only way to tell them apart
from a distance is by **smell**. We start with both smelling exactly the same (no distal clue at all)
and then, one step at a time, slide their smells apart. The question is: **how far apart do the two
smells have to be before the agent starts treating the dangerous one differently — keeping its
distance, fleeing, diving into a bush — *before* it ever gets bitten?** And in the runs where we also
make each animal's smell **jitter from episode to episode** (so the same smell is a predator one day
and a rabbit the next), does the agent's caution depend on **how hungry it is** — gambling and staying
to eat when starving, playing it safe and avoiding when full?

This is a **map**, not a single comparison. We lay down ten points on a 2-D grid:

- the horizontal axis is **separation `s`** — how far the predator's and rabbit's smells are slid
  apart (`s = 0` identical, `s = 0.5` completely different / orthogonal);
- the vertical axis is **per-episode noise σ** — how much each animal's smell randomly wobbles at the
  start of every episode (`σ = 0` fixed, `σ > 0` irreducibly ambiguous).

The two known endpoints are already established by earlier work: at `s = 0, σ = 0` (matched smell, no
wobble) a trained agent **cannot** tell them apart and does not avoid the rabbit; far-separated noisy
smell **can** be discriminated but only under confounds the June work exposed (a 2-rabbit layout and a
geometry artifact). Step 1 fills the clean space in between, in the simple 1-predator-vs-1-rabbit
world, so we can see *where the transition happens*.

**Formal hypotheses.**

> **H₀ (null — no distal discrimination):** Across the grid, the trained agent's *pre-contact*
> behaviour toward the predator is statistically indistinguishable from its behaviour toward the
> rabbit — no separation `s` is large enough (at any noise σ) to open a predator-vs-rabbit gap in
> closest-approach distance, flee rate, or bush-dive rate before contact.
>
> **H₁a (discrimination onset):** There exists a threshold separation `s*` above which the agent
> holds the predator at a greater distance / flees it / bush-dives away from it more than the rabbit,
> *before contact*; the gap is ≈ flat below `s*` and rises above it. Raising the per-episode noise σ
> shifts `s*` to the right (more separation needed when smell is noisier).
>
> **H₁b (hunger-gating, σ > 0 runs only):** In the noisy runs — where the smell is *irreducibly*
> ambiguous — the agent's avoidance of the ambiguous animal **depends on hunger**: when it starts
> satiated it avoids more (greater closest-approach distance / higher flee rate), and when it starts
> hungry it takes the risk and stays to eat (smaller distance / lower flee rate). The split appears in
> the σ > 0 runs and is absent (or much weaker) in the σ = 0 runs, where the smell→class mapping is
> fixed and learnable without reference to internal state.

A reader of this section needs no other document to know what is being asked and what counts as a
positive (H₁a / H₁b confirmed) vs. negative (H₀ retained) result. The symbolic / path-shaped detail
lives in §2–§5.

---

## Results & Verdict (filled 2026-06-20)

> **Headline (plain language).** Across the whole sweep the agent **never learned to keep
> meaningfully more distance from the harmful predator than from the byte-identical harmless
> rabbit.** A *small but statistically real* avoidance tilt — the agent lets the rabbit drift a
> little closer and flees the predator a little more, about **0.1–0.2 of a cell** — switches on the
> moment the two smells differ at all (even the smallest separation, `s = 0.05`), but it **never
> grows to the pre-registered "real discrimination" bar of half a cell** at any separation or noise
> level. So the *discrimination-onset* hypothesis is **refuted at threshold** (there's a signal, but
> it's far too weak to count), and there is **no clean onset `s*`** — the tilt is a low flat plateau,
> not a rising curve. Hunger did **not** gate avoidance either (the satiated-vs-hungry difference is
> ~0 or slightly *negative*), so the *hunger-gating* hypothesis is **refuted** — **but** the predator
> is lethal and the agent **dies in ~40–70 % of episodes**, so per our pre-registered watch-out the
> honest reading is **"the lethal predator likely prevented a hunger-gate from ever being
> reinforced," not "no hunger-gate exists."** The matched-smell anchor (`hg01`) is a **clean null**
> (gap ≈ 0, CI includes 0), so the small gaps elsewhere are a genuine *olfactory* effect, not the
> vision-count leak. Mechanistically the agent survives by **reactive tank-and-hide** (bush-diving
> and healing from injury), **not** by keeping its distance — which is exactly why the pre-contact
> distance gaps stay tiny even when the smells are fully distinct. **Single seed (42) → provisional.**

### The (s × σ) discrimination surface (final checkpoint, 200 eval episodes each)

`ca_gap` = closest-approach gap (predator − rabbit, cells; **positive = predator held farther**);
`gating_gap` = satiated − hungry closest-approach (σ>0 runs). H₁a bar = ca_gap ≥ 0.50 with CI
excluding 0 **and** flee_gap > 0, ≥2 of 3 measures agreeing.

| run | s | σ | survival | death | ca_gap (95% CI) | flee_gap | bdive_gap | H₁a | gating_gap (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| hg01 | 0.0 | 0.0 | 276 | 0.66 | +0.01 [-0.12, +0.13] | +0.003 | +0.019 | ✗ | — |
| hg02 | 0.05 | 0.0 | 325 | 0.48 | +0.15 [+0.03, +0.26] | +0.014 | +0.039 | ✗ | — |
| hg03 | 0.1 | 0.0 | 353 | 0.39 | +0.14 [+0.05, +0.22] | +0.042 | +0.058 | ✗ | — |
| hg04 | 0.25 | 0.0 | 362 | 0.40 | +0.13 [+0.02, +0.22] | +0.047 | +0.069 | ✗ | — |
| hg05 | 0.5 | 0.0 | 349 | 0.40 | +0.08 [-0.03, +0.18] | +0.061 | +0.050 | ✗ | — |
| hg06 | 0.1 | 0.2 | 298 | 0.56 | +0.14 [+0.03, +0.23] | -0.002 | +0.004 | ✗ | -0.15 [-0.26, -0.04] |
| hg07 | 0.25 | 0.2 | 308 | 0.54 | +0.20 [+0.09, +0.30] | +0.014 | +0.048 | ✗ | -0.00 [-0.10, +0.10] |
| hg08 | 0.5 | 0.2 | 333 | 0.47 | +0.21 [+0.12, +0.29] | +0.047 | +0.089 | ✗ | -0.02 [-0.11, +0.08] |
| hg09 | 0.1 | 0.4 | 242 | 0.69 | +0.21 [+0.09, +0.32] | -0.017 | +0.035 | ✗ | -0.08 [-0.22, +0.07] |
| hg10 | 0.5 | 0.4 | 315 | 0.54 | +0.16 [+0.05, +0.25] | +0.018 | +0.037 | ✗ | +0.02 [-0.08, +0.12] |

### Verdict against the pre-registered criteria

- **H₁a — discrimination onset: REFUTED at threshold.** No run reaches the 0.50-cell bar (max is
  `hg08` at +0.21). The gap *is* CI-positive on every separated run (a real, weak olfactory tilt that
  turns on at `s = 0.05` and never climbs), but it is 2–6× below the behaviourally-meaningful
  threshold. No `s*` exists on any σ row.
- **Anchor check — PASS (clean null).** `hg01` (no smell signal) shows ca_gap +0.01 [−0.12, +0.13],
  flee +0.003, bdive +0.019 — all CIs include 0. **No vision-count-elimination leak**, so the map is
  read against 0, and the small separated-run gaps are genuine smell-driven effects.
- **H₁b — hunger-gating: REFUTED, but "lethality masks gating" applies.** The satiated−hungry gap is
  ≈0 or *negative* on all σ>0 runs (`hg06` is −0.15 [−0.26, −0.04] — the *wrong* direction). With
  death rates of 47–69 % on those runs and the agent dying while pinned at distance 1, this triggers
  the pre-registered **"lethal [5,120] prevented the gate from being reinforced"** verdict — **not**
  "no hunger-gate exists." Revisit lethality before any "gating absent" claim.

### Mechanism (trajectory cross-check — means confirmed)

Visual channels are **contact-only** (`visual_sensor_range: 0`), so the agent's *only distal* class
cue is the olfactory gradient. At the anchor the two smell channels are byte-identical (nothing to
discriminate); at `s = 0.25/0.5` they separate and the agent uses the signal **weakly**. In the step
dumps the predator orbits at distance 1 for long stretches while the agent eats, rests, and dives
into bushes (injury oscillates 0→35→0) — it survives by **tanking and hiding**, never by holding a
standoff. This is why even fully-separable smells produce only a ~0.2-cell gap.

### Limitations & next step

Single seed → **provisional** (multi-seed hardening required before any onset/null is believed).
The clearest follow-up implied by the data: **a sub-lethal predator** (e.g. `[15,45]`) so a
hungry agent can *survive* taking a risk and a hunger-gate has a chance to be reinforced — directly
testing whether the H₁b null here is real or a lethality artifact. The contact-only visual channel
also means olfaction is the sole distal cue; if a stronger/learnable distal cue is wanted, that is a
separate design lever. Raw per-run numbers: `tmp/20260620_164156_hunger_gated_step1_results.json`
(analysis run via `experiment-analyzer`, eval-rollouts on the final 10M checkpoints).

## 2. Experimental Design

### 2.1 Independent Variables

Two olfactory knobs, crossed into a 10-point (`s`, σ) grid. The smell is **symmetric**: both animals
sit on olfactory channels 2 & 3, mirrored around 0.5, and slid apart by separation `s` — predator
`[0, 0.5+s, 0.5−s, 0, 0]`, rabbit `[0, 0.5−s, 0.5+s, 0, 0]`. `σ` (= `properties_std`) is applied to
channels 2 & 3 of **both** animals (`[0, σ, σ, 0, 0]`). The symmetric scheme is deliberate: it carries
**no presence/absence giveaway** (a pinned-predator scheme would turn channel 3 into a pure
"rabbit-present" flag). At `s = 0, σ = 0` both animals smell `[0, 0.5, 0.5, 0, 0]` — the matched anchor.

| Variable | Values | Rationale |
|----------|--------|-----------|
| separation `s` | 0, 0.05, 0.1, 0.25, 0.5 | dense along the σ = 0 row to locate the learnability onset cleanly; 0.5 makes the two smells orthogonal |
| per-episode noise σ (`properties_std`) | 0, 0.2, 0.4 | 0 = fixed (reducible) mapping; 0.2 mild, 0.4 strong irreducible ambiguity — the regime where hunger-gating should live |

The 10 grid points (one sparse config each):

| Run | Cell | `s` | σ | predator smell (ch2,ch3) | rabbit smell (ch2,ch3) | d′ (approx) | Role |
|-----|------|-----|---|--------------------------|------------------------|-------------|------|
| 1 | `s0_sig0` | 0 | 0 | 0.5 / 0.5 | 0.5 / 0.5 | 0 | **matched anchor (fresh baseline)** |
| 2 | `s0.05_sig0` | 0.05 | 0 | 0.55 / 0.45 | 0.45 / 0.55 | ∞ (reducible) | onset along `s` |
| 3 | `s0.1_sig0` | 0.1 | 0 | 0.6 / 0.4 | 0.4 / 0.6 | ∞ | onset along `s` |
| 4 | `s0.25_sig0` | 0.25 | 0 | 0.75 / 0.25 | 0.25 / 0.75 | ∞ | onset along `s` |
| 5 | `s0.5_sig0` | 0.5 | 0 | 1.0 / 0.0 | 0.0 / 1.0 | ∞ | orthogonal anchor (top of `s`) |
| 6 | `s0.1_sig0.2` | 0.1 | 0.2 | 0.6 / 0.4 | 0.4 / 0.6 | ≈ 1.4 | onset under mild noise |
| 7 | `s0.25_sig0.2` | 0.25 | 0.2 | 0.75 / 0.25 | 0.25 / 0.75 | ≈ 3.5 | onset under mild noise |
| 8 | `s0.5_sig0.2` | 0.5 | 0.2 | 1.0 / 0.0 | 0.0 / 1.0 | ≈ 7.1 | onset under mild noise |
| 9 | `s0.1_sig0.4` | 0.1 | 0.4 | 0.6 / 0.4 | 0.4 / 0.6 | ≈ 0.7 | onset under strong noise |
| 10 | `s0.5_sig0.4` | 0.5 | 0.4 | 1.0 / 0.0 | 0.0 / 1.0 | ≈ 3.5 | onset under strong noise |

d′ is the per-channel separation-to-noise ratio `2s / σ` (a sensitivity index; ∞ at σ = 0 because a
fixed mapping is perfectly learnable). Smell is drawn **once per episode** at reset and held fixed for
the whole episode, then clipped to `[0,1]`; at `s = 0.5` the means sit on the clip edges so σ jitter is
half-rectified — an acceptable, documented edge effect (the onset is read off the lower-`s` rungs).

### 2.2 Controlled Variables

Everything except the two smell knobs and the (intentionally randomised) initial internal state is
**pinned to the cell-08 scene**, reproduced as a sparse `extends: environment/default` override.
Verified against `config_loader.py` that all 10 configs load and reset correctly.

```yaml
# Held constant across all 10 runs (reproduced from cell-08):
environment:
  height: 10; width: 10; max_steps: 500
  resources: 8 food sources (4 quadrants × 2), [1,0,0,0,0] smell, no damage
  entities:
    - predator: class predator (is_damaging), behaviour hunt, count 1, damage [5,120] (LETHAL),
                nociception 0.9, full-grid spawn+patrol, disengage_on_contact true,
                detection_range 10, max_stamina 60, recovery 1.0, hunt_threshold 0.3, lose_interest 3.0
    - rabbit:   class neutral (harmless), behaviour hunt, count 1, damage [0,0], nociception 0.0
                — BYTE-IDENTICAL chase profile to the predator (only class/damage/smell differ)
    - hiding-predator slot: count 0 (inert schema template, zero instances)
  obstacles: 12 rocks (4×3) + 12 bushes (4×3, hides_agent true) + inert tree count 0
  visual_properties: class defaults (predator ch5, rabbit ch7, food ch3, rock/bush ch6) — NOT varied
body:
  random_start_nutrition: true,  start_nutrition_low: 10,  start_nutrition_high: 100
  random_start_injury:   true,   start_injury_low: 0,      start_injury_high: 80
  random_start_satiation: false  (satiation is DERIVED from nutrition: S = N at scaling=1 — no-op)
  metabolic_cost 1.0, food_nutrition_gain 6, death at injury >= 100, death_penalty 100
sensory / perceptual_noise / behavior_measures:  all inherited from default.yaml unchanged
                                                 (perceptual_noise.enabled: false — Step 2 adds it)
# Training: agent configs/models/recurrent_ppo.yaml, fresh-init (no warm-start),
#           single seed 42, ~10M episodes, num-envs 128, checkpoint-frequency 100k
```

**Initial-state bounds — reasoning (designer-chosen, env-config-auditor to confirm).**
Both `random_start_*` flags are **on** (locked decision), so the four range keys are
conditional-mandatory and present.

- **Nutrition `[10, 100]`.** `start_nutrition_high = 100 = max_nutrition` (full) down to a genuinely
  **hungry** floor of 10. Nutrition decays by `metabolic_cost = 1.0` per step, so a start of 10 gives
  ~10 steps of buffer — hungry and pressured, but not an instant-starvation artifact that would add
  reset noise without testing the hunger gate. Floor kept strictly above 0 for the same reason.
  Satiation is **derived** from nutrition at reset (`S = max_S·(N/max_N)^1 = N` here, verified in
  `core.py:jax_reset`), so randomising nutrition *is* the satiation contrast — no separate
  food-scarcity manipulation needed (matches the living plan's "satiation contrast" decision).
- **Injury `[0, 80]`.** Death fires at `injury ≥ max_injury = 100` (verified, `core.py`). A start at or
  above 100 is instant death, so `start_injury_high` must be **strictly below 100**; 80 spans
  healthy → badly-injured with a margin and never self-triggers death at reset. (A single lethal
  predator hit `[5,120]` can still kill from any start — that is intended.)

Reset draws were sampled across seeds to confirm: nutrition ∈ [10,100], injury ∈ [0,80], satiation ==
nutrition.

### 2.3 Confounds & Limitations

| Confound / limitation | Affected runs | Severity | Mitigation |
|-----------------------|---------------|----------|------------|
| **Single seed (42) per point** — cannot separate a real onset from seed-specific training noise | all 10 | High | first-wave map only; any onset flagged here is **provisional** and gets multi-seed hardening before it is believed. Read trajectories, not just means. |
| **Lethal predator `[5,120]` may prevent gating** — risk-taking while hungry is often fatal, so hunger-gated *staying* may never get reinforced | σ > 0 (6–10), worst on 8 & 10 | High | pre-registered watch-out (below): if H₁b fails *and* hungry-stay episodes die at high rate, the verdict is "lethality masks gating", not "no gating" → revisit lethality before concluding. |
| **Vision-count (1-vs-1) leak** — vision reports a per-class on-cell count; with 1 predator + 1 rabbit the agent can sometimes infer class by elimination, bypassing smell | all 10, strongest at anchor | Medium | confirm at the anchor (run 1): if the anchor already shows a pre-contact gap *with no smell signal*, the gap is a count leak, not discrimination — discount it across the map. |
| **σ clip half-rectification at s = 0.5** — means on the [0,1] clip edge, so σ jitter is one-sided | 5, 8, 10 | Low | onset is read off the lower-`s` rungs (3, 4, 6, 7, 9); the s = 0.5 runs are orthogonal anchors, not onset points. |
| **Geometry / spatial-encounter artifact** — bush-dive gap can track world geometry, not the agent | all 10 | Medium | class-blind + geometry control at EVAL (see §5) on the anchor + a mid rung. |

---

## 3. Launch Manifest

System-of-record for all 10 runs. Designer fills the planned columns; `training-runner` fills
Node / GPU / Launched at / WandB run ID / Log path at launch. No runs launched yet.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | s0_sig0 | `rppo_hg01_s0_sig0_s42` | hunger_gated | prod | 42 | 106 | cuda:0 | 2026-06-19T02:47:23 | lldqf9fh | logs/20260619_024723.log |
| 2 | running | s0.05_sig0 | `rppo_hg02_s0.05_sig0_s42` | hunger_gated | prod | 42 | 106 | cuda:1 | 2026-06-19T02:47:24 | jxuj38si | logs/20260619_024724.log |
| 3 | FAILED — node 107 NAS down; needs re-launch | s0.1_sig0 | `rppo_hg03_s0.1_sig0_s42` | hunger_gated | prod | 42 | — | — | — | — | — |
| 4 | FAILED — node 107 NAS down; needs re-launch | s0.25_sig0 | `rppo_hg04_s0.25_sig0_s42` | hunger_gated | prod | 42 | — | — | — | — | — |
| 5 | running (log shared with run 6 — see note) | s0.5_sig0 | `rppo_hg05_s0.5_sig0_s42` | hunger_gated | prod | 42 | 108 | cuda:0 | 2026-06-19T02:47:27 | (see note) | logs/20260619_024727.log (shared) |
| 6 | running (log shared with run 5 — see note) | s0.1_sig0.2 | `rppo_hg06_s0.1_sig0.2_s42` | hunger_gated | prod | 42 | 108 | cuda:1 | 2026-06-19T02:47:27 | 3c6k9szw | logs/20260619_024727.log (shared) |
| 7 | running | s0.25_sig0.2 | `rppo_hg07_s0.25_sig0.2_s42` | hunger_gated | prod | 42 | 110 | cuda:0 | 2026-06-19T02:47:28 | u59peb7n | logs/20260619_024728.log |
| 8 | running | s0.5_sig0.2 | `rppo_hg08_s0.5_sig0.2_s42` | hunger_gated | prod | 42 | 110 | cuda:1 | 2026-06-19T02:47:29 | 0vpvquax | logs/20260619_024729.log |
| 9 | running | s0.1_sig0.4 | `rppo_hg09_s0.1_sig0.4_s42` | hunger_gated | prod | 42 | 103 | cuda:0 | 2026-06-19T02:47:30 | pm1it6tg | logs/20260619_024730.log |
| 10 | running | s0.5_sig0.4 | `rppo_hg10_s0.5_sig0.4_s42` | hunger_gated | prod | 42 | 103 | cuda:1 | 2026-06-19T02:47:31 | wet6k6vp | logs/20260619_024731.log |

Tags are unique, parseable (`rppo_hg<NN>_s<sep>_sig<sigma>_s<seed>`), and identical to the
wandb-name. All rows share group `hunger_gated`.

### 3.1 Configs to Produce (designer-only, pre-launch)

All 10 environment configs vary only smell + (identical) initial-state block; no model hyperparameter
is under test, so every run uses the **same** agent config (`configs/models/recurrent_ppo.yaml`).

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/environment/experiment/hunger_gated/01-s0_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/environment/experiment/hunger_gated/02-s0.05_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 3 | `configs/environment/experiment/hunger_gated/03-s0.1_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 4 | `configs/environment/experiment/hunger_gated/04-s0.25_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 5 | `configs/environment/experiment/hunger_gated/05-s0.5_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 6 | `configs/environment/experiment/hunger_gated/06-s0.1_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 7 | `configs/environment/experiment/hunger_gated/07-s0.25_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 8 | `configs/environment/experiment/hunger_gated/08-s0.5_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 9 | `configs/environment/experiment/hunger_gated/09-s0.1_sig0.4.yaml` | `configs/models/recurrent_ppo.yaml` |
| 10 | `configs/environment/experiment/hunger_gated/10-s0.5_sig0.4.yaml` | `configs/models/recurrent_ppo.yaml` |

---

## 4. Predicted Outcomes (pre-registered)

- **H₁a (discrimination onset).** The pre-contact predator-vs-rabbit gap is ≈ flat near the anchor
  (runs 1–2) and **rises** once separation crosses a threshold `s*` on the σ = 0 row (somewhere in
  runs 3–5). On the noisy rows the same gap appears but the rise is **shifted right** (needs larger
  `s`) and/or shallower — i.e. `s*(σ=0.4) > s*(σ=0.2) > s*(σ=0)`. A flat surface everywhere retains H₀.
- **H₁b (hunger-gating).** Only on the σ > 0 runs (6–10): split the same measures by **starting
  satiation** (high vs. low, the randomised nutrition start). Predicted **interaction** — satiated
  starts avoid the ambiguous animal more, hungry starts stay/eat more. A null interaction (no
  hunger split) on runs where discrimination *is* present (H₁a holds) is the informative negative for
  H₁b.

---

## 5. Analysis Plan (pre-specified)

**Primary dependent variable (survival-step framing).** Headline performance is **survival steps**
per run across training; reward is a secondary diagnostic only. The *behavioural* read-out is the
**pre-contact predator-vs-rabbit gap**, measured three ways, each as predator-minus-rabbit:

1. **Closest-approach distance** — minimum agent-to-animal distance before first contact (predator
   held *farther* ⇒ positive gap).
2. **Flee rate** — fraction of pre-contact encounters where the agent increases distance within the
   K-step observation window.
3. **Bush-dive rate** — fraction of pre-contact encounters ending in a hides_agent bush cell (the M2
   defensive event from the toolkit).

**Statistics.** Per run, report each gap as mean ± 95% CI over the 200 eval episodes (fixed
`eval_seeds`, deterministic policy). With a single training seed the CI is **within-run** (episode
sampling) only — it does **not** license a cross-seed claim; the onset surface is provisional.

**Effect-size threshold for "discrimination onset".** A run counts as discriminating if the
predator-vs-rabbit closest-approach gap is **≥ 0.5 cell** with a 95% CI excluding 0 **and** the
flee-rate gap is positive — at least two of the three measures must agree (event-level, per the
carry-forward "event not mean" rule). `s*` is the smallest `s` (per σ row) meeting this.

**Hunger-gating test (σ > 0 runs).** Within each noisy run, split eval episodes at the **median
starting nutrition** into hungry vs. satiated halves; report the avoidance measures per half. H₁b
confirmed if the satiated-minus-hungry difference in closest-approach distance is **≥ 0.5 cell** with
a 95% CI excluding 0 (more avoidance when satiated), present on σ > 0 and absent/weaker on σ = 0.

**Temporal evolution (mandatory).** Track survival steps and the predator-vs-rabbit gap **across
training checkpoints** (every 100k episodes), not just at the end — the onset may be a convergence
phenomenon (gap emerges late) and a final-snapshot-only read would miss a slow-forming gate.

**Controls — run at EVAL, not training (cheap, no extra training runs).**
- **Class-blind control.** Re-evaluate a trained checkpoint with the visual class channel masked (or a
  known class-blind policy) on the anchor (run 1) + a mid rung (run 4 or 7). Any pre-contact gap the
  class-blind agent reproduces is a **geometry/encounter artifact**, subtracted from the real gap.
- **Geometry control.** Compare the gap against a spatial-encounter null on the same scene (the June
  spatial-encounter check) so a bush-dive gap that merely tracks world layout is discounted.

---

## 6. Failure-Mode Catalog (pre-decided)

- **Training instability (NaN / value explosion).** Refutes the **run**, not a hypothesis. Re-launch
  the affected grid point (same seed) once; if it recurs, flag the point as "no data" rather than
  reading it as H₀.
- **Lethality masks gating.** If H₁b fails on the σ > 0 runs **and** episodes with hungry starts show
  a high death rate when they stay near the ambiguous animal, the verdict is **"lethal `[5,120]`
  prevented the gate from being reinforced"**, not "no hunger-gating exists". This is the living
  plan's locked watch-out → revisit predator lethality (e.g. a sub-lethal follow-up) before any
  "gating absent" claim.
- **Anchor already shows a gap (count leak).** If run 1 (`s = 0, σ = 0`, no smell signal) shows a
  pre-contact predator-vs-rabbit gap, the vision-count elimination leak is active; the whole map's
  gaps are interpreted **relative to the anchor's** gap, not against zero.
- **Insufficient horizon.** If the gap is still rising at the end of training (temporal-evolution
  curve not plateaued), the onset estimate is a lower bound — extend that point rather than reading a
  null. Do not declare a flat surface until curves have plateaued.
- **Saturation at `s = 0.5`.** The orthogonal runs (5, 8, 10) sit on the smell clip edge; treat them
  as anchors for "fully separable", not as onset points — a ceiling there is by design, not a result.

---

## 7. Metrics Requested

None new are required for the **primary** read-out: closest-approach distance, flee rate, and bush
occupancy are already produced by the behaviour-measure toolkit (M1/M2 online + the M7 eval-rollout
protocol, both enabled in the inherited `behavior_measures` block), and survival steps + per-checkpoint
logging already exist.

One **convenience** request, optional and non-blocking: the **starting nutrition (or satiation) per
eval episode** logged alongside each rollout so the §5 hunger split can be computed without
re-deriving it from the reset RNG. Cheap (one scalar per episode), would live in the eval-rollout
recorder in `src/` (eval-rollout writer). If not added, the split is still computable offline by
replaying the fixed `eval_seeds` resets — so this does **not** gate the launch.

---

## 8. Handoff / Next Steps

1. **env-config-auditor** pre-flight on the 10 configs (obs↔noise width, smell symmetry, init-range
   bounds, list-replace completeness).
2. **PI consult** (multi-run launch decision) per the agent playbook.
3. **training-runner** launches the wave with user-supplied node + GPU. Suggested loop:
   ```bash
   for f in configs/environment/experiment/hunger_gated/*.yaml; do
     stem=$(basename "$f" .yaml)        # e.g. 01-s0_sig0
     # tag = rppo_hg<stem-with-dash-stripped>_s42  → see Launch Manifest for the exact per-run tag
     python train.py --config "$f" --agent_config configs/models/recurrent_ppo.yaml \
       --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
       --seed 42 --device cuda:<GPU> --tag "<manifest tag for this run>"
   done
   ```
   (The runner uses the exact per-run Tag from the §3 manifest, not an invented one.)
4. After training, results + verdict return to this doc (§4 outcomes filled, §5 analysis run).

## Links

- Living plan (Step 1 source of truth): [[20260616_1557_hunger_gated_avoidance]]
- Base scene (archived, frozen): `configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`
- Config authoring: [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md) (sparse `extends:`, list-replace footgun §1, init-state ranges §3.4)
- Carry-forward confounds: summary `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
