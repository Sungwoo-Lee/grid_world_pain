---
title: "Step 1 re-run — linear olfactory decay (longer-range distal smell cue)"
topic: hypervigilance
status: active
created: 2026-06-20
last_updated: 2026-06-20
phase: hunger_gated_avoidance
wandb_tag: hunger_gated_lindecay
develop_link: "[[20260616_1557_hunger_gated_avoidance]]"
---

# Step 1 re-run — linear olfactory decay (longer-range distal smell cue)

> **Status**: PLANNED (pre-registered — no runs launched yet)
> **Date**: 2026-06-20
> **Author**: experiment-designer
> **Related**: Step-1 (the comparison arm) [[20260619_hunger_gated_step1_discrimination_onset]];
> living plan [[20260616_1557_hunger_gated_avoidance]]; scarcity follow-up
> [[20260620_hypervig_scarcity_olfactory_ambiguity]]; config authoring
> [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md)

---

## 1. Research Question

**Plain-language framing.** In the first Step-1 sweep we put one dangerous predator and one harmless
rabbit in the same little world, made them behave identically, and let the agent tell them apart only
by **smell**. The agent never really learned to keep its distance from the predator: it survived by a
**reactive "tank-and-hide"** strategy — let the predator get right next to it, dive into a bush, heal,
repeat — rather than by **pre-emptively** avoiding the danger before contact. One strong suspect for
*why* is that smell, in that world, barely reaches: it falls off as **one-over-distance-squared**
(inverse-square), so by three cells away the predator's scent is only about a tenth of its on-cell
strength. With almost no advance warning, the agent had little choice but to react late.

This experiment changes **one knob and only one knob**: how fast smell fades with distance. We switch
the olfactory sensor from **inverse-square** fade (the steep default, exponent 2.0) to **linear** fade
(exponent 1.0), which lets smell carry roughly three times as far at three cells (~0.33 instead of
~0.11). Everything else — the lethal predator, the identical-behaviour rabbit, the eight food patches,
the randomised hunger and injury at the start of each episode, the exact smell vectors for all ten
cells of the original sweep — is held byte-identical. So this is a **clean paired comparison**: the
already-running Step-1 wave is the "steep smell" arm, and these ten new runs are the "long-range smell"
arm, differing in nothing but the fade exponent.

**The question.** Does a **longer-range distal smell cue** give the agent enough advance warning to
switch from reacting-after-contact to **avoiding-before-contact** — and does that show up as (a) a
larger predator-vs-rabbit gap in how close it lets each one get, and (b) the agent starting to increase
its distance from the predator at a **larger range** and **earlier in time** than under steep smell?

**Formal hypotheses.**

> **H₀ (null — longer range doesn't help):** Lengthening the olfactory cue (exponent 2.0 → 1.0) does
> not change the agent's pre-contact behaviour. Across the ten cells, the predator-vs-rabbit gap in
> closest-approach distance, flee rate, and bush-dive rate is statistically indistinguishable from the
> matched steep-decay Step-1 run, and the agent does not begin increasing its distance any earlier or
> at any larger range.

> **H₁ (longer range → pre-emptive avoidance):** With the longer-range cue, on the cells where the two
> smells differ (separation `s` > 0), the agent (a) holds the predator measurably farther than the
> rabbit *before contact* — a larger closest-approach gap than its steep-decay twin — and (b) begins
> backing away at a **larger predator range** and at an **earlier within-encounter timestep** than under
> steep decay. In short: the discrimination gap grows and the timing of avoidance moves earlier.

A reader needs no other document to know what is asked: does making smell carry farther turn a reactive
hide-and-heal agent into a pre-emptive avoider? Symbolic / numerical / path detail is in §2–§6.

---

## 2. Experimental Design

### 2.1 Independent Variable (the only one)

| Variable | Values | Where it lives | Note |
|----------|--------|----------------|------|
| olfactory distance-decay exponent `sensory.decay_power` | **2.0** (steep, inverse-square — the Step-1 arm) vs **1.0** (linear — this arm) | `configs/environment/default.yaml` (=2.0); overridden to 1.0 in the ten new configs | the single manipulated factor |

The exponent enters the olfactory sensor as `intensity = property / dist^decay_power` (capped 2.0
on-cell, summed within `sensory.sensor_radius`). At exponent 2.0 smell ≈ 0.11 at 3 cells, ≈ 0.04 at 5
cells; at exponent 1.0 it is ≈ 0.33 at 3 cells, ≈ 0.20 at 5 cells — a markedly longer-range distal cue.

The (separation `s`, per-episode noise σ) **grid is held identical** to Step-1 — it is *not* a second IV
here, it is the fixed backdrop that the decay manipulation is crossed against, one paired cell at a
time. The ten cells (smell vectors unchanged from Step-1):

| Run | Cell | `s` | σ | predator smell (ch2,ch3) | rabbit smell (ch2,ch3) | Step-1 twin (steep arm) |
|-----|------|-----|---|--------------------------|------------------------|-------------------------|
| 1 | `s0_sig0` | 0 | 0 | 0.5 / 0.5 | 0.5 / 0.5 | `rppo_hg01_s0_sig0_s42` |
| 2 | `s0.05_sig0` | 0.05 | 0 | 0.55 / 0.45 | 0.45 / 0.55 | `rppo_hg02_s0.05_sig0_s42` |
| 3 | `s0.1_sig0` | 0.1 | 0 | 0.6 / 0.4 | 0.4 / 0.6 | `rppo_hg03_s0.1_sig0_s42` |
| 4 | `s0.25_sig0` | 0.25 | 0 | 0.75 / 0.25 | 0.25 / 0.75 | `rppo_hg04_s0.25_sig0_s42` |
| 5 | `s0.5_sig0` | 0.5 | 0 | 1.0 / 0.0 | 0.0 / 1.0 | `rppo_hg05_s0.5_sig0_s42` |
| 6 | `s0.1_sig0.2` | 0.1 | 0.2 | 0.6 / 0.4 | 0.4 / 0.6 | `rppo_hg06_s0.1_sig0.2_s42` |
| 7 | `s0.25_sig0.2` | 0.25 | 0.2 | 0.75 / 0.25 | 0.25 / 0.75 | `rppo_hg07_s0.25_sig0.2_s42` |
| 8 | `s0.5_sig0.2` | 0.5 | 0.2 | 1.0 / 0.0 | 0.0 / 1.0 | `rppo_hg08_s0.5_sig0.2_s42` |
| 9 | `s0.1_sig0.4` | 0.1 | 0.4 | 0.6 / 0.4 | 0.4 / 0.6 | `rppo_hg09_s0.1_sig0.4_s42` |
| 10 | `s0.5_sig0.4` | 0.5 | 0.4 | 1.0 / 0.0 | 0.0 / 1.0 | `rppo_hg10_s0.5_sig0.4_s42` |

### 2.2 Dependent Variables

- **Primary (project convention): survival steps per run across training** — headline performance,
  reward is a secondary diagnostic only.
- **Behavioural pre-contact gap (predator − rabbit), the discrimination read-out**, measured three
  ways exactly as in Step-1 so the arms are directly comparable:
  1. **Closest-approach distance** — minimum agent-to-animal distance before first contact (predator
     held *farther* ⇒ positive gap).
  2. **Flee rate** — fraction of pre-contact encounters where the agent increases distance within the
     observation window.
  3. **Bush-dive rate** — fraction of pre-contact encounters ending in a `hides_agent` bush cell.
- **Pre-emptive-timing read-out (new emphasis vs Step-1):** the question here is not only *how big*
  the gap is but *when* the agent starts avoiding. Two measures, both predator-specific:
  - **Avoidance-onset range** — the predator-to-agent distance at the first step on which the agent
    begins increasing that distance within an encounter (larger = earlier-warning avoidance).
  - **Avoidance-onset timestep** — the within-encounter step index at which distance-increase begins
    (earlier = more pre-emptive). Both compared **arm-to-arm** (linear vs steep) per cell.

### 2.3 Controlled / Fixed Factors (everything else)

Pinned **byte-identical** to the matching Step-1 cell, *enforced structurally* rather than by hand:
each lindecay config `extends:` its Step-1 twin (which itself extends `environment/default`) and adds
**only** `sensory.decay_power: 1.0`. Because `sensory` is a dict, the deep-merge writes that one key
into the inherited sensory block and leaves `olfactory_enabled`, `sensor_radius`, `vector_size`, and
the visual sensor untouched. A live config-load parity check (§Verification) confirms every other
merged env parameter is identical to the Step-1 twin.

Held constant (inherited through the Step-1 twin): 10×10 grid, max 500 steps; lethal predator
`damage [5,120]`, nociception 0.9, full-grid hunt; byte-identical harmless rabbit (class neutral,
damage `[0,0]`); inert hiding-predator slot (count 0); 8 food patches (abundant, `[1,0,0,0,0]` smell,
no damage); 12 rocks + 12 hiding bushes; contact-only vision (`visual_sensor_range: 0`); randomised
initial state — nutrition `[10,100]`, injury `[0,80]`, satiation derived from nutrition; perceptual
noise off. Training: agent `configs/models/recurrent_ppo.yaml`, fresh-init, single seed 42, ~10M
episodes, num-envs 128, checkpoint-frequency 100k — all identical to Step-1.

### 2.4 The global-decay caveat (important, intrinsic)

`decay_power` is **global to the olfactory sensor**: lengthening the range lengthens the range of
**all** smells, not just the animals'. So in this arm the **food** cue (and the bush/rock smells) also
carry farther — the agent can smell food from farther away too. This is intrinsic to the manipulation;
there is no way to lengthen only the predator's smell with this knob. It does **not** break the paired
comparison — the Step-1 twin differs from this arm in `decay_power` and nothing else, so any
arm-to-arm difference is attributable to the longer-range *olfactory regime as a whole*. The
interpretive cost is that a survival or behaviour change could in principle be driven by easier
foraging rather than by earlier predator detection; the §5 analysis disambiguates this by reading the
**predator-specific** avoidance-onset and gap measures, which a foraging improvement alone would not
move.

### 2.5 Seeds & sample size

- **Seeds:** single training seed **42** per cell, matching the Step-1 arm exactly so the comparison is
  a true matched pair (same seed, same scene, only decay differs). This is a deliberate **provisional**
  first-wave choice (see §6 limitations) — any arm-to-arm difference flagged here is a candidate effect,
  not a confirmed one, and earns multi-seed hardening before it is believed.
- **Sample size:** 200 eval episodes per cell at each analysed checkpoint (fixed `eval_seeds`,
  deterministic policy), identical to Step-1; within-run CIs are episode-sampling only.
- **Compute (rough):** same as the Step-1 wave — ~10M episodes × 10 runs at the Step-1 per-run rate.

---

## 3. Launch Manifest

System-of-record for all 10 runs. Designer fills the planned columns; `training-runner` fills
Node / GPU / Launched at / WandB run ID / Log path at launch. **No runs launched yet.**

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | s0_sig0 | `rppo_hg01_s0_sig0_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 2 | planned | s0.05_sig0 | `rppo_hg02_s0.05_sig0_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 3 | planned | s0.1_sig0 | `rppo_hg03_s0.1_sig0_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 4 | planned | s0.25_sig0 | `rppo_hg04_s0.25_sig0_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 5 | planned | s0.5_sig0 | `rppo_hg05_s0.5_sig0_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 6 | planned | s0.1_sig0.2 | `rppo_hg06_s0.1_sig0.2_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 7 | planned | s0.25_sig0.2 | `rppo_hg07_s0.25_sig0.2_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 8 | planned | s0.5_sig0.2 | `rppo_hg08_s0.5_sig0.2_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 9 | planned | s0.1_sig0.4 | `rppo_hg09_s0.1_sig0.4_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |
| 10 | planned | s0.5_sig0.4 | `rppo_hg10_s0.5_sig0.4_dp1_s42` | hunger_gated_lindecay | prod | 42 | — | — | — | — | — |

Tags are unique, parseable (`rppo_hg<NN>_s<sep>_sig<sigma>_dp1_s<seed>`), carry a `dp1` marker that
pairs each row to its Step-1 twin by stripping it, and are identical to the wandb-name. All rows share
group `hunger_gated_lindecay`.

### 3.1 Configs to Produce (designer-only, pre-launch)

No model hyperparameter is under test, so every run uses the **same** agent config
(`configs/models/recurrent_ppo.yaml`). Each env config extends its Step-1 twin and overrides only
`sensory.decay_power: 1.0`.

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/environment/experiment/olfactory_ambiguity_lindecay/01-s0_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/environment/experiment/olfactory_ambiguity_lindecay/02-s0.05_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 3 | `configs/environment/experiment/olfactory_ambiguity_lindecay/03-s0.1_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 4 | `configs/environment/experiment/olfactory_ambiguity_lindecay/04-s0.25_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 5 | `configs/environment/experiment/olfactory_ambiguity_lindecay/05-s0.5_sig0.yaml` | `configs/models/recurrent_ppo.yaml` |
| 6 | `configs/environment/experiment/olfactory_ambiguity_lindecay/06-s0.1_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 7 | `configs/environment/experiment/olfactory_ambiguity_lindecay/07-s0.25_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 8 | `configs/environment/experiment/olfactory_ambiguity_lindecay/08-s0.5_sig0.2.yaml` | `configs/models/recurrent_ppo.yaml` |
| 9 | `configs/environment/experiment/olfactory_ambiguity_lindecay/09-s0.1_sig0.4.yaml` | `configs/models/recurrent_ppo.yaml` |
| 10 | `configs/environment/experiment/olfactory_ambiguity_lindecay/10-s0.5_sig0.4.yaml` | `configs/models/recurrent_ppo.yaml` |

### Verification (run at design time, 2026-06-20)

Live config load (`load_env_config` then `load_env_params`) on all 10 lindecay configs vs their
Step-1 twins confirmed: (a) `sensory.decay_power` resolves to **1.0** in every lindecay config and
**2.0** in every Step-1 twin; (b) with `decay_power` set aside, **every other merged env parameter is
byte-identical** to the matching Step-1 config (zero diffs on all 10 — smell vectors, lethal `[5,120]`,
1-vs-1 chase, randomised starts `[10,100]`/`[0,80]`, abundant food, obstacles, body block); (c) full
mandatory-key validation (`load_env_params`) passes on the merged result.

---

## 4. Predicted Outcomes (pre-registered)

- **Primary prediction (H₁).** On the separated cells (`s` ≥ 0.1, runs 3–10), the linear-decay arm
  shows a **larger** predator-vs-rabbit closest-approach gap than its steep-decay Step-1 twin, **and**
  the agent's avoidance-onset range is larger / its avoidance-onset timestep earlier — i.e. it starts
  backing away from the predator sooner and from farther out. The matched anchor (run 1, no smell
  difference) should show **no** arm-to-arm gap change — it is the within-experiment null.
- **Magnitude shape.** If the longer range matters, the arm-to-arm gap *increase* should be largest on
  the most discriminable cells (high `s`, low σ — runs 4, 5, 7, 8) where there is a clear class signal
  to act on at range, and smallest near the anchor.
- **Refutation (H₀ retained).** If the gaps and the avoidance-onset timing are statistically
  indistinguishable between arms across the separated cells, longer-range smell did **not** convert the
  agent to pre-emptive avoidance — the tank-and-hide strategy is robust to cue range, pointing the
  follow-up toward the lethality / incentive levers (the scarcity follow-up) rather than the sensor.
- **Survival is read but not the discriminator.** Survival steps may *rise* in this arm simply because
  food is easier to smell; a survival gain **without** a predator-specific avoidance-onset shift is
  scored as a foraging effect, **not** as pre-emptive avoidance (per the §2.4 caveat).

---

## 5. Analysis Plan (pre-specified)

**Comparison structure.** Every read-out is computed **per cell, per arm**, and the headline statistic
is the **arm-to-arm difference (linear − steep) at the matched cell** — a true paired comparison. The
Step-1 results table (already filled) supplies the steep-arm numbers; this arm supplies the linear-arm
numbers at the identical eval protocol (200 episodes, fixed `eval_seeds`, deterministic policy, final
10M checkpoint plus the per-checkpoint series).

**Statistics.** Per cell per arm, report each gap as mean ± 95% CI over the 200 eval episodes. The
arm-to-arm difference is reported with its CI; "the longer range helped at this cell" requires the
linear-minus-steep closest-approach-gap difference to be **positive with a 95% CI excluding 0**, with
the avoidance-onset measures pointing the same way (≥ 2 of {closest-approach gap, avoidance-onset
range, avoidance-onset timestep} agreeing). Single training seed → the CIs are within-run only and the
verdict is **provisional**, not a cross-seed claim.

**Pre-emptive-timing analysis (the new core read).** For each predator encounter, locate the first
step where the agent begins increasing predator distance; record (range at that step, timestep index).
Compare distributions arm-to-arm per cell. Pre-registered window: encounters are read from first
predator detection up to first contact; lag 0 (we ask *when within the encounter* avoidance starts, not
a cross-correlation lag).

**Temporal evolution (mandatory).** Track survival steps and the predator-vs-rabbit gap **across
training checkpoints** (every 100k episodes) for both arms — the longer-range cue may shift *when*
during training a gap emerges, and a final-snapshot-only read would miss an earlier-forming gate.

**Disambiguation control (for the global-decay caveat).** Report the **food-approach** behaviour
(time-to-first-food, mean food-smell at approach) alongside the predator measures. If survival rises
but only the food measures move and the predator avoidance-onset does not, the verdict is "easier
foraging", not "pre-emptive avoidance".

---

## 6. Failure-Mode Catalog (pre-decided)

- **Training instability (NaN / value explosion).** Refutes the **run**, not a hypothesis. Re-launch
  the affected cell (same seed) once; if it recurs, flag the cell as "no data" rather than reading it
  as H₀.
- **Lethality still masks the effect (carried from Step-1).** The predator is lethal `[5,120]`; the
  Step-1 verdict was that lethality likely prevented any hunger-gate from being reinforced. If, even
  with longer-range warning, the agent still dies at a high rate while pinned at distance 1, the read
  is "lethality dominates over cue range", and the lethality lever (sub-lethal follow-up) — not the
  sensor — is the next move. A longer-range cue that does **not** reduce death rate is informative:
  warning without survivable escape is not enough.
- **Foraging confound dominates (global-decay caveat).** If survival rises but the predator-specific
  avoidance-onset measures do not move, score as a **foraging** effect (§2.4 / §5 control), **not**
  pre-emptive avoidance. Do not claim H₁ on survival alone.
- **Insufficient horizon.** If the arm-to-arm gap or the avoidance-onset shift is still moving at the
  end of training (temporal curve not plateaued), the estimate is a lower bound — extend that cell
  rather than reading a null.
- **Anchor moves (run 1).** Run 1 has no smell difference, so its arm-to-arm gap change should be ≈ 0.
  If the linear arm opens a predator-vs-rabbit gap at the anchor where the steep arm did not, that is a
  longer-range **vision/geometry or count-leak** interaction, not olfactory discrimination — discount
  the separated-cell gaps by the anchor's arm-to-arm change.

---

## 7. Metrics Requested

None new are required for the **primary** read-out: closest-approach distance, flee rate, and bush
occupancy come from the inherited behaviour-measure toolkit (M1/M2 online + the M7 eval-rollout
protocol), and survival steps + per-checkpoint logging already exist. The **avoidance-onset
range/timestep** measures are derived offline from the eval-rollout recordings (`.rec.gz`), so no new
in-`src` logger is needed to compute them. The same optional convenience as Step-1 applies (logging
starting nutrition per eval episode) and remains **non-blocking**.

---

## 8. Handoff / Next Steps

1. **env-config-auditor** pre-flight on the 10 lindecay configs (confirm the single-key override,
   obs↔noise width unchanged, smell symmetry inherited intact, list-replace completeness via the
   extends chain).
2. **PI consult** (multi-run launch decision) per the agent playbook.
3. **training-runner** launches the wave with user-supplied node + GPU, using the exact per-run Tag
   from the §3 manifest.
4. After training, results + verdict (arm-to-arm comparison vs Step-1) return to this doc.

## Links

- **Comparison arm (steep decay, the 2.0 arm):** [[20260619_hunger_gated_step1_discrimination_onset]]
- Living plan (Step-1 source of truth): [[20260616_1557_hunger_gated_avoidance]]
- Scarcity follow-up (sibling lever): [[20260620_hypervig_scarcity_olfactory_ambiguity]]
- Config authoring: [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md) (sparse `extends:` layering,
  nested extends chains, list-replace footgun)
- Olfactory sensor decay: `src/environment/sensor.py` `sense_resource`; schema
  `docs/environment/02_config_schema.md` (`sensory.decay_power`)
