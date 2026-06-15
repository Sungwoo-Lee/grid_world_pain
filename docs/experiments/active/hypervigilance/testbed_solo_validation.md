---
title: "Solo-animal testbed validation — calibrating the predator-vs-rabbit discrimination metric"
topic: hypervigilance
status: active
created: 2026-06-16
last_updated: 2026-06-16
phase: hypervigilance
wandb_tag: hypervigilance
develop_link: "[[behavior_measurement_charter]]"
---

# Solo-animal testbeds — does our clean measure actually tell danger-recognition apart from general caution?

## Question / Purpose (plain-language entry point)

We want a single, scalable number that says "this agent recognises a dangerous predator
and treats it differently from a harmless rabbit." Earlier, averages computed in our rich
training world said every agent was "class-blind" (could not tell the two apart), yet a
slow, hand-read, step-by-step trajectory inspection showed one agent clearly *did*
discriminate — it kept eating near a rabbit but stopped eating and dived into a bush when
the predator approached. The averages were fooled because the complex world blends many
situations together and washes the difference out to zero.

This experiment **validates the fix before we trust it.** We built three stripped-down
**test worlds** (no retraining — we drop already-trained, frozen agents in and just watch):

- **predator world** — the agent alone with one dangerous animal,
- **rabbit world** — the agent alone with one harmless animal that looks and moves
  identically, and
- **forage-only world** — no animals at all, a no-threat baseline anchor.

Because each world has only one animal class, "discrimination" becomes the **difference in
behaviour between the predator world and the rabbit world** (we call this difference Δ,
"Delta"). We run two agents we already understand through all three worlds: a **known
discriminator** (the "Cell C" matched-smell agent, which the hand-read showed does tell
predator from rabbit) and a **known class-blind agent** (the "cell-08" single-predator-rabbit
agent, which behaves the same toward both). The testbed **passes** only if its averages
reproduce what the hand-read already told us: the discriminator's Δ is clearly non-zero and
points toward danger-avoidance, while the class-blind agent's Δ sits in a near-zero noise
band. If the clean world's averages cannot separate these two known agents, the testbed is
not isolated enough and must be reworked before we draw any new conclusion from it.

This is a **calibration / validation** experiment, not a hypothesis test about a new agent.
The "hypothesis" under test is about the **measurement instrument**, not the agents.

## Relationship to the charter

This doc operationalises the **Validation requirement** of
[[behavior_measurement_charter]] (calibration-against-controls before trust) for the first
two registered testbeds (predator-solo, rabbit-solo) plus a forage-only anchor. The charter
owns the *why*; this doc owns the *configs + pre-registered read-out + pass thresholds*.

## 1. Research Question

Does the predator-solo-vs-rabbit-solo cross-world behaviour contrast (Δ), measured on a
shared panel over the frozen Cell-C and cell-08 agents at 200 eval episodes per world per
agent, reproduce the trajectory-read ground truth — i.e. yield a clearly non-zero,
danger-avoidant Δ for the known discriminator (Cell C) on ≥2 of {standoff distance, cover
use, eat-suppression} while the known class-blind agent (cell-08) stays within a
pre-registered near-zero noise band on all of them?

## 2. Hypothesis & Predicted Outcomes (pre-registered)

The instrument-level hypothesis: **the solo testbeds isolate danger-class enough that
mean-level measures become an honest discrimination proxy.**

- **Validates (instrument trusted)** iff, on the shared panel:
  - Cell C (discriminator) shows |Δ| exceeding the "clear" threshold (below) on **≥2 of 3
    core measures** {standoff, cover/bush-dive, eat-suppression}, **with the sign matching
    danger-avoidance** (larger standoff, more cover, more eat-suppression, fewer/later
    contacts in the predator world), **AND**
  - cell-08 (class-blind) stays **within the noise band** (below) on **all 3 core
    measures**, **AND**
  - a `trajectory-story` spot-check on ≥3 representative episodes per world per agent
    agrees with the sign of each measure's Δ.
- **Refutes the instrument (testbed reworked, not the agents)** if any of:
  - Cell C's Δ is inside the noise band on ≥2 of 3 core measures (the clean world cannot
    surface a discrimination we know exists), **or**
  - cell-08's Δ exceeds the "clear" threshold on ≥2 of 3 core measures (the clean world
    manufactures discrimination where the hand-read says there is none — a confound is
    leaking through), **or**
  - the trajectory spot-check contradicts the mean's sign (a mean that disagrees with the
    trajectory is exactly the failure the charter forbids).

Predicted *shape* (from the prior trajectory read, charter §"Validation requirement"): Cell
C bush-dive ~0.76 predator vs ~0.44 rabbit (Δ ≈ +0.32), eat-suppression ~0.73 predator vs
~1.26 rabbit (Δ in the danger-avoidant direction); cell-08 bush-dive ~0.58 vs ~0.56 (Δ ≈
+0.02), eat-suppression ~1.0 vs ~1.0 (Δ ≈ 0). The forage-only anchor predicts both agents
forage freely with no standoff and no eat-suppression.

## 3. Launch Manifest (system-of-record for all eval rollouts)

These are **eval-only rollouts on frozen checkpoints** (`scripts/eval_rollout.py`), not
training runs — there are no WandB training runs, no nodes/GPUs locked here, no s/it. The
"Tag" column is the eval output-bucket label (also the wandb-group if eval logging is on).
2 agents × 3 worlds = 6 rollout jobs. Each world runs the 200 shared eval seeds the base
config carries.

| Run | World (config) | Agent (frozen ckpt) | Tag (= group label) | job-type | Episodes | Status |
|---|---|---|---|---|---|---|
| R1 | predator_solo | Cell C (discriminator) | `tb_predsolo_cellC` | eval | 200 | planned |
| R2 | rabbit_solo   | Cell C (discriminator) | `tb_rabsolo_cellC`  | eval | 200 | planned |
| R3 | forage_only   | Cell C (discriminator) | `tb_forage_cellC`   | eval | 200 | planned |
| R4 | predator_solo | cell-08 (class-blind)  | `tb_predsolo_c08`   | eval | 200 | planned |
| R5 | rabbit_solo   | cell-08 (class-blind)  | `tb_rabsolo_c08`    | eval | 200 | planned |
| R6 | forage_only   | cell-08 (class-blind)  | `tb_forage_c08`     | eval | 200 | planned |

Actual columns (eval output path, launched-at) left `—` for `training-runner` / the eval
launcher to fill at run time. The frozen checkpoint paths for Cell C and cell-08 are
supplied at launch (they are not configs this agent owns); the design only fixes which
agent × which world.

### 3.1 Configs to produce

| World | Env config (this experiment) | Agent config |
|---|---|---|
| predator_solo | `configs/experiment/hypervigilance/testbed_predator_solo.yaml` | (frozen ckpt's own model config, supplied at eval) |
| rabbit_solo   | `configs/experiment/hypervigilance/testbed_rabbit_solo.yaml`   | (frozen ckpt's own model config) |
| forage_only   | `configs/experiment/hypervigilance/testbed_forage_only.yaml`   | (frozen ckpt's own model config) |

Each env config is a verbatim copy of
`configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml` with **only** the
entity `count` fields changed (predator_solo: rabbit→0; rabbit_solo: predator→0;
forage_only: both→0). Food, bushes, body, sensors, `perceptual_noise`,
`behavior_measures` (enabled, cue_radius, obs_window, eval_n_episodes=200, the 200 eval
seeds, eval_max_steps=500), and `max_steps` are byte-identical across all three — the
worlds are confound-matched.

## 4. Experimental Design

- **Independent variables**: (i) **world** {predator_solo, rabbit_solo, forage_only} — the
  factor that defines Δ; (ii) **agent** {Cell C discriminator, cell-08 class-blind} — the
  two known-truth calibrators.
- **Dependent variable (primary)**: survival steps per episode (project headline metric).
- **Dependent variables (panel, per world)**:
  - **Standoff**: closest-approach distance to the single animal — median + 10th percentile
    of the per-episode minimum-distance distribution.
  - **Cover**: fraction of steps in a bush (M2 occupancy) + bush-dive rate on approach
    (dive into cover conditional on the animal being within cue_radius).
  - **Eat-suppression**: eat-rate near the animal (within cue_radius) vs far — reported as
    near/far ratio (lower near-rate ⇒ more suppression).
  - **Contact**: contact count per episode + latency (steps) to first contact.
- **Discrimination statistic**: Δ(measure) = value(predator_solo) − value(rabbit_solo), per
  agent. Forage_only is the no-threat anchor (expected: free foraging, no standoff/no
  suppression) — used to confirm the panel reads ~baseline when nothing dangerous exists,
  not part of Δ.
- **Controls / fixed factors (pinned, named)**: identical 10×10 grid, identical 8 food
  sources, identical 12 bushes + 12 rocks, identical body homeostasis, identical sensor
  suite (olfactory, nociception, visual range 0, proprioception, interoceptive
  convolution), `perceptual_noise.enabled: false`, identical animal chase profile (the one
  animal present is byte-identical between worlds except class), `max_steps: 500`,
  `eval_max_steps: 500`, `eval_policy_mode: deterministic`, `eval_obs_noise: training`, and
  the **same 200 eval seeds** in every world. The only thing that varies is which animal
  class is present.
- **Seeds / sample size**: 200 eval episodes per world per agent (the shared seed list the
  base config carries) → 200 × 3 worlds × 2 agents = 1 200 episodes total. Eval is cheap
  (frozen forward passes, 500 steps each); the 200-seed panel gives tight distribution
  estimates so a thin/skewed bin cannot masquerade as signal (charter design-rule 5).
- **No training**, no compute-budget estimate beyond rollout time.

## 5. Analysis Plan (pre-specified)

- **Primary statistic per measure**: mean ± 95% CI across the 200 episodes, plus the
  reported quantiles (median, 10th pct for standoff). Δ reported with a CI obtained by
  bootstrap over episode-paired differences where seeds are shared (they are).
- **Effect-size thresholds** — see §6 thresholds; "clear" vs "noise band" pre-registered
  numerically there.
- **Temporal-evolution check is N/A for training** (no training), but the panel is computed
  over the *within-episode* time course: eat-suppression and standoff are evaluated as
  conditional-on-proximity quantities, not single end-of-episode scalars, so a transient
  pre-contact response is not averaged away. Contact latency is itself a within-episode time
  measure.
- **Cross-checks**: `trajectory-story` read on ≥3 representative episodes per world per
  agent (charter design-rule 7) — the sign of each panel Δ must match the qualitative read.

## 6. Pre-registered PASS thresholds ("clear" vs "noise band")

Stated in the measure's own units. "Clear" = the Δ a discriminator must exceed; "noise
band" = the |Δ| a class-blind agent must stay within. The band is set comfortably below the
prior Cell-C effect sizes so the two known agents are separable with margin.

| Core measure | Δ unit | "Clear" (discriminator must exceed) | "Noise band" (class-blind must stay within) |
|---|---|---|---|
| **Standoff** — median closest-approach distance | grid cells | Δ ≥ +1.0 cell (predator world ≥1 cell farther) | |Δ| ≤ 0.4 cell |
| **Cover** — bush-dive-on-approach rate | rate (0–1) | Δ ≥ +0.15 (≥15 pts more diving in predator world) | |Δ| ≤ 0.07 |
| **Eat-suppression** — near-animal eat-rate (lower = more suppression) | rate (0–1) | Δ ≤ −0.20 (≥20 pts lower eating near predator) | |Δ| ≤ 0.08 |

Supporting (not gating, reported for the story): bush-occupancy fraction Δ; standoff 10th
percentile Δ; contact count Δ (discriminator: fewer in predator world); first-contact
latency Δ (discriminator: longer in predator world).

**Pass rule (restated):** the testbed **validates** iff Cell C exceeds "clear" on **≥2 of
3** core measures with the danger-avoidant sign, cell-08 stays within "noise band" on **all
3**, and the trajectory spot-check agrees. Anything else triggers the §2 refutation branch
(rework the testbed; do not reinterpret the agents).

Rationale for the numbers: prior Cell-C bush-dive Δ was ≈ +0.32 and cell-08 ≈ +0.02 (charter
§"Validation requirement"); a 0.15 clear / 0.07 band split puts the decision boundary
roughly midway on a log scale, giving each known agent ≥2× margin from the boundary.
Standoff +1.0/0.4 reflects that one full cell of extra standoff is behaviourally meaningful
on a 10×10 grid while sub-half-cell jitter is seed noise. Eat-suppression −0.20/0.08 mirrors
the cover split in magnitude.

## 7. Failure-Mode Catalog (pre-decided)

- **Both agents show large Δ in the same direction** → a confound (not class) is leaking;
  the world is not isolated. Refutes the *instrument*, rework testbed. (Most likely culprit:
  spawn geometry or contact dynamics differing by class — but configs are byte-identical
  except class, so this would point at a loader/engine asymmetry to report.)
- **forage_only shows standoff / eat-suppression > 0** → the panel mis-measures in the
  absence of any animal (e.g. distance-to-nonexistent-animal undefined). Treat as an
  instrument bug to report, not an agent result. The zero-animal loader path returns empty
  animal arrays (verified), so any animal-referenced measure must be defined as N/A, not 0,
  in forage_only.
- **Survival floors/ceilings identically across worlds** → expected and fine (predator
  world may show lower survival; if survival is identical that is itself informative but
  does not by itself fail the panel — discrimination is read on the behavioural measures,
  not survival).
- **Trajectory spot-check disagrees with a mean's sign** → hard fail per charter; the mean
  is not trusted.
- **Seed noise drowns Δ for Cell C** → 200 shared seeds already; if CI still overlaps the
  band, report as "instrument underpowered at 200 episodes", do not silently accept.

## 8. Configs verification (count: 0 behaviour)

Each config was loaded through `src.environment.config_loader.load_env_params` and the
resulting animal classes printed. `count: 0` cleanly drops the entity via the loader's
`for _ in range(count)` loop; the all-zero case (forage_only) takes the loader's explicit
zero-animal branch (`N == 0` → empty animal arrays) without crashing. Printout:

```
[predator_solo] animal_classes=('predator',)          predators=1 neutrals=0  behavior_measures.enabled=True  is_damaging=(True,)
[rabbit_solo]   animal_classes=('neutral',)           predators=0 neutrals=1  behavior_measures.enabled=True  is_damaging=(False,)
[forage_only]   animal_classes=()                     predators=0 neutrals=0  behavior_measures.enabled=True  is_damaging=()
[BASE_cell08]   animal_classes=('predator','neutral') predators=1 neutrals=1  behavior_measures.enabled=True  is_damaging=(True, False)
```

No `count: 0` breakage. `behavior_measures.enabled` stays True in all three (so M1/M2/M5
compute). Caveat carried into §7: in forage_only, any animal-referenced panel measure
(standoff, eat-near, contact) is **undefined**, not zero — the analysis must treat
forage_only's animal-referenced cells as N/A and use it only as the foraging-freedom anchor.

## Links

- Charter (the why): [[behavior_measurement_charter]]
- Candidate testbed menu: [[predator_rabbit_testbeds]]
- Base training config: `configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml`
- Trajectory-level cross-check tool: `.claude/skills/trajectory-story/SKILL.md`
- Eval-rollout entry point: `scripts/eval_rollout.py`

## Results / Analysis / Conclusions

_(blank — filled after the six eval rollouts complete.)_
