---
title: "SameProp Round 3 — predator behavioural randomisation: does avoidance generalise to neutrals when the predator's chase cue is unreliable?"
topic: hypervigilance
status: active
created: 2026-05-28
last_updated: 2026-05-28
phase: 2
wandb_tag: "hypervigilance-round3-distributional"
supersedes: []
---

# SameProp Round 3 — predator behavioural randomisation: does avoidance generalise to neutrals when the predator's chase cue is unreliable?

> **Related**:
> - Closing summary of the prior study line (sameProp Rounds 1–2.6): [`20260521_1546_sameprop_rabbit_avoidance_study.md`](../../summaries/20260521_1546_sameprop_rabbit_avoidance_study.md).
> - Seed-locked Round 2.6 Cell C baseline (the comparison reference): [`sameprop_round26_design.md`](sameprop_round26_design.md), §§9–12.
> - Behaviour-metric toolkit v1 (M1 / M2 / M5 / M7 operational definitions): [`behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md).
> - v2.0 env refactor that landed the per-episode `[low, high]` sampling used here: [`UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`](../../../develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md).
> - Parity-reference baseline config: [`02-sameProp_R2_decoupleFood.yaml`](../../../../configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml) — Cell C food-decoupling, predator behavioural parameters static.

---

## 0. What this round asks — plain-English entry point

In our previous studies, the patrolling predator and the two rabbits carry the **same olfactory smell** (the *matched-smell* or *sameProp* setup), so the agent's smell sensor cannot tell them apart. We expected this to make the agent treat all three animals the same way. It did, for *average distance* — but in the *seconds around an animal entering the agent's danger range*, the agent dives into a bush about **88% of the time when a predator approaches and only 51% of the time when a rabbit approaches** (a 37-percentage-point class gap, seed-locked across two independent training seeds). The remaining headline question was: *how is the agent still telling them apart?* One obvious candidate cue is **behaviour**: the predator chases the agent (state-machine in HUNT mode); the rabbits wander.

This round asks: if we **randomise the predator's chase behaviour on every episode** — wide enough that some episodes look like "always-chasing stalker" and other episodes look like "wandering harmless animal" — does the agent's class-conditional defence collapse? Concretely: does the in-cover rate near rabbits rise substantially (because the agent now treats anything matched-smell as potentially threatening), narrowing the 37-pp class gap? Or does the agent keep discriminating, suggesting the cue it uses is *not* the chase behaviour at all?

**Why this is possible now.** The v2.0 env refactor (just landed, branch `v2.0`) added per-episode uniform sampling for five predator behavioural parameters: how far the predator can detect the agent, how much chase-energy it carries, how fast that energy recovers, how depleted it must be before re-engaging, and how persistent the chase is past the detection boundary. Each is now specifiable as a `[low, high]` range in YAML; the env draws a fresh value at every episode reset. This is the first experiment that uses this new capability.

**Important honest caveat.** The agent's *visual* sensor channel-codes the animal class with a fixed one-hot (`predator → channel 5`, `neutral → channel 7`), regardless of olfactory match. So even under matched smells the agent CAN always tell predator from rabbit at close range by sight. Any avoidance generalisation we might observe in this round therefore must operate **through some other channel than vision** — for example, the agent learning a more generic "anything that moves fast at me / triggers the danger-range event = threat" representation, or the agent under-using the visual channel because its training signal was confounded enough to make the visual cue less reliable. We do not expect a complete collapse; the question is whether the class gap *narrows*.

**What success and failure look like.** We pre-register two thresholds before launch. **Confirmation** (the agent's avoidance generalises to neutrals): the in-cover rate gap between predator and rabbit drops from the seed-locked +37 pp baseline to below +20 pp. **Refutation** (the avoidance stays class-pinned): the gap remains at or above +30 pp. Anything in between is a borderline result that would need a second seed before any verdict.

**Compute.** One new training run at the full 10-million-episode budget on a single lab node. Comparison reference is the Round 2.6 Cell C run on seed 44 (`ja5fu5k3`); no new baseline training is needed. Post-hoc analysis uses the same 200-episode eval-rollout protocol that produced the seed-locked finding.

## 1. Research Question

> **Headline.** Setting the predator's five behavioural fields (`detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`) to per-episode-uniform ranges that span from "harmless wanderer" to "always-chasing stalker", under the otherwise-identical sameProp food-decoupling configuration of Cell C, will reduce the agent's class-conditional defence on M2 (in-cover rate at threat-onset) and M5 (eat-suppression near threat), measured against the Round 2.6 seed-44 reference.

The question is hypothesis-driven and pre-registered. Predicates:

- **H₁(generalised-avoidance) — confirmation.** The class gap on the headline metric collapses meaningfully: `Δ_M2_class < +20 pp` AND (`M2_neutral ≥ 0.65` OR `M5_neutral < 1.05`). Confirms the agent's class discrimination in the Round 2.6 baseline was leaning on the predator's chase behaviour as a cue, and that removing the cue forces the agent's avoidance to generalise toward the matched-smell neutrals.

- **H₀(class-pinned-avoidance) — refutation.** The class gap persists at near-baseline magnitude: `Δ_M2_class ≥ +30 pp` AND `M2_neutral < 0.60`. Confirms the agent's class discrimination was NOT primarily driven by the chase behaviour — most plausibly it leans on the visual one-hot channel codes (the architectural caveat above), and behavioural randomisation alone is not enough to force generalisation.

- **Borderline.** Anything in between (`+20 pp ≤ Δ_M2_class < +30 pp`, or one of the two H₁ secondaries met but not both). One additional seed required before any verdict; flagged for a Round 3.5 follow-up.

The "Headline" sentence is the falsifiable claim. The two predicate bands above are the threshold rules applied to the eval-rollout numbers; the borderline band is the escalation rule.

## 2. Hypothesis & Predicted Outcomes

### 2.1 What the seeds-locked Cell C baseline gave us

From the closing analysis of Round 2.6 (`docs/experiments/active/hypervigilance/sameprop_round26_design.md` §9.3, 200-episode deterministic eval-rollout on seeds 1000–1199):

| Measure | Seed 42 (R2.5 baseline) | Seed 44 (R2.6 seed-lock) |
|---|---:|---:|
| **M2** in-cover rate, predator-onset | 0.876 | **0.888** |
| **M2** in-cover rate, rabbit-onset (aggregated) | 0.508 | **0.515** |
| **Δ_M2_class** (predator − rabbit) | **+0.368** | **+0.373** |
| **M5** eat-suppression, predator | 0.748 | **0.769** |
| **M5** eat-suppression, rabbit | 1.186 | **1.202** |
| Mean survival (eval, n=200) | 405.4 | 414.7 |

The +37 pp class gap on M2 and the predator-vs-rabbit M5 split (predator < 1.0 < rabbit) are the load-bearing R2.6 numbers. These are the comparison points the §4 analysis applies the H₁ / H₀ thresholds to.

### 2.2 Designer's prior — what we expect to see

There are three plausible outcomes; only the first counts as confirmation of the *generalised-avoidance* hypothesis.

1. **Partial generalisation (the modal designer prior, ~50%).** The agent's M2_neutral rises from ~0.51 to ~0.65–0.70 — it dives into cover more often for rabbits because some episodes' rabbit-like predator (low detection_range, low stamina, high hunt_threshold) is sensorily indistinguishable from a rabbit in terms of *behaviour* during the threat-onset window. M2_predator stays near the +0.85–0.90 ceiling. Δ_M2_class drops to roughly +0.18–0.25, which crosses the < +20 pp confirmation threshold by a few percentage points.

2. **No generalisation — visual one-hot wins (~35%).** The agent learns from the visual channel that "channel 5 = damaging, channel 7 = neutral" robustly enough that behavioural unreliability does not matter. M2 numbers are essentially the same as Round 2.6 Cell C (Δ_M2_class ≈ +0.35). This refutes H₁ and is the load-bearing caveat in §0: vision is a fully sufficient cue for discrimination even when behaviour becomes unreliable.

3. **Over-generalisation / instability (~10%).** The agent treats the unreliable predator as a chronically uncertain threat and over-applies defensive behaviour, including suppressing eating substantially near rabbits. M5_neutral drops below 1.0 (instead of staying above 1.0 as in baseline). This would be a strong confirmation but in a way that suggests the agent's generalisation is over-eager. The designer flags it as "noteworthy if it happens" rather than as the primary predicted outcome.

4. **Predator becomes degenerate / training instability (~5%).** Some episode samples have predator parameters so low that the predator essentially never threatens; the agent's training signal becomes too easy and the policy collapses to a low-effort wanderer. Caught by sanity checks (§5).

The "no generalisation" prior (35%) is non-trivial precisely because of the visual-channel caveat. The designer is not confident this experiment will refute the strong baseline; the experiment is informative either way.

## 3. Experimental Design

### 3.1 Independent variable — the five distributional fields

The predator entry's five behavioural fields move from scalar values (as in Cell C) to per-episode uniform `[low, high]` ranges. Per the v2.0 env refactor, the env draws an independent value for each field from each range at every episode reset; the predator's behaviour stays fixed within an episode.

| Field | Static value in Cell C | Proposed `[low, high]` | Rationale |
|---|---:|---|---|
| `detection_range` (cells) | 5 | **`[0, 10]`** | 0 = predator never spots the agent, so HUNT never triggers — predator looks behaviourally indistinguishable from a wandering rabbit. 10 = predator can detect the agent anywhere on the 10×10 grid — always-chase stalker. Spans the full "wanderer ↔ omniscient stalker" regime. Static 5 is the midpoint. |
| `max_stamina` | 30 | **`[10, 60]`** | 10 = very brief chase before stamina depletes and predator drops to PATROL (gives up quickly). 60 = double today's stamina, can chase for many steps before disengaging. Static 30 is lower-tertile, so the range opens up the "longer-chase" regime. Lower bound stays well above 0 to avoid degenerate "always-out-of-stamina, never chases" episodes. |
| `stamina_recovery_rate` | 1 | **`[0.25, 2.0]`** | 0.25 = post-chase recovery takes ~4× longer than today (predator stays disengaged for many steps). 2.0 = recovers in half the time (can re-engage almost immediately). Static 1 is at the lower-third. Spans the "rare re-engagement" vs "frequent re-engagement" regime. |
| `hunt_stamina_threshold` | 0.7 | **`[0.3, 0.9]`** | 0.3 = predator re-engages HUNT as soon as stamina hits 30% of max (eager). 0.9 = waits until stamina is 90% of max (cautious). Static 0.7 sits inside the range. Spans "almost-always hunting when in detection range" vs "rarely re-engages". |
| `lose_interest_multiplier` | 1.5 | **`[1.0, 3.0]`** | 1.0 = drops the chase as soon as the agent crosses outside `detection_range × 1.0` (gives up at the detection boundary). 3.0 = pursues out to 3× the detection range (very persistent stalker, chases long after losing initial detection). Static 1.5 is lower-tertile. Spans "barely persistent" vs "very persistent". |

**Why these ranges are wide enough.** At the `(detection_range=0, max_stamina=10, lose_interest_multiplier=1.0)` corner the predator is sensorily and behaviourally indistinguishable from a wandering rabbit during the entire episode — HUNT mode never engages, the predator just wanders its patrol box. At the `(detection_range=10, max_stamina=60, lose_interest_multiplier=3.0)` corner the predator is a relentless stalker. The episode-to-episode variation crosses the boundary that today's static `(5, 30, 1.5)` sits roughly in the middle of, so the agent CANNOT reliably learn "the matched-smell entity that chases" as a sufficient discriminator — on a meaningful fraction of episodes there IS no chasing.

**Why these ranges are not too wide.** No lower bound goes to 0 on `max_stamina` (a zero-stamina predator would oscillate in the state machine — undefined behaviour for our purposes); `stamina_recovery_rate` stays positive (`0.25 > 0`); `hunt_stamina_threshold` stays in `(0, 1)`. `move_interval` stays at scalar 1 (the v2.0 refactor explicitly defers this field from the distributional scope), so the predator still moves every step on every episode.

**Static factors preserved from Cell C.** Every other field of the predator entry — `class` (predator), `properties` (matched-smell `[0, 1, 0, 0, 0]`), `properties_std` (zero), `move_interval` (1), `nociception_intensity` (0.9), `damage` (per-event `[15, 45]` — unchanged), `spawn_area` and `patrol_area` (full grid `[[1,1],[10,10]]`), `attack_delay` (3), `tag` (`full`). The two rabbit entries are byte-identical to Cell C (matched-smell `[0, 1, 0, 0, 0]`, no distributional fields, tags `TL` / `BR`). Food, obstacles, sensors, body, perceptual_noise, behavior_measures — all byte-identical to Cell C.

This satisfies the design discipline rule: the new config differs from `02-sameProp_R2_decoupleFood.yaml` **only** in the predator entry's five distributional fields (scalar → `[low, high]` list).

### 3.2 Dependent variables

| Tier | Metric | Why |
|---|---|---|
| **Primary** | **M2 in-cover rate** at threat-onset, per class, plus the class gap `Δ_M2_class = M2_predator − M2_neutral`. | Headline metric of the sameProp study line; +37 pp gap is the seed-locked baseline this experiment tries to reduce. |
| **Secondary** | **M5 eat-suppression ratio** per class. | Cross-validates M2: an avoidance-generalised agent should suppress eating near rabbits more than the baseline does (M5_neutral drops toward 1.0 or below). |
| **Secondary** | **M7 motif distribution** per class (6 k-means clusters). | Qualitative complement — does the agent's defensive *kind* of response change, or just the rate? |
| **Diagnostic** | Mean survival (`Episode/Steps`). | Cross-check the agent didn't degenerate. Cell C baseline ≈ 396 / 500 (training) and 414.7 / 500 (eval). A drop below 350 / 500 would suggest the randomisation made the env meaningfully harder. |
| **Diagnostic** | M1 interrupted-feeding rate per class. | Sanity check (M1 was +18 pp in R2.6; reading its evolution helps interpret M2 / M5 if those land in the borderline band). |
| **Sampling-diagnostic** | `Episode/sampled_*_<tag>` keys, if available in WandB during training. | Verifies the per-episode draws are actually firing during the training run. **NOTE**: see §7 — this WandB logging is not yet wired into `train.py`, even though the env-side sampling is fully active. The primary verdict does not depend on these keys. |

### 3.3 Controls / fixed factors

Inherited from Round 2.6 Cell C (the parity-reference baseline):

- **Smell preset**: matched (`sameProp`) — predator and rabbits both carry `[0, 1, 0, 0, 0]`, std zero.
- **Food layout**: decoupled — food spawns only in predator quadrants (TR, BL); rabbits inhabit TL / BR with no food.
- **Obstacles**: 12 rocks distributed across quadrants; 10 bushes (5 in TR, 5 in BL — `hides_agent: true`).
- **Neutral animals**: 2 rabbits tagged `TL` / `BR`, wander behaviour, no distributional fields.
- **Hiding predators**: 4 static rock-perched predators (one per quadrant), damage `[15, 45]`, nociception 0.9.
- **Sensors**: olfactory radius 20 + visual one-hot + nociception (interoceptive convolution, `tau=3.0`, kernel length 12) + proprioception + collision. **`location_sensor: false`**.
- **Body**: homeostatic reward, death penalty 100, recovery base 0.1 + accel 0.5, injury smoothing 3.
- **Perceptual noise**: disabled (matches Cell C and the rest of the sameProp study line).
- **Agent**: Recurrent PPO via `configs/models/recurrent_ppo/recurrent_ppo.yaml` (the same agent config that produced the seed-locked Round 2.6 result).
- **Step budget**: 10,000,000 episodes (one full run).
- **Parallel envs**: 128.
- **Behavior-measures block**: the same `behavior_measures:` block (with frozen 200-element `eval_seeds` list, `cue_radius=3.0`, `obs_window=5`, `eval_n_episodes=200`, `motif_kmeans_k=6`, `motif_kmeans_seed=42`) inherited byte-identical from Cell C — guarantees cross-cell metric comparability.

### 3.4 Seeds and sample size

**Seed choice.** **Seed 46** (one fresh, not re-using R2.5's 42 or R2.6's 44). Rationale:

- The Round 2.6 Cell C result is *seed-locked* (cross-seed agreement to within 1.2 pp on M2, 0.02 on M5). The baseline is not in doubt; we don't need to re-validate Cell C at seed 46.
- Using a fresh seed avoids "did the random-policy reset land in the same basin?" confound — a different seed makes the experiment's confirmation harder, not easier (a coincidental basin overlap with seed 44 would not be claimable).
- Seeds 42 / 43 / 44 / 45 are all spoken for by prior rounds. 46 is the smallest unused integer in the per-study seed namespace.

**One seed for the primary verdict.** Project convention is "≥ 3 seeds for marginal effects". This is an existence question, not a marginal one — we want to know whether the effect appears at all under behavioural randomisation. **If H₁ confirms** at seed 46 (Δ_M2_class < +20 pp), we run one additional seed (47) to seed-lock before claiming the result; the §4 verdict explicitly states this as a follow-up gate. **If H₀ confirms** (gap persists at ≥ +30 pp), one seed is sufficient given the seed-stability of the baseline (the baseline's +37 pp is so seed-stable that a single-seed +37 pp under this manipulation is strong evidence of class-pinning). **If borderline**, the single-seed result is uninformative — escalation to seed 47 is the §5 resolution.

**Sample size at evaluation.** 200 deterministic episodes on env-seeds 1000–1199 (the toolkit-v1 protocol). With M2 standard error ≈ 1.2 pp at n=200 in Round 2.6 numbers, the +37 pp → ≤ +20 pp drop the H₁ predicate looks for is ≈ 14σ wide — single-seed eval-rollout is statistically more than sufficient to test the threshold.

**Wall-clock estimate.** ~22 h on a 3090-class GPU; ~80 h on a 2080 Ti-class GPU. Matches Round 2.6's actuals (n101's 2080 Ti took 81 h 43 m for the same configuration; n106's 3090 took 22 h).

## 4. Launch Manifest

**Hardware**: 1 node, 1 GPU. Node and GPU to be locked by the user before runner handoff.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | C-dist — predator behavioural randomisation | `hypervigilance-round3-distributional-seed46` | hypervigilance | prod | 46 | — | — | — | — | — |

Tag uniqueness: `hypervigilance-round3-*` prefix collides with neither `hypervigilance-round25-*` nor `hypervigilance-round26-*`. The single-cell single-seed manifest is intentional (one fresh seed, see §3.4); if H₁ confirms, a Run 2 row for `seed47` will be appended as the seed-lock follow-up.

### 4.1 Configs to Produce

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/03-sameProp_R3_predatorDistributional.yaml` (NEW — this design's deliverable) | `configs/models/recurrent_ppo/recurrent_ppo.yaml` (reused, unchanged) |

The env config is the only artifact this experiment ships. It differs from `02-sameProp_R2_decoupleFood.yaml` exclusively in the predator entry's five distributional fields (scalar → `[low, high]` list, per §3.1).

### 4.2 Launch command (for `training-runner` reference)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/03-sameProp_R3_predatorDistributional.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 46 \
  --device cuda:<GPU> \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round3-distributional-seed46" \
  --tag "hypervigilance-round3-distributional-seed46"
```

The user picks `<GPU>` (and the node, via the standing `run_command.py` flow) at hand-off time.

## 5. Pre-Registered Analysis Plan

### 5.1 Primary verdict — applied to the §4 / §4.1 thresholds in §1

Same protocol as Round 2.6 Cell C: `scripts/eval_rollout.py` + `scripts/motif_cluster.py` on the final saved checkpoint, 200 deterministic episodes on env-seeds 1000–1199, R=3.0 cells, K=5 steps, K_motif=7 steps, k-means k=6 seed=42. Output directory: `results/eval/models/<final_step>/`.

| Outcome | Primary thresholds | Implies | Next step |
|---|---|---|---|
| **H₁(generalised-avoidance) confirmed** | `Δ_M2_class < +20 pp` AND (`M2_neutral ≥ 0.65` OR `M5_neutral < 1.05`) | The agent's class-conditional defence in the Round 2.6 baseline was leaning on the predator's behavioural cue. Randomising the cue forces avoidance to generalise toward the matched-smell neutrals. | Launch Run 2 at seed 47 to seed-lock the finding before claiming it. |
| **H₀(class-pinned-avoidance) confirmed** | `Δ_M2_class ≥ +30 pp` AND `M2_neutral < 0.60` | The class discrimination is NOT primarily driven by chase behaviour. Most plausibly leans on the visual one-hot. Single-seed sufficient (given the seed-stability of the +37 pp baseline at R2.6). | Close the round with a verdict; route the "what cue IS the agent using" question forward (vision-channel-blind ablation is the natural next experiment, out of scope here). |
| **Borderline** | `+20 pp ≤ Δ_M2_class < +30 pp` OR (`Δ_M2_class < +20 pp` AND BOTH H₁ secondaries fail) | The single-seed result does not cross either band cleanly. | Escalate to seed 47; re-evaluate after the second seed lands. |
| **Inverted / null** | `Δ_M2_class ≤ 0` (agent dives more for rabbits than predators) OR M2_predator < 0.50 | Either policy collapse or a genuinely surprising finding (e.g., the agent over-generalised so strongly that the visual cue inverted). | Treat as anomalous; route to a senior-developer / experiment-analyzer discussion before any verdict claim. |

### 5.2 Secondary cross-checks

- **Sampling-distribution sanity** (if `Episode/sampled_*_<tag>` keys are available; see §7). Plot the per-episode draws across training. Each field should look uniformly distributed across its range, with no clipping at the bounds and no concentration around the static-Cell-C value. If the distribution is degenerate (sharp peak / no spread), surface as a wiring bug.

- **Cell C → Cell C-dist cross-cell contrast.** Hold the cell intervention constant (sameProp + decoupleFood + all other Cell C factors), vary only the predator's behavioural-field distributional-vs-static treatment. The two reads are:
  - `Δ_M2_class` shift: Cell C (seed 44) `+0.373` → Cell C-dist (seed 46) `<value>`. Magnitude of the shift is the headline number.
  - `M2_neutral` shift: Cell C `0.515` → Cell C-dist `<value>`. A rise here is what the generalised-avoidance hypothesis predicts.

- **Per-tag rabbit fan-out** (M2 and M5, rabbit_TL vs rabbit_BR). Same sanity band as toolkit-v1: ±5 pp on M2, ±0.10 on M5. Rules out single-instance artifacts.

### 5.3 Temporal evolution

Mandatory per project convention. For the new run, partition training into 10 equal-episode windows. Plot:

1. `Episode/Steps` per window — confirms survival doesn't collapse.
2. `Episode/BushDiveRate_predator_full` and `Episode/BushDiveRate_rabbit` per window — confirms the class-conditional defence converges. Compare its trajectory to Cell C's monotonic plateau (Cell C reached its +0.80 / 0.45 plateau by episode 2 M and was stable thereafter).
3. `Episode/EatUnderThreatRatio_predator` and `_rabbit` per window — confirms the M5 ratio converges.
4. `Episode/sampled_*_<tag>` distribution shape across windows, **IF available**. If the keys are not in the WandB log (see §7), this temporal check is dropped; the offline eval is unaffected.

The §5.1 verdict uses the **last-10%-window training-time numbers + the eval-time numbers from the final checkpoint** (same as Round 2.6 §9 / §10). The eval-time numbers are the load-bearing ones for threshold-crossing.

### 5.4 Cross-round contrasts table

| Comparison | What it isolates |
|---|---|
| Cell C-dist (R3, seed 46) vs Cell C (R2.6, seed 44, `ja5fu5k3`) | **Primary contrast.** The role of the predator's chase behaviour as a class-discrimination cue under matched smells. |
| Cell C-dist (R3) per-tag rabbit fan-out | Rules out single-rabbit artifacts (toolkit ±5 pp on M2, ±0.10 on M5). |
| Cell C-dist per-episode `Episode/sampled_*_<tag>` distribution | Sanity: confirms the per-episode draws happened. |
| Cell C-dist motif distribution vs Cell C R2.6 motif distribution | Qualitative complement: does the agent's defensive repertoire reshape (e.g., predator-skewed cluster `predator_pursuit_with_bush` redistributes toward both classes)? |

## 6. Failure-Mode Catalog

Round 2.6 §5 still applies in full (M2 saturation, M5 sanity criteria, eval-rollout reproducibility, etc.). Round-3-specific additions:

| Failure | Resolution |
|---|---|
| **Survival collapses** (`Episode/Steps` < 350 / 500 at convergence). | The randomisation made the env meaningfully harder than Cell C and the agent could not converge. Treat as an "experimental setup failed", not a verdict refutation. Investigate: which corner of the predator-parameter space is unsurvivable, and whether the lower bounds need lifting. |
| **Per-episode sampling did not fire** (every episode's predator parameters land at the static Cell C value, or all five fields look uncorrelated with the agent's behaviour). | Wiring bug — either the v2.0 sampling path is not executing on this config, or `Episode/sampled_*_<tag>` is not being logged. Halt analysis; surface to `senior-developer`. Diagnostic step: load the saved checkpoint, manually call env-reset 100×, inspect `state.animal_detect_sampled[0]` etc., confirm uniform [0, 10] etc. |
| **`Δ_M2_class` lands in the borderline band** (`+20 pp ≤ Δ < +30 pp`). | Pre-registered §5.1 borderline rule: escalate to seed 47. Do not declare H₁ or H₀; the §5.1 outcome cell explicitly forbids a single-seed verdict in this band. |
| **Visual one-hot dependence cannot be ruled out** (H₁ confirms cleanly, BUT the designer-flagged "vision still channel-codes class" caveat is not separable from the avoidance shift in this experiment alone). | This is not a failure — it's a successor question. If H₁ confirms, the next natural experiment is to ablate the visual channel-class coding (channel 5 = channel 7) and re-test; that's a Round 4 design, not a Round 3 resolution. Flag in the §6 conclusions. |
| **Agent saturates at M2_predator ≈ 1.0** in randomised episodes. | If `Δ_M2_class` shrinks largely because `M2_neutral` rises (not because `M2_predator` drops), the ceiling effect on the predator side is not a concern. If `M2_predator` also drops to ≤ 0.80, flag as "the agent reduced predator-side defence" — this is informative; report alongside the headline number. |
| **Per-episode sampling triggers JIT recompile** (`Episode/Steps`/it drops by > 5%). | The v2.0 CP5 implementation report claims no recompile for bounds-only configs. If the s/it drops, surface as a v2.0 regression; do not let it block the verdict (offline eval is unaffected). |

## 7. Metrics Requested

**One known gap, flagged for the user.** The v2.0 refactor (CP5) added `build_episode_log_dict()` and `sampled_wandb_keys()` in `src/behavior/accumulators.py` to emit `Episode/sampled_detect_<tag>` / `_max_stamina_<tag>` / `_recovery_<tag>` / `_hunt_thresh_<tag>` / `_lose_interest_<tag>` per episode. These functions exist but **are not yet wired into `train.py`'s training loop** (verified by `grep -rn "build_episode_log_dict" train.py src/ | grep -v accumulators` → no callers).

**Impact on this experiment**: zero on the primary verdict — the per-episode random sampling in the env happens fully (it's `jax_reset` code, fully integrated in CP1–CP5); only the *WandB logging* of the sampled values is missing. The eval-rollout-time M2/M5/M7 calculations do not depend on these keys; they read from per-step state via `scripts/eval_rollout.py`, which works against any saved checkpoint.

**What's lost**: the training-time temporal evolution check on `Episode/sampled_*_<tag>` distributions (§5.3 row 4). That check would have answered "did the per-episode draws actually look uniform across training?" from inside WandB; without it, the only sanity is the offline checkpoint-replay diagnostic described in the §6 "per-episode sampling did not fire" row.

**Recommendation to the user**: this is a small developer follow-up (a few lines in the train.py episode-done branch to call `build_episode_log_dict(state, env_params)` and pass it to `wandb.log`). NOT urgent — does not block this experiment's launch. If the user wants it before launching, route it through `feature-workflow` (senior-developer plans, developer implements) as a small ticket. If not, launch proceeds without it and we rely on the offline diagnostic.

| Subfield | Content |
|---|---|
| **Metric** | `Episode/sampled_detect_<tag>`, `Episode/sampled_max_stamina_<tag>`, `Episode/sampled_recovery_<tag>`, `Episode/sampled_hunt_thresh_<tag>`, `Episode/sampled_lose_interest_<tag>` (5 keys × N animals — for this config, 1 predator + 2 rabbits, so 15 keys total though rabbits' values stay at degenerate `[0,0]`). |
| **Why now** | Verifies the per-episode draws are firing during training. Used by §5.3 temporal check and §6 sampling-sanity. |
| **Where it'd live** | `train.py` — call `build_episode_log_dict(state, env_params)` at episode-done, merge into the per-episode WandB log dict alongside `Episode/Steps`, `Episode/Term_*` etc. |
| **Cost** | Cheap — one host-side dict-build call per episode-done event, already implemented in `accumulators.py:492`. |

## 8. Launchable Status

**Configs**: ready (this design ships the new env config; agent config is reused).

**Pre-flight gates** (must pass before runner launches):
1. `env-config-auditor` audit of the new config (`03-sameProp_R3_predatorDistributional.yaml`). The auditor must confirm: byte-identical to Cell C except the five distributional fields; the `behavior_measures:` block frozen `eval_seeds` is the same 200-element list as Cell C; the `predator_enabled` key (removed in v2.0 CP1) is absent.
2. User decision on the §7 metrics question — proceed without `Episode/sampled_*_<tag>` (offline diagnostic only), or backfill the train.py wiring first.
3. User authorization — explicit go after auditor sign-off and seed/node/GPU lock.

**Runner handoff**: `training-runner` fills the actual columns of §4 (Status, Node, GPU, Launched at, WandB run ID, Log path) at launch time. Per the diary protocol the runner calls `diary training-start --tag hypervigilance-round3-distributional-seed46 --node <N> --gpu cuda:<G> --cell C-dist --wandb <id> --doc docs/experiments/active/hypervigilance/sameprop_predator_distributional.md`.

---

## 9. Results

To be filled by `experiment-analyzer` after training completes.

## 10. Analysis

To be filled by `experiment-analyzer` after training completes.

## 11. Conclusions

To be filled by `experiment-analyzer` after training completes.
