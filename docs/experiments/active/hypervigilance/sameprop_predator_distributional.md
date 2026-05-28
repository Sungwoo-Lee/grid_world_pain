---
title: "SameProp Round 3 — predator behavioural randomisation: does avoidance generalise to neutrals when the predator's chase cue is unreliable?"
topic: hypervigilance
status: active
created: 2026-05-28
last_updated: 2026-05-29
phase: 2
wandb_tag: "hypervigilance-round3-distributional"
supersedes: []
---

> **Status banner — 2026-05-29 02:30 KST.** MID-TRAINING PREVIEW at iter 36,144 / ~64,000 expected (episode 1,810,043 / 10,000,000 — 18.1 % of budget). The training run is still active on n113:GPU0 (PID 655905, launched 2026-05-28 16:18 KST). §§9–11 below report a preliminary readout to give the operator an early signal; the closing analysis will use the post-training checkpoint and may move the numbers materially. **Do not treat this as the closing verdict.**

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

## 9. Results — Cell C-dist seed 46, MID-TRAINING PREVIEW (ckpt 1,810,043 / 10 M, 0ikqqpvc)

### 9.0 What this section reports — plain-English entry point

The Cell C-dist run on seed 46 (the run that asks "if we randomise the predator's chase behaviour per episode, does the agent's predator-vs-rabbit class gap collapse?") is still training on n113's GPU at the time of this report — it has completed about 1.8 million of its planned 10 million episodes (18 % of budget) and will keep training for ~2–3 more days. The numbers below are computed off the **latest fully-written checkpoint** (episode 1,810,043, iteration 36,144), saved at 02:06 KST on 2026-05-29. The training run was NOT interrupted to produce this preview; the eval harness loads the saved checkpoint offline.

Two readouts: the first ("training-time last-20 %") is taken from the WandB log over the run's most recent ~370 K episodes, averaged across all 128 parallel envs as the run logs them; the second ("eval-time, deterministic policy, n = 200 episodes") replays the saved 1.81 M-episode checkpoint in a fresh evaluation harness with exploration off, using the same `scripts/eval_rollout.py` + `scripts/motif_cluster.py` pipeline and the same 200 frozen `eval_seeds` (the byte-identical list inherited from Cell C). The eval-time numbers are the apples-to-apples comparator against Round 2.6 seed 44; the §10 verdict applies the §1 / §5.1 thresholds to the eval-time row.

**The headline preview.** At 18 % of training the agent has already learned a substantial class gap — its in-cover rate at predator-onset is 69.3 %, at rabbit-onset 34.3 %, giving Δ_M2_class = +35.1 percentage points. The class gap is still LARGER than the §1 H₀-refutation floor of +30 pp; the agent is well below the +20 pp H₁-confirmation ceiling. Both M2 numbers are still climbing toward the R2.6 baseline (predator 88.8 %, rabbit 51.5 %), so the gap may shrink, hold, or widen as training continues. The closing-analysis verdict is NOT determined by this preview.

### 9.1 Run actuals (mid-training)

| Cell | Tag | WandB | Seed | Episodes (logged) | Iterations | Checkpoint analyzed | Wall-clock at preview |
|---|---|---|---|---:|---:|---|---:|
| C-dist — predator behavioural randomisation | `hypervigilance-round3-distributional-seed46` | [`0ikqqpvc`](https://wandb.ai/sungwoolee/grid_world_pain/runs/0ikqqpvc) | 46 | 1,820,035 (training) | 36,371 | episode 1,810,043 / iter 36,144 | ~10 h 12 m |

The training process (PID 655905 on n113) is still active and writing further checkpoints (1,820,035 finalized at 02:10; preview eval used 1,810,043 finalized at 02:06, one revision behind to ensure no race with orbax's atomic write). The 10 h-elapsed → 1.82 M-episode pace projects to ~56 h total wall-clock to 10 M episodes, which is roughly consistent with the n106 RTX 3090 baseline of ~22 h for Cell C (n113 is currently logging ~3.3 it/s based on the iteration-vs-elapsed math, somewhat slower than n106). The current pace puts ETA at ~2026-05-31.

### 9.2 Training-time last-20 % window readout (episodes ≈ 1.45 M – 1.82 M, n = 741 records)

Means ± std across the 741 last-20 %-window WandB log records. Cells annotated **bold** are the §1 / §5.1 primary verdict measures (training-time variant — useful as an early-warning signal; the §10 verdict uses the §9.3 eval-time numbers).

| Metric | R3 C-dist seed 46 (last-20 %, mid-train) | R2.6 C seed 44 (last-10 %, fully trained, §9.2 of `sameprop_round26_design.md`) |
|---|---:|---:|
| `Episode/Steps` (survival) | **371.7 ± 4.3** | 396.0 ± 4.5 |
| `Episode/Term_MaxSteps` | 0.485 ± 0.014 | 0.487 ± 0.017 |
| `Episode/Term_Injury` | 0.289 ± 0.032 | 0.276 ± 0.024 |
| `Episode/Term_Starvation` | 0.226 ± 0.030 | 0.239 ± 0.028 |
| `Episode/MeanDistRabbit` | 4.561 ± 0.024 | 4.587 ± 0.024 |
| `Episode/MeanDistPredator` | 4.210 ± 0.055 | 4.068 ± 0.049 |
| `Episode/MeanDistRabbit_TL` | 5.540 ± 0.041 | 5.537 ± 0.035 |
| `Episode/MeanDistRabbit_BR` | 5.778 ± 0.038 | 5.808 ± 0.037 |
| `Episode/MeanDistPredator_full` | 4.210 ± 0.055 | 4.068 ± 0.049 |
| `Episode/RabbitHits` | 0.956 ± 0.081 | 0.916 ± 0.070 |
| `Episode/PredatorHits` | 2.915 ± 0.170 | 3.315 ± 0.177 |
| `Episode/HidingPredatorHits` | 2.636 ± 0.099 | 2.527 ± 0.077 |
| `Episode/FoodEaten` | 72.92 ± 1.87 | 71.6 ± 1.4 |
| `Episode/Reward` | −176.5 ± 1.6 | −184.9 ± 1.7 |
| **`Episode/BushDiveRate_predator_full`** | **0.622 ± 0.008** | **0.804 ± 0.007** |
| **`Episode/BushDiveRate_rabbit`** | **0.350 ± 0.010** | **0.454 ± 0.011** |
| `Episode/BushDiveRate_rabbit_TL` | 0.432 ± 0.012 | 0.548 ± 0.013 |
| `Episode/BushDiveRate_rabbit_BR` | 0.409 ± 0.015 | 0.530 ± 0.015 |
| **`Episode/EatUnderThreatRatio_predator`** | **0.961 ± 0.041** | **0.931 ± 0.030** |
| **`Episode/EatUnderThreatRatio_rabbit`** | **1.274 ± 0.051** | **1.386 ± 0.046** |
| `Episode/EatUnderThreatRatio_rabbit_TL` | 1.242 ± 0.060 | 1.353 ± 0.060 |
| `Episode/EatUnderThreatRatio_rabbit_BR` | 1.237 ± 0.069 | 1.454 ± 0.069 |

The mid-training fingerprint vs the fully-trained R2.6 baseline shows the agent is **on the same monotonic trajectory the baseline followed but not yet plateaued**: both `BushDiveRate_predator_full` (0.622 vs the baseline plateau 0.804) and `BushDiveRate_rabbit` (0.350 vs 0.454) are still rising; their training-time Δ is +0.272 (preview) vs +0.350 (baseline). Survival is 24 steps below baseline (371.7 vs 396.0) — consistent with both (a) the harder env from randomised predator behaviour and (b) the policy still in its convergence phase. The training-time M5_predator at 0.961 is well above the R2.6 baseline's 0.931 and far above the toolkit's < 0.95 H₁(C-event) sub-threshold — the eat-suppression near predator is NOT yet established. (R2.6's training-time M5 also stayed above the eval-time M5 of 0.769 — see §10 of the R2.6 closing analysis.)

### 9.3 Derived primary statistics (eval-time, deterministic policy, n = 200 episodes)

Run via `scripts/eval_rollout.py` (200 deterministic episodes on the byte-identical R2.6 eval_seeds list, exploration off, host CPU, 265.2 s) + `scripts/motif_cluster.py` (k = 6, seed = 42, 10 features, zscore_pooled). Eval root: [`results/eval/models/1810043/`](../../../../results/eval/models/1810043) (script convention `<out_root>/models/<ckpt_name>/`). Git commit at eval time: `e3e6789`. Per-class headline numbers (cross-tab from [`tmp/20260529_0212_r3_seed46_eval_analysis.py`](../../../../tmp/20260529_0212_r3_seed46_eval_analysis.py)):

| Measure | R3 C-dist seed 46 (MID-TRAINING, ckpt 1.81 M) | R2.6 Cell C seed 44 (10 M reference) | R2.5 Cell C seed 42 (10 M reference) | Δ vs R2.6 |
|---|---:|---:|---:|---:|
| **M2** bush-dive rate, predator | **69.3 %** (1752 / 2527) | 88.8 % (2529 / 2849) | 87.6 % (2222 / 2537) | **−19.5 pp** |
| **M2** bush-dive rate, rabbit (aggregated) | **34.3 %** (881 / 2560) | 51.5 % (897 / 1741) | 50.8 % (851 / 1674) | **−17.2 pp** |
| **Δ_M2_class ≡ M2_pred − M2_rab** | **+35.1 pp** | **+37.3 pp** | **+36.8 pp** | **−2.3 pp** |
| **M5** eat-under-threat ratio, predator | **0.560** | 0.769 | 0.748 | **−0.209** |
| **M5** eat-under-threat ratio, rabbit | **1.124** | 1.202 | 1.186 | −0.078 |
| M1 interrupted-feeding rate, predator | 37.9 % (1314 / 3465) | 42.7 % (2151 / 5043) | 42.2 % (1926 / 4564) | −4.8 pp |
| M1 interrupted-feeding rate, rabbit | 19.6 % (567 / 2898) | 23.8 % (579 / 2432) | 24.1 % (603 / 2499) | −4.2 pp |
| Δ_M1_class | +18.3 pp | +18.9 pp | +18.1 pp | −0.6 pp |
| Mean survival (eval, n = 200) | 380.5 ± 142.5 | 414.7 ± 118.3 | 405.4 | −34 / +24.2 std |

The eval-time Δ_M2_class at 18 % of training is **+35.1 pp**, sitting just below the seed-locked R2.6 baseline (+37.3 pp at seed 44, +36.8 pp at seed 42) and well above the §1 H₀-refutation floor of +30 pp. Both M2 sides are 17–20 pp BELOW the baseline plateau — the agent has not finished training its in-cover defence. Critically: M2_predator and M2_rabbit are dropping by **approximately equal amounts** vs baseline (−19.5 pp and −17.2 pp respectively), so the class gap is preserved at the mid-training waypoint rather than narrowing. The eval-time M5_predator at 0.560 is even *more* suppressed than the baseline's 0.769 — the agent is eating less near the predator than the fully-trained R2.6 agent does, which is consistent with the harder env raising the marginal cost of food-grabbing near a (possibly stalking) predator.

Per-tag fan-out (the §5.2 secondary check):

| Tag | M1 | M2 | M5 ratio | R2.6 §9.3 reference (seed 44) |
|---|---:|---:|---:|---|
| predator_full | 37.9 % (1314 / 3465) | **69.3 % (1752 / 2527)** | **0.560** | M1 42.7 %, M2 88.8 %, M5 0.769 |
| rabbit_TL | 20.0 % (366 / 1826) | 33.6 % (504 / 1501) | 1.085 | M1 21.7 %, M2 49.6 %, M5 1.255 |
| rabbit_BR | 18.7 % (219 / 1170) | 35.6 % (377 / 1059) | 1.165 | M1 27.7 %, M2 53.3 %, M5 1.116 |

| Metric | rabbit_TL | rabbit_BR | gap | §5.2 ± band | Status |
|---|---:|---:|---:|---|---|
| M2 bush-dive rate | 33.6 % | 35.6 % | **2.0 pp** | ±5 pp | ✓ within band |
| M5 eat-under-threat ratio | 1.085 | 1.165 | **0.080** | ±0.10 | ✓ within band |

Per-tag fan-out is healthy at mid-training: the two rabbit tags agree to 2.0 pp on M2 and 0.080 on M5, BOTH within the toolkit-v1 fan-out band. (Notably the R2.6 reference §9.3 was 3.7 pp on M2 — within band — and 0.139 on M5 — marginally over the 0.10 band. The R3 preview is currently *tighter* than R2.6 on both per-tag fan-outs, but the M5 ratios themselves are not yet at the baseline magnitudes.)

### 9.4 Temporal evolution — 10 equal-episode windows across training-so-far

Mandatory per project convention. Windowed means across episodes 10,170 → 1,850,000 (n = 74 records per window for windows 1–9; n = 76–79 for window 10). The R3 training-time numbers:

| Window | ep range (M) | n | `BushDiveRate_predator_full` | `BushDiveRate_rabbit` | `EatUnderThreatRatio_predator` | `Steps` | `Term_MaxSteps` | `MeanDistPredator` |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 – 0.30 | 74 | 0.436 | 0.236 | 0.902 | 226.0 | 0.146 | 4.87 |
| 2 | 0.31 – 0.49 | 74 | 0.503 | 0.310 | 0.906 | 320.6 | 0.359 | 4.40 |
| 3 | 0.50 – 0.67 | 74 | 0.551 | 0.346 | 0.944 | 334.9 | 0.399 | 4.31 |
| 4 | 0.68 – 0.85 | 74 | 0.583 | 0.376 | 0.967 | 347.6 | 0.428 | 4.28 |
| 5 | 0.85 – 1.02 | 74 | 0.595 | 0.385 | 0.989 | 355.1 | 0.445 | 4.27 |
| 6 | 1.02 – 1.19 | 74 | 0.612 | 0.401 | 1.001 | 363.5 | 0.464 | 4.27 |
| 7 | 1.19 – 1.35 | 74 | 0.619 | 0.407 | 0.989 | 366.0 | 0.470 | 4.25 |
| 8 | 1.35 – 1.52 | 74 | 0.624 | 0.420 | 0.979 | 368.0 | 0.475 | 4.20 |
| 9 | 1.52 – 1.68 | 74 | 0.623 | 0.421 | 0.971 | 371.7 | 0.487 | 4.22 |
| 10 | 1.68 – 1.85 | 79 | 0.622 | 0.408 | 0.952 | 371.7 | 0.483 | 4.20 |

Two readings worth highlighting:

1. **The class-conditioned defence on M2 has plateaued in the last 3 windows on the predator side and the rabbit-aggregated side**, but at a level meaningfully below the R2.6 baseline plateau. `BushDiveRate_predator_full` over windows 8 / 9 / 10 = 0.624 / 0.623 / 0.622 (spread 0.002, within window σ ≈ 0.008) — the plateau is real at the training-time level. `BushDiveRate_rabbit` over windows 8 / 9 / 10 = 0.420 / 0.421 / 0.408 (spread 0.013, σ ≈ 0.011 — borderline but window 10 still has 76 records, smaller drop likely sampling). The R2.6 baseline plateau was 0.798 / 0.802 / 0.804 on predator and 0.450 / 0.454 / 0.454 on rabbit. **The R3 mid-training plateau is ~0.18 below R2.6 on predator and ~0.03 below on rabbit.** Whether this is a true policy plateau or a longer-trajectory inflection (with the R3 agent still on a path toward eventual baseline-magnitude) is the central open question for the closing analysis.

2. **Survival is also approaching its plateau**: windows 8 / 9 / 10 = 368.0 / 371.7 / 371.7 (spread 3.7, within-window σ ≈ 4.0). At convergence the R3 agent looks set to stabilise around 372 / 500 steps, ~24 steps below the R2.6 baseline. `Term_MaxSteps` over the same windows = 0.475 / 0.487 / 0.483 — flat to within noise, suggesting the policy is genuinely converged on its current operating point. **This is the survival-step plateau the user flagged in the brief.**

The M5_predator training-time number on the other hand is NOT yet plateaued — windows 6/7/8/9/10 = 1.001 / 0.989 / 0.979 / 0.971 / 0.952 — it has been monotonically decreasing for the last 5 windows, dropping by ~0.012 / 0.18 M episodes. Whether it continues to drop toward baseline's training-time 0.93 (which would still leave eval-time M5 in the 0.7–0.8 range like baseline) or whether it has another inflection is the second open question.

### 9.5 Diagnostic — did the per-episode predator sampling fire?

The design doc §7 noted that the v2.0 CP5 implementation deferred the `Episode/sampled_*_<tag>` WandB logging, so no training-time per-episode sampled-value distribution is available. As a substitute, I computed the **per-episode predator threat-fraction** (fraction of episode steps within R = 3.0 of the patrol predator) across all 200 eval episodes for R3 (this checkpoint) and R2.6 seed-44 reference:

| Run | n | mean | std | min | max | Range |
|---|---:|---:|---:|---:|---:|---:|
| R2.6 C seed 44 (static predator) | 200 | 0.415 | 0.148 | 0.066 | 0.869 | 0.803 |
| R3 C-dist seed 46 (mid-training) | 200 | 0.373 | **0.219** | **0.018** | **1.000** | **0.982** |

The R3 per-episode threat-fraction spread is **48 % wider in standard deviation and 22 % wider in range** than the static-predator R2.6 reference. The R3 distribution has at least one episode where the agent spent <2 % of steps near the predator (predator effectively never engaged HUNT) and at least one where the agent spent 100 % of steps in threat range (predator never disengaged). The static-predator R2.6 baseline cannot reach either extreme — its variation comes purely from spawn-position randomness and within-episode chase dynamics. **The widened distribution is a strong indirect signal that the per-episode `[low, high]` sampling on `detection_range`, `max_stamina`, etc. IS firing as designed.** This satisfies the §6 row "per-episode sampling did not fire" diagnostic via behavioural-trace proxy.

### 9.6 Motif distribution (M7)

200 episodes × ~3 threat-onsets per episode → 6,950 motif windows; silhouette mean = 0.180; the six k-means clusters partition predator vs rabbit windows as:

| cluster | size | frac | net_disp | path_len | min_threat_dist | bush_occ | eat_per_win | predator-frac | rabbit-frac |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1252 | 0.180 | 2.00 | 5.83 | 1.19 | 0.300 | 1.71 | **0.770** | 0.230 |
| 1 | 876 | 0.126 | 0.41 | 0.68 | 1.97 | 0.745 | 0.41 | 0.406 | 0.594 |
| 2 | 1711 | 0.246 | 0.82 | 3.00 | 1.30 | 0.544 | 2.19 | **0.699** | 0.301 |
| 3 | 1475 | 0.212 | 1.44 | 4.45 | 2.14 | 0.344 | 1.97 | 0.335 | **0.665** |
| 4 | 1050 | 0.151 | 0.56 | 0.83 | 2.04 | 0.124 | **4.79** | 0.250 | **0.750** |
| 5 | 586 | 0.084 | 5.29 | 7.52 | 1.49 | 0.136 | 0.64 | 0.355 | 0.645 |

The predator-skewed clusters are 0 ("moderate motion, low bush, moderate eating, close threat distance" — predator 77 %) and 2 ("low motion, in-bush, high eating, close threat" — predator 70 %); the rabbit-skewed clusters are 3 ("moderate motion, low bush, high eating, mid threat distance" — rabbit 67 %) and 4 ("low motion, very high eating, mid threat distance" — rabbit 75 %). The same qualitative split as R2.6 (the R2.6 §12 motif appendix had ~80/20 predator-skewed clusters 0 / 2 and ~70/30 rabbit-skewed clusters 3 / 4 — close to but not identical with R3 mid-training). The motif fingerprint at 18 % of training already shows a class-discriminating pattern, but the predator-skewed clusters are less "pure" than the baseline (R3 cluster 0 is 77 % predator vs R2.6's typical ~80 %+) — consistent with the agent's M2 numbers still climbing.

Per-class motif distribution (row-normalised):

| triggering_class | cluster_0 | cluster_1 | cluster_2 | cluster_3 | cluster_4 | cluster_5 |
|---|---:|---:|---:|---:|---:|---:|
| predator | **0.277** | 0.102 | **0.344** | 0.142 | 0.075 | 0.060 |
| rabbit | 0.083 | 0.150 | 0.148 | **0.283** | **0.227** | 0.109 |

The two predator-skewed clusters (0, 2) together capture 62 % of predator threat-onset windows; the two rabbit-skewed clusters (3, 4) together capture 51 % of rabbit threat-onset windows. The qualitative agent-response separation by class is intact at mid-training.

---

## 10. Analysis — applying §1 / §5.1 thresholds to §9 numbers (mid-training preview)

### 10.0 What this section does — plain-English entry point

§9 reported the numbers; §10 applies the pre-registered §5.1 verdict thresholds — *with the explicit caveat that this is a mid-training preview at 18 % of training, not the closing-analysis verdict*. The §5.1 confirmation criteria for Cell C-dist are written as two primary thresholds on the in-cover-rate measure: H₁ confirmation (avoidance generalised) needs `Δ_M2_class < +20 pp` AND `M2_neutral ≥ 0.65` OR `M5_neutral < 1.05`; H₀ confirmation (avoidance class-pinned) needs `Δ_M2_class ≥ +30 pp` AND `M2_neutral < 0.60`. Each row below holds an R3 eval-time number from §9.3 against its locked threshold and reports which band it falls in *at this waypoint*.

### 10.1 Pre-registered band classification at the mid-training preview

| §5.1 outcome | Primary thresholds | R3 eval-time observed (ckpt 1.81 M) | Within band? |
|---|---|---:|---|
| **H₁(generalised-avoidance) confirmed** | `Δ_M2_class < +20 pp` AND (`M2_neutral ≥ 0.65` OR `M5_neutral < 1.05`) | Δ_M2_class = **+35.1 pp**; M2_neutral = 0.343; M5_neutral = 1.124 | ✗ Δ_M2_class is +15 pp ABOVE the < +20 pp ceiling; M2_neutral is 0.31 below the ≥ 0.65 floor; M5_neutral is 0.07 ABOVE the < 1.05 ceiling. |
| **H₀(class-pinned-avoidance) confirmed** | `Δ_M2_class ≥ +30 pp` AND `M2_neutral < 0.60` | Δ_M2_class = **+35.1 pp** (≥ +30); M2_neutral = **0.343** (< 0.60) | ✓ both predicates satisfied. |
| **Borderline** | `+20 pp ≤ Δ_M2_class < +30 pp` | Δ_M2_class = +35.1 pp (above the borderline band) | ✗ |
| **Inverted / null** | `Δ_M2_class ≤ 0` OR `M2_predator < 0.50` | Δ_M2_class = +0.351 (positive, not ≤ 0); M2_predator = 0.693 (≥ 0.50) | ✗ |

**At the mid-training waypoint the data sits in the H₀(class-pinned-avoidance) band.** Both predicates clear with margin: Δ_M2_class +35.1 pp is 5.1 pp past the +30 pp floor; M2_neutral 0.343 is 0.257 (76 % of the way to zero) under the 0.60 ceiling. The agent is, at this point, NOT generalising avoidance to the matched-smell rabbits despite the predator's chase behaviour being randomised episode-by-episode.

**But — this is mid-training.** The honest hedge: the R2.6 baseline's training-time toolkit numbers, by analogous mid-training windowing, would have looked even *more* extreme at 18 % of training (R2.6 didn't have toolkit-v1 logged at 18 % of its R2.5 run, but at R2.6 window 2 of 10 — ~1.34 M episodes — `BushDiveRate_predator_full = 0.768`, vs R3 window 10 of 10 = 0.622; R3's predator-side is currently more like R2.6's mid-training pre-plateau, not post-plateau). The interpretation is **NOT** "H₀ is confirmed and the experiment is done"; it is "the agent has NOT YET reached a generalised-avoidance policy". The remaining 8.2 M episodes of training could move the eval-time numbers materially. The two open empirical questions for the closing analysis:

1. Does `M2_predator` continue to climb toward 0.80–0.90, and if so, does `M2_neutral` climb *with it* (closing some of the gap) or *less than it* (preserving the gap)?
2. Does `M5_predator` (currently 0.560 — already MORE suppressed than R2.6's 0.769) stay below 0.80 — providing strong H₁-secondary signal — or does it relax back toward 0.77 as the agent's exploration ramps further down?

### 10.2 §5.3 temporal-stability check (windows 8, 9, 10)

The §5.3 stability check asks whether the primary metric is stable across windows 8, 9, 10 — a precondition for the verdict to be taken seriously. At the mid-training waypoint these three windows correspond to episodes 1.35 M → 1.85 M (the last 26 % of training-so-far).

| Measure | Window 8 | Window 9 | Window 10 | Spread | Within-window σ | Stable? |
|---|---:|---:|---:|---:|---:|---|
| `BushDiveRate_predator_full` | 0.624 | 0.623 | 0.622 | 0.002 | 0.008 | ✓ |
| `BushDiveRate_rabbit` | 0.420 | 0.421 | 0.408 | 0.013 | 0.011 | ⚠ borderline |
| `EatUnderThreatRatio_predator` | 0.979 | 0.971 | 0.952 | 0.027 | 0.039 | ✓ but with downward trend |
| `Steps` | 368.0 | 371.7 | 371.7 | 3.7 | 4.0 | ✓ |
| `Term_MaxSteps` | 0.475 | 0.487 | 0.483 | 0.012 | 0.013 | ✓ |

The predator-side BushDive is rock-stable to ±0.002 (well below the within-window σ of 0.008), survival is stable to ±3.7 (within σ = 4.0), MaxSteps termination stable to within σ. The rabbit-side BushDive is borderline: the window-10 dip (0.408) is 0.012 below window 9 (0.421); within-window σ is 0.011; this could be a stochastic dip or the early sign of a regression. M5_predator is monotonically decreasing across the last 5 windows (1.001 → 0.989 → 0.979 → 0.971 → 0.952), which is the *good* direction but means the metric is NOT yet plateaued.

**The mid-training plateau is partial.** The agent's survival policy is stable; the predator-side defence is stable; the rabbit-side defence and the eat-suppression metric are still moving. This is consistent with the user's report that survival has plateaued but loss may still be decreasing slowly — the operative interpretation is "the policy's gross structure is set, fine-grained class-conditional behaviour is still being learned".

### 10.3 §5.2 cross-cell + cross-round contrasts — what the mid-training preview locks in

The §5.2 contrast table predicted that "C-dist (R3, seed 46) vs C (R2.6, seed 44, `ja5fu5k3`)" should isolate the role of the predator's chase behaviour as a class-discrimination cue. Observed at mid-training:

| Quantity | R3 C-dist seed 46 (mid-train, ckpt 1.81 M) | R2.6 C seed 44 (10 M baseline) | Δ |
|---|---:|---:|---:|
| Eval-time M2 predator | 69.3 % | 88.8 % | −19.5 pp |
| Eval-time M2 rabbit | 34.3 % | 51.5 % | −17.2 pp |
| Eval-time Δ_M2_class | **+35.1 pp** | **+37.3 pp** | **−2.3 pp** |
| Eval-time M5 predator | 0.560 | 0.769 | −0.209 |
| Eval-time M5 rabbit | 1.124 | 1.202 | −0.078 |
| Mean survival (eval, n = 200) | 380.5 ± 142.5 | 414.7 ± 118.3 | −34 steps |

Three observations:

- **Both M2 sides drop by approximately equal amounts** (−19.5 pp predator, −17.2 pp rabbit) vs the baseline, so the class gap is preserved (−2.3 pp from baseline). Whatever discriminative cue the R3 agent is using is being applied with similar weight on both classes, just at a lower absolute defence level (consistent with the policy still in training).
- **M5_predator drops harder than M2_predator does** (−0.21 on a 0.77 baseline = 27 % drop, vs M2's 22 % drop). The agent at mid-training is MORE eat-suppressed near the predator than the fully-trained R2.6 agent is. This is the interesting tell — under behavioural randomisation, the predator becomes a "more reliably-dangerous-when-present" cue (because the agent has learned across all the randomised episodes that the predator is sometimes a stalker), and the agent responds by eating less when the predator is anywhere within R = 3.0.
- **Survival drops by 34 steps** (8 % below baseline) — consistent with the harder env. The survival std is wider (142.5 vs 118.3), reflecting episode-by-episode variation in predator difficulty.

The headline mid-training reading is that the **predator-side defence is muted but the class gap is mostly preserved**. The §0 plain-English question — "does the class gap collapse from +37 pp toward ≤ +20 pp?" — has its mid-training answer "no, the gap is still +35 pp; it has shrunk by only 2.3 pp", which sits in the H₀ band of the §5.1 verdict table. Whether it collapses further as training continues is the open empirical question.

### 10.4 Did the random-predator-behaviour cue make the agent more or less defensive overall?

The user's brief asked the sub-question explicitly: did the distributional behaviour make the agent (a) MORE defensive both sides, (b) LESS defensive both sides, (c) narrowed the class gap, or (d) some other reshape?

Reading the §10.3 contrast: **both sides are LESS defensive on M2** (predator −19.5 pp, rabbit −17.2 pp) than the fully-trained R2.6 baseline. **Predator-side M5 is MORE defensive** (eat-suppression 0.560 vs baseline 0.769 — eating is HALF the safe-baseline rate, vs three-quarters in baseline). The class gap shape is preserved (Δ_M2 = +35.1 pp vs +37.3 pp baseline; Δ_M5 = +0.564 vs +0.433 baseline — the M5 class gap actually WIDENED by 0.13 because predator-side M5 dropped more than rabbit-side M5 did).

The most useful summary of the mid-training shape: **the agent has a more aggressive predator-side eat-suppression and a less complete bush-dive defence on both classes, with the predator-vs-rabbit gap on the in-cover measure essentially preserved**. The interpretation is that randomised predator behaviour pushed the agent into a "stay away from food when predator is nearby" defence (which is class-conditional on the visual channel — see §0 caveat) but the agent hasn't yet trained the bush-dive arm of the defence to its full magnitude.

### 10.5 §6 failure-mode catalog — which rows fired at mid-training?

- **§6 "Survival collapses < 350 / 500"**: training-time `Episode/Steps` = 371.7, eval-time mean = 380.5 — both ABOVE the 350 floor. The mid-training agent has NOT collapsed. **Did not fire.**
- **§6 "Per-episode sampling did not fire"**: §9.5 behavioural-trace diagnostic (per-episode predator threat-fraction std 0.219 vs baseline 0.148, 48 % wider; range [0.018, 1.000] vs baseline [0.066, 0.869]) is strong indirect evidence that the per-episode sampling IS firing. **Did not fire.**
- **§6 "Δ_M2_class lands in the borderline band"**: Δ_M2_class = +35.1 pp, NOT in the [+20, +30] band. **Did not fire** at mid-training (could fire at the closing analysis if the gap shrinks below +30).
- **§6 "Visual one-hot dependence cannot be ruled out"**: the H₁ has not (yet) confirmed, so this row's "successor question" path is not triggered. If the closing analysis ends in H₀, the visual-channel caveat in §0 becomes the load-bearing follow-up question. **Latent.**
- **§6 "Agent saturates at M2_predator ≈ 1.0"**: M2_predator = 0.693, far from 1.0. **Did not fire.**
- **§6 "Per-episode sampling triggers JIT recompile"**: training is logging 3.3 it/s (estimated), and the run started cleanly (no recompile spikes visible in the WandB log). **Did not fire.**

---

## 11. Conclusions

### 11.0 What this section says — plain-English entry point

This section records the **mid-training preview verdict for the R3 distributional-predator experiment at ~18 % of its 10-million-episode training budget**, NOT the closing verdict. The closing analysis will be re-run on the post-training checkpoint and may move the numbers in either direction. With those caveats stated up front:

**Plain-English preview.** At 1.81 M episodes of training, the agent has NOT generalised its avoidance to the matched-smell rabbits despite the predator's chase behaviour being randomised episode-by-episode. Its in-cover rate at predator-onset is 69.3 %; at rabbit-onset 34.3 %. The class gap on the headline in-cover measure is +35.1 percentage points, almost identical in magnitude to the seed-locked +37 pp Round 2.6 baseline (seed-paired drop of only 2.3 pp). On the eat-suppression measure the agent is *more* defensive near the predator than the baseline (eating at 56 % of safe-baseline rate vs the baseline's 77 %), confirming the agent treats the predator as a reliably-dangerous-when-present cue under randomisation. The predator-side M2 is still climbing across windows 6–10 toward its plateau, so the gap may shrink further by the end of training, but the current trajectory suggests the agent will land closer to the +30 pp H₀ band than the +20 pp H₁ band.

The §0 "important honest caveat" — the agent's visual sensor channel-codes class regardless of olfactory match — is the load-bearing interpretive frame for this mid-training result. The mid-training reading is consistent with the H₀ prediction in §2.2 (35 % designer prior): the visual one-hot is doing the discrimination work, and behavioural randomisation is not changing what the agent uses as the class cue.

### 11.1 Per-cell preview verdict — Cell C-dist seed 46 at iter 36,144 / 10 M

**Plain English:** at the mid-training waypoint the agent at seed 46 has trained a class-conditional defence very similar in *shape* to the Round 2.6 baseline (predator in-cover rate well above rabbit; eat-suppression near predator below safe-baseline; rabbit-tag fan-out symmetric) but at a *lower* absolute defence level on the in-cover measure. The class gap on in-cover rate is +35.1 pp — almost the +37 pp seed-locked baseline. If this mid-training plateau persists, the experiment's pre-registered verdict at closing will be **H₀(class-pinned-avoidance) confirmed** — the predator's chase behaviour is NOT the cue the agent uses for class discrimination under matched smells, and the most plausible remaining channel is the visual one-hot (the §0 caveat).

**Formal preview predicate: H₀(class-pinned-avoidance) provisionally band-matched.** Δ_M2_class = +35.1 pp (≥ +30 pp); M2_neutral = 0.343 (< 0.60). Both H₀ predicates clear. Per-tag rabbit fan-out is within band on both M2 (2.0 pp ≤ 5 pp) and M5 (0.080 ≤ 0.10) — cleaner than the R2.6 reference. The H₀ verdict at mid-training is *consistent with* but not *committed to* — the closing analysis on the 10 M-episode checkpoint will lock it.

**Elevation:** none claimed. The §5.1 verdict thresholds are written for the closing-analysis checkpoint; a mid-training preview does not satisfy the elevation rule.

### 11.2 What the closing analysis will check that this preview cannot

The mid-training preview cannot decide three questions:

1. **Does the rabbit-side defence catch up?** The R2.6 baseline rabbit-side M2 sits at 0.515; R3 mid-training is 0.343 (−0.17). If the rabbit-side defence rises further while the predator-side defence stays put, Δ_M2 could shrink into the borderline (+20, +30) band, triggering the seed-47 escalation rule. If both sides rise proportionally, the H₀ verdict will hold. Window 9 → 10 rabbit-side dropped 0.013 (within noise) — no escalation trajectory visible yet, but 8 more M episodes of training is substantial.
2. **Does M5_predator stay below 0.80?** Currently 0.560 — comfortably below the H₁-secondary 1.05 ceiling — but at R2.6 the training-time M5 stayed in the 0.93 range while the eval-time M5 dropped to 0.77 only at the converged checkpoint. R3's mid-training M5_predator at 0.560 is much LOWER than R2.6's mid-training would have been (R2.6's M5 was 0.93 in last-10 %), so the M5 path looks like a different shape entirely. The closing analysis will read the M5 plateau, not its mid-trajectory.
3. **Does Δ_M2_class drift down into the borderline band?** Currently +35.1 pp; the H₀ floor is +30 pp; the borderline ceiling is +30 pp. A 5 pp drop over 8 M more episodes is plausible if the rabbit-side defence trains further. The closing analysis at 10 M will lock the answer.

### 11.3 §2.2 designer's priors versus mid-training observed

| §2.2 prior | Stated probability | Mid-training observation | Holding? |
|---|---:|---|---|
| Partial generalisation (H₁ confirmed via M2_neutral rising into 0.65–0.70) | 50 % | M2_neutral = 0.343 — rabbit defence rising but FAR below the 0.65 threshold | Not yet — but mid-training is too early to refute. |
| No generalisation (visual one-hot wins, H₀ confirmed) | 35 % | Δ_M2_class = +35.1 pp; both H₀ predicates band-matched | Currently in band. |
| Over-generalisation (M5_neutral drops below 1.0) | 10 % | M5_neutral = 1.124 — modestly above 1.0 but well below R2.6's 1.20 | Latent (could activate if M5_neutral trends below 1.05). |
| Predator becomes degenerate / training instability | 5 % | Survival 371.7 / 380.5, no policy collapse, M2 stable in windows 8–10 | Refuted. |

The H₀ prior (35 %) is the current band-match; the H₁-partial prior (50 %) cannot be confirmed but cannot yet be refuted either (M2_neutral could rise; the mid-training trajectory is at window 10's value 0.408 vs window 8 at 0.420 — flat-ish, but 8 M more episodes of training is substantial in absolute terms even if the slope is shallow). The closing-analysis verdict will pick between these two priors.

### 11.4 What to do next — closing-analysis routing

**Do not act on the mid-training H₀ preview as a final answer.** The pre-registered §5.1 verdict thresholds are written for the closing checkpoint; this preview is logged so the user has an early read on training health and can plan downstream experiments, NOT to declare the verdict.

The closing-analysis decision tree (to be applied when the run completes):
- **If closing Δ_M2_class lands ≥ +30 pp with M2_neutral < 0.60** (current trajectory's projection): H₀ confirms; single seed is sufficient per §3.4 ("If H₀ confirms, one seed is sufficient given the seed-stability of the baseline"). Route to the next experiment in the line — the visual-channel-blind ablation (Round 4) — as the natural follow-up to settle the §0 visual-cue caveat.
- **If closing Δ_M2_class lands < +20 pp with M2_neutral ≥ 0.65 OR M5_neutral < 1.05**: H₁ confirms; launch seed 47 as the §3.4 seed-lock follow-up before claiming the result.
- **If closing Δ_M2_class lands in [+20, +30] pp**: borderline; escalate to seed 47 per §5.1.
- **If closing Δ_M2_class ≤ 0 OR M2_predator < 0.50**: anomalous; route to senior-developer / experiment-analyzer discussion before any verdict.

The closing analysis should rerun the entire §9–§11 pipeline against the post-training checkpoint and append a new dated section (e.g. § 12 "Closing analysis — Cell C-dist seed 46, 10 M") rather than overwriting these mid-training §§9–11.

### 11.5 Metrics requested

**One known gap, unchanged from §7.** The `Episode/sampled_*_<tag>` keys remain unwired in `train.py` (verified by `discover` listing 65 metrics, none matching `sampled_*` / `detect_*` / `max_stamina` / `hunt_thresh` / `lose_interest` / `recovery`). The mid-training preview uses the §9.5 behavioural-trace proxy (per-episode predator threat-fraction spread) as the sampling-firing diagnostic, and it cleanly distinguishes R3 (std 0.219) from the static-predator R2.6 reference (std 0.148). The diagnostic is sufficient for the preview verdict.

**Recommendation to the user:** the proxy diagnostic is sufficient. Wiring `Episode/sampled_*_<tag>` into `train.py` would let the closing analysis show the actual distributional draws (window-by-window) and confirm uniformity across the per-episode samples; this is a small `feature-workflow` ticket (a few lines in train.py's episode-done branch) and remains worth doing for the closing analysis, but is not blocking.

### 11.6 Related issues

- **No bugs surfaced** in the mid-training preview. Training is healthy, eval-rollout ran cleanly (265 s for 200 episodes), motif clustering converged with silhouette 0.180 (typical for this protocol).
- **No `feature-workflow` plan** is needed for the preview verdict; the `Episode/sampled_*_<tag>` wiring remains a soft request for the closing analysis but does not block.
- **No `bug-fix-workflow` plan** is needed.
- The training process (PID 655905 on n113) is **explicitly NOT to be interrupted** per the user's brief. The preview was produced offline against a saved checkpoint and does not touch the running training.

### 11.7 Open question for the operator before the closing analysis

The single most important open question from this mid-training preview: **does the rabbit-side defence on M2 catch up to its R2.6 plateau (rising from 0.34 toward 0.45–0.51) over the next 8 M episodes, or does it flatten near its current 0.34?** If it rises proportionally to the predator-side, Δ_M2_class shrinks toward the borderline band and the §5.1 escalation rule fires. If it flattens while the predator-side keeps rising, Δ_M2_class will WIDEN at the closing checkpoint, putting H₀ deeper into its band. The window-10 rabbit-side dip to 0.408 (from window 9's 0.421) is a minor wobble within noise, not yet a regression signal; the operator may want to monitor windows 11, 12, 13 of `Episode/BushDiveRate_rabbit` as the run continues.

A second secondary question worth flagging: M5_predator at 0.560 is **dramatically more suppressed than R2.6's 0.769**. If this holds at the closing checkpoint, it is itself an informative finding — the agent under behavioural randomisation reduces eating-near-predator MORE than the baseline does, suggesting the predator's variable behaviour pushes the agent's defensive strategy toward eat-suppression (which works regardless of whether the predator is currently hunting) rather than purely toward bush-diving (which is wasted on a non-hunting predator). This could be a Round-4 successor question.

---

## 12. Mid-training preview manifest

Files produced by this preview (all gitignored except the design doc itself):

- `results/eval/models/1810043/` — eval-rollout output (200 episodes, deterministic policy, 265.2 s wall-clock at git `e3e6789`).
  - `metadata.json` — config snapshot.
  - `episodes/*.npz` — 200 per-episode step arrays.
  - `windows/threat_onsets.parquet` — threat-onset window index.
  - `online_replay.json` — M1 / M2 / M5 online-replay sanity (M1 reads 0 due to a known online_replay quirk on the predator side; cross-tab in `tmp/20260529_r3_seed46_aggregates.json` is the authoritative number).
  - `motifs/feature_vectors.parquet` + `cluster_assignments.parquet` + `cluster_centroids.npy` + `silhouette.json` + `motif_distribution.json` + `exemplars.json` — M7 k-means k = 6 seed = 42.
- `tmp/20260529_0212_r3_seed46_eval_analysis.py` — cross-tab analysis script (mirror of `tmp/20260521_r26_c_seed44_eval_analysis.py`).
- `tmp/20260529_r3_seed46_aggregates.json` — per-class and per-tag M1 / M2 / M5 numerator/denominator/ratio.
- `tmp/20260529_r3_seed46_motifs.csv` — per-cluster centroid feature means + class fractions.
- `tmp/20260529_r3_seed46_motif_by_class.csv` — class × cluster cross-tab (row-normalised).
- `tmp/20260529_r3_seed46_motif_by_tag.csv` — tag × cluster cross-tab (row-normalised).
- `tmp/20260529_0212_r3_online_replay.json` — snapshot of online_replay for traceability.
- `tmp/20260529_0212_r3_aggregates.json` — duplicate of the seed-46 aggregates JSON with the date-time prefix for the working-file convention.
