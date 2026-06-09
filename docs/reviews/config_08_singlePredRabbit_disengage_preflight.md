# Config Audit — 08-singlePredRabbit_disengage (pre-flight)

**Scope:** pre-flight before ~2-day 10M-episode training run on n107:GPU0
**Files audited:** `configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml`
**Lineage:** derived from `06-matchedAggression_chasingRabbit_eval.yaml`
**Design doc:** `docs/experiments/active/hypervigilance/single_pred_rabbit_disengage.md`
**Audited by:** env-config-auditor
**Date:** 2026-06-09

---

## Summary

This config is the "cleanest predator-vs-rabbit contrast" cell in the hypervigilance line: one predator and one rabbit that are byte-identical on every observable and chase-driving parameter, differing only in whether contact hurts (predator: yes, rabbit: no). The config was live-loaded via `load_env_params` without error. The observation breakdown and noise modality set are fully consistent. All five mandatory chase fields are present and correctly parsed for both hunt animals. The `disengage_on_contact` feature (shipped today) parsed `True` for both animals as intended. No latent-bug recurrences are triggered. One design-intent note is raised regarding the effective lethality of the damage range, which is slightly higher than the config comment suggests but is coherent with the stated design goal and does not block launch.

**Verdict: safe to launch.**

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| nit | `configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml` line 9 (comment) | Comment says "~1-in-6 contacts is instantly lethal"; actual P(single hit >= 100) = 17.4% (1-in-5.75). The design intent (occasionally lethal) is correct; only the comment is slightly optimistic. | Update comment from "~1-in-6" to "~1-in-6 (17%)" or leave as acceptable approximation. |
| info | `environment.random_start_pos: true` | Known latent behaviour (documented in ENVIRONMENT_SUMMARY FAQ): agent may spawn on top of a predator or rabbit cell at episode start; contact effects fire on step 0. This is expected for a training config (promotes exploration) and is not a bug here. | No action required for training. Flag if this config is ever re-used as a deterministic evaluation baseline. |
| info | Expected early-training episode length | With E[damage per contact] = 62.5 on a 0-100 injury scale and attack_delay=3, an untrained agent accumulates fatal injury in roughly 1-2 contacts without recovery. Simulation shows: 17% die on first contact, 52% die on second contact, 99%+ dead by third contact (assuming no recovery between contacts). With active recovery (recovery_base_rate=0.1, rest action enabled) a cautious trained agent can survive much longer. The high lethality creates strong pressure to learn avoidance; this matches the stated design goal. | No action required. |

---

## Checklist

### 1. Observation / Noise Modality Consistency

Live `get_observation_breakdown` result (7 active sensors, 27 total dims):

| Sensor | Dims | Noise index | Noise mode |
|--------|------|-------------|------------|
| Satiation | 1 | 2 | state_dependent, sigma=0.1 |
| Interoceptive Nociception | 1 | 3 | state_dependent, sigma=0.1 |
| Extero Nociception | 1 | 4 | state_dependent, sigma=0.1 |
| Olfaction | 5 | 5 | state_dependent, sigma=0.2 |
| Collision | 5 | 6 | constant, sigma=0.01 |
| Proprioception | 6 | 7 | constant, sigma=0.05 |
| Visual | 8 | 8 | state_dependent, sigma=0.2 |

`perceptual_noise.enabled: false` — `apply_perceptual_noise` is a static pass-through at JIT trace time; the modality-map lookup loop is inside the `if params.perceptual_noise_enabled:` guard and never executes. Even if noise were enabled, all 7 active sensors have entries in `noise_modality_order` and the lookup would succeed.

Disabled sensors (Injury, Nutrition, Location) have noise entries in the YAML but are absent from the breakdown — Desync 2 (safe direction, no crash). Padding: 10 configured modalities + 3 zero-pad slots = 13 total, within the static budget.

**Result: PASS**

### 1.5. Behavior-measures Bush Presence

`behavior_measures.enabled: true`. M2 (bush_dive_rate) requires `hides_agent: true` obstacles.

Live check: `obs_hides_agent.sum() = 12` (four quadrant groups of 3 bushes each). Bush entries confirmed at lines 184-219 of the config, all with `hides_agent: true`.

**Result: PASS**

### 2. Mandatory-Key Discipline

**Hunt-specific mandatory chase fields — both entities (predator and rabbit both have `behaviour: hunt`):**

| Field | Predator (ent 0) | Rabbit (ent 1) |
|-------|-----------------|----------------|
| `detection_range` | 10 (degenerate [10,10]) | 10 (degenerate [10,10]) |
| `max_stamina` | 60 (degenerate [60,60]) | 60 (degenerate [60,60]) |
| `stamina_recovery_rate` | 1.0 | 1.0 |
| `hunt_stamina_threshold` | 0.3 | 0.3 |
| `lose_interest_multiplier` | 3.0 | 3.0 |

All five mandatory hunt fields present on both entities. Loader confirmed no `ValueError`.

**`disengage_on_contact`:** parsed as `[True, True]` for both animals — confirmed via `params.animal_disengage_on_contact`. The feature is optional by design (defaults `False` for backward compat) and is correctly set `true` on both entities in this config.

**Schema-required but runtime-ignored body fields** (`body.start_satiation`, `body.random_start_satiation`): present in config (lines 241-242), not being relied upon by any new caller. No issue.

**`properties` key discipline:** Only the plural `properties:` key appears throughout the config (resources, entities, obstacles). No legacy `property:` (singular) key. No `DeprecationWarning` risk.

**`behavior_measures` block (14 mandatory keys):** All 14 keys present. Seed list: 200 seeds, all unique, all non-negative, all < 2^31, matches `eval_n_episodes: 200`. `eval_max_steps: 500` matches `environment.max_steps: 500`. `eval_policy_mode: deterministic`, `eval_obs_noise: training`, `motif_standardise: zscore_pooled` — all valid enum values. `motif_features` contains 10 features, all in the v1 allowed set. `cue_radius: 3.0 > 0`, `obs_window: 5 >= 1`, `motif_kmeans_k: 6 >= 2`.

**Result: PASS**

### 3. Static-Field / JIT Recompile Risk

Single config, single run — no sweep. Static fields that differ from defaults but are stable within this run:

| Static field | Value | Notes |
|---|---|---|
| `visual_sensor_enabled` | `true` | N/A for single run |
| `visual_sensor_range` | `0` | 1 cell (agent's own cell only) |
| `interoceptive_kernel_length` | `12` | N/A |
| `placement_mode` | `per_entity` | N/A |
| `injury_observable` | `false` | N/A |
| `nutrition_observable` | `false` | N/A |
| `use_homeostatic_reward` | `true` | N/A |
| `hunt_idx` | `(0, 1)` | Both animals hunt |
| `predator_indices` | `(0,)` | 1 predator |
| `neutral_indices` | `(1,)` | 1 neutral (rabbit) |
| `noise_modality_order` | 10-tuple | Stable (noise disabled) |

No sweep across static fields. No recompile concern.

`predator_indices=(0,)` and `neutral_indices=(1,)` map correctly: `dist_per_predator` tracks the one predator (index 0); `dist_per_neutral` tracks the rabbit (index 1). `animal_visual_channel=[5, 7]` is correct (predator class=5, neutral class=7). `animal_is_damaging=[True, False]` is correct.

**Result: PASS**

### 4. Known Latent-Bug Recurrences

- **`overeating_death`:** `false` — the bug (flag sets termination_reason=3 but not done=True) is not triggered.
- **`random_start_pos: true`:** active — agent may spawn on a predator or rabbit cell at episode start, contact effects fire on step 0. This is expected behaviour for training and is noted as INFO above. Recorded here as a known latent behaviour, not a blocker.
- **`property` vs `properties`:** only `properties:` (plural) used. No DeprecationWarning.
- **Per-entity `properties` key missing:** all entities (resources, obstacles, entity definitions) carry `properties:` explicitly. No hard `ValueError` risk.
- **Resource respawn at `core.py:300-305` (occupancy not checked):** 8 food sources with `regeneration_delay: 0` (instant respawn). This could cause multiple resources to stack at the same cell if consumption and respawn race. With `regeneration_delay: 0` this is an elevated risk, but the effect (locally concentrated food) is unlikely to degenerate training in a 10x10 grid with 8 food sources spread across 4 quadrants. Not a blocker.
- **`terminated` vs `done`:** no config assumption about divergence.

**Result: PASS (with one INFO note on `random_start_pos`)**

### 5. Schema Padding and Modality-Count

10 configured modalities, padded to 13. `noise_modes.shape = (13,)` confirmed by live load. Padding slots (indices 10-12) are zero. No 14th sensor. Budget intact.

**Result: PASS**

### 6. Cross-Config Coherence

N/A — single config, not a sweep.

---

## Design-Intent Verification (per brief)

**Check 4 — Wide damage range `[5, 120]`:**
- Legal: `jax.random.uniform(key, (), minval=5.0, maxval=120.0)` — valid per-event sample.
- Not a crash: the damage value flows into the injury ring buffer and is spread over `injury_smoothing_duration=3` steps, then accumulates toward `max_injury=100`. `damage=120` delivers 120 total injury over 3 steps regardless of the per-step smoothing — the agent dies within those 3 steps if starting from zero injury.
- P(single contact is one-hit lethal given zero prior injury) = (120-100)/(120-5) = 17.4% (~1-in-5.8). Config comment says "~1-in-6" — close enough to be accurate in spirit.
- Early-training death rate: a fully untrained agent (random actions) will die in 1-2 contacts ~70% of the time. This is high pressure but not degenerate — the agent receives meaningful gradient signal from episodes that survive past the first contact, and the `disengage_on_contact` mechanic ensures the predator retreats after each hit, giving the agent a recovery window between attacks.

**Check 4b — Training viability:**
- 8 food sources, `regeneration_delay: 0` (instant regen), `food_nutrition_gain: 6`, `metabolic_cost: 1.0`. Starting nutrition 100; agent has 100 steps before starvation without eating. Food is abundant and the foraging task is real but not a starvation trap.
- 12 bushes (3 per quadrant) present for cover. Predator detection is full-grid (`detection_range: 10`, 10x10 grid), but `hides_agent: true` bushes break detection — the agent has a meaningful hide option.
- Environment is not degenerate: the agent has both a foraging incentive and an avoidance incentive, and the tools (rest, eat, move, hide in bushes) to act on both.

---

## Conclusion

Safe to launch. No blockers. One nit (comment accuracy on lethality rate), two info notes (expected `random_start_pos` spawn overlap, expected high early-training mortality that is by design). All mandatory fields present, all sensors covered in noise block, `disengage_on_contact` parsed correctly for both animals, behavior-measures block fully valid.

**Audited by: env-config-auditor**
