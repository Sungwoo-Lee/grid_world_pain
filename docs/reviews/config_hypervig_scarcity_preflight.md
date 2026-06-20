# Config Audit — Hypervigilance Scarcity Pre-flight

**Scope:** Two-config sweep pre-flight (scarce vs abundant arms of the hypervigilance-scarcity experiment)
**Files audited:**
- `configs/environment/experiment/hypervig_scarcity/01-scarce.yaml`
- `configs/environment/experiment/hypervig_scarcity/02-abundant.yaml`
**Design doc:** `docs/experiments/active/hypervigilance/20260620_hypervig_scarcity_olfactory_ambiguity.md`
**Audited by:** env-config-auditor
**Date:** 2026-06-20

---

## Summary

This is a two-arm experiment where one agent trains in a food-scarce world and another in a food-abundant world, with everything else — the dangerous predator, the harmless rabbit that smells almost identical to the predator, the obstacle layout, the body parameters, and the noise settings — held identical. The goal is to test whether scarcity forces the agent to make a hunger-sensitive stay-vs-flee decision near an ambiguous animal.

Both configs load cleanly through the v3.0 config system, produce a valid 27-dimensional observation vector, survive a reset-plus-step runtime check without error, and correctly implement the food-only contrast. All eight audited properties pass. There are two notes that do not block launch: a starvation-floor concern for agents that start with very low nutrition in the scarce arm (by design, flagged for awareness), and a routine XLA recompilation that will occur if both configs are loaded into the same Python process during analysis (also expected and benign for separate training runs).

**Verdict: PASS-WITH-NOTES. Safe to launch.**

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| nit | `01-scarce.yaml` (runtime consequence) | At minimum starting nutrition (10), the scarce arm's maximum-foraging budget (480 net nutrition) minus metabolic drain (500) = -10: a perfect forager starting at nutrition 10 would just starve. This is by construction (the design doc explicitly targets "survivable only with sensible foraging") but the narrow margin means hungry-start episodes will have elevated starvation deaths independent of animal contact. If the scarce-world hungry-start death rate hits the pre-registered 60% threshold, evaluate whether starvation rather than predator contact is the primary cause before concluding "too punishing." | Monitor: in the post-run analysis, split hungry-start deaths by `termination_reason` (code 2 = starvation vs code 1 = injury/predator). No config change needed. |
| nit | Both configs (load-time, separate training processes) | The two configs differ in two static `EnvParams` fields: `max_per_type` (7 in scarce, 8 in abundant) and `num_entities` (28 vs 34). These drive JAX/XLA array shapes; loading both into the same Python interpreter session will force an XLA recompile when switching between them. | This is expected behaviour for configs with different entity counts. Each arm is a separate training run in a separate process, so no action needed. Only relevant if a post-analysis script loads both configs in sequence without clearing the JIT cache. |

---

## Checklist

### 1. Observation / Noise Modality Consistency

Both configs inherit `perceptual_noise` and all sensor flags from `default.yaml` unchanged. The live loader confirms identical 27-dimensional observation vectors:

| Sensor | Dim |
|---|---|
| Satiation | 1 |
| Interoceptive Nociception | 1 |
| Extero Nociception | 1 |
| Olfaction | 5 |
| Collision | 5 |
| Proprioception | 6 |
| Visual | 8 |
| **Total** | **27** |

`Injury` and `Nutrition` are observable-gated off (inherited: `injury_observable: false`, `nutrition_observable: false`), so they do not appear in the obs vector — consistent with the noise config that carries them at `sigma: 0.0`. `noise_modality_order` is identical between both arms: `('Injury', 'Nutrition', 'Satiation', 'Interoceptive Nociception', 'Extero Nociception', 'Olfaction', 'Collision', 'Proprioception', 'Visual', 'Location')`.

`perceptual_noise_enabled: false` in both arms (inherited from default). The noise modality list is present and consistent; no `KeyError` risk. The design doc's note "sensory / perceptual_noise: inherited from default.yaml unchanged (perceptual_noise.enabled: false)" is correct.

**Result: PASS**

### 1.5 Behavior-measures Bush Presence

`behavior_measures.enabled: true` (inherited). M2 (bush-dive rate) is in scope. Both configs declare 12 bushes (4 quadrants x 3), all with `hides_agent: true`. Live loader confirms: `obs_hides_agent` array contains 12 `True` entries in both arms, in all four quadrants. `info['agent_in_bush']` will fire correctly; M2 is structurally interpretable.

**Result: PASS**

### 2. Mandatory-Key Discipline

Both configs use `extends: environment/default`, so all mandatory keys flow through the base. The configs themselves introduce no new keys — only override existing YAML values within known sections (`resources:`, `entities:`, `obstacles:`, `body:`). Conditional-mandatory keys for initial-state randomization are all present and correct:

- `random_start_nutrition: true` with `start_nutrition_low: 10`, `start_nutrition_high: 100` (low < high, high = max_nutrition)
- `random_start_injury: true` with `start_injury_low: 0`, `start_injury_high: 80` (low < high, high < 100 death line)
- `random_start_satiation: false` (documented no-op; `start_satiation` and `body.random_start_satiation` are required by schema, present via default, and intentionally not relied upon per FAQ)

The loader confirms: `start_nutrition_low=10.0`, `start_nutrition_high=100.0`, `start_injury_low=0.0`, `start_injury_high=80.0` in both arms.

**Result: PASS**

### 3. Static-Field and JIT Recompile Risk

The two arms share all experiment-critical static fields: `height=10`, `width=10`, `max_steps=500`, `placement_mode='per_entity'`, `visual_sensor_enabled=True`, `visual_sensor_range=0`, `olfactory_enabled=True`, `sensor_range=1`, `interoceptive_kernel_length=12`, `noise_modality_order` (identical tuple), `perceptual_noise_enabled=False`, `random_start_pos=True`.

Two static fields differ: `max_per_type` (7 vs 8) and `num_entities` (28 vs 34). These differ because the abundant arm has 8 food resource instances versus 2 in the scarce arm, changing the internal entity-pool size. Both are `struct.field(pytree_node=False)`. Since the two arms run as separate training processes (each gets its own JIT trace), this causes no problem during training. The recompile note applies only to post-analysis scripts that load both configs in the same Python session — see Findings.

No sweep varies a static field to test a hypothesis — the food-block difference correctly falls in the non-static resource parameter arrays (`res_max_cons`, `res_reg_delay`, `res_spawn_area`).

**Result: PASS (nit noted)**

### 4. Known Latent-Bug Recurrences

- **`overeating_death: false`** (inherited from default): the latent bug (flag sets `termination_reason=3` but does not fire `done=True`) is inert because the flag is off. No issue.
- **`random_start_pos: true`** (both arms): agent may spawn on a predator or rabbit at reset — contact effects fire on step 0. This is the existing and accepted condition for this experiment family (same as the Step-1 lineage, already audited). The designer is aware.
- **`property` vs `properties` legacy key**: all entity definitions in both configs use the canonical `properties` (plural) key. No `DeprecationWarning` risk.
- **Per-entity olfactory key missing**: all entities (predator, rabbit, inert-count-0 predator, all resource and obstacle entries) carry explicit `properties` keys. No hard `ValueError` risk.
- **Resource respawn occupancy check**: `regen_delay=40` in the scarce arm means food respawns are infrequent; stacking risk exists but is low at 2 total resources. Not a concern here.
- **`terminated` vs `done`**: no config assumption on divergence. N/A.
- **`body.start_satiation` / `body.random_start_satiation`**: present via default (mandatory in schema), not relied upon in any caller (documented no-ops). `random_start_satiation: false` is set explicitly in both configs, consistent with the design intent that satiation is derived from nutrition.

**Result: PASS**

### 5. Schema Padding and Modality-Count

`noise_modality_order` has 10 modalities in both arms. The zero-padding to a fixed static shape of 13 is handled by the loader; no new sensor is introduced, so padding length is unchanged. The observation is 27-dimensional (well within the 13-slot noise indexing envelope). No 14th sensor is introduced.

**Result: PASS**

### 6. Cross-Config Coherence (Sweep)

**Food-only diff is pure.** The YAML diff contains only: (a) comment block (arm-description text), (b) the `resources:` list (scarce has 2 entries, abundant has 4; counts, max_consumption, regeneration_delay, and spawn areas differ as designed). Every other YAML block — `entities:`, `obstacles:`, `body:`, `location_areas:`, the `extends:` directive, `environment.height/width/max_steps/placement`, and all inherited blocks — is byte-identical between the two files. Live `EnvParams` comparison confirms the diff is confined to the resource arrays.

**Smell vectors verified (live EnvParams):**
- Predator (index 0): `animal_property=[0, 0.55, 0.45, 0, 0]`, `animal_property_std=[0, 0.4, 0.4, 0, 0]`
- Rabbit (index 1): `animal_property=[0, 0.45, 0.55, 0, 0]`, `animal_property_std=[0, 0.4, 0.4, 0, 0]`
- Identical between scarce and abundant. Symmetric layout confirmed (predator ch2=0.55, ch3=0.45; rabbit mirror).

**Lethality and scene fidelity (live EnvParams):**
- `animal_damage`: predator `[5.0, 120.0]` (lethal), rabbit `[0.0, 0.0]` (harmless) — in both arms.
- `animal_disengage_on_contact`: `[True, True]` for predator and rabbit — in both arms.
- `animal_is_damaging`: `[True, False]` — in both arms.
- `animal_visual_channel`: `[5, 7]` (predator ch5, rabbit ch7) — in both arms.
- Detection range 10, max_stamina 60, stamina_recovery 1.0, hunt_threshold 0.3, lose_interest 3.0, attack_delay 3 — all identical.
- Inert `count=0` entity (hiding-predator slot): filtered out at load time; does not appear in `animal_tags`, `animal_damage`, or any other animal array. Confirmed: `animal_tags=('full', 'rab')` in both arms.
- No hiding-predator resources in either config (`res_type` is all food type 0 in both arms).

**Seed alignment:** both arms use training seed 42, `eval_seeds` generated identically from `{rng: 42, sort: true}` (200 episodes, byte-identical seed list). Episode-index comparability is preserved.

**WandB tags** (`rppo_hvs_scarce_s42` / `rppo_hvs_abundant_s42`) are unique, arm-parseable, and follow the `rppo_hvs_<arm>_s<seed>` convention from the manifest. Group `hypervig_scarcity` is shared.

**Result: PASS**

---

## Item 7 — Scarcity Survivability Sanity (custom check per audit request)

Computed from loaded `EnvParams` values:

- Scarce: 2 food cells, `res_max_cons=[4,4]`, `res_reg_delay=[40,40]`, `food_nutrition_gain=6`, `eating_nutrition_cost=1.0`
- Net nutrition per eat: 6 − 1 = +5
- Regen cycles per 500-step episode: floor(500/40) = 12
- Total eat events (max, both cells): 2 × 4 × 12 = 96
- Total net nutrition from food: 96 × 5 = **+480** (matches design doc)
- Metabolic drain: 500 × 1.0 = **-500**
- Net from food alone: **-20** (always a slight deficit from food)

Survivability by starting nutrition:
- start = 100 (full): 100 + 480 - 500 = **+80** — survives comfortably
- start = 55 (median): 55 + 480 - 500 = **+35** — survives with good foraging
- start = 10 (minimum): 10 + 480 - 500 = **-10** — borderline starvation even with perfect foraging

The design intent is met: scarcity forces active foraging and genuinely pressures hungry-start agents. The minimum-start margin of -10 is narrow and will produce some starvation deaths in hungry episodes without any predator contact. This is pre-registered in the design doc as the intended pressure (§4.1 "survivable ONLY with sensible foraging"). It is not a config error. The pre-registered failure-mode verdict covers this: if the hungry-start death rate reaches 60%, check `termination_reason` (code 2 = starvation vs code 1 = predator injury) before concluding the lethality-scarcity combination is "too punishing."

Abundant arm: 8 food cells, `res_max_cons=12`, `res_reg_delay=0` (instant) — effectively unlimited throughput. Not starvation-pressured by design. Confirmed.

**Result: PASS (margin noted in Findings)**

---

## Conclusion

Both configs are correctly layered (sparse `extends: environment/default`), resolve to a valid complete scene, and are identical in every non-food dimension. The food-only contrast is confirmed pure by both YAML diff and live `EnvParams` comparison. The observation is 27-dimensional with consistent noise coverage. All known latent-bug triggers are absent or inert. The scarce arm's food budget is tight by construction and matches the design doc's +480 net estimate. Runtime checks (reset + step) pass on both arms without error.

**Safe to launch. Fix blockers before launch: none. Two nits (starvation-floor monitoring, routine recompile) noted for awareness.**

Audited by: env-config-auditor
