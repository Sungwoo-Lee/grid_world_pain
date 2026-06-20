# Config Audit — Hunger-Gated Step-1 Linear-Olfactory-Decay Sweep (Pre-Flight)

**Scope:** Multi-config sweep (10 v3.0 nested-extends sparse configs)
**Files audited:**
- `configs/environment/experiment/hunger_gated_lindecay/01-s0_sig0.yaml` through `10-s0.5_sig0.4.yaml` (10 new files)
- `configs/environment/experiment/hunger_gated/01-s0_sig0.yaml` through `10-s0.5_sig0.4.yaml` (10 Step-1 twins, read-only reference)
**Audited by:** env-config-auditor
**Date:** 2026-06-20

---

## Purpose

This audit guards the pre-flight of a 10-run "linear olfactory decay" sweep. The experiment asks
whether giving an agent a **longer-range smell cue** — letting predator scent carry three times
further (linear fade instead of inverse-square fade) — shifts its behaviour from reacting after
contact with the predator to avoiding it before contact. Each of the 10 new configs is a thin
override that inherits its matched Step-1 "steep smell" twin in full, changing only the single
olfactory-decay exponent key (`sensory.decay_power`) from 2.0 (the default) to 1.0 (linear).

The Step-1 sweep already passed its own pre-flight audit (at
`docs/reviews/config_hunger_gated_step1_preflight.md`); this audit's job is to confirm that the
nested-extends mechanism correctly threads the single-key change through, that the inherited
scene is bit-for-bit identical to the Step-1 twin in every other dimension, and that no
correctness-level issue has been introduced.

---

## Summary

**PASS.** All 10 lindecay configs load cleanly through the full `load_env_config` +
`load_env_params` pipeline. The nested `extends:` chain (lindecay config → Step-1 twin →
`environment/default`) resolves correctly on all 10 pairs. A live per-field diff of the merged
`EnvParams` against each Step-1 twin confirms that `sensor_decay` (loaded from
`sensory.decay_power` via `get_mandatory`) is the single and only parameter that differs, taking
the value 1.0 in every lindecay config versus 2.0 in every Step-1 twin. Every other field
(smell vectors, lethal predator `damage=[5,120]`, 1-vs-1 chase, randomised starts, food patches,
obstacles, body, behavior_measures, noise modality order) is byte-identical to the twin across
all 10 pairs. The 27-dimensional observation vector and 10-modality noise order (zero-padded to
13 slots) are unchanged from the Step-1 sweep. Two informational notes from the Step-1 audit
carry forward unchanged and remain non-blocking.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| NOTE (carry-forward) | Configs 08 and 10 — `entities[0].properties_std` and `entities[1].properties_std` | At separation `s=0.5`, per-episode smell draws with standard deviation 0.2 or 0.4 can produce negative channel values (e.g. predator channel 3 = 0.0 minus 0.4). This is identical to the Step-1 condition. Whether the olfaction sensor clips negative draws is a sensor-code question, not a config defect. | Verify `apply_olfaction` clips sampled properties at 0.0 before pooling. Documented in Step-1 audit. Non-blocking; identical exposure in both arms of the paired comparison. |
| NOTE (carry-forward) | All 10 — `environment.random_start_pos: true` | Agent may spawn on a predator or resource at reset (documented latent bug; placement does not participate in the occupancy mask). Identical to Step-1; expected design. | No fix required. Consistent with the Step-1 baseline and cell-08 origin. |

No new findings beyond the two carry-forward notes. No blockers.

---

## Checklist

### 1. Observation / Noise Modality Consistency

All 10 lindecay configs resolve to the identical 27-dimension observation breakdown:

`{Satiation: 1, Interoceptive Nociception: 1, Extero Nociception: 1, Olfaction: 5, Collision: 5, Proprioception: 6, Visual: 8}`

`decay_power` governs **how much** olfactory signal reaches the sensor (the weighted sum of
property vectors), but does not change **how many** channels are emitted. The observation width
is therefore unchanged; no modality is added, removed, or renamed.

The `noise_modality_order` tuple is identical across all 10 (confirmed by live static-field
comparison against config 01):

`('Injury', 'Nutrition', 'Satiation', 'Interoceptive Nociception', 'Extero Nociception', 'Olfaction', 'Collision', 'Proprioception', 'Visual', 'Location')`

10 configured modalities, zero-padded to 13-slot arrays (`noise_modes`, `noise_sigmas`,
`noise_injury_scales`, `noise_clip_min`, `noise_clip_max` — all shape `(13,)`).

Every observation modality has a corresponding noise entry. The three noise-side modalities that
have no observation counterpart (`Injury`, `Nutrition`, `Location`) are expected and carry
`mode=none` — identical to Step-1.

Status: **PASS**

### 1.5 Behavior-Measures Bush Presence

`behavior_measures.enabled: true` (inherited from the Step-1 twin, which inherits it from
`default.yaml`). The merged `EnvParams` carries `obs_hides_agent` with 12 `True` entries,
confirming 12 bush obstacle slots with `hides_agent=True` (4 quadrant groups × 3 each).
The M2 bush-dive-rate measure is operable.

Status: **PASS**

### 2. Mandatory-Key Discipline

`sensory.decay_power` is loaded exclusively via `config.get_mandatory('sensory.decay_power')`
at `src/environment/config_loader.py:1210`. Confirmed by grep and by the fact that omitting the
key in a lindecay config would raise `ValueError` at load time (all 10 load cleanly, so the key
is present in the merged config from the Step-1 twin in every case, with the lindecay override
writing 1.0 on top).

The `sensory` block in each lindecay config is a dict overlay. The v3.0 deep-merge rule merges
dicts key-by-key, so `decay_power: 1.0` is written into the inherited `sensory` block without
disturbing `olfactory_enabled`, `sensor_radius`, `vector_size`, or any visual-sensor key. All
other critical keys (`environment.height`, `body.random_start_nutrition`, etc.) are mandatory in
the Step-1 twin, inherited without change.

`body.start_satiation` / `body.random_start_satiation: false` — no-op keys present for schema
compliance; not relied upon by any caller. Confirmed identical to Step-1.

Legacy `property` (singular) key: absent from all 10 lindecay configs. All entities in the
inherited Step-1 layer use `properties` (plural). No `DeprecationWarning` will fire.

Status: **PASS**

### 3. Static-Field and JIT Recompile Risk

All 10 lindecay configs share identical static fields (verified by per-field equality against
config 01):

| Static field | Value |
|---|---|
| `height` | 10 |
| `width` | 10 |
| `placement_mode` | `per_entity` |
| `use_homeostatic_reward` | `True` |
| `visual_sensor_enabled` | `True` |
| `interoceptive_kernel_length` | 12 |
| `noise_modality_order` | 10-tuple, identical across all 10 |

`sensor_decay` (the one changed field) has **empty** `struct` metadata — confirmed by inspection
of `EnvParams` field metadata. It is a traced JAX leaf, not a static field. Changing its value
from 2.0 to 1.0 does NOT force XLA recompilation. The lindecay configs can share a single
compiled XLA graph (they share all static fields with each other and with the Step-1 twins that
use those same static fields). No per-config recompile beyond the normal first-run trace.

Status: **PASS**

### 4. Known Latent-Bug Recurrences

- `overeating_death: false` — confirmed on all 10 (inherited from Step-1). The latent bug
  (termination_reason=3 without triggering done=True) is not triggered.
- `random_start_pos: true` — present on all 10. The spawn-on-predator latent behaviour is
  active and is the intended design (see NOTE in Findings). Identical to Step-1 and cell-08.
- `body.start_satiation` / `body.random_start_satiation: false` — no-op; no caller relies on it.
- Legacy `property` (singular): absent from all 10 configs.
- Per-entity missing `properties`: all entities carry `properties` and `properties_std`.
- Resource respawn stacking: not a concern at this density (8 food sources,
  `regeneration_delay=0`, no high regen rate).
- `terminated` vs `done`: no config-level assumption about divergence between the two.
- Predator count and lethality: `animal_damage=[[5.0, 120.0], [0.0, 0.0]]`,
  `predator_indices=(0,)` — 1 lethal predator, 1 harmless rabbit. Confirmed on merged params.

Status: **PASS**

### 5. Schema Padding and Modality-Count

10 active modalities; noise arrays padded to 13 slots for JIT stability. No new sensor
introduced. 13-slot static padding is unchanged from the Step-1 sweep and from `default.yaml`.
No 14th sensor is introduced. `decay_power` affects the *values* inside the 5-dimensional
olfaction slice, not its width.

Status: **PASS**

### 6. Cross-Config Coherence (Sweep Audit)

Live `EnvParams` diff of all 10 lindecay configs against their respective Step-1 twins:

| Pair | Diff result |
|------|-------------|
| 01 (s=0, σ=0) | PASS — only `sensor_decay` differs (2.0 → 1.0) |
| 02 (s=0.05, σ=0) | PASS — only `sensor_decay` differs |
| 03 (s=0.1, σ=0) | PASS — only `sensor_decay` differs |
| 04 (s=0.25, σ=0) | PASS — only `sensor_decay` differs |
| 05 (s=0.5, σ=0) | PASS — only `sensor_decay` differs |
| 06 (s=0.1, σ=0.2) | PASS — only `sensor_decay` differs |
| 07 (s=0.25, σ=0.2) | PASS — only `sensor_decay` differs |
| 08 (s=0.5, σ=0.2) | PASS — only `sensor_decay` differs |
| 09 (s=0.1, σ=0.4) | PASS — only `sensor_decay` differs |
| 10 (s=0.5, σ=0.4) | PASS — only `sensor_decay` differs |

Within the lindecay sweep, all 10 configs share identical static fields (see §3). The only
inter-config differences are the inherited `properties` / `properties_std` vectors on the two
animal entities (the smell-separation and noise design from the Step-1 sweep), which is the
correct and intentional design.

WandB tags encode the swept variables cleanly:
`rppo_hg<NN>_s<sep>_sig<sigma>_dp1_s<seed>`. The `dp1` marker distinguishes this wave from the
Step-1 wave and is strippable for paired-comparison queries. All 10 share group
`hunger_gated_lindecay`. Training seed is 42 on all 10, matching the Step-1 twins for a true
matched-pair comparison.

Status: **PASS**

---

## Live-Load Verification

All 10 lindecay configs loaded successfully via:

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
from src.environment.config_loader import load_env_config, load_env_params
from src.environment.sensor import get_observation_breakdown
p = load_env_params(load_env_config('configs/environment/experiment/hunger_gated_lindecay/<name>.yaml'))
print(get_observation_breakdown(p))
"
```

No error raised on any of the 10. `get_observation_breakdown` returns the expected 27-dim
breakdown on all 10. `sensor_decay = 1.0` confirmed on all 10 merged `EnvParams` objects.

---

## Conclusion

Safe to launch. No blockers. Two informational notes carry forward from the Step-1 audit (potential negative Gaussian smell draws at extreme s=0.5, σ>0 grid points; random-start-pos spawn-on-entity latency) — both are identical to the Step-1 arm and are expected in the experimental design.

Audited by: env-config-auditor
