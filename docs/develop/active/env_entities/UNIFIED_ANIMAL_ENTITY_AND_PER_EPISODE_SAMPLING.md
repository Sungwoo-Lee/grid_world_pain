---
title: "Env Refactor — Unified Animal Entity + Per-Episode Sampling of Behavioural Params"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [unified_animal_entity, env_entities_unified_animal, env_entities_step_1]
---

# Env Refactor — Unified Animal Entity + Per-Episode Sampling of Behavioural Params

> **Status**: PLANNED
> **Opened**: 2026-05-28
> **Branch**: `v2.0` (from `v1.4@72186aa`, latest: `f0fe297`)
> **Related**: [hypervigilance/01-interoNocicept_sameProp.yaml](../../../../configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml) (the parity-reference config), [FRONTMATTER_CONTRACT](../meta/FRONTMATTER_CONTRACT.md), [AGENT_PLAYBOOK](../../../AGENT_PLAYBOOK.md)

---

## Context

Today the gridworld environment models "the things that move and have a smell" as **two separate kinds of code**: a *patrolling predator* with a hunt/patrol/return state-machine, and a *neutral animal* (e.g., rabbit) that wanders randomly. The two share roughly 80% of their state — position, move-timer, olfactory signature, patrol box — but every step of the env, every sensor channel, every analysis script branches on which array (`pred_*` vs `neutral_*`) the entity lives in. That split shows up as duplicated code in `src/environment/core.py` (`update_predators` at line 132 vs `update_neutral_animals` at line 253), duplicated fields in `EnvState` and `EnvParams`, hard-coded visual channels in `src/environment/sensor.py:184–193`, and parallel per-class metric arrays in `src/behavior/accumulators.py`. Adding any new animal kind (e.g., a fleeing prey, a static decoy) means another full vertical of duplicated code.

This refactor collapses both into **one "animal" entity** whose behaviour is selected by a static tag (`behaviour_mode ∈ {wander, hunt, static}`) and whose *class* (`predator` / `neutral` / future kinds) remains as a separate static tag — the class is what tells the visual sensor which channel to light up and what tells the eval-rollout which per-class metric bucket to drop a sample into. **Class and behaviour are now orthogonal**: a "rabbit that hunts" or a "predator that wanders" becomes expressible without code changes.

Alongside the structural change, this refactor adds a third sampling cadence — **per-episode uniform sampling** — so that behavioural parameters like `detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, and `lose_interest_multiplier` can be specified as ranges (e.g., `detection_range: [0, 5]`) and re-sampled at the start of every episode, independently for each animal instance. The existing scalar form (`detection_range: 5`) continues to work — it is treated as the degenerate range `[5, 5]`. The existing per-event cadence (e.g., `damage: [15, 45]` re-sampled on every collision) and per-spawn cadence (e.g., olfactory `properties_std` re-sampled on every respawn) are unchanged.

**Hard constraints (locked by the v2.0 design discussion):**

1. **Class is a static tag** on each entity (`struct.field(pytree_node=False)` on the unified arrays). It drives (a) visual-sensor channel mapping and (b) per-class metric fan-out in eval-rollout — both load-bearing for prior studies and downstream analyses.
2. **Parity gate** — a unified-schema config equivalent to today's two-class config must produce byte-identical per-step state and byte-identical visual-channel observations. This is the load-bearing constraint that lets us land the refactor without re-running every prior study.
3. **Backward-compatible loader** — the existing `predators:` and `neutral_animals:` YAML sections (used by all 74 configs under `configs/experiment/**/*.yaml`) keep working unchanged.
4. **First-implementation scope for distributional sampling** is exactly five fields: `detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`. `move_interval`, `patrol_area`, and `nociception_intensity` stay scalar in this phase (deferred).

This is the first design-level artifact of the `v2.0` branch. Independently, the user has flagged that the visual sensor channel-codes class regardless of olfactory signature — so under matched smells, only the smell channel is matched but the agent can still see "predator vs. rabbit" in the visual one-hot. This refactor preserves that behaviour exactly; addressing visual-channel class-blindness is a separate experimental-design conversation and is **out of scope here**.


## Analysis

### Where the current split lives

| Concern | Predator code path | Neutral code path |
|---|---|---|
| Update step | `core.py:132 update_predators()` — HUNT/PATROL/RETURN state machine + stamina + attack_timer | `core.py:253 update_neutral_animals()` — random jitter |
| State arrays | `state.py:19–24` — `pred_pos`, `pred_state`, `pred_stamina`, `pred_move_timer`, `pred_attack_timer`, `pred_property_sampled` | `state.py:27–29` — `neutral_pos`, `neutral_move_timer`, `neutral_property_sampled` |
| Params arrays | `state.py:74–89` — 12 `pred_*` fields incl. `pred_detect`, `pred_max_stamina`, `pred_recovery`, `pred_hunt_thresh`, `pred_lose_interest_mult`, `predator_tags` | `state.py:104–110` — 7 `neutral_*` fields incl. `neutral_tags` |
| Visual sensor | `sensor.py:191` — `one_hot(full((num_pred,), 5), 8)` → channel 5 | `sensor.py:193` — `one_hot(full((num_neutral,), 7), 8)` → channel 7 |
| Olfactory sensor | `sensor.py:292` — separate `sense_resource` call on `pred_pos` / `pred_property_sampled` | `sensor.py:294` — separate `sense_resource` call on `neutral_pos` / `neutral_property_sampled` |
| Damage logic | `core.py:395–404` — `at_predator = all(new_pred_pos == agent_pos)`; per-event uniform sample from `params.pred_damage[:, 0..1]` | none (neutrals never damage) |
| Per-step distances | `core.py:505–509` — `dist_per_predator = norm(state.pred_pos - new_agent_pos)` | `core.py:500–504` — `dist_per_neutral` |
| Config loader | `config_loader.py:228–284` — reads `environment.predators:` | `config_loader.py:331–366` — reads `environment.neutral_animals:` |
| Metric fan-out | `accumulators.py:38, 54, 111–120, 346, 392, 432, 477` — `num_predator_tags` slice, predator-tagged keys | same file, `num_neutral_tags` slice, neutral-tagged keys |

### Where the unified shape lands

| Concern | After refactor |
|---|---|
| Update step | One `update_animals()` consuming `state.animal_*` arrays; behaviour dispatch inside vmap (see §4 below) |
| State arrays | `animal_pos`, `animal_stamina`, `animal_state`, `animal_move_timer`, `animal_attack_timer`, `animal_property_sampled`, **`animal_detect_sampled`, `animal_max_stamina_sampled`, `animal_recovery_sampled`, `animal_hunt_thresh_sampled`, `animal_lose_interest_sampled`** (the five per-episode-sampled fields) |
| Params arrays | Per-entity `animal_property`, `animal_property_std`, `animal_nociception`, `animal_move_int`, `animal_damage`, `animal_patrol`, `animal_spawn_area`, `animal_attack_delay`; **`animal_detect_low/high`, `animal_max_stamina_low/high`, `animal_recovery_low/high`, `animal_hunt_thresh_low/high`, `animal_lose_interest_low/high`** (the five low/high bound pairs) |
| Static tags | `animal_classes: tuple[str, ...]` (e.g., `('predator', 'predator', ..., 'neutral', 'neutral')`); `animal_behaviours: tuple[str, ...]` (e.g., `('hunt', 'hunt', ..., 'wander', 'wander')`); `animal_tags: tuple[str, ...]` (metric labels). All three are `struct.field(pytree_node=False)` on `EnvParams`. |
| Visual sensor | One per-entity scatter: `class_to_channel = {'predator': 5, 'neutral': 7}`; visual props built from `animal_classes` mapped to channel ints |
| Olfactory sensor | One `sense_resource` call on `animal_pos` / `animal_property_sampled` |
| Damage logic | `at_animal = all(animal_pos == agent_pos)` masked by `is_damaging_class = (animal_classes == 'predator')` (precomputed as a static jnp.array at param-build time) |
| Per-step distances | `info['dist_per_animal']` (shape `[num_animals]`) + the helper `select_by_class(state, class)` returning a boolean mask on the animal axis |
| Config loader | Two YAML paths: (a) new `entities:` section preferred; (b) `predators:` + `neutral_animals:` legacy sections re-projected into the unified schema. Loader produces a single set of `animal_*` arrays |
| Metric fan-out | Per-class slicing done in Python at logger setup using `animal_classes` and `animal_tags`; runtime layout matches today's behaviour |

### Why uniform-only and per-episode sampling

The user locked these in the design discussion. Briefly:

- **Uniform only** — gives the same expressive power as the existing `damage: [min, max]` cadence (per-event), and keeps the YAML shorthand `[low, high]` consistent across all three cadences. Gaussian / log-uniform / mixture variants can be added later as `{type: gaussian, mean: m, std: s}` long-form, but are not needed now.
- **Per-episode for behavioural params** — the five fields under scope drive the predator state machine. The user wants population-level heterogeneity within a single training run (e.g., "the agent meets predators with detection range drawn uniformly from `[0, 5]`"), not within-episode drift. Per-episode is the right cadence: a single episode is the unit of agent–environment coupling, and re-sampling at reset preserves "the predator with detection 4 stays detection 4 for this whole episode."
- **Per-instance independence** — N animals of the same class get N independent draws. The four hiding-predators in `01-interoNocicept_sameProp.yaml` each get their own `detection_range` per episode.

### Risks called out by this analysis

1. **JIT recompile** if `lax.switch` is used naively with Python-side `behaviour_mode` strings. Solution: convert `behaviours` to an int-coded jnp.array (`0=wander, 1=hunt, 2=static`) at param-build time, and use `lax.switch(beh_int, branches)` per-entity inside the vmap — or, equivalently, run all three branches and select via mask. We pick the mask approach (see §4 below) because the three behaviours are cheap and the masked-combine avoids the branch-divergence cost in vmap. Compile-time cost is amortised: shapes are static across all configs with the same entity counts.
2. **Parity tail risks**: (a) PRNG key consumption order — current `jax_step` splits `key, respawn_key, predator_key, neutral_key, damage_key, property_key` (6 keys). After refactor it becomes `key, respawn_key, animal_key, damage_key, property_key` (5 keys). This *will* change the per-step PRNG stream and therefore byte-level parity. To preserve parity, the refactored step must **preserve the 6-key split** (renaming `predator_key` + `neutral_key` to `hunt_animal_key` + `wander_animal_key`, and routing them by behaviour-mode mask inside `update_animals`). This is documented and tested as part of CP1's parity gate.
3. **`damage_key` is reused 3× today** (`core.py:351, 398, 410`) — same key used for resource damage, predator damage, obstacle damage. That is a pre-existing PRNG bug (not technically a bug since `jax.random.uniform` with different shapes yields different streams, but it is suspicious). **Do not fix it in this refactor** — flag it as a separate issue if confirmed. Refactor must preserve this exact behaviour.
4. **`predator_enabled: bool` flag** (`state.py:87`) currently disables the *predator path*. **Fully removed in CP1.** The 86 existing configs that reference this flag are migrated atomically in CP1: 85 of them set `predator_enabled: true` (the default — sed-stripped, no semantic change); 1 of them (`configs/verification/olfaction_parity_neutral.yaml`) sets `predator_enabled: false` and is hand-migrated to an empty `predators: []` list. After CP1 lands, the loader does not read the flag; configs that still carry the key raise `ValueError` (caught by the CP1 migration sweep itself).
5. **`grid_world.py` + `renderer.py` + `renderer_v2.py`** all index `state.pred_pos` / `state.neutral_pos` for rendering. They need to switch to slicing `state.animal_pos` by class (using `select_by_class`). The visual output must remain byte-identical for the parity-reference config.

## Implementation Plan

### Design

#### Unified config schema (the new way)

A new top-level `environment.entities:` list. Each entry has fields: `class` (`predator|neutral|...`), `behaviour` (`hunt|wander|static`), `tag`, `count`, `properties`, `properties_std`, `nociception_intensity`, `move_interval`, `damage`, `spawn_area`, `patrol_area`, and the five distributional behavioural fields. Each of those five accepts either a scalar (degenerate range) or a `[low, high]` list.

**Worked example** — equivalent of the `01-interoNocicept_sameProp.yaml` predator + rabbits, in the new schema (predators get a per-episode `detection_range` draw, rabbits use matched smell `[0.0, 1.0, 0.0, 0.0, 0.0]`):

```yaml
environment:
  entities:
    # ── Predator: full-grid patrol, per-episode-randomised detection range and stamina ──
    - class: "predator"
      behaviour: "hunt"
      tag: "full"
      count: 1
      properties:     [0.0, 1.0, 0.0, 0.0, 0.0]   # same smell as rabbit (matched-smell setup)
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [15.0, 45.0]                          # per-event uniform sample (unchanged)
      spawn_area:  [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      # ── per-episode uniform sampling (NEW) ──
      detection_range:         [0, 5]               # uniform over [0, 5] each episode
      max_stamina:             [20, 40]
      stamina_recovery_rate:   [0.5, 1.5]
      hunt_stamina_threshold:  [0.5, 0.9]
      lose_interest_multiplier: 1.5                 # scalar — treated as [1.5, 1.5]
      attack_delay: 3

    # ── Rabbit: matched smell, per-episode wander ──
    - class: "neutral"
      behaviour: "wander"
      tag: "TL"
      count: 1
      properties:     [0.0, 1.0, 0.0, 0.0, 0.0]    # matched-smell rabbit
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      damage: [0.0, 0.0]                            # rabbits don't damage
      spawn_area:  [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
      # ── per-episode fields default to neutral-friendly placeholders ──
      # (loader fills these with zeros; the wander branch ignores them)
```

**Backward-compat schema (the existing way, all 74 configs)** — `predators:` + `neutral_animals:` keep working. The loader detects which form is present and dispatches:

| If config has … | Loader does … |
|---|---|
| `environment.entities:` (new) | Parse as unified list; ignore `predators:` / `neutral_animals:` if also present and emit a deprecation warning. |
| `environment.predators:` and/or `environment.neutral_animals:` (legacy) | Re-project each `predators:` entry into a unified entry with `class='predator'`, `behaviour='hunt'`, and each `neutral_animals:` entry with `class='neutral'`, `behaviour='wander'`. The five distributional fields are read as scalars from the legacy YAML (since the legacy schema is scalar-only) and stored as degenerate ranges `[s, s]`. (The `predator_enabled` flag is removed; see Risks item 4 + CP1 migration sweep.) |
| Neither | Treat as zero animals (empty arrays). |

#### `EnvState` / `EnvParams` field layout

All `[num_animals, ...]` arrays in the table below are pytree-node JAX arrays. `tuple[str, ...]` fields are `struct.field(pytree_node=False)` static metadata.

`EnvParams` — additions and removals:

| Field | Shape / Type | Static? | Notes |
|---|---|:---:|---|
| `animal_classes` | `tuple[str, ...]`, len `N` | ✓ | per-entity class string |
| `animal_behaviours` | `tuple[str, ...]`, len `N` | ✓ | per-entity behaviour string |
| `animal_behaviours_int` | `jnp.array[N] int32` | — | `{wander:0, hunt:1, static:2}` — runtime dispatch key |
| `animal_classes_int` | `jnp.array[N] int32` | — | `{predator:0, neutral:1, ...}` — runtime dispatch key for visual scatter + damage mask |
| `animal_tags` | `tuple[str, ...]`, len `N` | ✓ | metric label suffix |
| `animal_is_damaging` | `jnp.array[N] bool` | — | precomputed from `animal_classes == 'predator'`; used by damage mask |
| `animal_property` | `[N, V]` float | — | olfactory signature mean |
| `animal_property_std` | `[N, V]` float | — | per-spawn re-sample std |
| `animal_nociception` | `[N]` float | — | extero-noc intensity on contact |
| `animal_move_int` | `[N]` int32 | — | move-timer reset value |
| `animal_damage` | `[N, 2]` float | — | per-event uniform low/high |
| `animal_attack_delay` | `[N]` int32 | — | post-attack cooldown |
| `animal_spawn_area` | `[N, 4]` int32 | — | min_r, min_c, max_r, max_c |
| `animal_patrol` | `[N, 4]` int32 | — | patrol-box clamp |
| `animal_detect_low` / `animal_detect_high` | `[N]` float | — | per-episode uniform bounds (NEW) |
| `animal_max_stamina_low` / `animal_max_stamina_high` | `[N]` float | — | per-episode uniform bounds (NEW) |
| `animal_recovery_low` / `animal_recovery_high` | `[N]` float | — | per-episode uniform bounds (NEW) |
| `animal_hunt_thresh_low` / `animal_hunt_thresh_high` | `[N]` float | — | per-episode uniform bounds (NEW) |
| `animal_lose_interest_low` / `animal_lose_interest_high` | `[N]` float | — | per-episode uniform bounds (NEW) |

To be removed: `pred_property`, `pred_property_std`, `pred_nociception`, `pred_move_int`, `pred_damage`, `pred_patrol`, `pred_detect`, `pred_max_stamina`, `pred_recovery`, `pred_hunt_thresh`, `pred_attack_delay`, `pred_lose_interest_mult`, `pred_spawn_area`, `predator_tags`, and the seven `neutral_*` fields. (See §"File Changes" for line numbers.)

`EnvState` — additions and removals:

| Field | Shape | Notes |
|---|---|---|
| `animal_pos` | `[N, 2]` int32 | replaces `pred_pos` + `neutral_pos` |
| `animal_state` | `[N]` int32 | replaces `pred_state`; carries the HUNT/PATROL/RETURN code; for `wander/static` behaviour this stays 0 (unused) |
| `animal_stamina` | `[N]` float | replaces `pred_stamina`; for `wander/static`, stays at `animal_max_stamina_sampled` (unused but kept for shape stability) |
| `animal_move_timer` | `[N]` int32 | replaces `pred_move_timer` + `neutral_move_timer` |
| `animal_attack_timer` | `[N]` int32 | replaces `pred_attack_timer`; zero for non-hunt entities |
| `animal_property_sampled` | `[N, V]` float | replaces `pred_property_sampled` + `neutral_property_sampled` |
| `animal_detect_sampled` | `[N]` float | per-episode draw (NEW) |
| `animal_max_stamina_sampled` | `[N]` float | per-episode draw (NEW) |
| `animal_recovery_sampled` | `[N]` float | per-episode draw (NEW) |
| `animal_hunt_thresh_sampled` | `[N]` float | per-episode draw (NEW) |
| `animal_lose_interest_sampled` | `[N]` float | per-episode draw (NEW) |

To be removed: `pred_pos`, `pred_state`, `pred_stamina`, `pred_move_timer`, `pred_attack_timer`, `pred_property_sampled`, `neutral_pos`, `neutral_move_timer`, `neutral_property_sampled`.

**JIT shape-stability check.** `N = num_animals` is known at config-load time and identical across all calls in a training run, so all `[N, ...]` shapes are static. Different configs with the same `N` and same flag-set will *not* trigger recompile. Configs with different `N` will recompile — same as today.

#### `update_animals()` dispatch

The function takes the full unified `animal_*` state and the agent position, and returns the updated state. Inside, we compute **all three behaviour branches simultaneously per entity** (vectorised), then **mask-select** by `animal_behaviours_int`. This avoids the JIT compile cost of `lax.switch` per-entity-in-vmap, which can blow up when JAX has to specialise per-branch.

```
# pseudocode (real implementation is in core.py)

def update_animals(animal_pos, animal_state, animal_stamina,
                   animal_move_timer, animal_attack_timer,
                   animal_detect_sampled, animal_max_stamina_sampled,
                   animal_recovery_sampled, animal_hunt_thresh_sampled,
                   animal_lose_interest_sampled,
                   agent_pos, obs_pos, obs_blocking, params, key):

    beh_int = params.animal_behaviours_int   # [N], static at trace time

    # ── Branch A: HUNT (the current update_predators body, vectorised) ──
    hunt_pos, hunt_state, hunt_stam, hunt_mt, hunt_at, key = _hunt_step(
        animal_pos, animal_state, animal_stamina,
        animal_move_timer, animal_attack_timer,
        animal_detect_sampled, animal_max_stamina_sampled,
        animal_recovery_sampled, animal_hunt_thresh_sampled,
        animal_lose_interest_sampled,
        agent_pos, obs_pos, obs_blocking, params, key)

    # ── Branch B: WANDER (the current update_neutral_animals body, vectorised) ──
    wand_pos, wand_mt, key = _wander_step(
        animal_pos, animal_move_timer,
        obs_pos, obs_blocking, params, key)

    # ── Branch C: STATIC (no-op) ──
    static_pos = animal_pos
    static_mt = animal_move_timer

    # ── Mask-combine by behaviour ──
    is_hunt   = (beh_int == 1)
    is_wander = (beh_int == 0)
    # static is implicit complement

    new_pos = jnp.where(is_hunt[:, None], hunt_pos,
                jnp.where(is_wander[:, None], wand_pos, static_pos))
    new_state   = jnp.where(is_hunt, hunt_state, animal_state)      # 0 for wander/static
    new_stamina = jnp.where(is_hunt, hunt_stam, animal_stamina)     # frozen for wander/static
    new_mt = jnp.where(is_hunt, hunt_mt,
              jnp.where(is_wander, wand_mt, static_mt))
    new_at = jnp.where(is_hunt, hunt_at, animal_attack_timer)       # 0 for wander/static
    return new_pos, new_state, new_stamina, new_mt, new_at, key
```

Rationale for **mask-combine over `lax.switch`**: with N small (≤10 in current configs) and three cheap branches, the FLOPs cost of running all three is negligible compared to the compile-time blowup `lax.switch` can introduce when wrapped in vmap. The mask form is also straightforward to keep parity-equivalent to today's code: `_hunt_step` is a near-verbatim transcription of `update_predators` (only the per-entity bound arrays change from `params.pred_*` to the new `animal_*_sampled`), and `_wander_step` is a verbatim transcription of `update_neutral_animals`.

#### Per-episode sampling inside `jax_reset`

```
# inside jax_reset, after the existing key splits:

key, animal_episode_key = jax.random.split(key)
ek1, ek2, ek3, ek4, ek5 = jax.random.split(animal_episode_key, 5)

animal_detect_sampled  = jax.random.uniform(ek1, (N,),
    minval=params.animal_detect_low,  maxval=params.animal_detect_high)
animal_max_stamina_sampled = jax.random.uniform(ek2, (N,),
    minval=params.animal_max_stamina_low, maxval=params.animal_max_stamina_high)
animal_recovery_sampled    = jax.random.uniform(ek3, (N,),
    minval=params.animal_recovery_low,    maxval=params.animal_recovery_high)
animal_hunt_thresh_sampled = jax.random.uniform(ek4, (N,),
    minval=params.animal_hunt_thresh_low, maxval=params.animal_hunt_thresh_high)
animal_lose_interest_sampled = jax.random.uniform(ek5, (N,),
    minval=params.animal_lose_interest_low, maxval=params.animal_lose_interest_high)
```

Each `jax.random.uniform` over `[low, high]` with `low == high` returns the constant `low`, so the degenerate-range case is correct without a special branch. This means `detection_range: 5` (scalar in legacy YAML) → `low = high = 5.0` → sampled value `5.0` every episode → byte-identical to today's behaviour.

**Initial stamina** at reset uses the sampled max: `animal_stamina = animal_max_stamina_sampled` (replaces today's `pred_stamina = full(num_pred, params.pred_max_stamina)`).

#### Sensor refactor (visual scatter, olfactory unified)

`sense_visual` (`sensor.py:137`):

```
# OLD: lines 184-193
res_props     = jax.nn.one_hot(jnp.where(params.res_type == 0, 3, 4), 8)
pred_props    = jax.nn.one_hot(jnp.full((num_pred,), 5), 8)
obs_props     = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)
neutral_props = jax.nn.one_hot(jnp.full((num_neutral,), 7), 8)

# NEW:
res_props     = jax.nn.one_hot(jnp.where(params.res_type == 0, 3, 4), 8)
# class_to_channel: 'predator' -> 5, 'neutral' -> 7. Encoded statically at param-build time.
animal_channels = params.animal_visual_channel   # jnp.array[N] int32 — NEW EnvParams field
animal_props  = jax.nn.one_hot(animal_channels, 8)
obs_props     = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)

all_pos   = jnp.concatenate([state.res_pos, state.animal_pos, state.obs_pos], axis=0)
all_active = jnp.concatenate([
    state.res_active,
    jnp.ones(state.animal_pos.shape[0], dtype=jnp.bool_),
    jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_),
], axis=0)
all_props = jnp.concatenate([res_props, animal_props, obs_props], axis=0)
```

`animal_visual_channel` is a new `[N] int32` `EnvParams` field, built at param-load time from `animal_classes` via the table `{'predator': 5, 'neutral': 7}`. Future classes append entries to this table; channel 6 stays reserved for rocks. The concat order changes from `[res, pred, obs, neutral]` to `[res, animal, obs]`, so the matmul result for any cell is `sum_e match(cell, e) * one_hot(channel_of_e)` — and since the channels for predator and neutral animals are unchanged, the visual vector for the parity-reference config is byte-identical.

`get_observation` olfactory block (`sensor.py:289–295`):

```
# OLD:
pred_chem    = sense_resource(..., state.pred_pos, ..., state.pred_property_sampled, ...)
neutral_chem = sense_resource(..., state.neutral_pos, ..., state.neutral_property_sampled, ...)
obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)

# NEW:
animal_chem = sense_resource(state.agent_pos, state.animal_pos,
    jnp.ones(state.animal_pos.shape[0], dtype=jnp.bool_),
    state.animal_property_sampled, params.sensor_radius, params.sensor_decay)
obs_parts.append(res_chem + animal_chem + obs_chem)
```

Olfactory parity follows from the fact that the resource summation is order-invariant (it's a sum), so concatenating predators + neutrals into one `animal_*` array and summing gives the same scalar.

#### Damage logic (`core.py:395–404`)

```
# OLD:
at_predator = jnp.all(new_pred_pos == new_agent_pos, axis=-1)
sampled_pred_damage = jax.random.uniform(damage_key, (params.pred_damage.shape[0],),
    minval=params.pred_damage[:, 0], maxval=params.pred_damage[:, 1])
damage_pred = jnp.sum(jnp.where(at_predator, sampled_pred_damage, 0.0))
new_pred_attack_timer = jnp.where(at_predator, params.pred_attack_delay, new_pred_attack_timer)

# NEW:
at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)
# Only damaging classes (predator) deal damage; neutrals don't.
at_damaging = jnp.logical_and(at_animal, params.animal_is_damaging)
sampled_animal_damage = jax.random.uniform(damage_key, (params.animal_damage.shape[0],),
    minval=params.animal_damage[:, 0], maxval=params.animal_damage[:, 1])
damage_pred = jnp.sum(jnp.where(at_damaging, sampled_animal_damage, 0.0))
new_animal_attack_timer = jnp.where(at_damaging, params.animal_attack_delay, new_animal_attack_timer)

# Also: info['hit_predator'] = any(at_damaging) ; info['hit_neutral'] = any(at_animal & ~animal_is_damaging)
```

Parity check: with the parity-reference config, `animal_is_damaging` is `[1, 0, 0, 0]` (1 predator + 3 rabbits — note the reference config has rabbits not 3 but 2; the example is illustrative), and `at_damaging` picks out exactly the predator hits, identical to today.

#### Analysis-side helper

In `src/behavior/util.py` (new file, ~30 LOC):

```python
def select_by_class(params, class_name: str) -> jnp.ndarray:
    """Return [N] bool mask selecting animals of class_name. Static at trace time."""
    return jnp.array([c == class_name for c in params.animal_classes], dtype=jnp.bool_)
```

Used by accumulators and eval_rollout to compute per-class slices identical to today's `dist_per_predator` / `dist_per_neutral`. The `info` dict carries:

- `dist_per_animal` — `[N]` float, replaces `dist_per_predator` + `dist_per_neutral`.
- For backward-compat with existing analysis scripts and notebooks, **also** populate the legacy keys `dist_per_predator` and `dist_per_neutral` from `dist_per_animal[predator_mask]` and `dist_per_animal[neutral_mask]`. Both are static slices known at param-build time. Same for `info['hit_predator']`, `info['hit_neutral']`.

This keeps every downstream consumer (`accumulators.py`, `distance_aggregator.py`, `motif_cluster.py`, `eval_rollout.py`) working **without code changes** for the first cut. The analysis-side touch-points still get refactored in CP6 to consume `dist_per_animal` + `select_by_class` directly, but the legacy keys remain as compatibility aliases until CP6 lands.

#### Per-episode logging of sampled values

Add to `info` in `jax_reset`'s return path: a one-time per-episode summary written to `EnvState` (it's already there as the five `*_sampled` fields). The per-step logger reads these at episode boundaries:

- WandB metric keys (one per animal-tag-and-field combination):
  - `Episode/sampled_detect_<tag>` (mean per episode across N envs)
  - `Episode/sampled_max_stamina_<tag>`
  - `Episode/sampled_recovery_<tag>`
  - `Episode/sampled_hunt_thresh_<tag>`
  - `Episode/sampled_lose_interest_<tag>`

Plumbed through `src/behavior/accumulators.py` (`build_episode_log_dict`) — same shape as the existing distance-per-tag plumbing. Since the values are constant within an episode, the logger only needs the value at any single step per episode; we pick step 0 (right after reset).

### File Changes

#### `src/environment/state.py` (~190 lines total today)

**Remove** (lines 19–24, 27–29, 74–89, 104–110): all `pred_*` and `neutral_*` fields on both `EnvState` and `EnvParams`. **Also remove `predator_enabled: bool`** (the 86 configs that reference it are migrated in CP1's sweep — see CP1 spec).

**Add** to `EnvState`:

```python
# Animals (unified)
animal_pos: jnp.ndarray              # [N, 2]
animal_state: jnp.ndarray            # [N] int (PATROL=0, HUNT=1, RETURN=2; unused for wander/static)
animal_stamina: jnp.ndarray          # [N] float
animal_move_timer: jnp.ndarray       # [N] int
animal_attack_timer: jnp.ndarray     # [N] int
animal_property_sampled: jnp.ndarray # [N, V]
# Per-episode-sampled behavioural params
animal_detect_sampled: jnp.ndarray         # [N]
animal_max_stamina_sampled: jnp.ndarray    # [N]
animal_recovery_sampled: jnp.ndarray       # [N]
animal_hunt_thresh_sampled: jnp.ndarray    # [N]
animal_lose_interest_sampled: jnp.ndarray  # [N]
```

**Add** to `EnvParams`:

```python
# Animals (unified)
animal_property: jnp.ndarray       # [N, V]
animal_property_std: jnp.ndarray   # [N, V]
animal_nociception: jnp.ndarray    # [N]
animal_move_int: jnp.ndarray       # [N] int
animal_damage: jnp.ndarray         # [N, 2]
animal_attack_delay: jnp.ndarray   # [N] int
animal_spawn_area: jnp.ndarray     # [N, 4] int
animal_patrol: jnp.ndarray         # [N, 4] int
# Per-episode uniform bounds
animal_detect_low: jnp.ndarray         # [N]
animal_detect_high: jnp.ndarray        # [N]
animal_max_stamina_low: jnp.ndarray    # [N]
animal_max_stamina_high: jnp.ndarray   # [N]
animal_recovery_low: jnp.ndarray       # [N]
animal_recovery_high: jnp.ndarray      # [N]
animal_hunt_thresh_low: jnp.ndarray    # [N]
animal_hunt_thresh_high: jnp.ndarray   # [N]
animal_lose_interest_low: jnp.ndarray  # [N]
animal_lose_interest_high: jnp.ndarray # [N]
# Per-entity static dispatch keys (int-coded)
animal_classes_int: jnp.ndarray        # [N] int (0=predator, 1=neutral, ...)
animal_behaviours_int: jnp.ndarray     # [N] int (0=wander, 1=hunt, 2=static)
animal_is_damaging: jnp.ndarray        # [N] bool (precomputed from class)
animal_visual_channel: jnp.ndarray     # [N] int (5=predator, 7=neutral, ...)
# Static tags / labels
animal_classes: tuple = struct.field(pytree_node=False)      # len N strings
animal_behaviours: tuple = struct.field(pytree_node=False)   # len N strings
animal_tags: tuple = struct.field(pytree_node=False)         # len N strings
```

#### `src/environment/config_loader.py` (today 251–366 for the two paths)

**Add** at module top: class/behaviour constant maps:

```python
ANIMAL_CLASS_TO_INT = {"predator": 0, "neutral": 1}
ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}
ANIMAL_DAMAGING_CLASSES = {"predator"}
ANIMAL_BEHAVIOUR_TO_INT = {"wander": 0, "hunt": 1, "static": 2}
DISTRIBUTIONAL_FIELDS = (
    "detection_range",
    "max_stamina",
    "stamina_recovery_rate",
    "hunt_stamina_threshold",
    "lose_interest_multiplier",
)
```

**Replace** lines 228–366 with a single `_load_animals()` helper:

1. If `environment.entities` is present → parse the unified list directly.
2. Else, build the unified list by re-projecting `environment.predators` (each entry gets `class='predator'`, `behaviour='hunt'`) and `environment.neutral_animals` (each entry gets `class='neutral'`, `behaviour='wander'`). Concatenate in that order so per-class metric slicing yields the same layout as today.
3. For each entry, for each field in `DISTRIBUTIONAL_FIELDS`: read scalar or `[low, high]`; if scalar `s`, treat as `[s, s]`. Missing → raise `ValueError` (no fallback default, per project rule).
4. Build all `animal_*` jnp arrays at the bottom of the function and return them.

(The loader no longer reads `predator_enabled` — that flag is fully removed in this CP; the 86 affected configs are migrated atomically with the schema change.)

`get_mandatory` semantics: `environment.predators` and `environment.neutral_animals` are kept as `get_mandatory` reads (legacy invariant — every existing config has them, even if empty list). When `entities:` is present, both legacy keys become optional and the loader allows them to be missing or `None`. Document this in the loader docstring.

#### `src/environment/core.py`

**Remove** `update_predators` (lines 132–251) and `update_neutral_animals` (lines 253–286).

**Add** `update_animals(state, agent_pos, params, key)` per the pseudocode above. Internals:

- `_hunt_step` is a near-verbatim transcription of today's `update_predators`, with `params.pred_detect → state.animal_detect_sampled`, `params.pred_max_stamina → state.animal_max_stamina_sampled`, `params.pred_recovery → state.animal_recovery_sampled`, `params.pred_hunt_thresh → state.animal_hunt_thresh_sampled`, `params.pred_lose_interest_mult → state.animal_lose_interest_sampled`. All other inputs stay as `params.*` (`pred_patrol → animal_patrol`, `pred_move_int → animal_move_int`, `pred_attack_delay → animal_attack_delay`).
- `_wander_step` is a verbatim transcription of `update_neutral_animals` with `neutral_patrol → animal_patrol`, `neutral_move_int → animal_move_int`.

**Patch** `jax_step` (lines 288–555):

- Line 293: keep the 6-way key split (rename: `key, respawn_key, hunt_key, wander_key, damage_key, property_key`). Combine `hunt_key + wander_key` inside `update_animals` so per-step PRNG-stream parity holds against `update_predators(predator_key)` + `update_neutral_animals(neutral_key)`.
- Lines 321–330: replace the two separate update calls with one `update_animals` call.
- Lines 395–404: change damage logic per §"Damage logic" above.
- Lines 440–443: `info['hit_neutral']` becomes `any(at_animal & ~animal_is_damaging)`.
- Lines 495–515: `dist_to_pred` and `dist_to_neutral` derived from `state.animal_pos` masked by `params.animal_is_damaging` (predator distances) and `~animal_is_damaging` (neutral distances). `dist_per_predator` and `dist_per_neutral` similarly. `dist_per_animal` is the unmasked vector.
- Lines 528–553: update `state._replace(...)` to use the new `animal_*` field names.

**Patch** `jax_reset` (lines 657–817):

- Lines 666: extend the 5-way key split to 6-way (`key, agent_key, placement_key, body_key, property_key, animal_episode_key`).
- Lines 673–676: replace the `num_pred`, `num_neutral` derivation with `num_animals = params.animal_property.shape[0]`.
- Lines 678–751 (placement modes): the per-entity and per-type placement scans currently treat predator and neutral as separate spawn-area groups. Re-project: in `per_entity` mode, concatenate `res / animal / obs` spawn areas (was `res / pred / obs / neutral`). In `per_type` mode, the YAML's `placement.types:` definition is loaded by the existing per-type logic — verify the config-loader produces the right `type_entity_map` indices when animals are unified. **Detailed check needed in CP3**: per-type placement reads `type_entity_map` indices, which today partition the flat `[res, pred, obs, neutral]` index space; the loader's `type_entity_map` builder needs updating to use `[res, animal, obs]` ordering. The reference config uses `per_entity` mode, so the per-type path is exercised only by configs that opt into it — list them in CP3 and verify all 74 parity tests.
- After placement, add the per-episode uniform sampling block (§"Per-episode sampling inside `jax_reset`" above).
- Lines 776–784 (property sampling): collapse `pred_property_sampled` + `neutral_property_sampled` into one `animal_property_sampled`. Use a single `prop_key_animal`.
- Lines 786–815 (state assembly): replace `pred_*` and `neutral_*` initialisers with `animal_*` initialisers (per the `EnvState` schema above). `animal_state = jnp.zeros(N, dtype=jnp.int32)` (PATROL=0; semantically inert for wander/static).

#### `src/environment/sensor.py`

- Line 168–198 (`sense_visual`): replace the 4-way concat with the 3-way concat (res, animal, obs) and use `params.animal_visual_channel` for animal channels. See §"Sensor refactor" above for exact diff.
- Lines 290–295 (`get_observation` olfactory): replace `pred_chem + neutral_chem` with `animal_chem`.

#### `src/environment/grid_world.py` (lines 445, 466)

`p_pos = np.array(state.pred_pos)` → derive predator positions from `state.animal_pos` filtered by `params.animal_is_damaging` (host-side). Similarly for `n_pos` from neutrals. Wrap in a small host-side helper.

#### `src/environment/renderer.py` (lines 461, 482) and `src/environment/renderer_v2.py` (lines 374, 380)

Same change as `grid_world.py`. Both renderers visually distinguish predators from neutrals by colour; that distinction comes from the static class slice.

#### `src/utils/evaluation_core.py` (lines 64–65, 121–128, 249, 387–388, 465–466, 534, 579–580, 662)

Every `state.pred_pos` / `state.neutral_pos` read becomes a `select_by_class` host-side slice on `state.animal_pos`. The function's CSV output columns must remain in the same order (predator-cols-first, then neutral-cols) so existing eval-data consumers don't break — use `params.animal_classes` to emit the columns in `predator → neutral` order.

#### `src/utils/eval_recording.py` (lines 34–35)

Same as above — replace `state.pred_pos` and `state.neutral_pos` with class-sliced reads from `state.animal_pos`.

#### `src/behavior/util.py` (new file)

```python
"""Behaviour-analysis utilities for the unified animal-entity layout (v2.0)."""
import jax.numpy as jnp


def select_by_class(params, class_name: str) -> jnp.ndarray:
    """Return [N] bool mask selecting animals of class_name.

    Static at trace time (depends only on params.animal_classes, which is a
    pytree_node=False tuple). Use as a constant inside jax.jit.
    """
    return jnp.array([c == class_name for c in params.animal_classes], dtype=jnp.bool_)


def class_indices(params, class_name: str) -> tuple[int, ...]:
    """Return tuple of integer indices i such that animal_classes[i] == class_name.

    Host-side only. Use for static slicing in Python before jit (e.g., when
    building per-class metric loggers).
    """
    return tuple(i for i, c in enumerate(params.animal_classes) if c == class_name)
```

#### `src/behavior/accumulators.py` (multiple lines per inventory above)

- Lines 38–39, 111–120: replace `num_predator_tags` + `num_neutral_tags` with one `num_animal_tags` + a Python-side `predator_indices` / `neutral_indices` derived from `params.animal_classes`. The runtime arrays continue to expose two slices, but the slicing math now uses `class_indices(params, 'predator')` and `class_indices(params, 'neutral')`.
- Lines 175–176: read `info['dist_per_animal']` (unified). The legacy keys `dist_per_predator` / `dist_per_neutral` are kept as compatibility aliases for one release cycle and **also** read here so existing logger output is unchanged.
- Lines 392–395, 432–444, 460–482: per-tag loops over `predator_tags` and `neutral_tags` keep their names — the tags now come from sliced `animal_tags`. The output WandB metric key namespace is preserved (`Episode/MeanDistPredator_<tag>` and `Episode/MeanDistNeutral_<tag>`).
- **Add** per-episode sampled-value logging: at episode reset (or step 0), pull `state.animal_*_sampled[i]` for each animal `i` and emit:
  - `Episode/sampled_detect_<tag>`
  - `Episode/sampled_max_stamina_<tag>`
  - `Episode/sampled_recovery_<tag>`
  - `Episode/sampled_hunt_thresh_<tag>`
  - `Episode/sampled_lose_interest_<tag>`
- Document the new metric key family in the function docstring and in `docs/develop/active/behavior/` (out of scope for this plan — flag as a follow-up).

#### `src/behavior/distance_aggregator.py` (lines 67–75, 97–98, 114–117, 124–139)

- Continue reading the legacy keys `dist_per_predator` / `dist_per_neutral` from `info` (kept as aliases per above).
- Continue accepting `predator_tags` and `neutral_tags` as separate kwargs to public API — internally they are now sliced from `animal_tags`.

#### `scripts/eval_rollout.py` (lines 87–88, 106–109, 127–128, 162–163, 220–221)

Read `dist_per_animal` if present (preferred) and fall back to `dist_per_predator` / `dist_per_neutral` for backward compat with older roll-out data. Add a CLI flag `--unified` (default `False`) that emits `dist_per_animal` directly instead of the two legacy slices, for downstream notebooks that have been updated. **Defer the CLI-flag UX work to a follow-up plan** — for CP6, just keep both keys live in output.

#### `scripts/motif_cluster.py` (lines 78–79)

Same — read either form, prefer the unified key.

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` and `src/models/dreamer_v3_trainer.py`, `src/models/recurrent_ppo_trainer.py`

These files appear in the grep but only via the same WandB-key-suffix machinery — they consume `predator_tags` / `neutral_tags` through `accumulators.py` and don't reach into `state.pred_*` directly. **Verify in CP4** that no trainer file reads `state.pred_*` or `state.neutral_*` directly. Likely no diff here.

#### `scripts/verification/check_olfaction_parity.py` and `scripts/verify_noise.py`

These existing parity scripts already do byte-level comparison of observations. They will be reused as the parity-gate for CP1 — no edits required, but the developer should run them against the parity-reference config before and after the refactor and confirm zero diff.

#### `scripts/benchmark_render.py`

Same as the renderers — replace `state.pred_pos` / `state.neutral_pos` reads with `select_by_class` slices.

### Checkpoints

The refactor lands in six checkpoints. CP1 alone is the **minimum viable landing** (schema + loader + backward-compat shim + parity tests). CP2–CP5 add the per-episode sampling layer. CP6 is the analysis-side cleanup; until CP6, the legacy `dist_per_predator` / `dist_per_neutral` / `pred_*` arrays survive as compatibility aliases.

- [ ] **CP1 — Unified state + loader + backward-compat parity + `predator_enabled` migration sweep.** Implement `EnvState` / `EnvParams` field changes (including full removal of `predator_enabled`), `update_animals` (with all distributional sampling no-ops: bounds set to `[scalar, scalar]` from legacy YAML), `config_loader._load_animals`. **Atomically migrate all 86 configs that reference `predator_enabled`**: (a) sed-strip the 85 `predator_enabled: true` lines (no behavioural change); (b) hand-migrate `configs/verification/olfaction_parity_neutral.yaml` to use an empty `predators: []` list in place of `predator_enabled: false`. **Verifies:** `pytest tests/env/test_unified_parity.py` (new) passes — for each of all 74 `configs/experiment/**/*.yaml`, running 100 steps from seed 0 produces byte-identical `obs` vectors before and after the refactor. The existing parity scripts `scripts/verification/check_olfaction_parity.py` and `scripts/verify_noise.py` also exit 0. **No new YAML fields are read in this CP** — the per-episode bounds come from re-projecting the legacy scalar to `[scalar, scalar]`.
- [ ] **CP2 — Per-episode sampling at reset.** Add the 5-way key split in `jax_reset` and the `jax.random.uniform` calls; populate the 5 `animal_*_sampled` fields. **Verifies:** with all legacy configs (degenerate ranges) the parity gate still passes byte-for-byte (uniform`[s, s]` ≡ `s`). A new test `tests/env/test_per_episode_sampling.py` verifies: (a) same `key` ⇒ same sampled values; (b) different `key` ⇒ different sampled values; (c) N entities of the same class give N independent samples (not N copies). 
- [ ] **CP3 — `entities:` schema in config loader.** Add the new `environment.entities:` YAML path; loader prefers it over legacy if both present (with warning). Update `placement.types:` handling: ensure `type_entity_map` uses `[res, animal, obs]` indexing in `per_type` mode. **Verifies:** a new `configs/experiment/v2_smoke/01-entities-smoke.yaml` config that uses the unified schema for the same parity-reference setup produces byte-identical behaviour to `01-interoNocicept_sameProp.yaml`. New test `tests/env/test_entities_schema.py` covers: (a) the unified config loads; (b) byte-parity vs legacy; (c) loader warns when both legacy and unified sections are present; (d) configs that still contain the removed `predator_enabled` key raise `ValueError` with a clear migration message (sanity check; the CP1 sweep should have removed all of them already).
- [ ] **CP4 — Sensor + damage + step path parity.** Patch `sense_visual` (class scatter), `get_observation` olfactory (unified `animal_chem`), and `jax_step` damage logic (`animal_is_damaging`). **Verifies:** the CP1 parity test (all 74 configs × 100 steps) still passes — this CP shouldn't *add* parity coverage but must *not break* it. Additionally, a new `tests/env/test_visual_parity.py` runs one episode (1000 steps) on the parity-reference config and asserts the visual one-hot per cell is byte-identical to a pinned reference dump (committed under `tests/env/fixtures/visual_parity_ref.npz`). Trainer verification per CP4 above (no trainer reads `state.pred_*` directly).
- [ ] **CP5 — Distributional schema + per-episode logging.** Add YAML parsing for `[low, high]` ranges on the five fields under scope; emit `Episode/sampled_*_<tag>` WandB metrics. **Verifies:** new `tests/env/test_distributional_yaml.py` covers (a) scalar `5` → degenerate range `[5, 5]`, (b) `[0, 5]` → bounds stored correctly, (c) malformed range like `[5]` raises `ValueError`. End-to-end smoke: run 1000 training steps with `configs/experiment/v2_smoke/02-entities-distributional.yaml` (NEW — has `detection_range: [0, 5]` per predator) and confirm the 5 `sampled_*` WandB metrics appear with non-degenerate values. **JIT-recompile check**: swap two configs with the same animal counts but different bounds; confirm JAX does not recompile (use `jax.jit.lower(...).compile()` traces or `jax.config.update("jax_log_compiles", True)`).
- [ ] **CP6 — Analysis-side cleanup.** Update `accumulators.py`, `distance_aggregator.py`, `eval_rollout.py`, `motif_cluster.py`, `evaluation_core.py`, `eval_recording.py`, `grid_world.py`, `renderer.py`, `renderer_v2.py`, `benchmark_render.py` to consume `dist_per_animal` + `select_by_class` directly. **Verifies:** `pytest tests/behavior/test_accumulators.py` passes with the same metric layout as before; the WandB metric key names are unchanged (`MeanDistPredator_<tag>`, `MeanDistNeutral_<tag>` still produced). Legacy aliases `dist_per_predator` / `dist_per_neutral` removed from `info` (one release-cycle later, not in this CP).

### Test Plan

The test plan operationalises the locked parity gate. The full test surface:

#### (a) Per-step parity test — `tests/env/test_unified_parity.py` (NEW, CP1)

For each `config_path in glob('configs/experiment/**/*.yaml')`:
1. Load with the **old** loader path (pre-refactor commit — pulled via git via a one-shot script that builds a comparison snapshot, OR equivalently against a hard-pinned fixture per config).
2. Load with the **new** loader path.
3. Run 100 steps from seed 0 (using `jax_reset(params, jax.random.PRNGKey(0))` then 100 `jax_step` calls with actions `[0, 1, 2, 3, 4] * 20`).
4. Assert `jnp.array_equal(old_obs[t], new_obs[t])` for every step `t`.
5. Assert `jnp.array_equal(old_state.<field>, new_state.<field>)` for every common field (post-renaming: predator/neutral fields → matching `animal_*` slices).

**Failure mode:** any single step diff blocks CP1.

#### (b) Per-episode sampling test — `tests/env/test_per_episode_sampling.py` (NEW, CP2)

Use a config where `detection_range: [0, 5]` for 4 predators:
1. Reset with key `K`, capture `state.animal_detect_sampled` → call this `s_K`.
2. Reset again with key `K`, capture sampled → assert `s_K == s'`.
3. Reset with key `K + 1`, capture → assert `s != s_K` (with high probability — N=4 gives effectively no false positives at sigma=2.89 uniform).
4. Assert the 4 values inside `s_K` are not all-identical (per-instance independence).

#### (c) Backward-compat test — `tests/env/test_backward_compat_configs.py` (NEW, CP1)

Loop over all `configs/experiment/**/*.yaml`, call `load_env_params(Config.from_file(path))`. Assert no exception. Assert `params.animal_property.shape[0] == num_predators_in_yaml + num_neutrals_in_yaml` (where the YAML's `count:` fields are summed by class). This catches any config that the new loader can't handle.

#### (d) Visual parity test — `tests/env/test_visual_parity.py` (NEW, CP4)

For the parity-reference config (`01-interoNocicept_sameProp.yaml`):
1. Run one episode of 1000 steps from seed 0 with actions `[0, 1, 2, 3, 4] * 200`.
2. Save the visual-sensor slice of each `obs` to `tests/env/fixtures/visual_parity_ref.npz` (committed alongside the test).
3. The test re-runs and asserts byte-equality with the fixture.

This guards against future regressions and locks the visual-channel-class-coding behaviour the user explicitly wants preserved.

#### (e) JIT-recompile test — `tests/env/test_no_recompile.py` (NEW, CP5)

Two configs with the same animal counts (`1 predator + 2 rabbits`) but different `detection_range` bounds (`[0, 5]` vs `[2, 7]`). Use `jax.config.update("jax_log_compiles", True)` capture; build env-step jit for config A, run 10 steps, build for config B (same counts), run 10 steps, assert recompile count is 1 (not 2).

#### (f) Distributional schema test — `tests/env/test_distributional_yaml.py` (NEW, CP5)

YAML fixtures under `tests/env/fixtures/`:
- `dist_scalar.yaml` — `detection_range: 5` → bounds `[5, 5]`
- `dist_range.yaml` — `detection_range: [0, 5]` → bounds `[0, 5]`
- `dist_malformed_single.yaml` — `detection_range: [5]` → raise `ValueError`
- `dist_malformed_str.yaml` — `detection_range: "five"` → raise `ValueError`

#### (g) End-to-end training smoke — manual, CP5

Launch a 1000-step PPO training on `configs/experiment/v2_smoke/02-entities-distributional.yaml` via `train_command-agent.sh` (single node, single GPU). Verify in WandB that the 5 `Episode/sampled_*_<tag>` keys appear and have non-degenerate distributions across episodes. Verify training-step throughput (s/it) is within 5% of pre-refactor baseline (use `01-interoNocicept_sameProp.yaml` as the baseline) — if a regression of >5% appears, escalate.

## Implementation Report

> **Implemented by**: [to be filled by `developer` agent]
> **Date**: [to be filled]

<!-- Filled by developer after CP1–CP6 each land. Use one sub-section per CP. -->

## Verification Report

> **Verified by**: [to be filled by `senior-developer` after each CP]
> **Date**: [to be filled]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [to be filled]

---

## Reviews needed

- **`code-reviewer` (mandatory, after CP1 + after CP5)** — JAX/Flax/vmap/PRNG correctness. Three high-risk surfaces: (a) the masked-combine in `update_animals` and whether `jnp.where` mask broadcasting interacts safely with the position update; (b) the PRNG-stream parity claim (6-way key split preserved, `damage_key` reuse preserved); (c) JIT shape-stability of the `[N, ...]` arrays across configs with different bound values.
- **`env-config-auditor` (mandatory, after CP3 and CP5)** — YAML schema soundness, obs↔noise channel ordering still matches the `noise_modality_order` tuple, and the new `entities:` schema is documented in any `default.yaml` or schema-reference file the auditor uses.
- **`math-reviewer` (not needed)** — no new equations.
- **`pi` (not needed)** — this is a structural refactor with a parity gate, not a research-direction call.

## Open questions surfaced during planning

1. ~~**`predator_enabled: bool` — keep or remove?**~~ **RESOLVED 2026-05-28 — fully removed.** The 86 configs that reference the flag are migrated atomically in CP1: 85 trivially (sed-strip `predator_enabled: true`); 1 substantively (`configs/verification/olfaction_parity_neutral.yaml` → empty `predators: []`). See Risks item 4 and CP1 spec.
2. **`damage_key` reuse across resource / predator / obstacle damage (`core.py:351, 398, 410`).** Pre-existing PRNG sharing — preserved by this refactor. Worth a separate bug-fix triage but not in scope. Flagged in §"Analysis → Risks".
3. **Per-type placement mode (`placement: mode: per_type`).** The plan describes the index re-mapping but does not enumerate which configs opt into it. CP3 includes a sweep over all 74 configs; if any use `per_type` with non-trivial type-group structure that overlaps the predator/neutral split, the developer must surface them before CP3 lands so we can verify the `type_entity_map` re-projection is correct.
4. **WandB metric-key surface.** Plan preserves `MeanDistPredator_<tag>` and `MeanDistNeutral_<tag>` exactly. The new `sampled_*_<tag>` keys are additive. If the user wants a wholesale rename (e.g., to `MeanDist_<class>_<tag>`), that's a separate logging refactor — flag as a follow-up.
5. **`entities:` schema doc location.** Not part of this plan, but once CP3 lands, the YAML schema reference (likely `configs/CLAUDE.md` if it exists, or a new `configs/SCHEMA.md`) should document the unified form. Open question: where does this reference live today? Plan defers documenting until the developer surfaces the existing schema-doc home.

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
