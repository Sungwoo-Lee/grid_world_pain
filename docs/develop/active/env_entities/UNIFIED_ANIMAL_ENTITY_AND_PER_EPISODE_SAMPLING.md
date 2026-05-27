---
title: "Env Refactor — Unified Animal Entity + Per-Episode Sampling of Behavioural Params"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [unified_animal_entity, env_entities_unified_animal, env_entities_step_1]
---

# Env Refactor — Unified Animal Entity + Per-Episode Sampling of Behavioural Params

> **Status**: PLANNED (v0.2 — post-review revision)
> **Opened**: 2026-05-28
> **Branch**: `v2.0` (from `v1.4@72186aa`, latest: `f0fe297`)
> **Related**: [hypervigilance/01-interoNocicept_sameProp.yaml](../../../../configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml) (the parity-reference config), [code review](../../../reviews/env_entities_plan_review_code.md), [config audit](../../../reviews/env_entities_plan_audit_config.md), [FRONTMATTER_CONTRACT](../meta/FRONTMATTER_CONTRACT.md), [AGENT_PLAYBOOK](../../../AGENT_PLAYBOOK.md)

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

1. **PRNG byte-parity requires per-subset call shapes, not masked-combine.** Naively running both `_hunt_step` and `_wander_step` on the full unified array of length `N_animals` and selecting via `jnp.where` is **incorrect**: today's `update_predators` draws over shape `(N_pred,)` and today's `update_neutral_animals` draws over shape `(N_neutral,)`. JAX's threefry is prefix-stable, so `randint(subkey, (N_animals,), ...)[N_pred:]` differs from `randint(subkey, (N_neutral,), ...)` — predators (front of array) would retain parity, neutrals (positions ≥ N_pred) would not. **Solution adopted: per-subset call pattern.** The hunt subset and wander subset are statically sliced from the full animal arrays using host-side index tuples derived from `params.animal_classes` / `params.animal_behaviours` (both `pytree_node=False`, so the slicing is JIT-static and free). `_hunt_step` is called with shape `(N_pred,)`; `_wander_step` is called with shape `(N_neutral,)`. Results are scattered back into the unified `animal_*` arrays via `.at[idx].set(...)`. The draw shapes are byte-identical to today's, so threefry yields the same bytes. See §"`update_animals()` dispatch" for the implementation pattern. This was the resolution to a blocker raised by `code-reviewer`.
2. **JIT recompile** is controlled by making behaviour/class tuples `pytree_node=False`. A config with the same per-class entity counts (e.g., `1 predator + 3 rabbits`) but different distributional bounds **does not** recompile because the bounds are stored in traced `[N]` arrays. A config with the same N but a different per-entity class ordering (e.g., `[predator, predator, neutral]` vs `[predator, neutral, predator]`) **does** recompile, because `animal_classes` is `pytree_node=False` and the host-side hunt/wander index tuples differ. This is the expected behaviour and is positively asserted in the CP5 JIT-recompile test (positive control: swap class orderings, assert recompile fires).
3. **Parity tail risks (PRNG key consumption order)**: current `jax_step` splits `key, respawn_key, predator_key, neutral_key, damage_key, property_key` (6 keys). The refactor **preserves the 6-key split** (renamed `predator_key` → `hunt_key`, `neutral_key` → `wander_key`). Each subset call consumes its own key with the matching draw shape. Tested as part of CP1's parity gate.
4. **`damage_key` is reused 3× today** (`core.py:351, 398, 410`) — same key used for resource damage, predator damage, obstacle damage. That is a pre-existing PRNG bug (not technically a bug since `jax.random.uniform` with different shapes yields different streams, but it is suspicious). **Do not fix it in this refactor** — flag it as a separate issue if confirmed. Refactor must preserve this exact behaviour.
5. **`predator_enabled: bool` flag** (`state.py:87`) currently disables the *predator path*. **Fully removed in CP1.** The 86 existing configs that reference this flag are migrated atomically in CP1: 85 of them set `predator_enabled: true` (the default — sed-stripped, no semantic change); 1 of them (`configs/verification/olfaction_parity_neutral.yaml`) sets `predator_enabled: false` and `predators: []` already on L11 — the migration is a one-line strip of `predator_enabled: false`, not a restructuring. After CP1 lands, the loader does not read the flag; configs that still carry the key raise `ValueError` (caught by the CP1 migration sweep itself).
6. **`grid_world.py` + `renderer.py` + `renderer_v2.py`** all index `state.pred_pos` / `state.neutral_pos` for rendering. They need to switch to slicing `state.animal_pos` by class (using `select_by_class`). The visual output must remain byte-identical for the parity-reference config.

## Implementation Plan

### Design

#### Unified config schema (the new way)

A new top-level `environment.entities:` list. Each entry has fields: `class` (`predator|neutral|...`), `behaviour` (`hunt|wander|static`), `tag`, `count`, `properties`, `properties_std`, `nociception_intensity`, `move_interval`, `damage`, `spawn_area`, `patrol_area`, `attack_delay`, and the five distributional behavioural fields (`detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`). Each of those five accepts either a scalar (degenerate range) or a `[low, high]` list.

**Mandatory-key rule by behaviour (B-CFG-1 resolution).** The five distributional behavioural fields drive the predator state machine; for non-hunting entities they are unused. The rule is:

| Field group | `behaviour: hunt` entry | `behaviour: wander` / `static` entry (unified `entities:` schema) | Legacy `predators:` / `neutral_animals:` re-projection |
|---|---|---|---|
| `class`, `behaviour`, `tag`, `count`, `properties`, `properties_std`, `nociception_intensity`, `move_interval`, `damage`, `spawn_area`, `patrol_area` | **mandatory** — missing → `ValueError` | **mandatory** | already mandatory today |
| `attack_delay` | **mandatory** | **mandatory** (matches today's behaviour: `p_get('attack_delay')` at `config_loader.py:262` is unconditional) | mandatory (`p_get`) |
| 5 distributional fields (`detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`) | **mandatory** — missing → `ValueError` (no fallback default, per project rule) | **optional** — if missing, loader auto-fills `[0, 0]` and records a debug-log entry. The wander / static branch never reads these fields, so the auto-fill is an internal projection detail, not a user-facing fallback default. | **internal auto-fill** — legacy neutrals don't carry these fields at all; the loader auto-fills `[0, 0]` during re-projection. This is an internal projection detail (not a fallback default on a user-facing key), so it does not violate the "no fallback defaults" rule. |

This resolves the apparent contradiction the config auditor flagged between "Missing → raise `ValueError`" (loader code) and "loader fills these with zeros; the wander branch ignores them" (worked example): the `ValueError` applies only to `behaviour: hunt`; the zero-fill applies to `behaviour: wander` / `static` entries and to the legacy `neutral_animals:` re-projection path. The loader docstring documents this explicitly.

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
      # ── per-episode fields are OPTIONAL for behaviour: wander / static ──
      # If omitted, the loader auto-fills [0, 0] (internal projection detail,
      # not a fallback default — the wander branch never reads these fields).
      # See "Mandatory-key rule by behaviour" above (B-CFG-1 resolution).
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
| `animal_behaviours_int` | `jnp.array[N] int32` | — | `{wander:0, hunt:1, static:2}` — used by damage / metric masking (no longer used as runtime dispatch key, see §"`update_animals()` dispatch") |
| `animal_classes_int` | `jnp.array[N] int32` | — | `{predator:0, neutral:1, ...}` — runtime dispatch key for visual scatter + damage mask |
| `animal_tags` | `tuple[str, ...]`, len `N` | ✓ | metric label suffix |
| `animal_is_damaging` | `jnp.array[N] bool` | — | precomputed from `animal_classes == 'predator'`; used by damage mask |
| `hunt_idx` | `tuple[int, ...]`, len `N_pred` | ✓ | host-side index tuple for animals with `behaviour == 'hunt'`; used by `update_animals` to slice the hunt subset (preserves PRNG draw shape `(N_pred,)`) |
| `wander_idx` | `tuple[int, ...]`, len `N_neutral` | ✓ | host-side index tuple for animals with `behaviour == 'wander'`; preserves PRNG draw shape `(N_neutral,)` |
| `static_idx` | `tuple[int, ...]`, len `N_static` | ✓ | host-side index tuple for animals with `behaviour == 'static'`; no PRNG draws |
| `predator_tags` (legacy alias, @property) | derived | — | returns `tuple(animal_tags[i] for i in class_indices('predator'))`. Read by `dreamer_srl_main.py:522-523`. See "Legacy aliases" below. |
| `neutral_tags` (legacy alias, @property) | derived | — | returns `tuple(animal_tags[i] for i in class_indices('neutral'))`. Read by `dreamer_srl_main.py:522-523`. See "Legacy aliases" below. |
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

To be removed: `pred_property`, `pred_property_std`, `pred_nociception`, `pred_move_int`, `pred_damage`, `pred_patrol`, `pred_detect`, `pred_max_stamina`, `pred_recovery`, `pred_hunt_thresh`, `pred_attack_delay`, `pred_lose_interest_mult`, `pred_spawn_area`, and the seven `neutral_*` fields (`neutral_property`, `neutral_property_std`, `neutral_nociception`, `neutral_move_int`, `neutral_damage`, `neutral_patrol`, `neutral_spawn_area`). (See §"File Changes" for line numbers.)

**Legacy aliases (kept for backward-compat, one release cycle).** `predator_tags` and `neutral_tags` are **not** removed — they are re-implemented as `@property` accessors on `EnvParams` that derive from `animal_tags` + `animal_classes`. This matches the same legacy-alias strategy already used for the info-dict keys `dist_per_predator` / `dist_per_neutral` / `hit_predator` / `hit_neutral`. The accessor is host-side (it operates on `pytree_node=False` tuples, so it's free and JIT-safe):

```python
@property
def predator_tags(self) -> tuple[str, ...]:
    """Legacy alias — derived from animal_tags filtered by class == 'predator'.

    Read by src/algorithms/dreamer_srl/dreamer_srl_main.py:522-523 and
    accumulators.py setup. Will be removed once all consumers migrate to
    consume animal_tags + class_indices() directly (post-CP6 + one release).
    """
    return tuple(t for t, c in zip(self.animal_tags, self.animal_classes) if c == 'predator')

@property
def neutral_tags(self) -> tuple[str, ...]:
    """Legacy alias — derived from animal_tags filtered by class == 'neutral'."""
    return tuple(t for t, c in zip(self.animal_tags, self.animal_classes) if c == 'neutral')
```

This was the resolution to a blocker raised by `code-reviewer` (B3): `dreamer_srl_main.py:522-523` reads `env_params.predator_tags` and `env_params.neutral_tags` today; the original plan claimed no trainer touched these, which was wrong.

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

#### `update_animals()` dispatch — per-subset call pattern

The function takes the full unified `animal_*` state and the agent position, and returns the updated state. Inside, we **statically slice the hunt-subset and wander-subset** from the full animal arrays, call each behaviour's update on its own subset (with the original draw shape), then scatter the results back into the unified arrays. This is the resolution to the code-reviewer's B1 blocker: the per-subset call shapes are `(N_pred,)` and `(N_neutral,)`, byte-identical to today's `update_predators` / `update_neutral_animals` calls, so threefry yields the same bytes.

**Static slicing is free.** `params.animal_classes` and `params.animal_behaviours` are `pytree_node=False`, so `hunt_idx = tuple(i for i, b in enumerate(params.animal_behaviours) if b == 'hunt')` is computed once at trace time as a Python int tuple. JAX treats `jnp.asarray(hunt_idx)` as a static constant. There is no per-step host-device round-trip.

```python
# pseudocode (real implementation is in core.py)
#
# hunt_idx and wander_idx are precomputed once at param-build time as Python
# tuples of ints. They live alongside animal_classes / animal_behaviours on
# EnvParams as `pytree_node=False` fields:
#   params.hunt_idx   : tuple[int, ...]  e.g. (0,)        for 1 predator
#   params.wander_idx : tuple[int, ...]  e.g. (1, 2, 3)   for 3 rabbits
#   params.static_idx : tuple[int, ...]  e.g. ()          (typically empty)

def update_animals(animal_pos, animal_state, animal_stamina,
                   animal_move_timer, animal_attack_timer,
                   animal_detect_sampled, animal_max_stamina_sampled,
                   animal_recovery_sampled, animal_hunt_thresh_sampled,
                   animal_lose_interest_sampled,
                   agent_pos, obs_pos, obs_blocking, params,
                   hunt_key, wander_key):

    hunt_idx_arr   = jnp.asarray(params.hunt_idx,   dtype=jnp.int32)   # static; len N_pred
    wander_idx_arr = jnp.asarray(params.wander_idx, dtype=jnp.int32)   # static; len N_neutral

    # ── Branch A: HUNT — call with shape (N_pred,), exactly matching today's update_predators ──
    if len(params.hunt_idx) > 0:
        hunt_pos_in       = animal_pos[hunt_idx_arr]
        hunt_state_in     = animal_state[hunt_idx_arr]
        hunt_stam_in      = animal_stamina[hunt_idx_arr]
        hunt_mt_in        = animal_move_timer[hunt_idx_arr]
        hunt_at_in        = animal_attack_timer[hunt_idx_arr]
        hunt_detect       = animal_detect_sampled[hunt_idx_arr]
        hunt_max_stam     = animal_max_stamina_sampled[hunt_idx_arr]
        hunt_recovery     = animal_recovery_sampled[hunt_idx_arr]
        hunt_thresh       = animal_hunt_thresh_sampled[hunt_idx_arr]
        hunt_lose_int     = animal_lose_interest_sampled[hunt_idx_arr]
        # Per-entity params (static after slice — still pytree-traced arrays):
        hunt_patrol       = params.animal_patrol[hunt_idx_arr]
        hunt_move_int     = params.animal_move_int[hunt_idx_arr]
        # _hunt_step is the verbatim body of update_predators, lifted to take
        # the sliced arrays. Its internal jax.random.{uniform,randint} calls
        # use shape (N_pred,) — byte-identical to today.
        (new_hunt_pos, new_hunt_state, new_hunt_stam,
         new_hunt_mt, new_hunt_at) = _hunt_step(
            hunt_pos_in, hunt_state_in, hunt_stam_in, hunt_mt_in, hunt_at_in,
            hunt_detect, hunt_max_stam, hunt_recovery, hunt_thresh, hunt_lose_int,
            hunt_patrol, hunt_move_int,
            agent_pos, obs_pos, obs_blocking, hunt_key)
    else:
        new_hunt_pos = new_hunt_state = new_hunt_stam = new_hunt_mt = new_hunt_at = None

    # ── Branch B: WANDER — call with shape (N_neutral,), exactly matching today's update_neutral_animals ──
    if len(params.wander_idx) > 0:
        wand_pos_in       = animal_pos[wander_idx_arr]
        wand_mt_in        = animal_move_timer[wander_idx_arr]
        wand_patrol       = params.animal_patrol[wander_idx_arr]
        wand_move_int     = params.animal_move_int[wander_idx_arr]
        # _wander_step is the verbatim body of update_neutral_animals, lifted to
        # take the sliced arrays. Its internal jax.random.randint uses shape
        # (N_neutral,) — byte-identical to today.
        new_wand_pos, new_wand_mt = _wander_step(
            wand_pos_in, wand_mt_in, wand_patrol, wand_move_int,
            obs_pos, obs_blocking, wander_key)
    else:
        new_wand_pos = new_wand_mt = None

    # ── Branch C: STATIC — pass-through, no draws ──
    # (No randoms consumed, so no parity concern.)

    # ── Scatter back into unified arrays via .at[idx].set(...) ──
    new_pos     = animal_pos
    new_state   = animal_state
    new_stamina = animal_stamina
    new_mt      = animal_move_timer
    new_at      = animal_attack_timer

    if len(params.hunt_idx) > 0:
        new_pos     = new_pos.at[hunt_idx_arr].set(new_hunt_pos)
        new_state   = new_state.at[hunt_idx_arr].set(new_hunt_state)
        new_stamina = new_stamina.at[hunt_idx_arr].set(new_hunt_stam)
        new_mt      = new_mt.at[hunt_idx_arr].set(new_hunt_mt)
        new_at      = new_at.at[hunt_idx_arr].set(new_hunt_at)
    if len(params.wander_idx) > 0:
        new_pos = new_pos.at[wander_idx_arr].set(new_wand_pos)
        new_mt  = new_mt.at[wander_idx_arr].set(new_wand_mt)
    # static: no scatter (positions and timers stay as-is)

    return new_pos, new_state, new_stamina, new_mt, new_at
```

**Draw-shape preservation guarantee.** The hunt path consumes a `hunt_key` of identical draw-shape `(N_pred,)` to today's `predator_key` consumption inside `update_predators`. The wander path consumes a `wander_key` of identical draw-shape `(N_neutral,)` to today's `neutral_key` consumption inside `update_neutral_animals`. Both behaviours' internal `jax.random.{randint, uniform}` calls produce byte-identical bytes to today, because JAX threefry produces deterministic streams keyed by `(key, shape, dtype)`.

**`_hunt_step` and `_wander_step` rationale.** `_hunt_step` is a near-verbatim transcription of today's `update_predators` (lines 132–251), with the only changes being that `params.pred_*` reads become passed-in arguments (e.g., `params.pred_detect → hunt_detect` argument) — every `jax.random.*` shape, every order of operations, and every operator stays identical. `_wander_step` is a verbatim transcription of today's `update_neutral_animals` (lines 253–286), with the same input-renaming pattern.

**Static-vs-traced split.** `params.hunt_idx` / `params.wander_idx` / `params.static_idx` are `pytree_node=False` Python int tuples. Slicing `animal_pos[jnp.asarray(hunt_idx)]` is JIT-static (the slice indices are constants under the trace). The if-branches on `len(params.hunt_idx) > 0` are Python-level (host-side), so a config with zero predators or zero neutrals compiles the corresponding branch out entirely — no wasted FLOPs, no parity perturbation.

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

`sense_extero_nociception` block (`sensor.py:59–87`):

```
# OLD:
dist_pred = jnp.linalg.norm(state.pred_pos - agent_pos, axis=-1)
nociception = jnp.sum(jnp.where(dist_pred <= params.extero_noc_radius,
                                params.pred_nociception, 0.0))

# NEW (only damaging classes contribute, preserving today's "predators only" behaviour):
dist_animal = jnp.linalg.norm(state.animal_pos - agent_pos, axis=-1)
contributes = (dist_animal <= params.extero_noc_radius) & params.animal_is_damaging
nociception = jnp.sum(jnp.where(contributes, params.animal_nociception, 0.0))
```

Parity check: `params.animal_is_damaging` is True only where `animal_classes == 'predator'`, so the sum is identical to today's `pred_nociception`-only contribution. Rabbits and other neutrals never appear in the extero-noc sum.

This was a blocker raised by `code-reviewer` (B2). The CP4 parity gate explicitly covers extero-nociception in addition to visual and olfactory — see Test Plan §(d).

#### Damage logic (`core.py:395–404`) and the `hit_neutral` pre-step asymmetry

```
# OLD:
at_predator = jnp.all(new_pred_pos == new_agent_pos, axis=-1)             # POST-step
sampled_pred_damage = jax.random.uniform(damage_key, (params.pred_damage.shape[0],),
    minval=params.pred_damage[:, 0], maxval=params.pred_damage[:, 1])
damage_pred = jnp.sum(jnp.where(at_predator, sampled_pred_damage, 0.0))
new_pred_attack_timer = jnp.where(at_predator, params.pred_attack_delay, new_pred_attack_timer)
# Today, core.py:441:
info['hit_neutral'] = jnp.any(jnp.all(state.neutral_pos == new_agent_pos, axis=-1))  # PRE-step

# NEW (preserves the pre-existing pre/post asymmetry exactly):
at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)              # POST-step (predator-side)
at_damaging = jnp.logical_and(at_animal, params.animal_is_damaging)
sampled_animal_damage = jax.random.uniform(damage_key, (params.animal_damage.shape[0],),
    minval=params.animal_damage[:, 0], maxval=params.animal_damage[:, 1])
damage_pred = jnp.sum(jnp.where(at_damaging, sampled_animal_damage, 0.0))
new_animal_attack_timer = jnp.where(at_damaging, params.animal_attack_delay, new_animal_attack_timer)

# info['hit_predator'] uses the POST-step `at_damaging` (unchanged semantics):
info['hit_predator'] = jnp.any(at_damaging)

# info['hit_neutral'] MUST use the PRE-step animal positions to preserve today's behaviour.
# This is a pre-existing quirk in core.py:441 — agent moving onto a rabbit's OLD square
# (before the rabbit's move) registers hit_neutral=True. The refactor preserves this exactly
# by reading `state.animal_pos` (pre-update) rather than `new_animal_pos` (post-update):
at_neutral_pre = jnp.all(state.animal_pos == new_agent_pos, axis=-1) & ~params.animal_is_damaging
info['hit_neutral'] = jnp.any(at_neutral_pre)
```

**Pre/post asymmetry preservation.** This was a blocker raised by `code-reviewer` (B5). Today's `hit_predator` uses `new_pred_pos` (post-predator-move) while `hit_neutral` uses `state.neutral_pos` (pre-neutral-move). The original plan unified both onto `new_animal_pos` (post-move), which is a behaviour change. The revised plan computes two separate masks:

- `at_animal` from `new_animal_pos` → feeds `hit_predator`, damage accumulation, and `new_animal_attack_timer`.
- `at_neutral_pre` from `state.animal_pos` masked by `~animal_is_damaging` → feeds `hit_neutral` only.

This is byte-equivalent to today's logic for the parity-reference config and every other legacy config.

Parity check: with the parity-reference config, `animal_is_damaging` is `[1, 0, 0]` (1 predator + 2 rabbits), `at_damaging` picks out exactly the predator hits identical to today, and `at_neutral_pre` picks out exactly the rabbit-hit events identical to today's `state.neutral_pos == new_agent_pos`.

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
# Static per-subset index tuples (B1 fix — used by update_animals to slice
# hunt / wander / static subsets while preserving today's PRNG draw shapes).
hunt_idx: tuple = struct.field(pytree_node=False)            # tuple[int, ...], len N_pred
wander_idx: tuple = struct.field(pytree_node=False)          # tuple[int, ...], len N_neutral
static_idx: tuple = struct.field(pytree_node=False)          # tuple[int, ...], len N_static
```

Plus the `@property` legacy accessors `predator_tags` / `neutral_tags` defined on the `EnvParams` class body (see "Legacy aliases" earlier in this section, B3 fix).

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
3. For each entry, for each field in `DISTRIBUTIONAL_FIELDS`:
   - **If `behaviour == 'hunt'`**: field is **mandatory**. Read scalar or `[low, high]`; if scalar `s`, treat as `[s, s]`. Missing → raise `ValueError` (no fallback default, per project rule).
   - **If `behaviour ∈ {'wander', 'static'}`**: field is **optional**. Read scalar or `[low, high]` if present; if scalar `s`, treat as `[s, s]`. If missing, auto-fill `[0, 0]` and log at debug level. This is an internal projection detail (the wander / static code path never reads these arrays), not a fallback default on a user-facing key — see "Mandatory-key rule by behaviour" above (B-CFG-1).
   - **Legacy `predators:` entries** behave as `behaviour='hunt'` (mandatory rule). **Legacy `neutral_animals:` entries** behave as `behaviour='wander'` (optional / auto-fill `[0, 0]`). This preserves today's behaviour: legacy neutrals don't carry these fields and the auto-fill is purely an internal re-projection.
4. Build all `animal_*` jnp arrays at the bottom of the function and return them.

(The loader no longer reads `predator_enabled` — that flag is fully removed in this CP; the 86 affected configs are migrated atomically with the schema change.)

**Note on `lose_interest_multiplier` mandatory promotion** (C-CFG-config auditor concern). The current loader at `config_loader.py:263` uses `p.get('lose_interest_multiplier', 2.0)` — a soft default of `2.0`. The refactor promotes the field to mandatory for `behaviour: hunt`. All 86 existing predator entries carry an explicit value, so no real-world config breaks; flag the change in the loader docstring as a v1.x → v2.0 semantic-change note.

`get_mandatory` semantics: `environment.predators` and `environment.neutral_animals` are kept as `get_mandatory` reads (legacy invariant — every existing config has them, even if empty list). When `entities:` is present, both legacy keys become optional and the loader allows them to be missing or `None`. Document this in the loader docstring.

**YAML `[lo, hi]` cadence-distinction convention (C-CFG-6).** Two YAML fields use identical `[lo, hi]` syntax but have different sampling cadences:

- `damage: [lo, hi]` → re-sampled **per event** (every collision) — pre-existing.
- `detection_range: [lo, hi]` (and the four sibling distributional fields) → re-sampled **per episode** (at reset) — NEW.

The cadence is determined by field name (membership in `DISTRIBUTIONAL_FIELDS`), not by syntax. To prevent confusion for future YAML authors, the loader's `_load_animals()` docstring lists which fields use which cadence, and the canonical examples under `configs/experiment/v2_smoke/` include inline comments:

```yaml
  damage: [15.0, 45.0]          # per-event uniform sample
  detection_range: [0, 5]       # per-episode uniform sample (one draw per reset)
```

#### `src/environment/core.py`

**Remove** `update_predators` (lines 132–251) and `update_neutral_animals` (lines 253–286).

**Add** `update_animals(state, agent_pos, params, key)` per the pseudocode above. Internals:

- `_hunt_step` is a near-verbatim transcription of today's `update_predators`, with `params.pred_detect → state.animal_detect_sampled`, `params.pred_max_stamina → state.animal_max_stamina_sampled`, `params.pred_recovery → state.animal_recovery_sampled`, `params.pred_hunt_thresh → state.animal_hunt_thresh_sampled`, `params.pred_lose_interest_mult → state.animal_lose_interest_sampled`. All other inputs stay as `params.*` (`pred_patrol → animal_patrol`, `pred_move_int → animal_move_int`, `pred_attack_delay → animal_attack_delay`).
- `_wander_step` is a verbatim transcription of `update_neutral_animals` with `neutral_patrol → animal_patrol`, `neutral_move_int → animal_move_int`.

**Patch** `jax_step` (lines 288–555):

- Line 293: keep the 6-way key split (rename: `key, respawn_key, hunt_key, wander_key, damage_key, property_key`). Pass `hunt_key` and `wander_key` separately to `update_animals` — each consumed by its own per-subset call, with draw-shape `(N_pred,)` / `(N_neutral,)` byte-identical to today's `update_predators(predator_key)` / `update_neutral_animals(neutral_key)`. See §"`update_animals()` dispatch" for the per-subset call pattern (B1 fix).
- Lines 321–330: replace the two separate update calls with one `update_animals` call (the per-subset slicing + scatter happens inside).
- Lines 395–404: change damage logic per §"Damage logic" above (use `at_damaging` for damage + `hit_predator`).
- Lines 440–443: `info['hit_neutral']` uses **pre-step** `state.animal_pos` masked by `~animal_is_damaging` (B5 fix), not post-step `new_animal_pos`. See §"Damage logic" for the exact diff.
- Lines 495–515: `dist_to_pred` and `dist_to_neutral` derived from `state.animal_pos` masked by `params.animal_is_damaging` (predator distances) and `~animal_is_damaging` (neutral distances). `dist_per_predator` and `dist_per_neutral` similarly — kept as legacy info-dict aliases (C5). `dist_per_animal` is the unmasked vector.
- Lines 528–553: update `state._replace(...)` to use the new `animal_*` field names.

**Patch** `jax_reset` (lines 657–817):

- Lines 666: extend the 5-way key split to 6-way (`key, agent_key, placement_key, body_key, property_key, animal_episode_key`).
- Lines 673–676: replace the `num_pred`, `num_neutral` derivation with `num_animals = params.animal_property.shape[0]`.
- Lines 678–751 (placement modes): the per-entity and per-type placement scans currently treat predator and neutral as separate spawn-area groups. Re-project: in `per_entity` mode, concatenate `res / animal / obs` spawn areas (was `res / pred / obs / neutral`). In `per_type` mode, the YAML's `placement.types:` definition is loaded by the existing per-type logic — verify the config-loader produces the right `type_entity_map` indices when animals are unified. **Detailed check needed in CP3**: per-type placement reads `type_entity_map` indices, which today partition the flat `[res, pred, obs, neutral]` index space; the loader's `type_entity_map` builder needs updating to use `[res, animal, obs]` ordering. The reference config uses `per_entity` mode, so the per-type path is exercised only by configs that opt into it — list them in CP3 and verify all 74 parity tests.
- After placement, add the per-episode uniform sampling block (§"Per-episode sampling inside `jax_reset`" above).
- Lines 776–784 (property sampling): collapse `pred_property_sampled` + `neutral_property_sampled` into one `animal_property_sampled`. Use a single `prop_key_animal`.
- Lines 786–815 (state assembly): replace `pred_*` and `neutral_*` initialisers with `animal_*` initialisers (per the `EnvState` schema above). `animal_state = jnp.zeros(N, dtype=jnp.int32)` (PATROL=0; semantically inert for wander/static).

#### `src/environment/sensor.py`

- Lines 59–87 (`sense_extero_nociception`): replace `state.pred_pos` / `params.pred_nociception` reads with `state.animal_pos` / `params.animal_nociception`, masked by `params.animal_is_damaging` so only predators contribute (preserves today's behaviour exactly). See §"Sensor refactor" above for exact diff. **(B2 fix: this was missing from the original plan.)**
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

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` (lines 522–523) and `src/models/dreamer_v3_trainer.py`, `src/models/recurrent_ppo_trainer.py`

**B3 correction (raised by code-reviewer):** the original plan claimed no trainer reads `predator_tags` / `neutral_tags`. Wrong. `dreamer_srl_main.py:522-523` reads `env_params.predator_tags` and `env_params.neutral_tags` directly.

**Fix (minimum-diff):** the legacy aliases `predator_tags` and `neutral_tags` are kept on `EnvParams` as `@property` accessors derived from `animal_tags` + `animal_classes` (see §"`EnvState` / `EnvParams` field layout → Legacy aliases"). With the aliases in place, `dreamer_srl_main.py:522-523` requires **no edit** — it transparently reads through the property. The `@property` returns a host-side Python tuple (not a jnp array), matching today's type exactly.

`dreamer_v3_trainer.py` and `recurrent_ppo_trainer.py` use these tags via the same `accumulators.py` plumbing — no direct `state.pred_*` reads. **CP4 verification**: grep `src/algorithms/` and `src/models/` for `pred_` and `neutral_` to confirm no surviving direct reads, document the grep result in the CP4 implementation report.

#### `scripts/verification/check_olfaction_parity.py`

This existing parity script does byte-level comparison of observations. It will be reused as part of the parity-gate for CP1 — no edits required, but the developer should run it against the parity-reference config before and after the refactor and confirm zero diff.

#### `scripts/verify_noise.py` (lines 37–43) — REWRITE

`scripts/verify_noise.py:37-43` today manually constructs an `EnvState` with the now-removed fields `pred_pos`, `pred_state`, `pred_stamina`, `pred_move_timer`, `pred_attack_timer`, `neutral_pos`, `neutral_move_timer`. After CP1 these field names don't exist; the script would `TypeError`. **B4 fix (raised by code-reviewer):** rewrite the constructor block to use `animal_*` fields:

```python
# OLD:
state = EnvState(
    ...,
    pred_pos=..., pred_state=..., pred_stamina=...,
    pred_move_timer=..., pred_attack_timer=...,
    neutral_pos=..., neutral_move_timer=...,
)

# NEW (N_animals = N_pred + N_neutral, predators-first ordering):
state = EnvState(
    ...,
    animal_pos=animal_pos,                       # [N, 2]
    animal_state=jnp.zeros(N, dtype=jnp.int32),
    animal_stamina=animal_max_stamina_sampled,   # init = max
    animal_move_timer=jnp.zeros(N, dtype=jnp.int32),
    animal_attack_timer=jnp.zeros(N, dtype=jnp.int32),
    animal_property_sampled=animal_property,
    animal_detect_sampled=...,                   # per-episode sampled
    animal_max_stamina_sampled=...,
    animal_recovery_sampled=...,
    animal_hunt_thresh_sampled=...,
    animal_lose_interest_sampled=...,
)
```

The developer should land this rewrite atomically with CP1 (alongside the loader change and the migration sweep) so the script keeps working through the refactor. The script is then re-run end-to-end as part of CP1's parity gate to confirm zero diff.

#### `scripts/benchmark_render.py`

Same as the renderers — replace `state.pred_pos` / `state.neutral_pos` reads with `select_by_class` slices.

### Checkpoints

The refactor lands in six checkpoints. CP1 alone is the **minimum viable landing** (schema + loader + backward-compat shim + parity tests). CP2–CP5 add the per-episode sampling layer. CP6 is the analysis-side cleanup; until CP6, the legacy `dist_per_predator` / `dist_per_neutral` / `pred_*` arrays survive as compatibility aliases.

- [ ] **CP1 — Unified state + loader + backward-compat parity + `predator_enabled` migration sweep.** Implement `EnvState` / `EnvParams` field changes (including full removal of `predator_enabled` and the per-subset hunt_idx / wander_idx / static_idx static tuples), `update_animals` (per-subset call pattern — see §"`update_animals()` dispatch" — with all distributional sampling no-ops: bounds set to `[scalar, scalar]` from legacy YAML), `config_loader._load_animals`, the legacy `@property predator_tags` / `neutral_tags` on `EnvParams`, the `sense_extero_nociception` refactor (B2), and the `scripts/verify_noise.py` `EnvState` constructor rewrite (B4). Pre-step `hit_neutral` asymmetry preserved (B5). **Atomic commit boundary (B-CFG-config-auditor C-CFG-4)**: the loader change, the `verify_noise.py` rewrite, and the 86-config migration sweep all land in a single atomic commit so no intermediate state has 86 configs failing to load. **Atomically migrate all 86 configs that reference `predator_enabled`**: (a) sed-strip the 85 `predator_enabled: true` lines (no behavioural change); (b) hand-migrate `configs/verification/olfaction_parity_neutral.yaml` — strip the single line `predator_enabled: false` (L10); `predators: []` is already present on L11. **Pre-flight (C2)**: run `grep -rln "mode: per_type" configs/` to enumerate per-type-placement configs before touching loader code — surface them in the implementation report so the developer can verify the `type_entity_map` re-projection on each one. **Verifies:** `pytest tests/env/test_unified_parity.py` (new) passes — for each of all **86** migrated configs (see Test Plan §(a) for the full glob), running 100 steps from seed 0 produces byte-identical `obs` vectors before and after the refactor, AND the parity assertion includes `state.animal_pos[wander_idx]` (catches B1 regressions) and `info['hit_neutral']` / `info['hit_predator']` (catches B5 regressions). The existing parity scripts `scripts/verification/check_olfaction_parity.py` and `scripts/verify_noise.py` (post-rewrite) also exit 0. **No new YAML fields are read in this CP** — the per-episode bounds come from re-projecting the legacy scalar to `[scalar, scalar]`.
- [ ] **CP2 — Per-episode sampling at reset.** Add the 5-way key split in `jax_reset` and the `jax.random.uniform` calls; populate the 5 `animal_*_sampled` fields. **Verifies:** with all legacy configs (degenerate ranges) the parity gate still passes byte-for-byte (uniform`[s, s]` ≡ `s`). A new test `tests/env/test_per_episode_sampling.py` verifies, **using a config explicitly with non-degenerate ranges** (e.g., `detection_range: [0, 5]`, 4 predators — degenerate ranges would fail the divergence check spuriously, C4): (a) same `key` ⇒ same sampled values; (b) different `key` ⇒ different sampled values; (c) N entities of the same class give N independent samples (not N copies); (d) **cross-field independence**: `detect`, `max_stamina`, `recovery`, `hunt_thresh`, `lose_interest` samples are statistically independent across episodes (correlation `< 0.5` over 100 keys per field pair — guards against accidentally reusing one subkey for multiple fields); (e) **per-episode-sampled fields live on `EnvState`, not `EnvParams`** — assert `hasattr(state, 'animal_detect_sampled')` and `not hasattr(params, 'animal_detect_sampled')` (code-reviewer Missed-failure-mode #1); (f) **wander/static get unused per-episode draws** — assert `state.animal_state[wander_idx]` and `state.animal_state[static_idx]` stay at 0 after 1000 steps (intentional shape uniformity, code-reviewer Missed-failure-mode #3 / #4). 
- [ ] **CP3 — `entities:` schema in config loader.** Add the new `environment.entities:` YAML path; loader prefers it over legacy if both present (with warning). Update `placement.types:` handling: ensure `type_entity_map` uses `[res, animal, obs]` indexing in `per_type` mode. **Pre-flight (C2)**: enumerate `per_type`-mode configs via `grep -l "mode: per_type" configs/`; for each one, verify the loader produces the same `type_entity_map` indices after unification (predators-first, then neutrals, preserves today's `[res, pred, obs, neutral]` index space mapped to `[res, animal, obs]`). Surface the list of `per_type` configs in the CP3 implementation report. **Verifies:** a new `configs/experiment/v2_smoke/01-entities-smoke.yaml` config that uses the unified schema for the same parity-reference setup produces byte-identical behaviour to `01-interoNocicept_sameProp.yaml`. New test `tests/env/test_entities_schema.py` covers: (a) the unified config loads; (b) byte-parity vs legacy; (c) loader warns when both legacy and unified sections are present; (d) configs that still contain the removed `predator_enabled` key raise `ValueError` with a clear migration message (sanity check; the CP1 sweep should have removed all of them already); (e) a 4-entity config with `behaviour: [hunt, wander, hunt, static]` loads correctly and the resulting `hunt_idx == (0, 2)`, `wander_idx == (1,)`, `static_idx == (3,)` (behaviour-mask axis test from code-reviewer Missed-failure-mode #2).
- [ ] **CP4 — Sensor + damage + step path parity.** Patch `sense_visual` (class scatter), `sense_extero_nociception` (animal_is_damaging mask — B2), `get_observation` olfactory (unified `animal_chem`), and `jax_step` damage logic (`animal_is_damaging` + the pre-step `hit_neutral` asymmetry — B5). **Verifies:** the CP1 parity test (all **86** migrated configs × 100 steps) still passes — this CP shouldn't *add* parity coverage but must *not break* it. Additionally, a new `tests/env/test_visual_parity.py` runs one episode (1000 steps) on the parity-reference config and asserts the visual one-hot per cell is byte-identical to a pinned reference dump (committed under `tests/env/fixtures/visual_parity_ref.npz`). **Extero-noc parity gate (B2)**: the same fixture also pins the extero-noc channel per step, byte-identical to today. **Trainer verification**: grep `src/algorithms/` and `src/models/` for `state.pred_` / `state.neutral_` / `pred_pos` / `neutral_pos`; document the grep result. With the `predator_tags` / `neutral_tags` legacy aliases in place (B3), `dreamer_srl_main.py:522-523` requires no edit and the test passes.
- [ ] **CP5 — Distributional schema + per-episode logging.** Add YAML parsing for `[low, high]` ranges on the five fields under scope; emit `Episode/sampled_*_<tag>` WandB metrics. **Verifies:** new `tests/env/test_distributional_yaml.py` covers (a) scalar `5` → degenerate range `[5, 5]`, (b) `[0, 5]` → bounds stored correctly, (c) malformed range like `[5]` raises `ValueError`. End-to-end smoke: run 1000 training steps with `configs/experiment/v2_smoke/02-entities-distributional.yaml` (NEW — has `detection_range: [0, 5]` per predator) and confirm the 5 `sampled_*` WandB metrics appear with non-degenerate values. **JIT-recompile check (C3)**: capture `jax_log_compiles` output via `jax.config.update("jax_log_compiles", True)`; build env-step jit for config A (`1 pred + 2 neutral`, `detection_range: [0, 5]`), run 10 steps; build for config B (same counts, `detection_range: [2, 7]`), run 10 steps; assert `log.count("Compiling jax_step") == 1` (regex check). **Positive control (C-CFG-5)**: swap `animal_classes` ordering — same N, different per-entity class ordering like `[pred, neutral, pred]` vs `[pred, pred, neutral]` — and assert `log.count("Compiling jax_step") == 2` (recompile IS expected because `animal_classes` is `pytree_node=False` and the hunt_idx / wander_idx tuples differ). Documents the boundary: same N + same class ordering = no recompile; same N + different class ordering = recompile.
- [ ] **CP6 — Analysis-side cleanup.** Update `accumulators.py`, `distance_aggregator.py`, `eval_rollout.py`, `motif_cluster.py`, `evaluation_core.py`, `eval_recording.py`, `grid_world.py`, `renderer.py`, `renderer_v2.py`, `benchmark_render.py` to consume `dist_per_animal` + `select_by_class` directly. **Verifies:** `pytest tests/behavior/test_accumulators.py` passes with the same metric layout as before; the WandB metric key names are unchanged (`MeanDistPredator_<tag>`, `MeanDistNeutral_<tag>` still produced). Legacy aliases `dist_per_predator` / `dist_per_neutral` removed from `info` (one release-cycle later, not in this CP).

### Test Plan

The test plan operationalises the locked parity gate. The full test surface:

#### (a) Per-step parity test — `tests/env/test_unified_parity.py` (NEW, CP1)

**Glob covers all 86 migrated configs**, not just the 74 under `configs/experiment/`. This was HC-1 raised by both reviewers (also C1 in the code review and "HARD CONCERN" in the config audit). The original glob `glob('configs/experiment/**/*.yaml')` without `recursive=True` would miss both:

- `configs/experiment/2X2_area.yaml` (depth 0 — not matched by the `**/` glob without `recursive=True`)
- 12 of 86 migrated configs outside `configs/experiment/`

Use:

```python
import glob
configs = sorted(
    glob.glob('configs/experiment/**/*.yaml', recursive=True) +
    glob.glob('configs/continual/**/*.yaml', recursive=True) +
    glob.glob('configs/verification/**/*.yaml', recursive=True) +
    ['configs/environment/default.yaml']
)
```

For each `config_path in configs`:
1. Load with the **old** loader path (pre-refactor commit — pulled via git via a one-shot script that builds a comparison snapshot, OR equivalently against a hard-pinned fixture per config).
2. Load with the **new** loader path.
3. Run 100 steps from seed 0 (using `jax_reset(params, jax.random.PRNGKey(0))` then 100 `jax_step` calls with actions `[0, 1, 2, 3, 4] * 20`).
4. Assert `jnp.array_equal(old_obs[t], new_obs[t])` for every step `t`.
5. Assert `jnp.array_equal(old_state.<field>, new_state.<field>)` for every common field (post-renaming: predator/neutral fields → matching `animal_*` slices, including `state.animal_pos[wander_idx]` to catch B1 regressions).
6. Assert `info['hit_neutral']` and `info['hit_predator']` byte-equal old (catches B5 regressions).

**Failure mode:** any single step diff blocks CP1.

**Coverage breakdown** (86 = 74 experiment + 5 continual + 6 verification + 1 environment/default):

| Directory | Count | Included via |
|---|---:|---|
| `configs/experiment/**/*.yaml` | 74 | `experiment/**/*.yaml` glob (recursive=True catches `2X2_area.yaml` at depth 0) |
| `configs/continual/nmn_double_return_stages/*.yaml` | 5 | `continual/**/*.yaml` glob |
| `configs/verification/*.yaml` | 6 | `verification/**/*.yaml` glob (includes 4 observability gates + 2 olfaction parity) |
| `configs/environment/default.yaml` | 1 | explicit |
| **Total** | **86** | |

#### (b) Per-episode sampling test — `tests/env/test_per_episode_sampling.py` (NEW, CP2)

Use a config where `detection_range: [0, 5]`, `max_stamina: [20, 40]`, etc. with **non-degenerate ranges** for 4 predators (C4 from code-reviewer — degenerate ranges would fail the divergence check spuriously):

1. Reset with key `K`, capture `state.animal_detect_sampled` → call this `s_K`.
2. Reset again with key `K`, capture sampled → assert `s_K == s'` (reproducibility).
3. Reset with key `K + 1`, capture → assert `s != s_K` (divergence; with high probability — N=4 gives effectively no false positives at sigma=2.89 uniform).
4. Assert the 4 values inside `s_K` are not all-identical (per-instance independence).
5. **Cross-field independence (C4)**: collect samples from 100 distinct keys for each of the 5 sampled fields (`detect`, `max_stamina`, `recovery`, `hunt_thresh`, `lose_interest`); assert pairwise Pearson correlation `|r| < 0.5` for every pair. Guards against accidentally reusing one subkey across fields.
6. **State-vs-params placement (code-reviewer Missed-failure-mode #1)**: assert `hasattr(state, 'animal_detect_sampled')` AND `not hasattr(params, 'animal_detect_sampled')`. Per-episode samples live on `EnvState`, never on `EnvParams`.
7. **Wander/static get unused draws (code-reviewer Missed-failure-mode #3, #4)**: use a 4-entity config with `behaviour: [hunt, wander, hunt, static]`; reset with key `K`, run 1000 `jax_step` calls; assert `state.animal_state[wander_idx]` and `state.animal_state[static_idx]` stay at 0 throughout. Intentional shape uniformity — wander/static entities carry the five `*_sampled` fields but the code paths never read them.

#### (c) Backward-compat test — `tests/env/test_backward_compat_configs.py` (NEW, CP1)

Loop over all `configs/experiment/**/*.yaml`, call `load_env_params(Config.from_file(path))`. Assert no exception. Assert `params.animal_property.shape[0] == num_predators_in_yaml + num_neutrals_in_yaml` (where the YAML's `count:` fields are summed by class). This catches any config that the new loader can't handle.

#### (d) Visual parity test — `tests/env/test_visual_parity.py` (NEW, CP4)

For the parity-reference config (`01-interoNocicept_sameProp.yaml`):
1. Run one episode of 1000 steps from seed 0 with actions `[0, 1, 2, 3, 4] * 200`.
2. Save the visual-sensor slice of each `obs` to `tests/env/fixtures/visual_parity_ref.npz` (committed alongside the test).
3. The test re-runs and asserts byte-equality with the fixture.

This guards against future regressions and locks the visual-channel-class-coding behaviour the user explicitly wants preserved.

#### (e) JIT-recompile test — `tests/env/test_no_recompile.py` (NEW, CP5)

Two parts:

**Part 1 — negative control (no recompile expected, C3).** Two configs with the same animal counts (`1 predator + 2 rabbits`) AND the same per-entity class ordering, but different `detection_range` bounds (`[0, 5]` vs `[2, 7]`). Use `jax.config.update("jax_log_compiles", True)` capture into a `logging.StreamHandler` → string buffer; build env-step jit for config A, run 10 steps; build for config B, run 10 steps. Assert `log.count("Compiling jax_step") == 1` (regex check, exactly one compile).

**Part 2 — positive control (recompile EXPECTED, C-CFG-5).** Two configs with same N=3 but different class ordering: A has `[pred, neutral, pred]`, B has `[pred, pred, neutral]`. Because `animal_classes` is `pytree_node=False`, the host-side `hunt_idx` / `wander_idx` tuples differ — JAX must re-trace. Assert `log.count("Compiling jax_step") == 2` (one compile per config). This documents the boundary: same N + same class ordering = no recompile (Part 1); same N + different class ordering = recompile (Part 2).

#### (f) Distributional schema test — `tests/env/test_distributional_yaml.py` (NEW, CP5)

YAML fixtures under `tests/env/fixtures/`:
- `dist_scalar.yaml` — `detection_range: 5` → bounds `[5, 5]`
- `dist_range.yaml` — `detection_range: [0, 5]` → bounds `[0, 5]`
- `dist_malformed_single.yaml` — `detection_range: [5]` → raise `ValueError`
- `dist_malformed_str.yaml` — `detection_range: "five"` → raise `ValueError`

#### (h) Info-dict legacy-alias parity — `tests/env/test_info_dict_aliases.py` (NEW, CP1)

C5 from code-reviewer: the original plan addressed `hit_neutral` (B5) but did not enumerate the full set of legacy info-dict keys. Sweep `grep -n "info\[" src/environment/core.py` and confirm every legacy info-dict key is preserved name-for-name on the new step output. Required key list (minimum — extend if grep finds more):

- `info['hit_predator']` — from `at_damaging` (POST-step), unchanged semantics.
- `info['hit_neutral']` — from `at_neutral_pre` (PRE-step, B5 fix), preserves today's asymmetry.
- `info['damage_predator']` — from `jnp.sum(jnp.where(at_damaging, sampled_animal_damage, 0.0))`. Same key, same scalar semantics.
- `info['damage_obstacle']` — unchanged (no animal-side change).
- `info['damage_hiding_predator']` — preserved if today's code emits it; verify via grep.
- `info['dist_per_predator']` / `info['dist_per_neutral']` — derived from `info['dist_per_animal']` masked by `animal_is_damaging` / `~animal_is_damaging`. Legacy aliases kept for one release cycle.

Test loads the parity-reference config, runs 100 steps from seed 0, and asserts each legacy key (a) is present in `info` and (b) has the same value as the pre-refactor reference dump. Failure of any key blocks CP1.

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
3. **Per-type placement mode (`placement: mode: per_type`).** The plan describes the index re-mapping but does not enumerate which configs opt into it. **Updated in v0.2 (C2 from code-reviewer):** CP1 pre-flight runs `grep -rln "mode: per_type" configs/` and surfaces the list in the implementation report; CP3 sweeps all 86 migrated configs (not just 74); if any use `per_type` with non-trivial type-group structure that overlaps the predator/neutral split, the developer must verify the `type_entity_map` re-projection in CP3.
4. **WandB metric-key surface.** Plan preserves `MeanDistPredator_<tag>` and `MeanDistNeutral_<tag>` exactly. The new `sampled_*_<tag>` keys are additive. If the user wants a wholesale rename (e.g., to `MeanDist_<class>_<tag>`), that's a separate logging refactor — flag as a follow-up.
5. **`entities:` schema doc location.** Not part of this plan, but once CP3 lands, the YAML schema reference (likely `configs/CLAUDE.md` if it exists, or a new `configs/SCHEMA.md`) should document the unified form. Open question: where does this reference live today? Plan defers documenting until the developer surfaces the existing schema-doc home.

---

## Revision log

### v0.2 — 2026-05-28 (post-review revision)

Plan revised by `senior-developer` to address every blocker raised by the two pre-implementation reviews. Original commit `7110323`; this revision builds on that without restructuring CP1–CP6 (per the user's explicit constraint).

**Reviewer reports addressed:**

- [code review](../../../reviews/env_entities_plan_review_code.md) — verdict **REJECT** → now resolved (B1–B5).
- [config audit](../../../reviews/env_entities_plan_audit_config.md) — verdict **ACCEPT-WITH-REVISIONS** → now resolved (B-CFG-1, B-CFG-2, HC-1).

**Blockers addressed**

| ID | Source | What changed | Where |
|---|---|---|---|
| B1 | code-reviewer | Replaced the masked-combine `update_animals` design with a per-subset call pattern that statically slices the hunt subset (`(N_pred,)` draws) and wander subset (`(N_neutral,)` draws), then scatters results back into the unified `animal_*` arrays via `.at[idx].set(...)`. Draw shapes byte-identical to today, so threefry PRNG bytes match exactly. Added new static `hunt_idx` / `wander_idx` / `static_idx` `pytree_node=False` index tuples to `EnvParams`. Updated Risks item 1 to remove the obsolete masked-combine rationale. | §"`update_animals()` dispatch", EnvParams table, EnvParams Add block, Risks item 1, CP1 description |
| B2 | code-reviewer | Added `sense_extero_nociception` (lines 59–87 of `sensor.py`) to the File Changes section, with the exact diff: `state.pred_pos` / `params.pred_nociception` reads become `state.animal_pos` / `params.animal_nociception`, masked by `params.animal_is_damaging` so only predators contribute. CP4 parity gate now explicitly covers extero-noc. | §"Sensor refactor", File Changes → `sensor.py`, CP4 description |
| B3 | code-reviewer | Kept `predator_tags` and `neutral_tags` on `EnvParams` as `@property` accessors derived from `animal_tags` + `animal_classes`. `dreamer_srl_main.py:522-523` requires no edit; the property returns a host-side Python tuple matching today's type. Added a dedicated "Legacy aliases" subsection. Updated the trainer-files File Changes entry to correct the original plan's "no trainer reads these" claim. | §"`EnvState` / `EnvParams` field layout → Legacy aliases", EnvParams table rows, File Changes → `dreamer_srl_main.py` |
| B4 | code-reviewer | Added `scripts/verify_noise.py` (lines 37–43) to File Changes with full rewrite of the `EnvState` constructor block using `animal_*` fields. Atomic-commit boundary now explicitly includes the script rewrite alongside the loader change. | File Changes → `scripts/verify_noise.py`, CP1 description |
| B5 | code-reviewer | Preserved the pre-existing PRE-step / POST-step asymmetry between `info['hit_neutral']` (today reads `state.neutral_pos`, pre-rabbit-move) and `info['hit_predator']` (today reads `new_pred_pos`, post-predator-move). New plan computes two separate masks: `at_animal` (from `new_animal_pos`, feeds damage + `hit_predator` + `attack_timer`) and `at_neutral_pre` (from `state.animal_pos`, feeds `hit_neutral` only). | §"Damage logic", File Changes → `jax_step` patch |
| B-CFG-1 | env-config-auditor | Reconciled the apparent contradiction between "Missing → raise `ValueError`" (loader code) and "loader fills these with zeros" (worked example) via a behaviour-conditional rule: distributional fields are mandatory for `behaviour: hunt` and optional (auto-fill `[0, 0]`) for `behaviour: wander` / `static`; legacy `neutral_animals:` entries re-project to `behaviour: wander` with auto-fill, an internal projection detail not a user-facing fallback default. Added explicit "Mandatory-key rule by behaviour" table. Reconciled the worked-example comment and the loader numbered steps. | §"Unified config schema", worked-example YAML comment, loader numbered steps |
| B-CFG-2 | env-config-auditor | Added `attack_delay` to the schema description text in the opening paragraph of §"Unified config schema". | §"Unified config schema" |
| HC-1 / C1 | both | Widened the CP1 parity-test glob to cover all 86 migrated configs (74 experiment + 5 continual + 6 verification + 1 environment/default) with `recursive=True` explicit, using the four-glob pattern the reviewer suggested. Added a coverage breakdown table. CP1, CP3, CP4 descriptions now say "86" not "74". | Test Plan §(a), CP1, CP3, CP4 descriptions |

**Non-blocking suggestions — incorporated inline**

| ID | What changed | Where |
|---|---|---|
| C2 | CP1 pre-flight enumerates per-type-mode configs via `grep -rln "mode: per_type" configs/`; CP3 sweeps all 86, surfaces the list in the implementation report. | CP1 + CP3 descriptions, open question #3 |
| C3 | JIT-recompile test (CP5) now uses explicit `log.count("Compiling jax_step") == 1` regex assertion on captured `jax_log_compiles` output. | Test Plan §(e), CP5 description |
| C4 | Per-episode sampling test (CP2) explicitly uses a non-degenerate-range config; cross-field independence test added (Pearson `\|r\| < 0.5` over 100 keys per field pair); state-vs-params placement assertion added; wander/static-zero-state assertion added. | Test Plan §(b), CP2 description |
| C5 | New test (h) `test_info_dict_aliases.py` sweeps the full info-dict legacy-alias surface (`hit_predator`, `hit_neutral`, `damage_predator`, `damage_obstacle`, `damage_hiding_predator`, `dist_per_predator`, `dist_per_neutral`) and asserts each is preserved name-for-name. | Test Plan §(h) |
| C-CFG-3 | Reworded the olfaction_parity_neutral.yaml migration description: it's a one-line strip of `predator_enabled: false`, not a restructuring — `predators: []` is already present on L11. | Risks item 5, CP1 description |
| C-CFG-4 | Added explicit atomic-commit note on CP1: loader change + `verify_noise.py` rewrite + 86-config migration sweep land in a single atomic commit. | CP1 description |
| C-CFG-5 | CP5 JIT-recompile test now has a positive control (Part 2): swap class orderings at same N, assert `log.count("Compiling jax_step") == 2` (recompile IS expected because `animal_classes` is `pytree_node=False`). Added Risks item 2 documenting the boundary. | Test Plan §(e), CP5 description, Risks item 2 |
| C-CFG-6 | Added YAML cadence-distinction convention to the loader section: `damage: [lo, hi]` is per-event, `detection_range: [lo, hi]` (and four siblings) are per-episode; cadence is field-name-determined, not syntax-determined; canonical examples carry inline comments. | §"Unified config schema" (after loader numbered steps) |
| code-reviewer Missed-failure-modes 1–5 | (1) Per-episode-sampled fields live on EnvState not EnvParams — assertion added to test (b). (2) Behaviour-mask axis test with `[hunt, wander, hunt, static]` — added to CP3 test. (3, 4) `state.animal_state[wander_idx] == 0` after 1000 steps + documentation that wander/static get unused per-episode draws — added to test (b). (5) `per_type` enumeration in CP3 pre-flight — added to CP1 + CP3. | Test Plan §(b), CP3 description, Risks |

**Non-blocking suggestions — deferred**

| ID | Why deferred | Owner |
|---|---|---|
| `lose_interest_multiplier` mandatory-promotion semantic-change note | Inline-documented in the loader docstring (added in the loader numbered-steps text); no separate deferral needed. | resolved in this revision |
| `placement.mode: per_entity` soft-default cleanup (config-auditor NIT) | Pre-existing issue, not introduced by this plan. Flag as a separate developer ticket. | separate plan |
| `name` field vs `tag` field in unified schema (config-auditor NIT) | Both are present today; the new schema uses `tag` for the metric label suffix and keeps `name` as an optional human-readable identifier. Not a blocker. | resolved (existing `name` semantics preserved) |
| `damage_key` reuse triage (Risks item 4) | Pre-existing, explicitly preserved by the refactor. Out of scope. | separate plan |

**New open questions surfaced during this revision**

- **None of consequence.** The per-subset call pattern (B1 fix) introduces `hunt_idx` / `wander_idx` / `static_idx` as new `pytree_node=False` fields on `EnvParams`. These are constructed at param-build time from `animal_behaviours` (also `pytree_node=False`), so they cannot drift; the constructor is a one-liner. No new design decision required.

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
