---
title: "Configurable per-entity visual properties — make the visual sensor work like olfaction (refactor context + plan seed)"
topic: sensors
status: active
created: 2026-06-16
last_updated: 2026-06-16
phase: null
aliases: [configurable-visual-properties, visual-properties-refactor]
---

# Configurable per-entity visual properties

> **For the new implementation session.** This is the single entry-point context doc for the
> visual-observation refactor on the **`v3.0`** branch. Read this first, then the linked code and
> the experiment-environment context docs. The formal checkpointed plan should be authored by
> `senior-developer` under this same `docs/develop/active/sensors/` topic; this doc is the seed +
> rationale it builds on.

## Purpose (read this first)

Today, when we add or change an entity type in the world, the agent's **visual observation** can
break in a way that forces a full retrain. The reason: each entity's appearance is a **hardcoded
one-hot "channel"** baked into the code (predator = channel 5, neutral = channel 7, food = 3,
hiding-predator = 4, rock = 6, …). Adding a genuinely new appearance means editing that fixed
channel list, which changes the size/meaning of the observation vector and **breaks compatibility
with every already-trained agent** (the "27-dim observation parity" we rely on to load frozen
checkpoints).

The fix is to make the visual sensor work **exactly like the olfactory (smell) sensor already
does**: every entity carries a small **property vector set from its config** (smell already does
this — `properties: [...]`), and the sensor just sums those vectors. Appearance becomes a config
value, not a code constant. Then a new entity look — a lurking "hiding predator" that should look
different from a chasing predator, or "different-coloured predators" that encode different
properties — is a **config edit**, with **no code change and no observation-size break**.

The decisive fact (verified in code): **the visual sensor is *already* an olfactory-style
property-matrix matmul.** It already builds a `[num_entities, 8]` property matrix and does
`matches @ all_props`. The *only* rigid part is that this matrix is built from
`jax.nn.one_hot(channel)` instead of a config vector. Swap that one construction for a
config-supplied `visual_properties` vector and the whole class→channel→obs-size coupling problem
disappears.

This work is being version-bumped to **`v3.0`** precisely because it changes the config system
broadly (every entity in ~86 configs gains an optional `visual_properties` field).

## The problem in one concrete example

We want a **hiding predator** (a predator that lurks in place) to be visually distinguishable from a
**chasing predator** in rendered episodes. Right now both are `class: predator`, so both draw the
same `predator.png` and both feed the same visual channel (5). To split them today you'd add a new
`class` / channel — which grows the observation vector and breaks frozen-checkpoint loading. With
configurable visual properties, you just give the hiding predator a different `visual_properties`
vector — and the *same vector can drive the rendered asset* — with zero obs-size change.

(Decision already taken with the user for the hiding predator specifically: keep it **render-only**
distinct — the agent should still *perceive* it as a predator. The visual-properties system makes
both the render-only and the perceptually-distinct options trivial config choices later.)

## How it works today (code map)

**Visual sensor — `src/environment/sensor.py::sense_visual` (~L145–215).** Builds a per-entity
property matrix `all_props` of shape `[total_entities, 8]`, then `vis_entities = matches @ all_props`
(exact-cell match within `visual_sensor_range`). The hardcoded constructions:
- animals: `animal_props = jax.nn.one_hot(params.animal_visual_channel, 8)`  ← **L202, the one rigid line**
- resources: `res_props = jax.nn.one_hot(where(res_type==0, 3, 4), 8)` (food=3, hiding-pred=4)
- obstacles: `obs_props = jax.nn.one_hot(6, 8)` (all → rock=6)
- background/location: one-hot into channels 0/1/2 (grass/sand/plain)

**Olfactory sensor — `src/environment/sensor.py::sense_resource` (L5–20) + L299–302.** The template
to copy. Each entity already carries a config `property` vector (`res_property`,
`animal_property_sampled`, `obs_property_sampled`); the sensor does
`weighted_props = property * distance_decay * mask`, summed. Same shape as visual — the only
difference is olfaction uses distance decay over a radius while visual uses exact-cell match. **We
keep the visual aggregation as-is; we only change where the per-entity vector comes from.**

**Class → channel map — `src/environment/config_loader.py` L32–35:**
`ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}` (also `ANIMAL_CLASS_TO_INT`,
`ANIMAL_DAMAGING_CLASSES`, `ANIMAL_BEHAVIOUR_TO_INT`). `visual_channel_list` is built at L539 and
stored as `animal_visual_channel` (L581); the field lives on `EnvParams`
(`src/environment/state.py` L125).

**The 8 visual channels** (labels at `sensor.py` ~L425):
`['GRS','SND','PLN','FOD','DNG','PRD','NEU','RCK']`. ⚠️ **Latent labeling bug to reconcile**: this
label list puts `NEU` at index 6 and `RCK` at 7, but the actual encoding is rock=6, neutral=7 (per
`ANIMAL_CLASS_TO_VIS_CHANNEL` + `obs_props=one_hot(6)`). The new session should fix this labeling
while it's in here.

**Rendering — `src/environment/renderer_v2.py` (L375/L382) and `grid_world.py` (L42–56, L420–485).**
Draws animals by class (`select_by_class(params,'predator')` → `predator.png`). The dedicated
`hiding_predator.png` asset is wired only to the legacy *resource* path (`res_type==1`). Once
entities carry `visual_properties`, an **asset resolver** can pick/tint the asset from that same
vector — one source of truth for perception + appearance.

## Proposed solution (v1 scope)

Add an optional per-entity **`visual_properties`** vector (mirroring olfactory `properties`) to every
entity type (resources, animals/entities, obstacles) in config + loader. The visual sensor consumes
these vectors directly instead of `one_hot(channel)`.

**The make-or-break design principle — byte-parity via defaults.** Every entity's
`visual_properties` **defaults to its current one-hot** (predator → `[0,0,0,0,0,1,0,0]`, neutral →
channel 7, food → 3, hiding-predator → 4, rock → 6, locations → 0/1/2). Then **all ~86 existing
configs reproduce today's visual observation byte-for-byte**; the new flexibility is opt-in. This is
the same defaults-preserve-parity discipline that made the v2.0 unified-animal-entity refactor land
cleanly — see [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]].

### Open design decisions (resolve in the senior-developer plan)
1. **Vector size** — keep **8** (preserves current channels + obs parity) for v1, vs make it
   configurable like olfactory `vector_size: 5`. Recommend **keep 8** for v1.
2. **Static vs sampled** — ship **static** `visual_properties` (no per-episode `_std` sampling) for
   v1: no new PRNG draw → trivial parity. Olfaction *does* sample (`*_property_sampled` via
   `property_key`), so if visual sampling is added later, **preserve the PRNG draw order** (the same
   trap the unified-animal refactor solved with the per-subset call pattern).
3. **Keep or retire `class→channel`** — **keep** `ANIMAL_CLASS_TO_VIS_CHANNEL` as the
   *default-vector generator* so old configs need no edits; do not delete it.

### Files this will touch (indicative, not a substitute for the plan)
- `src/environment/config_loader.py` — parse `visual_properties` per entity (default = one-hot of the
  current channel); build `animal_visual_property` / `res_visual_property` / `obs_visual_property`.
- `src/environment/state.py` — add the visual-property arrays to `EnvParams`.
- `src/environment/sensor.py::sense_visual` — replace the `one_hot(...)` constructions with the
  config vectors; fix the NEU/RCK label order.
- `src/environment/renderer_v2.py` / `grid_world.py` — optional asset resolver keyed on
  `visual_properties` (can be a later slice).
- `configs/**` — opt-in `visual_properties` only where a distinct look is wanted; everything else
  unchanged.
- tests — a **parity test** (obs byte-identical for existing configs) is the primary gate, plus
  new-vector behaviour tests.

## Coordinate with
- **`v3.0` branch** — all of this lands on v3.0 (created from the v2.0 tip `871f146`).
- A parallel refactor, **`docs/develop/active/refactors/CONFIGURABLE_INITIAL_STATE_RANGES.md`** (other
  session), also touches the config system on v3.0 — check for overlap before large edits.

## Why this matters (experiment-environment context — owned by the other session)
This refactor is the platform enabler for the **behavior-probe** program. Read these for the
motivating context, but do not edit them (they are the other session's home):
- [[experiment_environment_design_perspective]] — the guiding perspective: experiment environments
  interpret a frozen trained agent; adding/altering entities to induce behavior is the whole point,
  so entity changes must be cheap and parity-safe.
- [[experiment_environment_designs_v1]] — the first concrete probes (foraging template uses the
  hiding predator), where the render-distinction need first surfaced.

## Recommended agent flow
Spawn `agent-manager` for a routing plan, then: `senior-developer` (author the checkpointed
implementation plan in `docs/develop/active/sensors/`) → `developer` (implement on v3.0) →
`code-reviewer` (JAX/parity correctness — the byte-parity test is the key gate) + `env-config-auditor`
(config soundness across the migrated YAMLs). `math-reviewer` not required (no equations change).

## References
- Code: `src/environment/sensor.py` (`sense_visual` L145–215, `sense_resource` L5–20),
  `src/environment/config_loader.py` (L32–35, L539, L581), `src/environment/state.py` (L125),
  `src/environment/renderer_v2.py` (L375/L382), `src/environment/grid_world.py` (L42–56, L420–485).
- Precedent: [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]] (the v2.0 unify + per-episode sampling
  refactor; same parity-via-defaults discipline) and memory insight
  `20260529_1823_unified_animal_entity_v2_0_arch`.
- Assets present: `assets/{predator,neutral,hiding_predator,agent_hiding_predator,danger,...}.png`.
