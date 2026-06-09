---
title: "Env Feature — disengage_on_contact (strike-and-retreat for hunting animals)"
topic: env_entities
status: implemented
created: 2026-06-09
last_updated: 2026-06-09
aliases: [disengage_on_contact, env_entities_strike_retreat]
---

# Env Feature — `disengage_on_contact` (strike-and-retreat for hunting animals)

> **Status**: IMPLEMENTED
> **Opened**: 2026-06-09
> **Related**: [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]] (the unified-animal refactor this builds on); memory insights `20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride` and `20260609_1747_avoidance_is_post_contact_not_preemptive` (the chasing-rabbit study that motivated it).

---

## Context

In the chasing-rabbit study, a **harmless hunting rabbit** — an animal that chases the agent but deals zero damage — was observed to **sit on top of the agent's cell forever after it caught up**. That "riding" happens because the only thing that makes a hunting animal pause and back off today is *taking damage*: the post-contact pause is wired to the attack-delay timer, which only fires for animals that actually hurt the agent. A zero-damage chaser never triggers it, so it just stays glued to the agent.

The user wants a cleaner, reusable **"strike-and-retreat"** behaviour: the moment a hunting animal touches the agent, it should lose interest, walk back toward the middle of its home patch to "catch its breath", rest until recovered, and only then start chasing again. The same mechanic will later be useful for the real predator (a bite-and-retreat hunter), so we want it as a general, opt-in switch rather than a one-off rabbit hack.

The good news: the environment **already has the full disengage → retreat → recover → re-chase cycle** built into the hunt behaviour. It is driven entirely by the animal's "stamina" (an energy bar): when stamina hits zero, the animal gives up the chase and heads home to recover; once rested it re-engages. The **only missing piece** is a trigger that drains that energy bar to zero on contact. This plan adds exactly that — a single per-animal on/off flag, `disengage_on_contact`, plus one line in the step function that zeroes an animal's stamina when it lands on the agent's cell (but only for animals that opted in). Everything downstream — the retreat, the rest, the re-chase — is the existing machinery, untouched.

This is a deliberately **small, surgical** change: one optional config flag, one boolean array threaded through the loader and the params object, and one line in the step. It is **opt-in and defaults to off**, so all 86 existing configs and every existing animal behave byte-for-byte exactly as before.

---

## Analysis

### The existing hunt cycle (already present — we are NOT changing it)

The hunt state machine lives in `src/environment/core.py:_hunt_step`. Each hunting animal carries an integer `state` (0 = PATROL / wander in home patch, 1 = HUNT / chase agent, 2 = RETURN / walk back to patch centre) and a float `stamina` energy bar. The transitions that matter here, verbatim from the current code:

- **Give up the chase** (`lose_interest`, core.py ~173–176): becomes true when the agent is too far (`dist > detect × lose_interest`), **OR `hunt_stamina <= 0`**, OR the agent is hidden in a bush. While in HUNT (state 1), `lose_interest` flips the animal to RETURN (state 2) (~line 180).
- **Walk home and reset** (core.py ~182–187): RETURN heads to the patrol-area centre `(tr_return, tc_return)`; when it arrives (`dist_to_center <= 2`) it drops back to PATROL (state 0).
- **Recover while not hunting** (core.py ~236): `new_stamina = where(state == 1, stamina − 1, stamina + recovery)` — stamina drains only while actively chasing and refills in every other state, clipped to `[0, max_stamina]`.
- **Re-engage once rested** (`become_hunt`, core.py ~167–171): becomes true when the agent is back in detection range (`dist <= detect`), the animal is **rested enough** (`stamina >= max_stamina × hunt_thresh`), and the agent is not hidden — flipping PATROL/RETURN back to HUNT.

So `stamina → 0` is *already* the canonical "disengage now" signal. An animal whose stamina is forced to 0 will, on the very next step, satisfy `lose_interest`, transition HUNT → RETURN, walk home, refill stamina (`+recovery` each step), and re-HUNT when `stamina >= max_stamina × hunt_thresh` and the agent is back in range. **No new state-machine logic is required.**

### The only missing trigger: drain stamina on contact

The step function `src/environment/core.py:jax_step` already computes the post-step animal stamina and the contact mask we need:

- `new_animal_stamina` is returned from `update_animals(...)` at core.py ~403 (it is the per-animal stamina *after* this step's hunt update).
- `at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)` at core.py ~471 — a per-animal boolean: True where the animal's **post-step** cell equals the agent's **post-step** cell (i.e. it caught the agent this step). This is the same mask the predator-damage block uses.

The feature is therefore a one-line override applied after both quantities exist and before the state is assembled (core.py ~627):

```python
new_animal_stamina = jnp.where(at_animal & params.animal_disengage_on_contact, 0.0, new_animal_stamina)
```

For a flagged animal that contacted the agent this is `where(True, 0.0, …)` → stamina forced to 0 → next step `lose_interest` fires → HUNT → RETURN → recover → re-HUNT, exactly the desired strike-and-retreat. For any animal that did not contact, or that did not opt in, the predicate is False → `new_animal_stamina` passes through unchanged.

### Why this is byte-safe for the 86 existing configs

Two independent guarantees:

1. **Default off.** `disengage_on_contact` is parsed as optional with default `False`. Every existing config and every existing animal therefore has `params.animal_disengage_on_contact == False` for all entries, so the override is `where(False, 0.0, new_animal_stamina) == new_animal_stamina` — an algebraic no-op, leaf-by-leaf identical to today's array.
2. **No PRNG consumption.** The override is a pure `jnp.where` on already-materialised arrays — it draws **no random numbers**. The v2.0 byte-parity contract (the unified-animal refactor preserved exact per-subset PRNG draw shapes so trajectories stay byte-identical to the pre-refactor code) is about *the sequence and shapes of `jax.random` draws*. This change adds zero draws, so that contract is untouched: a config without the flag produces byte-identical trajectories, obs, and info dicts.

### Why `disengage_on_contact` is OPTIONAL, not a `get_mandatory` key

Project rule: critical config params use `config.get_mandatory('key')` so a missing key raises `ValueError` rather than silently taking a fallback. That rule exists to prevent *silent* divergence on params that change physics for runs that are supposed to specify them. `disengage_on_contact` is the opposite case:

- It is **new opt-in behaviour**. The 86 existing configs predate it and legitimately do not mention it; making it mandatory would raise `ValueError` on all of them and break every existing run and every parity test.
- Its **default (`False`) reproduces today's behaviour exactly** — absence is meaningful and safe, not an ambiguous oversight.

So the correct discipline here is `ent.get('disengage_on_contact', False)` (per-entity, optional, default False), parallel to how optional per-entity fields like `nociception_intensity` are already read with `.get(...)` defaults in the loader. This is *not* a fallback default for a critical key — it is the documented "off" state of an opt-in switch.

---

## Implementation Plan

### Design

Thread one new per-entity boolean from YAML → loader → `EnvParams` (as a pytree leaf array, because it is read inside the JIT'd step) → one override line in `jax_step`. The cycle it triggers is entirely pre-existing. Concretely:

1. **`EnvParams`** gains a leaf array `animal_disengage_on_contact: jnp.ndarray  # [N] bool`.
2. **`config_loader.py`** reads the optional per-entity flag (default False) on **both** the `entities:` path and the legacy `predators:` / `neutral_animals:` re-projection (legacy entries default to False), builds the `jnp.bool_` array, threads it through the `_load_animals` return tuple (both the zero-N and non-zero branches and the caller's unpacking), and passes it into the `EnvParams(...)` constructor.
3. **`core.py:jax_step`** applies the one-line override after `at_animal` and `new_animal_stamina` both exist and before `state._replace(...)`.

Ordering note for the override line: it must come **after** the attack-delay block (~480) is fine either way (the two are independent — attack-delay touches `new_animal_at`, the override touches `new_animal_stamina`), but it MUST be before line ~638 where `animal_stamina=new_animal_stamina` is read into the new state. Place it immediately after the predator-damage block (right after ~line 480) for locality with `at_animal`.

### File Changes

#### `src/environment/state.py` (EnvParams, after line 123)

Add the new leaf array field next to the other per-entity animal arrays. It is a **pytree leaf** (a JAX array used inside the JIT'd step), NOT a `struct.field(pytree_node=False)` static field.

```python
# BEFORE:
    animal_is_damaging: jnp.ndarray        # [N] bool (precomputed from class)
    animal_visual_channel: jnp.ndarray     # [N] int (5=predator, 7=neutral, ...)

# AFTER:
    animal_is_damaging: jnp.ndarray        # [N] bool (precomputed from class)
    animal_disengage_on_contact: jnp.ndarray  # [N] bool (opt-in: drain stamina→0 on agent contact)
    animal_visual_channel: jnp.ndarray     # [N] int (5=predator, 7=neutral, ...)
```

#### `src/environment/config_loader.py`

**(a) `entities:` path — read the optional flag (after line 327).** Add to the `entries.append({...})` dict in the `entities:` branch:

```python
# BEFORE:
                    'damage': ent.get('damage'),
                    'attack_delay': ent.get('attack_delay'),
                    'spawn_area': ent.get('spawn_area'),

# AFTER:
                    'damage': ent.get('damage'),
                    'attack_delay': ent.get('attack_delay'),
                    'disengage_on_contact': bool(ent.get('disengage_on_contact', False)),
                    'spawn_area': ent.get('spawn_area'),
```

**(b) Legacy `predators:` re-projection — default False (after line 359).** Legacy predators may opt in if a `disengage_on_contact` key is present in the YAML; absent → False:

```python
# BEFORE:
                    'damage': _p_get('damage'),
                    'attack_delay': _p_get('attack_delay'),
                    'spawn_area': p.get('spawn_area'),

# AFTER:
                    'damage': _p_get('damage'),
                    'attack_delay': _p_get('attack_delay'),
                    'disengage_on_contact': bool(p.get('disengage_on_contact', False)),
                    'spawn_area': p.get('spawn_area'),
```

**(c) Legacy `neutral_animals:` re-projection — default False (after line 392).** Legacy neutrals may opt in (this is the chasing-rabbit path); absent → False:

```python
# BEFORE:
                    'damage': [0.0, 0.0],
                    'attack_delay': 0,
                    'spawn_area': n.get('spawn_area'),

# AFTER:
                    'damage': [0.0, 0.0],
                    'attack_delay': 0,
                    'disengage_on_contact': bool(n.get('disengage_on_contact', False)),
                    'spawn_area': n.get('spawn_area'),
```

> Note: legacy neutrals are re-projected with `behaviour: 'wander'`, not `'hunt'`. The chasing-rabbit study uses a hunting rabbit — confirm the rabbit config it targets routes through a `hunt` behaviour (entities: path or a `predators:` entry with zero damage), because the disengage cycle only runs inside `_hunt_step`. A pure `wander` animal never enters HUNT/RETURN, so the stamina drain would have no visible effect on it. Reading the flag on the wander path is still harmless (default False, and the override is a no-op for wander animals whose stamina is unused), so we read it on all three paths for uniformity; the behavioural effect only manifests for `hunt` animals. **This plan reads the flag everywhere but does not change behaviour routing — see Checkpoint 4.**

**(d) Zero-N branch — empty array + return tuple (lines ~424, ~446).** Add the empty array beside `animal_is_damaging` and insert it into the zero-N return tuple in the same position used everywhere else (immediately after `animal_is_damaging`):

```python
# BEFORE (~line 424):
        animal_is_damaging = jnp.zeros(0, dtype=jnp.bool_)
        animal_visual_channel = jnp.zeros(0, dtype=jnp.int32)

# AFTER:
        animal_is_damaging = jnp.zeros(0, dtype=jnp.bool_)
        animal_disengage_on_contact = jnp.zeros(0, dtype=jnp.bool_)
        animal_visual_channel = jnp.zeros(0, dtype=jnp.int32)
```

```python
# BEFORE (zero-N return, ~line 446):
            animal_is_damaging, animal_visual_channel,

# AFTER:
            animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
```

**(e) Non-zero build — collect into a list + build the array (lines ~505, ~532, ~573).** Add a collection list, append per entry inside the existing `for i, e in enumerate(entries)` loop, and build the bool array next to `animal_is_damaging`:

```python
# BEFORE (~line 505):
    is_damaging_list = []
    visual_channel_list = []

# AFTER:
    is_damaging_list = []
    disengage_on_contact_list = []
    visual_channel_list = []
```

```python
# BEFORE (inside the loop, ~line 532):
        is_damaging_list.append(cls in ANIMAL_DAMAGING_CLASSES)
        visual_channel_list.append(ANIMAL_CLASS_TO_VIS_CHANNEL[cls])

# AFTER:
        is_damaging_list.append(cls in ANIMAL_DAMAGING_CLASSES)
        disengage_on_contact_list.append(bool(e['disengage_on_contact']))
        visual_channel_list.append(ANIMAL_CLASS_TO_VIS_CHANNEL[cls])
```

```python
# BEFORE (~line 573):
    animal_is_damaging = jnp.array(is_damaging_list, dtype=jnp.bool_)
    animal_visual_channel = jnp.array(visual_channel_list, dtype=jnp.int32)

# AFTER:
    animal_is_damaging = jnp.array(is_damaging_list, dtype=jnp.bool_)
    animal_disengage_on_contact = jnp.array(disengage_on_contact_list, dtype=jnp.bool_)
    animal_visual_channel = jnp.array(visual_channel_list, dtype=jnp.int32)
```

**(f) Non-zero return tuple (line ~606).** Same position as elsewhere:

```python
# BEFORE:
        animal_is_damaging, animal_visual_channel,

# AFTER:
        animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
```

**(g) Caller unpacking in `load_env_params` (line ~684).** The tuple is unpacked here — add the name in the same slot:

```python
# BEFORE:
        animal_is_damaging, animal_visual_channel,

# AFTER:
        animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
```

**(h) `EnvParams(...)` constructor (after line 862).** Pass the array through:

```python
# BEFORE:
        animal_is_damaging=animal_is_damaging,
        animal_visual_channel=animal_visual_channel,

# AFTER:
        animal_is_damaging=animal_is_damaging,
        animal_disengage_on_contact=animal_disengage_on_contact,
        animal_visual_channel=animal_visual_channel,
```

> **New config key (per project convention — explicit listing):**
> - YAML path: `environment.entities[i].disengage_on_contact` (and the legacy aliases `environment.predators[i].disengage_on_contact`, `environment.neutral_animals[i].disengage_on_contact`)
> - Type: `bool`
> - Default: `False` (optional — NOT a `get_mandatory` key; see Analysis)
> - Effect: when `True`, the animal's stamina is forced to 0 on the step it contacts the agent, triggering the existing disengage → retreat → recover → re-engage cycle. Only behaviourally meaningful for `hunt` animals.
> - This plan does NOT add the key to any existing config (that is experiment-designer's next step). All 86 existing configs remain byte-unchanged.

#### `src/environment/core.py:jax_step` (after line ~480, before the state assembly at ~627)

Insert the single override line. Place it right after the attack-delay block so it sits next to the `at_animal` mask it consumes:

```python
# BEFORE (~line 479–481):
    # Trigger Attack Delay for damaging animals that hit the agent
    new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)

# AFTER:
    # Trigger Attack Delay for damaging animals that hit the agent
    new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)

    # Strike-and-retreat: opt-in animals lose all stamina on contact, which makes
    # the existing hunt cycle disengage (HUNT→RETURN), retreat to patrol centre,
    # recover, and re-engage. No-op (where(False, ...)) for animals that did not
    # opt in or did not contact this step; draws no PRNG → byte-parity preserved.
    new_animal_stamina = jnp.where(
        at_animal & params.animal_disengage_on_contact, 0.0, new_animal_stamina
    )
```

`at_animal` (core.py ~471) is the post-step contact mask; `new_animal_stamina` (from `update_animals`, core.py ~403) is the post-hunt stamina. Both are in scope at this point. The result flows unchanged into `state._replace(animal_stamina=new_animal_stamina, ...)` at ~638.

### Test plan (developer MUST execute)

Add a new test module `tests/env/test_disengage_on_contact.py`. All three tests below must be implemented and pass.

**Test 1 — full strike-and-retreat cycle (unit).**
Build a minimal config (or construct `EnvParams` directly) with a single **hunting** animal that has `disengage_on_contact: True`, a small patrol area, finite `max_stamina`, and a positive `stamina_recovery_rate`. Drive the env so the animal is placed on / steps onto the agent's cell, then assert the full cycle:
  1. On the step where `at_animal` is True for that animal, the next state's `animal_stamina[i] == 0.0`.
  2. The following step, the animal's `animal_state[i]` transitions from HUNT (1) to RETURN (2) (it satisfies `lose_interest` because stamina is 0).
  3. With the agent moved out of detection range, over subsequent steps `animal_stamina[i]` increases (recovery) and the animal reaches its patrol centre / returns to PATROL (0).
  4. With the agent brought back into detection range **and** the animal rested (`stamina >= max_stamina × hunt_thresh`), `animal_state[i]` re-enters HUNT (1).
Assert all four transitions in sequence (the cycle), not just the stamina zeroing.

**Test 2 — byte-parity for non-opted-in configs.**
Pick an existing legacy config that contains a hunting animal (e.g. one of the hypervigilance configs already used as the parity reference, `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`). Run 100 steps from seed 0 with the fixed action sequence `[0,1,2,3,4]*20` **before logic** and **after logic** are not directly comparable across a code edit, so instead assert against the committed parity fixtures: confirm this config still passes the existing `tests/env/test_unified_parity.py` (byte-identical `animal_pos`, `animal_stamina`, `animal_state`, info-dict keys) — i.e. the change introduces **zero** trajectory drift for configs without the flag. Additionally, in the new module, assert directly that for a no-flag config the per-animal `params.animal_disengage_on_contact` is all-False and that two 100-step rollouts (with vs. without the new override line conceptually — i.e. the override is a verified no-op) produce identical `animal_stamina` arrays. The pre-existing parity suite passing is the load-bearing proof; do not weaken it.

**Test 3 — composes with unified arrays / `hunt_idx` subset and `vmap` over envs.**
Build a mixed config with ≥3 animals across behaviours (e.g. `hunt_idx == (0, 2)`, `wander_idx == (1,)`), with `disengage_on_contact: True` on at least one hunt entry and the default (absent → False) on the others. Assert:
  1. `params.animal_disengage_on_contact` has shape `[N]` and matches the per-entry flags in entity order (the override is applied over the **full** unified animal array, not the `hunt_idx` subset — confirm a False-flagged hunt animal is unaffected and a True-flagged one disengages).
  2. `jax.vmap(jax_step)` over a batch of envs (or the project's existing vmapped-step harness) runs without shape errors and the override broadcasts correctly per env. Reuse the vmap pattern from existing env tests (e.g. `test_no_recompile.py` / `test_unified_parity.py`) rather than inventing a new one.

> Bug-triage note: this is a **feature**, not a bug fix, so there is no pre-fix failing reproducer to gate on. Test 1 is the behavioural acceptance test; Test 2 is the backward-compat guard; Test 3 is the integration/vmap guard.

### Reusability

`disengage_on_contact` is class-agnostic — it reads `at_animal` (contact) and a per-entity bool, with no dependency on `is_damaging`. The same flag can later be set on the **predator** (a damaging `hunt` animal) to give it a **bite-and-retreat** profile: on contact it deals its damage (existing attack-delay path, unchanged) AND drains its stamina to 0 (this feature), so it strikes, backs off to its patch, recovers, and re-stalks — a more naturalistic predator than today's perpetually-attached chaser. No further code is needed for that; it is purely a config choice handed to experiment-designer.

## Checkpoints

What the implementing agent should verify **during** implementation:

- [x] **Checkpoint 1 — tuple arity.** Verified: `animal_disengage_on_contact` sits immediately after `animal_is_damaging` in all 3 slots (zero-N return, non-zero return, caller unpacking). `params.animal_visual_channel.dtype == int32` confirmed — slots not crossed.
- [x] **Checkpoint 2 — array shape + dtype.** N=3 config: `shape=(3,) dtype=bool`; zero-N config: `shape=(0,) dtype=bool`. Both verified.
- [x] **Checkpoint 3 — no-op for existing configs.** `test_unified_parity.py` + `test_entities_schema.py`: 38 passed, 71 skipped. Full env suite: 134 passed, 132 skipped, 0 failures.
- [x] **Checkpoint 4 — flag only matters for hunt animals.** Confirmed: wander animal with `disengage_on_contact: True` loads correctly; override fires but is behaviorally a no-op (wander never reads stamina for state transitions). Noted in report.
- [x] **Checkpoint 5 — no PRNG drift.** The override block contains zero `jax.random` calls. Confirmed via grep.

## Implementation Report

> **Implemented by**: developer (Claude Sonnet 4.6)
> **Date**: 2026-06-09

### Summary

All changes implemented exactly as specified in the plan's File Changes section.

**`src/environment/state.py`**: Added `animal_disengage_on_contact: jnp.ndarray  # [N] bool` as a pytree leaf immediately after `animal_is_damaging` (line 124).

**`src/environment/config_loader.py`**: Eight edits in total:
- (a) `entities:` path: added `'disengage_on_contact': bool(ent.get('disengage_on_contact', False))` to the `entries.append({...})` dict.
- (b) Legacy `predators:` path: added same key with `p.get(...)`.
- (c) Legacy `neutral_animals:` path: added same key with `n.get(...)`.
- (d) Zero-N branch: added `animal_disengage_on_contact = jnp.zeros(0, dtype=jnp.bool_)` and inserted it in the return tuple immediately after `animal_is_damaging`.
- (e) Non-zero build: added `disengage_on_contact_list = []`, loop append `disengage_on_contact_list.append(bool(e['disengage_on_contact']))`, and `animal_disengage_on_contact = jnp.array(disengage_on_contact_list, dtype=jnp.bool_)`.
- (f) Non-zero return tuple: inserted `animal_disengage_on_contact` in same position.
- (g) Caller unpacking in `load_env_params`: same position.
- (h) `EnvParams(...)` constructor: `animal_disengage_on_contact=animal_disengage_on_contact` added.

**`src/environment/core.py:jax_step`**: Added the one-line override block after the attack-delay block (after line 480):
```python
new_animal_stamina = jnp.where(
    at_animal & params.animal_disengage_on_contact, 0.0, new_animal_stamina
)
```

**`tests/env/test_disengage_on_contact.py`**: Created with 3 tests (all pass). Test config uses `start_pos: [1,1]` (0-indexed [0,0]) and `spawn_area: [[1,1],[1,1]]` so the animal starts on the agent's cell, with `patrol_area: [[1,1],[10,10]]` (center at [5,5]) far enough from the agent that after HUNT→RETURN the animal doesn't immediately satisfy `dist_to_center<=2` and skip RETURN.

### Test results

```
tests/env/test_disengage_on_contact.py: 3 passed in 22.0s
tests/env/test_unified_parity.py + test_entities_schema.py: 38 passed, 71 skipped
Full env suite: 134 passed, 132 skipped, 0 failures
```

### Validation smoke output

```
disengage_on_contact flag: True
Patrol area stored: [ 0  0 10 10], center: (5,5)

Initial: agent=[0 0], animal=[0 0]

 Step   Stamina   HuntState   Dist-to-Agent  Note
----------------------------------------------------------------------
    1       0.0        HUNT               0  <-- CONTACT: stamina→0
    2       2.0      RETURN               1  <-- HUNT→RETURN (disengage)
    3       4.0      RETURN               2
    4       6.0      RETURN               3  <-- recovering
    5       8.0      RETURN               4
    6      10.0      RETURN               5
   11      20.0      PATROL               8
   15      25.0        HUNT               5  <-- RE-ENGAGES HUNT

Cycle phases observed:
  CONTACT + stamina→0 : True
  HUNT→RETURN         : True
  Recovering stamina  : True
  RE-ENGAGES HUNT     : True
```

Full cycle completed in 15 steps: contact → stamina forced to 0 → HUNT→RETURN → retreats toward patrol center [5,5] → stamina recovers (+2/step) → re-engages HUNT when stamina≥25 and agent is still in range.

### Speed check

Config: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` (3 animals), seed 0, 1000 steps post-warmup.
- **Result**: 668 steps/sec.
- The change adds a single `jnp.where` over a 3-element already-materialised array — no measurable overhead. No before/after comparison taken (change cannot plausibly affect runtime).

### Deviations from plan

**None.** All 8 config_loader edits, the state.py field, and the core.py line were implemented exactly as the plan specified.

**Notable: test coord system.** The plan's test config placed the animal at `spawn_area: [[5,5],[5,5]]` with agent at `start_pos: [5,5]`. After debugging, the patrol area `[[1,1],[3,3]]` in the plan's test was too small to include the agent (the animal gets clipped to patrol bounds and can never reach [4,4]). The final test uses `start_pos: [1,1]` (agent at 0-indexed [0,0]) with `patrol_area: [[1,1],[10,10]]` (covers the whole grid, center at [5,5] = far from agent). The tested behavior is identical to what the plan specified; only the specific coord values changed.

**Checkpoint 4 note (as requested by plan):** A `wander` animal with `disengage_on_contact: True` will have its (unused) stamina zeroed on contact, but this is behaviorally a no-op because `_wander_step` never reads stamina for state transitions. This is expected, not a bug.

### Blockers / follow-up

None. The feature is ready for use by `experiment-designer` — set `disengage_on_contact: true` on any `behaviour: hunt` entity in the YAML to enable the strike-and-retreat cycle.

**Signed: Implemented by: developer**

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/state.py` | add `animal_disengage_on_contact` leaf | | |
| `src/environment/config_loader.py` | parse flag (3 paths) + thread array | | |
| `src/environment/core.py` | one-line stamina override in `jax_step` | | |
| `tests/env/test_disengage_on_contact.py` | 3 tests (cycle / parity / vmap) | | |

**Conclusion**: [one-line summary]

<!-- For ⚠️/❌ items, add detailed sections below the table with root cause,
     affected lines, and recommended fix. -->
