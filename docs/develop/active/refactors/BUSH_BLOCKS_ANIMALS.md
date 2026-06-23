---
title: "Bush as Perfect Refuge — Phase 1: Animal-Blocking Toggle"
topic: refactors
status: active
created: 2026-06-23
last_updated: 2026-06-23
---

# Bush as Perfect Refuge — Phase 1: Animal-Blocking Toggle

## Purpose

This document describes the "bush as perfect refuge" feature, Phase 1 (movement blocking only). The goal is to let any obstacle — particularly a bush — be marked as **impassable to animals** while remaining **freely enterable by the agent**. This gives the agent a defensive safe zone: step into a bush and predators/rabbits cannot follow.

Default: the flag is `false`, so all existing configs and trained runs behave byte-identically.

## The Toggle: `blocks_animals`

A new per-obstacle boolean field `blocks_animals` (default `false`) is added parallel to the existing `blocking` and `hides_agent` flags.

| Flag | Effect |
|---|---|
| `blocking: true` | Blocks both agent AND animals from entering |
| `hides_agent: true` | Agent hidden from predator detection when inside |
| `blocks_animals: true` | **New.** Blocks animal movement into this cell; agent can still enter |

Example YAML (bush as perfect refuge):
```yaml
obstacles:
  - name: bush
    blocking: false       # agent can enter
    hides_agent: true     # agent is hidden while inside
    blocks_animals: true  # animals cannot follow the agent in
    ...
```

## Phase 1 Scope: Movement Only

Phase 1 implements **movement blocking only**. It does NOT change spawn logic — if a bush occupies a cell, an animal can still be spawned there at episode start (spawn exclusion is Phase 2, deferred).

### What changes
- Animals (both wandering neutrals and hunting predators) cannot move onto a cell occupied by an active obstacle with `blocks_animals: true`.
- The agent's `move_agent` call is **unchanged** — it still uses plain `obs_blocking`, so the agent enters freely.
- Inactive obstacles (per the per-episode count-range feature) are still transparent to animals even if `blocks_animals: true`.

### What does NOT change (Phase 2, deferred)
- Spawn exclusion: preventing animals from spawning on `blocks_animals` cells at episode reset.
- Speed check: a dedicated before/after SPS measurement is deferred to Phase 2 (Phase 1 adds only a single `|` operation on two small bool arrays before the animal update loop, negligible cost).

## Implementation Details

### Files changed

| File | Change |
|---|---|
| `src/environment/state.py` | Added `obs_blocks_animals: jnp.ndarray` field to `EnvParams` alongside `obs_blocking`/`obs_hides_agent` |
| `src/environment/config_loader.py` | Load `obs_blocks_animals` from YAML (optional, default False); add zero-array fallback for empty-obstacles case; pass into `EnvParams(...)` |
| `src/environment/core.py` | In `update_animals`: compute `obs_block_for_animals = params.obs_blocking \| params.obs_blocks_animals` once; pass this merged array to `_hunt_step` and `_wander_step` instead of plain `params.obs_blocking`. `move_agent` (agent path) unchanged. |
| `configs/environment/default.yaml` | Added `blocks_animals: false` to the bush entry with comment |
| `tests/env/test_bush_blocks_animals.py` | Four focused tests (see below) |
| `docs/environment/CONFIG_GUIDE.md` | Updated obstacle field table |
| `docs/environment/02_config_schema.md` | Updated obstacle field table + defaults table |

### Threading (core.py)

```python
# In update_animals(), before Branch A / Branch B:
obs_block_for_animals = params.obs_blocking | params.obs_blocks_animals

# Branch A: _hunt_step(..., obs_block_for_animals, ...)   ← was params.obs_blocking
# Branch B: _wander_step(..., obs_block_for_animals, ...) ← was params.obs_blocking
# move_agent ← params.obs_blocking  (UNCHANGED)
```

The `_hunt_step` and `_wander_step` signatures are unchanged; only the value fed to the existing `obs_blocking_for_collision` / `obs_blocking` argument changes.

## Tests

Located at `tests/env/test_bush_blocks_animals.py`:

| Test | What it checks |
|---|---|
| `test_wander_blocked_by_bush` | Wandering rabbit never enters a `blocks_animals=true` bush over 500 steps |
| `test_wander_not_blocked_when_flag_false` | With `blocks_animals=false`, rabbit CAN enter the bush (validates the flag actually does something) |
| `test_hunt_blocked_by_bush` | Hunting predator (agent standing ON the bush) never lands on the bush cell over 500 steps |
| `test_default_off_parity` | No-key config vs. explicit-false config: `obs_blocks_animals` is all-False and animal positions are byte-identical over 200 steps |

Parity tests (`test_unified_parity.py`, `test_visual_parity.py`) pass unchanged — 34 passed, 173 skipped, 0 failures.

## Phase 2 (Deferred): Spawn Exclusion + Speed Check

- **Spawn exclusion**: at `jax_reset`, mask out `blocks_animals` cells from animal spawn placement (analogous to how `obs_blocking` works for the agent spawn). Requires changes to `jax_reset`'s placement logic.
- **Speed check**: SPS before/after measurement once Phase 2 lands (Phase 1's single `|` op is provably negligible; Phase 2 may involve a non-trivial mask scan at reset).

---

## Implementation Report

### Summary

Implemented Phase 1 of the `blocks_animals` feature. All changes are byte-transparent by default (flag defaults to `false`, so `obs_blocking | False == obs_blocking` — identical behaviour for all existing configs).

### File-by-file

- **`src/environment/state.py`**: Added `obs_blocks_animals: jnp.ndarray` to `EnvParams`, next to `obs_hides_agent`. Normal pytree leaf (not `struct.field(pytree_node=False)`), matching the pattern of `obs_blocking` and `obs_hides_agent`.
- **`src/environment/config_loader.py`**: Added `obs_blocks_animals` load in the `expanded_obstacles` branch (line ~987) and the zero-array fallback in the else branch (line ~1023). Passed `obs_blocks_animals=obs_blocks_animals` into `EnvParams(...)` next to `obs_hides_agent`.
- **`src/environment/core.py`**: In `update_animals`, compute `obs_block_for_animals = params.obs_blocking | params.obs_blocks_animals` once before Branch A and B. Feed this merged array to `_hunt_step` and `_wander_step` in place of `params.obs_blocking`. The `move_agent` call at line ~458 is untouched.
- **`configs/environment/default.yaml`**: Added `blocks_animals: false  # animals cannot enter when true; agent still can` to the bush entry.
- **`tests/env/test_bush_blocks_animals.py`**: 4 focused tests. All pass.
- **`docs/environment/CONFIG_GUIDE.md`**: (see below — updated obstacle field table).
- **`docs/environment/02_config_schema.md`**: (see below — updated obstacle field table + defaults table).

### Test results

```
tests/env/test_bush_blocks_animals.py   4 passed  (25s)
tests/env/test_unified_parity.py       34 passed, 173 skipped  (316s)
tests/env/test_visual_parity.py        [included in above run]
tests/env/  (full suite, excl. parity) 136 passed, 164 skipped, 1 warning (302s)
```

All tests pass. No failures. No regressions in parity. The 1 warning is a pre-existing `DeprecationWarning` in `test_behaviour_validation.py` (unrelated to this change).

### Speed check

Phase 1 adds a single `|` bitwise-or between two small bool arrays (`obs_blocking` and `obs_blocks_animals`, shape `[num_obs]`) once per call to `update_animals`. This is provably negligible (O(num_obs) ops before the PRNG-heavy movement logic). Speed check is deferred to Phase 2 per the plan.

### Deviations from plan

None. All files listed in the plan's File Changes section were modified. No undisclosed files touched.

Implemented by: developer
