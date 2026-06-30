---
title: "Code review — per-episode entity-count activation masks"
topic: reviews
status: active
created: 2026-06-22
last_updated: 2026-06-22
---

# Code review — per-episode entity-count activation masks

## Verdict (plain language)

**CHANGES-REQUESTED.** The feature being reviewed lets each episode start with a
*random* number of food / bushes / rocks / predators / rabbits (e.g. "between 2
and 6 food") instead of a fixed count, by allocating the maximum number of slots
and switching the unused ones "off" for that episode. An off slot is supposed to
be completely invisible: it can't bite, hide the agent, be smelled, or collide.

For two of the three entity families — animals (predator/rabbit) and obstacles
(rock/bush) — the off-switch is wired through every place that matters, and I
could not find a leak. **The third family, resources (food and hiding-predator),
has a real escape bug**: a slot that starts the episode "off" turns itself back
**on** on the very first step and gets re-placed on the grid. I reproduced this
directly — a reset that draws 1 active food (3 off) becomes 4 active food after
one `jax_step`. The cause is that the resource code reuses the same `False` flag
to mean two different things ("eaten, regenerating" vs. "never existed this
episode"), and the regeneration logic happily "regrows" the never-existed slots.

The net effect: **per-episode count variance is silently defeated for food and
hiding-predator** — every episode ends up with the maximum count after one step,
which is exactly the layout-memorisation confound the feature was built to
remove. The animal/obstacle masking is correct and can ship; the resource path
needs the fix below before this is sound. There is also a test gap: the
within-episode-stability test never checks `res_active`, which is why this slipped
through.

Scope reviewed: the diff `90d687f~1..7fa6183` across `src/environment/{config_loader,core,sensor,state}.py`,
against the plan [[PER_EPISODE_ENV_VARIANCE]].

## Findings

| # | Severity | Site | Issue | Suggested fix |
|---|----------|------|-------|---------------|
| 1 | 🔴 blocker | `src/environment/core.py:130-141` (`update_resources`), reached from `core.py:400` | **Inactive resource slots are revived on the first step.** An off count-range slot is initialised with `res_active=False` and `res_reg_timer=0` (`core.py:1175-1176`). `update_resources` computes `respawn_mask = ~res_active & (new_reg_timer <= 0)` → `~False & (0<=0) = True`, so it flips the slot to `new_active=True` and `res_pos_after_reg` re-places it on-grid (`core.py:413`). The activation mask is defeated for **food and hiding_predator** after one step. Reproduced: reset K=1/4 food → 4/4 active after one `jax_step`. | Track which slots were ever allocated this episode and gate respawn with it: keep the reset activation mask as a per-episode "allocated" mask (it is constant within the episode) and set `respawn_mask = ~res_active & (new_reg_timer <= 0) & allocated`. Cheapest concrete option: thread the reset mask in — but `res_active` already carries it at reset, so the minimal fix is to AND `respawn_mask` (and `new_active`) with a slot-allocated mask derived from `params` at step time, OR set inactive-slot `res_reg_timer` to a never-reaching sentinel **and** make `update_resources` not decrement/trip it. The allocated-mask route is the clean one. Add a `res_active`-stability assertion to the test (see #5). |
| 2 | 🟡 concern | `tests/env/test_per_episode_count.py:455-506` (`test_mask_stable_within_episode`) | The within-episode stability test asserts only `animal_active` and `obs_active` are constant across 50 steps — it **omits `res_active`**, and its config has no resource range. This is exactly the blind spot that let finding #1 through. `test_inactive_food_not_sensed` (`:338`) senses the *reset* state and steps only once for `dist_to_food`, but never asserts the mask survives stepping. | After fixing #1, extend `test_mask_stable_within_episode` to include a food `count_low<count_high` entry and assert `jnp.sum(state.res_active)` stays equal to the reset K across all 50 steps (not just that the boolean array is stable — respawn changes it, so equality of the count is the real guard). |
| 3 | 🟢 nit | `src/environment/core.py:1126-1142` (`_build_activation_mask` callers) | The off-grid park is applied only when `params.has_*_range` is True (`core.py:1138/1140/1142`), so for degenerate configs the positions array is byte-identical (good, parity-preserving). But the mask itself (`res_activation_mask` etc.) is always built and stored even in the degenerate case (it returns `jnp.ones`). That is correct and intended; no action — flagging only so the reviewer of the fix for #1 knows the all-True mask is a safe "allocated" mask to reuse. | None — informational. |
| 4 | 🟢 nit | `src/environment/core.py:1135-1142` | Off-grid park value is `(height, width)`. Agent is clamped to `[0,height-1]×[0,width-1]` (`core.py:24-27`), animals to the same (`:241`, `:290`), so an off-grid entity at `(10,10)` can never be coincident with any on-grid actor — sound. Distance metrics additionally mask via the active masks (`:628-629`, `:640`, `:648`), so even the off-grid norm (~`sqrt(2)*10` from a corner) is excluded. Belt-and-suspenders as the plan intended. | None. |

## Per-check results

### 1. Inertness at every consumption site

**Animals (`animal_active`)** — fully threaded, verified each site:
- damage / bite: `core.py:509` (`at_damaging & params.animal_is_damaging & state.animal_active`). ✅
- neutral contact: `core.py:561` (`& state.animal_active`). ✅
- movement (hunt/wander collision + concealment receive `obs_active`, and animals are parked off-grid so they don't move onto the grid): `update_animals` passes `obs_active` at `:358`/`:381`; inactive animals stay parked. ✅
- olfaction: `sensor.py:322` passes `state.animal_active` into `sense_resource`. ✅
- visual: `sensor.py:209` appends `state.animal_active`. ✅
- extero-nociception: `sensor.py:77` ANDs `params.animal_is_damaging & state.animal_active`. ✅
- distance metrics: `core.py:640` / `:648` AND the pred/neutral masks with `state.animal_active` → inactive read 99.0. ✅

**Obstacles (`obs_active`)** — fully threaded:
- agent collision: `move_agent` `core.py:36-42` (`obs_blocking & obs_active`); called with `obs_active` at `:434`. ✅
- animal collision (hunt/wander): `core.py:244-246`, `:293-295`; passed at `:358`/`:381`. ✅
- bush concealment in hunt-step: `core.py:178-181` (`obs_hides_agent & obs_active`). ✅
- bush concealment for `agent_in_bush` info: `core.py:670` (`& state.obs_active`). ✅
- overlap damage: `core.py:535` (`& state.obs_active`). ✅
- collision damage + collision noc: `core.py:540`, `:543` (`at_attempted_obs & state.obs_active`). ✅
- olfaction / visual: `sensor.py:323` / `sensor.py:207`. ✅
- collision sensor: `sensor.py:40` (`params.obs_blocking & state.obs_active`). ✅

**Resources (`res_active`)** — the consumption-side reads are correct (`interact_resource`
at `core.py:446` uses post-respawn `new_active`; `dist_to_food`/`dist_to_hiding_predator`
at `:628-629` AND `res_active`; olfaction/visual use `res_active`). **But** the
*respawn* path reactivates inactive slots — see finding #1. So inertness holds at
reset and would hold across steps *if the slot stayed inactive*, but it does not.
❌ (blocked on #1).

### 2. Per-episode K-sampling & PRNG threading

- K is drawn from `fold_in(property_key, <class-const>)` with distinct constants
  `0xC0A1/0xC0A2/0xC0A3` (`core.py:1058-1060`), then `split` per entry and
  `randint(ek, (), lo, hi+1)` per entry (`core.py:1083-1090`). `fold_in` yields an
  independent stream and does **not** consume `property_key`, so reusing
  `property_key` as the fold base across the three classes **and** the existing
  property sampling is safe — no key reuse, no stream collision. ✅
- Degenerate-only classes skip the `jax.random` call entirely (`has_*_range` static
  bool, `core.py:1077`), preserving byte-identical PRNG streams for non-opted-in
  configs. Verified by `test_degenerate_range_parity` + the green parity suite. ✅
- chosen-K → mask mapping: `lax.scan` cumcount gives each slot its within-entry
  rank; `mask = slot_rank < K[entry]` → exactly K active per entry, deterministic
  given key. ✅ Reproduced: same key → same `res_active` (`test_reproducibility`).

### 3. Static-shape / recompile safety

- Allocation is always `count_high` (loader `config_loader.py` expansion loops use
  `range(hi)`), so array shapes are fixed regardless of K. ✅
- K and all masks are traced JAX values; `has_*_range` are the only new branch
  predicates and they are `struct.field(pytree_node=False)` static bools
  (`state.py:170-172`) — correct, they gate a Python-level `if` that must not be
  traced. ✅
- count-bound arrays (`*_count_low/high`, `*_entry_id`) are **traced** int arrays
  (no `pytree_node=False`) — correct, they are not shape-determining (shape comes
  from the slot count, fixed at load). ✅
- No new Python-int branch on a traced value; `num_slots == 0` and `not has_range`
  are static. No recompile across differing-K resets (covered by Checkpoint 5 /
  `test_shape_stability_across_k_values`). ✅

### 4. vmap-safety

- `_build_activation_mask`'s `lax.scan` carry is `[num_entries]` fixed; operates on
  static-shaped `entry_id_arr`. vmapping `jax_reset` over envs is clean — no shape
  leak. ✅
- Off-grid park `(height, width)` cannot collide with any clamped on-grid actor
  (finding #4). Placement uniqueness at reset (`resolve_overlaps_global`) runs
  **before** the off-grid park, so active slots are de-duplicated among themselves;
  inactive slots are then moved off-grid and excluded from collision. ✅
- Masked reductions (`jnp.where(mask, …, 99.0/0.0)` + `jnp.max/min`) are
  shape-preserving and vmap-safe. ✅

### 5. Backward-compat (degenerate `count: N`)

- `_resolve_count_range` returns `(N, N)` for scalar `count`, `(1,1)` for neither,
  and raises `ValueError` if both styles present (`config_loader.py:783-820`).
  `0 <= low <= high` validated. ✅
- Degenerate → `has_*_range=False` → no K-draw, all-True mask, no off-grid park →
  byte-identical reset (parity suite green, `test_degenerate_range_parity`). ✅
- **Caveat:** the degenerate path is byte-identical *because finding #1 never fires
  when the mask is all-True* (every slot allocated → respawn revival is a no-op).
  So the backward-compat path is genuinely safe; the bug is confined to configs
  that actually opt into a resource range (`food`/`hiding_predator` with
  `count_low < count_high`) — which includes the new `default.yaml` (food 2–6,
  hiding_predator 2–4). The default scene is therefore affected in production.

## Conclusion

CHANGES-REQUESTED — one 🔴 blocker (resource slots revive on step 1, defeating
count variance for food + hiding_predator, including the new default scene) and
one 🟡 test-coverage gap that hid it. Animal and obstacle masking is correct and
complete across all consumption sites; PRNG, static-shape, vmap, and degenerate
backward-compat checks all pass. Fix #1 + extend the stability test (#2), then
re-run `tests/env/test_per_episode_count.py` and the parity suite.

Conventions audit: pytree ✅ · JIT/recompile ✅ · vmap ✅ · PRNG ✅ · sensor/obs-breakdown sync ✅ · config protocol ✅ · **resource-inertness ❌ (blocker #1)**

Reviewed by: code-reviewer
