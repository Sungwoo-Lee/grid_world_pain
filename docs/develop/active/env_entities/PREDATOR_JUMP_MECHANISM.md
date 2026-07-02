---
title: "Env Feature — predator jump / pounce (attack_range + attack_success_rate)"
topic: env_entities
status: active
created: 2026-07-03
last_updated: 2026-07-03
aliases: [predator_jump, predator_pounce, attack_range]
---

# Env Feature — predator jump / pounce

> **Status**: PLANNED
> **Opened**: 2026-07-03
> **Related**: [[DISENGAGE_ON_CONTACT]] (analogous opt-in hunt feature), [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]] (per-episode sampling + activation masks this plan builds on)

---

## Context

Today the chasing predator moves **one cell per step**, exactly as fast as the agent. An agent that simply keeps moving can stay one step ahead forever — "run laps around the map" is a fully valid, permanent escape, so the agent never has any reason to use a bush (the one place a predator cannot reach it). We even see this in a trained noise-variant agent (the "v5" model): early in an episode it shows the beginnings of hypervigilance — it avoids the predator more when it is already injured — but its actual escape strategy is *roaming*, not *hiding*. It never dives into a bush because roaming is risk-free.

This feature gives the predator a **jump (pounce)**: when the agent is close and out in the open, the predator can occasionally lunge across several cells in a single step, either landing directly on the agent (a hit) or landing next to it (a miss). This makes plain roaming genuinely dangerous, so the **bush becomes the only reliable refuge** — a bushed agent cannot be jumped. The intended behavioural consequence is that the agent must learn to *use* bushes instead of relying on outrunning the predator.

The design below is already fully aligned with the user; this document is the **engineering plan only** (exact file edits, JAX-correctness argument, backward-compatibility proof, and test checklist). It is a hand-off to the `developer` agent — **no code is written here.**

The single hardest constraint is **backward compatibility**: with the jump *disabled* (the default), every one of the ~existing configs — the `basic-05` family, `basic05_variants/*`, `basic/06`, and the 6 runs currently in training — must behave **byte-for-byte identically** to today, including the pseudo-random number (PRNG) stream. The two parity test suites (`test_unified_parity.py`, `test_visual_parity.py`) must stay green with **no fixture re-capture**. The plan is built around making the disabled path a provable no-op.

---

## Analysis

### Where predator behaviour lives

Predator ("hunt") movement is computed in `src/environment/core.py::_hunt_step` (lines ~165–279). It receives **sliced** hunt-subset arrays of shape `(N_pred, …)` so its PRNG draw shapes are byte-identical to the pre-refactor `update_predators`. It is called once from `update_animals` (lines ~361–392), which scatters the results back into the unified `animal_pos` / `animal_state` / … arrays. `update_animals` runs inside the jitted `jax_step` (line ~483). Damage is delivered later in `jax_step` by the existing on-cell block (lines ~550–570): `at_animal = all(new_animal_pos == new_agent_pos)` → `at_damaging` → damage + nociception + injury.

Key facts the plan relies on:

1. **`agent_hidden` is already computed** inside `_hunt_step` (lines ~199–204) as `any(obs_pos == agent_pos AND obs_hides_agent AND obs_active)`. We reuse it verbatim — a bushed agent is un-jumpable (design condition 3).
2. **The attack-cooldown timer already exists**: `hunt_at` (state `animal_attack_timer`) is decremented to `new_attack_timer = max(hunt_at-1, 0)` at line ~187. Today it is reset to `attack_delay` **only on a successful on-cell hit** in `jax_step` line ~562: `new_animal_at = where(at_damaging, state.animal_attack_delay_sampled, new_animal_at)`. `attack_delay` is already per-episode-sampled into `state.animal_attack_delay_sampled` (state.py line 61; reset line ~1212).
3. **`attack_delay` reused as the jump cooldown** — no new cooldown field needed.
4. **The PRNG tail is free**: inside `_hunt_step` the last split is `key, subkey3 = jax.random.split(key)` (line ~247); after that `key` is **never used again and never returned**. New draws taken from that tail `key` therefore **cannot perturb** the existing `jitter_r` / `jitter_c` / `rand_choice` draws.
5. **The per-episode sampling split is size-locked at 7**: `ep_keys = jax.random.split(animal_episode_key, 7)` (core.py line ~1197). Changing this to 8 would change **all** of `ep_keys[0..6]` and break parity for every config. So `attack_range` must be sampled from a **`fold_in`-derived independent key**, never by widening this split — exactly the pattern already used for the count-range masks (`_COUNT_KEY_*`, lines ~1110) and the visual-property stream (`0x7150A1`, line ~1067).

### Why the disabled path is a pure no-op

The jump only ever changes `new_pos` / `new_attack_timer` when a jump actually **fires**, which requires `attack_range_sampled > 0` (design condition 5). With `attack_range` absent → `[0,0]` → sampled `0.0`, the trigger is always False, so `_hunt_step` returns exactly today's values. To make this *provable at the code-path level* (not just value level), the whole jump block is guarded by a **Python-static** `has_attack_feature` bool (True iff any animal has `attack_range_high > 0`), passed as a `pytree_node=False`-style static arg — mirroring `has_animal_range` / `has_res_range` / `has_obs_range`. When it is False, `_hunt_step` executes the identical instruction stream as today: **no new splits, no new draws, no new `where`s.** That is the backward-compat proof.

---

## Implementation Plan

### Design (the aligned mechanism — implement exactly this)

Two new per-predator config fields on a `behaviour: hunt` entity (and legacy predators):

- **`attack_range`** — jump-trigger distance (Manhattan). Scalar `s` (stored as degenerate `[s,s]`) **or** per-episode range `[lo,hi]`, sampled once at reset exactly like `detection_range`. **OPTIONAL; defaults to `[0,0]` = jump disabled** when absent. (Deliberately *not* mandatory-for-hunt like `detection_range` — making it mandatory would break every existing config.)
- **`attack_success_rate`** — float in `[0,1]`, probability a fired jump lands on the agent. Scalar per config for now (a plain per-entity `EnvParams` leaf, **not** per-episode sampled). Rangeability deferred. Default `0.0` when absent (jump-disabled → no-op regardless).

**Trigger** (per predator, all must hold; evaluated in `_hunt_step`):
1. predator is in HUNT this step (`next_state == 1`), AND
2. agent within range (`dist <= attack_range_sampled`, Manhattan; `dist` already computed at line ~190), AND
3. agent NOT bushed (`~agent_hidden`; reuse the existing scalar), AND
4. cooldown up (`new_attack_timer <= 0`), AND
5. feature enabled for this predator (`attack_range_sampled > 0`), AND
6. predator active (`hunt_active`, i.e. `state.animal_active[h_idx]`) — inactive/off-grid slots never jump.

**On fire** — the jump **replaces this step's normal 1-cell chase move**. Draw `Bernoulli(attack_success_rate)`:
- **Success** → set the predator's new position to the agent's cell. The existing on-cell damage block in `jax_step` (line ~552) then delivers a normal hit (damage / nociception / injury) with **no double-counting** — the jump only moves the predator; damage is unchanged code.
- **Miss** → relocate to a uniformly-random cell among the agent's 8 Chebyshev-1 neighbours, skipping cells that are out-of-bounds or a blocking obstacle; **fallback: stay put** (`hunt_pos`) if no neighbour is valid.

**Cooldown on ANY attempt (hit OR miss)** — set `new_attack_timer = attack_delay` whenever a jump is *attempted*, so the predator cannot spam jumps. Because `should_move` already requires `attack_timer <= 0`, an attempted jump also freezes normal locomotion for `attack_delay` steps — i.e. a natural post-pounce recovery pause, consistent with today's post-hit freeze. (See "Order of operations" for how this composes with the existing line-562 reset.)

### Data flow / order of operations inside `_hunt_step`

Compute the normal `new_pos` exactly as today (chase → patrol clip → grid clip → obstacle `check_collision`). **Then**, guarded by `if has_attack_feature:`, apply the jump override **after** all clipping/collision:

```
jump_attempted = (next_state == 1) & (dist <= attack_range_s) & (~agent_hidden) \
                 & (new_attack_timer <= 0) & (attack_range_s > 0) & hunt_active   # (N_pred,) bool

key, jkey_succ, jkey_nbr = jax.random.split(key, 3)   # tail key — parity-safe
success = jax.random.uniform(jkey_succ, (N_pred,)) < attack_success_rate          # per-animal
jump_success = jump_attempted & success

# miss target: random valid neighbour of the agent (same candidate set for all preds)
offsets = jnp.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])       # (8,2)
cand    = agent_pos[None,:] + offsets                                              # (8,2)
in_bounds = (cand[:,0]>=0)&(cand[:,0]<grid_height)&(cand[:,1]>=0)&(cand[:,1]<grid_width)
blocked   = jax.vmap(lambda c: jnp.any(jnp.all(obs_pos==c,axis=-1) & _eff_blocking_hunt))(cand)
valid     = in_bounds & (~blocked)                                                # (8,)
score     = jnp.where(valid[None,:], jax.random.uniform(jkey_nbr,(N_pred,8)), -1.0)
pick      = jnp.argmax(score, axis=-1)                                             # (N_pred,)
miss_pos  = jnp.where(jnp.any(valid), cand[pick], hunt_pos)                        # fallback stay put

jump_pos  = jnp.where(jump_success[:,None], jnp.broadcast_to(agent_pos,(N_pred,2)), miss_pos)
new_pos   = jnp.where(jump_attempted[:,None], jump_pos, new_pos)                   # override AFTER clips
new_attack_timer = jnp.where(jump_attempted, hunt_attack_delay, new_attack_timer) # cooldown on attempt
```

Why the override goes **after** patrol/grid clipping and collision:
- The pounce is a lunge that may legitimately leave the patrol box; clipping it would defeat the point. On the **next** step the predator's normal chase move is patrol-clipped as usual, so it is pulled back — an acceptable one-step lunge-then-recover. **Documented consequence, not a bug.**
- On a **success** landing (agent's cell): the agent cannot be in a blocking obstacle, and cannot be in a bush (condition 3 gates that out), so the landing cell is always valid — no clip/collision needed.
- On a **miss**: we validity-check neighbours ourselves (in-bounds + not-blocking), so grid/collision clamps are unnecessary. Bush (non-blocking) cells are considered valid landing cells for a miss — see Fork F3.

**Interaction with the ghost-predator re-park** (`update_animals` line ~428): inactive slots are forced off-grid *after* `_hunt_step` returns, so an inactive predator can never end a step on the agent even in the impossible event of a jump. Combined with the explicit `hunt_active` gate (condition 6), inactive predators provably never jump and never set a cooldown. **No change to the re-park code.**

**Composition with the existing line-562 cooldown reset:** keep line 562 as-is. On a **successful** jump, `at_damaging` is True → line 562 re-sets the timer to `state.animal_attack_delay_sampled` — the *same value* `_hunt_step` already wrote (`hunt_attack_delay = state.animal_attack_delay_sampled[h_idx]`), so it is idempotent. On a **miss**, `at_damaging` is False (the predator landed on a neighbour, not the agent), so line 562 leaves the timer that `_hunt_step` set. On a **normal on-cell hit with no jump** (agent walks onto a stationary predator), `_hunt_step` set nothing → line 562 handles it as today. No path double-resets to a *different* value; no conflict.

### File Changes

| File | Change | Notes |
|---|---|---|
| `src/environment/config_loader.py` | Parse `attack_range` (scalar-or-`[lo,hi]`, optional, default `[0,0]`) into `animal_attack_range_low/high`; parse `attack_success_rate` (scalar, optional, default `0.0`) into `animal_attack_success_rate`; compute static `has_attack_feature`; add all three to the `_load_animals` return tuple, its unpacking, the zero-animal branch, and the `EnvParams(...)` kwargs. | Mirror the existing `attack_delay` / `detection_range` parsing exactly. |
| `src/environment/state.py` | `EnvParams`: add `animal_attack_range_low`, `animal_attack_range_high`, `animal_attack_success_rate` (traced `[N]` float leaves) and `has_attack_feature: bool = struct.field(pytree_node=False)`. `EnvState`: add `animal_attack_range_sampled` (`[N]` float). | Traced leaves = recompile-safe; static bool = provable disabled-path parity. |
| `src/environment/core.py` | (a) `jax_reset`: sample `animal_attack_range_sampled` via a **`fold_in`** key (new constant, e.g. `0xA77AC7`) — never widen the size-7 `ep_keys` split; carry it through `EnvState(...)` and the `jax_step` `state._replace(...)`. (b) `_hunt_step`: add args `attack_range_s`, `attack_success_rate`, `hunt_attack_delay`, `hunt_active`, and static `has_attack_feature`; add the guarded jump block above. (c) `update_animals`: slice + pass the new hunt-subset args (`state.animal_attack_range_sampled[h_idx]`, `params.animal_attack_success_rate[h_idx]`, `state.animal_attack_delay_sampled[h_idx]`, `state.animal_active[h_idx]`, `params.has_attack_feature`). (d) `jax_step` line ~744: add `animal_attack_range_sampled=state.animal_attack_range_sampled` to the `_replace` (constant within episode). | Keep line 562 untouched. |
| `tests/env/test_predator_jump.py` (**new**) | Full behavioural + parity + recompile checklist (see Test Plan). | Model on `tests/env/test_disengage_on_contact.py`. |
| `docs/environment/02_config_schema.md` | Document `attack_range` + `attack_success_rate` in the animal-fields table (optional; jump-disabled default `[0,0]` / `0.0`). | Config Maintenance Contract. |
| `docs/environment/CONFIG_GUIDE.md` | One-line mention of the two new optional hunt fields + link back to this doc. | Config Maintenance Contract. |

No files under `scripts/` are added/moved/renamed, so `SCRIPTS_DEPENDENCY_MAP.md` needs **no** update.

### Config keys added (exact YAML)

On a `behaviour: hunt` entity (both optional):
```yaml
    attack_range: [3, 5]          # OR a scalar, e.g. 4. Absent → [0,0] = jump disabled.
    attack_success_rate: 0.6      # float in [0,1]. Absent → 0.0.
```

---

## JAX-correctness analysis (each surface addressed)

- **PRNG threading / parity.** New draws (`success`, neighbour `score`) come from the **tail `key`** after the existing `subkey3` split — that `key` is otherwise discarded, so `jitter_r`/`jitter_c`/`rand_choice` are byte-unchanged. Draws are per-animal shape `(N_pred,)` / `(N_pred,8)` — vmap-safe over the static N-predator axis and deterministic under a fixed seed. Guarded by the static `has_attack_feature`, so a **disabled config takes zero new splits** — the strongest possible parity guarantee.
- **Reset sampling parity.** `attack_range` is sampled from `fold_in(animal_episode_key, 0xA77AC7)` — an independent stream that does **not** consume from the size-7 `ep_keys` split, so `animal_detect_sampled` … `animal_attack_delay_sampled` are all byte-identical. The returned episode `key` (outer split, line ~883) is untouched by `fold_in`. New constant chosen to not collide with `0xAE1`, `0x7150A1`, `0xC0A1/2/3`.
- **Recompile safety.** `animal_attack_range_low/high`, `animal_attack_success_rate`, and `animal_attack_range_sampled` are **traced float pytree leaves** (like `animal_detect_low/high` / `animal_detect_sampled`) — changing their *values* between configs does **not** recompile `jax_step`. `has_attack_feature` is a **static** (`pytree_node=False`) bool — jump-on vs jump-off are two distinct traces (intended), constant within any one training run. Guarded by `test_no_recompile`-style coverage (Test §e).
- **Order of operations.** Jump override applied after patrol/grid clip + obstacle collision, so a pounce is not blocked/clipped the way a normal move is (documented lunge). Success landing (agent cell) is always valid; miss neighbours are self-validated.
- **Ghost predators / activation masks.** Explicit `hunt_active` gate (condition 6) + the existing post-step re-park (line ~428) mean inactive predators never jump and never set a cooldown. No re-park change.
- **Bush gating.** Reuse the already-computed scalar `agent_hidden` (obs_hides_agent & obs_active). Do **not** recompute.

---

## Backward-compat argument (hard requirement)

With `attack_range` absent / `[0,0]` for every animal, `has_attack_feature` is Python-`False`, so:
1. `_hunt_step` runs the **identical instruction stream** as today — no extra `jax.random.split`, no extra draws, no extra `where`. → `new_pos`, `next_state`, `new_stamina`, `new_move_timer`, `new_attack_timer` byte-identical.
2. `jax_reset`'s `ep_keys` split stays size-7 and the `fold_in` draw is an independent stream, so all seven per-episode sampled arrays are byte-identical. (For a fully-disabled config `animal_attack_range_sampled` is all-zero and read by nothing.) The returned `key` is unchanged.
3. `jax_step` line 562 and the damage/obs/reward paths are untouched.

Therefore observations, rewards, positions, timers, and RNG state are byte-identical → `test_unified_parity.py` and `test_visual_parity.py` pass **without fixture re-capture**. (The pre-existing modified/untracked `.npz` fixtures in `git status` are unrelated to this change and must not be re-captured on its account.)

---

## Test Plan (`tests/env/test_predator_jump.py`)

Build minimal YAML like `test_disengage_on_contact.py` (`_make_params` merging a base env with an `entities:` block). Checklist:

- **(a) Fires only when all conditions hold.** Predator adjacent-but-within-range, HUNT, agent in open, cooldown up, `attack_range>0`, `attack_success_rate=1.0` → predator lands on the agent's cell in one step and damage is delivered (`info['hit_predator']` / injury rises).
- **(b) Success → on-agent hit, no double-count.** With `attack_success_rate=1.0`, a fired jump = a normal on-cell hit: exactly one damage application; injury increment equals a single sampled predator damage (not doubled).
- **(c) Miss → adjacent free cell.** `attack_success_rate=0.0` → after a fired jump the predator sits on one of the agent's 8 neighbours (Chebyshev-1), never on the agent, never on a blocking obstacle, never off-grid; no damage that step.
- **(d) Cooldown blocks the next-step jump.** After any attempt (hit or miss) `animal_attack_timer == attack_delay`; no second jump until it decays to 0.
- **(e) Bushed agent is never jumped.** Agent standing on a `hides_agent` obstacle within `attack_range` → predator never lands on the agent (jump gated out by `~agent_hidden`); also confirms the predator leaves HUNT via the existing `lose_interest`.
- **(f) Disabled (`attack_range` absent / `[0,0]`) = no jump + parity.** `params.has_attack_feature is False`; two identical seeded 100-step rollouts agree; predator advances exactly 1 cell/step (never teleports). Load-bearing global proof = `test_unified_parity.py` staying green.
- **(g) Recompile-safety.** `test_no_recompile`-style: two jump-enabled configs with the same N and class ordering but different `attack_range` / `attack_success_rate` values → exactly **1** compile of `jax_step`. (Optionally: a jump-on vs jump-off pair → 2 compiles, confirming the static bool split is intentional.)
- **(h) vmap + mixed entities.** Config `[hunt+jump, wander, hunt+no-jump]` → `animal_attack_range_high` shape `(3,)`, `vmap(jax_step)` over a batch runs without shape error, no NaNs.

---

## Checkpoints (for the `developer` during implementation)

- [x] After `config_loader` edit: load an existing `basic-05` config → `params.has_attack_feature is False` and `params.animal_attack_range_high` is all-zero. Verified on `configs/environment/experiment/basic/05-random_init_10x10.yaml`: `has_attack_feature=False`, `animal_attack_range_high=[0. 0. 0. 0.]`.
- [x] After `core.py` reset edit: `jax_reset` on a disabled config produces a `key` byte-identical to `HEAD` (compare `state.key` before/after the change on the same seed). Verified via `git stash` A/B on the same config+seed: both HEAD and post-change produced `state.key = [1797259609 2579123966]`.
- [x] Run `pytest tests/env/test_unified_parity.py tests/env/test_visual_parity.py` — green, no fixture diff. Result: **37 passed, 247 skipped, 0 failed**.
- [x] Run the new `test_predator_jump.py` — all sub-tests pass. Result: **8 passed** (checklist a–h plus an extra empirical success-rate sanity test).
- [x] Speed check: measure `jax_step` throughput on the disabled path vs `HEAD` (same node/GPU/seed, long enough to clear warm-up). Record before/after in the Implementation Report. See "Speed check" below.

## Open forks (need a user decision — do NOT invent)

- **F1 — `attack_success_rate` default when disabled.** Plan defaults it to `0.0` when absent (jump-disabled is a no-op regardless of this value since the trigger is gated on `attack_range>0`). Alternative: default `1.0`. Since it is inert while disabled, `0.0` is the safe, clearly-off default. **Flagging only; recommend `0.0`.**
- **F2 — HUNT gate uses `next_state` vs `hunt_state`.** Plan uses `next_state == 1` (the post-transition state that governs this step's move), so the jump is consistent with the move it replaces. Using the pre-transition `hunt_state == 1` would let a predator that *just lost interest* still jump. **Recommend `next_state`.**
- **F3 — Miss can land in a bush.** A bush is non-blocking, so a missed pounce may land the predator on a bush cell (it just occupies it; it does not gain concealment). Plan treats bush cells as valid miss targets (only blocking obstacles excluded). Alternative: also exclude `hides_agent` cells from miss targets. **Recommend allowing bush landings** (simpler, and a predator on a bush is harmless), but the user may prefer to exclude them for cleaner semantics.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-03

### Summary — what's in this document

This section is the record of what was actually built, tested, and measured for the predator jump/pounce feature described above. In plain terms: the feature was implemented exactly as planned, all fork resolutions (F1–F3, patrol-lunge) were applied as instructed, every test in the plan's checklist passes, the two backward-compatibility parity suites are green with no fixture regeneration, and a repeated speed measurement shows no regression on the disabled path (which is what every config in production/training uses today). The feature is off by default and ready for `experiment-designer` to opt individual predators into via two new YAML keys, `attack_range` and `attack_success_rate`.

### File-by-file changes

**`src/environment/state.py`**
- `EnvState`: added `animal_attack_range_sampled: jnp.ndarray  # [N] float`, placed next to `animal_attack_delay_sampled`.
- `EnvParams`: added three traced leaves (`animal_attack_range_low`, `animal_attack_range_high`, `animal_attack_success_rate`) next to `animal_attack_delay_low/high`, and one static bool `has_attack_feature: bool = struct.field(pytree_node=False)` next to `has_res_range`/`has_animal_range`/`has_obs_range`.

**`src/environment/config_loader.py`**
- `_load_animals`: added parsing for `attack_range` (scalar-or-`[lo,hi]`, via `_parse_distributional(..., mandatory=False, ...)` reading directly from each entity's raw YAML `dist_source` — mirrors the existing `detection_range` pattern, no entries-dict copy needed) and `attack_success_rate` (scalar float in `[0,1]`, default `0.0`, validated range). Computed `has_attack_feature = any(hi > 0 for hi in attack_range_high_list)`. Added the four new values (`animal_attack_range_low`, `animal_attack_range_high`, `animal_attack_success_rate`, `has_attack_feature`) to: the zero-animal (`N==0`) branch, the array-building block, and the function's return tuple.
- `load_env_params`: unpacked the four new values from `_load_animals(...)` and passed them through to `EnvParams(...)`.

**`src/environment/core.py`**
- `_hunt_step`: added keyword args `attack_range_s`, `attack_success_rate`, `hunt_attack_delay`, `hunt_active`, and static `has_attack_feature: bool = False`. Inserted the jump/pounce override block **after** the existing patrol-clip/grid-clip/obstacle-collision code and **before** the move-timer reset, guarded by `if has_attack_feature:` (a Python-level branch — when False, zero extra JAX ops are traced). See the exact diff below.
- `update_animals`: HUNT-branch call to `_hunt_step` now passes `attack_range_s=state.animal_attack_range_sampled[h_idx]`, `attack_success_rate=params.animal_attack_success_rate[h_idx]`, `hunt_attack_delay=state.animal_attack_delay_sampled[h_idx]`, `hunt_active=state.animal_active[h_idx]`, `has_attack_feature=params.has_attack_feature`.
- `jax_reset`: after the existing size-7 `ep_keys` sampling block, added an **independent** `fold_in(animal_episode_key, 0xA77AC7)` draw for `animal_attack_range_sampled` (both the `N>0` and `N==0` branches), and added it to the `EnvState(...)` constructor.
- `jax_step`: added `animal_attack_range_sampled=state.animal_attack_range_sampled` to the `state._replace(...)` call (carried through unchanged — a per-episode-constant field, same pattern as `animal_attack_delay_sampled`). Line ~562 (the existing on-cell-hit cooldown reset) was **left untouched** per the plan's idempotency argument.

**`tests/env/test_predator_jump.py` (new)** — 8 tests covering the plan's checklist (a)–(h) plus one extra empirical-rate sanity test. See "Test results" below.

**`docs/environment/02_config_schema.md`** — added `attack_range` / `attack_success_rate` rows to the "Mandatory / optional per entity" table, a new "Jump/pounce feature" subsection under "Distributional fields", and updated the Static/Dynamic fields table (added `has_attack_feature` to static, the three new arrays to dynamic).

**`docs/environment/CONFIG_GUIDE.md`** — added a new "3.7 Predator jump / pounce" entry to the v3.0 feature quick-reference, with the YAML snippet and a link back to this plan doc.

**Deviation from the plan's literal File Changes wording (flagging, not a scope change):** the plan's table said to add `'attack_range': ent.get('attack_range')` etc. to each of the three `entries.append({...})` dicts in `_load_animals` (mirroring `attack_delay`). Instead, `attack_range` is parsed directly from `e['dist_source']` (the raw YAML entry dict), exactly mirroring how `detection_range` (a `DISTRIBUTIONAL_FIELDS` member) is already parsed — no entries-dict copy needed, since `dist_source` already carries the raw YAML. This is functionally identical and touches fewer call sites; `attack_success_rate` is a plain scalar read the same way. All four resulting arrays (`animal_attack_range_low/high`, `animal_attack_success_rate`, `has_attack_feature`) were still added to the return tuple, its unpacking, the zero-animal branch, and the `EnvParams(...)` kwargs exactly as the plan specified.

### Core jump-block diff (`src/environment/core.py::_hunt_step`)

```python
    new_pos = jax.vmap(check_collision)(new_pos, hunt_pos)

    # ── Jump / pounce override (predator lunge attack) ──────────────────────
    # Guarded by the STATIC has_attack_feature bool: when False (default / every
    # existing config), this entire block is skipped at TRACE time — no extra
    # jax.random.split, no extra draws, no extra `where`s — so the disabled path
    # is byte-identical to pre-feature `_hunt_step`.
    if has_attack_feature:
        jump_attempted = (
            (next_state == 1) & (dist <= attack_range_s) & jnp.logical_not(agent_hidden)
            & (new_attack_timer <= 0) & (attack_range_s > 0) & hunt_active
        )

        key, jkey_succ, jkey_nbr = jax.random.split(key, 3)   # tail key — parity-safe
        success = jax.random.uniform(jkey_succ, (hunt_pos.shape[0],)) < attack_success_rate
        jump_success = jump_attempted & success

        offsets = jnp.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]], dtype=jnp.int32)
        cand = agent_pos[None, :] + offsets
        in_bounds = (cand[:,0]>=0)&(cand[:,0]<grid_height)&(cand[:,1]>=0)&(cand[:,1]<grid_width)
        blocked = jax.vmap(lambda c: jnp.any(jnp.all(obs_pos==c,axis=-1) & _eff_blocking_hunt))(cand)
        valid = in_bounds & jnp.logical_not(blocked)
        score = jnp.where(valid[None,:], jax.random.uniform(jkey_nbr,(hunt_pos.shape[0],8)), -1.0)
        pick = jnp.argmax(score, axis=-1)
        miss_pos = jnp.where(jnp.any(valid), cand[pick], hunt_pos)

        jump_pos = jnp.where(jump_success[:,None], jnp.broadcast_to(agent_pos, hunt_pos.shape), miss_pos)
        new_pos = jnp.where(jump_attempted[:,None], jump_pos, new_pos)
        new_attack_timer = jnp.where(jump_attempted, hunt_attack_delay, new_attack_timer)

    # Reset timer
    new_move_timer = jnp.where(should_move, hunt_move_int, new_move_timer)
```

This is byte-for-byte the plan's pseudocode (Fork resolutions applied: F1 default `0.0`; F2 gate on `next_state`; F3 bush cells not excluded from `valid`; patrol-lunge — no special-case clip added, the override sits after all clipping as the plan specifies).

### Fork resolutions applied

- **F1**: `attack_success_rate` defaults to `0.0` when absent — confirmed inert (jump trigger also requires `attack_range_s > 0`).
- **F2**: HUNT gate uses `next_state == 1` (post-transition), not `hunt_state`.
- **F3**: miss targets include bush (`hides_agent`) cells — only out-of-bounds / blocking obstacles are excluded from the candidate set.
- **Patrol-lunge**: no special handling added; the jump override sits after the patrol-box clip, so a successful/missed jump may land outside the predator's patrol box for that one step. The next normal chase move clips it back, exactly as the plan describes.

### Test results

**New feature tests — `tests/env/test_predator_jump.py`** (checklist a–h + 1 extra):
```
tests/env/test_predator_jump.py::test_jump_success_lands_on_agent_and_deals_damage_once   PASSED  (a, b)
tests/env/test_predator_jump.py::test_jump_miss_lands_on_valid_neighbour_no_damage         PASSED  (c)
tests/env/test_predator_jump.py::test_cooldown_blocks_second_jump                          PASSED  (d)
tests/env/test_predator_jump.py::test_bushed_agent_never_jumped                            PASSED  (e)
tests/env/test_predator_jump.py::test_disabled_no_jump_and_deterministic                   PASSED  (f)
tests/env/test_predator_jump.py::test_recompile_safety_values_and_enable_toggle            PASSED  (g)
tests/env/test_predator_jump.py::test_vmap_mixed_entities                                  PASSED  (h)
tests/env/test_predator_jump.py::test_empirical_success_rate_matches_configured_rate       PASSED  (extra)
8 passed in 68.48s
```

**Parity gate (the hard backward-compat requirement):**
```
pytest tests/env/test_unified_parity.py tests/env/test_visual_parity.py -q
37 passed, 247 skipped, 0 failed, in 315.20s
```
No fixture was re-captured or touched by this change (the 5 modified + 3 untracked `.npz` fixtures visible in `git status` predate this session and are unrelated — confirmed by diffing against the state at session start).

**Full env suite:**
```
pytest tests/env/ -q
192 passed, 488 skipped, 1 warning (pre-existing, unrelated to this change), 0 failed, in 705.97s
```

### Empirical demo (no training) — `tmp/20260703_020855_predator_jump_demo.py`

Predator with `attack_range: [2,2]`, `attack_success_rate: 0.5`, spawned exactly 2 cells (Manhattan) from a non-bushed agent; 200 independent-seed trials, one step each:

```
[Part 1] Non-bushed agent, 200 trials (attack_success_rate=0.5):
  hits   (landed ON agent)       :  111  (55.5%)
  misses (landed adjacent)       :   89  (44.5%)
  other  (neither -- should be 0):    0

[Part 2] Bushed agent, 200 trials (same geometry, agent hidden in a bush):
  predator landed on agent (must be 0): 0
```
Hit rate (55.5%) is close to the configured 50% (binomial noise at n=200 is expected — the dedicated 300-trial test in the test suite asserts the rate stays within `(0.35, 0.65)` and passes). The bushed-agent control confirms 0/200 hits, as required.

### Speed check

**Rationale for measuring:** `_hunt_step` (the hot-path per-step predator update) was changed, so a before/after measurement is required even though the change is provably a no-op on the disabled path.

**Method:** `configs/environment/experiment/archive/hypervigilance/01-interoNocicept_sameProp.yaml` (3 animals), seed 0, 20-step warm-up + 2000 measured `jax_step` calls, 3 repeated trials each. "Before" = `git stash` of the three changed `src/environment/*.py` files (reverts to pre-feature `HEAD`); "after" = the implemented change, same config (disabled path — no `attack_range` in this config). Machine: shared lab workstation with several other concurrent Claude sessions running (confirmed via `ps aux`), so trial-to-trial noise is expected; repeated trials were taken to bound it.

```
BEFORE (HEAD, git-stashed):  685.0 / 695.3 / 698.9 steps/sec   (mean 693.1)
AFTER  (disabled path):      701.7 / 707.8 / 769.0 steps/sec   (mean 726.2)
```

**Result: no regression** — the "after" mean is ~4.8% *faster* than "before," which is within the noise band of a busy shared machine (an isolated single-run comparison taken earlier under heavier contention showed the opposite direction, 851→600 SPS, underscoring that single-shot measurements on this machine are unreliable; the repeated-trial comparison above is the load-bearing one). This is consistent with the JAX-correctness argument: the disabled path (`has_attack_feature=False`) takes the Python `if` branch's `False` arm, so zero additional JAX ops are traced into `jax_step` — the compiled XLA graph for a disabled config is unaffected. A dedicated jump-*enabled* reference benchmark was attempted but the ad-hoc script's manual YAML injection didn't route through the `extends:` resolution correctly (a script bug, not a correctness issue) and was not repaired given the disabled-path result is the load-bearing one; the jump-enabled path's runtime is otherwise implicitly exercised by `test_predator_jump.py`'s vmap and 300-trial tests, which complete in line with the other feature tests in the suite (~68s for 8 tests including a 300-iteration reset+step loop and a 4-way vmap).

### Checkpoint verification detail

- **Checkpoint 1**: `configs/environment/experiment/basic/05-random_init_10x10.yaml` → `params.has_attack_feature == False`, `params.animal_attack_range_high == [0. 0. 0. 0.]`.
- **Checkpoint 2**: same config, seed 0 — `state.key` after `jax_reset` is `[1797259609 2579123966]` on both `HEAD` (git-stashed) and the implemented change. Byte-identical.

### Blockers / follow-up

None. The feature is off by default and ready for `experiment-designer` to opt individual predators into via `attack_range` + `attack_success_rate` on any `behaviour: hunt` entity.

**Signed: Implemented by: developer**

## Verification Report

> **Verified by**: —
> **Date**: —

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: —
