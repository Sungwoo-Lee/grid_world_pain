# Diagnosis findings — src/environment/core.py + src/environment/state.py (2026-07-23)

Reviewer unit: environment step/reset/termination/reward/entity logic + state dataclasses.
Method: full read of both files, call-in verification into wrapper.py / sensor.py / recurrent_ppo_trainer.py, and read-only PRNG checks with the project interpreter (JAX 0.9.0.1).
Known-bugs context honored: OPEN rows not re-reported; LATENT rows verified with added depth; FIXED rows regression-checked.

---

## Findings

### F1 [P2, verified LATENT row] core.py:117-126 + 702-712 — `overeating_death` never ends the episode; `termination_reason=3` is stamped on non-terminal steps

**Claim.** `update_body` has no satiation-at-max death check, so with `overeating_death: true` the episode never terminates from overeating; meanwhile `jax_step` stamps `reason=3` whenever `new_satiation >= max_satiation`, i.e. on ANY step where the agent is merely full — with `done=False`.

**Failure scenario.** Config sets `overeating_death: true`. Agent eats to full nutrition -> `satiation == max_satiation` (the power-law mapping hits max exactly at full nutrition) -> `info['termination_reason'] == 3` for many consecutive LIVE steps. Any consumer treating `reason != 0` as episode-over (or logging reason at arbitrary steps) mislabels; the promised death never comes.

**Evidence.** `update_body` (lines 117-126) sets `done` only from `new_nutrition <= 0` (if with_nutrition), `new_injury >= max_injury` (if with_injury), or `damage > 0` (no-injury instant death). No `overeating_death` branch. `jax_step` lines 708-709: `reason = where(new_satiation >= max_satiation, 3, reason)` gated only on the static flag, not on any death event. The rPPO trainer explicitly guards this quirk (`recurrent_ppo_trainer.py:63,81-82` — "overeating sets termination_reason=3 without setting done"; real-death mask = done AND terminated), confirming the quirk is real and load-bearing.

**Status/severity.** Verified (was LATENT "needs verification"). No config in `configs/` currently sets `overeating_death: true` (grep clean), so P2 latent; becomes P1 the day someone flips the flag expecting a death mode.

**Fix direction.** Either add `done |= (overeating_death & satiation >= max)` in `update_body` and stamp reason 3 only on that event, or remove the flag + reason-3 code path entirely.

### F2 [P2, verified LATENT row — with new downstream depth] core.py:124-126 + 704-712 -> recurrent_ppo_trainer.py:373/386 — instant death (no-injury configs) leaves `termination_reason < 2`, so the trainer's real-death mask classifies the death as a timeout

**Claim.** In configs with `with_injury: false`, any positive damage kills instantly (`done = damage > 0`, core.py:126), but no reason code >= 2 is ever stamped for that path — `reason` stays 0 (or 1 if the same step truncates). The rPPO trainer derives real-death flags as `termination_reason >= 2` (`recurrent_ppo_trainer.py:373,386`), so an instant death is treated as a TRUNCATION: the GAE/MC bootstrap across the death is RETAINED while the death penalty is still applied to the reward — inconsistent value targets.

**Failure scenario.** Config: `with_injury: false`, one predator, extrinsic reward. Predator contacts agent at step t -> `done=True`, `reward -= death_penalty`, `reason=0`. Trainer sees done with `terminateds=0` -> bootstraps V(s_{t+1}) past a real death. The value function is trained toward "penalty + continuing value" — systematically wrong at exactly the events the task is about.

**Evidence.** Reason codes assigned only for truncation (1), starvation (2, gated with_nutrition), overeating (3, gated flag), injury (4). The `damage > 0` instant-death branch (line 126) has no corresponding reason assignment. Trainer mapping verified by grep.

**Status/severity.** Verified depth on LATENT row "termination-reason unreliable when a body system is off". No current non-archive config has `with_injury: false` (grep — only archive/), so P2 latent; P1 if no-injury sandbox configs are revived.

**Fix direction.** Add a reason code (e.g. 5 = instant-death) in the no-injury branch, or stamp reason 4 when `damage > 0 & ~with_injury`; keep trainer mapping `>= 2`.

Related note (same row): `reason=4` (line 710) is not gated on `with_injury`; with `with_injury: false` + `random_start_injury` bounds reaching `max_injury`, a live agent can carry `reason=4` while not dying. Same root cause: reason codes are computed from raw state comparisons, not from the actual `done` events.

### F3 [P2] core.py:571-573, 622-624, 643-645 — `damage_key` reused for three independent damage draws; cross-source damage values are bit-identical index-wise

**Claim.** The same `damage_key` (from the 6-way split at line 505) is passed to `jax.random.uniform` three times — resource damage (571), animal damage (622), obstacle damage (643). JAX PRNG is counter-based, so draw i of each array consumes identical bits: uniform quantiles are equal index-wise across the three entity classes (verified empirically: same key + same shape -> allclose exactly; different lengths share the common prefix).

**Failure scenario.** Config with hiding predator slot 0 and animal slot 0 sharing the same [min,max] damage range: on any step where both hit the agent, the two sampled damages are IDENTICAL numbers — doubling exactly instead of varying independently. Within-step damage randomness across sources is perfectly rank-correlated in general.

**Evidence.** Read of the three `jax.random.uniform(damage_key, ...)` calls; Python check with the project interpreter confirmed bit-identity.

**Severity.** P2 — marginals unaffected, cross-source co-occurrence rare, but genuine key reuse.

**Fix direction.** Split `damage_key` into three subkeys (changes the PRNG stream — coordinate with parity fixtures).

### F4 [P2] core.py:865-885 (`resolve_overlaps_global`) — when an entity's spawn area has no free cell, the entity silently teleports to grid cell (0,0)

**Claim.** If `is_taken` is True and no valid replacement exists (`valid` all-False), `first_valid_mask` is all-False, so `replacement_flat = jnp.where(all-False, perm, 0).sum() = 0` — the entity is placed at flat cell 0 = (0,0), outside its spawn area, and multiple starved entities stack there.

**Failure scenario.** A 2x2 spawn area shared by 5 entities in per_entity mode: the 5th entity finds its sampled cell taken and every area cell occupied -> lands at (0,0). If (0,0) is the agent start or another type's zone, entity semantics (e.g. a predator pinned at the corner, outside its patrol) silently break.

**Evidence.** Lines 876-879: `first_valid_mask = valid_in_perm & (cumsum == 1)`; sum over an all-False mask is 0; `new_flat = where(is_taken, 0, flat_idx)`; a second starved entity repeats the same fallback (its `valid` is also empty).

**Severity.** P2 — only fires on over-packed spawn areas; nothing validates capacity, and the failure is silent.

**Fix direction.** Park starved entities off-grid (as the count-range feature already does) or assert capacity at config-load time.

### F5 [P2] core.py:894-941 + 1039-1058 (`place_in_area`, per_type mode) — when valid cells < requested count, surplus entities are silently parked off-grid at (height, 0)

**Claim.** `selected_flat` pads unfilled selections with sentinel `total_cells`; after sort + slice, slots with index < `num_entities` that got no real cell keep the sentinel, whose decoded position is `(total_cells // width, 0) = (height, 0)` — off-grid. `place_type_group` then writes that position for every `valid[j]` slot (valid = `arange < count`, not "got a cell"), so the entity exists in state but is invisible/uninteractable all episode.

**Failure scenario.** per_type config asks for 6 food items in a 2x2 area: 4 are placed, 2 sit at (height, 0) forever — active resources the agent can never reach or see; effective food density silently below the configured value; survival results wrong for that config.

**Evidence.** Lines 930-936 (sentinel + `where(arange < num_entities, selected_flat, 0)` — the 0 padding applies only to slots >= num_entities, not to starved slots below it), lines 1048-1058 (scatter gated on `valid[j] = j < count`).

**Severity.** P2 — config-dependent, silent; no capacity validation exists.

**Fix direction.** Validate area capacity >= count at config-load; or mark starved slots inactive via the existing activation-mask machinery.

### F6 [P2] core.py:968-970 (`jax_reset`) — agent placement ignores all entity occupancy

**Claim.** With `random_start_pos: true` (the default in `configs/environment/default.yaml`), the agent position is drawn uniformly over the whole grid independently of entity placement, and the entity overlap-resolve never considers the agent cell. The agent can spawn on a hiding-predator resource (auto-damage on step 1), on food (free meal), inside a blocking obstacle, or in a bush (spawn-concealed).

**Failure scenario.** 10x10 grid, 3 hiding predators -> ~3% of episodes begin with an unavoidable damage tick at step 1, adding injury-onset variance uncorrelated with behaviour. This is the same mechanism behind the FIXED "observability gate fires at step 0" bug (Finding G2, commit 84014e4) — that fix pinned the start position in one diagnosis config; the mechanism itself is untouched and live in every random-start config.

**Evidence.** Lines 968-970: `random_pos = randint(...)`; `agent_pos = where(random_start_pos, random_pos, start_pos)`; no occupancy check anywhere afterward.

**Fix direction.** Include the agent cell in the occupancy mask (or resample agent position avoiding occupied cells) at reset.

### F7 [P2] core.py:517-541 + 556 — step-time resource respawn skips overlap resolution

**Claim.** A respawning resource samples a fresh position from its spawn area (lines 519-524) with no occupancy check: it can land on a blocking obstacle's cell (agent can never enter -> resource unreachable while active), on another resource's cell, or directly under the agent (instant auto-eat the same step, since `at_resource` uses `res_pos_after_reg`, line 556).

**Failure scenario.** Sparse-food config whose food spawn area overlaps rock positions: over a long episode a food item respawns onto a rock and is permanently unreachable — the effective food budget quietly shrinks and the agent starves "for no reason" in a fraction of episodes.

**Evidence.** `sample_res_pos` vmap (519-524) uses only `res_spawn_area`; contrast with `jax_reset`, which runs `resolve_overlaps_global` for initial placement.

**Fix direction.** Mask occupied cells when sampling the respawn position.

### F8 [P2] core.py:756-786 — `dist_per_animal` / `dist_per_predator` / `dist_per_neutral` info fields are not masked by `animal_active`

**Claim.** The aggregated `dist_to_pred` / `dist_to_neutral` correctly mask inactive slots to 99.0 (lines 765-766, 773-774), but the per-entity arrays exported in the same info dict include parked slots at the off-grid sentinel (height, width) — a FINITE, small distance on typical grids (agent at (9,9) on a 10x10 grid is distance ~1.4 from the sentinel).

**Failure scenario.** Count-range config (`has_animal_range=True`) with 2-4 predators allocated: behaviour metrics / accumulators consuming `dist_per_predator` see a phantom "predator" hovering just past the bottom-right corner, biasing mean-distance and approach/flee statistics precisely in count-varied experiments.

**Evidence.** Lines 757-761 compute `dist_per_animal` from raw `state.animal_pos` (which holds the sentinel for inactive slots, re-parked each step at line 495) with no `animal_active` mask; the per-class slices (767, 775) inherit this.

**Severity.** P2 — stats/analysis only (reward and termination unaffected), but it corrupts exactly the behaviour metrics the project reports.

**Fix direction.** `jnp.where(state.animal_active, dist_per_animal, 99.0)` before slicing per-class arrays.

### F9 [P2] state.py:86 / core.py:717+837 — `EnvState.terminated` actually stores `done | truncated` (death OR timeout), contradicting its name

**Claim.** Line 717 merges truncation into `done`, and line 837 stores that merged flag into `state.terminated`. In post-Finding-B vocabulary (real death vs timeout), the field name is a trap: any future consumer reading `state.terminated` as "real death" reintroduces the survival-timeout-punished-as-death class of bug. Current consumers avoid it (trainer uses `info['termination_reason']`; grep found no live `.terminated` readers in src), so this is a naming/latent hazard only.

**Fix direction.** Rename to `episode_done`, or store `real_death` in the field and derive episode-end where needed.

---

## Verified-latent recap (from the known-bugs LATENT section)

- "Over-eating never ends episode" — VERIFIED TRUE (F1).
- "Termination-reason unreliable when a body system is off" — VERIFIED TRUE, with new downstream GAE depth (F2).
- "auto_reset_step() ignores its `key` arg" — VERIFIED TRUE in `src/environment/wrapper.py:35-46` (outside this unit's two files): the `key` parameter is never used; `reset_key` is derived from `state.key`, and because JAX splits are counter-based, `split(state.key, 2)[0] == split(state.key, 6)[0]` — i.e. `reset_key` equals exactly the carried key `jax_step` just stored in `next_state.key`, so the reset episode's stream is correlated with the would-be continuation stream. The live rPPO trainer no longer uses this helper (commit b8eb286 fixed the trainer-side draw); `wrapper.auto_reset_step` appears to be dead-but-exported code. Recommend deletion or fixing the key threading.

## Fixed-bug regression check — all PASS

- Ghost predators stuck to agent: re-park block present at core.py:483-495 (`jnp.where(animal_active[:, None], new_pos, off_grid)` after all subset moves). Not regressed.
- Dead resource slots revived: `res_allocated` gate present in `update_resources` (core.py:130-163), threaded from `jax_step` (510-513), set once at reset (1321-1325), carried unmutated (805). Not regressed.
- Survival-timeout punished as death: `real_death` captured before the truncation merge (696), death penalty gated on `real_death` in both reward branches (734, 738); truncation merged into `done` only afterward (717). Not regressed.
- Observability gate at step 0 (G2): fix was config-level (pinned start position, commit 84014e4) — nothing in core.py to regress; the underlying mechanism remains and is recorded here as F6.
- Fractional attack/detection range inclusive-integer fix: `randint(lo, hi+1)` inclusive sampling present for detect (1280-1281), move_int (1292-1293), attack_delay (1294-1295), attack_range (1302-1303). Not regressed.
- "Chasing rabbit" glued to agent (OPEN, core.py:629): root-cause confirmed as recorded — `new_animal_at = where(at_damaging, attack_delay_sampled, ...)` gates the post-contact pause on `animal_is_damaging`, so non-damaging contact animals never pause and `disengage_on_contact` (635-637) is the only opt-out. No new depth to add.

## Reviewed but clean

- `update_body` nutrition/satiation/injury dynamics: decay, eat gain, power-law satiation, instant-start smoothing buffer roll (`temp_buffer[0]` applied now, rolled out correctly), rest-streak exponential recovery, clip-then-check-death ordering (clip floor 0.0 does not mask the `<= 0` death check).
- Truncation off-by-one: `next_step >= max_steps` with `current_step` starting at 0 gives exactly `max_steps` steps; consistent.
- `_hunt_step` / `_wander_step` PRNG threading: keys split and consumed linearly; jump block draws only from the otherwise-discarded tail key under the static `has_attack_feature` guard (trace-time skip verified — Python `if` on a `pytree_node=False` bool is JIT-safe).
- Jump/pounce block: attempt gating (`hunt_active`, `agent_hidden`, cooldown), miss-target selection (valid-neighbour argmax over per-predator uniforms, stay-put fallback), cooldown on any attempt — all consistent with the documented design.
- `update_animals` scatter/gather via static index tuples (`hunt_idx`/`wander_idx`): shapes and argument order verified against `_hunt_step`/`_wander_step` signatures; static Python `if len(...) > 0` branches are trace-safe (tuple lengths are static).
- Inactive-slot damage gating: `at_damaging`, `at_neutral_pre`, obstacle overlap/collision damage, and `collision_noc` all AND with the relevant `*_active` masks; parked sentinel (h, w) cannot equal any in-grid agent position, so `disengage_on_contact` (ungated by active) is still safe.
- Resource lifecycle: deactivate-only-if-active guard, reg-timer set/decrement/respawn sequencing (delay=k -> exactly k inactive steps), consumption-count reset on respawn.
- Eat-action semantics: `eat_action_idx = where(rest_enabled, 5, 4)`, auto-eat vs action-eat, lifecycle vs satiation flags mutually consistent.
- Reward path: homeostatic drive delta, per-branch death penalty, eating penalty — consistent with Finding-B semantics.
- `jax_reset` PRNG layout: 5-way outer split, fold_in constants (0xAE1, 0x7150A1, 0xC0A1-3, 0xA77AC7) distinct and collision-safe against small split counters; count-range activation masks (`_cumcount` scan, first-K-per-entry semantics) correct; degenerate-range parity guards (static `has_*_range` bools) correct.
- `jax.random.split(key, 0)` for zero-entity configs in `jax_step` (line 517 has no zero guard, unlike `jax_reset`): verified non-crashing on JAX 0.9.0.1 (returns shape (0,2)).
- Dtypes/overflow: int32 counters safe at any realistic horizon; `move_timer` can go negative while attack-frozen but is bounded by timer reset on next move.
- `in_zone` (core.py:205-208) is computed but unused — pre-existing dead code kept for parity; not a bug, mention only.
