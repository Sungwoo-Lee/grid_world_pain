---
title: "Independent diagnosis — environment core (step/reward/termination, entities, sensors, reset boundary)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Environment-core correctness diagnosis (independent bug hunt, 2026-07-04)

## Purpose (plain-language entry point)

This is an independent correctness sweep of the simulated grid-world's **core engine** — the code
that moves the agent and animals each step, computes hunger/injury and reward, decides when an
episode ends, assembles what the agent senses, and restarts episodes when they finish. A prior
audit already found and fixed a cluster of bugs here (most famously: surviving to the time limit
was punished as if the agent had died). This document does two things: (1) **verifies that five of
those fixes really landed and are complete**, and (2) hunts for **new** problems the earlier audit
missed.

Headline results: all five prior fixes are **correct and complete** in the live code. The most
significant new finding is a **random-number key-reuse bug in the live training loop's episode
auto-restart** — the key used to initialize a fresh episode is byte-identical to the master key
that drives all of the next step's randomness, so "how a new episode is laid out" and "what the
agents do / what the world rolls next" are systematically entangled (reproducibility is unharmed;
statistical independence is not). A second new finding: the three per-step damage rolls (traps,
animals, rocks) share one key, so their magnitudes are perfectly rank-correlated within a step.
I also confirmed two known-latent termination bugs are real and — since the recent timeout fix
made trainers *trust* the termination-reason code — they are now one config flag away from
corrupting training rather than just mislabeling statistics.

Scope reviewed: `src/environment/core.py`, `state.py`, `sensor.py`, `grid_world.py`, `wrapper.py`,
plus the trainer-side auto-reset boundary (`src/models/recurrent_ppo_trainer.py`) because the
auto-reset semantics named in the hunt list live there, not in the environment package.

---

## Part 1 — Verification of prior fixes (all five: VERIFIED)

### V1. Timeout no longer punished as death (`ef0fd25`) — VERIFIED, complete
`core.py:696` captures `real_death` (update_body's done: starvation / injury / instant damage)
**before** merging with `truncated` at `core.py:716`; the death penalty is gated on `real_death`
only (`core.py:733`, `core.py:737`). Edge cases checked:
- **Simultaneous death + timeout**: the reason codes (`core.py:704-709`) let death codes 2/3/4
  overwrite the timeout code 1, and the penalty fires via `real_death` — resolves to death,
  which is correct.
- `state.terminated` still stores the *merged* flag (death OR timeout), but its only consumer in
  the whole repo is the parity-fixture generator — no live desync risk.

### V2. Ghost predators parked off-grid (`3634887`) — VERIFIED, complete; one small residue
`core.py:494-495` re-parks inactive animal slots at the off-grid sentinel `(height, width)` after
every movement update. All harm/sense gates check `animal_active`: damage (`core.py:620`),
extero-nociception (`sensor.py:78`), olfaction (`sensor.py:323`), vision (`sensor.py:211` via
`all_active` mask), and `dist_to_pred`/`dist_to_neutral` mins (`core.py:764`, `core.py:772`).
Residue → Finding N5 (unmasked per-entity distance arrays).

### V3. Dead resource slots stay dead (`db8bd03`) — VERIFIED, complete
`res_allocated` is set once at reset (`core.py:1324`), carried through untouched
(`core.py:804`), and gates respawn (`core.py:156-159`). Inactive slots can no longer revive.

### V4. Inclusive-integer `attack_range` / `detection_range` sampling (`7ff8d1f`) — VERIFIED
Both sample as `randint(lo, hi+1)` (`core.py:1279-1280`, `core.py:1301-1302`); the comparisons
`dist <= hunt_detect` (`core.py:219`) and `dist <= attack_range_s` (`core.py:293`) are now
int-vs-int. Nit: `state.py:54` and `state.py:64` comments still say `[N] float` for
`animal_detect_sampled` / `animal_attack_range_sampled` — stale, they are int.

### V5. Observability gate step-0 contact (`84014e4`) — VERIFIED (config-scoped)
Commit touches exactly the four verification-gate YAMLs (`random_start_pos: false`). Note the
general mechanism remains: any config with `random_start_pos: true` samples the agent start
uniformly over the whole grid with no occupancy check (`core.py:968`), so a training agent can
still spawn on a predator/food and take step-0 contact. That is accepted randomness, not a
regression — recorded here for awareness only.

---

## Part 2 — New findings

### N1. [Med] Auto-reset key aliasing in the live rPPO rollout — reset stream == next step's master stream — NEW
**Where**: `src/models/recurrent_ppo_trainer.py:216` (trainer side of the env auto-reset boundary).

**What happens**: the rollout scan derives the episode-reset key as
`reset_key, _ = jax.random.split(key)` but then carries `key` forward **unchanged**. Next
iteration begins with `key, act_key = jax.random.split(key)` — the same `split` of the same
input. Therefore `reset_key(t)` is **byte-identical** to the rollout's master key at step t+1
(verified empirically in the project conda env: `array_equal == True`).

Because JAX's `split(k, n)[i]` is identical for every n (counter-prefix property — also verified:
`split(k,2)[0] == split(k,64)[0]`), the aliasing cascades:
- env 0's per-episode reset key at step t == `split(reset_key, 2)[0]` == the master key at t+2's
  ancestor; env **1**'s reset key == `act_key(t+1)` — the very key that samples all envs' actions
  next step;
- a resetting env's fresh `state.key` (`jax_reset`'s `split(key,5)[0]`) == the trainer master key
  two steps later, whose subsequent 6-way step split collides key-for-key with `jax_reset`'s
  5-way split (respawn key == agent-placement key, etc.).

**Concrete failure scenario**: every time any env finishes an episode, the layout of its new
episode (agent start, entity placement, body randomization) is drawn from the exact keys that
generate the next steps' action sampling and env events for *all* envs. Same-key-different-draw
reuse produces correlated uniforms (verified: `uniform(k,(3,)) == uniform(k,(5,))[:3]`), so e.g.
"where env 1's food spawned" is statistically entangled with "what actions the policy samples one
step later". Determinism/reproducibility is unaffected; independence assumptions are not.

**Contrast**: the sibling trainers do this correctly — `ppo_trainer.py:163` and
`dreamer_v3_trainer.py:652` advance the key (`reset_key, key = split(key)` /
`current_key, reset_key = split(current_key)`). The **live** trainer (rPPO) is the odd one out.

**Suggested fix**: `key, reset_key = jax.random.split(key)` at `recurrent_ppo_trainer.py:216`.
Warning: this shifts the whole PRNG stream — golden/parity fixtures and any byte-reproduction
tests will need regeneration; flag as a deliberate stream break.

### N2. [Low–Med] One `damage_key` shared by three independent damage rolls — NEW
**Where**: `core.py:571` (hiding-predator/resource damage), `core.py:622` (animal damage),
`core.py:643` (obstacle damage) — all three `jax.random.uniform` calls use the same `damage_key`.

**What happens**: uniform draws of different shapes from the same key share their leading values
(verified). So within a step, the i-th resource damage roll, the i-th animal damage roll, and the
i-th obstacle damage roll are the *same* underlying uniform scaled to different `[min,max]`
ranges — perfectly rank-correlated.

**Concrete failure scenario**: an agent simultaneously bumped by predator 0 and standing on
damaging rock 0 receives two damage values that are always high-together / low-together, never
independent. Marginal distributions are correct and determinism holds, so the practical bias is
small — but any analysis of damage variance on multi-source-contact steps is distorted.
**Suggested fix**: split `damage_key` three ways (stream break — same fixture caveat as N1).

### N3. [Med, latent → now load-bearing] `termination_reason` gaps can corrupt training the day a body flag flips — KNOWN (memory `20260609_1726`), NEW consequence
Since the timeout fixes (`3c60f6f`, `926c2c3`, `5b093bf`), all three trainers derive "real death"
from `termination_reason >= 2` (`recurrent_ppo_trainer.py:330`, `ppo_trainer.py:226`,
`dreamer_v3_trainer.py:245-246`). That promotes two documented reason-code defects from
"mislabeled statistics" to "wrong value targets":

1. **`with_injury: false` instant death has no reason code.** `core.py:126` sets `done=True` on
   any damage when the injury system is off, but the reason ladder (`core.py:704-709`) never
   assigns a death code for this path — reason stays 0 (or 1 if the same step truncates). The
   trainer would then treat a real death as a truncation: GAE bootstraps `V(s')` *through the
   death* while the env simultaneously applies `-death_penalty` — contradictory targets; Dreamer's
   continue head learns "episode continues" on death. **Currently latent**: only two archived
   labmeeting configs disable a body system; no live config does.
2. **`overeating_death: true` fires the death code on live steps.** `core.py:707-708` sets
   reason=3 whenever `satiation >= max_satiation` — which (since satiation is a capped
   deterministic function of nutrition) means *whenever the stomach is full*, a common,
   non-terminal state. `done` never fires (see N4), so a GAE-mode trainer would see
   `terminateds=1` on ordinary mid-episode steps and zero the bootstrap there. **Currently
   latent**: no config enables `overeating_death`.

**Suggested fix direction**: add an explicit reason code for instant (injury-system-off) death;
gate the reason=2/3/4 assignments on the corresponding `done` contribution actually firing; or
have trainers consume `real_death` (exported into `info`) instead of re-deriving from reason codes.

### N4. [Med, confirmed latent] Over-eating never ends the episode — KNOWN (memory `20260609_1726`), triage complete
Confirmed at the code level: `update_body`'s termination block (`core.py:117-126`) contains no
satiation branch; `params.overeating_death` touches only the reason code (`core.py:707-708`),
never `done`. Additional triage detail: because satiation is `max_satiation * (nutrition/max_nutrition)^k`
with the ratio clipped to 1 (`core.py:74-75`), `satiation >= max_satiation` is reachable **only**
when nutrition sits exactly at its cap — i.e., the coded condition means "stomach full", not a
plausible death condition. As written, `overeating_death` can *literally never* terminate an
episode; the flag is a misnomer plus a reason-code polluter (see N3.2).

### N5. [Low] Per-entity distance arrays in `info` do not mask inactive (parked) animals — NEW (residue of the ghost-predator fix)
`core.py:756-785`: `dist_per_animal`, `dist_per_predator`, `dist_per_neutral` are computed from
raw positions with no `animal_active` mask (unlike `dist_to_pred`/`dist_to_neutral`, which mask
to 99.0). An inactive slot is parked at `(height, width)`, which is Chebyshev-adjacent to the
bottom-right corner — an agent near that corner logs a phantom "predator at distance ~1.4" in
the per-entity arrays consumed by the behavior accumulators (`StepInfo.dist_per_predator`,
`recurrent_ppo_trainer.py:249`). Distorts distance-based behavior metrics only in configs with
per-episode animal count ranges; no effect on training reward. Sibling nit: inactive hunt slots
still run the hunt state machine, so `animal_state` can read HUNT for a parked ghost — harmless
today, a footgun for future consumers.

### N6. [Low] `src/environment/grid_world.py` is an orphaned renderer duplicate with a latent crash — NEW
The file is a stale copy of `renderer.py` (not the vectorized env its name suggests). Nothing in
`src/` or `scripts/` imports it (all callers import `renderer.py` / `renderer_v2.py`). Latent
crash if ever revived: `grid_world.py:602` reads `COLORS['dmg_hiding_predator']` /
`'dmg_predator'` / `'dmg_obstacle'`, which are **missing from its own COLORS dict** (they exist
only in `renderer.py:129-131`) → guaranteed `KeyError` whenever rendering a frame with
`info['damage'] > 0`. Remove-on-touch candidate; at minimum, its misleading name should not
anchor future work (this very diagnosis was scoped to it as "vectorized env").

### N7. [Low] `wrapper.py:auto_reset_step` — extra detail on the known dead-code row
Beyond ignoring its `key` argument (already in KNOWN_BUGS), it contains the same class of key
reuse as N1: `reset_key = split(state.key)[0]` while `jax_step` consumes `split(state.key, 6)` —
by the split-prefix property, `reset_key` equals the stepped state's **new main key** exactly. If
this dead helper is ever revived, every auto-reset episode would be seeded with the key driving
the next step's event randomness. Reinforces the existing "delete on touch" recommendation.

### N8. [Low] Placement fallbacks silently misplace entities when a spawn area is saturated — NEW
- `resolve_overlaps_global` (`core.py:876-878`): if an entity's cell is taken and its spawn area
  has **no** free cell, `replacement_flat` sums to 0 → the entity is silently placed at grid cell
  `(0,0)`, outside its configured area.
- `place_in_area` (`core.py:929-935`): if the area holds fewer free cells than requested, the
  overflow entities resolve to the sentinel `total_cells` → position `(height, 0)`, an off-grid
  row (silent under-placement).
Both only bite over-packed configs; no live config is known to saturate an area. Worth a loader-
side count-vs-area capacity assert.

### N9. [Nit] Minor items
- `core.py:205-208`: `in_zone` computed in `_hunt_step`, never used — dead code.
- `core.py:1360`: initial `last_action` = 4 when `rest_action_enabled` → the first observation's
  proprioception one-hot falsely reports "I rested last step" at every episode start (with rest
  disabled, the 5 one-hots to an all-zero sentinel — inconsistent sentinel choice).
- `sensor.py:284-285`: the clip-min/max comprehensions filter `if name in modality_map` but the
  sigma/alpha/mode loop (`sensor.py:260-264`) does not — the filter is unreachable dead logic
  (the loop above would already have raised the documented `KeyError`).
- `state.py:54`, `state.py:64`: stale `float` comments on now-integer sampled fields (see V4).

---

## Part 3 — Checked and found sound (no finding)

- **Obs assembly ↔ noise slicing**: the sensor order in `get_observation` (`sensor.py:291-343`)
  matches `get_observation_breakdown` (`sensor.py:350-390`) one-for-one, so perceptual noise
  lands on the right slices with the right per-modality clip bounds.
- **Zero-entity configs**: `jax.random.split(key, 0)` is legal (tested) — the unguarded
  `split(respawn_key, num_res)` at `core.py:517` does not crash for zero-resource configs.
- **Predator contact off-by-one / tunneling**: animals chase the agent's *post-move* position
  (`core.py:550-551`), and damage uses post-move coincidence (`core.py:619`), so stepping onto a
  predator and being caught are both detected; no swap-through miss exists. The pre-step
  `hit_neutral` asymmetry is documented, intentional parity (B5).
- **Clamping edges**: nutrition clips to `[0,max]` then dies at `<= 0` (equality reachable);
  injury clips to `[0,max]` then dies at `>= max` (equality reachable) — no unreachable-death
  clamp bug.
- **Eating + dying same step**: gains apply before the death check; penalty gating unaffected.
- **vmap hygiene**: `ParallelEnv` broadcasts `EnvParams`, batches state on axis 0; per-env noise
  keys derive from per-env `state.key` — no cross-env contamination found.
- **Off-grid parked entities vs sensing**: all senses gate on the active masks (see V2), and the
  visual sensor additionally zeroes out-of-bounds cells — a parked ghost cannot be smelled, seen,
  or felt (only the N5 info arrays leak it).

Known-OPEN item *not* re-reported per instructions: the chasing-rabbit post-contact pause firing
only for damaging animals (`core.py:629`). One added detail: the newer opt-in
`animal_disengage_on_contact` field (`core.py:635-637`) does apply to non-damaging animals and
can serve as a config-level mitigation for the rabbit-ride until the row is decided.

---

## Verdict

**The environment core is sound for the live training path.** All five prior fixes verified
correct and complete; no regressions found. The step/reward/termination pipeline, entity
mechanics, and sensor assembly are correct for every currently-live configuration. The two items
that deserve action are *outside the hot semantics*: the rPPO auto-reset key aliasing (N1 — the
one genuine correctness defect in live code, statistical not behavioral) and the
termination-reason gaps (N3/N4 — latent, but now one YAML flag away from corrupting value
targets, so worth closing before anyone flips `with_injury: false` or `overeating_death: true`).

Reviewed by: code-reviewer (independent diagnosis session, Fable 5)
