#!/usr/bin/env python3
"""traj_scan.py — the batched rollout kernel for the trajectory-collection pipeline.

Plan: docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md (§D5, §D8, §D12)

What this module does, in one paragraph: it runs `batch_size` episodes at once by
`vmap`-ing the environment over episodes and `lax.scan`-ing over time, all inside a
single `nnx.jit`, then flattens the result on the host into exactly the fixed column
set defined by `src/utils/trajectory_store.py`.  It is deliberately SEPARATE from
`scripts/eval/eval_rollout.py`'s `_rollout_scan_jit` so the dwell-sweep pipeline is
untouched by this change.

THE ONE THING YOU MUST NOT BREAK (plan §A5).  After `nnx.update(model, restored_tree)`
restores a checkpoint, calling the model EAGERLY reads a stale view of the restored
parameters and every trajectory silently diverges from step 0 onward.  `lax.scan`
alone does not fix it — it compiles the outer XLA loop but does not perform nnx's
graphdef/state split.  `_rollout_scan` therefore refuses to run outside a trace, and
the only supported entry point is `nnx.jit`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # scripts/eval/traj_collect -> repo root (3 up)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown


# `float16` overflows to inf above 65,504.  OBS_ABS_MAX sits comfortably above any
# bounded sensor and comfortably below the cliff — the outer backstop.  Plan §D12.
OBS_ABS_MAX = 1e4

# The TIGHT range guard: the absolute float16 round-trip error the store is willing to
# accept.  This is the clause with teeth, and it is set from measurement rather than
# from a round number.
#
# WHY ABSOLUTE AND NOT RELATIVE (a plan correction — see the Implementation Report).
# The plan specified a RELATIVE round-trip ceiling of 1e-3, on the reasoning that it
# "additionally catches subnormal underflow".  Run against a real 5,000-episode
# collection it fired immediately — on an interoceptive-nociception reading of
# 1.34e-06, which float16 stores as 1.37e-06.  That is a relative error of 2.2 % and an
# ABSOLUTE error of 2.9e-08: about 8,000x below the store's own stated error budget and
# ~3e5 x below the sigma = 0.10 noise the environment injects into that channel on
# purpose.  A relative criterion is simply the wrong instrument near zero, and as
# written it made the plan's DEFAULT precision impossible to collect at.
#
# The store's error budget is absolute, so the guard is too.  Measured on a real
# 5,000-episode store (1,028,932 step rows) from the reference training run:
#   - worst-case absolute round-trip error anywhere in the store: 1.95e-03,
#     on an olfaction channel whose magnitude reaches 6.95 (olfaction is NOT
#     bounded in [0,1], contrary to the plan's assumption);
#   - that 1.95e-03 is still ~100x below olfaction's own injected sigma of 0.20.
# 1e-2 therefore leaves ~5x headroom over measured reality, while corresponding to
# float16's half-ulp at |x| ~ 20 — so the guard fires as soon as any channel's
# magnitude reaches ~20, long before the 65,504 overflow cliff and long before the
# quantisation error becomes comparable to any injected sigma.  That is a realistic
# drift (a larger grid or a higher `sensor_radius` could push olfaction there), which
# is what makes this clause falsifiable rather than decorative.
OBS_ABS_ERR_MAX = 1e-2
# float16 worst-case absolute error on values in [0, 1], quoted in error messages.
FLOAT16_ABS_ERR_ON_UNIT_INTERVAL = 2.44e-04


# ── Reset-row helper (plan §D8) ───────────────────────────────────────────────

def _agent_in_bush(state, params):
    """The environment's OWN definition of bush occupancy, transcribed from
    `src/environment/core.py:793-797`, for the reset row (t = 0), where the env
    produces no `info` dict.

    Lives here and NOT in `src/environment/` — this plan touches no environment code.
    Verification V4 asserts this helper and the env agree at every step, by
    recomputing the same expression in pure NumPy from the stored columns.

    Note what it is NOT: `scripts/behavior_measures/avoidance_stats_heatmap.py:77-79`
    hardcodes obstacle slot 0 and ignores both `obs_hides_agent` and `obs_active`,
    which is systematically wrong whenever bush count varies per episode — i.e. always,
    in a training environment.  Do not copy that.
    """
    if state.obs_pos.shape[0] == 0:
        return jnp.array(False)
    return jnp.any(jnp.logical_and(
        jnp.all(state.obs_pos == state.agent_pos, axis=-1),
        params.obs_hides_agent & state.obs_active,
    ))


# ── PRNG parity guard (plan §D5) ──────────────────────────────────────────────

def _prng_parity_guard(params, seeds_i64, states0):
    """Vectorised, ONE-host-sync-per-chunk replacement for `eval_rollout.py:405-415`.

    The invariant it protects: the per-episode reset key MUST be the directly-stacked
    `jax.random.PRNGKey(seed)` and NOT `ParallelEnv.reset`, whose internal
    `jax.random.split(key, num_envs)` derives a completely different key set and would
    silently corrupt every trajectory.

    The reference path derives its keys from the INTEGER SEEDS itself (not from the
    caller's key array), so a wrong key construction in the collector is caught rather
    than echoed back.  `jax.lax.map` lowers to `lax.scan` — genuinely NOT `vmap` — so it
    is an independent execution path for the same invariant.

    What it cannot catch, stated so nobody over-trusts it: a wrong seed-to-episode
    ASSIGNMENT (e.g. reusing one fixed seed for every episode, or an off-by-one in block
    indexing).  Both sides consume the same seed list.  V1 (realised draws vs. an
    independent replay) and the store's no-duplicate-seed check cover that.
    """
    ref = jax.lax.map(lambda s: jax_reset(params, jax.random.PRNGKey(s)).key, seeds_i64)
    if not bool(jnp.all(ref == states0.key)):
        bad = int(jnp.argmax(jnp.any(ref != states0.key, axis=-1)))
        raise RuntimeError(
            f"Batched reset PRNG parity check FAILED at chunk index {bad} "
            f"(seed={int(seeds_i64[bad])}): state.key from vmap(jax_reset) over "
            "jnp.stack([PRNGKey(s) for s in seeds]) does not match an independent "
            "lax.map(jax_reset(PRNGKey(s))) reference. The batched reset must NOT go "
            "through ParallelEnv.reset (whose internal jax.random.split(key, num_envs) "
            "derives a different key set)."
        )


# ── The scan kernel — entered ONLY via nnx.jit ────────────────────────────────

def _rollout_scan(model, params, states0, h0, obs0, obs_true0, max_steps, policy_step):
    """Scan body.  MUST be entered via `nnx.jit` — see the module docstring.

    Returns `(final_draws, scan_out)` where `scan_out` fields are (max_steps, B, ...)
    and `final_draws` is the per-episode draw block read off the FINAL carry state
    (checkpoint C5 compares it against the same block read at reset; `jax_step` copies
    every one of these fields through unchanged, `core.py:818-828`, so they must agree).

    The observation is CARRIED, not recomputed.  That is not an optimisation for its own
    sake: it makes the row convention structural.  Row `t`'s observation is literally the
    array the policy consumed when choosing the action recorded in row `t + 1`, rather
    than a value recomputed later and argued to be equal.
    """
    if not isinstance(states0.agent_pos, jax.core.Tracer):
        raise RuntimeError(
            "_rollout_scan was called EAGERLY (its arguments are concrete, not traced). "
            "It must be entered through nnx.jit. Calling the model eagerly after "
            "nnx.update(model, restored_tree) reads a STALE view of the restored "
            "parameters and every trajectory silently diverges from step 0 onward — "
            "see the module docstring and eval_rollout.py:293-311."
        )

    v_step = jax.vmap(jax_step, in_axes=(0, 0, None))
    v_obs = jax.vmap(get_observation, in_axes=(0, None))
    v_obs_true = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False), in_axes=(0, None))

    def scan_fn(carry, _):
        state, h, obs, _obs_true = carry
        action, h_new = policy_step(model, obs, h)
        next_state, reward, done, info = v_step(state, action, params)
        next_obs = v_obs(next_state, params)
        next_obs_true = v_obs_true(next_state, params)

        step_out = {
            # -- arriving quantities (the step that produced state t+1) --------
            "action": action,
            "reward": reward,
            "done": done,
            "damage": info["damage"],
            "ate_food": info["ate_food"],
            "rested": info["rested"],
            "hit_predator": info["hit_predator"],
            "hit_neutral": info["hit_neutral"],
            "hit_hiding_predator": info["hit_hiding_predator"],
            "event_collided": info["event_collided"],
            "termination_reason": info["termination_reason"],
            "agent_in_bush": info["agent_in_bush"],
            # -- state at t+1 ---------------------------------------------------
            "agent_pos": next_state.agent_pos,
            "satiation": next_state.satiation,
            "nutrition": next_state.nutrition,
            "injury_level": next_state.injury_level,
            "rest_streak": next_state.rest_streak,
            "last_collision_noc": next_state.last_collision_noc,
            "terminated": next_state.terminated,
            "animal_pos": next_state.animal_pos,
            "animal_state": next_state.animal_state,
            "animal_stamina": next_state.animal_stamina,
            "animal_move_timer": next_state.animal_move_timer,
            "animal_attack_timer": next_state.animal_attack_timer,
            "res_pos": next_state.res_pos,
            "res_active": next_state.res_active,
            "res_cons_count": next_state.res_cons_count,
            "res_reg_timer": next_state.res_reg_timer,
            "obs_pos": next_state.obs_pos,
            "obs_noised": next_obs,
            "obs_true": next_obs_true,
        }
        return (next_state, h_new, next_obs, next_obs_true), step_out

    (final_state, _h, _o, _ot), scan_out = jax.lax.scan(
        scan_fn, (states0, h0, obs0, obs_true0), None, length=max_steps
    )
    return _draw_block(final_state), scan_out


# Which per-episode draw fields are ACTUALLY constant for the whole episode.
#
# `jax_step` copies the animal draws and the activation masks through unchanged — see the
# `state._replace(...)` call at `src/environment/core.py:800-838` (`res_allocated` at 805,
# the animal block at 812-828, `obs_active` at 830).  The two RESOURCE property draws are
# the exception and are deliberately NOT in this set: `jax_step` RE-DRAWS them whenever a
# resource regenerates (`core.py:526-541` builds `res_property_sampled_after_reg` /
# `res_visual_property_sampled_after_reg`; `core.py:808-809` assigns them into the new
# state).  That is exactly why the schema names them `*_init` — "the draw at reset" — and
# why the schema doc carries a caveat about treating them as episode constants.
#
# Asserting reset == final over ALL of `_draw_block` would therefore encode an invariant
# the environment contradicts. Any config with regenerating resources AND a nonzero
# resource `properties_std` would abort mid-collection, and — because a resumed block is
# redone from the same seeds — would abort at the same block forever.  No config in the
# tree has a nonzero resource std today, so it cannot fire yet; per-episode-variance
# environments are precisely the scientific target of this store, so it would.
CONSTANT_DRAW_FIELDS: frozenset = frozenset({
    "animal_active", "animal_detect_sampled", "animal_max_stamina_sampled",
    "animal_recovery_sampled", "animal_hunt_thresh_sampled", "animal_lose_interest_sampled",
    "animal_move_int_sampled", "animal_attack_delay_sampled", "animal_attack_range_sampled",
    "animal_property_sampled", "animal_visual_property_sampled",
    "res_allocated", "obs_active", "obs_property_sampled", "obs_visual_property_sampled",
})

# Draw fields the environment may legitimately mutate mid-episode (resource regeneration).
# Recorded at reset only; excluded from the C5 constancy check BY DESIGN, not by omission.
MUTABLE_DRAW_FIELDS: frozenset = frozenset({
    "res_property_sampled_init", "res_visual_property_sampled_init",
})


def _draw_block(state):
    """The per-episode realised-draw fields, read off a state (plan §D4.2 cols 7-23)."""
    return {
        "animal_active": state.animal_active,
        "animal_detect_sampled": state.animal_detect_sampled,
        "animal_max_stamina_sampled": state.animal_max_stamina_sampled,
        "animal_recovery_sampled": state.animal_recovery_sampled,
        "animal_hunt_thresh_sampled": state.animal_hunt_thresh_sampled,
        "animal_lose_interest_sampled": state.animal_lose_interest_sampled,
        "animal_move_int_sampled": state.animal_move_int_sampled,
        "animal_attack_delay_sampled": state.animal_attack_delay_sampled,
        "animal_attack_range_sampled": state.animal_attack_range_sampled,
        "animal_property_sampled": state.animal_property_sampled,
        "animal_visual_property_sampled": state.animal_visual_property_sampled,
        "res_allocated": state.res_allocated,
        "res_property_sampled_init": state.res_property_sampled,
        "res_visual_property_sampled_init": state.res_visual_property_sampled,
        "obs_active": state.obs_active,
        "obs_property_sampled": state.obs_property_sampled,
        "obs_visual_property_sampled": state.obs_visual_property_sampled,
    }


def _reset_row(states0, params, obs0, obs_true0):
    """The `t = 0` row group (plan §D2): the reset STATE, with every arriving field at
    its zero value and `action = -1`, `reward = 0.0`."""
    B = states0.agent_pos.shape[0]
    zeros_b = jnp.zeros((B,), dtype=jnp.float32)
    false_b = jnp.zeros((B,), dtype=jnp.bool_)
    v_bush = jax.vmap(_agent_in_bush, in_axes=(0, None))
    return {
        "action": jnp.full((B,), -1, dtype=jnp.int32),
        "reward": zeros_b,
        "done": false_b,
        "damage": zeros_b,
        "ate_food": false_b,
        "rested": false_b,
        "hit_predator": false_b,
        "hit_neutral": false_b,
        "hit_hiding_predator": false_b,
        "event_collided": false_b,
        "termination_reason": jnp.zeros((B,), dtype=jnp.int32),
        "agent_in_bush": jnp.broadcast_to(v_bush(states0, params), (B,)),
        "agent_pos": states0.agent_pos,
        "satiation": states0.satiation,
        "nutrition": states0.nutrition,
        "injury_level": states0.injury_level,
        "rest_streak": states0.rest_streak,
        "last_collision_noc": states0.last_collision_noc,
        "terminated": states0.terminated,
        "animal_pos": states0.animal_pos,
        "animal_state": states0.animal_state,
        "animal_stamina": states0.animal_stamina,
        "animal_move_timer": states0.animal_move_timer,
        "animal_attack_timer": states0.animal_attack_timer,
        "res_pos": states0.res_pos,
        "res_active": states0.res_active,
        "res_cons_count": states0.res_cons_count,
        "res_reg_timer": states0.res_reg_timer,
        "obs_pos": states0.obs_pos,
        "obs_noised": obs0,
        "obs_true": obs_true0,
    }


# ── Host-side flatten (plan §D5) ──────────────────────────────────────────────

def _stack_rows(reset_arr, scan_arr):
    """(B, ...) reset row + (T_scan, B, ...) scan rows -> (B, T_scan+1, ...)."""
    r = np.asarray(reset_arr)
    s = np.moveaxis(np.asarray(scan_arr), 0, 1)
    if r.ndim < s.ndim - 1:
        # DEFENSIVE ONLY — dead on every path exercised today.  `_reset_row` builds every
        # field explicitly at batch width (constants are `jnp.broadcast_to`-ed there), so
        # a rank-deficient reset field would mean `_reset_row` and the scan disagree about
        # a column's shape.  Kept as a loud-shape-fix rather than a silent broadcast
        # failure; if it ever fires, `_reset_row` is what needs fixing.
        r = np.broadcast_to(r, s.shape[:1] + s.shape[2:])
    return np.concatenate([r[:, None, ...], s], axis=1)


def _flatten_to_rows(scan_out, reset_rows, T, seeds):
    """Fully vectorised NumPy flatten, replacing the Python `for i in range(num_envs)`
    loop at `eval_rollout.py:456-475` — one boolean gather per column, no per-episode
    Python work at any point.

    Output rows are ordered by (episode, t), matching the store's sort order.  Returns
    a {name: array} dict keyed by the SCAN's field names; the caller maps them onto the
    schema's column names.
    """
    B = len(seeds)
    n_scan = np.asarray(scan_out["action"]).shape[0]
    t_grid = np.broadcast_to(np.arange(n_scan + 1, dtype=np.int64), (B, n_scan + 1))
    mask = t_grid <= np.asarray(T)[:, None]        # (B, n_scan+1)

    out = {}
    for name, scan_arr in scan_out.items():
        if name == "done":
            continue
        full = _stack_rows(reset_rows[name], scan_arr)
        out[name] = full[mask]
    out["t"] = t_grid[mask]
    out["episode_seed"] = np.repeat(np.asarray(seeds, dtype=np.int64), np.asarray(T) + 1)
    return out


# ── The float16 range guard (plan §D12) ───────────────────────────────────────

def _sensor_of_index(params, idx: int) -> str:
    """Resolve an observation index to the sensor it belongs to, so the operator learns
    'dimension 22 (Visual) exceeded the safe range at 1.2e5', not 'assertion failed'."""
    off = 0
    for name, dim in get_observation_breakdown(params).items():
        if off <= idx < off + dim:
            return f"{name} (dims {off}..{off + dim - 1})"
        off += dim
    return f"<unknown sensor; observation breakdown covers 0..{off - 1}>"


def assert_obs_representable(obs_f32, obs_precision: str, params, column: str = "obs") -> None:
    """HARD RUNTIME GUARD, not a post-hoc check.  Called once per chunk, after the host
    flatten and BEFORE the shard write; a raise therefore leaves nothing on disk,
    because shards are written by atomic rename.

    A store that silently clipped an out-of-range channel is worse than a store that is
    twice as large: the damage would be invisible at read time and would invalidate
    every analysis touching that sensor.  Do NOT downgrade this to a warning and do NOT
    move it after the write.
    """
    obs_f32 = np.asarray(obs_f32, dtype=np.float32)
    if obs_f32.size == 0:
        return

    if not np.isfinite(obs_f32).all():
        i = np.argwhere(~np.isfinite(obs_f32))[0]
        d = int(i[-1])
        raise ValueError(
            f"Observation column {column!r} contains a non-finite value "
            f"({obs_f32[tuple(i)]!r}) at observation index {d} — sensor "
            f"{_sensor_of_index(params, d)}. The environment produced inf/nan; this is "
            "not a precision problem. Nothing has been written."
        )

    if obs_precision == "float32":
        # DORMANCY, stated rather than left to be discovered: under float32 BOTH magnitude
        # clauses below (the OBS_ABS_ERR_MAX round-trip budget and the OBS_ABS_MAX
        # backstop) are skipped.  That is defensible — float32 cannot misrepresent these
        # magnitudes, so there is nothing for them to catch — but it has a consequence: a
        # float32 store carries no recorded evidence about observation RANGE, so deciding
        # later that a run "could have been float16" is not supported by anything the
        # store contains.  Measure it explicitly if you need it.
        return
    if obs_precision != "float16":
        raise ValueError(f"Unknown obs_precision {obs_precision!r}")

    amax = np.abs(obs_f32).max(axis=0)
    over = np.nonzero(amax > OBS_ABS_MAX)[0]
    if over.size:
        d = int(over[0])
        raise ValueError(
            f"Observation index {d} — sensor {_sensor_of_index(params, d)} — reached "
            f"|value| = {amax[d]:.6g} in column {column!r}, above the OBS_ABS_MAX safety "
            f"ceiling of {OBS_ABS_MAX:g}. float16 overflows to inf at 65504, so this "
            "store would silently mangle that channel. Re-run this collection with "
            "--obs-precision float32. Nothing has been written."
        )

    rt = np.asarray(obs_f32.astype(np.float16), dtype=np.float32)
    if not np.isfinite(rt).all():
        i = np.argwhere(~np.isfinite(rt))[0]
        d = int(i[-1])
        raise ValueError(
            f"Observation index {d} — sensor {_sensor_of_index(params, d)} — OVERFLOWS "
            f"float16 (value {obs_f32[tuple(i)]!r} -> inf) in column {column!r}. "
            "Re-run with --obs-precision float32. Nothing has been written."
        )
    aerr = np.abs(obs_f32 - rt)
    aerrmax = aerr.max(axis=0)
    bad = np.nonzero(aerrmax > OBS_ABS_ERR_MAX)[0]
    if bad.size:
        d = int(bad[0])
        j = int(np.argmax(aerr[:, d]))
        raise ValueError(
            f"Observation index {d} — sensor {_sensor_of_index(params, d)} — has a "
            f"float16 round-trip ABSOLUTE error of {aerrmax[d]:.3g} (value "
            f"{obs_f32[j, d]!r} -> {rt[j, d]!r}, |value| reaches "
            f"{np.abs(obs_f32[:, d]).max():.4g}) in column {column!r}, above the "
            f"{OBS_ABS_ERR_MAX:g} budget. float16's error grows with magnitude "
            f"(half-ulp = 2**-11 * |x|), so this means the channel's RANGE has grown "
            f"beyond what half precision can measure at this store's accuracy — on "
            f"values in [0,1] the worst case is {FLOAT16_ABS_ERR_ON_UNIT_INTERVAL:g}. "
            "Re-run with --obs-precision float32. Nothing has been written."
        )
