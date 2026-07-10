"""Regression tests for WP-NNX F1 (registry K1) — DreamerV3-NNX collection
never resets the agent's belief state at episode boundaries.

Plain-language context: during data collection, `collect_sequence` stages an
`is_first` flag on the carried Dreamer state whenever an episode ends, but
`get_action` ignored it — it hardcoded `is_first = 0` into `RSSM.step` and
carried the previous episode's `prev_action` (and, when modulation is enabled,
the modulator hidden state `mod_h`) straight across the reset. So the first
actions of every new episode were conditioned on the *dead* previous episode's
memory. The canonical recipe resets per-env state on done (sheeprl
`player.init_states`, vendor agent.py:643-659). The fix: `get_action` consumes
the staged `prev_state['is_first']` (RSSM.step then masks deter/stoch itself),
zeroes the stale `prev_action`, and resets `mod_h`.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F1) and 05_dreamer_v3_nnx_conventions.md (K1).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dreamer_nnx_fixtures import (
    build_trainer_and_env, TINY_MODULATION_CONFIG,
)

import jax
import jax.numpy as jnp
from src.environment.sensor import get_observation


def _obs_batched(env_state, params):
    return jax.vmap(get_observation, in_axes=(0, None))(env_state, params)


def _garbage_state(trainer, B, scale=100.0, seed=123, with_mod_h=False):
    """A Dreamer carry full of garbage history, flagged is_first=1."""
    fresh = trainer.agent.wm.rssm.initial(B)
    act_dim = trainer.agent.ac.actor.net.layers[-1].out_features
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    state = {
        'deter': scale * jax.random.normal(k1, fresh['deter'].shape),
        'stoch': scale * jax.random.normal(k2, fresh['stoch'].shape),
        'logits': scale * jax.random.normal(k3, fresh['logits'].shape),
        'prev_action': jax.nn.one_hot(jnp.full((B,), 2), act_dim),
        'is_first': jnp.ones((B, 1)),
    }
    if with_mod_h:
        mod_init = trainer.agent.wm.modulator.initial_state(B)
        state['mod_h'] = mod_init + 37.0
    return state


def _fresh_state(trainer, B, with_mod_h=False):
    """The reference: a genuinely fresh carry, also flagged is_first=1."""
    act_dim = trainer.agent.ac.actor.net.layers[-1].out_features
    state = dict(trainer.agent.wm.rssm.initial(B))
    state['prev_action'] = jnp.zeros((B, act_dim))
    state['is_first'] = jnp.ones((B, 1))
    if with_mod_h:
        state['mod_h'] = trainer.agent.wm.modulator.initial_state(B)
    return state


def test_get_action_is_first_resets_garbage_state():
    """T1(a): with is_first=1, get_action's output must be bit-identical
    whether the carry holds garbage history or a fresh initial state — the
    flag must wipe deter/stoch (RSSM mask), prev_action, everything.
    Pre-fix: is_first is hardcoded to 0 inside get_action, so the garbage
    deter/stoch/prev_action leak through and the outputs differ."""
    trainer, params, env_state, _ = build_trainer_and_env(max_steps=50)
    B = 1
    obs = _obs_batched(env_state, params)
    rng = jax.random.PRNGKey(7)

    state_garbage = _garbage_state(trainer, B)
    state_fresh = _fresh_state(trainer, B)

    a_g, next_g = trainer.get_action(obs, state_garbage, rng=rng)
    a_f, next_f = trainer.get_action(obs, state_fresh, rng=rng)

    assert jnp.array_equal(a_g, a_f), (
        "is_first=1 must make the action independent of the (garbage) carried "
        "history — pre-fix the flag is ignored and stale state drives the action"
    )
    for k in ('deter', 'stoch', 'prev_action'):
        assert jnp.array_equal(next_g[k], next_f[k]), (
            f"next_state['{k}'] differs between garbage-carry and fresh-carry "
            f"despite is_first=1 — episode-boundary reset not applied"
        )


def test_get_action_is_first_zero_keeps_history():
    """Non-vacuity control (must pass pre- AND post-fix): with is_first=0 the
    same garbage carry must produce a DIFFERENT belief than the fresh carry —
    otherwise the reset assertion above proves nothing."""
    trainer, params, env_state, _ = build_trainer_and_env(max_steps=50)
    B = 1
    obs = _obs_batched(env_state, params)
    rng = jax.random.PRNGKey(7)

    state_garbage = _garbage_state(trainer, B)
    state_garbage['is_first'] = jnp.zeros((B, 1))
    state_fresh = _fresh_state(trainer, B)
    state_fresh['is_first'] = jnp.zeros((B, 1))

    _, next_g = trainer.get_action(obs, state_garbage, rng=rng)
    _, next_f = trainer.get_action(obs, state_fresh, rng=rng)

    assert not jnp.allclose(next_g['deter'], next_f['deter']), (
        "vacuous fixture: garbage history does not influence the belief even "
        "with is_first=0 — the reset test cannot discriminate"
    )


def test_get_action_is_first_resets_modulator_state():
    """T1(a), modulated variant: the modulator hidden state mod_h must reset
    at episode boundaries too. Pre-fix, a garbage mod_h leaks through
    forward_obs into the encoder modulation and the RSSM gate bias."""
    trainer, params, env_state, _ = build_trainer_and_env(
        max_steps=50, modulation_config=TINY_MODULATION_CONFIG)
    B = 1
    obs = _obs_batched(env_state, params)
    rng = jax.random.PRNGKey(7)

    state_garbage = _garbage_state(trainer, B, with_mod_h=True)
    state_fresh = _fresh_state(trainer, B, with_mod_h=True)

    a_g, next_g = trainer.get_action(obs, state_garbage, rng=rng)
    a_f, next_f = trainer.get_action(obs, state_fresh, rng=rng)

    assert jnp.array_equal(a_g, a_f)
    for k in ('deter', 'stoch', 'prev_action', 'mod_h'):
        assert jnp.array_equal(next_g[k], next_f[k]), (
            f"next_state['{k}'] differs despite is_first=1 (modulated path) — "
            f"mod_h / belief reset not applied at the episode boundary"
        )


def test_collect_sequence_post_boundary_rows_invariant_to_prefix():
    """T1(b) integration: run collect_sequence twice across a deterministic
    timeout boundary (max_steps=4 -> done at row 3), once from a fresh carry
    and once from a garbage carry. The two runs take different actions BEFORE
    the boundary (control asserts this), but every row AFTER the boundary must
    be bit-identical: the reset draws the same new env state in both runs
    (same key stream), and post-fix the belief is wiped at the boundary, so
    nothing of the pre-done trajectory can influence the new episode.
    Pre-fix: stale deter/stoch carry across the reset makes rows 4-5 differ."""
    NUM_STEPS = 6
    K = 3  # timeout row (max_steps=4)
    trainer, params, env_state, _ = build_trainer_and_env(max_steps=4)
    B = 1
    key = jax.random.PRNGKey(42)

    # Run 1: fresh carry (None -> collect_sequence builds the initial state).
    _, fd1, _, tr1 = trainer.collect_sequence(env_state, params, NUM_STEPS, key)

    # Run 2: garbage carry, flagged NOT-first so the garbage acts as real
    # history and produces different pre-boundary behaviour.
    d0 = _garbage_state(trainer, B, scale=30.0, seed=5)
    d0['is_first'] = jnp.zeros((B, 1))
    _, fd2, _, tr2 = trainer.collect_sequence(env_state, params, NUM_STEPS, key, d0)

    # Fixture sanity: the timeout fires at row K in BOTH runs and nowhere else
    # (death-free config; timeout is action-independent).
    for tr in (tr1, tr2):
        assert bool(tr['terminal'][K, 0]), "timeout did not fire at row K"
        for t in [i for i in range(NUM_STEPS) if i != K]:
            assert not bool(tr['terminal'][t, 0]), f"unexpected done at row {t}"

    # Non-vacuity control: the carry ENTERING the post-boundary row must
    # differ between the two runs (the garbage history must have propagated
    # through the pre-boundary steps), otherwise the invariance assertion
    # below proves nothing. A 4-step collect (rows 0..K) returns exactly that
    # carry; its deter encodes the differing histories pre- AND post-fix
    # (the wipe happens on CONSUMPTION, i.e. at row K+1).
    _, fd1_pre, _, _ = trainer.collect_sequence(env_state, params, K + 1, key)
    _, fd2_pre, _, _ = trainer.collect_sequence(env_state, params, K + 1, key, d0)
    assert not jnp.allclose(fd1_pre['deter'], fd2_pre['deter']), (
        "vacuous fixture: garbage initial carry did not propagate to the "
        "boundary — the invariance assertion below proves nothing"
    )
    assert bool(jnp.squeeze(fd1_pre['is_first'][0])) and \
        bool(jnp.squeeze(fd2_pre['is_first'][0])), (
        "fixture error: is_first not staged on the boundary carry"
    )

    # Core F1 assertion: everything after the boundary is prefix-invariant.
    for field in ('action', 'obs', 'reward'):
        assert jnp.array_equal(tr1[field][K + 1:], tr2[field][K + 1:]), (
            f"post-boundary '{field}' rows depend on the pre-done trajectory — "
            f"belief state not reset at the episode boundary (K1)"
        )
    for k in ('deter', 'stoch', 'prev_action'):
        assert jnp.array_equal(fd1[k], fd2[k]), (
            f"final carry '{k}' depends on the pre-done trajectory across a "
            f"boundary — belief state not reset (K1)"
        )
