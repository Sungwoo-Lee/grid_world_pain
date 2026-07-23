"""Regression test for Finding B: surviving to the time limit must NOT be
punished like death.

Background (plain English): this project scores an agent by how many steps it
survives. An episode ends either because the agent dies (starves, over-eats,
or is injured past a threshold) or because the clock runs out (it survives to
`max_steps` — the SUCCESS outcome of a survival task). The pre-fix bug applied
the large `death_penalty` (e.g. -100) on BOTH kinds of ending, because the
environment merged "real death" and "timeout" into a single `done` flag before
gating the penalty on it. This test drives the real `jax_step`/`jax_reset` to
each ending and asserts the penalty fires on real death only.

See docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md for the full
diagnosis and fix plan. This test must FAIL on the pre-fix code (reward on
timeout ~= -100) and PASS after the fix (reward on timeout ~= the ordinary
homeostatic step value, |reward| << death_penalty).
"""
import os
import sys

import jax
import jax.numpy as jnp
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.environment.config_loader import load_env_config, load_env_params
from src.environment import core

# basic ladder re-level (b093023): 06-sensory_noise -> 05-sensory_noise (content unchanged).
CFG = "configs/environment/experiment/basic/05-sensory_noise_10x10.yaml"


def _load_params():
    config = load_env_config(CFG)
    return load_env_params(config)


def _run_episode(params, seed, max_steps_override, action_fn):
    """Drive jax_step/jax_reset until `done`, return the final step's
    (reward, info, done)."""
    p = params._replace(max_steps=int(max_steps_override))
    step_fn = jax.jit(core.jax_step)
    state = core.jax_reset(p, jax.random.PRNGKey(seed))
    last = None
    for t in range(max_steps_override + 5):
        a = action_fn(t)
        state, reward, done, info = step_fn(state, jnp.int32(a), p)
        last = (reward, info, done)
        if bool(done):
            break
    return last


def test_timeout_no_death_penalty():
    """Surviving to max_steps (truncation) must NOT incur the death penalty."""
    params = _load_params()
    assert params.use_homeostatic_reward, (
        "Test assumes homeostatic reward mode (per basic/05 config); "
        "adjust if the config changes."
    )

    reason = None
    reward = None
    for seed in range(50):
        reward, info, done = _run_episode(
            params, seed=seed, max_steps_override=3, action_fn=lambda t: 0
        )
        reason = int(info['termination_reason'])
        if reason == 1:
            break
    assert reason == 1, (
        f"Could not drive a clean timeout episode (last reason={reason}); "
        "test setup issue, not the fix under test."
    )

    death_penalty = float(params.death_penalty)
    # The ordinary per-step homeostatic reward is small (~0.2 in magnitude).
    # A death penalty of e.g. 100 would make |reward| >> 1.0.
    assert reward > -1.0, (
        f"Timeout step reward = {reward:.5f}; expected ~ -0.2 (no death penalty). "
        f"death_penalty = {death_penalty} appears to have been applied on timeout."
    )
    assert reward > -death_penalty / 2, (
        f"Timeout step reward = {reward:.5f} is on the order of -death_penalty "
        f"({death_penalty}); the penalty must not fire on truncation."
    )


def test_starvation_applies_death_penalty():
    """Dying of starvation before max_steps must still incur the death penalty
    (guards against over-correction — i.e. the fix must not remove the
    penalty from real deaths too)."""
    params = _load_params()
    death_penalty = float(params.death_penalty)
    assert death_penalty > 0.0, "Test requires a nonzero death_penalty in basic/05."

    reward, info, done = _run_episode(
        params, seed=0, max_steps_override=10000, action_fn=lambda t: 0
    )
    reason = int(info['termination_reason'])
    assert reason == 2, (
        f"Expected starvation (reason=2) driving with wander-only actions, got {reason}. "
        "Test setup issue, not the fix under test."
    )
    assert reward < -death_penalty / 2, (
        f"Starvation step reward = {reward:.5f}; expected the death_penalty "
        f"({death_penalty}) to be applied (reward strongly negative)."
    )
