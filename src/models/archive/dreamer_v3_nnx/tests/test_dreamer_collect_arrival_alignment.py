"""Regression tests for H6 — DreamerV3-NNX (obs, action) one-step misalignment
in `collect_sequence` (WP-D, Finding 1 of the 2026-07-04 Fable 5 re-diagnosis).

Plain-language context: DreamerV3's recurrent world model (the RSSM) is defined
so that the action fed alongside an observation is the action that PRODUCED that
observation (paper Eq. 3: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})). Our inference
path honors this — `get_action` feeds (current obs, the action that led to it) —
but data collection stored, in each buffer row, the PRE-step observation
together with the action chosen AFTER seeing it. The training scan
(dreamer_v3_trainer.py, `env_inputs = (action, is_first)`, no shift) therefore
trained the model with the action leaking one step of future information about
the observation it must predict, and the model was then used at act time under
the opposite convention. The fix (chosen over a train-time shift, which would
erase every death event from world-model training — see §A3 of the plan) is to
store the ARRIVAL observation: the one sensed from the post-step, PRE-reset env
state, so the death/timeout observation is retained on the terminal row.

Ground-truth replay is exact because `jax_step(state, action, params)` is
deterministic given the state (the PRNG lives inside `EnvState.key`), so we can
re-run the recorded actions from the same initial state and compute, for every
step, both the departure observation (pre-step) and the arrival observation
(post-step) — then check which one `collect_sequence` stored.

See docs/develop/active/issues/diag_fable5_20260704/
fix_plan_h6h7_dreamer_v3_world_model.md (plan) and 05_dreamer_v3_nnx.md
(Finding 1).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import pytest
from flax import nnx

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown
from src.models.dreamer_v3_trainer import DreamerTrainer

# Death-free fixture env: no mobile predator, no static hiding_predator,
# starvation horizon ~100 steps (start nutrition 100, metabolic cost 1/step),
# overeating_death false — so within a <=6-step window NO death can occur and
# the only possible episode end is the max_steps timeout we control per test.
ENV_CONFIG_NO_DEATH = os.path.join(
    _REPO, "configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml"
)
DREAMER_AGENT_CONFIG = os.path.join(_REPO, "configs/models/dreamer_v3/dreamer_v3.yaml")


def _build_config(max_steps):
    """Env default -> death-free env config -> live dreamer agent config,
    shrunk to a fast test size (tiny nets, flat encoding)."""
    cfg = get_default_config()
    cfg.merge(Config.load_yaml(ENV_CONFIG_NO_DEATH))
    cfg.merge(Config.load_yaml(DREAMER_AGENT_CONFIG))
    cfg.set("environment.max_steps", max_steps)
    # Tiny model — this test checks data-collection indexing, not learning.
    cfg.set("agent.encoder_dim", 16)
    cfg.set("agent.encoder_fc_layers", [16])
    cfg.set("agent.rssm_deter_dim", 16)
    cfg.set("agent.rssm_stoch_dim", 4)
    cfg.set("agent.rssm_classes", 4)
    cfg.set("agent.decoder_fc_layers", [16])
    cfg.set("agent.reward_fc_layers", [16])
    cfg.set("agent.continue_fc_layers", [16])
    cfg.set("agent.actor_fc_layers", [16])
    cfg.set("agent.critic_fc_layers", [16])
    cfg.set("agent.encoding_mode", "flat")
    return cfg


def _build_trainer_and_env(max_steps, reset_seed=0):
    cfg = _build_config(max_steps)
    params = load_env_params(cfg)

    probe_state = jax_reset(params, jax.random.PRNGKey(reset_seed))
    input_dim = get_observation(probe_state, params).shape[0]
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

    trainer = DreamerTrainer(
        input_dim, action_dim, cfg, rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        obs_breakdown=get_observation_breakdown(params),
        modulation_config=None,
    )

    # B=1 batched env state, fixed key -> fully deterministic collection.
    env_state = jax.vmap(jax_reset, in_axes=(None, 0))(
        params, jax.random.split(jax.random.PRNGKey(reset_seed), 1))
    return trainer, params, env_state


def _replay(env_state, actions_onehot, params, num_steps):
    """Replay the recorded actions from the same initial state (valid up to —
    and including — the first done; must NOT be trusted past it, because on
    done the collection carry switches to a reset state drawn from the scan's
    own key stream).

    Returns per-step lists of (departure obs, arrival obs, reward, done),
    each batched (B=1). `arrival` is sensed from the RAW post-step state
    (pre-reset), matching what the fixed collect_sequence must store.
    """
    step_v = jax.vmap(jax_step, in_axes=(0, 0, None))
    obs_v = jax.vmap(get_observation, in_axes=(0, None))

    s = env_state
    records = []
    for t in range(num_steps):
        a = jnp.argmax(actions_onehot[t], axis=-1).astype(jnp.int32)
        obs_departure = obs_v(s, params)
        s_next, reward, done, _info = step_v(s, a, params)
        obs_arrival = obs_v(s_next, params)
        records.append((obs_departure, obs_arrival, reward, done))
        s = s_next
    return records


def test_stored_obs_is_arrival_of_stored_action():
    """Core H6 pin (train ≡ inference pairing). The training scan feeds each
    buffer row's own (obs, action) pair to RSSM.step, while inference
    (get_action) feeds (current obs, the action that produced it). The row
    pair equals an inference pair iff the stored obs is the ARRIVAL
    observation of the stored action. Pre-fix, collect_sequence stores the
    DEPARTURE observation (the one the policy acted from) -> this test fails.
    """
    NUM_STEPS = 6
    trainer, params, env_state = _build_trainer_and_env(max_steps=50)

    _, _, _, transitions = trainer.collect_sequence(
        env_state, params, NUM_STEPS, jax.random.PRNGKey(42))

    # Precondition: the death-free config + generous max_steps guarantee no
    # episode boundary inside the window, so replay is exact for all rows.
    assert not bool(jnp.any(transitions['terminal'])), (
        "fixture must be done-free in the test window; replay would diverge"
    )

    records = _replay(env_state, transitions['action'], params, NUM_STEPS)

    for t, (obs_dep, obs_arr, reward, done) in enumerate(records):
        # Non-vacuity guard: departure and arrival observations must actually
        # differ at every compared step (the interoceptive satiation channel
        # decrements every step), otherwise the core assertion proves nothing.
        assert not jnp.allclose(obs_arr[0], obs_dep[0]), (
            f"step {t}: arrival obs equals departure obs — test window is "
            f"vacuous, cannot distinguish the two conventions"
        )

        # Core assertion: row t stores the arrival observation of action a_t.
        assert jnp.allclose(transitions['obs'][t, 0], obs_arr[0]), (
            f"step {t}: stored obs is NOT the arrival observation of the "
            f"stored action (H6: it is the pre-step/departure obs — the "
            f"(obs, action) row pairing leaks the action one step into the "
            f"future at world-model training time)"
        )

        # Index-stability guards (must pass pre- AND post-fix): only the
        # 'obs' field changes meaning; reward/terminal keep their index.
        assert jnp.allclose(transitions['reward'][t, 0], reward[0]), (
            f"step {t}: reward index moved — H6 fix must not re-index rewards"
        )
        assert bool(transitions['terminal'][t, 0]) == bool(done[0]), (
            f"step {t}: terminal index moved — H6 fix must not re-index dones"
        )


def test_terminal_row_keeps_arrival_obs_and_no_bleed_into_next_episode():
    """Episode-boundary pin. With max_steps=4 a timeout fires deterministically
    at window step k=3. (a) The terminal row must store the TERMINAL arrival
    observation — sensed from the post-step, PRE-reset state that auto-reset
    would otherwise discard (fails pre-fix: the departure obs is stored).
    (b) Boundary flags sit on the expected rows. (c) H5-class bleed guard
    (invariant pre- and post-fix): the new episode's first row must carry no
    leftover timeout/death flags — the exact corruption class the dreamer_srl
    H5 fix removed must not be introduced here.
    """
    NUM_STEPS = 6
    K = 3  # timeout row: 4th env step of the episode (max_steps = 4)
    trainer, params, env_state = _build_trainer_and_env(max_steps=4)

    _, _, _, transitions = trainer.collect_sequence(
        env_state, params, NUM_STEPS, jax.random.PRNGKey(42))

    # Replay rows 0..K only (replay past the first done is invalid — the
    # collection carry switches to a reset state we cannot reproduce here).
    records = _replay(env_state, transitions['action'], params, K + 1)
    obs_dep_k, obs_arr_k, _, done_k = records[K]

    # Sanity: the timeout really fires at row K and nowhere before.
    assert bool(done_k[0]), "fixture error: timeout did not fire at row K"
    for t in range(K):
        assert not bool(records[t][3][0]), f"unexpected done at row {t}"

    # (b) boundary flags on the expected rows.
    assert bool(transitions['terminal'][K, 0]), "terminal flag missing on row K"
    assert float(transitions['termination_reason'][K, 0]) == 1.0, (
        "termination_reason on the timeout row must be 1 (timeout)"
    )
    assert float(jnp.squeeze(transitions['is_first'][K + 1, 0])) == 1.0, (
        "row K+1 (new episode's first row) must be marked is_first"
    )

    # (a) FAILS PRE-FIX: the terminal row must store the terminal ARRIVAL
    # observation (post-step, pre-reset), not the departure obs.
    assert not jnp.allclose(obs_arr_k[0], obs_dep_k[0]), (
        "vacuous boundary window: terminal arrival obs equals departure obs"
    )
    assert jnp.allclose(transitions['obs'][K, 0], obs_arr_k[0]), (
        "terminal row does not store the terminal arrival observation "
        "(H6: auto-reset discarded the episode-ending observation)"
    )

    # (c) H5-class bleed guard: no terminal/timeout flags leak into the new
    # episode's first row.
    assert not bool(transitions['terminal'][K + 1, 0]), (
        "terminal flag bled into the new episode's first row (H5 bug class)"
    )
    assert float(transitions['termination_reason'][K + 1, 0]) == 0.0, (
        "termination_reason bled into the new episode's first row (H5 bug class)"
    )
