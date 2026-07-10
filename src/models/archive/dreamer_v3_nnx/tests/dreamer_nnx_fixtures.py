"""Shared tiny-trainer/env builders for the WP-NNX regression tests.

Reuses the fixture pattern established in
tests/models/test_dreamer_collect_arrival_alignment.py:57-97 (H6 tests):
death-free env config + live dreamer agent config shrunk to a fast test size.

Used by the WP-NNX (recipe-alignment) test files:
test_dreamer_nnx_is_first_reset.py, test_dreamer_nnx_stop_gradients.py,
test_dreamer_nnx_obs_loss_sum.py, test_dreamer_nnx_online_bootstrap.py,
test_dreamer_nnx_terminal_start_weights.py,
test_dreamer_nnx_replay_ratio_semantics.py.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from flax import nnx

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown
from src.models.dreamer_v3_trainer import DreamerTrainer

# Death-free fixture env (same rationale as the H6 tests): no mobile predator,
# no static hiding_predator, starvation horizon ~100 steps, overeating_death
# false — within short test windows the only possible episode end is the
# max_steps timeout we control per test.
ENV_CONFIG_NO_DEATH = os.path.join(
    _REPO, "configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml"
)
DREAMER_AGENT_CONFIG = os.path.join(_REPO, "configs/models/dreamer_v3/dreamer_v3.yaml")


def build_config(max_steps):
    """Env default -> death-free env config -> live dreamer agent config,
    shrunk to a fast test size (tiny nets, flat encoding)."""
    cfg = get_default_config()
    cfg.merge(Config.load_yaml(ENV_CONFIG_NO_DEATH))
    cfg.merge(Config.load_yaml(DREAMER_AGENT_CONFIG))
    cfg.set("environment.max_steps", max_steps)
    # Tiny model — these tests check gradient structure / loss semantics /
    # collection indexing, not learning.
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


def build_trainer_and_env(max_steps, reset_seed=0, num_envs=1,
                          modulation_config=None, cfg_overrides=None):
    """Build a tiny DreamerTrainer plus a batched env state.

    Returns (trainer, params, env_state, cfg).
    """
    cfg = build_config(max_steps)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            cfg.set(k, v)
    params = load_env_params(cfg)

    probe_state = jax_reset(params, jax.random.PRNGKey(reset_seed))
    input_dim = get_observation(probe_state, params).shape[0]
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

    trainer = DreamerTrainer(
        input_dim, action_dim, cfg, rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        obs_breakdown=get_observation_breakdown(params),
        modulation_config=modulation_config,
    )

    env_state = jax.vmap(jax_reset, in_axes=(None, 0))(
        params, jax.random.split(jax.random.PRNGKey(reset_seed), num_envs))
    return trainer, params, env_state, cfg


# Small but valid modulation config (mirrors the live
# neuromodulated_dreamer_v3.yaml block, shrunk).
TINY_MODULATION_CONFIG = {
    'type': 'Multiplicative',
    'mod_hidden_size': 8,
    'grouping_size': 1,
    'percept_bias_init': 2.0,
    'percept_add_bias_init': 0.0,
    'memory_bias_init': 0.0,
    'reward_bias_init': 2.0,
}


def make_replay_batch(trainer, params, env_state, B, T, key,
                      term_reason_value=0.0, terminal_from_reason=True):
    """Hand-built replay batch of shape (B, T, ...) with controllable
    termination_reason codes (float), for train_step-level tests.

    Observations come from real env resets (so the encoder sees plausible
    inputs); actions are random one-hots; rewards small random values.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs_dim = get_observation(jax.tree.map(lambda x: x[0], env_state), params).shape[0]
    act_dim = trainer.agent.ac.actor.net.layers[-1].out_features

    obs = jax.random.normal(k1, (B, T, obs_dim))
    action_idx = jax.random.randint(k2, (B, T), 0, act_dim)
    action = jax.nn.one_hot(action_idx, act_dim)
    reward = 0.1 * jax.random.normal(k3, (B, T))
    term_reason = jnp.full((B, T), float(term_reason_value), dtype=jnp.float32)
    if terminal_from_reason:
        terminal = (term_reason >= 1.0).astype(jnp.float32)
    else:
        terminal = jnp.zeros((B, T), dtype=jnp.float32)
    is_first = jnp.zeros((B, T), dtype=jnp.float32)
    return {
        'obs': obs,
        'action': action,
        'reward': reward,
        'terminal': terminal,
        'term_reason': term_reason,
        'is_first': is_first,
    }
