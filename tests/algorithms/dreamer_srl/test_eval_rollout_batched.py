"""Regression tests for dreamer_srl_eval_rollout_batched() -- the vmapped
sibling of dreamer_srl_eval_rollout() (Tier 2 perf path for offline Dreamer
probe sweeps).

Two families of test here:

  1. Basic smoke tests (mirrors test_eval_rollout.py's coverage for the legacy
     single-env function) -- runs, returns plausible stats, writes recordings.

  2. A deterministic cross-check that isolates "is the vmapped RSSM + actor +
     env computation itself correct" from "the RNG streams differ across
     episodes by design" (see dreamer_srl_eval_rollout_batched's docstring for
     the full RNG-convention argument). This is the real correctness gate for
     the batching machinery: feed ONE fixed reset key + ONE fixed sequence of
     RSSM sample keys to (a) a hand-written batch=1 step-by-step reference
     loop and (b) `_dreamer_rollout_scan_jit` (the vmap+lax.scan body used
     inside the batched function), and confirm the per-step action/latent
     outputs are bit-identical. No checkpoint restore is involved in either
     path here (a freshly-built agent is used), so the nnx.jit-vs-eager
     restore-parity trap documented in scripts/eval/eval_rollout.py's
     `_rollout_scan_jit` docstring does not apply -- both the reference and
     the scan body are wrapped in nnx.jit for a clean, apples-to-apples
     comparison of the vmap/reshape/broadcast logic only.
"""
import os
import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from flax import nnx
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.eval import (
    dreamer_srl_eval_rollout_batched,
    _dreamer_rollout_scan_jit,
)


@pytest.fixture(scope='module')
def eval_agent_and_params():
    """Build a tiny agent + env_params for eval tests (same fixture as
    test_eval_rollout.py, kept independent/duplicated per-file rather than
    imported, matching that file's own self-contained style)."""
    _root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    env_cfg = get_default_config()
    env_cfg.merge(Config.load_yaml(f'{_root}/configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml'))
    for rel in ['configs/train/default.yaml', 'configs/evaluation/default.yaml',
                'configs/visualization/default.yaml']:
        env_cfg.merge(Config.load_yaml(os.path.join(_root, rel)))

    agent_cfg = Config.load_yaml(f'{_root}/configs/models/dreamer_srl/01_food_only.yaml')

    env_params = load_env_params(env_cfg)
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)
    obs_dim = 19  # known for 5x5 food-only

    key = jax.random.PRNGKey(7)
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=agent_cfg.to_dict(),
        rngs=rngs,
    )
    return world_model, actor, env_params, env_cfg


# ---------------------------------------------------------------------------
# 1. Basic smoke tests
# ---------------------------------------------------------------------------

def test_batched_eval_rollout_returns_stats(eval_agent_and_params, tmp_path):
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout_batched(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=3,
        seed=0,
        results_dir=str(tmp_path),
        checkpoint_pct=50,
        render_video=False,
        quiet=True,
    )
    assert 'mean_reward' in result
    assert 'mean_length' in result
    assert 'episode_rewards' in result
    assert 'episode_lengths' in result
    assert len(result['episode_rewards']) == 3
    assert len(result['episode_lengths']) == 3


def test_batched_eval_rollout_episode_lengths_valid(eval_agent_and_params, tmp_path):
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout_batched(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=4,
        seed=1,
        results_dir=str(tmp_path),
        checkpoint_pct=100,
        render_video=False,
        quiet=True,
    )
    for ep_len in result['episode_lengths']:
        assert 1 <= ep_len <= int(env_params.max_steps), \
            f"episode_length={ep_len} outside [1, {env_params.max_steps}]"


def test_batched_eval_rollout_recordings_exist(eval_agent_and_params, tmp_path):
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout_batched(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=3,
        seed=2,
        results_dir=str(tmp_path),
        checkpoint_pct=200,
        render_video=True,
        quiet=True,
    )
    rdir = result['recordings_dir']
    assert rdir is not None
    from pathlib import Path
    rpath = Path(rdir)
    assert rpath.is_dir(), f"recordings_dir {rpath} not created"
    rec_files = list(rpath.glob('episode_*.rec.gz'))
    assert len(rec_files) == 3, f"Expected 3 .rec.gz files, got {len(rec_files)}"
    meta_file = rpath / 'run_meta.pkl'
    assert meta_file.exists(), "run_meta.pkl not written"

    # run_meta.pkl must document the RNG convention explicitly (so downstream
    # consumers never mistake a batched recording for a legacy bit-identical one).
    from src.utils.eval_recording import load_run_meta
    meta = load_run_meta(rpath)
    assert meta['extras']['rollout_mode'] == 'dreamer_batched'
    assert 'independent per-episode' in meta['extras']['rng_convention']


def test_batched_eval_rollout_episode_measures_computable(eval_agent_and_params, tmp_path):
    """The written .rec.gz files must be consumable by the SAME
    episode_measures() function the legacy path's recordings feed
    (scripts/behavior_measures/avoidance_stats_heatmap.py) -- confirms
    format compatibility end-to-end, not just presence of files."""
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout_batched(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=2,
        seed=3,
        results_dir=str(tmp_path),
        checkpoint_pct=300,
        render_video=True,
        quiet=True,
    )
    from pathlib import Path
    sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/behavior_measures')
    from avoidance_stats_heatmap import episode_measures
    from src.utils.eval_recording import load_episode

    rpath = Path(result['recordings_dir'])
    for rec_file in sorted(rpath.glob('episode_*.rec.gz')):
        ep = load_episode(rec_file)
        measures = episode_measures(ep)
        assert set(measures.keys()) == {
            'bush_use_rate', 'bush_entry_step', 'bush_dwell', 'fid',
            'time_near_animal', 'closest_approach', 'time_moving',
            'spatial_spread', 'pursuit_duration', 'injury_change', 'survival_steps',
        }


# ---------------------------------------------------------------------------
# 2. Deterministic cross-check -- vmap machinery correctness, RNG-independent
# ---------------------------------------------------------------------------

def _reference_batch1_step(world_model, actor, env_params, state, recurrent_state,
                            posterior_state, prev_action, is_first, step_key, action_dim):
    """Hand-written batch=1 single step, mirroring dreamer_srl_eval_rollout's
    per-step loop body (src/algorithms/dreamer_srl/eval.py:130-171) exactly,
    but returning (next_state, ..., action) instead of recording/looping.
    """
    obs = get_observation(state, env_params)
    obs_b = jnp.asarray(obs, dtype=jnp.float32)[None, :]
    embedded = jax.vmap(world_model.encoder)(obs_b)

    step_key, k_rssm = jax.random.split(step_key)
    recurrent_state, posterior_state, _, _, _ = world_model.rssm.dynamic(
        posterior_state, recurrent_state, prev_action, embedded, is_first, k_rssm,
    )

    posterior_flat = posterior_state.reshape(1, -1)
    latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)

    logits = actor.forward_logits(latent)
    action = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    next_state, reward, done, _info = jax_step(state, action[0], env_params)

    prev_action_next = jax.nn.one_hot(action, action_dim, dtype=jnp.float32)
    is_first_next = jnp.zeros_like(is_first)

    return (next_state, recurrent_state, posterior_state, prev_action_next,
            is_first_next, step_key, action, reward, done, latent)


def test_batched_scan_matches_reference_step_bit_exact(eval_agent_and_params):
    """Deterministic cross-check (corrected acceptance gate item 3): fix ONE
    reset key + ONE step-key thread, run 5 steps through (a) a hand-written
    batch=1 reference loop and (b) `_dreamer_rollout_scan_jit` (the exact
    vmap+lax.scan body `dreamer_srl_eval_rollout_batched` uses), and confirm
    the per-step actions and latents are bit-identical. This isolates "is the
    vmapped RSSM+actor+env computation correct" from "the RNG streams differ
    across episodes by design" -- the RNG streams here are IDENTICAL (both
    paths consume the same fixed keys in the same order), so any divergence
    would indicate a real bug in the vmap/scan formulation (wrong axis,
    wrong reshape order, wrong is_first broadcast, wrong key-consumption
    order), not expected sampling noise.
    """
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    action_dim = int(actor.action_dim)
    n_steps = 5

    k_reset = jax.random.PRNGKey(123)
    k_step0 = jax.random.PRNGKey(456)

    # --- Reference: hand-written batch=1 loop, step-by-step, nnx.jit-wrapped
    # per call (fresh agent, no checkpoint restore involved -- see module
    # docstring for why the restore-parity trap does not apply here). ---
    ref_step_jit = nnx.jit(_reference_batch1_step, static_argnames=("action_dim",))

    state = jax_reset(env_params, k_reset)
    h, z = world_model.rssm.get_initial_states(1)
    prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)
    is_first = jnp.ones((1, 1), dtype=jnp.float32)
    step_key = k_step0

    ref_actions = []
    ref_latents = []
    for _ in range(n_steps):
        (state, h, z, prev_action, is_first, step_key,
         action, reward, done, latent) = ref_step_jit(
            world_model, actor, env_params, state, h, z, prev_action, is_first,
            step_key, action_dim,
        )
        ref_actions.append(np.asarray(action)[0])
        ref_latents.append(np.asarray(latent)[0])

    # --- Batched machinery: N=1, same reset key (stacked to shape (1,2) via
    # vmap, matching dreamer_srl_eval_rollout_batched's own reset construction)
    # and the SAME step-key thread. ---
    v_reset = jax.vmap(jax_reset, in_axes=(None, 0))
    states0 = v_reset(env_params, k_reset[None, :])
    h0, z0 = world_model.rssm.get_initial_states(1)
    prev_action0 = jnp.zeros((1, action_dim), dtype=jnp.float32)
    is_first0 = jnp.ones((1, 1), dtype=jnp.float32)

    scan_out = nnx.jit(_dreamer_rollout_scan_jit, static_argnames=("action_dim", "max_steps"))(
        world_model, actor, env_params, states0, h0, z0, prev_action0, is_first0,
        k_step0, action_dim, n_steps,
    )

    batched_actions = np.asarray(scan_out["action"])[:, 0]   # [n_steps]

    assert list(batched_actions) == ref_actions, (
        f"Batched scan actions {list(batched_actions)} != reference "
        f"batch=1 actions {ref_actions} for identical fixed keys -- vmap "
        f"machinery bug (wrong axis / reshape / key-consumption order)."
    )
