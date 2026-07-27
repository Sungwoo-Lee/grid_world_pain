"""dreamer_srl eval_rollout — deterministic evaluation episodes.

Mirrors evaluate_jax_checkpoint() from src/utils/evaluation_core.py but
adapted for the dreamer-srl agent interface (WorldModel + Actor via Player).

Key differences from evaluate_jax_checkpoint:
  - No single model() callable; instead encoder + rssm.dynamic + actor.forward_logits.
  - Uses actor.forward_logits() for argmax (deterministic) instead of sampling.
  - Reuses EpisodeRecorder, write_run_meta (already algorithm-agnostic).

Ported from:
  src/utils/evaluation_core.py:L141-L321  (structure + recording helpers)
  src/algorithms/dreamer_srl/dreamer_srl_main.py:Player.get_actions  (RSSM step)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def dreamer_srl_eval_rollout(
    world_model,
    actor,
    env_params,
    config,
    num_episodes: int,
    seed: int,
    results_dir: str,
    checkpoint_pct: int,
    render_video: bool = True,
    quiet: bool = True,
) -> dict:
    """Deterministic single-env eval rollout for dreamer-srl.

    Mirrors evaluate_jax_checkpoint() (src/utils/evaluation_core.py:L141)
    but calls the dreamer-srl agent directly (WorldModel.rssm.dynamic +
    Actor.forward_logits) rather than a generic_inference() shim.

    Action selection: argmax(actor.forward_logits(latent)) — deterministic,
    no Gumbel noise. Ported from evaluation_core.py:L36-L39 (argmax path).

    Args:
        world_model: dreamer-srl WorldModel (encoder + rssm).
        actor: dreamer-srl Actor (forward_logits for deterministic eval).
        env_params: EnvParams for the environment.
        config: merged Config object (for visualization.icons etc.).
        num_episodes: number of eval episodes to run.
        seed: RNG seed (deterministic from this seed).
        results_dir: root results directory (recordings written under here).
        checkpoint_pct: episode count label for this checkpoint.
        render_video: if True, write EpisodeRecorder .rec.gz files.
        quiet: suppress per-episode stdout.

    Returns:
        dict with keys:
          mean_reward, mean_length, episode_rewards, episode_lengths,
          recordings_dir (str or None).
    """
    from src.environment.core import jax_reset, jax_step
    from src.environment.sensor import get_observation
    from src.utils.eval_recording import EpisodeRecorder, write_run_meta

    # ------------------------------------------------------------------
    # Setup recordings dir + run_meta.pkl
    # Ported from evaluation_core.py:L152-L170
    # ------------------------------------------------------------------
    recordings_dir = Path(results_dir) / 'recordings' / str(checkpoint_pct)
    if render_video:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _action_map = ['Up', 'Right', 'Down', 'Left']
        if env_params.rest_action_enabled:
            _action_map.append('Rest')
        if env_params.eat_action_enabled:
            _action_map.append('Eat')
        icon_config = config.get('visualization.icons', None)
        write_run_meta(
            recordings_dir,
            env_params,
            icon_config,
            _action_map,
            getattr(config, 'source_path', ''),
            extras={'checkpoint_pct': checkpoint_pct, 'seed': seed},
        )

    # ------------------------------------------------------------------
    # Per-episode loop — deterministic
    # Ported from evaluation_core.py:L338-L455 (_run_single_env_eval)
    # ------------------------------------------------------------------
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)
    key = jax.random.PRNGKey(seed)
    episode_rewards: list = []
    episode_lengths: list = []

    for ep_idx in range(num_episodes):
        key, k_reset = jax.random.split(key)
        state = jax_reset(env_params, k_reset)
        obs = get_observation(state, env_params)

        # Initialise RSSM state — mirrors Player.init_states() all-envs path.
        # Ported from dreamer_srl_main.py:Player.init_states L77-L81
        h0, z0 = world_model.rssm.get_initial_states(1)
        recurrent_state = h0    # [1, recurrent_state_size]
        posterior_state = z0    # [1, num_cat, num_cls]
        prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)
        is_first = jnp.ones((1, 1), dtype=jnp.float32)

        # EpisodeRecorder — step 0 is the initial state (action=-1, reward=0)
        # Ported from evaluation_core.py:L390-L396
        recorder: Optional[EpisodeRecorder] = None
        if render_video:
            recorder = EpisodeRecorder(ep_idx + 1, checkpoint_pct, seed)
            recorder.append(
                jax.device_get(state),
                obs,
                None,        # true_obs not recorded for dreamer-srl (no noise API)
                action_idx=-1,
                reward=0.0,
            )

        total_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < int(env_params.max_steps):
            # -----------------------------------------------------------
            # RSSM step — mirrors Player.get_actions L120-L147
            # Ported from dreamer_srl_main.py:L120-L147
            # -----------------------------------------------------------
            obs_b = jnp.asarray(obs, dtype=jnp.float32)[None, :]   # [1, obs_dim]
            embedded = jax.vmap(world_model.encoder)(obs_b)         # [1, dense_units]

            key, k_rssm = jax.random.split(key)
            recurrent_state, posterior_state, _, _, _ = world_model.rssm.dynamic(
                posterior_state,    # [1, S, D]
                recurrent_state,    # [1, recurrent_state_size]
                prev_action,        # [1, action_dim]
                embedded,           # [1, dense_units]
                is_first,           # [1, 1]
                k_rssm,
            )

            # Build latent: cat(posterior_flat, recurrent_state)
            # Ported from dreamer_srl_main.py:L139-L141
            posterior_flat = posterior_state.reshape(1, -1)                       # [1, S*D]
            latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)  # [1, latent_dim]

            # DETERMINISTIC action = argmax(post-unimix logits)
            # Mirrors evaluation_core.py:L36-L39 argmax path
            logits = actor.forward_logits(latent)   # [1, action_dim]
            action_idx = int(jnp.argmax(logits, axis=-1)[0])
            prev_action = jax.nn.one_hot(
                jnp.array([action_idx]), action_dim, dtype=jnp.float32
            )   # [1, action_dim]
            is_first = jnp.zeros((1, 1), dtype=jnp.float32)

            # -----------------------------------------------------------
            # Step environment
            # Ported from evaluation_core.py:L439
            # -----------------------------------------------------------
            next_state, reward, done, _info = jax_step(state, action_idx, env_params)
            total_reward += float(reward)
            step_count += 1
            state = next_state
            obs = get_observation(state, env_params)

            if recorder is not None:
                recorder.append(
                    jax.device_get(state),
                    obs,
                    None,
                    action_idx=action_idx,
                    reward=float(reward),
                )

        # Write episode recording
        if recorder is not None:
            ep_path = recordings_dir / f'episode_{ep_idx + 1:06d}.rec.gz'
            recorder.write(ep_path)

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        if not quiet:
            print(f'[eval] ep {ep_idx + 1}/{num_episodes}: '
                  f'reward={total_reward:.2f} steps={step_count}')

    # ------------------------------------------------------------------
    # Return summary
    # Mirrors evaluation_core.py return contract
    # ------------------------------------------------------------------
    return {
        'mean_reward':     float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'mean_length':     float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'recordings_dir':  str(recordings_dir) if render_video else None,
    }


def _dreamer_rollout_scan_jit(world_model, actor, env_params, states0, h0, z0,
                               prev_action0, is_first0, step_key0,
                               action_dim, max_steps):
    """nnx.jit-wrapped scan body for the batched Dreamer eval rollout (Tier 2).

    MUST be entered via `nnx.jit`, not called eagerly, and MUST NOT be wrapped in a
    bare `jax.lax.scan` with no enclosing `nnx.jit` anywhere in the call stack.
    See `scripts/eval/eval_rollout.py::_rollout_scan_jit`'s docstring for the full
    root-cause argument (found empirically while building the rPPO batched eval):
    after `nnx.update(model, restored_tree)` restores a checkpoint, an EAGER
    `model(...)` call (or a bare `lax.scan` around one) reads a stale/inconsistent
    view of the restored parameters -- only an `nnx.jit`-traced read correctly
    materializes them. `world_model` and `actor` here are nnx.Module instances
    restored via `nnx.update(...)` in exactly the same way, so the same trap
    applies identically.

    Per-step body mirrors `Player.get_actions` (dreamer_srl_main.py:304-352) --
    the training-time rollout-collection path, which already runs encoder +
    rssm.dynamic + actor forward at batch=num_envs with ONE shared PRNG key per
    call (not one key per env). We reuse that exact convention here: `step_key0`
    is ONE evolving key thread, split once per scan step and applied to the
    WHOLE batch of N episodes via `world_model.rssm.dynamic(..., key=k_rssm)` --
    this is the RSSM's native, already-tested batched interface, not a per-env
    vmapped key split. The only departure from Player.get_actions is the action
    head: eval uses `actor.forward_logits(latent)` + deterministic argmax (no
    Gumbel-softmax sampling), matching `dreamer_srl_eval_rollout`'s single-env
    action selection.

    `is_first` is 1.0 only at t=0 (the initial RSSM state, set by the caller)
    and 0.0 for every subsequent scan step -- unlike a live training rollout,
    a dead episode is never "auto-reset" mid-scan here; steps recorded after an
    episode's death are simply discarded by the caller via per-episode T[i]
    slicing (mirrors `_run_episodes_batched`'s `T = argmax(done_seq, axis=0)+1`
    in scripts/eval/eval_rollout.py), so post-death padding steps need no
    special-cased is_first handling.
    """
    from src.environment.core import jax_step
    from src.environment.sensor import get_observation

    v_step = jax.vmap(jax_step, in_axes=(0, 0, None))
    v_obs = jax.vmap(get_observation, in_axes=(0, None))

    def scan_fn(carry, _):
        state, recurrent_state, posterior_state, prev_action, is_first, step_key = carry

        obs = v_obs(state, env_params)                          # [N, obs_dim] -- pre-step obs
        embedded = jax.vmap(world_model.encoder)(obs)             # [N, dense_units]

        step_key, k_rssm = jax.random.split(step_key)
        recurrent_state, posterior_state, _prior, _post_logits, _prior_logits = world_model.rssm.dynamic(
            posterior_state, recurrent_state, prev_action, embedded, is_first, k_rssm,
        )

        posterior_flat = posterior_state.reshape(posterior_state.shape[0], -1)   # [N, S*D]
        latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)      # [N, latent_dim]

        logits = actor.forward_logits(latent)                     # [N, action_dim]
        action = jnp.argmax(logits, axis=-1).astype(jnp.int32)    # [N] deterministic

        next_state, reward, done, _info = v_step(state, action, env_params)
        next_obs = v_obs(next_state, env_params)

        prev_action_next = jax.nn.one_hot(action, action_dim, dtype=jnp.float32)
        is_first_next = jnp.zeros_like(is_first)

        step_out = {
            "action": action,
            "reward": reward,
            "done": done,
            "snap_agent_pos": next_state.agent_pos,
            "snap_satiation": next_state.satiation,
            "snap_nutrition": next_state.nutrition,
            "snap_injury_level": next_state.injury_level,
            "snap_rest_streak": next_state.rest_streak,
            "snap_res_pos": next_state.res_pos,
            "snap_res_active": next_state.res_active,
            "snap_animal_pos": next_state.animal_pos,
            "snap_obs_pos": next_state.obs_pos,
            "obs": next_obs,
        }
        new_carry = (next_state, recurrent_state, posterior_state, prev_action_next,
                     is_first_next, step_key)
        return new_carry, step_out

    init_carry = (states0, h0, z0, prev_action0, is_first0, step_key0)
    _final_carry, scan_out = jax.lax.scan(scan_fn, init_carry, None, length=max_steps)
    return scan_out


def dreamer_srl_eval_rollout_batched(
    world_model,
    actor,
    env_params,
    config,
    num_episodes: int,
    seed: int,
    results_dir: str,
    checkpoint_pct: int,
    render_video: bool = True,
    quiet: bool = True,
) -> dict:
    """Batched (vmapped) sibling of `dreamer_srl_eval_rollout` -- ALL `num_episodes`
    episodes run as ONE vmapped batch + `jax.lax.scan` over `max_steps`, instead of
    a Python per-episode loop. Same signature, same return schema, same `.rec.gz`
    recording format (episode_measures / render_recordings.py consume it unchanged).
    Additive only -- `dreamer_srl_eval_rollout` is untouched and remains the
    training-time eval path.

    RNG convention -- DELIBERATELY DIFFERENT from the legacy single-env path,
    per the design decision recorded in the implementation report (developer/
    coordinator exchange, dated during this function's construction):

      dreamer_srl_eval_rollout (legacy): ONE master key threaded SEQUENTIALLY
      across ALL episodes AND all of their steps (`key, subkey =
      jax.random.split(key)` fires once per reset and once per step, carrying
      forward from episode i into episode i+1's reset). Episode i+1's starting
      randomness is therefore NOT a precomputable function of the episode index
      alone -- it depends on how many steps episode i happened to take, which is
      itself a stochastic outcome of that trajectory (the RSSM's posterior
      sampling consumes a key every step; the eval "argmax" is only deterministic
      GIVEN that stochastic latent). This makes bit-exact reproduction by any
      real (independent-stream) vmapped batch structurally impossible for N>1
      episodes -- vmap requires each episode's RNG stream to be knowable up
      front, not entangled with a sibling episode's emergent trajectory length.

      dreamer_srl_eval_rollout_batched (this function): N INDEPENDENT per-episode
      reset keys via `jax.random.split(jax.random.PRNGKey(seed), num_episodes)`
      (mirrors the rPPO batched path's "each episode is a pure function of its
      own reset key" argument). The RSSM/actor's per-step stochastic sampling
      uses ONE shared step-key thread applied across the WHOLE batch at each
      scan step -- this is the SAME convention `Player.get_actions`
      (dreamer_srl_main.py:304-352) already uses natively during training
      rollout collection at batch=num_envs, so it is not a novel/untested key
      pattern.

      Net effect: same `seed` reproduces the same DISTRIBUTION of behavior
      (same aggregate statistics within Monte Carlo error), NOT bit-identical
      per-episode trajectories vs. the legacy function. This is recorded
      explicitly in the recordings' `run_meta.pkl` `extras['rng_convention']`
      field so downstream consumers are never misled into diffing trajectories
      episode-by-episode against a legacy-path recording.

    Args: identical to `dreamer_srl_eval_rollout` (see that docstring).

    Returns:
        dict with keys: mean_reward, mean_length, episode_rewards,
        episode_lengths, recordings_dir (str or None) -- identical schema to
        `dreamer_srl_eval_rollout`.
    """
    import flax.nnx as nnx
    from types import SimpleNamespace
    from src.environment.core import jax_reset
    from src.environment.sensor import get_observation
    from src.utils.eval_recording import EpisodeRecorder, write_run_meta, _snapshot_state

    N = int(num_episodes)
    max_steps = int(env_params.max_steps)
    action_dim = int(actor.action_dim)

    # ------------------------------------------------------------------
    # Setup recordings dir + run_meta.pkl (same layout as the legacy path;
    # extras carries the RNG-convention note above so results are never
    # mistaken for legacy-path bit-identical trajectories).
    # ------------------------------------------------------------------
    recordings_dir = Path(results_dir) / 'recordings' / str(checkpoint_pct)
    if render_video:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _action_map = ['Up', 'Right', 'Down', 'Left']
        if env_params.rest_action_enabled:
            _action_map.append('Rest')
        if env_params.eat_action_enabled:
            _action_map.append('Eat')
        icon_config = config.get('visualization.icons', None)
        write_run_meta(
            recordings_dir,
            env_params,
            icon_config,
            _action_map,
            getattr(config, 'source_path', ''),
            extras={
                'checkpoint_pct': checkpoint_pct,
                'seed': seed,
                'rollout_mode': 'dreamer_batched',
                'rng_convention': (
                    'independent per-episode reset keys via '
                    'jax.random.split(PRNGKey(seed), num_episodes); RSSM/actor '
                    'stochastic sampling uses ONE shared step-key thread applied '
                    'across the whole batch at each scan step (matches the '
                    'native training-time Player.get_actions batched '
                    'convention). DIFFERENT RNG scheme from '
                    'dreamer_srl_eval_rollout (legacy single-env path threads '
                    'ONE master key sequentially across ALL episodes) -- same '
                    'seed therefore reproduces the same behavior DISTRIBUTION, '
                    'not bit-identical per-episode trajectories.'
                ),
            },
        )

    # ------------------------------------------------------------------
    # RNG setup: N independent per-episode reset keys + one shared step-key
    # thread (see docstring above).
    # ------------------------------------------------------------------
    master_key = jax.random.PRNGKey(seed)
    reset_master, step_master = jax.random.split(master_key)
    episode_reset_keys = jax.random.split(reset_master, N)   # [N, 2]

    v_reset = jax.vmap(jax_reset, in_axes=(None, 0))
    v_obs = jax.vmap(get_observation, in_axes=(0, None))

    states0 = v_reset(env_params, episode_reset_keys)

    h0, z0 = world_model.rssm.get_initial_states(N)
    prev_action0 = jnp.zeros((N, action_dim), dtype=jnp.float32)
    is_first0 = jnp.ones((N, 1), dtype=jnp.float32)

    scan_out = nnx.jit(_dreamer_rollout_scan_jit, static_argnames=("action_dim", "max_steps"))(
        world_model, actor, env_params, states0, h0, z0, prev_action0, is_first0,
        step_master, action_dim, max_steps,
    )
    assert scan_out["action"].shape == (max_steps, N), (
        f"Expected batched action shape (max_steps={max_steps}, N={N}), "
        f"got {scan_out['action'].shape}"
    )
    assert bool(jnp.all(scan_out["action"] < action_dim)), (
        f"Batched action indices out of range for action_dim={action_dim} -- "
        "argmax likely reduced over the wrong axis."
    )

    # Bring everything to host ONCE.
    scan_out = jax.tree_util.tree_map(np.asarray, scan_out)
    obs0_np = np.asarray(v_obs(states0, env_params))
    states0_np = jax.tree_util.tree_map(np.asarray, states0)

    done_seq = scan_out["done"]   # (max_steps, N) bool
    # `max_steps` truncation guarantees every env reaches done=True within the
    # scan window (core.py: truncated = next_step >= params.max_steps fires on
    # the max_steps-th step call at the latest), so argmax always finds a hit.
    T = np.argmax(done_seq, axis=0) + 1   # (N,) -- episode length (death step)

    episode_rewards: list = []
    episode_lengths: list = []

    for i in range(N):
        Ti = int(T[i])
        total_reward = float(np.sum(scan_out["reward"][:Ti, i]))
        episode_rewards.append(total_reward)
        episode_lengths.append(Ti)

        if render_video:
            recorder = EpisodeRecorder(i + 1, checkpoint_pct, seed)

            # Initial snapshot (pre-loop) -- matches
            # dreamer_srl_eval_rollout's recorder.append(state, obs, None,
            # action_idx=-1, reward=0.0) call.
            init_state = SimpleNamespace(
                agent_pos=states0_np.agent_pos[i], satiation=states0_np.satiation[i],
                nutrition=states0_np.nutrition[i], injury_level=states0_np.injury_level[i],
                rest_streak=states0_np.rest_streak[i], res_pos=states0_np.res_pos[i],
                res_active=states0_np.res_active[i], animal_pos=states0_np.animal_pos[i],
                obs_pos=states0_np.obs_pos[i],
            )
            recorder.snapshots.append(_snapshot_state(init_state))
            recorder.obs.append(np.asarray(obs0_np[i]))
            recorder.true_obs.append(None)   # true_obs never recorded for dreamer-srl (no noise API)
            recorder.actions.append(-1)
            recorder.rewards.append(0.0)

            for t in range(Ti):
                step_state = SimpleNamespace(
                    agent_pos=scan_out["snap_agent_pos"][t, i],
                    satiation=scan_out["snap_satiation"][t, i],
                    nutrition=scan_out["snap_nutrition"][t, i],
                    injury_level=scan_out["snap_injury_level"][t, i],
                    rest_streak=scan_out["snap_rest_streak"][t, i],
                    res_pos=scan_out["snap_res_pos"][t, i],
                    res_active=scan_out["snap_res_active"][t, i],
                    animal_pos=scan_out["snap_animal_pos"][t, i],
                    obs_pos=scan_out["snap_obs_pos"][t, i],
                )
                recorder.snapshots.append(_snapshot_state(step_state))
                recorder.obs.append(np.asarray(scan_out["obs"][t, i]))
                recorder.true_obs.append(None)
                recorder.actions.append(int(scan_out["action"][t, i]))
                recorder.rewards.append(float(scan_out["reward"][t, i]))

            ep_path = recordings_dir / f'episode_{i + 1:06d}.rec.gz'
            recorder.write(ep_path)

        if not quiet:
            print(f'[eval-batched] ep {i + 1}/{N}: '
                  f'reward={total_reward:.2f} steps={Ti}')

    return {
        'mean_reward':     float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'mean_length':     float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'recordings_dir':  str(recordings_dir) if render_video else None,
    }


def _render_and_upload(
    recordings_dir: str,
    results_dir: str,
    checkpoint_pct: int,
    fps: int,
    wandb_enabled: bool,
    policy_step: int,
    quiet: bool = True,
) -> Optional[str]:
    """Subprocess-render recordings → consolidated MP4, then WandB-upload.

    Near-verbatim port of evaluation_core.py:L289-L321.
    Ported from src/utils/evaluation_core.py:L289-L321.

    NOTE: this is the BLOCKING render path — kept as the in-training fallback
    (kill-switch `training.async_video_render: false`) and unused otherwise.
    The default in-training path is the non-blocking dispatch/poll/drain in
    src/utils/async_render.py (docs/develop/active/refactors/
    ASYNC_CHECKPOINT_VIDEO_RENDER.md), which Popens the exact same command.

    Args:
        recordings_dir: path to <results_dir>/recordings/<checkpoint_pct>/.
        results_dir: root results dir (videos written to results_dir/videos/).
        checkpoint_pct: episode label for the video filename AND the
            `eval/checkpoint_episode` payload (episode count, NOT the WandB
            step axis).
        fps: frames-per-second for the rendered video.
        wandb_enabled: if True, upload the MP4 to the active WandB run.
        policy_step: WandB step axis (env-step clock). The run's dashboard
            timeline is driven by `policy_step`, not the episode count, so
            the upload must be stamped with `step=policy_step` — otherwise
            it lands backward on the timeline and WandB silently drops it.
        quiet: suppress subprocess stdout (stderr still printed on failure).

    Returns:
        Path to the consolidated MP4, or None if render failed.
    """
    _project_root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    render_script = os.path.join(_project_root, 'scripts', 'eval', 'render_recordings.py')
    consolidated = os.path.join(results_dir, 'videos', f'eval_{checkpoint_pct}.mp4')

    cmd = [
        sys.executable,
        render_script,
        str(recordings_dir),
        '--concat',
        '--skip-existing',
        '--cleanup-per-episode',
        '--fps', str(fps),
    ]
    # Renderer must run on CPU only (matplotlib; avoids GPU OOM with training process)
    # Ported from evaluation_core.py:L289-L295
    child_env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}

    result = subprocess.run(
        cmd,
        env=child_env,
        capture_output=quiet,
        text=True,
    )
    if result.returncode != 0:
        print(f'[eval] Warning: render_recordings.py failed '
              f'(rc={result.returncode}). stderr: '
              f'{(result.stderr or "")[:500]}')
        return None

    # WandB upload — mirrors evaluation_core.py:L318-L321
    if wandb_enabled and os.path.exists(consolidated):
        try:
            from src.utils.wandb_utils import upload_video
            upload_video(
                consolidated,
                episode=checkpoint_pct,   # -> eval/checkpoint_episode payload (episode label)
                step=policy_step,         # env-step clock: forward step, no longer dropped
                caption=f'Episode {checkpoint_pct}',
                quiet=True,
            )
        except Exception as _e:
            print(f'[eval] Warning: WandB video upload failed: {_e}')

    return consolidated if os.path.exists(consolidated) else None
