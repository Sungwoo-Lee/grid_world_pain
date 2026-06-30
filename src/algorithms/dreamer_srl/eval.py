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


def _render_and_upload(
    recordings_dir: str,
    results_dir: str,
    checkpoint_pct: int,
    fps: int,
    wandb_enabled: bool,
    quiet: bool = True,
) -> Optional[str]:
    """Subprocess-render recordings → consolidated MP4, then WandB-upload.

    Near-verbatim port of evaluation_core.py:L289-L321.
    Ported from src/utils/evaluation_core.py:L289-L321.

    Args:
        recordings_dir: path to <results_dir>/recordings/<checkpoint_pct>/.
        results_dir: root results dir (videos written to results_dir/videos/).
        checkpoint_pct: episode label for the video filename.
        fps: frames-per-second for the rendered video.
        wandb_enabled: if True, upload the MP4 to the active WandB run.
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
                episode=checkpoint_pct,
                step=checkpoint_pct,
                caption=f'Episode {checkpoint_pct}',
                quiet=True,
            )
        except Exception as _e:
            print(f'[eval] Warning: WandB video upload failed: {_e}')

    return consolidated if os.path.exists(consolidated) else None
