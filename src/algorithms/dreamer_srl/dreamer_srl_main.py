"""dreamer_srl_main.py — training-loop driver for dreamer-srl v3.

Ports vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L361-L765 (main())
to JAX/Flax-NNX, stripped per v3 §B:
  - No Hydra / Fabric / RestartOnException / MLflow / memmap / all_gather
  - ParallelEnv replaces gym.vector.SyncVectorEnv
  - Single observation key "obs" (vector obs, no CNN)
  - WandB replaced with optional wandb; --no-wandb runs silently

Usage::

    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        src/algorithms/dreamer_srl/dreamer_srl_main.py \\
        --env-config configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml \\
        --agent-config configs/models/dreamer_srl/01_food_only.yaml \\
        --total-steps 5000 --num-envs 1 --seed 0 \\
        --wandb-project grid_world_pain \\
        --wandb-name dreamer_srl_v3_schema_migration_smoke_s0

CP9 note: learning_starts=0 (D-012 pre-declared in DEVIATION_LOG.md).
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

# Project imports
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
from src.utils.config import Config, dump_config_yaml as _dump_config_yaml
from src.environment.config_loader import load_env_params, load_env_config
from src.environment.wrapper import ParallelEnv
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.buffers import (
    EnvIndependentSequentialReplayBuffer,
    SequentialReplayBuffer,
    validate_per_env_capacity,
)
from src.algorithms.dreamer_srl.loss import TwoHotEncoding
from src.algorithms.dreamer_srl.train import make_train_step, polyak_update
from src.algorithms.dreamer_srl.utils import (
    ACTOR_CLIP_NORM,
    CRITIC_CLIP_NORM,
    WM_CLIP_NORM,
    Ratio,
    derive_prefill,
    make_optim_tx,
    moments_init,
)


# ---------------------------------------------------------------------------
# ContinualSchedule — curriculum engine (ported from train.py:134-201)
# ---------------------------------------------------------------------------

@dataclass
class ContinualSchedule:
    """Describes a multi-stage curriculum: one env config + schedule per stage.

    Ported from train.py:L135-L153 with adaptation for dreamer-srl's split
    env-config / agent-config style (stage configs are env configs only).

    Attributes:
        stage_config_paths: Absolute paths to stage env YAML files (alphabetic).
        stage_names: Basename without extension (e.g. "01_5x5_food_hide_rock").
        stage_configs: Fully-merged Config objects, one per stage.
        episode_boundaries: Cumulative episode count ending each stage
            (strictly increasing; last = total episodes).
        checkpoint_frequencies: Checkpoint-save period (in episodes) per stage.
    """
    stage_config_paths: List[str]
    stage_names: List[str]
    stage_configs: List  # List[Config]
    episode_boundaries: List[int]
    checkpoint_frequencies: List[int]

    @property
    def num_stages(self) -> int:
        return len(self.stage_config_paths)

    def stage_for_episode(self, episode: int) -> int:
        """Return the stage index that owns `episode` (0-based).

        Mirrors train.py:L146-L153 stage_for_episode logic.
        """
        for i, b in enumerate(self.episode_boundaries):
            if episode < b:
                return i
        return self.num_stages - 1


def _load_stage_env_cfg(project_root: str, stage_yaml_path: str):
    """Build a full env Config for one stage: defaults + stage YAML.

    Mirrors dreamer_srl_main.py single-config merge (lines 221-234), applied
    per stage. Returns a merged Config object ready for load_env_params().

    Args:
        project_root: Absolute path to the repo root.
        stage_yaml_path: Absolute path to the stage env YAML.

    Returns:
        Config: Fully merged env config for this stage.
    """
    import os as _os
    from src.utils.config import get_default_config, Config as _Config
    from src.environment.config_loader import load_env_config as _load_env_config
    cfg = get_default_config()  # seed base (archived/standalone stage configs rely on this)
    for rel in [
        'configs/train/default.yaml',
        'configs/evaluation/default.yaml',
        'configs/visualization/default.yaml',
    ]:
        p = _os.path.join(project_root, rel)
        if _os.path.exists(p):
            cfg.merge(_Config.load_yaml(p))
    # load_env_config: if stage YAML has `extends:`, its base is merged inside;
    # if standalone (archived), it loads as-is and merges on top of defaults.
    cfg.merge(_load_env_config(stage_yaml_path))
    return cfg


def _build_continual_schedule(project_root: str, configs_dir: str,
                               schedule_path: str) -> ContinualSchedule:
    """Discover stage YAMLs + load + validate the schedule file.

    Ported from train.py:L154-L201 _build_continual_schedule with adaptation:
    stage Config is built by _load_stage_env_cfg (dreamer-srl merge style)
    instead of train.py's single-config clone.

    Args:
        project_root: Absolute path to the repo root.
        configs_dir: Directory containing stage env YAML files (loaded
            alphabetically — prefix 01_, 02_, ...).
        schedule_path: Path to schedule YAML with
            continual.episode_boundaries and continual.checkpoint_frequencies.

    Returns:
        ContinualSchedule: Validated schedule ready for use.

    Raises:
        ValueError: If directory missing, no YAMLs found, schedule YAML
            invalid, lengths mismatched, or boundaries non-monotonic.
    """
    import os as _os
    from src.utils.config import Config as _Config

    if not _os.path.isdir(configs_dir):
        raise ValueError(f"--configs-dir '{configs_dir}' is not a directory.")
    paths = sorted(glob.glob(_os.path.join(configs_dir, "*.yaml")))
    if not paths:
        raise ValueError(f"No *.yaml files found in {configs_dir}.")
    names = [_os.path.splitext(_os.path.basename(p))[0] for p in paths]

    # Load + validate schedule YAML — ported from train.py:L165-L186
    sched_cfg = _Config.load_yaml(schedule_path)
    boundaries = sched_cfg.get_mandatory("continual.episode_boundaries")
    ckpt_freqs = sched_cfg.get_mandatory("continual.checkpoint_frequencies")

    if len(boundaries) != len(paths):
        raise ValueError(
            f"episode_boundaries length ({len(boundaries)}) != "
            f"number of stage configs ({len(paths)})."
        )
    if len(ckpt_freqs) != len(paths):
        raise ValueError(
            f"checkpoint_frequencies length ({len(ckpt_freqs)}) != "
            f"number of stage configs ({len(paths)})."
        )
    if sorted(boundaries) != list(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(
            f"episode_boundaries must be strictly increasing: {boundaries}"
        )
    if boundaries[0] <= 0:
        raise ValueError(
            f"episode_boundaries[0] must be > 0 (got {boundaries[0]})."
        )
    if any(f <= 0 for f in ckpt_freqs):
        raise ValueError(
            f"checkpoint_frequencies must all be > 0: {ckpt_freqs}"
        )

    stage_configs = [_load_stage_env_cfg(project_root, p) for p in paths]
    return ContinualSchedule(
        stage_config_paths=paths,
        stage_names=names,
        stage_configs=stage_configs,
        episode_boundaries=list(boundaries),
        checkpoint_frequencies=list(ckpt_freqs),
    )


# ---------------------------------------------------------------------------
# Player — stateful inference wrapper (single env)
# Sheeprl reference: agent.py:L596-L691 PlayerDV3.get_actions / init_states
# ---------------------------------------------------------------------------

class Player:
    """Stateful inference wrapper for a single-env dreamer agent.

    Maintains (recurrent_state, posterior_state) across env steps.
    Call init_states() to reset (e.g. after episode done).

    Sheeprl reference: agent.py:L596-L691 PlayerDV3.
    """

    def __init__(self, world_model, actor, num_envs: int) -> None:
        self.world_model = world_model
        self.actor = actor
        self.num_envs = num_envs
        self._recurrent_state: Optional[jax.Array] = None
        self._posterior_state: Optional[jax.Array] = None
        self._prev_action: Optional[jax.Array] = None

    def init_states(self, reset_envs=None, done_mask: Optional[np.ndarray] = None) -> None:
        """Reset player recurrent + posterior state.

        Sheeprl PlayerDV3.init_states — reset all envs if reset_envs is None.
        Called at startup and after done episodes.

        Fix 1 (recompile-storm): when done_mask is provided (preferred path for the
        training loop), use a fixed-width boolean-mask reset over all num_envs.
        This ensures get_initial_states(num_envs) is called with a CONSTANT batch
        size, eliminating the variable-shape XLA recompiles that fired when
        get_initial_states(n_done) was called with n_done in 1..16 on desynchronized
        episodes.

        Idiom mirrors rPPO's jnp.where(done, reset, keep) pattern — same outcome,
        fixed shape.  Bit-for-bit equivalent to the old scatter (only the done slots
        receive h0; living slots keep their exact values).

        done_mask: bool/float array of shape [num_envs] — 1.0 for done envs.
                   When provided, reset_envs is ignored.
        reset_envs: legacy list of done env indices.  Used only when done_mask
                    is None (full-reset path on curriculum stage transition or startup).
        """
        if done_mask is not None:
            # Fixed-width masked reset (Fix 1 — recompile-storm fix).
            # get_initial_states(num_envs) is constant-shape → compiled once.
            h0_full, z0_full = self.world_model.rssm.get_initial_states(self.num_envs)
            action_dim = self.actor.action_dim

            mask = jnp.asarray(done_mask, dtype=jnp.float32)  # [B]

            # Recurrent state: [B, recurrent_state_size]
            m_h = mask[:, None]  # [B, 1]
            self._recurrent_state = (1.0 - m_h) * self._recurrent_state + m_h * h0_full

            # Posterior state: [B, num_cat, num_cls]
            m_z = mask[:, None, None]  # [B, 1, 1]
            self._posterior_state = (1.0 - m_z) * self._posterior_state + m_z * z0_full

            # Previous action: [B, action_dim]
            m_a = mask[:, None]  # [B, 1]
            zeros_a = jnp.zeros((self.num_envs, action_dim), dtype=jnp.float32)
            self._prev_action = (1.0 - m_a) * self._prev_action + m_a * zeros_a
            return

        if reset_envs is None:
            # Reset all envs (startup / curriculum stage transition)
            h0, z0 = self.world_model.rssm.get_initial_states(self.num_envs)
            self._recurrent_state = h0   # [B, recurrent_state_size]
            self._posterior_state = z0   # [B, num_cat, num_cls]
            action_dim = self.actor.action_dim
            self._prev_action = jnp.zeros((self.num_envs, action_dim), dtype=jnp.float32)
        else:
            # Legacy path: reset only the done envs via index scatter.
            # Kept for any caller that passes reset_envs without done_mask;
            # the hot training-loop path now uses done_mask instead.
            n_done = len(reset_envs)
            if n_done == 0:
                return
            h0_reset, z0_reset = self.world_model.rssm.get_initial_states(n_done)
            action_dim = self.actor.action_dim

            # Scatter reset states into the full batch
            h = self._recurrent_state
            z = self._posterior_state
            a = self._prev_action

            idx = jnp.array(reset_envs)
            h = h.at[idx].set(h0_reset)
            z = z.at[idx].set(z0_reset)
            a = a.at[idx].set(jnp.zeros((n_done, action_dim), dtype=jnp.float32))

            self._recurrent_state = h
            self._posterior_state = z
            self._prev_action = a

    def get_actions(self, obs: np.ndarray, is_first: np.ndarray, key: jax.Array) -> np.ndarray:
        """Get actions from the current observation.

        Runs encoder + RSSM dynamic step + actor forward.
        Updates internal (recurrent_state, posterior_state) in-place.

        Sheeprl PlayerDV3.get_actions L661-L691.

        Args:
            obs: [B, obs_dim] numpy array
            is_first: [B, 1] numpy array (1.0 on episode start)
            key: PRNG key

        Returns:
            actions: [B, action_dim] numpy array (one-hot)
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        is_first_jax = jnp.asarray(is_first, dtype=jnp.float32)

        B = obs_jax.shape[0]

        # Encode obs (symlog applied inside MLPEncoder)
        embedded_obs = jax.vmap(self.world_model.encoder)(obs_jax)  # [B, dense_units]

        # RSSM dynamic step (sheeprl PlayerDV3 L679-L688)
        key, k_rssm = jax.random.split(key)
        recurrent_state, posterior_state, _, _, _ = self.world_model.rssm.dynamic(
            self._posterior_state,  # [B, S, D]
            self._recurrent_state,  # [B, recurrent_state_size]
            self._prev_action,      # [B, action_dim]
            embedded_obs,           # [B, dense_units]
            is_first_jax,           # [B, 1]
            k_rssm,
        )

        # Build latent state: cat(posterior_flat, recurrent_state)
        posterior_flat = posterior_state.reshape(B, -1)  # [B, S*D]
        latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)  # [B, latent_dim]

        # Actor forward
        key, k_act = jax.random.split(key)
        actions, _, _ = self.actor(latent, k_act)  # [B, action_dim]

        # Update internal state
        self._recurrent_state = recurrent_state
        self._posterior_state = posterior_state
        self._prev_action = actions

        return np.asarray(actions)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _reset_terminal_step_data(step_data: dict, dones_idxes: list) -> None:
    """Zero the staged reward/termination flags and set is_first for done envs.

    Restores sheeprl's post-done "Reset already inserted step data" block, which
    the JAX port previously dropped (only the is_first set was ported). Without
    this, the next row written for a done env inherits the *previous* episode's
    terminal reward and death flag while showing the fresh reset observation.

    In-place, pure numpy. Zeroes ONLY the done-env columns — non-done envs keep
    their live in-flight reward/flag values for their own next buffer.add.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L653-L656.
    Fix for H5 — see docs/develop/active/issues/diag_fable5_20260704/
    fix_plan_h5_dreamer_srl_buffer_reset.md
    """
    step_data["rewards"][:, dones_idxes]    = 0.0
    step_data["terminated"][:, dones_idxes] = 0.0
    step_data["truncated"][:, dones_idxes]  = 0.0
    step_data["is_first"][:, dones_idxes]   = 1.0


def _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones,
                              *, stage_swapped: bool = False) -> None:
    """WP-SRL P5: advance per-env episode counters, excluding envs done this iter.

    In-place, pure numpy. The done-block logging already counts the in-flight
    terminal transition (`ep_len = counters + 1`, `ep_rew = counters +
    rewards[i]`) and zeroes the done envs' counters; the previous unconditional
    `episode_lengths += 1; episode_rewards += rewards` then leaked +1 survival
    step (the project's headline metric) and the terminal reward into the
    SUCCESSOR episode's counters ([[00_master_comparison]] §3 P5, area report
    04 D-06). sheeprl needs no counters — it reads the gym wrapper's
    `final_info["episode"]` (dreamer_v3.py:610-618).

    C1 (review_srl_parity_fixes.md): `stage_swapped=True` no-ops the advance.
    On a curriculum stage-transition iteration the swap block has already
    wiped ALL envs' counters (partial episodes deliberately dropped); the
    pre-swap step's +1/reward must NOT be credited to the new stage's fresh
    counters — consistent with the behavior/dist accumulators, which are
    wiped at the same point and receive no post-wipe credit.

    Regression test: tests/algorithms/dreamer_srl/test_episode_metrics.py
    """
    if stage_swapped:
        return
    alive = ~np.asarray(dones, dtype=bool)
    episode_lengths[alive] += 1
    episode_rewards[alive] += rewards[alive].astype(np.float32)


def main() -> None:
    """Training-loop driver — port of sheeprl dreamer_v3.py:L361-L765 main()."""

    # -----------------------------------------------------------------------
    # 1. CLI args
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="dreamer-srl v3 training driver")
    parser.add_argument("--env-config", type=str, default=None,
                        help="Path to env YAML config (single-config mode). "
                             "Mutually exclusive with --configs-dir.")
    parser.add_argument("--agent-config", type=str, required=True,
                        help="Path to dreamer-srl agent YAML config")
    parser.add_argument("--configs-dir", type=str, default=None,
                        help="Directory of stage env-config YAMLs for curriculum learning. "
                             "Loaded alphabetically (prefix 01_, 02_, ...). "
                             "Mutually exclusive with --env-config / --episodes / --total-steps.")
    parser.add_argument("--continual-schedule", type=str, default=None,
                        help="Schedule YAML (continual.episode_boundaries, "
                             "continual.checkpoint_frequencies). Required with --configs-dir.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Primary stop condition: total episodes to complete. "
                             "Mirrors train.py rPPO + JAX Dreamer-V3 dual-mode "
                             "pattern (train.py:1169). When set, --total-steps / "
                             "--total-timesteps are ignored unless --episodes=0.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Env-step fallback budget. Used only when --episodes "
                             "is not passed AND training.episodes is not in the env "
                             "config. Default: None (no env-step override).")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Alias for --total-steps for rPPO CLI compat (train.py:235).")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=None,
                        help="WandB logging interval in iterations (overrides config). "
                             "Mirrors train.py:242 rPPO CLI. Priority: CLI > "
                             "agent_cfg.training.log_interval > env_cfg.training.log_interval > default(50).")
    parser.add_argument("--wandb-project", type=str,
                        default="grid_world_pain",
                        help="WandB project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (for Checkpoint D micro-dry-run)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step stdout (for Checkpoint D)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory (default: results/JAX_DreamerSRL/<run-name>/)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info at startup")
    parser.add_argument("--buffer-device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Buffer storage device: 'cpu' (default, preserves all tests) or "
                             "'gpu' (Step 2 opt-in, stores replay buffer as jnp.ndarray). "
                             "See docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 2")
    parser.add_argument("--legacy-grad-loop", action="store_true", default=False,
                        help="Step 3 opt-in: use the pre-Step-3 Python for-loop over gradient "
                             "steps instead of the default jax.lax.scan path. "
                             "Preserves bit-identity for grad-parity tests (test_grad_parity.py). "
                             "Default: False (scan path). "
                             "See docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 3")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Load configs (env + agent)
    # Load default env config first, then merge training/eval/viz defaults on
    # top, then the experiment config.  Mirrors train.py:L295-L367.
    # -----------------------------------------------------------------------
    import os as _os
    _project_root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    from src.utils.config import get_default_config

    # --- Mutual-exclusion guard + curriculum schedule build ---
    # Exactly one of --env-config or --configs-dir is required.
    # Mirrors train.py:329-336, 441-442.
    schedule: Optional[ContinualSchedule] = None
    if args.configs_dir is not None:
        # Curriculum mode: --configs-dir replaces --env-config.
        if args.env_config is not None:
            raise ValueError(
                "--configs-dir and --env-config are mutually exclusive. "
                "Pass one or the other, not both."
            )
        if args.continual_schedule is None:
            raise ValueError(
                "--continual-schedule is required when --configs-dir is set."
            )
        if any(x is not None for x in (args.episodes, args.total_steps, args.total_timesteps)):
            raise ValueError(
                "--episodes / --total-steps / --total-timesteps are not allowed "
                "with --configs-dir. The episode budget comes from "
                "episode_boundaries[-1] in the schedule file."
            )
        schedule = _build_continual_schedule(
            _project_root, args.configs_dir, args.continual_schedule
        )
        # Stage-0 config is the live env config at startup.
        env_cfg = schedule.stage_configs[0]
    else:
        # Single-config mode: existing path unchanged.
        if args.env_config is None:
            raise ValueError(
                "Exactly one of --env-config or --configs-dir is required."
            )
        env_cfg = get_default_config()  # seed base (archived/standalone configs rely on this)
        # Merge Training defaults (training.checkpoint_frequency etc.)
        # Ported from train.py:L297-L327
        for _cfg_rel in [
            'configs/train/default.yaml',
            'configs/evaluation/default.yaml',
            'configs/visualization/default.yaml',
        ]:
            _cfg_path = _os.path.join(_project_root, _cfg_rel)
            if _os.path.exists(_cfg_path):
                env_cfg.merge(Config.load_yaml(_cfg_path))
        # load_env_config: if env YAML has `extends:`, its base is merged inside;
        # if standalone (archived), it loads as-is and merges on top of defaults.
        env_cfg.merge(load_env_config(args.env_config))

    agent_cfg = Config.load_yaml(args.agent_config)

    # Mandatory agent config reads (no fallback defaults per CLAUDE.md)
    learning_starts_cfg = agent_cfg.get_mandatory("algo.learning_starts", int)
    replay_ratio = agent_cfg.get_mandatory("algo.replay_ratio", float)
    seq_len = agent_cfg.get_mandatory("algo.per_rank_sequence_length", int)
    batch_size = agent_cfg.get_mandatory("algo.per_rank_batch_size", int)
    num_envs = args.num_envs  # CLI sets this; needed for total_timesteps calculation below
    # WP-SRL P6: the config value is an ENV-STEP count (sheeprl semantics) —
    # the prefill covers learning_starts_cfg env steps at EVERY env count.
    # Pre-fix the raw value gated ITERATIONS, making prefill num_envs x too
    # long. Ported from sheeprl@33b6366:dreamer_v3.py:508-511 (world_size==1).
    learning_starts, prefill_steps = derive_prefill(learning_starts_cfg, num_envs)

    # Episode-budget resolution — mirrors train.py:439-451 dual-mode pattern.
    # Priority: --episodes > --total-(time)steps > training.episodes (mandatory).
    # If --total-(time)steps is set, force episodes=0 to drop into env-step branch.
    env_max_steps = env_cfg.get_mandatory('environment.max_steps', int)
    env_step_override = args.total_timesteps or args.total_steps  # either CLI alias
    if schedule is not None:
        # Curriculum mode: budget comes from the schedule (mutual-exclusion
        # guard above ensures --episodes / --total-steps are not set).
        episodes = schedule.episode_boundaries[-1]
        total_timesteps = episodes * env_max_steps * num_envs
    elif args.episodes is not None:
        episodes = args.episodes
        total_timesteps = env_step_override or (episodes * env_max_steps * num_envs)
    elif env_step_override is not None:
        episodes = 0                       # env-step mode
        total_timesteps = env_step_override
    else:
        episodes = env_cfg.get_mandatory('training.episodes', int)
        total_timesteps = episodes * env_max_steps * num_envs
    # Legacy variable kept for the final-log print; equals the budgeted env-step cap.
    total_steps = total_timesteps
    buffer_size = agent_cfg.get_mandatory("buffer.size", int)

    gamma = agent_cfg.get_mandatory("algo.gamma", float)
    lmbda = agent_cfg.get_mandatory("algo.lmbda", float)
    horizon = agent_cfg.get_mandatory("algo.horizon", int)
    ent_coef = agent_cfg.get_mandatory("algo.actor.ent_coef", float)
    kl_dynamic = agent_cfg.get_mandatory("algo.kl_dynamic", float)
    kl_representation = agent_cfg.get_mandatory("algo.kl_representation", float)
    kl_free_nats = agent_cfg.get_mandatory("algo.kl_free_nats", float)
    kl_regularizer = agent_cfg.get_mandatory("algo.kl_regularizer", float)
    continue_scale_factor = agent_cfg.get_mandatory("algo.continue_scale_factor", float)

    moments_decay = agent_cfg.get_mandatory("algo.actor.moments.decay", float)
    moments_max = agent_cfg.get_mandatory("algo.actor.moments.max", float)
    moments_pct_low = agent_cfg.get_mandatory("algo.actor.moments.percentile.low", float)
    moments_pct_high = agent_cfg.get_mandatory("algo.actor.moments.percentile.high", float)

    critic_tau = agent_cfg.get_mandatory("algo.critic.tau", float)
    target_update_freq = agent_cfg.get_mandatory("algo.critic.per_rank_target_network_update_freq", int)

    wm_lr = agent_cfg.get_mandatory("algo.world_model.optimizer.lr", float)
    wm_eps = agent_cfg.get_mandatory("algo.world_model.optimizer.eps", float)
    actor_lr = agent_cfg.get_mandatory("algo.actor.optimizer.lr", float)
    actor_eps = agent_cfg.get_mandatory("algo.actor.optimizer.eps", float)
    critic_lr = agent_cfg.get_mandatory("algo.critic.optimizer.lr", float)
    critic_eps = agent_cfg.get_mandatory("algo.critic.optimizer.eps", float)

    # -----------------------------------------------------------------------
    # 2b. Eval-video config — read from env_cfg (which now holds training.*,
    #     visualization.*, testing.* merged in above).
    #     Ported from train.py:L2429-L2435
    # -----------------------------------------------------------------------
    video_during_training = env_cfg.get_mandatory('training.video_during_training')
    eval_video_episodes   = env_cfg.get_mandatory('training.eval_video_episodes', int)
    stats_during_training = env_cfg.get_mandatory('training.stats_during_training')
    eval_stats_episodes   = env_cfg.get_mandatory('training.eval_stats_episodes', int)
    eval_stats_num_envs   = env_cfg.get_mandatory('training.eval_stats_num_envs', int)
    checkpoint_frequency  = env_cfg.get_mandatory('training.checkpoint_frequency', int)
    max_checkpoints_keep  = env_cfg.get_mandatory('training.max_checkpoints_to_keep', int)
    viz_enabled           = env_cfg.get_mandatory('visualization.enabled')
    viz_fps               = env_cfg.get_mandatory('visualization.fps', int)
    auto_render           = env_cfg.get_mandatory('testing.auto_render_after_eval')

    if args.debug:
        print(f"[dreamer-srl] eval config: video_during_training={video_during_training}, "
              f"eval_video_episodes={eval_video_episodes}, stats_during_training={stats_during_training}, "
              f"checkpoint_frequency={checkpoint_frequency}, viz_fps={viz_fps}, "
              f"auto_render={auto_render}, max_checkpoints_keep={max_checkpoints_keep}")

    # -----------------------------------------------------------------------
    # 3. Seed + JAX device setup
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # -----------------------------------------------------------------------
    # 4. Construct env + probe obs_dim / action_dim
    # -----------------------------------------------------------------------
    env_params = load_env_params(env_cfg)
    env = ParallelEnv(env_params)

    # Probe obs_dim by resetting
    key, k_reset = jax.random.split(key)
    states, probe_obs = env.reset(k_reset, num_envs)
    obs_dim = int(probe_obs.shape[-1])
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)

    print(f"[dreamer-srl] obs_dim={obs_dim}, action_dim={action_dim}, num_envs={num_envs}")
    print(f"[dreamer-srl] episodes={episodes} (0=env-step mode), "
          f"total_timesteps={total_timesteps}, learning_starts={learning_starts} iters "
          f"(= {learning_starts_cfg} env steps / {num_envs} envs; WP-SRL P6)")
    print(f"[dreamer-srl] seq_len={seq_len}, batch_size={batch_size}, horizon={horizon}")

    # -----------------------------------------------------------------------
    # 4b. Pre-flight obs/action + modality-fingerprint check (curriculum only)
    # Ported from train.py:483-534. Validates that all stage configs produce
    # the same obs_dim, action_dim, and 13-field modality fingerprint so the
    # retained weights fit every stage. Fails fast with a descriptive error.
    # -----------------------------------------------------------------------
    def _modality_fingerprint(p):
        """13-field tuple of sensor enables + shape params affecting obs layout.

        Ported verbatim from train.py:L485-L503.
        If any two stages produce different fingerprints, obs semantics differ
        even when obs_dim happens to be the same (same-dim modality swap).
        """
        return (
            p.visual_sensor_enabled,
            p.visual_sensor_range,
            p.local_view_size,
            p.olfactory_enabled,
            p.olfactory_vector_size,
            p.nociception_enabled,
            p.nociception_size,
            p.interoceptive_nociception_enabled,
            p.location_sensor_enabled,
            p.proprioception_enabled,
            p.injury_observable,
            p.nutrition_observable,
            p.sensor_range,
        )

    if schedule is not None:
        stage0_fingerprint = _modality_fingerprint(env_params)
        probe_key = jax.random.PRNGKey(0)
        for _i in range(1, schedule.num_stages):
            _p_i = load_env_params(schedule.stage_configs[_i])
            _env_i = ParallelEnv(_p_i)
            _, _obs_i = _env_i.reset(probe_key, 1)
            _a_i = 4 + int(_p_i.rest_action_enabled) + int(_p_i.eat_action_enabled)
            if int(_obs_i.shape[-1]) != obs_dim or _a_i != action_dim:
                raise ValueError(
                    f"Stage {_i} ({schedule.stage_names[_i]}) changes "
                    f"obs_dim ({obs_dim} -> {int(_obs_i.shape[-1])}) or "
                    f"action_dim ({action_dim} -> {_a_i}). "
                    "Continual learning forbids architecture-visible dimension changes."
                )
            _fp_i = _modality_fingerprint(_p_i)
            if _fp_i != stage0_fingerprint:
                raise ValueError(
                    f"Stage {_i} ({schedule.stage_names[_i]}) has a different sensor "
                    f"modality fingerprint than stage 0, which would scramble the "
                    f"observation semantics even if obs_dim is unchanged.\n"
                    f"  Stage 0 fingerprint: {stage0_fingerprint}\n"
                    f"  Stage {_i} fingerprint: {_fp_i}"
                )
        if not args.quiet:
            print(
                f"[dreamer-srl] Continual mode: obs_dim={obs_dim}, "
                f"action_dim={action_dim} validated consistent across "
                f"{schedule.num_stages} stages."
            )

    # -----------------------------------------------------------------------
    # 5. Build agent
    # -----------------------------------------------------------------------
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=agent_cfg.to_dict(),
        rngs=rngs,
    )
    print("[dreamer-srl] build_agent OK")

    # -----------------------------------------------------------------------
    # 6. Construct optimizers (Flax 0.12.4: wrt=nnx.Param required)
    # WP-SRL P1: gradient clipping restored — sheeprl clips on every step
    # (dreamer_v3.py:193-197 WM norm 1000, :300-302 actor 100, :320-324
    # critic 100). The unclipped WM loss was observed spiking to ~1e29 in
    # live smokes; a single spike poisons Adam's second moment for thousands
    # of steps (tests/algorithms/dreamer_srl/test_grad_clip.py).
    # -----------------------------------------------------------------------
    wm_opt = nnx.Optimizer(world_model, make_optim_tx(wm_lr, wm_eps, WM_CLIP_NORM), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(actor, make_optim_tx(actor_lr, actor_eps, ACTOR_CLIP_NORM), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, make_optim_tx(critic_lr, critic_eps, CRITIC_CLIP_NORM), wrt=nnx.Param)

    # -----------------------------------------------------------------------
    # 7. Construct moments + ratio + buffer + player
    # -----------------------------------------------------------------------
    moments = moments_init()
    ratio = Ratio(ratio=replay_ratio, pretrain_steps=0)
    # WP-SRL P2: per-env independent buffers + sheeprl capacity semantics.
    # Sizing ported from sheeprl@33b6366:dreamer_v3.py:478
    # (buffer_size = cfg.buffer.size // (num_envs * world_size), world_size == 1).
    # The old construction passed the full cfg size per env column (num_envs x
    # the reference capacity) AND shared one write head across envs (hole rows
    # at partial dones — area report 04 D-03).
    per_env_buffer_size = buffer_size // num_envs
    # N1 (review_srl_parity_fixes.md): fail fast at startup — a per-env
    # capacity below seq_len passes ready_to_sample() once wrapped-full but
    # crashes at the first post-prefill sample() (buffers.py:367-371).
    validate_per_env_capacity(
        per_env_buffer_size,
        seq_len,
        configured_buffer_size=buffer_size,
        num_envs=num_envs,
    )
    if args.buffer_device == "gpu":
        if num_envs != 1:
            raise ValueError(
                "--buffer-device gpu supports num_envs == 1 only (WP-SRL P2: "
                "bincount sampling across per-env buffers would recompile per "
                "allocation pattern on the traced GPU path)."
            )
        # Single-env: the plain buffer is semantically identical to a
        # wrapper-of-one; keep it so the traced GPU add/sample path survives.
        buffer = SequentialReplayBuffer(
            buffer_size=per_env_buffer_size,
            n_envs=1,
            obs_keys=("obs",),
            device="gpu",  # Step 2 opt-in via --buffer-device
        )
    else:
        buffer = EnvIndependentSequentialReplayBuffer(
            buffer_size=per_env_buffer_size,
            n_envs=num_envs,
            obs_keys=("obs",),
        )
    player = Player(world_model, actor, num_envs)

    # -----------------------------------------------------------------------
    # 8. Build JIT'd train_step (factory captures horizon + scalars statically)
    # -----------------------------------------------------------------------
    train_step = make_train_step(
        horizon=horizon,
        gamma=gamma,
        lmbda=lmbda,
        ent_coef=ent_coef,
        kl_dynamic=kl_dynamic,
        kl_representation=kl_representation,
        kl_free_nats=kl_free_nats,
        kl_regularizer=kl_regularizer,
        continue_scale_factor=continue_scale_factor,
        moments_decay=moments_decay,
        moments_max=moments_max,
        moments_pct_low=moments_pct_low,
        moments_pct_high=moments_pct_high,
    )

    # -----------------------------------------------------------------------
    # 9. Init WandB
    # Ported from train.py:L599-L617 — flatten full env + agent config trees
    # into wandb.init config so runs are filterable by agent.algorithm, etc.
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            # Build flat config payload mirroring train.py:L599-L617
            env_config_dict = env_cfg.to_dict()
            agent_config_dict = agent_cfg.to_dict()
            wandb_config = {
                # Explicit top-level identifiers (mirrors train.py:L600-L601)
                "algorithm": "DreamerV3",
                "framework": "JAX/Flax NNX",
                # Backward-compat scalars (previously hand-picked subset)
                "env_config": args.env_config,
                "agent_config": args.agent_config,
                "configs_dir": args.configs_dir,
                "continual_schedule": args.continual_schedule,
                "current_stage": 0,                      # curriculum: start stage
                "total_steps":      total_timesteps,     # env-step cap (legacy key)
                "total_timesteps":  total_timesteps,     # rPPO-style alias
                "episodes":         episodes,            # primary budget
                "num_envs": num_envs,
                "seed": args.seed,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "horizon": horizon,
                "gamma": gamma,
                "lmbda": lmbda,
                "learning_starts": learning_starts,               # derived ITERATION count (WP-SRL P6)
                "learning_starts_env_steps": learning_starts_cfg,  # config env-step value
                "seq_len": seq_len,
                "batch_size": batch_size,
                # Spread env YAML first, then agent YAML — agent wins on collision.
                # Mirrors train.py:L599-L617 spread pattern so the agent YAML's `agent:`
                # block lands at wandb.config.agent (clean, single level) — makes
                # `agent.algorithm = DreamerV3` filterable in the WandB UI.
                # Merge rule: agent-config wins on key collision (algorithm-identity and
                # training-cadence live there). Current canonical configs are disjoint at
                # top level (env: environment/body/sensory/visualization/perceptual_noise/
                # behavior_measures; agent: agent/algo/buffer/training/env), so no actual
                # overrides today — the rule only matters for future configs.
                **env_config_dict,
                **agent_config_dict,
            }
            # Defense-in-depth: if a variant YAML ever omits agent.algorithm,
            # the WandB filter still works. (Cheap, idempotent.)
            wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"

            run = wandb.init(
                project=args.wandb_project,
                entity="sungwoolee",   # Mirrors configs/logger/wandb.yaml:L4
                name=args.wandb_name,
                config=wandb_config,
            )
            print(f"[dreamer-srl] WandB run: {run.url}")
            # Commit 2: define_metric so Episode/* keys plot against Episode/Number
            # Mirrors train.py:L625-L634
            wandb.define_metric("iteration")
            wandb.define_metric("timesteps")
            wandb.define_metric("Episode/Number")
            wandb.define_metric("*",            step_metric="timesteps")            # catch-all fallback
            wandb.define_metric("Episode/*",    step_metric="Episode/Number")
            # Explicit prefix routes so metrics don't silently rely on the catch-all.
            # Uppercase prefixes match the actual logged keys (WandB patterns are case-sensitive).
            # Ported from train.py:L630-L632 — corrected to uppercase.
            wandb.define_metric("Loss/*",       step_metric="timesteps")            # Ported from train.py:L630
            wandb.define_metric("WorldModel/*", step_metric="timesteps")            # Ported from train.py:L632 (new)
            wandb.define_metric("Behavior/*",   step_metric="timesteps")            # Ported from train.py:L632
            wandb.define_metric("Time/*",       step_metric="timesteps")
            wandb.define_metric("Params/*",     step_metric="timesteps")
            wandb.define_metric("Diagnostic/*", step_metric="timesteps")
            # Commit A (eval-video): Eval/* step metric + checkpoint_episode
            # Mirrors train.py:L2466-L2467 Eval/* pattern
            wandb.define_metric("eval/checkpoint_episode")
            wandb.define_metric("Eval/*",       step_metric="timesteps")
            # Curriculum: stage/* metrics indexed on Episode/Number.
            # Mirrors plan §(h) — logged at each stage boundary.
            wandb.define_metric("stage/*",      step_metric="Episode/Number")
        except ImportError:
            print("[dreamer-srl] WandB not installed — disabling WandB logging")
            use_wandb = False
    else:
        print("[dreamer-srl] WandB disabled (--no-wandb)")

    # -----------------------------------------------------------------------
    # 9b. Results directory — mirrors train.py:L538-L545 JAX_DreamerV3 convention.
    # Convention: results/JAX_DreamerSRL/<YYYYMMDD-HHMMSS>_<wandb-run-name>/.
    # Falls back to tmp/JAX_DreamerSRL_<timestamp>/ when --no-wandb.
    # -----------------------------------------------------------------------
    from datetime import datetime as _dt
    _timestamp = _dt.now().strftime("%Y%m%d-%H%M%S")
    if args.results_dir:
        results_dir = args.results_dir
    elif use_wandb and wandb.run is not None:
        results_dir = _os.path.join(_project_root, 'results', 'JAX_DreamerSRL', f"{_timestamp}_{wandb.run.name}")
    else:
        results_dir = _os.path.join(_project_root, 'tmp', f'JAX_DreamerSRL_{_timestamp}')
    _os.makedirs(results_dir, exist_ok=True)
    print(f"[dreamer-srl] results_dir={results_dir}")

    # Persist both source YAMLs next to checkpoints for reproducibility.
    # Mirrors train.py:L561-L566 — but kept SEPARATE (env_config.yaml + agent_config.yaml)
    # instead of merged, to preserve the --env-config / --agent-config CLI provenance.
    import yaml as _yaml
    _models_dir = _os.path.join(results_dir, 'models')
    _os.makedirs(_models_dir, exist_ok=True)
    _env_save = _os.path.join(_models_dir, 'env_config.yaml')
    _agent_save = _os.path.join(_models_dir, 'agent_config.yaml')
    with open(_env_save, 'w') as _f:
        _dump_config_yaml(env_cfg.to_dict(), _f)
    with open(_agent_save, 'w') as _f:
        _dump_config_yaml(agent_cfg.to_dict(), _f)
    print(f"[dreamer-srl] saved env_config → {_env_save}")
    print(f"[dreamer-srl] saved agent_config → {_agent_save}")

    # In curriculum mode: dump each stage config + the schedule for auditability.
    # Mirrors train.py:L568-L582.
    if schedule is not None:
        for _si, (_sname, _scfg) in enumerate(zip(schedule.stage_names, schedule.stage_configs)):
            _stage_out = _os.path.join(_models_dir, f"stage_{_si:02d}_{_sname}.yaml")
            with open(_stage_out, 'w') as _f:
                _dump_config_yaml(_scfg.to_dict(), _f)
        _sched_dump = {
            "continual": {
                "episode_boundaries":     schedule.episode_boundaries,
                "checkpoint_frequencies": schedule.checkpoint_frequencies,
                "stage_names":            schedule.stage_names,
            }
        }
        _sched_out = _os.path.join(_models_dir, "schedule.yaml")
        with open(_sched_out, 'w') as _f:
            _dump_config_yaml(_sched_dump, _f)
        print(f"[dreamer-srl] Saved {schedule.num_stages} stage configs + schedule.yaml → {_models_dir}/")

    # Checkpoint state for Commit B
    last_ckpt_episode: int = 0   # tracks last episode count at which we saved

    # Initialize Orbax CheckpointManager — Commit B
    # Ported from train.py:L555-L559
    from src.algorithms.dreamer_srl.checkpoint import make_checkpoint_manager, save_checkpoint as _save_checkpoint
    _ckpt_manager = make_checkpoint_manager(results_dir, max_to_keep=max_checkpoints_keep)
    print(f"[dreamer-srl] Checkpoint manager ready at {results_dir}/checkpoints/")

    # -----------------------------------------------------------------------
    # 10. Initial env reset + player state init
    #     Matches sheeprl main() L540-L547
    # -----------------------------------------------------------------------
    obs = np.asarray(probe_obs)   # [B, obs_dim] — already reset above
    player.init_states()

    # step_data — the accumulating buffer entry for the current step
    # Sheeprl L541-L546: initialize with obs + zeros for other fields
    step_data = {
        "obs":        obs[np.newaxis],                            # [1, B, obs_dim]
        "actions":    np.zeros((1, num_envs, action_dim), dtype=np.float32),
        "rewards":    np.zeros((1, num_envs, 1), dtype=np.float32),
        "terminated": np.zeros((1, num_envs, 1), dtype=np.float32),
        "truncated":  np.zeros((1, num_envs, 1), dtype=np.float32),
        "is_first":   np.ones((1, num_envs, 1), dtype=np.float32),  # first step = is_first
    }

    # -----------------------------------------------------------------------
    # 11. Episode tracking
    # -----------------------------------------------------------------------
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    is_first = np.ones((num_envs, 1), dtype=np.float32)  # for player inference

    # Commit 2: per-iteration episode buffer + total episode counter
    # Mirrors train.py:L1062 (Dreamer) and train.py:L1349 (rPPO)
    iteration_episodes: list = []          # cleared each log_every window
    total_episodes_completed: int = 0      # running total for Episode/Number step metric

    # Commit 3: per-env behavior-event + distance accumulators.
    # Mirrors train.py:L943-L950
    BEHAVIOR_KEYS = ['ate_food', 'hit_predator', 'hit_hiding_predator',
                     'hit_neutral', 'event_collided', 'rested',
                     'damage', 'damage_predator', 'damage_hiding_predator',
                     'damage_obstacle']
    BEHAVIOR_DIST_KEYS = ['dist_to_food', 'dist_to_pred',
                          'dist_to_neutral', 'dist_to_hiding_predator']
    episode_behavior  = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
    episode_dist_sums = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_DIST_KEYS}

    # Commit 4: per-instance (tag-based) accumulators.
    # Mirrors train.py:L953-L958
    neutral_tags  = tuple(env_params.neutral_tags)
    predator_tags = tuple(env_params.predator_tags)
    num_neutral_for_log  = len(neutral_tags)
    num_predator_for_log = len(predator_tags)
    episode_dist_per_neutral_sums  = np.zeros((num_envs, num_neutral_for_log),  dtype=np.float32)
    episode_dist_per_predator_sums = np.zeros((num_envs, num_predator_for_log), dtype=np.float32)

    # Commit 5: Behavior-Measure toolkit v1 state.
    # Mirrors train.py:L962-L982
    from src.behavior.accumulators import (
        make_bm_state, bm_step_update, bm_reset_env,
        bm_finalise_episode as _bm_finalise_episode_shared,
    )
    from src.environment.config_loader import load_behavior_measure_cfg
    bm_cfg = load_behavior_measure_cfg(env_cfg)
    bm_enabled = bm_cfg is not None and bm_cfg.enabled
    if bm_enabled:
        bm_R = float(bm_cfg.cue_radius)
        bm_K = int(bm_cfg.obs_window)
        _bm_state = make_bm_state(
            num_envs=num_envs,
            num_predator_tags=num_predator_for_log,
            num_neutral_tags=num_neutral_for_log,
            bm_R=bm_R,
            bm_K=bm_K,
        )
    else:
        _bm_state = None

    # -----------------------------------------------------------------------
    # 11b. Curriculum stage tracker + active checkpoint frequency.
    # current_stage: 0-based index into schedule.stage_configs.
    # checkpoint_frequency_active: per-stage checkpoint cadence (episodes).
    #   In single-config mode mirrors the existing checkpoint_frequency.
    # Mirrors plan §(i) and train.py:1093 (stage bookkeeping).
    # -----------------------------------------------------------------------
    current_stage: int = 0
    checkpoint_frequency_active: int = (
        schedule.checkpoint_frequencies[0] if schedule is not None
        else checkpoint_frequency
    )

    # -----------------------------------------------------------------------
    # 11c. Fix 1 (compile-once) — build a persistent jitted gradient-scan function.
    #
    # Root cause: the bare jax.lax.scan in the training loop (lines 1449–1591 old)
    # was not wrapped in an outer @jax.jit.  XLA compiled the full ~76k-HLO-line
    # gradient program from scratch every training iteration → multi-hour "hang".
    #
    # Fix: extract the scan into _scan_grad_steps(), decorated @jax.jit, with:
    #   - graphdef_* captured in the Python closure (hashable static metadata)
    #   - module/optimizer states passed as explicit pytree arguments
    #   - scan length fixed by the leading axis of scan_xs (Fix 2 makes this const)
    #
    # This function is built ONCE here (graphdefs are immutable after build_agent).
    # The training loop calls it by name; JAX caches the compiled executable across
    # all iterations as long as the input shapes and dtypes don't change.
    #
    # Pattern: src/models/dreamer_v3_trainer.py:683-833 (_scan_train_gpu)
    # Insight: docs/memory/memories/dreamer_diagnosis/
    #          20260519_1509_nnx_lax_scan_split_merge_pattern.md
    # Plan:    docs/develop/active/dreamer_srl_v2/
    #          TRAIN_STEP_COMPILE_DIAGNOSIS_AND_FIX_PLAN.md §Fix 1
    # -----------------------------------------------------------------------
    # Capture graphdefs once (immutable structural descriptions of each module).
    _graphdef_wm,      _ = nnx.split(world_model)
    _graphdef_ac,      _ = nnx.split(actor)
    _graphdef_cr,      _ = nnx.split(critic)
    _graphdef_tg,      _ = nnx.split(target_critic)
    _graphdef_wm_opt,  _ = nnx.split(wm_opt)
    _graphdef_ac_opt,  _ = nnx.split(actor_opt)
    _graphdef_cr_opt,  _ = nnx.split(critic_opt)

    # Python-int hyperparameters captured in closure (hashable, never change).
    _tuf = target_update_freq
    _tau = critic_tau

    @jax.jit
    def _scan_grad_steps(
        state_wm, state_ac, state_cr, state_tg,
        state_wm_opt, state_ac_opt, state_cr_opt,
        moments_in, key_in,
        step_idx_in,
        scan_xs,
    ):
        """Single jax.jit boundary for the whole per-iteration gradient scan.

        All mutable state enters as pure JAX pytrees (state_*) or arrays
        (moments_in, key_in, step_idx_in).  The graphdefs are captured in
        the outer Python closure — they are hashable static metadata and never
        enter the traced computation.  scan_xs is a dict[str, array] with
        leading axis = n_grad_steps (fixed per Fix 2 → one compiled instance).

        Returns updated state pytrees + final moments/key/step_idx + stacked losses.
        """
        init_carry = (
            state_wm, state_ac, state_cr, state_tg,
            state_wm_opt, state_ac_opt, state_cr_opt,
            moments_in, key_in,
            step_idx_in,
        )

        def _scan_body(carry, batch_i):
            (s_wm, s_ac, s_cr, s_tg,
             s_wm_opt, s_ac_opt, s_cr_opt,
             carry_moments, carry_key, step_idx) = carry

            # --- Polyak update on target_critic ---
            # jnp.where replaces the Python if-conditional so the body is XLA-pure.
            do_update = (step_idx % _tuf) == 0
            tau_val = jnp.where(step_idx == 0,
                                jnp.float32(1.0),
                                jnp.float32(_tau))

            _cr_for_polyak = nnx.merge(_graphdef_cr, s_cr)
            _tg_for_polyak = nnx.merge(_graphdef_tg, s_tg)
            online_params = nnx.state(_cr_for_polyak, nnx.Param)
            target_params = nnx.state(_tg_for_polyak, nnx.Param)

            new_target_params = jax.tree.map(
                lambda c, t: jnp.where(
                    do_update,
                    (1.0 - tau_val) * t + tau_val * c,
                    t,
                ),
                online_params, target_params,
            )
            nnx.update(_tg_for_polyak, new_target_params)
            s_tg_new = nnx.state(_tg_for_polyak)

            # --- Reconstruct all modules for train_step ---
            _wm   = nnx.merge(_graphdef_wm,     s_wm)
            _ac   = nnx.merge(_graphdef_ac,     s_ac)
            _cr   = nnx.merge(_graphdef_cr,     s_cr)
            _tg   = nnx.merge(_graphdef_tg,     s_tg_new)
            _wm_o = nnx.merge(_graphdef_wm_opt, s_wm_opt)
            _ac_o = nnx.merge(_graphdef_ac_opt, s_ac_opt)
            _cr_o = nnx.merge(_graphdef_cr_opt, s_cr_opt)

            # --- Gradient step ---
            carry_key, k_train = jax.random.split(carry_key)
            new_moments, losses = train_step(
                _wm, _ac, _cr, _tg,
                _wm_o, _ac_o, _cr_o,
                carry_moments, batch_i, k_train,
            )

            # --- Extract updated state pytrees for next carry ---
            new_carry = (
                nnx.state(_wm),
                nnx.state(_ac),
                nnx.state(_cr),
                s_tg_new,       # target_critic: Polyak-updated above
                nnx.state(_wm_o),
                nnx.state(_ac_o),
                nnx.state(_cr_o),
                new_moments,
                carry_key,
                step_idx + 1,
            )
            return new_carry, losses

        final_carry, losses_stack = jax.lax.scan(_scan_body, init_carry, scan_xs)
        return final_carry, losses_stack

    # -----------------------------------------------------------------------
    # 11d. Fix 2 (constant scan length) — quantized grad-step count.
    #
    # Root cause: Ratio.__call__ returns 15, 16, 17, … per iteration (float
    # accumulation).  Each distinct scan length → a distinct XLA compiled instance,
    # so Fix 1's executable would recompile on every new length encountered.
    #
    # WP-SRL P7 / D-015: the scan path runs a CONSTANT _G gradient steps per
    # training iteration (one XLA executable — Fix 2, recompile-storm). The
    # former fractional remainder carry was dead code and has been deleted;
    # fractional replay ratios are QUANTIZED on this path (declared deviation
    # D-015 in DEVIATION_LOG.md). Exact Ratio cadence: use --legacy-grad-loop.
    # The Ratio scheduler is still called (for checkpoint/log continuity) but its
    # output is replaced by the quantized value in the scan path.
    # -----------------------------------------------------------------------
    _G: int = max(1, int(replay_ratio * num_envs))  # steady-state grad steps per iter

    # -----------------------------------------------------------------------
    # 12. Training loop (sheeprl main() L550-L765)
    # -----------------------------------------------------------------------
    cumulative_grad_steps = 0
    policy_step = 0
    iter_num = 0
    last_losses: Dict = {}

    # log_every: iterations between WandB metric logs.
    # Resolution order mirrors rPPO's CLI > config pattern (train.py:447):
    #   --log-interval (CLI)
    #   > agent_cfg.training.log_interval (per-config; e.g. buf256k.yaml)
    #   > env_cfg.training.log_interval   (env / global default at configs/train/default.yaml:15)
    #   > 50 (built-in fallback)
    # CAUTION: dreamer-srl "iteration" = one env-step batch (num_envs env-steps),
    # whereas rPPO "iteration" = one rollout (num_steps * num_envs env-steps). To get
    # rPPO-equivalent env-step cadence at num_envs=16, use log_interval ≈ 50000
    # (≈ 800k env-steps/log, matching rPPO @ log_interval=50, num_steps=128, num_envs=128).
    log_every = (
        args.log_interval
        or agent_cfg.get('training.log_interval')
        or env_cfg.get('training.log_interval', 50)
    )
    last_log_step = 0

    t_start = time.time()
    t_env_total = 0.0
    t_train_total = 0.0

    if episodes > 0:
        print(f"[dreamer-srl] Starting training loop: until {episodes} episodes "
              f"completed × {num_envs} envs (env-step cap: {total_timesteps})")
    else:
        total_iters_estimate = total_timesteps // num_envs
        print(f"[dreamer-srl] Starting training loop: {total_iters_estimate} "
              f"iterations × {num_envs} envs (env-step mode)")

    # -------------------------------------------------------------------------
    # Startup config banner (rPPO style) — mirrors train.py:690-754.
    # -------------------------------------------------------------------------
    if not args.quiet:
        _banner_width = 60
        print("\n" + "=" * _banner_width)
        print(" JAX/FLAX DREAMER-SRL CONFIGURATION ".center(_banner_width, "="))
        print("=" * _banner_width)

        def _print_section(title, data):
            print(f"\n[{title}]")
            for k, v in data.items():
                print(f"  ● {k:.<25} {v}")

        _with_satiation = env_cfg.get_mandatory('body.with_satiation')
        _print_section("Environment", {
            "Grid Size":  f"{env_params.height}x{env_params.width}",
            "Max Steps":  env_max_steps,
            "Mode":       "Interoceptive (Homeostasis)" if _with_satiation else "Conventional (Goal-driven)",
        })
        _print_section("Training", {
            "Framework":       "JAX/Flax NNX (Dreamer-SRL)",
            "Total Timesteps": f"{total_timesteps:,}",
            "Episodes":        episodes if episodes > 0 else "(env-step mode)",
            "Parallel Envs":   num_envs,
            "Seq Length":      seq_len,
            "Batch Size":      batch_size,
            "Horizon":         horizon,
            "Seed":            args.seed,
            "Results":         results_dir,
            "WandB":           "Enabled" if use_wandb else "Disabled",
        })
        print("\n--- RL API Specifications ---")
        print(f"Action Dim: {action_dim}")
        print(f"Observation Dim: {obs_dim}")
        print("=" * _banner_width + "\n")

    pbar = tqdm(total=episodes if episodes > 0 else None, disable=args.quiet, desc="Training")

    # Dual-mode while-loop — mirrors train.py:1169.
    # episodes > 0  → episode-driven (rPPO + JAX Dreamer-V3 default).
    # episodes == 0 → env-step fallback (preserves --total-steps backward compat).
    while (total_episodes_completed < episodes) if episodes > 0 else (policy_step < total_timesteps):
        iter_num += 1
        policy_step += num_envs

        # -------------------------------------------------------------------
        # ENV INTERACTION — collect one transition
        # Sheeprl L556-L656
        # -------------------------------------------------------------------
        t_env_start = time.time()

        # Get actions from player, OR uniform-random if before learning_starts (§S3).
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571.
        # Gate is inclusive (`<=`): iteration `learning_starts` is the LAST prefill
        # iteration; iteration `learning_starts + 1` is the first policy iteration.
        # WP-SRL P6: `learning_starts` is the DERIVED iteration count
        # (cfg env-step value // num_envs) and the train gate below feeds the
        # Ratio scheduler `ratio_steps = policy_step - prefill_steps * num_envs`
        # — both line-for-line matches of sheeprl dreamer_v3.py:508-511, 660-661
        # at world_size == 1 (DEVIATION_LOG D-014 is historical as of this fix;
        # the one-shot debt-repayment burst it described no longer occurs).
        # The "no gradient before learning_starts" hard invariant is preserved
        # by the OUTER `if iter_num >= learning_starts` guard at the train gate.
        key, k_player = jax.random.split(key)
        if iter_num <= learning_starts:
            # §S3 uniform-random prefill — seeds the buffer with diverse data.
            action_idx = jax.random.randint(
                k_player, shape=(num_envs,), minval=0, maxval=action_dim
            )  # [B] int32 in [0, action_dim)
            actions_oh = np.asarray(
                jax.nn.one_hot(action_idx, num_classes=action_dim, dtype=jnp.float32)
            )  # [B, action_dim]
        else:
            actions_oh = player.get_actions(obs, is_first, k_player)  # [B, action_dim]

        # Store actions in step_data (sheeprl L586: step_data["actions"] = ...)
        step_data["actions"] = actions_oh[np.newaxis]  # [1, B, action_dim]
        # Add to buffer BEFORE stepping env (sheeprl L587: rb.add(step_data))
        buffer.add(step_data, validate_args=False)

        # Step environment (sheeprl L589: next_obs, rewards, terminated, truncated, infos = envs.step(...))
        # Our env uses argmax of one-hot for the discrete action
        action_indices = np.argmax(actions_oh, axis=-1)  # [B] int indices
        states, next_obs_jax, rewards_jax, dones_jax, infos = env.step(states, action_indices)
        next_obs = np.array(next_obs_jax)               # [B, obs_dim] — writable copy
        rewards = np.asarray(rewards_jax)              # [B]
        dones = np.asarray(dones_jax).astype(bool)     # [B]

        # CP7-P2 fix: separate terminated vs truncated using env's termination_reason.
        # reason==1 → max-steps (truncated); reason>=2 → death event (terminated).
        # §S5 true-continue splice must zero bootstrap only on hard terminations, not
        # truncations — conflating them makes the agent treat every episode end as death,
        # giving a misleading value signal in food-only (where all dones are truncations).
        # Sheeprl ref: env returns separate terminated/truncated booleans natively.
        # Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P2
        # Ported from sheeprl@33b6366:dreamer_v3.py:L600-L610 (env API separation)
        term_reason = np.asarray(infos['termination_reason'])    # [B] int32
        terminated_np = (term_reason >= 2).astype(np.float32)[:, np.newaxis]  # [B, 1]
        truncated_np  = (term_reason == 1).astype(np.float32)[:, np.newaxis]  # [B, 1]

        # Commit 3: per-step behavior-event + distance accumulation.
        # Mirrors train.py:L1327-L1332
        info_np_t = {k: np.asarray(infos[k]) for k in (BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason'])}
        for k in BEHAVIOR_KEYS:
            episode_behavior[k] += info_np_t[k]
        for k in BEHAVIOR_DIST_KEYS:
            episode_dist_sums[k] += info_np_t[k]

        # Commit 4: per-tag distance accumulation.
        # Mirrors train.py:L1340-L1341
        if num_neutral_for_log > 0 and 'dist_per_neutral' in infos:
            episode_dist_per_neutral_sums  += np.asarray(infos['dist_per_neutral'])
        if num_predator_for_log > 0 and 'dist_per_predator' in infos:
            episode_dist_per_predator_sums += np.asarray(infos['dist_per_predator'])

        # Commit 5: BM toolkit per-step update.
        # Mirrors train.py:L1339-L1342
        if bm_enabled:
            _bm_info_t = {
                'ate_food':      np.asarray(infos['ate_food']),
                'agent_in_bush': np.asarray(infos['agent_in_bush']),
            }
            if 'dist_per_predator' in infos:
                _bm_info_t['dist_per_predator'] = np.asarray(infos['dist_per_predator'])
            if 'dist_per_neutral' in infos:
                _bm_info_t['dist_per_neutral']  = np.asarray(infos['dist_per_neutral'])
            bm_step_update(_bm_state, _bm_info_t, dones.astype(bool))

        t_env_total += time.time() - t_env_start

        # Update step_data for next iteration (sheeprl L628-L638)
        is_first_next = np.zeros((num_envs, 1), dtype=np.float32)
        step_data["obs"] = next_obs[np.newaxis]                           # [1, B, obs_dim]
        step_data["rewards"] = rewards[:, np.newaxis][np.newaxis].astype(np.float32)  # [1, B, 1]
        step_data["terminated"] = terminated_np[np.newaxis]               # [1, B, 1]
        step_data["truncated"] = truncated_np[np.newaxis]                 # [1, B, 1]
        step_data["is_first"] = np.zeros((1, num_envs, 1), dtype=np.float32)

        # Handle done environments (sheeprl L639-L657)
        # C1 (review_srl_parity_fixes.md): set on curriculum stage swaps so the
        # end-of-iteration counter advance is skipped (swap wipes ALL counters;
        # the pre-swap step must not credit the new stage's fresh counters).
        _stage_swapped_this_iter = False
        dones_idxes = list(np.where(dones)[0])
        if dones_idxes:
            # Log episode info BEFORE resetting (sheeprl L610-L618)
            # Commits 2/3/4/5: buffer episodes for per-iteration aggregation with
            # behavior-event, per-tag distance, termination-reason and BM fields.
            # Mirrors train.py:L1349-L1377 (rPPO)
            for i in dones_idxes:
                ep_len = int(episode_lengths[i]) + 1
                ep_rew = float(episode_rewards[i]) + float(rewards[i])
                total_episodes_completed += 1
                ep_data = {'r': ep_rew, 'l': ep_len}

                # Commit 3: behavior-event + dist + termination_reason.
                # Mirrors train.py:L1354-L1359
                for k in BEHAVIOR_KEYS:
                    ep_data[k] = float(episode_behavior[k][i])
                for k in BEHAVIOR_DIST_KEYS:
                    ep_data[k] = float(episode_dist_sums[k][i] / max(ep_len, 1))
                ep_data['termination_reason'] = int(info_np_t['termination_reason'][i])

                # Commit 4: per-tag distance means.
                # Mirrors train.py:L1361-L1369
                ep_l_safe = max(ep_len, 1)
                if num_neutral_for_log > 0:
                    means = episode_dist_per_neutral_sums[i] / ep_l_safe
                    for j, tag in enumerate(neutral_tags):
                        ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                if num_predator_for_log > 0:
                    means = episode_dist_per_predator_sums[i] / ep_l_safe
                    for j, tag in enumerate(predator_tags):
                        ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])

                # Commit 5: BM per-episode finalisation.
                # Mirrors train.py:L1371-L1372
                if bm_enabled:
                    ep_data.update(_bm_finalise_episode_shared(
                        _bm_state, i, predator_tags, neutral_tags,
                    ))

                iteration_episodes.append(ep_data)
                if args.debug:
                    pbar.write(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")

            # CP7-P1 fix: write reset_data second buffer entry at done boundaries.
            # Sheeprl writes TWO rows per done: (1) the normal step_data row (already
            # done via buffer.add(step_data) above), and (2) this reset_data row with
            # the real terminal obs (before auto-reset) and is_first=0.  The RSSM
            # reads is_first=1 from the NEXT step_data row to gate hidden-state reset.
            # Ported from sheeprl@33b6366:dreamer_v3.py:L639-L657
            # Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P1
            #
            # Fix 3 (recompile-storm): build reset_data at fixed width [1, num_envs, ...]
            # (all envs, not just done envs), then write only done-env columns via the
            # buffer.add done_mask path.  Previously used [1, R, ...] with R=len(dones_idxes)
            # (variable), which is pure numpy but keeps the pattern consistent and future-safe
            # if any downstream path ever traces through JAX.
            # next_obs still holds the true terminal obs (env auto-reset not yet run)
            reset_data = {
                "obs":        next_obs[np.newaxis],                                       # [1, B, obs_dim]
                "actions":    np.zeros((1, num_envs, actions_oh.shape[-1]),
                                       dtype=np.float32),                                 # [1, B, action_dim]
                "rewards":    step_data["rewards"],                                       # [1, B, 1]
                "terminated": step_data["terminated"],                                    # [1, B, 1]
                "truncated":  step_data["truncated"],                                     # [1, B, 1]
                "is_first":   np.zeros((1, num_envs, 1), dtype=np.float32),              # [1, B, 1]
            }
            buffer.add(reset_data, done_mask=dones, validate_args=False)

            # Reset player state for done envs — use fixed-width boolean-mask path
            # (Fix 1 — recompile-storm fix): builds get_initial_states(num_envs) which is
            # constant-shape, instead of get_initial_states(n_done) which varies 1..16.
            _done_mask = dones.astype(np.float32)  # [num_envs] — 1.0 for done envs
            player.init_states(done_mask=_done_mask)

            # Reset the already-staged step_data for done envs so the NEXT row
            # written (buffer.add(step_data) at the top of the next iteration) does
            # NOT inherit this episode's terminal reward / death flag. The port
            # previously set only is_first here and dropped sheeprl's reward/
            # terminated/truncated zeroing (H5). Restored via the helper.
            # Ported from sheeprl@33b6366:dreamer_v3.py:L652-L656 ("Reset already
            # inserted step data"). Fix H5 — see fix_plan_h5_dreamer_srl_buffer_reset.md
            is_first_next[dones_idxes] = 1.0
            _reset_terminal_step_data(step_data, dones_idxes)

            # Reset episode tracking for done envs
            episode_lengths[list(dones_idxes)] = 0
            episode_rewards[list(dones_idxes)] = 0.0
            # Commits 3/4/5: reset per-env behavior + per-tag + BM accumulators.
            # Mirrors train.py:L1382-L1391
            for i in dones_idxes:
                for k in BEHAVIOR_KEYS:
                    episode_behavior[k][i] = 0.0
                for k in BEHAVIOR_DIST_KEYS:
                    episode_dist_sums[k][i] = 0.0
                if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                # Commit 5: BM per-env reset. Mirrors train.py:L1390-L1391
                if bm_enabled:
                    bm_reset_env(_bm_state, i)

            # Env auto-reset: get fresh obs for done envs.
            # Fix 2 (recompile-storm): split into num_envs keys (constant shape) and
            # index by env_idx directly — instead of splitting into len(dones_idxes)
            # keys (variable 1..16) which caused XLA to build a new _threefry_split
            # executable for each distinct count.
            key, k_autoreset = jax.random.split(key)
            reset_keys = jax.random.split(k_autoreset, num_envs)  # [num_envs, 2] — constant shape
            for env_idx in dones_idxes:
                k_env = reset_keys[env_idx]
                # Reset single env by overwriting its state
                from src.environment.core import jax_reset
                from src.environment.sensor import get_observation
                new_state = jax_reset(env_params, k_env)
                new_obs_single = get_observation(new_state, env_params)
                # Update states and obs for this env
                states = jax.tree_util.tree_map(
                    lambda s, ns: s.at[env_idx].set(ns), states, new_state
                )
                next_obs[env_idx] = np.asarray(new_obs_single)

            # -------------------------------------------------------------------
            # Curriculum stage-transition block (gated on schedule is not None).
            # Ported from train.py:1176-1270. MUST run AFTER per-env autoreset loop
            # and BEFORE "obs = next_obs" (the fresh-stage full-env reset is
            # authoritative; the per-env autoreset above would otherwise leave
            # individual done-env obs drawn from the old env_params).
            # -------------------------------------------------------------------
            if schedule is not None:
                _new_stage = schedule.stage_for_episode(total_episodes_completed)
                if _new_stage != current_stage:
                    if not args.quiet:
                        print(
                            f"[STAGE] {current_stage}:{schedule.stage_names[current_stage]}"
                            f" -> {_new_stage}:{schedule.stage_names[_new_stage]}"
                            f" at ep={total_episodes_completed}"
                        )
                    # 1. Rebuild env (env_params reassigned — autoreset path reads it).
                    env_params = load_env_params(schedule.stage_configs[_new_stage])
                    env = ParallelEnv(env_params)
                    key, _k_stage_reset = jax.random.split(key)
                    states, _next_obs_jax = env.reset(_k_stage_reset, num_envs)
                    next_obs = np.array(_next_obs_jax)

                    # Re-sync step_data staged row to the fresh stage.
                    # step_data["obs"] was a NumPy view aliasing the *old* next_obs;
                    # after rebinding next_obs above the view still points at the
                    # pre-swap array.  buffer.add(step_data) runs at the TOP of the
                    # next iteration (before line ~991 re-writes obs), so without this
                    # re-sync the first row of the cleared buffer would pair the new
                    # stage's is_first=1 anchor with the old stage's obs/reward/term.
                    # Fix: mirror the startup init at lines ~783-789 exactly.
                    step_data["obs"]        = next_obs[np.newaxis]                    # [1, B, obs_dim]
                    step_data["rewards"]    = np.zeros((1, num_envs, 1), dtype=np.float32)
                    step_data["terminated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
                    step_data["truncated"]  = np.zeros((1, num_envs, 1), dtype=np.float32)

                    # 2. Reset player recurrent + posterior state (weights retained).
                    player.init_states()
                    is_first_next[:] = 1.0
                    step_data["is_first"][:] = 1.0

                    # 3. Clear replay buffer — prevents cross-stage dynamics
                    #    contamination. Matches train.py:1224-1246. Cheap counter reset.
                    _pre_size = buffer.filled_size  # WP-SRL P2: class-agnostic (wrapper sums sub-buffers)
                    buffer.reset()

                    # 4. Wipe in-flight episode accumulators (all envs — partial
                    #    episodes dropped). Risk 3 mitigation: clear iteration_episodes
                    #    so pre-swap per-tag keys don't reach WandB fan-out.
                    #    C1: flag the swap so _advance_episode_counters below skips
                    #    this iteration — otherwise the pre-swap step's +1/reward
                    #    lands on non-done envs' freshly wiped counters.
                    _stage_swapped_this_iter = True
                    episode_lengths[:] = 0
                    episode_rewards[:] = 0.0
                    for _k in BEHAVIOR_KEYS:
                        episode_behavior[_k][:] = 0.0
                    for _k in BEHAVIOR_DIST_KEYS:
                        episode_dist_sums[_k][:] = 0.0
                    iteration_episodes = []

                    # 5. Rebuild per-tag accumulators + BM state for the new tag roster.
                    neutral_tags  = tuple(env_params.neutral_tags)
                    predator_tags = tuple(env_params.predator_tags)
                    num_neutral_for_log  = len(neutral_tags)
                    num_predator_for_log = len(predator_tags)
                    episode_dist_per_neutral_sums  = np.zeros(
                        (num_envs, num_neutral_for_log),  dtype=np.float32
                    )
                    episode_dist_per_predator_sums = np.zeros(
                        (num_envs, num_predator_for_log), dtype=np.float32
                    )
                    if bm_enabled:
                        _bm_state = make_bm_state(
                            num_envs=num_envs,
                            num_predator_tags=num_predator_for_log,
                            num_neutral_tags=num_neutral_for_log,
                            bm_R=bm_R,
                            bm_K=bm_K,
                        )

                    # 6. Switch checkpoint frequency + log transition to WandB.
                    current_stage = _new_stage
                    checkpoint_frequency_active = schedule.checkpoint_frequencies[current_stage]
                    if use_wandb:
                        import wandb as _wandb_stage
                        _wandb_stage.log({
                            "stage/index":         current_stage,
                            "stage/transition":    1,
                            "stage/buffer_cleared": _pre_size,
                            "Episode/Number":      total_episodes_completed,
                        })

            # -------------------------------------------------------------------
            # Commit B — Orbax checkpoint trigger (episode-based).
            # Ported from train.py:L2398-L2426 checkpoint-save block.
            # Fires when total_episodes_completed crosses a new multiple of
            # checkpoint_frequency_active (per-stage cadence in curriculum mode,
            # static checkpoint_frequency in single-config mode).
            # -------------------------------------------------------------------
            _just_saved_ckpt = False
            if (total_episodes_completed > 0 and
                    total_episodes_completed // checkpoint_frequency_active >
                    last_ckpt_episode // checkpoint_frequency_active):
                pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iter_num})...")
                _save_checkpoint(
                    _ckpt_manager,
                    episode=total_episodes_completed,
                    world_model=world_model,
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    moments=moments,
                    key=key,
                    iter_num=iter_num,
                    policy_step=policy_step,
                    total_episodes_completed=total_episodes_completed,
                    cumulative_grad_steps=cumulative_grad_steps,
                    stage=current_stage,
                )
                last_ckpt_episode = total_episodes_completed
                _just_saved_ckpt = True
                pbar.write("[CHECKPOINT] Saved.")

                # -----------------------------------------------------------
                # Commit F — Checkpoint-triggered eval (mirrors train.py:L2429-L2467)
                # -----------------------------------------------------------
                if video_during_training or stats_during_training:
                    from src.algorithms.dreamer_srl.eval import (
                        dreamer_srl_eval_rollout, _render_and_upload,
                    )
                    # Pass 1: Video
                    if video_during_training:
                        if not args.quiet:
                            pbar.write(f'[eval] checkpoint @ ep={total_episodes_completed}: video pass')
                        _eval_result = dreamer_srl_eval_rollout(
                            world_model=world_model,
                            actor=actor,
                            env_params=env_params,
                            config=env_cfg,
                            num_episodes=eval_video_episodes,
                            seed=args.seed,
                            results_dir=results_dir,
                            checkpoint_pct=total_episodes_completed,
                            render_video=True,
                            quiet=args.quiet,
                        )
                        if auto_render and _eval_result['recordings_dir']:
                            _render_and_upload(
                                recordings_dir=_eval_result['recordings_dir'],
                                results_dir=results_dir,
                                checkpoint_pct=total_episodes_completed,
                                fps=viz_fps,
                                wandb_enabled=use_wandb,
                                quiet=args.quiet,
                            )
                        # Log Eval/* to WandB (mirrors train.py:L2466-L2467)
                        # Note: no explicit step= kwarg — render subprocess (~13s) advances
                        # WandB's internal step counter during eval, so passing the old
                        # policy_step triggers "Tried to log to step N < current step M".
                        # define_metric("Eval/*", step_metric="timesteps") routes the
                        # X-axis via the "timesteps" key in the dict instead.
                        if use_wandb:
                            import wandb as _wandb
                            _wandb.log({
                                'Eval/MeanReward': _eval_result['mean_reward'],
                                'Eval/MeanLength': _eval_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            })

                    # Pass 2: Stats (no video; just scalar metrics)
                    if stats_during_training:
                        if not args.quiet:
                            pbar.write(f'[eval] checkpoint @ ep={total_episodes_completed}: stats pass')
                        _stats_result = dreamer_srl_eval_rollout(
                            world_model=world_model,
                            actor=actor,
                            env_params=env_params,
                            config=env_cfg,
                            num_episodes=eval_stats_episodes,
                            seed=args.seed,
                            results_dir=results_dir,
                            checkpoint_pct=total_episodes_completed,
                            render_video=False,
                            quiet=args.quiet,
                        )
                        if use_wandb:
                            import wandb as _wandb
                            _wandb.log({
                                'Eval/MeanReward': _stats_result['mean_reward'],
                                'Eval/MeanLength': _stats_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            })

        else:
            _just_saved_ckpt = False

        # Update obs + is_first for next iteration
        obs = next_obs
        is_first = is_first_next

        # Update episode tracking — WP-SRL P5: skip envs that finished THIS
        # iteration. Their terminal transition was already counted into the
        # logged episode (ep_len = counters+1, ep_rew = counters+rewards[i] at
        # the done block above); the unconditional increment previously leaked
        # +1 step and the terminal reward into the SUCCESSOR episode's counters.
        # sheeprl needs no counters (gym wrapper final_info, dreamer_v3.py:610-618).
        # C1: skipped entirely on curriculum stage-swap iterations — the swap
        # block wiped all counters and the pre-swap step belongs to dropped
        # partial episodes, not the new stage's first episode.
        _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones,
                                  stage_swapped=_stage_swapped_this_iter)

        # -------------------------------------------------------------------
        # TRAIN GATE (sheeprl L660-L698)
        # -------------------------------------------------------------------
        if iter_num >= learning_starts:
            # Compute how many gradient steps are owed.
            # WP-SRL P6: subtract prefill env-steps so the Ratio scheduler sees
            # only policy-phase steps. `prefill_steps` carries sheeprl's
            # intentional off-by-one (learning_starts - int(learning_starts > 0),
            # dreamer_v3.py:508-511) — the earlier `learning_starts * num_envs`
            # variant here (the stale "D-014 fix", S-01 in area report 04) is
            # replaced by the line-for-line sheeprl formula. At
            # learning_starts_cfg=0 (D-012 smoke configs) this is a no-op.
            # Ported from sheeprl@33b6366:dreamer_v3.py:L660-L661 (world_size==1)
            ratio_steps = policy_step - prefill_steps * num_envs
            n_grad_steps = ratio(ratio_steps)

            # D-015: constant _G on the scan path (quantized ratio, declared);
            # legacy path keeps the exact Ratio return.
            if not args.legacy_grad_loop and n_grad_steps > 0:
                n_grad_steps_scan = _G
            else:
                n_grad_steps_scan = n_grad_steps  # legacy path: exact Ratio value

            # WP-SRL P8: a wrapped-full buffer is sampleable; the old expression
            # (`buffer._pos >= seq_len`) skipped up to seq_len-1 owed grad steps
            # after every ring wrap because `_pos` cycles low (area report 04 D-07).
            if n_grad_steps > 0 and buffer.ready_to_sample(seq_len):
                t_train_start = time.time()

                # Sample once for all gradient steps (sheeprl L664-L671)
                # Step 2 — GPU-buffer opt-in: when buffer.device=="gpu", pass a JAX PRNG key
                # and skip the subsequent H2D transfer (data is already on device).
                # When buffer.device=="cpu" (default), key=None keeps existing behaviour.
                # Fix 2: scan path samples exactly n_grad_steps_scan (== _G) steps.
                _n_samples = n_grad_steps_scan if not args.legacy_grad_loop else n_grad_steps
                if buffer._on_gpu:
                    key, sample_key = jax.random.split(key)
                    local_data = buffer.sample(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        n_samples=_n_samples,
                        key=sample_key,
                    )
                else:
                    local_data = buffer.sample(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        n_samples=_n_samples,
                    )
                # local_data shape: [_n_samples, seq_len, batch_size, ...]

                # Step 1 — Option S: single H2D for the whole iteration.
                # BEFORE: jnp.asarray(v[i], dtype=jnp.float32) was called once
                # per gradient step (line 885 in pre-Step-1 code), launching one
                # host→device transfer per grad step (tens per training iteration).
                # AFTER: one jax.tree.map does the full [n_grad_steps, ...] array
                # in a single transfer; v[i] inside the loop is a zero-copy JAX
                # slice (dispatches to lax.dynamic_index_in_dim).
                # Bit-identity is preserved: same numbers, same dtype (float32),
                # just fewer CUDA launches.
                # Step 2: when buffer._on_gpu, data is already jnp.ndarray — the
                # jax.tree.map still ensures float32 dtype consistency but avoids
                # any H2D transfer (no-op for GPU arrays already on device).
                # Reference: docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 1 §Step 2
                local_data_gpu = jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=jnp.float32),
                    local_data,
                )
                # local_data_gpu[k] shape: [_n_samples, seq_len, batch_size, ...] on device

                if args.legacy_grad_loop:
                    # -------------------------------------------------------
                    # LEGACY PATH (pre-Step-3 Python for-loop).
                    # Activated via --legacy-grad-loop flag.
                    # Preserves bit-identity for test_grad_parity.py.
                    # Step 3 plan: docs/develop/active/dreamer_srl_v2/
                    #              buffer_perf_fix_plan_option_L.md §Step 3
                    # -------------------------------------------------------
                    for i in range(n_grad_steps):
                        # Polyak update BEFORE train step (sheeprl L675-L680)
                        if cumulative_grad_steps % target_update_freq == 0:
                            tau = 1.0 if cumulative_grad_steps == 0 else critic_tau
                            online_params = nnx.state(critic, nnx.Param)
                            target_params = nnx.state(target_critic, nnx.Param)
                            new_target_params = jax.tree.map(
                                lambda c, t: (1.0 - tau) * t + tau * c,
                                online_params, target_params,
                            )
                            nnx.update(target_critic, new_target_params)

                        # Extract batch for this gradient step (sheeprl L681)
                        batch = {k: v[i] for k, v in local_data_gpu.items()}

                        # one_train_step (sheeprl L682-L698 train(...) call)
                        key, k_train = jax.random.split(key)
                        moments, losses = train_step(
                            world_model, actor, critic, target_critic,
                            wm_opt, actor_opt, critic_opt,
                            moments, batch, k_train,
                        )
                        last_losses = losses
                        cumulative_grad_steps += 1
                else:
                    # -------------------------------------------------------
                    # SCAN PATH — Fix 1 (compile-once) + Fix 2 (constant length).
                    #
                    # Calls the pre-built _scan_grad_steps (defined at §11c above,
                    # outside this loop).  That function is @jax.jit-decorated with
                    # graphdefs captured in its closure.  JAX compiles it ONCE on
                    # the first call and reuses the cached executable for all
                    # subsequent iterations (because input shapes / dtypes are fixed:
                    # Fix 2 makes the leading axis of scan_xs always == _G).
                    #
                    # Step 3 plan: docs/develop/active/dreamer_srl_v2/
                    #   buffer_perf_fix_plan_option_L.md §Step 3
                    # Fix plan:    docs/develop/active/dreamer_srl_v2/
                    #   TRAIN_STEP_COMPILE_DIAGNOSIS_AND_FIX_PLAN.md §Fix 1 §Fix 2
                    # -------------------------------------------------------

                    # Split live module states into pure pytrees for the jit boundary.
                    _, state_wm     = nnx.split(world_model)
                    _, state_ac     = nnx.split(actor)
                    _, state_cr     = nnx.split(critic)
                    _, state_tg     = nnx.split(target_critic)
                    _, state_wm_opt = nnx.split(wm_opt)
                    _, state_ac_opt = nnx.split(actor_opt)
                    _, state_cr_opt = nnx.split(critic_opt)

                    final_carry, losses_stack = _scan_grad_steps(
                        state_wm, state_ac, state_cr, state_tg,
                        state_wm_opt, state_ac_opt, state_cr_opt,
                        moments, key,
                        jnp.array(cumulative_grad_steps, dtype=jnp.int32),
                        local_data_gpu,  # scan_xs: leading axis == _G (fixed shape → cached compile)
                    )

                    # Unpack final carry and write state back to live modules.
                    (s_wm_f, s_ac_f, s_cr_f, s_tg_f,
                     s_wm_opt_f, s_ac_opt_f, s_cr_opt_f,
                     moments, key, _step_idx_f) = final_carry

                    nnx.update(world_model,   s_wm_f)
                    nnx.update(actor,         s_ac_f)
                    nnx.update(critic,        s_cr_f)
                    nnx.update(target_critic, s_tg_f)
                    nnx.update(wm_opt,        s_wm_opt_f)
                    nnx.update(actor_opt,     s_ac_opt_f)
                    nnx.update(critic_opt,    s_cr_opt_f)

                    # Recover last-step losses from the stacked output.
                    last_losses = jax.tree.map(lambda x: x[-1], losses_stack)

                    # Advance the Python counter (Fix 2: use quantized count).
                    cumulative_grad_steps += n_grad_steps_scan

                t_train_total += time.time() - t_train_start

                # CP8 forward-looking guard #1: print moments_invscale at step 100
                if cumulative_grad_steps == 100:
                    inv = float(last_losses.get("moments_invscale", float("nan")))
                    print(f"[step 100] moments_invscale={inv:.6f} (should be >= 1.0)")
                    if inv < 1.0:
                        print(f"[WARN] moments_invscale < 1.0 at step 100 — §S7 floor may be violated!")

        # -------------------------------------------------------------------
        # LOG metrics (sheeprl L702-L730)
        # -------------------------------------------------------------------
        # Log either every log_every iters, OR on the final iteration (whichever
        # comes first). "Final iter" is detected by checking whether the loop
        # condition will fail on the NEXT iter — done implicitly by logging
        # whenever the budget is about to be exhausted.
        will_be_last = (
            (total_episodes_completed >= episodes) if episodes > 0
            else (policy_step >= total_timesteps)
        )
        if last_losses and (iter_num - last_log_step >= log_every or will_be_last):
            last_log_step = iter_num
            sps_env = policy_step / max(time.time() - t_start, 1e-9)

            # Commit 2: per-iteration episode aggregation block.
            # Mirrors train.py:L1397-L1404 (rPPO) and train.py:L1713-L1720 (Dreamer).
            # One WandB row per log_every iterations, averaged over all episodes in the window.
            if use_wandb and iteration_episodes:
                ep_rewards = [ep['r'] for ep in iteration_episodes]
                ep_lengths = [ep['l'] for ep in iteration_episodes]
                ep_log = {
                    "Episode/Reward":     float(np.mean(ep_rewards)),
                    "Episode/Reward_Min": float(np.min(ep_rewards)),
                    "Episode/Reward_Max": float(np.max(ep_rewards)),
                    "Episode/Steps":      float(np.mean(ep_lengths)),
                    "Episode/Number":     total_episodes_completed,
                }

                # Commit 3: behavior-event + distance + termination fan-out
                # Mirrors train.py:L1406-L1428
                if 'ate_food' in iteration_episodes[0]:
                    ep_log.update({
                        "Episode/FoodEaten":     float(np.mean([ep['ate_food']             for ep in iteration_episodes])),
                        "Episode/PredatorHits":  float(np.mean([ep['hit_predator']         for ep in iteration_episodes])),
                        "Episode/DangerHits":    float(np.mean([ep['hit_hiding_predator']  for ep in iteration_episodes])),
                        "Episode/RestCount":     float(np.mean([ep['rested']               for ep in iteration_episodes])),
                        "Episode/Collisions":    float(np.mean([ep['event_collided']       for ep in iteration_episodes])),
                        "Episode/TotalDamage":   float(np.mean([ep['damage']               for ep in iteration_episodes])),
                        "Episode/DamagePredator":float(np.mean([ep['damage_predator']      for ep in iteration_episodes])),
                        "Episode/DamageDanger":  float(np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes])),
                        "Episode/DamageObstacle":float(np.mean([ep['damage_obstacle']      for ep in iteration_episodes])),
                        "Episode/MeanDistFood":  float(np.mean([ep['dist_to_food']         for ep in iteration_episodes])),
                        "Episode/MeanDistPredator":        float(np.mean([ep['dist_to_pred']              for ep in iteration_episodes])),
                        "Episode/MeanDistRabbit":          float(np.mean([ep['dist_to_neutral']           for ep in iteration_episodes])),
                        "Episode/MeanDistHidingPredator":  float(np.mean([ep['dist_to_hiding_predator']   for ep in iteration_episodes])),
                        "Episode/RabbitHits":              float(np.mean([ep['hit_neutral']               for ep in iteration_episodes])),
                        "Episode/HidingPredatorHits":      float(np.mean([ep['hit_hiding_predator']       for ep in iteration_episodes])),
                    })
                    term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                    for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                        ep_log[f"Episode/Term_{name}"] = float(np.mean([1.0 if r == code else 0.0 for r in term_reasons]))

                    # Commit 4: per-tag fan-out for distance keys.
                    # Mirrors train.py:L1430-L1433
                    from src.utils.episode_logging import append_per_tag_means
                    append_per_tag_means(ep_log, iteration_episodes, neutral_tags,
                                         'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                    append_per_tag_means(ep_log, iteration_episodes, predator_tags,
                                         'mean_dist_predator', 'Episode/MeanDistPredator')

                    # Commit 5: BM toolkit fan-out (gated on bm_enabled).
                    # Mirrors train.py:L1434-L1436
                    if bm_enabled:
                        from src.utils.episode_logging import bm_log_wandb
                        bm_log_wandb(ep_log, iteration_episodes, predator_tags, neutral_tags)

                wandb.log(ep_log, step=policy_step)
                iteration_episodes = []  # clear for next window

            # Commit 6: Dreamer-style prefix-sorted loss dict.
            # Replaces the flat Loss/* mapping with WorldModel/* + Behavior/* routing.
            # Mirrors train.py:L1797-L1810 prefix-sort routing.
            log_dict = {
                "Params/effective_replay_ratio": cumulative_grad_steps / max(policy_step, 1),
                "Time/sps_env":                  sps_env,
                "Diagnostic/moments_invscale":   float(last_losses.get("moments_invscale", float("nan"))),
                # X-axis keys: required so define_metric("*", step_metric="timesteps") has a
                # value to plot against.  Mirrors train.py:L1482-L1483 + L1788.
                "timesteps":                     int(policy_step),   # Ported from train.py:L1788
                "iteration":                     int(iter_num),      # Ported from train.py:L1483
            }
            for mk, mv in last_losses.items():
                v = float(mv)
                if mk.startswith('loss_actor') or mk.startswith('mean_') or mk == 'entropy' \
                        or mk in ('value_mae',):
                    log_dict[f"Behavior/{mk}"] = v
                elif mk.startswith('loss_model') or mk.startswith('loss_recon') or \
                     mk.startswith('loss_kl') or mk.startswith('loss_rew') or \
                     mk.startswith('loss_cont') or mk.startswith('loss_dyn') or \
                     mk.startswith('loss_rep')  or mk.startswith('model_'):
                    log_dict[f"WorldModel/{mk}"] = v
                elif mk in ('world_model_loss', 'observation_loss', 'reward_loss',
                            'state_loss', 'continue_loss'):
                    # Legacy sheeprl-style keys → re-namespace under WorldModel
                    # Mirrors train.py:L1801 WorldModel/* routing
                    alias = {
                        'world_model_loss': 'loss_model',
                        'observation_loss': 'loss_recon',
                        'reward_loss':      'loss_rew',
                        'state_loss':       'loss_kl',
                        'continue_loss':    'loss_cont',
                    }[mk]
                    log_dict[f"WorldModel/{alias}"] = v
                elif mk in ('value_loss', 'policy_loss'):
                    alias = {'value_loss': 'loss_critic', 'policy_loss': 'loss_actor'}[mk]
                    log_dict[f"Behavior/{alias}"] = v
                elif mk == 'moments_invscale':
                    pass  # already added at top of dict
                else:
                    log_dict[mk] = v

            if use_wandb:
                wandb.log(log_dict, step=policy_step)

            wm_loss = log_dict.get("WorldModel/loss_model", float("nan"))
            inv_s = log_dict["Diagnostic/moments_invscale"]
            pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
            pbar.set_postfix({
                "iter":  iter_num,
                "step":  policy_step,
                "wm":    f"{wm_loss:.4f}",
                "invsc": f"{inv_s:.3f}",
                "sps":   f"{sps_env:.1f}",
            })
            pbar.refresh()
            if args.debug:
                # In episode-mode, show ep-progress; in env-step-mode, show iter-progress.
                if episodes > 0:
                    progress = f"ep {total_episodes_completed}/{episodes}"
                else:
                    progress = f"iter {iter_num}"
                pbar.write(
                    f"[{progress}] "
                    f"policy_step={policy_step} "
                    f"world_model_loss={wm_loss:.4f} "
                    f"moments_invscale={inv_s:.4f} "
                    f"sps={sps_env:.1f}"
                )

    pbar.close()

    # -----------------------------------------------------------------------
    # 13. Final log + finish
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n[dreamer-srl] Done. Total time: {elapsed:.1f}s "
          f"({policy_step/elapsed:.1f} env-steps/s, "
          f"{total_episodes_completed} episodes completed)")
    print(f"[dreamer-srl] grad_steps={cumulative_grad_steps}")
    if last_losses:
        print(f"[dreamer-srl] Final losses:")
        for k, v in last_losses.items():
            print(f"  {k}: {float(v):.6f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
