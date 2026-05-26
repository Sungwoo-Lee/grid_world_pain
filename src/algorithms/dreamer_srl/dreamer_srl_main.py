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
        --env-config configs/experiment/dreamer_curriculum/01_food_only.yaml \\
        --agent-config configs/dreamer_srl/01_food_only.yaml \\
        --total-steps 5000 --num-envs 1 --seed 0 \\
        --wandb-project grid_world_pain \\
        --wandb-name dreamer_srl_v3_schema_migration_smoke_s0

CP9 note: learning_starts=0 (D-012 pre-declared in DEVIATION_LOG.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Project imports
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer
from src.algorithms.dreamer_srl.loss import TwoHotEncoding
from src.algorithms.dreamer_srl.train import make_train_step, polyak_update
from src.algorithms.dreamer_srl.utils import Ratio, moments_init


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

    def init_states(self, reset_envs=None) -> None:
        """Reset player recurrent + posterior state.

        Sheeprl PlayerDV3.init_states — reset all envs if reset_envs is None.
        Called at startup and after done episodes.
        """
        if reset_envs is None:
            # Reset all envs
            h0, z0 = self.world_model.rssm.get_initial_states(self.num_envs)
            self._recurrent_state = h0   # [B, recurrent_state_size]
            self._posterior_state = z0   # [B, num_cat, num_cls]
            action_dim = self.actor.action_dim
            self._prev_action = jnp.zeros((self.num_envs, action_dim), dtype=jnp.float32)
        else:
            # Reset only the done envs
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

def main() -> None:
    """Training-loop driver — port of sheeprl dreamer_v3.py:L361-L765 main()."""

    # -----------------------------------------------------------------------
    # 1. CLI args
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="dreamer-srl v3 training driver")
    parser.add_argument("--env-config", type=str, required=True,
                        help="Path to env YAML config")
    parser.add_argument("--agent-config", type=str, required=True,
                        help="Path to dreamer-srl agent YAML config")
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
    env_cfg = get_default_config()  # loads configs/environment/default.yaml

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

    env_cfg.merge(Config.load_yaml(args.env_config))  # experiment-specific overrides
    agent_cfg = Config.load_yaml(args.agent_config)

    # Mandatory agent config reads (no fallback defaults per CLAUDE.md)
    learning_starts = agent_cfg.get_mandatory("algo.learning_starts", int)
    replay_ratio = agent_cfg.get_mandatory("algo.replay_ratio", float)
    seq_len = agent_cfg.get_mandatory("algo.per_rank_sequence_length", int)
    batch_size = agent_cfg.get_mandatory("algo.per_rank_batch_size", int)
    num_envs = args.num_envs  # CLI sets this; needed for total_timesteps calculation below

    # Episode-budget resolution — mirrors train.py:439-451 dual-mode pattern.
    # Priority: --episodes > --total-(time)steps > training.episodes (mandatory).
    # If --total-(time)steps is set, force episodes=0 to drop into env-step branch.
    env_max_steps = env_cfg.get_mandatory('environment.max_steps', int)
    env_step_override = args.total_timesteps or args.total_steps  # either CLI alias
    if args.episodes is not None:
        episodes = args.episodes
    elif env_step_override is not None:
        episodes = 0                       # env-step mode
    else:
        episodes = env_cfg.get_mandatory('training.episodes', int)
    total_timesteps = env_step_override or (episodes * env_max_steps * num_envs)
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
          f"total_timesteps={total_timesteps}, learning_starts={learning_starts}")
    print(f"[dreamer-srl] seq_len={seq_len}, batch_size={batch_size}, horizon={horizon}")

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
    # -----------------------------------------------------------------------
    wm_opt = nnx.Optimizer(world_model, optax.adam(wm_lr, eps=wm_eps), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(actor, optax.adam(actor_lr, eps=actor_eps), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, optax.adam(critic_lr, eps=critic_eps), wrt=nnx.Param)

    # -----------------------------------------------------------------------
    # 7. Construct moments + ratio + buffer + player
    # -----------------------------------------------------------------------
    moments = moments_init()
    ratio = Ratio(ratio=replay_ratio, pretrain_steps=0)
    buffer = SequentialReplayBuffer(
        buffer_size=buffer_size,
        n_envs=num_envs,
        obs_keys=("obs",),
        device=args.buffer_device,  # Step 2: "cpu" (default) or "gpu" (opt-in via --buffer-device)
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
                "learning_starts": learning_starts,
                "seq_len": seq_len,
                "batch_size": batch_size,
                # Full config trees — agent.algorithm becomes filterable in WandB UI
                "env": env_config_dict,
                "agent": agent_config_dict,
            }
            # Defense-in-depth: force agent.algorithm even if a variant YAML omits the field
            # Ported from train.py:L600 + 189f0df defense
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
        # The train-gate at L490 uses `>=` and does NOT subtract `prefill_steps`
        # from `policy_step` (unlike sheeprl L661 — see D-014). At iter_num ==
        # learning_starts the train-gate fires for the first time and the Ratio
        # scheduler returns `int(learning_starts * replay_ratio)` grad steps in a
        # one-shot debt-repayment burst, then steady-state `replay_ratio` per iter
        # from learning_starts+1 onwards. The "no gradient before learning_starts"
        # hard invariant is preserved by the OUTER `if iter_num >= learning_starts`
        # guard at L490 — NOT by `ratio(0) == 0`. Long-run replay ratio matches
        # sheeprl by construction (Ratio class's self-correcting design).
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
                if not args.quiet:
                    print(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")

            # CP7-P1 fix: write reset_data second buffer entry at done boundaries.
            # Sheeprl writes TWO rows per done: (1) the normal step_data row (already
            # done via buffer.add(step_data) above), and (2) this reset_data row with
            # the real terminal obs (before auto-reset) and is_first=0.  The RSSM
            # reads is_first=1 from the NEXT step_data row to gate hidden-state reset.
            # Ported from sheeprl@33b6366:dreamer_v3.py:L639-L657
            # Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P1
            reset_envs = len(dones_idxes)
            # next_obs still holds the true terminal obs (env auto-reset not yet run)
            reset_data = {
                "obs":        next_obs[dones_idxes][np.newaxis],                         # [1, R, obs_dim]
                "actions":    np.zeros((1, reset_envs, actions_oh.shape[-1]),
                                       dtype=np.float32),                                 # [1, R, action_dim]
                "rewards":    step_data["rewards"][:, dones_idxes],                      # [1, R, 1]
                "terminated": step_data["terminated"][:, dones_idxes],                   # [1, R, 1]
                "truncated":  step_data["truncated"][:, dones_idxes],                    # [1, R, 1]
                "is_first":   np.zeros((1, reset_envs, 1), dtype=np.float32),            # [1, R, 1]
            }
            buffer.add(reset_data, env_idxes=dones_idxes, validate_args=False)

            # Reset player state for done envs
            player.init_states(reset_envs=dones_idxes)

            # Set is_first=1 in step_data so the NEXT row written has is_first=1
            # (sheeprl L656: step_data["is_first"][:, dones_idxes] = ones_like(...))
            is_first_next[dones_idxes] = 1.0
            step_data["is_first"][:, dones_idxes] = 1.0

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

            # Env auto-reset: get fresh obs for done envs
            key, k_autoreset = jax.random.split(key)
            reset_keys = jax.random.split(k_autoreset, len(dones_idxes))
            for idx_local, env_idx in enumerate(dones_idxes):
                k_env = reset_keys[idx_local]
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
            # Commit B — Orbax checkpoint trigger (episode-based).
            # Ported from train.py:L2398-L2426 checkpoint-save block.
            # Fires when total_episodes_completed crosses a new multiple of
            # checkpoint_frequency (mirrors train.py:L2395 episode modulo gate).
            # -------------------------------------------------------------------
            _just_saved_ckpt = False
            if (total_episodes_completed > 0 and
                    total_episodes_completed // checkpoint_frequency >
                    last_ckpt_episode // checkpoint_frequency):
                print(f"[dreamer-srl] Saving checkpoint @ episode {total_episodes_completed}...")
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
                )
                last_ckpt_episode = total_episodes_completed
                _just_saved_ckpt = True
                print(f"[dreamer-srl] Checkpoint saved.")

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
                            print(f'[eval] checkpoint @ ep={total_episodes_completed}: video pass')
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
                            print(f'[eval] checkpoint @ ep={total_episodes_completed}: stats pass')
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

        # Update episode tracking
        episode_lengths += 1
        episode_rewards += rewards.astype(np.float32)

        # -------------------------------------------------------------------
        # TRAIN GATE (sheeprl L660-L698)
        # -------------------------------------------------------------------
        if iter_num >= learning_starts:
            # Compute how many gradient steps are owed.
            # D-014 fix: subtract prefill env-steps so the Ratio scheduler sees only
            # policy-phase steps, not the accumulated prefill debt.  At learning_starts=0
            # (D-012, food-only config) this is a no-op (prefill_steps * num_envs == 0).
            # Sheeprl: ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            # Ported from sheeprl@33b6366:dreamer_v3.py:L661
            # Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P3
            ratio_steps = policy_step - learning_starts * num_envs
            n_grad_steps = ratio(ratio_steps)

            if n_grad_steps > 0 and buffer._pos >= seq_len:
                t_train_start = time.time()

                # Sample once for all gradient steps (sheeprl L664-L671)
                # Step 2 — GPU-buffer opt-in: when buffer.device=="gpu", pass a JAX PRNG key
                # and skip the subsequent H2D transfer (data is already on device).
                # When buffer.device=="cpu" (default), key=None keeps existing behaviour.
                if buffer._on_gpu:
                    key, sample_key = jax.random.split(key)
                    local_data = buffer.sample(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        n_samples=n_grad_steps,
                        key=sample_key,
                    )
                else:
                    local_data = buffer.sample(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        n_samples=n_grad_steps,
                    )
                # local_data shape: [n_grad_steps, seq_len, batch_size, ...]

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
                # local_data_gpu[k] shape: [n_grad_steps, seq_len, batch_size, ...] on device

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
                    # SCAN PATH (Step 3 default) — jax.lax.scan over grad steps.
                    # Replaces the Python for-loop with a single fused XLA op.
                    # Uses nnx.split / nnx.merge pattern from insight:
                    #   docs/memory/memories/dreamer_diagnosis/
                    #   20260519_1509_nnx_lax_scan_split_merge_pattern.md
                    # Anti-pattern avoided: graphdef captured in closure (hashable
                    # static); state pytree in carry (pure JAX array tree).
                    # Reference: src/models/dreamer_v3_trainer.py:685-833 (_scan_train_gpu)
                    # -------------------------------------------------------

                    # Split all modules + optimizers into (graphdef, state) pairs.
                    # graphdef is static metadata (hashable, captured in closure).
                    # state is a pure pytree of arrays (goes in the scan carry).
                    graphdef_wm,      state_wm      = nnx.split(world_model)
                    graphdef_ac,      state_ac      = nnx.split(actor)
                    graphdef_cr,      state_cr      = nnx.split(critic)
                    graphdef_tg,      state_tg      = nnx.split(target_critic)
                    graphdef_wm_opt,  state_wm_opt  = nnx.split(wm_opt)
                    graphdef_ac_opt,  state_ac_opt  = nnx.split(actor_opt)
                    graphdef_cr_opt,  state_cr_opt  = nnx.split(critic_opt)

                    # Carry: all mutable state as pure pytrees.
                    # step_idx tracks the cumulative grad step count for Polyak
                    # scheduling; it's a JAX int so jnp.where can act on it traced.
                    init_carry = (
                        state_wm, state_ac, state_cr, state_tg,
                        state_wm_opt, state_ac_opt, state_cr_opt,
                        moments, key,
                        jnp.array(cumulative_grad_steps, dtype=jnp.int32),
                    )

                    # Scan inputs: the pre-stacked local_data_gpu dict, already
                    # on device. lax.scan slices axis 0 → each body call receives
                    # {k: v[i]} automatically.
                    scan_xs = local_data_gpu  # dict[str, [n_grad_steps, ...]]

                    def _scan_body(carry, batch_i):
                        """One gradient step inside jax.lax.scan.

                        carry = (s_wm, s_ac, s_cr, s_tg,
                                 s_wm_opt, s_ac_opt, s_cr_opt,
                                 moments, key, step_idx)
                        batch_i = {k: v[i, ...]} — one slice of local_data_gpu.

                        graphdef_* are captured via Python closure (hashable static
                        metadata) — they never enter the carry, which prevents
                        JIT retracing. This is rule 1+3 from the nnx.split/merge
                        insight: docs/memory/memories/dreamer_diagnosis/
                        20260519_1509_nnx_lax_scan_split_merge_pattern.md
                        target_update_freq, critic_tau captured as Python ints/floats.
                        """
                        (s_wm, s_ac, s_cr, s_tg,
                         s_wm_opt, s_ac_opt, s_cr_opt,
                         carry_moments, carry_key, step_idx) = carry

                        # --- Polyak update on target_critic (scan-safe form) ---
                        # Replaces the Python if-conditional (lines 921-929 legacy path).
                        # jnp.where is elementwise: both branches always evaluated
                        # (XLA requirement), but only one is selected. This is safe
                        # because both branches are pure arithmetic on JAX arrays.
                        do_update = (step_idx % target_update_freq) == 0
                        tau_val = jnp.where(step_idx == 0,
                                            jnp.float32(1.0),
                                            jnp.float32(critic_tau))

                        # Reconstruct critic + target_critic to extract Param substate.
                        # Mirrors CPU path: nnx.state(critic, nnx.Param).
                        # Note: _cr_for_polyak and _tg_for_polyak are temporary;
                        # _cr reconstructed again below for train_step (same graphdef).
                        _cr_for_polyak = nnx.merge(graphdef_cr, s_cr)
                        _tg_for_polyak = nnx.merge(graphdef_tg, s_tg)
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
                        # Write updated Param substate back; extract full state
                        nnx.update(_tg_for_polyak, new_target_params)
                        s_tg_new = nnx.state(_tg_for_polyak)

                        # --- Reconstruct all modules for train_step ---
                        _wm   = nnx.merge(graphdef_wm,     s_wm)
                        _ac   = nnx.merge(graphdef_ac,     s_ac)
                        _cr   = nnx.merge(graphdef_cr,     s_cr)
                        _tg   = nnx.merge(graphdef_tg,     s_tg_new)
                        _wm_o = nnx.merge(graphdef_wm_opt, s_wm_opt)
                        _ac_o = nnx.merge(graphdef_ac_opt, s_ac_opt)
                        _cr_o = nnx.merge(graphdef_cr_opt, s_cr_opt)

                        # --- Train step ---
                        carry_key, k_train = jax.random.split(carry_key)
                        new_moments, losses = train_step(
                            _wm, _ac, _cr, _tg,
                            _wm_o, _ac_o, _cr_o,
                            carry_moments, batch_i, k_train,
                        )

                        # --- Extract updated state pytrees for next iteration ---
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

                    # Run the scan over all n_grad_steps
                    final_carry, losses_stack = jax.lax.scan(
                        _scan_body, init_carry, scan_xs,
                    )

                    # Unpack final carry and write state back to live modules
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

                    # Recover last-step losses from the stacked output
                    last_losses = jax.tree.map(lambda x: x[-1], losses_stack)

                    # Advance the Python counter by n_grad_steps (CP3 continuity)
                    cumulative_grad_steps += n_grad_steps

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

            if not args.quiet:
                wm_loss = log_dict.get("WorldModel/loss_model", float("nan"))
                inv_s = log_dict["Diagnostic/moments_invscale"]
                # In episode-mode, show ep-progress; in env-step-mode, show iter-progress.
                if episodes > 0:
                    progress = f"ep {total_episodes_completed}/{episodes}"
                else:
                    progress = f"iter {iter_num}"
                print(
                    f"[{progress}] "
                    f"policy_step={policy_step} "
                    f"world_model_loss={wm_loss:.4f} "
                    f"moments_invscale={inv_s:.4f} "
                    f"sps={sps_env:.1f}"
                )

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
