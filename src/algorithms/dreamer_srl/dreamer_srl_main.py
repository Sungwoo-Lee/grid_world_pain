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
        --wandb-project grid_world_pain_dreamer_srl_smoke \\
        --wandb-name dreamer_srl_cp9_dryrun_s0

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
    parser.add_argument("--total-steps", type=int, default=5000,
                        help="Total environment steps to run")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--wandb-project", type=str,
                        default="grid_world_pain_dreamer_srl_smoke",
                        help="WandB project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (for Checkpoint D micro-dry-run)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step stdout (for Checkpoint D)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Load configs (env + agent)
    # Load default env config first, then merge the experiment config on top.
    # This mirrors train.py:L295-L359 (get_default_config() + merge).
    # -----------------------------------------------------------------------
    from src.utils.config import get_default_config
    env_cfg = get_default_config()  # loads configs/environment/default.yaml
    env_cfg.merge(Config.load_yaml(args.env_config))  # experiment-specific overrides
    agent_cfg = Config.load_yaml(args.agent_config)

    # Mandatory agent config reads (no fallback defaults per CLAUDE.md)
    learning_starts = agent_cfg.get_mandatory("algo.learning_starts", int)
    replay_ratio = agent_cfg.get_mandatory("algo.replay_ratio", float)
    seq_len = agent_cfg.get_mandatory("algo.per_rank_sequence_length", int)
    batch_size = agent_cfg.get_mandatory("algo.per_rank_batch_size", int)
    total_steps = args.total_steps  # CLI overrides config
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

    num_envs = args.num_envs  # CLI sets this; config is just documentation

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
    print(f"[dreamer-srl] total_steps={total_steps}, learning_starts={learning_starts}")
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
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "env_config": args.env_config,
                    "agent_config": args.agent_config,
                    "total_steps": total_steps,
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
                },
            )
            print(f"[dreamer-srl] WandB run: {run.url}")
        except ImportError:
            print("[dreamer-srl] WandB not installed — disabling WandB logging")
            use_wandb = False
    else:
        print("[dreamer-srl] WandB disabled (--no-wandb)")

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

    # -----------------------------------------------------------------------
    # 12. Training loop (sheeprl main() L550-L765)
    # -----------------------------------------------------------------------
    cumulative_grad_steps = 0
    policy_step = 0
    last_losses: Dict = {}
    total_iters = total_steps // num_envs  # one iter = num_envs env steps

    last_log_step = 0
    log_every = max(1, total_iters // 100)  # log ~100 times over the run

    t_start = time.time()
    t_env_total = 0.0
    t_train_total = 0.0

    print(f"[dreamer-srl] Starting training loop: {total_iters} iterations × {num_envs} envs")

    for iter_num in range(1, total_iters + 1):
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

        # Extract terminated flag from env state (§S10: terminated ≠ truncated)
        # Our env's done = terminated_by_injury_or_starvation OR truncated_by_maxsteps
        # For §S10, we want ONLY the hard-termination events (not max-steps truncation)
        # The env stores this in states.terminated — we treat done as terminated for CP9
        # (fine because food-only has no death events, so done = truncated = maxsteps)
        terminated_np = dones.astype(np.float32)[:, np.newaxis]  # [B, 1]
        truncated_np = dones.astype(np.float32)[:, np.newaxis]   # [B, 1] (same for now)

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
            for i in dones_idxes:
                ep_len = int(episode_lengths[i]) + 1
                ep_rew = float(episode_rewards[i]) + float(rewards[i])
                if use_wandb:
                    wandb.log({"Game/ep_len_avg": ep_len, "Rewards/rew_avg": ep_rew},
                              step=policy_step)
                if not args.quiet:
                    print(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")

            # Reset player state for done envs
            player.init_states(reset_envs=dones_idxes)

            # Set is_first for the next obs of done envs
            is_first_next[dones_idxes] = 1.0
            step_data["is_first"][:, dones_idxes] = 1.0

            # Reset episode tracking for done envs
            episode_lengths[list(dones_idxes)] = 0
            episode_rewards[list(dones_idxes)] = 0.0

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
        else:
            pass

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
            # Compute how many gradient steps are owed
            ratio_steps = policy_step
            n_grad_steps = ratio(ratio_steps)

            if n_grad_steps > 0 and buffer._pos >= seq_len:
                t_train_start = time.time()

                # Sample once for all gradient steps (sheeprl L664-L671)
                local_data = buffer.sample(
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    n_samples=n_grad_steps,
                )
                # local_data shape: [n_grad_steps, seq_len, batch_size, ...]

                for i in range(n_grad_steps):
                    # Polyak update BEFORE train step (sheeprl L675-L680)
                    # Use jax.tree.map directly on NNX State pytrees (polyak_update
                    # expects a plain dict, so we use tree.map inline here).
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
                    # local_data[key] shape: [n_grad_steps, seq_len, batch_size, ...]
                    # Slice i to get [seq_len, batch_size, ...]
                    batch = {}
                    for k, v in local_data.items():
                        arr = jnp.asarray(v[i], dtype=jnp.float32)
                        # v[i] shape: [seq_len, batch_size, ...] → matches [T, B, ...]
                        batch[k] = arr

                    # one_train_step (sheeprl L682-L698 train(...) call)
                    key, k_train = jax.random.split(key)
                    moments, losses = train_step(
                        world_model, actor, critic, target_critic,
                        wm_opt, actor_opt, critic_opt,
                        moments, batch, k_train,
                    )
                    last_losses = losses
                    cumulative_grad_steps += 1

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
        if last_losses and (iter_num - last_log_step >= log_every or iter_num == total_iters):
            last_log_step = iter_num
            sps_env = policy_step / max(time.time() - t_start, 1e-9)
            log_dict = {
                "Loss/world_model_loss": float(last_losses.get("world_model_loss", float("nan"))),
                "Loss/observation_loss": float(last_losses.get("observation_loss", float("nan"))),
                "Loss/reward_loss":      float(last_losses.get("reward_loss", float("nan"))),
                "Loss/state_loss":       float(last_losses.get("state_loss", float("nan"))),
                "Loss/continue_loss":    float(last_losses.get("continue_loss", float("nan"))),
                "Loss/value_loss":       float(last_losses.get("value_loss", float("nan"))),
                "Loss/policy_loss":      float(last_losses.get("policy_loss", float("nan"))),
                "Params/replay_ratio":   cumulative_grad_steps / max(policy_step, 1),
                "Time/sps_env":          sps_env,
                "Diagnostic/moments_invscale": float(last_losses.get("moments_invscale", float("nan"))),
            }
            if use_wandb:
                wandb.log(log_dict, step=policy_step)

            if not args.quiet:
                wm_loss = log_dict["Loss/world_model_loss"]
                inv_s = log_dict["Diagnostic/moments_invscale"]
                print(
                    f"[iter {iter_num}/{total_iters}] "
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
          f"({total_steps/elapsed:.1f} env-steps/s)")
    print(f"[dreamer-srl] grad_steps={cumulative_grad_steps}")
    if last_losses:
        print(f"[dreamer-srl] Final losses:")
        for k, v in last_losses.items():
            print(f"  {k}: {float(v):.6f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
