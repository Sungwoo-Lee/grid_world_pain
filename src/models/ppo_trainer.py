import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Tuple, Any

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    # V(true next state), computed BEFORE auto-reset overwrites it with a fresh episode start
    # (Finding B, Part 2 — docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md). Used as
    # the GAE bootstrap value so truncation (timeout) retains gamma*V(s') instead of losing it to
    # the rollout's auto-reset. Only populated when return_mode == "GAE" (see collect_trajectories);
    # zeros otherwise, since MC mode never reads it.
    next_value: jnp.ndarray = None
    step_info: Any = None

class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    hit_predator: jnp.ndarray
    hit_hiding_predator: jnp.ndarray
    event_collided: jnp.ndarray
    rested: jnp.ndarray
    damage: jnp.ndarray
    damage_predator: jnp.ndarray
    damage_hiding_predator: jnp.ndarray
    damage_obstacle: jnp.ndarray
    dist_to_food: jnp.ndarray
    dist_to_pred: jnp.ndarray
    termination_reason: jnp.ndarray

class PPOBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray
    targets: jnp.ndarray

def compute_gae(rewards, values, values_next, dones, terminateds, gamma, lmbda):
    """Computes Generalized Advantage Estimation.

    RL-correct truncation handling (Finding B, Part 2 —
    docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md):
      - the delta bootstrap (`gamma * next_value`) is gated on `terminated` (REAL death,
        termination_reason in {2,3,4}) — RETAINED (not zeroed) on timeout (truncation).
      - the GAE accumulation reset is still gated on `done` (death OR timeout) — an episode
        boundary still cuts the advantage chain on both, since a new episode starts either way.
    `values_next` MUST be V(s') of the TRUE next state (Transition.next_value, computed pre-reset
    in collect_trajectories), not a value computed on an auto-reset fresh-episode observation.

    Args:
        rewards:      (T,) rewards at each timestep
        values:       (T,) V(s_t) for t = 0..T-1 — the advantage baseline
        values_next:  (T,) V(s_{t+1}) for t = 0..T-1 — TRUE next-state value, pre-auto-reset
        dones:        (T,) episode-end flags (real death OR timeout) — gates accumulation reset only
        terminateds:  (T,) real-termination flags (real death only) — gates the value bootstrap
        gamma:        discount factor
        lmbda:        GAE lambda
    """
    def gae_scan(gae, x):
        reward, value, next_value, done, terminated = x
        delta = reward + gamma * next_value * (1 - terminated) - value
        gae = delta + gamma * lmbda * (1 - done) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        gae_scan,
        0.0,
        (rewards, values, values_next, dones, terminateds),
        reverse=True
    )
    return advantages

def compute_mc_returns(rewards, dones, gamma):
    """Computes Monte Carlo returns."""
    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        ret = jnp.where(done, 0.0, ret)
        ret = reward + gamma * ret
        return ret, ret

    _, returns = jax.lax.scan(
        mc_scan, 
        0.0, 
        (rewards, dones),
        reverse=True
    )
    return returns

def ppo_loss_fn(model, batch, clip_eps, ent_coef, vf_coef):
    """PPO loss function for a batch using an NNX model."""
    logits, new_values = model(batch.obs)
    new_values = new_values.squeeze()
    
    log_probs = jax.nn.log_softmax(logits)
    # Get log_prob of actions
    new_log_probs = jax.vmap(lambda lp, a: lp[a])(log_probs, batch.actions)
    
    # Entropy
    probs = jax.nn.softmax(logits)
    entropies = -jnp.sum(probs * log_probs, axis=-1)
    
    # 1. Policy Loss
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    surr1 = ratio * batch.advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # 2. Value Loss
    value_loss = 0.5 * jnp.mean(jnp.square(new_values - batch.targets))
    
    # 3. Entropy Loss
    entropy_loss = -jnp.mean(entropies)
    
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    return total_loss, (policy_loss, value_loss, entropy_loss)

def collect_trajectories(model, env_params, last_state, last_key, num_steps, return_mode="MC"):
    """Collects parallel trajectories using jax.lax.scan."""
    from src.environment.core import jax_step, jax_reset
    from src.environment.sensor import get_observation

    # `return_mode` is a static (non-traced) string from `config`, which is itself a static jit
    # argument (see train.py's `nnx.jit(train_iteration_ppo, static_argnums=(5,))`). This plain
    # Python comparison is therefore resolved once at trace time, not per-step at runtime.
    use_gae_bootstrap = return_mode.upper() == "GAE"

    def scan_fn(carry, _):
        state, key = carry

        obs = jax.vmap(get_observation, in_axes=(0, None))(state, env_params)
        key, act_key = jax.random.split(key)
        act_keys = jax.random.split(act_key, state.agent_pos.shape[0])

        from .ppo_network import get_action_and_value_ppo_nnx
        action, log_prob, value = jax.vmap(
            get_action_and_value_ppo_nnx, in_axes=(None, 0, 0)
        )(model, obs, act_keys)

        next_state, reward, done, info = jax.vmap(jax_step, in_axes=(0, 0, None))(state, action, env_params)

        # Bootstrap value: V(TRUE next state), computed BEFORE auto-reset (below) overwrites
        # `next_state` with a fresh episode start. Finding B, Part 2 —
        # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md. Gated on GAE mode:
        # `compute_gae` is the only consumer of `next_value` (MC mode uses `compute_mc_returns`,
        # which never reads it). No live `ppo` config uses GAE mode, so this static, trace-time
        # branch keeps the extra value-head forward pass entirely out of the MC-mode hot path.
        if use_gae_bootstrap:
            obs_next_true = jax.vmap(get_observation, in_axes=(0, None))(next_state, env_params)
            _, next_value = model(obs_next_true)
            next_value = next_value.squeeze(-1)
        else:
            next_value = jnp.zeros_like(value)

        reset_key, key = jax.random.split(key)
        reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(env_params, jax.random.split(reset_key, state.agent_pos.shape[0]))
        
        final_state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(done.reshape((done.shape[0],) + (1,) * (r.ndim - 1)), r, n).astype(r.dtype),
            reset_state, next_state
        )
        
        step_info = StepInfo(
            ate_food=info['ate_food'],
            hit_predator=info['hit_predator'],
            hit_hiding_predator=info['hit_hiding_predator'],
            event_collided=info['event_collided'],
            rested=info['rested'],
            damage=info['damage'],
            damage_predator=info['damage_predator'],
            damage_hiding_predator=info['damage_hiding_predator'],
            damage_obstacle=info['damage_obstacle'],
            dist_to_food=info['dist_to_food'],
            dist_to_pred=info['dist_to_pred'],
            termination_reason=info['termination_reason'],
        )
        trans = Transition(
            obs=obs, action=action, reward=reward, done=done,
            log_prob=log_prob, value=value, step_info=step_info, next_value=next_value
        )
        return (final_state, key), trans

    (final_state, final_key), trajectories = jax.lax.scan(
        scan_fn, (last_state, last_key), None, length=num_steps
    )
    return trajectories, final_state, final_key

def update_step_ppo(model, optimizer, batch, config):
    """Performs a single PPO update step."""
    def loss_fn(m):
        return ppo_loss_fn(m, batch, config.clip_eps, config.ent_coef, config.vf_coef)
    
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux

def train_iteration_ppo(model, optimizer, env_params, env_state, key, config):
    """Performs one full PPO iteration (collect + N epochs)."""
    return_mode = getattr(config, 'return_mode', 'GAE')
    # Plain PPO implements "MC" and "GAE" only. "MC_FIXED" (raw returns as the critic
    # target + normalised advantages), "GAE_NORM" (GAE estimator with MC's z-scored
    # target and un-rescaled residual advantages) and "MC_RAW" (raw returns as the target
    # AND the raw residual as the advantage — nothing normalised) exist in the
    # RecurrentPPO trainer only; reject them loudly instead of letting the `else` below
    # silently run GAE. No fallback default.
    if return_mode.upper() not in ("MC", "GAE"):
        raise ValueError(
            f"Unknown agent.return_mode {return_mode!r} for algorithm PPO. "
            "Plain PPO supports 'MC' and 'GAE'; 'MC_FIXED', 'GAE_NORM' and 'MC_RAW' are "
            "implemented in the RecurrentPPO trainer only "
            "(src/models/recurrent_ppo_trainer.py)."
        )

    # 1. Collect rollouts
    trajectories, next_env_state, key = collect_trajectories(
        model, env_params, env_state, key, config.num_steps, return_mode=return_mode
    )

    # 2. Compute Advantages and Targets
    if return_mode.upper() == "MC":
        returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, None), out_axes=1)(
            trajectories.reward, trajectories.done, config.gamma
        )
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)
        targets = returns
        advantages = returns - trajectories.value
    else:
        # GAE. Real-termination mask (Finding B, Part 2): termination_reason 2/3/4 = real death;
        # 1 = timeout (truncation); 0 = still active. Gates the value bootstrap only — the
        # accumulation reset inside compute_gae still uses `done` (death OR timeout).
        terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
        # trajectories.next_value already holds V(s') of the TRUE next state, computed
        # pre-auto-reset inside collect_trajectories — supersedes the old final-value forward
        # pass + concatenate-and-shift (values_with_next) approach, which used the auto-reset
        # (fresh episode start) value on every done step, including timeouts.
        advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, 1, 1, None, None), out_axes=1)(
            trajectories.reward, trajectories.value, trajectories.next_value,
            trajectories.done, terminateds, config.gamma, config.gae_lambda
        )
        targets = advantages + trajectories.value
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    # Reshape for updates: (num_steps * num_envs, ...)
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    batch = PPOBatch(
        obs=flatten(trajectories.obs),
        actions=flatten(trajectories.action),
        log_probs=flatten(trajectories.log_prob),
        values=flatten(trajectories.value),
        advantages=flatten(advantages),
        targets=flatten(targets)
    )
    
    # 3. Update epochs
    epoch_losses = []
    for _ in range(config.num_epochs):
        # We process the whole batch at once or could sub-batch
        loss, aux = update_step_ppo(model, optimizer, batch, config)
        epoch_losses.append((loss, aux))
    
    num_completed = jnp.sum(trajectories.done)
    return next_env_state, key, epoch_losses, num_completed, trajectories
