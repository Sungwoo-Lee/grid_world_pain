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
    step_info: Any = None

class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    hit_predator: jnp.ndarray
    hit_danger: jnp.ndarray
    event_collided: jnp.ndarray
    rested: jnp.ndarray
    damage: jnp.ndarray
    damage_predator: jnp.ndarray
    damage_danger: jnp.ndarray
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

def compute_gae(rewards, values_next, dones, gamma, lmbda):
    """Computes Generalized Advantage Estimation."""
    def gae_scan(carry, x):
        gae, next_v = carry
        reward, next_v_val, done = x
        delta = reward + gamma * next_v_val * (1 - done) - next_v
        gae = delta + gamma * lmbda * (1 - done) * gae
        return (gae, next_v_val), gae

    _, advantages = jax.lax.scan(
        gae_scan, 
        (0.0, values_next[-1]), 
        (rewards, values_next, dones),
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

def collect_trajectories(model, env_params, last_state, last_key, num_steps):
    """Collects parallel trajectories using jax.lax.scan."""
    from src.environment.core import jax_step, jax_reset
    from src.environment.sensor import get_observation

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
        
        reset_key, key = jax.random.split(key)
        reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(env_params, jax.random.split(reset_key, state.agent_pos.shape[0]))
        
        final_state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(done.reshape((done.shape[0],) + (1,) * (r.ndim - 1)), r, n).astype(r.dtype),
            reset_state, next_state
        )
        
        step_info = StepInfo(
            ate_food=info['ate_food'],
            hit_predator=info['hit_predator'],
            hit_danger=info['hit_danger'],
            event_collided=info['event_collided'],
            rested=info['rested'],
            damage=info['damage'],
            damage_predator=info['damage_predator'],
            damage_danger=info['damage_danger'],
            damage_obstacle=info['damage_obstacle'],
            dist_to_food=info['dist_to_food'],
            dist_to_pred=info['dist_to_pred'],
            termination_reason=info['termination_reason'],
        )
        trans = Transition(
            obs=obs, action=action, reward=reward, done=done, 
            log_prob=log_prob, value=value, step_info=step_info
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
    from src.environment.sensor import get_observation
    
    # 1. Collect rollouts
    trajectories, next_env_state, key = collect_trajectories(
        model, env_params, env_state, key, config.num_steps
    )
    
    # 2. Compute Advantages and Targets
    if getattr(config, 'return_mode', 'GAE').upper() == "MC":
        returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, None), out_axes=1)(
            trajectories.reward, trajectories.done, config.gamma
        )
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)
        targets = returns
        advantages = returns - trajectories.value
    else:
        obs_final = jax.vmap(get_observation, in_axes=(0, None))(next_env_state, env_params)
        _, final_v = model(obs_final)
        final_v = final_v.squeeze()
        
        values_with_next = jnp.concatenate([trajectories.value, final_v.reshape(1, -1)], axis=0)
        advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, None, None), out_axes=1)(
            trajectories.reward, values_with_next[1:], trajectories.done, config.gamma, config.gae_lambda
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
