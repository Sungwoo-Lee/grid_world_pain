import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Tuple, Any, Union

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    mod_info: Any  # Neuromodulator outputs (z_percept, z_memory, temperature)
    # h_state can be an array (GRU) or a tuple of arrays (LSTM)
    # We store it as a PyTree

class PPOBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray
    targets: jnp.ndarray
    dones: jnp.ndarray
    h_init: Any  # Can be tuple or array

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
    """Computes Monte Carlo returns (matching PyTorch baseline)."""
    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        # Reset return at episode boundary
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

def _h_vmap_axes(h_state):
    """Infer vmap in_axes for a hidden state PyTree (batch dim = 0 for all leaves)."""
    return jax.tree_util.tree_map(lambda _: 0, h_state)

def _h_reset_on_done(h_state, done):
    """Reset all leaves of a hidden state PyTree to zero where done=True.
    
    Uses efficient broadcasting that works for both scalar and vector 'done'.
    """
    return jax.tree_util.tree_map(
        lambda h: jnp.where(jnp.reshape(done, (done.shape + (1,) * (h.ndim - done.ndim))), 0.0, h),
        h_state
    )

def _h_get_first_timestep(h_states):
    """Extract the first timestep from stacked hidden states (scan output)."""
    return jax.tree_util.tree_map(lambda h: h[0], h_states)

def ppo_loss_fn(model, batch, clip_eps, ent_coef, vf_coef):
    """PPO loss function for a trajectory batch using an NNX model."""
    
    def scan_fn(h, x):
        obs, action, done = x
        logits, value, h_new, _ = model(obs, h)
        
        log_probs = jax.nn.log_softmax(logits)
        new_log_prob = log_probs[action]
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs)
        
        # RNN Fix: Reset hidden state for the NEXT step if this step is DONE
        # This prevents Episode 2 from seeing Episode 1's history
        h_reset = _h_reset_on_done(h_new, done)
        
        return h_reset, (new_log_prob, value.squeeze(), entropy)

    _, (new_log_probs, new_values, entropies) = jax.lax.scan(
        scan_fn, batch.h_init, (batch.obs, batch.actions, batch.dones)
    )
    
    # 1. Policy Loss
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    surr1 = ratio * batch.advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # 2. Value Loss (simple MSE for MC mode, clipped for GAE)
    value_loss = 0.5 * jnp.mean(jnp.square(new_values - batch.targets))
    
    # 3. Entropy Loss
    entropy_loss = -jnp.mean(entropies)
    
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    return total_loss, (policy_loss, value_loss, entropy_loss)

def collect_trajectories(model, env_params, last_state, last_h_state, last_key, num_steps, rnn_type="LSTM"):
    """Collects parallel trajectories using jax.lax.scan and NNX model."""
    from src.environment.core import jax_step
    from src.environment.sensor import get_observation

    # Infer vmap axes from the hidden state structure (handles any PyTree)
    h_axes = _h_vmap_axes(last_h_state)
    
    def scan_fn(carry, _):
        state, h_state, key = carry
        
        # 1. Sense and Action
        obs = jax.vmap(get_observation, in_axes=(0, None))(state, env_params)
        key, act_key = jax.random.split(key)
        act_keys = jax.random.split(act_key, state.agent_pos.shape[0])
        
        from .recurrent_ppo_network import get_action_and_value_nnx
        
        # Generic vmap over batch — h_axes handles any PyTree structure
        action, log_prob, value, h_new, mod_info = jax.vmap(
            get_action_and_value_nnx, in_axes=(None, 0, h_axes, 0)
        )(model, obs, h_state, act_keys)
        
        # 2. Step Env
        next_state, reward, done, _ = jax.vmap(jax_step, in_axes=(0, 0, None))(state, action, env_params)
        
        # 3. Handle Auto-Reset
        reset_key, _ = jax.random.split(key)
        from src.environment.core import jax_reset
        reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(env_params, jax.random.split(reset_key, state.agent_pos.shape[0]))
        
        def select_done(d, r, n):
            d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
            return jnp.where(d_expanded, r, n)
            
        final_state = jax.tree_util.tree_map(
            lambda r, n: select_done(done, r, n),
            reset_state, next_state
        )
        
        # Reset hidden state on done (generic PyTree reset)
        final_h = _h_reset_on_done(h_new, done)
        
        trans = Transition(
            obs=obs, action=action, reward=reward, done=done,
            log_prob=log_prob, value=value, mod_info=mod_info
        )
        
        return (final_state, final_h, key), (trans, h_state)

    (final_state, final_h, final_key), (trajectories, h_states) = jax.lax.scan(
        scan_fn, (last_state, last_h_state, last_key), None, length=num_steps
    )
    
    return trajectories, h_states, final_state, final_h, final_key

def update_step(model, optimizer, batch, config):
    """Performs a single PPO update step using NNX patterns."""
    
    def batch_loss_wrapped(model):
        def compute_loss(m, obs_b):
            return ppo_loss_fn(m, obs_b, config.clip_eps, config.ent_coef, config.vf_coef)
        
        b_axes = PPOBatch(obs=1, actions=1, log_probs=1, values=1, advantages=1, targets=1, dones=1, h_init=0)
        losses, aux = jax.vmap(compute_loss, in_axes=(None, b_axes))(model, batch)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, aux)
    
    (loss, aux), grads = nnx.value_and_grad(batch_loss_wrapped, has_aux=True)(model)
    
    # 4. Calculate Gradient Norms
    grad_norm = optax.global_norm(grads)
    
    # Try to extract modulator-specific gradient norm if visible in the State
    mod_grad_norm = 0.0
    try:
        # NNX stats are nested; we look for the modulator key
        if hasattr(model, 'modulation_enabled') and model.modulation_enabled:
            # Note: The structure of 'grads' matches the structure of 'model'
            # We can use jax.tree_util to find sub-trees, but a simple check often works for Param states
            if 'modulator' in grads:
                mod_grad_norm = optax.global_norm(grads['modulator'])
    except:
        pass
    
    optimizer.update(model, grads)
    
    # Combine aux info with grad norms
    ppo_loss, v_loss, ent_loss = aux
    return loss, (ppo_loss, v_loss, ent_loss, grad_norm, mod_grad_norm)

def train_iteration(model, optimizer, env_params, env_state, h_state, key, config):
    """Performs one full PPO iteration (collect + N epochs) with NNX."""
    from src.environment.sensor import get_observation
    
    rnn_type = config.rnn_type
    return_mode = config.return_mode
    
    # 1. Collect rollouts
    trajectories, h_states, next_env_state, next_h_state, key = collect_trajectories(
        model, env_params, env_state, h_state, key, config.num_steps, rnn_type=rnn_type
    )
    
    # 2. Compute Advantages and Targets
    if return_mode.upper() == "MC":
        # Monte Carlo Returns (PyTorch parity)
        # Use vmap over batch dimension (axis 1)
        returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, None), out_axes=1)(
            trajectories.reward, trajectories.done, config.gamma
        )
        # Normalize returns
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)
        targets = returns
        advantages = returns - trajectories.value
    else:
        # GAE (original JAX)
        obs_final = jax.vmap(get_observation, in_axes=(0, None))(next_env_state, env_params)
        _, final_v, _ = jax.vmap(model)(obs_final, next_h_state)
        
        values_with_next = jnp.concatenate([trajectories.value, final_v.reshape(1, -1)], axis=0)
        
        advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, None, None), out_axes=1)(
            trajectories.reward, values_with_next[1:], trajectories.done, config.gamma, config.gae_lambda
        )
        targets = advantages + trajectories.value
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    # Get initial h_state for loss computation (generic PyTree extraction)
    h_init = _h_get_first_timestep(h_states)
    
    batch = PPOBatch(
        obs=trajectories.obs,
        actions=trajectories.action,
        log_probs=trajectories.log_prob,
        values=trajectories.value,
        advantages=advantages,
        targets=targets,
        dones=trajectories.done,
        h_init=h_init
    )
    
    # 3. Update epochs
    epoch_losses = []
    for _ in range(config.num_epochs):
        loss, aux = update_step(model, optimizer, batch, config)
        epoch_losses.append((loss, aux))
    
    # 4. Count completed episodes
    num_completed = jnp.sum(trajectories.done)
    
    return next_env_state, next_h_state, key, epoch_losses, num_completed, trajectories
