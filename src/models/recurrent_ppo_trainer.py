import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Tuple, Any, Union

class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    hit_predator: jnp.ndarray
    hit_hiding_predator: jnp.ndarray
    hit_neutral: jnp.ndarray
    event_collided: jnp.ndarray
    rested: jnp.ndarray
    damage: jnp.ndarray
    damage_predator: jnp.ndarray
    damage_hiding_predator: jnp.ndarray
    damage_obstacle: jnp.ndarray
    dist_to_food: jnp.ndarray
    dist_to_pred: jnp.ndarray
    dist_to_neutral: jnp.ndarray
    dist_to_hiding_predator: jnp.ndarray
    termination_reason: jnp.ndarray
    dist_per_neutral: jnp.ndarray   # [num_neutral]  per-rabbit distance, unreduced
    dist_per_predator: jnp.ndarray  # [num_predator] per-predator distance, unreduced
    agent_in_bush: jnp.ndarray      # bool — True iff agent is on a hides_agent obstacle (behavior toolkit v1)

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    mod_info: Any  # Neuromodulator outputs (z_percept, z_memory, temperature)
    step_info: Any = None  # StepInfo for behavioral metrics (optional for backward compat)
    # V(true next state), computed BEFORE auto-reset overwrites it with a fresh episode start
    # (Finding B, Part 2 — docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md). Used as
    # the GAE bootstrap value so truncation (timeout) retains gamma*V(s') instead of losing it to
    # the rollout's auto-reset.
    next_value: jnp.ndarray = None
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

def compute_gae(rewards, values, values_next, dones, terminateds, gamma, lmbda):
    """Computes Generalized Advantage Estimation.

    RL-correct truncation handling (Finding B, Part 2 —
    docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md):
      - the delta bootstrap is gated on `done AND terminated` (REAL death: the episode ended
        AND the reason was death, termination_reason in {2,3,4}) — it is RETAINED (not zeroed)
        on timeout (truncation). The AND with `done` guards the known env quirk where
        `overeating_death=True` stamps termination_reason=3 without setting done (KNOWN_BUGS) —
        a continuing episode must keep its bootstrap (mirror of `compute_mc_returns`' edge gate).
      - the GAE accumulation reset is still gated on `done` (death OR timeout) — an episode
        boundary still cuts the advantage chain on both, since a new episode starts either way.
    `values_next` MUST be V(s') of the TRUE next state (Transition.next_value, computed pre-reset
    in the rollout), not a value computed on an auto-reset fresh-episode observation.

    Args:
        rewards:      (T,) rewards at each timestep
        values:       (T,) V(s_t) for t = 0..T-1
        values_next:  (T,) V(s_{t+1}) for t = 0..T-1 — TRUE next-state value, pre-auto-reset
        dones:        (T,) episode-end flags (real death OR timeout) — gates accumulation reset only
        terminateds:  (T,) real-termination flags — ANDed with `dones` to gate the value bootstrap
        gamma:        discount factor
        lmbda:        GAE lambda
    """
    def gae_scan(gae, x):
        reward, value, next_value, done, terminated = x
        # Real death = done AND terminated (overeating-quirk guard — see docstring).
        real_death = done * terminated
        delta = reward + gamma * next_value * (1 - real_death) - value
        gae = delta + gamma * lmbda * (1 - done) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        gae_scan,
        0.0,
        (rewards, values, values_next, dones, terminateds),
        reverse=True
    )
    return advantages

def compute_mc_returns(rewards, dones, terminateds, bootstrap_value, gamma):
    """Computes Monte Carlo returns with a value bootstrap at the rollout-window edge.

    H4 fix (docs/develop/active/issues/diag_fable5_20260704/fix_plan_h4_mc_window_bootstrap.md):
    the reverse-scan carry is initialised with V(s') of the window's LAST step instead of
    0.0, so steps near the window edge keep an estimate of their future return. Edge-step
    gating follows the 3c60f6f truncation semantics:
      - real death at the edge  (done AND terminated)  -> bootstrap zeroed (future truly gone)
      - timeout at the edge     (done, NOT terminated) -> bootstrap retained
      - window cut mid-episode  (NOT done)             -> bootstrap retained
    Mid-window episode boundaries are unchanged: the return still resets on merged `done`
    (finite-horizon MC treatment of mid-window timeouts is deliberate — 02_rppo_stack.md,
    Finding 1 sibling note).

    Args:
        rewards:         (T,) rewards
        dones:           (T,) merged episode-end flags (death OR timeout)
        terminateds:     (T,) real-death flags (termination_reason >= 2)
        bootstrap_value: ()  V(s') of the TRUE (pre-auto-reset) next state after step T-1
        gamma:           discount factor
    """
    # Edge gate: zero the carry only on REAL death at the window edge. The AND with done
    # guards against the known env quirk where overeating sets termination_reason=3
    # without done=True (KNOWN_BUGS) — a continuing episode must keep its bootstrap.
    edge_death = jnp.logical_and(dones[-1].astype(bool), terminateds[-1].astype(bool))
    dones_for_reset = dones.at[-1].set(edge_death.astype(dones.dtype))

    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        # Reset return at episode boundary
        ret = jnp.where(done, 0.0, ret)
        ret = reward + gamma * ret
        return ret, ret

    _, returns = jax.lax.scan(
        mc_scan,
        bootstrap_value,
        (rewards, dones_for_reset),
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
    
    # 2. Value Loss (plain MSE, used for both MC and GAE modes)
    value_loss = 0.5 * jnp.mean(jnp.square(new_values - batch.targets))
    
    # 3. Entropy Loss
    entropy_loss = -jnp.mean(entropies)
    
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    return total_loss, (policy_loss, value_loss, entropy_loss)

def collect_trajectories(model, env_params, last_state, last_h_state, last_key, num_steps, rnn_type="LSTM", return_mode="MC"):
    """Collects parallel trajectories using jax.lax.scan and NNX model."""
    from src.environment.core import jax_step
    from src.environment.sensor import get_observation

    # Infer vmap axes from the hidden state structure (handles any PyTree)
    h_axes = _h_vmap_axes(last_h_state)
    # `return_mode` is a static (non-traced) string from `config`, which is itself a static
    # jit argument (see train.py's `nnx.jit(train_iteration, static_argnums=(6,))`). This plain
    # Python comparison is therefore resolved once at trace time, not per-step at runtime.
    use_gae_bootstrap = return_mode.upper() == "GAE"

    def scan_fn(carry, _):
        state, h_state, key, _prev_next_state, _prev_h_new = carry

        # 1. Sense
        with jax.named_scope("rppo_sense"):
            obs = jax.vmap(get_observation, in_axes=(0, None))(state, env_params)

        # 2. Action
        with jax.named_scope("rppo_act"):
            key, act_key = jax.random.split(key)
            act_keys = jax.random.split(act_key, state.agent_pos.shape[0])

            from .recurrent_ppo_network import get_action_and_value_nnx

            # Generic vmap over batch — h_axes handles any PyTree structure
            action, log_prob, value, h_new, mod_info = jax.vmap(
                get_action_and_value_nnx, in_axes=(None, 0, h_axes, 0)
            )(model, obs, h_state, act_keys)

        # 3. Step Env
        with jax.named_scope("rppo_env_step"):
            next_state, reward, done, info = jax.vmap(jax_step, in_axes=(0, 0, None))(state, action, env_params)

        # 3b. Bootstrap value: V(TRUE next state), computed BEFORE auto-reset (below) overwrites
        # `next_state` with a fresh episode start. Uses h_new (this step's post-forward hidden
        # state) so it reflects the RNN state that would carry into the continuation.
        # Finding B, Part 2 — docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md.
        # Gated on GAE mode: `compute_gae` is the only consumer of the PER-STEP `next_value`
        # (MC mode uses `compute_mc_returns`, which never reads it). Note: since the H4 fix,
        # MC mode DOES take a SINGLE window-edge bootstrap value, computed post-scan from the
        # pre-reset (next_state, h_new) carried out of the scan — one forward per window, not
        # per step. Every currently-live rPPO config sets
        # `return_mode: "MC"`, so this Python-level (static, trace-time) branch keeps the extra
        # value-head forward pass — and its speed cost — entirely out of the MC-mode hot path.
        if use_gae_bootstrap:
            with jax.named_scope("rppo_bootstrap_value"):
                obs_next_true = jax.vmap(get_observation, in_axes=(0, None))(next_state, env_params)
                _, next_value, _, _ = jax.vmap(model, in_axes=(0, h_axes))(obs_next_true, h_new)
                next_value = next_value.squeeze(-1)
        else:
            next_value = jnp.zeros_like(value)

        # 4. Handle Auto-Reset
        with jax.named_scope("rppo_env_reset"):
            # Advance the carried key (never reuse a sub-key): the old
            # `reset_key, _ = split(key)` re-split the already-carried key without
            # advancing it, so step t's reset key was byte-identical to step t+1's
            # master key (split(k, 2) == split(k, N)[:2]) — see
            # docs/reviews/review_nmn_trainer_parity_20260722.md, findings row 1.
            key, reset_key = jax.random.split(key)
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
        
        step_info = StepInfo(
            ate_food=info['ate_food'],
            hit_predator=info['hit_predator'],
            hit_hiding_predator=info['hit_hiding_predator'],
            hit_neutral=info['hit_neutral'],
            event_collided=info['event_collided'],
            rested=info['rested'],
            damage=info['damage'],
            damage_predator=info['damage_predator'],
            damage_hiding_predator=info['damage_hiding_predator'],
            damage_obstacle=info['damage_obstacle'],
            dist_to_food=info['dist_to_food'],
            dist_to_pred=info['dist_to_pred'],
            dist_to_neutral=info['dist_to_neutral'],
            dist_to_hiding_predator=info['dist_to_hiding_predator'],
            termination_reason=info['termination_reason'],
            dist_per_neutral=info['dist_per_neutral'],
            dist_per_predator=info['dist_per_predator'],
            agent_in_bush=info['agent_in_bush'],
        )
        trans = Transition(
            obs=obs, action=action, reward=reward, done=done,
            log_prob=log_prob, value=value, mod_info=mod_info,
            step_info=step_info, next_value=next_value
        )
        
        return (final_state, final_h, key, next_state, h_new), (trans, h_state)

    (final_state, final_h, final_key, boot_state, boot_h), (trajectories, h_states) = jax.lax.scan(
        scan_fn, (last_state, last_h_state, last_key, last_state, last_h_state), None,
        length=num_steps
    )

    # H4 fix: window-edge bootstrap value = V(TRUE next state) of the LAST window step,
    # from the pre-auto-reset (boot_state, boot_h) carried out of the scan. One value
    # forward per 128-step window — negligible vs the per-step collection forward.
    # In GAE mode the exact per-step quantity already exists; expose its edge slice so
    # the return signature is uniform (train_iteration's GAE branch does not consume it).
    if use_gae_bootstrap:
        bootstrap_value = trajectories.next_value[-1]
    else:
        with jax.named_scope("rppo_mc_edge_bootstrap"):
            obs_boot = jax.vmap(get_observation, in_axes=(0, None))(boot_state, env_params)
            _, v_boot, _, _ = jax.vmap(model, in_axes=(0, h_axes))(obs_boot, boot_h)
            bootstrap_value = v_boot.squeeze(-1)

    return trajectories, h_states, final_state, final_h, final_key, bootstrap_value

def update_step(model, optimizer, batch, config):
    """Performs a single PPO update step using NNX patterns."""
    
    def batch_loss_wrapped(model):
        def compute_loss(m, obs_b):
            return ppo_loss_fn(m, obs_b, config.clip_eps, config.ent_coef, config.vf_coef)
        
        b_axes = PPOBatch(obs=1, actions=1, log_probs=1, values=1, advantages=1, targets=1, dones=1, h_init=0)
        losses, aux = jax.vmap(compute_loss, in_axes=(None, b_axes))(model, batch)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, aux)
    
    with jax.named_scope("rppo_loss_grad"):
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

    with jax.named_scope("rppo_optim"):
        optimizer.update(model, grads)
    
    # Combine aux info with grad norms
    ppo_loss, v_loss, ent_loss = aux
    return loss, (ppo_loss, v_loss, ent_loss, grad_norm, mod_grad_norm)

def train_iteration(model, optimizer, env_params, env_state, h_state, key, config):
    """Performs one full PPO iteration (collect + N epochs) with NNX."""
    rnn_type = config.rnn_type
    return_mode = config.return_mode
    
    # 1. Collect rollouts
    with jax.named_scope("rppo_collect_trajectories"):
        trajectories, h_states, next_env_state, next_h_state, key, bootstrap_value = collect_trajectories(
            model, env_params, env_state, h_state, key, config.num_steps, rnn_type=rnn_type, return_mode=return_mode
        )

    # 2. Compute Advantages and Targets
    with jax.named_scope("rppo_advantages"):
        if return_mode.upper() == "MC":
            # Monte Carlo Returns (PyTorch parity) with window-edge bootstrap (H4 fix).
            # Real-death mask: same termination_reason >= 2 mapping as the GAE branch below.
            # Use vmap over batch dimension (axis 1); bootstrap_value is (num_envs,) -> axis 0.
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
                trajectories.reward, trajectories.done, terminateds, bootstrap_value, config.gamma
            )
            # Normalize returns
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)
            targets = returns
            advantages = returns - trajectories.value
        else:
            # GAE (original JAX)
            # Real-termination mask (Finding B, Part 2): termination_reason 2/3/4 = real death;
            # 1 = timeout (truncation); 0 = still active. Gates the value bootstrap only — the
            # accumulation reset inside compute_gae still uses `done` (death OR timeout).
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            # trajectories.next_value already holds V(s') of the TRUE next state, computed
            # pre-auto-reset inside collect_trajectories — supersedes the old final-value forward
            # + concatenate-and-shift (values_with_next) approach, which used the auto-reset
            # (fresh episode start) value on every done step, including timeouts.
            advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, 1, 1, None, None), out_axes=1)(
                trajectories.reward, trajectories.value, trajectories.next_value,
                trajectories.done, terminateds, config.gamma, config.gae_lambda
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
    with jax.named_scope("rppo_update"):
        epoch_losses = []
        for _ in range(config.num_epochs):
            loss, aux = update_step(model, optimizer, batch, config)
            epoch_losses.append((loss, aux))
    
    # 4. Count completed episodes
    num_completed = jnp.sum(trajectories.done)
    
    return next_env_state, next_h_state, key, epoch_losses, num_completed, trajectories
