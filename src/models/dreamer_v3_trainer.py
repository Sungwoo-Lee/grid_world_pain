import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Tuple, Any, Optional
from jax import random
from functools import partial

from src.models.dreamer_v3_nnx import DreamerV3Agent, RSSM, WorldModel, ActorCritic
from src.models.dreamer_v3_util import symlog, symexp, to_twohot, from_twohot, OneHotDist
from src.environment.core import jax_step, jax_reset
from src.environment.sensor import get_observation

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FREE_NATS = 1.0
KL_SCALE = 1.0
DYN_SCALE = 0.5
REP_SCALE = 0.1
HORIZON = 15
GAMMA = 0.997
LAMBDA = 0.95

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def compute_lambda_values(rewards, values, continues, LAMBDA=0.95):
    """
    Lambda-return calculation with global GAMMA.
    rewards: (H, B)
    values: (H+1, B)
    continues: (H, B) - should include both Head output and global discount
    """
    next_vals = values[1:]
    
    # We incorporate GAMMA here if not already in continues
    # In DreamerV3, 'continues' usually comes from the head * global_discount
    
    def scan_fn(next_return, inputs):
        r, v, c = inputs
        bootstrap = (1 - LAMBDA) * v + LAMBDA * next_return
        current_return = r + c * bootstrap
        return current_return, current_return

    inputs = (rewards, next_vals, continues)
    last_val = values[-1]

    _, returns = jax.lax.scan(scan_fn, last_val, inputs, reverse=True)

    return returns

# -----------------------------------------------------------------------------
# Training Step
# -----------------------------------------------------------------------------

class DreamerTrainer(nnx.Module):
    def __init__(self, obs_dim, act_dim, config, rngs: nnx.Rngs,
                 obs_breakdown: Optional[dict] = None,
                 modulation_config=None):
        self.config = config
        self.modulation_config = modulation_config

        agent_config = {
            'encoder_dim': config.get_mandatory('agent.encoder_dim', int),
            'encoder_fc_layers': config.get_mandatory('agent.encoder_fc_layers'),
            'rssm_deter_dim': config.get_mandatory('agent.rssm_deter_dim', int),
            'rssm_stoch_dim': config.get_mandatory('agent.rssm_stoch_dim', int),
            'rssm_classes': config.get_mandatory('agent.rssm_classes', int),
            'decoder_fc_layers': config.get_mandatory('agent.decoder_fc_layers'),
            'reward_fc_layers': config.get_mandatory('agent.reward_fc_layers'),
            'continue_fc_layers': config.get_mandatory('agent.continue_fc_layers'),
            'actor_fc_layers': config.get_mandatory('agent.actor_fc_layers'),
            'critic_fc_layers': config.get_mandatory('agent.critic_fc_layers'),
            'use_layer_norm': config.get_mandatory('agent.use_layer_norm', bool),
            
            # Hierarchical Encoding Params
            'encoding_mode': config.get_mandatory('agent.encoding_mode', str),
            'hierarchical_params': config.to_dict().get('agent', {}).get('hierarchical_params', {})
        }

        self.agent = DreamerV3Agent(obs_dim, act_dim, agent_config, rngs=rngs,
                                    obs_breakdown=obs_breakdown,
                                    modulation_config=modulation_config)

        feat_dim = self.agent.wm.deter_dim + self.agent.wm.stoch_dim * self.agent.wm.discrete
        self.target_critic = ActorCritic(feat_dim, act_dim, agent_config, rngs=rngs).critic

        from src.models.dreamer_v3_util import Moments
        self.moments = Moments(decay=0.99, max_=1.0, percentile_low=0.05, percentile_high=0.95)

        self.model_opt = nnx.Optimizer(
            self.agent.wm,
            optax.chain(
                optax.clip_by_global_norm(1000.0),
                optax.adam(config.get_mandatory('agent.model_lr', float), eps=1e-8)
            ),
            wrt=nnx.Param
        )
        self.actor_opt = nnx.Optimizer(
            self.agent.ac.actor,
            optax.chain(
                optax.clip_by_global_norm(100.0),  # Actor usually clipped more strictly
                optax.adam(config.get_mandatory('agent.actor_lr', float), eps=1e-5)
            ),
            wrt=nnx.Param
        )
        self.critic_opt = nnx.Optimizer(
            self.agent.ac.critic,
            optax.chain(
                optax.clip_by_global_norm(100.0),
                optax.adam(config.get_mandatory('agent.value_lr', float), eps=1e-5)
            ),
            wrt=nnx.Param
        )

        self.step_count = jnp.array(0, dtype=jnp.int32)


    @nnx.jit
    def train_step(self, batch, rng):
        obs = symlog(batch['obs'])
        action = batch['action']
        reward = batch['reward']
        terminal = batch['terminal']
        is_first = batch['is_first']

        B, T, _ = obs.shape
        modulation_enabled = self.agent.wm.modulation_enabled

        # --- 1. World Model Learning ---
        def model_loss_fn(wm, rng):
            # OPTIMIZATION: Pre-compute encoder embeddings for all timesteps (outside scan)
            # This enables batched parallel encoding instead of sequential per-timestep encoding
            # Expected speedup: 2-5x on world model training step
            with jax.named_scope("wm_encoder"):
                if modulation_enabled:
                    # For modulated path, we need modulator outputs for both encoding AND memory gate
                    # We do one scan for modulator+encoder, then main RSSM scan
                    def mod_scan(h_mod, o):
                        mod_output, h_mod_new = wm.modulator.forward_obs(o, h_mod)
                        return h_mod_new, (mod_output, h_mod_new)

                    obs_T = jnp.swapaxes(obs, 0, 1)  # (T, B, obs_dim)
                    h_mod_init = wm.modulator.initial_state(B)
                    _, (mod_outputs_T, h_mods_T) = jax.lax.scan(mod_scan, h_mod_init, obs_T)

                    # Vectorized encoder call
                    mod_outputs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), mod_outputs_T)
                    embeds = wm.encoder.forward_with_modulation(
                        obs, mod_outputs, wm.modulation_type,
                        unimodal_ln=getattr(wm, 'mod_unimodal_ln', None),
                        multimodal_ln=getattr(wm, 'mod_multimodal_ln', None),
                        flat_ln=getattr(wm, 'mod_flat_ln', None)
                    )
                    embeds_T = jnp.swapaxes(embeds, 0, 1)

                    # Now main RSSM scan uses pre-computed embeddings and modulator outputs
                    def scan_step(prev_state, inputs):
                        embed, a, f, k, mod_output = inputs
                        post, prior = wm.rssm.step(
                            prev_state, embed, a, f, k,
                            gate_bias=mod_output.z_memory)
                        return post, (post, prior)

                    init_carry = wm.rssm.initial(B)
                else:
                    embeds = wm.encoder(
                        obs,
                        unimodal_ln=getattr(wm, 'mod_unimodal_ln', None),
                        multimodal_ln=getattr(wm, 'mod_multimodal_ln', None),
                        flat_ln=getattr(wm, 'mod_flat_ln', None)
                    )
                    embeds_T = jnp.swapaxes(embeds, 0, 1)  # (T, B, embed_dim)

                    def scan_step(prev_state, inputs):
                        embed, a, f, k = inputs
                        post, prior = wm.rssm.step(prev_state, embed, a, f, k)
                        return post, (post, prior)

                    init_carry = wm.rssm.initial(B)

            with jax.named_scope("wm_rssm_scan"):
                rng, scan_rng = random.split(rng)
                # Split keys for (T, B) to ensure independent sampling per environment per step
                scan_rngs = random.split(scan_rng, T * B).reshape((T, B, -1))

                env_inputs = (action, is_first)
                env_inputs_T = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), env_inputs)

                if modulation_enabled:
                    inputs_T = (embeds_T, *env_inputs_T, scan_rngs, mod_outputs_T)
                else:
                    inputs_T = (embeds_T, *env_inputs_T, scan_rngs)

                _, scan_outputs = jax.lax.scan(scan_step, init_carry, inputs_T)

                # Extract posts and priors from scan outputs
                if modulation_enabled:
                    posts_T, priors_T = scan_outputs
                    # h_mods_T already computed in encode_scan
                    h_mods_all = jnp.swapaxes(h_mods_T, 0, 1)
                else:
                    posts_T, priors_T = scan_outputs
                    h_mods_all = None

                posts = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), posts_T)
                priors = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), priors_T)

            with jax.named_scope("wm_losses"):
                # Reconstruction Loss
                feat = wm.get_feat(posts)
                recon = wm.decoder(feat)
                loss_recon = jnp.mean(jnp.square(recon - obs))

                # Reward Loss
                rew_pred = wm.reward_head(feat)
                rew_target = to_twohot(reward)
                loss_rew = -jnp.mean(jnp.sum(rew_target * jax.nn.log_softmax(rew_pred), axis=-1))

                # Continue Loss
                cont_pred = wm.continue_head(feat)
                loss_cont = optax.sigmoid_binary_cross_entropy(cont_pred, 1.0 - terminal[..., None]).mean()

                # KL Loss
                q_logits = posts['logits']
                p_logits = priors['logits']

                def kl_div_categ(p_logits, q_logits):
                    p_dist = jax.nn.softmax(p_logits)
                    p_log = jax.nn.log_softmax(p_logits)
                    q_log = jax.nn.log_softmax(q_logits)
                    return jnp.sum(p_dist * (p_log - q_log), axis=-1)

                q_logits_sg = jax.lax.stop_gradient(q_logits)
                p_logits_sg = jax.lax.stop_gradient(p_logits)

                dyn_kl = kl_div_categ(q_logits_sg, p_logits)  # (B, T, stoch, discrete) -> (B, T, stoch)
                rep_kl = kl_div_categ(q_logits, p_logits_sg)

                # Sum over latent groups (stoch_dim) to get total info loss per state
                dyn_kl = jnp.sum(dyn_kl, axis=-1)
                rep_kl = jnp.sum(rep_kl, axis=-1)

                dyn_kl = jnp.maximum(dyn_kl, FREE_NATS)
                rep_kl = jnp.maximum(rep_kl, FREE_NATS)

                loss_kl = DYN_SCALE * jnp.mean(dyn_kl) + REP_SCALE * jnp.mean(rep_kl)

                total_loss = loss_recon + loss_rew + loss_cont + loss_kl

            # Error metrics (non-gradient)
            rew_pred_val = from_twohot(rew_pred)
            rew_error = jnp.mean(jnp.abs(rew_pred_val - reward))
            
            # Directional Reward MAE
            pos_mask = (reward > 0.01).astype(jnp.float32)
            neg_mask = (reward < -0.01).astype(jnp.float32)
            rew_mae_pos = jnp.sum(jnp.abs(rew_pred_val - reward) * pos_mask) / (jnp.sum(pos_mask) + 1e-8)
            rew_mae_neg = jnp.sum(jnp.abs(rew_pred_val - reward) * neg_mask) / (jnp.sum(neg_mask) + 1e-8)

            # Latent Entropy
            q_dist = jax.nn.softmax(q_logits)
            latent_entropy = -jnp.sum(q_dist * jax.nn.log_softmax(q_logits), axis=-1).mean()

            # Continue Accuracy
            cont_target = 1.0 - terminal[..., None]
            cont_acc = jnp.mean((nnx.sigmoid(cont_pred) > 0.5) == cont_target.astype(jnp.bool_))

            metrics = {
                'loss_model': total_loss,
                'loss_recon': loss_recon,
                'loss_rew': loss_rew,
                'loss_cont': loss_cont,
                'loss_dyn_kl': jnp.mean(dyn_kl),
                'loss_rep_kl': jnp.mean(rep_kl),
                'loss_kl': loss_kl,
                'model_reward_mae': rew_error,
                'model_reward_mae_pos': rew_mae_pos,
                'model_reward_mae_neg': rew_mae_neg,
                'model_latent_entropy': latent_entropy,
                'model_cont_acc': cont_acc,
            }

            if modulation_enabled:
                if wm.modulation_type == "FiLM":
                    effective_gamma_uni = mod_outputs_T.z_unimodal
                    effective_gamma_multi = mod_outputs_T.z_multimodal
                else:
                    effective_gamma_uni = jax.nn.sigmoid(mod_outputs_T.z_unimodal)
                    effective_gamma_multi = jax.nn.sigmoid(mod_outputs_T.z_multimodal)

                metrics.update({
                    'mod_z_unimodal_mean': jnp.mean(effective_gamma_uni),
                    'mod_z_unimodal_std': jnp.std(effective_gamma_uni),
                    'mod_z_multimodal_mean': jnp.mean(effective_gamma_multi),
                    'mod_z_multimodal_std': jnp.std(effective_gamma_multi),
                    'mod_memory_mean': jnp.mean(mod_outputs_T.z_memory),
                    'mod_memory_std': jnp.std(mod_outputs_T.z_memory),
                    'mod_z_reward_mean': jnp.mean(mod_outputs_T.z_reward),
                    'mod_z_reward_std': jnp.std(mod_outputs_T.z_reward),
                })
                if wm.modulation_type in ("PreActivation", "FiLM"):
                    metrics.update({
                        'mod_beta_unimodal_mean': jnp.mean(mod_outputs_T.z_unimodal_add),
                        'mod_beta_multimodal_mean': jnp.mean(mod_outputs_T.z_multimodal_add),
                    })

            return total_loss, (metrics, posts, h_mods_all)

        with jax.named_scope("dreamer_optim"):
            grads_model, (model_metrics, posts, h_mods_all) = nnx.grad(
                model_loss_fn, has_aux=True)(self.agent.wm, rng)
            self.model_opt.update(self.agent.wm, grads_model)

        # --- 2. Behavior Learning (Imagination) ---
        start_state = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), posts)
        start_state = jax.lax.stop_gradient(start_state)

        # Prepare modulator hidden state for imagination initialization
        if modulation_enabled:
            h_mod_start = h_mods_all.reshape((-1,) + h_mods_all.shape[2:])
            h_mod_start = jax.lax.stop_gradient(h_mod_start)

        # Pre-compute moments parameters for advantage normalization (OUTSIDE grad)
        # This avoids tracing through self.moments inside nnx.grad which causes OOM
        moments_low = jax.lax.stop_gradient(self.moments.low.value)
        moments_high = jax.lax.stop_gradient(self.moments.high.value)
        moments_invscale = jnp.maximum(1.0 / self.moments.max_, moments_high - moments_low)

        def behavior_loss_fn(actor, critic, rng):
            if modulation_enabled:
                def scan_imag(carry, key):
                    prev_state, h_mod = carry
                    feat = self.agent.wm.get_feat(prev_state)
                    actor_out = actor(feat)
                    dist = OneHotDist(actor_out)
                    action = dist.sample(key)

                    # Modulator imagination mode
                    mod_input = jnp.concatenate([feat, action], axis=-1)
                    mod_output, h_mod_new = self.agent.wm.modulator.forward_imagine(
                        mod_input, h_mod)

                    # RSSM imagine step with gate-bias
                    prior = self.agent.wm.rssm.imagine_step(
                        prev_state, action, key,
                        gate_bias=mod_output.z_memory)

                    next_feat = self.agent.wm.get_feat(prior)
                    rew = from_twohot(self.agent.wm.reward_head(next_feat))
                    # Injection C: Reward interpretation scale (imagination only)
                    rew = rew * mod_output.z_reward.squeeze(-1)
                    cont = nnx.sigmoid(
                        self.agent.wm.continue_head(next_feat)).squeeze(-1)
                    val = from_twohot(self.target_critic(next_feat))

                    step_info = {
                        'reward': rew, 'continue': cont, 'value': val,
                        'feat': feat, 'action_dist': actor_out, 'action': action
                    }
                    return (prior, h_mod_new), step_info

                imag_init = (start_state, h_mod_start)
            else:
                def scan_imag(prev_state, key):
                    feat = self.agent.wm.get_feat(prev_state)
                    actor_out = actor(feat)
                    dist = OneHotDist(actor_out)
                    action = dist.sample(key)
                    prior = self.agent.wm.rssm.imagine_step(prev_state, action, key)

                    next_feat = self.agent.wm.get_feat(prior)
                    rew = from_twohot(self.agent.wm.reward_head(next_feat))
                    cont = nnx.sigmoid(
                        self.agent.wm.continue_head(next_feat)).squeeze(-1)
                    val = from_twohot(self.target_critic(next_feat))

                    step_info = {
                        'reward': rew, 'continue': cont, 'value': val,
                        'feat': feat, 'action_dist': actor_out, 'action': action
                    }
                    return prior, step_info

                imag_init = start_state

            with jax.named_scope("ac_imagine_scan"):
                # Split keys for (HORIZON, IMAG_BATCH) for behavior learning
                # IMAG_BATCH = B * T (flattened start_state)
                imag_batch = start_state['deter'].shape[0]
                rng_imag = random.split(rng, HORIZON * imag_batch).reshape((HORIZON, imag_batch, -1))
                _, rollouts = jax.lax.scan(scan_imag, imag_init, rng_imag)

            with jax.named_scope("ac_losses"):
                rews = rollouts['reward']
                conts = rollouts['continue']
                vals = rollouts['value']

                start_feat = self.agent.wm.get_feat(start_state)
                v_start = from_twohot(self.target_critic(start_feat))

                all_vals = jnp.concatenate([v_start[None], vals], axis=0)

                # Lambda returns with global discount
                lambda_returns = compute_lambda_values(rews, all_vals, conts * GAMMA)

                norm_returns = (lambda_returns - moments_low) / moments_invscale

                # Cumulative Discount Weighting
                # weights[t] = \prod_{i=0}^{t-1} (conts[i] * GAMMA)
                discount_weights = jnp.concatenate([jnp.ones_like(conts[:1]), conts[:-1] * GAMMA], axis=0)
                discount_weights = jnp.cumprod(discount_weights, axis=0)
                discount_weights = jax.lax.stop_gradient(discount_weights)

                # Critic Loss — train on RAW lambda_returns (canonical DreamerV3)
                v_pred_logits = critic(rollouts['feat'])
                target_twohot = to_twohot(jax.lax.stop_gradient(lambda_returns))
                loss_critic_step = -jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
                loss_critic = jnp.mean(loss_critic_step * discount_weights)

                # Actor Loss — normalize BOTH sides for consistent advantage
                baseline = from_twohot(v_pred_logits)
                norm_baseline = (baseline - moments_low) / moments_invscale
                advantage = jax.lax.stop_gradient(norm_returns - norm_baseline)

                actions = rollouts['action']
                logits = rollouts['action_dist']
                log_probs = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1)

                ENTROPY_SCALE = self.config.get_mandatory('agent.entropy_scale', float)
                entropy = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)

                loss_actor_step = -(log_probs * advantage + ENTROPY_SCALE * entropy)
                loss_actor = jnp.mean(loss_actor_step * discount_weights)

                metrics = {
                    'loss_critic': loss_critic,
                    'loss_actor': loss_actor,
                    'loss_actor_policy': jnp.mean(-log_probs * advantage * discount_weights),
                    'loss_actor_entropy': jnp.mean(-ENTROPY_SCALE * entropy * discount_weights),
                    'mean_return': jnp.mean(lambda_returns),
                    'mean_norm_return': jnp.mean(norm_returns),
                    'mean_value': jnp.mean(baseline),
                    'mean_advantage': jnp.mean(advantage),
                    'mean_entropy': jnp.mean(entropy),
                    'value_mae': jnp.mean(jnp.abs(baseline - jax.lax.stop_gradient(lambda_returns)))
                }
            return (loss_actor + loss_critic), (metrics, lambda_returns)

        with jax.named_scope("dreamer_optim"):
            grads_ac, (behavior_metrics, lambda_returns) = nnx.grad(behavior_loss_fn, argnums=(0,1), has_aux=True)(
                self.agent.ac.actor,
                self.agent.ac.critic,
                rng
            )

            # --- 3. Update Moments (Outside Grad/Trace) ---
            self.moments.update(lambda_returns)

            grads_actor, grads_critic = grads_ac
            self.actor_opt.update(self.agent.ac.actor, grads_actor)
            self.critic_opt.update(self.agent.ac.critic, grads_critic)

            # EMA Update
            current_st = nnx.state(self.agent.ac.critic, nnx.Param)
            target_st = nnx.state(self.target_critic, nnx.Param)
            new_target_st = jax.tree.map(lambda t, c: 0.98 * t + 0.02 * c, target_st, current_st)
            nnx.update(self.target_critic, new_target_st)

        return {**model_metrics, **behavior_metrics}

    def get_action(self, obs, prev_state=None, eval_mode=False, rng=None):
        """Inference method with optional neuromodulation.

        Args:
            obs: Observation, shape (B, O).
            prev_state: RSSM state dict. If None, initialized to zeros.
                        When modulation is enabled, may contain 'mod_h' key.
            eval_mode: If True, use argmax (greedy) action selection.
            rng: PRNG key for stochastic sampling.
        """
        B = obs.shape[0]
        modulation_enabled = self.agent.wm.modulation_enabled

        if prev_state is None:
            prev_state = self.agent.wm.rssm.initial(B)

        if 'prev_action' not in prev_state:
            prev_state['prev_action'] = jnp.zeros(
                (B, self.agent.ac.actor.net.layers[-1].out_features))

        prev_action = prev_state['prev_action']
        obs_symlog = symlog(obs)

        key = random.split(rng)[0] if rng is not None else random.PRNGKey(0)
        is_first = jnp.zeros((B, 1))

        if modulation_enabled:
            if 'mod_h' not in prev_state:
                mod_h = self.agent.wm.modulator.initial_state(B)
            else:
                mod_h = prev_state['mod_h']

            mod_output, mod_h_new = self.agent.wm.modulator.forward_obs(
                obs_symlog, mod_h)

            embed = self.agent.wm.encoder.forward_with_modulation(
                obs_symlog, mod_output, self.agent.wm.modulation_type,
                unimodal_ln=getattr(self.agent.wm, 'mod_unimodal_ln', None),
                multimodal_ln=getattr(self.agent.wm, 'mod_multimodal_ln', None),
                flat_ln=getattr(self.agent.wm, 'mod_flat_ln', None)
            )

            gate_bias = mod_output.z_memory
        else:
            embed = self.agent.wm.encoder(
                obs_symlog,
                unimodal_ln=getattr(self.agent.wm, 'mod_unimodal_ln', None),
                multimodal_ln=getattr(self.agent.wm, 'mod_multimodal_ln', None),
                flat_ln=getattr(self.agent.wm, 'mod_flat_ln', None)
            )
            gate_bias = None

        post, _ = self.agent.wm.rssm.step(
            prev_state, embed, prev_action, is_first, key,
            gate_bias=gate_bias)

        feat = self.agent.wm.get_feat(post)
        actor_out = self.agent.ac.actor(feat)

        if eval_mode:
            action_idx = jnp.argmax(actor_out, axis=-1)
        else:
            dist = OneHotDist(actor_out)
            action_idx = jnp.argmax(dist.sample(key), axis=-1)

        action_onehot = jax.nn.one_hot(
            action_idx, self.agent.ac.actor.net.layers[-1].out_features)

        next_state = post
        next_state['prev_action'] = action_onehot
        if modulation_enabled:
            next_state['mod_h'] = mod_h_new

        return action_idx, next_state

    @nnx.jit(static_argnums=(3,))
    def collect_sequence(self, env_state, params, num_steps, key, dreamer_state=None):
        """Collects a sequence of transitions using jax.lax.scan.
        Includes auto-reset on done.
        """
        B = env_state.agent_pos.shape[0]
        if dreamer_state is None:
            dreamer_state = self.agent.wm.rssm.initial(B)
            # Add prev_action for consistency if not present
            if 'prev_action' not in dreamer_state:
                dreamer_state['prev_action'] = jnp.zeros(
                    (B, self.agent.ac.actor.net.layers[-1].out_features))
            # Initial step is always 'first'
            dreamer_state['is_first'] = jnp.ones((B, 1))
            
            # Initial modulator state if enabled
            if self.agent.wm.modulation_enabled:
                dreamer_state['mod_h'] = self.agent.wm.modulator.initial_state(B)
        
        def scan_fn(carry, _):
            state, d_state, current_key = carry

            # 1. Sensing
            with jax.named_scope("dreamer_sense"):
                obs = jax.vmap(get_observation, in_axes=(0, None))(state, params)

            # 2. Action selection
            with jax.named_scope("dreamer_act"):
                current_key, act_key = jax.random.split(current_key)
                action_idx, next_d_state = self.get_action(
                    obs, d_state, eval_mode=False, rng=act_key)

            # 3. Step Environment
            with jax.named_scope("dreamer_env_step"):
                action_idx = action_idx.astype(jnp.int32)
                next_state_raw, reward, done, info = jax.vmap(
                    jax_step, in_axes=(0, 0, None))(state, action_idx, params)

            # 4. Auto-Reset
            with jax.named_scope("dreamer_env_reset"):
                current_key, reset_key = jax.random.split(current_key)
                reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(
                    params, jax.random.split(reset_key, B))

                def select_done(d, r, n):
                    d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
                    return jnp.where(d_expanded, r, n)

                final_env_state = jax.tree_util.tree_map(
                    lambda r, n: select_done(done, r, n),
                    reset_state, next_state_raw
                )
            
            # 5. Prepare Dreamer state for NEXT step
            # On reset, we should reset RSSM state too? 
            # Dreamer typically handles this via 'is_first' flag in RSSM.step
            # but here get_action handles it. 
            # We need to set 'is_first' for the NEXT get_action call.
            next_d_state['is_first'] = done[..., None].astype(jnp.float32)
            
            # Record transition
            transition = {
                'obs': obs,
                'action': jax.nn.one_hot(action_idx, self.agent.ac.actor.net.layers[-1].out_features),
                'reward': reward,
                'terminal': done,
                'is_first': d_state.get('is_first', jnp.zeros((B, 1))),
                'ate_food': info['ate_food'].astype(jnp.float32),
                'hit_predator': info['hit_predator'].astype(jnp.float32),
                'hit_hiding_predator': info['hit_hiding_predator'].astype(jnp.float32),
                'event_collided': info['event_collided'].astype(jnp.float32),
                'rested': info['rested'].astype(jnp.float32),
                'damage': info['damage'],
                'damage_predator': info['damage_predator'],
                'damage_hiding_predator': info['damage_hiding_predator'],
                'damage_obstacle': info['damage_obstacle'],
                'dist_to_food': info['dist_to_food'],
                'dist_to_pred': info['dist_to_pred'],
                'termination_reason': info['termination_reason'].astype(jnp.float32),
            }
            
            return (final_env_state, next_d_state, current_key), transition

        (final_env_state, final_d_state, final_key), transitions = jax.lax.scan(
            scan_fn, (env_state, dreamer_state, key), None, length=num_steps)
        
        return final_env_state, final_d_state, final_key, transitions

    @nnx.jit(static_argnums=(1, 2, 7, 8, 10, 11, 12, 13))
    def _scan_train_gpu(self, graphdef, num_steps, rng,
                        main_arrays, pos_arrays,
                        b_size, b_cap, b_seq_len,
                        pos_size, pos_cap,
                        pos_slots, recent_slots, recent_window, buf_idx):
        obs, actions, rewards, dones, is_first = main_arrays
        pos_obs, pos_actions, pos_rewards, pos_dones, pos_is_first = pos_arrays

        num_blocks = b_size // b_seq_len
        max_blocks = b_cap // b_seq_len
        num_pos_blocks = pos_size // b_seq_len
        max_pos_blocks = pos_cap // b_seq_len
        seq_range = jnp.arange(b_seq_len)

        # Functional state split
        _, state = nnx.split(self)

        def scan_body(carry, _):
            current_state, rng = carry
            
            # Re-create a local trainer instance inside the JIT trace
            trainer = nnx.merge(graphdef, current_state)
            batch_size = trainer.config.get_mandatory('agent.batch_size', int)
            
            # Mixture sampling logic inside JIT (zero host involvement)
            rng, key_pos, key_recent, key_uniform, train_key = jax.random.split(rng, 5)
            uniform_slots = batch_size - pos_slots - recent_slots

            with jax.named_scope("replay_mixture_sample"):
                # --- Pool 1: Positive-reward buffer ---
                # Sample uniformly from the positive buffer
                pos_block_idx = jax.random.randint(
                    key_pos, (pos_slots,), 0, jnp.maximum(num_pos_blocks, 1))
                pos_starts = pos_block_idx * b_seq_len
                pos_indices = (pos_starts[:, None] + seq_range[None, :]) % pos_cap

                pos_batch_obs = pos_obs[pos_indices]           # (pos_slots, seq_len, obs_dim)
                pos_batch_act = pos_actions[pos_indices]       # (pos_slots, seq_len, act_dim)
                pos_batch_rew = pos_rewards[pos_indices]       # (pos_slots, seq_len)
                pos_batch_done = pos_dones[pos_indices]        # (pos_slots, seq_len)
                pos_batch_first = pos_is_first[pos_indices]    # (pos_slots, seq_len)

                # --- Pool 2: Recent blocks from main buffer ---
                recent_blocks_count = jnp.minimum(recent_window // b_seq_len, num_blocks)
                buf_block = buf_idx // b_seq_len
                recent_offsets = jax.random.randint(
                    key_recent, (recent_slots,), 0, jnp.maximum(recent_blocks_count, 1))
                recent_block_idx = (buf_block - recent_blocks_count + recent_offsets) % max_blocks

                # Fallback: if not enough data, use uniform from main buffer
                recent_fallback = jax.random.randint(key_recent, (recent_slots,), 0, num_blocks)
                recent_block_idx = jnp.where(recent_blocks_count > 0, recent_block_idx, recent_fallback)

                recent_starts = recent_block_idx * b_seq_len
                recent_indices = (recent_starts[:, None] + seq_range[None, :]) % b_cap

                recent_batch_obs = obs[recent_indices]
                recent_batch_act = actions[recent_indices]
                recent_batch_rew = rewards[recent_indices]
                recent_batch_done = dones[recent_indices]
                recent_batch_first = is_first[recent_indices]

                # --- Pool 3: Uniform random from main buffer (existing behavior) ---
                uniform_block_idx = jax.random.randint(key_uniform, (uniform_slots,), 0, num_blocks)
                uniform_starts = uniform_block_idx * b_seq_len
                uniform_indices = (uniform_starts[:, None] + seq_range[None, :]) % b_cap

                uniform_batch_obs = obs[uniform_indices]
                uniform_batch_act = actions[uniform_indices]
                uniform_batch_rew = rewards[uniform_indices]
                uniform_batch_done = dones[uniform_indices]
                uniform_batch_first = is_first[uniform_indices]

                # --- Fallback: if positive buffer is empty, replace with uniform from main ---
                # When num_pos_blocks == 0, pos_batch_* contains garbage or zeros
                fallback_block_idx = jax.random.randint(key_pos, (pos_slots,), 0, num_blocks)
                fallback_starts = fallback_block_idx * b_seq_len
                fallback_indices = (fallback_starts[:, None] + seq_range[None, :]) % b_cap

                has_positive_data = num_pos_blocks > 0
                pos_batch_obs = jnp.where(has_positive_data, pos_batch_obs, obs[fallback_indices])
                pos_batch_act = jnp.where(has_positive_data, pos_batch_act, actions[fallback_indices])
                pos_batch_rew = jnp.where(has_positive_data, pos_batch_rew, rewards[fallback_indices])
                pos_batch_done = jnp.where(has_positive_data, pos_batch_done, dones[fallback_indices])
                pos_batch_first = jnp.where(has_positive_data, pos_batch_first, is_first[fallback_indices])

            # --- Concatenate all pools into the training batch ---
            with jax.named_scope("replay_concat"):
                batch = {
                    'obs': jnp.concatenate([pos_batch_obs, recent_batch_obs, uniform_batch_obs], axis=0),
                    'action': jnp.concatenate([pos_batch_act, recent_batch_act, uniform_batch_act], axis=0),
                    'reward': jnp.concatenate([pos_batch_rew, recent_batch_rew, uniform_batch_rew], axis=0),
                    'terminal': jnp.concatenate([pos_batch_done, recent_batch_done, uniform_batch_done], axis=0),
                    'is_first': jnp.concatenate([pos_batch_first, recent_batch_first, uniform_batch_first], axis=0),
                }
            
            # gradient step
            metrics = trainer.train_step(batch, train_key)
            
            # Extract the locally mutated state to carry forward
            new_state = nnx.state(trainer)
            return (new_state, rng), metrics
        
        final_carry, all_metrics = jax.lax.scan(scan_body, (state, rng), None, length=num_steps)
        metrics_mean = jax.tree.map(jnp.mean, all_metrics)
        return final_carry[0], metrics_mean, final_carry[1]

    def train_multiple_gpu(self, buffer, num_steps, rng, positive_buffer=None):
        """Fully JIT-compiled training loop for GPU buffer.
        
        Samples and trains inside lax.scan — zero host involvement.
        """
        # 1. Functional split for structural description
        graphdef, _ = nnx.split(self)
        
        # Extract buffer arrays to pass explicitly (prevents JIT retracing)
        buffer_arrays = (buffer.obs, buffer.actions, buffer.rewards, buffer.dones, buffer.is_first)
        
        # Mixture sampling config (static)
        sampling_mode = self.config.get_mandatory('agent.sampling_mode')
        pos_slots = self.config.get_mandatory('agent.mixture_positive_slots') if sampling_mode == 'mixture' else 0
        recent_slots = self.config.get_mandatory('agent.mixture_recent_slots') if sampling_mode == 'mixture' else 0
        recent_window = self.config.get_mandatory('agent.mixture_recent_window') if sampling_mode == 'mixture' else 0

        # Positive buffer arrays (or zeros if not using mixture / positive buffer empty)
        if positive_buffer is not None and positive_buffer.size > 0:
            pos_arrays = (positive_buffer.obs, positive_buffer.actions, positive_buffer.rewards,
                          positive_buffer.dones, positive_buffer.is_first)
            pos_size = positive_buffer.size
            pos_cap = positive_buffer.capacity
        else:
            # Dummy arrays — will be ignored when pos_slots fallback triggers
            pos_arrays = (buffer.obs, buffer.actions, buffer.rewards, buffer.dones, buffer.is_first)
            pos_size = 0
            pos_cap = buffer.capacity

        # 2. Execute the stable JIT training loop
        # We pass graphdef as it is hashable and constant
        final_state, metrics_mean, rng = self._scan_train_gpu(
            graphdef, int(num_steps), rng, 
            buffer_arrays, pos_arrays,
            buffer.size, buffer.capacity, buffer.sequence_length,
            pos_size, pos_cap,
            pos_slots, recent_slots, recent_window, buffer.idx
        )
        
        # 3. Apply the final aggregated state back to our real self
        nnx.update(self, final_state)
        
        return metrics_mean, rng

    def train_multiple_cpu(self, stacked_batches, rng):
        """Batched JIT training for CPU buffer (Path A fallback).
        
        stacked_batches: pre-sampled dict of (num_steps, batch_size, seq_len, dim)
        """
        graphdef, state = nnx.split(self)
        
        @nnx.jit
        def _scan_train(state, stacked_batches, rng):
            def scan_body(carry, batch):
                current_state, rng = carry
                
                trainer = nnx.merge(graphdef, current_state)
                
                rng, train_key = jax.random.split(rng)
                metrics = trainer.train_step(batch, train_key)
                
                new_state = nnx.state(trainer)
                return (new_state, rng), metrics
            
            final_carry, all_metrics = jax.lax.scan(scan_body, (state, rng), stacked_batches)
            return final_carry[0], jax.tree.map(jnp.mean, all_metrics), final_carry[1]
        
        final_state, metrics_mean, rng = _scan_train(state, stacked_batches, rng)
        nnx.update(self, final_state)
        
        return metrics_mean, rng

    def _sample_mixture_cpu(self, buffer, positive_buffer, num_batches, batch_size):
        """CPU-path mixture sampling (called outside JIT)."""
        seq_len = buffer.sequence_length
        sampling_mode = self.config.get('agent.sampling_mode', 'uniform')
        pos_slots = self.config.get('agent.mixture_positive_slots', 0) if sampling_mode == 'mixture' else 0
        recent_slots = self.config.get('agent.mixture_recent_slots', 0) if sampling_mode == 'mixture' else 0
        recent_window = self.config.get('agent.mixture_recent_window', 10000) if sampling_mode == 'mixture' else 0
        uniform_slots = batch_size - pos_slots - recent_slots
        
        all_batches = []
        for _ in range(num_batches):
            # Pool 1: Positive buffer
            if positive_buffer is not None and positive_buffer.size >= seq_len:
                pos_batch = positive_buffer.sample(pos_slots)
            else:
                pos_batch = buffer.sample(pos_slots)  # fallback
            
            if pos_batch is None: # Extreme fallback
                 pos_batch = buffer.sample(pos_slots)

            # Pool 2: Recent from main buffer
            num_blocks = buffer.size // seq_len
            max_blocks = buffer.capacity // seq_len
            recent_blocks_count = min(recent_window // seq_len, num_blocks)
            buf_block = buffer.idx // seq_len

            if recent_blocks_count > 0:
                offsets = np.random.randint(0, recent_blocks_count, size=recent_slots)
                recent_block_idx = (buf_block - recent_blocks_count + offsets) % max_blocks
            else:
                recent_block_idx = np.random.randint(0, max(num_blocks, 1), size=recent_slots)

            seq_range = np.arange(seq_len)
            recent_indices = (recent_block_idx[:, None] * seq_len + seq_range[None, :]) % buffer.capacity
            recent_batch = {
                'obs': buffer.obs[recent_indices],
                'action': buffer.actions[recent_indices],
                'reward': buffer.rewards[recent_indices],
                'terminal': buffer.dones[recent_indices],
                'is_first': buffer.is_first[recent_indices],
            }

            # Pool 3: Uniform from main buffer
            uniform_batch = buffer.sample(uniform_slots)

            # Concatenate
            combined = {}
            for key in ['obs', 'action', 'reward', 'terminal', 'is_first']:
                combined[key] = np.concatenate([pos_batch[key], recent_batch[key], uniform_batch[key]], axis=0)
            all_batches.append(combined)
        
        # Stack into (num_batches, batch_size, seq_len, dim) and transfer once
        stacked = {k: jnp.array(np.stack([b[k] for b in all_batches])) for k in all_batches[0]}
        return stacked

# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10_000, sequence_length=16, obs_dim=33, action_dim=4, device="gpu"):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        self._on_gpu = (device == "gpu")
        
        if self._on_gpu:
            self.obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
            self.actions = jnp.zeros((capacity, action_dim), dtype=jnp.float32)
            self.rewards = jnp.zeros((capacity,), dtype=jnp.float32)
            self.dones = jnp.zeros((capacity,), dtype=jnp.float32)
            self.is_first = jnp.zeros((capacity,), dtype=jnp.float32)
        else:
            self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rewards = np.zeros((capacity,), dtype=np.float32)
            self.dones = np.zeros((capacity,), dtype=np.float32)
            self.is_first = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add_batch(self, obs, actions, rewards, dones, is_firsts):
        """Vectorized addition of a batch of transitions.
        
        Caller must pass data in ENV-MAJOR order: the first sequence_length
        entries = env0's trajectory, next sequence_length = env1's, etc.
        This ensures sample() returns temporal sequences for the RSSM.
        
        obs: (num_items, obs_dim)
        actions: (num_items, act_dim)
        rewards: (num_items,)
        dones: (num_items,)
        is_firsts: (num_items,)
        """
        num_items = obs.shape[0]
        
        if self._on_gpu:
            indices = (self.idx + jnp.arange(num_items)) % self.capacity
            self.obs = self.obs.at[indices].set(obs)
            self.actions = self.actions.at[indices].set(actions)
            self.rewards = self.rewards.at[indices].set(rewards)
            self.dones = self.dones.at[indices].set(dones)
            self.is_first = self.is_first.at[indices].set(is_firsts)
        else:
            indices = (self.idx + np.arange(num_items)) % self.capacity
            self.obs[indices] = obs
            self.actions[indices] = actions
            self.rewards[indices] = rewards
            self.dones[indices] = dones
            self.is_first[indices] = is_firsts
        
        self.idx = (self.idx + num_items) % self.capacity
        self.size = min(self.size + num_items, self.capacity)

    def sample(self, batch_size, key=None):
        """Sample a batch of sequences.
        
        For GPU mode: `key` is a JAX PRNG key (required).
        For CPU mode: `key` is ignored, uses numpy RNG.
        """
        # Sample sequences that are TEMPORAL (one env over time).
        # Buffer is stored in env-major order: block of sequence_length consecutive
        # slots = one env's trajectory. So we sample start indices that are multiples
        # of sequence_length.
        if self.size <= self.sequence_length:
            return None  # Not enough data
        num_blocks = self.size // self.sequence_length
        if num_blocks < 1:
            return None
            
        seq_range = (jnp.arange if self._on_gpu else np.arange)(self.sequence_length)
        
        if self._on_gpu:
            block_indices = jax.random.randint(key, (batch_size,), 0, num_blocks)
            starts = block_indices * self.sequence_length
            indices = (starts[:, None] + seq_range[None, :]) % self.capacity
        else:
            block_indices = np.random.randint(0, num_blocks, size=batch_size)
            starts = block_indices * self.sequence_length
            indices = (starts[:, None] + seq_range[None, :]) % self.capacity

        return {
            'obs': self.obs[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'terminal': self.dones[indices],
            'is_first': self.is_first[indices] if self._on_gpu else self.is_first[indices].astype(np.float32)
        }

    def sample_multiple(self, num_batches, batch_size, key=None):
        """Pre-sample `num_batches` batches at once (for CPU mode batched JIT)."""
        if self._on_gpu:
            # For GPU: not needed — sample inside lax.scan instead
            raise NotImplementedError("Use sample() inside lax.scan for GPU mode")
        
        batches = [self.sample(batch_size) for _ in range(num_batches)]
        # Stack into (num_batches, batch_size, seq_len, dim) and transfer once
        stacked = {k: jnp.array(np.stack([b[k] for b in batches])) for k in batches[0]}
        return stacked
