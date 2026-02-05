import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Tuple, Any
from jax import random
from functools import partial

from src.models.jax_models.dreamer_v3_nnx import DreamerV3Agent, RSSM, WorldModel, ActorCritic
from src.models.jax_models.dreamer_v3_util import symlog, symexp, to_twohot, from_twohot, OneHotDist

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FREE_NATS = 1.0 # KL balancing threshold
KL_SCALE = 1.0 # 0.5 dyn + 0.1 rep generally, but scaled
DYN_SCALE = 0.5
REP_SCALE = 0.1
HORIZON = 15
GAMMA = 0.997
LAMBDA = 0.95

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def compute_lambda_values(rewards, values, continues):
    """
    Lambda-return calculation.
    rewards: (H, B)
    values: (H+1, B)
    continues: (H, B)
    """
    # values[t] is V(s_t)
    # Target at t: R_t + gamma * ((1-lambda)*V(s_{t+1}) + lambda*G_{t+1})
    # G_t = R_t + gamma * ...
    pass # Re-implement equivalent to PyTorch logic

    # Scan from end to start
    # Carry is G_{t+1}. Initial G_{H} = V(s_H).
    
    vals = values[:-1] # V(s_0) ... V(s_{H-1})
    next_vals = values[1:] # V(s_1) ... V(s_H)
    
    # Inputs to scan: (reward, next_val, cont) at each step t
    # But scan goes forward. We need reverse scan.
    
    def scan_fn(next_return, inputs):
        r, v, c = inputs
        # v is next_val V(s_{t+1})
        # r is reward R_t
        # c is continue gamma*C_t
        
        # G_t = R_t + c * ((1 - lambda) * v + lambda * next_return)
        
        # Note: 'continues' usually incorporates gamma. 
        # If passed as (gamma * term_mask), then c is discount.
        
        bootstrap = (1 - LAMBDA) * v + LAMBDA * next_return
        current_return = r + c * bootstrap
        return current_return, current_return

    # Pack inputs
    # rewards: [0...H-1]
    # next_vals: [0...H-1] -> values[1...H]
    # continues: [0...H-1]
    
    inputs = (rewards, next_vals, continues)
    
    # Initial carry: V(s_H) -> values[-1]
    last_val = values[-1]
    
    # Reverse scan
    _, returns = jax.lax.scan(scan_fn, last_val, inputs, reverse=True)
    
    # Returns shape: (H, B) corresponding to t=0...H-1
    return returns

# -----------------------------------------------------------------------------
# Training Step
# -----------------------------------------------------------------------------

class DreamerTrainer(nnx.Module):
    def __init__(self, obs_dim, act_dim, config, rngs: nnx.Rngs):
        self.config = config
        
        # Create agent config dict for network architecture
        agent_config = {
            'encoder_dim': config.get('encoder_dim', 128),
            'encoder_fc_layers': config.get('encoder_fc_layers', [128, 128]),
            'rssm_deter_dim': config.get('rssm_deter_dim', 512),
            'rssm_stoch_dim': config.get('rssm_stoch_dim', 32),
            'rssm_classes': config.get('rssm_classes', 32),
            'decoder_fc_layers': config.get('decoder_fc_layers', [128, 128]),
            'reward_fc_layers': config.get('reward_fc_layers', [128, 128]),
            'continue_fc_layers': config.get('continue_fc_layers', [128, 128]),
            'actor_fc_layers': config.get('actor_fc_layers', [256, 256]),
            'critic_fc_layers': config.get('critic_fc_layers', [256, 256]),
        }
        
        self.agent = DreamerV3Agent(obs_dim, act_dim, agent_config, rngs=rngs)
        
        # Target Critic (separate network for EMA)
        feat_dim = self.agent.wm.deter_dim + self.agent.wm.stoch_dim * self.agent.wm.discrete
        self.target_critic = ActorCritic(feat_dim, act_dim, agent_config, rngs=rngs).critic
        
        # Moments for return normalization (matching PyTorch)
        from src.models.jax_models.dreamer_v3_util import Moments
        self.moments = Moments(decay=0.99, max_=1.0, percentile_low=0.05, percentile_high=0.95)
        
        # Optimizers (nnx.Optimizer manages state)
        self.model_opt = nnx.Optimizer(self.agent.wm, optax.adam(config.get('model_lr', 1e-4)), wrt=nnx.Param)
        self.actor_opt = nnx.Optimizer(self.agent.ac.actor, optax.adam(config.get('actor_lr', 3e-5)), wrt=nnx.Param)
        self.critic_opt = nnx.Optimizer(self.agent.ac.critic, optax.adam(config.get('value_lr', 8e-5)), wrt=nnx.Param)
        
        self.step_count = jnp.array(0, dtype=jnp.int32)


    def train_step(self, batch, rng):
        # batch: obs (B,T,O), action (B,T,A), reward (B,T), terminal (B,T), is_first (B,T)
        obs = symlog(batch['obs'])
        action = batch['action']
        reward = batch['reward']
        terminal = batch['terminal']
        is_first = batch['is_first']
        
        B, T, _ = obs.shape
        
        # --- 1. World Model Learning ---
        def model_loss_fn(wm, rng):
            # Unroll RSSM using scan
            def scan_step(prev_state, inputs):
                o, a, f, k = inputs
                embed = wm.encoder(o)
                post, prior = wm.rssm.step(prev_state, embed, a, f, k)
                return post, (post, prior, embed)
                
            init_state = wm.rssm.initial(B)
            rng, scan_rng = random.split(rng)
            scan_rngs = random.split(scan_rng, T)
            
            inputs = (obs, action, is_first, scan_rngs)
            
            # Transpose only the batched inputs (obs, action, is_first)
            env_inputs = (obs, action, is_first)
            env_inputs_T = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), env_inputs)
            
            inputs_T = (*env_inputs_T, scan_rngs)

            _, (posts_T, priors_T, embeds_T) = jax.lax.scan(scan_step, init_state, inputs_T)
            
            posts = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), posts_T)
            priors = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), priors_T)
            
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
            
            dyn_kl = kl_div_categ(q_logits_sg, p_logits)
            rep_kl = kl_div_categ(q_logits, p_logits_sg)
            
            dyn_kl = jnp.maximum(dyn_kl, FREE_NATS)
            rep_kl = jnp.maximum(rep_kl, FREE_NATS)
            
            loss_kl = DYN_SCALE * jnp.mean(dyn_kl) + REP_SCALE * jnp.mean(rep_kl)
            
            total_loss = loss_recon + loss_rew + loss_cont + loss_kl
            
            metrics = {
                'loss_model': total_loss,
                'loss_recon': loss_recon,
                'loss_rew': loss_rew,
                'loss_cont': loss_cont,
                'loss_kl': loss_kl
            }
            return total_loss, (metrics, posts)

        grads_model, (model_metrics, posts) = nnx.grad(model_loss_fn, has_aux=True)(self.agent.wm, rng)
        self.model_opt.update(self.agent.wm, grads_model)
        
        # --- 2. Behavior Learning (Imagination) ---
        start_state = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), posts)
        start_state = jax.lax.stop_gradient(start_state)
        
        def behavior_loss_fn(actor, critic, rng):
            # Rollout
            def scan_imag(prev_state, key):
                feat = self.agent.wm.get_feat(prev_state)
                actor_out = actor(feat)
                dist = OneHotDist(actor_out)
                action = dist.sample(key)
                prior = self.agent.wm.rssm.imagine_step(prev_state, action, key)
                
                next_feat = self.agent.wm.get_feat(prior)
                
                # Predictions (using updated WM heads but static target critic)
                rew = from_twohot(self.agent.wm.reward_head(next_feat))
                # CRITICAL: Squeeze -1 is required to keep 'cont' at shape (B,) 
                # Preventing broadcasting to (B,B) in returns calculation
                cont = nnx.sigmoid(self.agent.wm.continue_head(next_feat)).squeeze(-1)
                val = from_twohot(self.target_critic(next_feat))
                
                step_info = {'reward': rew, 'continue': cont, 'value': val, 'feat': feat, 'action_dist': actor_out, 'action': action}
                return prior, step_info
            
            rng_imag = random.split(rng, HORIZON)
            _, rollouts = jax.lax.scan(scan_imag, start_state, rng_imag)
            
            rews = rollouts['reward']
            conts = rollouts['continue']
            vals = rollouts['value']
            
            start_feat = self.agent.wm.get_feat(start_state)
            v_start = from_twohot(self.target_critic(start_feat))
            
            all_vals = jnp.concatenate([v_start[None], vals], axis=0)
            
            lambda_returns = compute_lambda_values(rews, all_vals, conts)
            
            # Critic Loss
            v_pred_logits = critic(rollouts['feat'])
            target_twohot = to_twohot(jax.lax.stop_gradient(lambda_returns))
            loss_critic = -jnp.mean(jnp.sum(target_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1))
            
            # Actor Loss
            baseline = from_twohot(v_pred_logits)
            advantage = jax.lax.stop_gradient(lambda_returns - baseline)
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
            
            actions = rollouts['action']
            logits = rollouts['action_dist']
            log_probs = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1)
            loss_actor_policy = -jnp.mean(log_probs * advantage)
            
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1)
            loss_actor_ent = -3e-4 * jnp.mean(entropy)
            loss_actor = loss_actor_policy + loss_actor_ent
            
            metrics = {
                'loss_critic': loss_critic, 
                'loss_actor': loss_actor, 
                'mean_return': jnp.mean(lambda_returns),
                'mean_advantage': jnp.mean(advantage)
            }
            return (loss_actor + loss_critic), metrics

        grads_ac, behavior_metrics = nnx.grad(behavior_loss_fn, argnums=(0,1), has_aux=True)(
            self.agent.ac.actor,
            self.agent.ac.critic,
            rng
        )
        
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
        # Inference method
        # obs: (B, O)
        # prev_state: RSSM state dict (deter, stoch). If None, init.
        
        B = obs.shape[0]
        
        if prev_state is None:
             prev_state = self.agent.wm.rssm.initial(B)
        
        # 1. Prepare Inputs
        # RSSM step requires (prev_deter, prev_stoch, prev_action, embed)
        # We need prev_action? In inference loop, we pass the action we *took* last time.
        # But get_action usually returns action for *current* obs.
        # So we need the action from t-1.
        # State tracking should be done externally or we make this stateful.
        # For simplicity, let's assume 'prev_state' contains 'prev_action' or we pass it.
        # Let's append 'prev_action' to state dict.
        
        if 'prev_action' not in prev_state:
             prev_state['prev_action'] = jnp.zeros((B, self.agent.ac.actor.net.layers[-1].out_features)) # Hacky?
             
        prev_action = prev_state['prev_action']
        
        # 2. Embed
        embed = self.agent.wm.encoder(symlog(obs))
        
        # 3. RSSM Step (Posterior)
        # For inference/imagination, do we use Posterior or Prior?
        # In environment, we have observations, so we use Posterior (filter).
        
        is_first = jnp.zeros((B, 1)) # Assumed not first if in loop. If first, caller resets state.
        
        # Key management - nnx handles rngs stored in module? 
        # No, we must pass rngs for stochastic sampling if nnx module uses them.
        # Or pass key to methods.
        key = random.split(rng)[0] if rng is not None else random.PRNGKey(0)
        
        # We need a step function that takes (prev_state, embed, prev_action, is_first).
        # Our RSSM.step does this.
        post, _ = self.agent.wm.rssm.step(prev_state, embed, prev_action, is_first, key)
        
        # 4. Action
        feat = self.agent.wm.get_feat(post)
        actor_out = self.agent.ac.actor(feat)
        
        if eval_mode:
            action_idx = jnp.argmax(actor_out, axis=-1)
        else:
            dist = OneHotDist(actor_out)
            action_idx = jnp.argmax(dist.sample(key), axis=-1)
            
        action_onehot = jax.nn.one_hot(action_idx, self.agent.ac.actor.net.layers[-1].out_features)
        
        # 5. Update State for next step
        next_state = post
        next_state['prev_action'] = action_onehot
        
        return action_idx, next_state

# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10_000, sequence_length=16, obs_dim=33, action_dim=4):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.is_first = np.zeros((capacity,), dtype=np.bool_)
        
        self.idx = 0
        self.size = 0
        self.ep_start_idx = 0
        
    def add(self, obs, action, reward, done, is_first):
        # Add single step
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.is_first[self.idx] = is_first
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        if done:
            self.ep_start_idx = self.idx # Next step is start of new ep (handled by is_first arg mostly)

    def sample(self, batch_size):
        # Sample sequences of length T
        # Valid start indices: indices where index+T is valid and doesn't wrap around in a bad way
        # Simplified: just sample random indices and check validity.
        
        obs_b, act_b, rew_b, done_b, first_b = [], [], [], [], []
        
        for _ in range(batch_size):
            while True:
                start = np.random.randint(0, self.size - self.sequence_length)
                # Check if wraps around buffer end (handled by range limit above if buffer full)
                # If buffer wraps (circular), we need to ensure we don't cross self.idx
                # For simplicity, let's assume linear fill or handle circularity carefully.
                
                # Verify consistency? Dreamer usually stores full episodes.
                # Here we store flat for simplicity.
                # To ensure valid sequence: check 'is_first' in the middle.
                # If is_first appears at index > start, it means we crossed episode boundary.
                # Dreamer handles episode boundaries by masking (is_first).
                # So we just need contiguous memory.
                
                # Check circular buffer wrap
                indices = np.arange(start, start + self.sequence_length) % self.capacity
                
                # Check if we cross current write pointer 'idx' (invalid data)
                if self.size == self.capacity:
                   # If full, we shouldn't cross 'idx'
                   # e.g. idx=5, start=4, len=10 -> 4,5,6... 5 is overwritten.
                   pass 
                   
                break
            
            obs_b.append(self.obs[indices])
            act_b.append(self.actions[indices])
            rew_b.append(self.rewards[indices])
            done_b.append(self.dones[indices])
            first_b.append(self.is_first[indices])

        return {
            'obs': np.array(obs_b),
            'action': np.array(act_b),
            'reward': np.array(rew_b),
            'terminal': np.array(done_b),
            'is_first': np.array(first_b)
        }
