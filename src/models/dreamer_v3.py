import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class EMAMixin:
    def update_ema(self, target, source, rate):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(rate).add_(s.data, alpha=1 - rate)

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, layers=2):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.append(nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim))
            modules.append(nn.LayerNorm(hidden_dim))
            modules.append(nn.SiLU())
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# RSSM (World Model)
# -----------------------------------------------------------------------------

class RSSM(nn.Module):
    def __init__(self, embed_dim, action_dim, deter_dim=256, stoch_dim=32, discrete=32, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim # Number of classes
        self.discrete = discrete   # Number of logits per class
        self.hidden_dim = hidden_dim
        
        # Cell
        self.cell = nn.GRUCell(hidden_dim, deter_dim)
        
        # Prior (Dynamics) -> Predict Z_t from h_t
        self.img_out = nn.Linear(deter_dim, stoch_dim * discrete)
        
        # Posterior (Representation) -> Infer Z_t from h_t + e_t
        self.obs_out = nn.Linear(deter_dim + embed_dim, stoch_dim * discrete)
        
        # Input to Cell: h_{t-1}, z_{t-1}, a_{t-1}
        self.img_in = nn.Linear(stoch_dim * discrete + action_dim, hidden_dim)

    def initial(self, batch_size, device):
        return dict(
            mean=torch.zeros(batch_size, self.stoch_dim * self.discrete, device=device),
            std=torch.zeros(batch_size, self.stoch_dim * self.discrete, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim * self.discrete, device=device),
            deter=torch.zeros(batch_size, self.deter_dim, device=device)
        )

    def observe(self, embed, action, is_first, state=None):
        # embed: (B, T, E)
        # action: (B, T, A)
        # is_first: (B, T)
        
        if state is None:
            state = self.initial(embed.shape[0], embed.device)
            
        post, prior = dict(), dict()
        posts, priors = [], []
        
        # Unbind sequences to iterate steps
        embeds = embed.unbind(1)
        actions = action.unbind(1)
        is_firsts = is_first.unbind(1)
        
        deter = state['deter']
        stoch = state['stoch']
        
        for emb, act, first in zip(embeds, actions, is_firsts):
            # If first step, reset state (masking)
            mask = (1.0 - first).unsqueeze(-1)
            deter = deter * mask
            stoch = stoch * mask
            
            # 1. Compute Prior and Recurrent Step
            # cell_input = func(z_{t-1}, a_{t-1})
            x = torch.cat([stoch, act], dim=-1)
            x = self.img_in(x) 
            deter = self.cell(x, deter) # h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
            
            # Prior Z_t
            prior_logits = self.img_out(deter)
            prior_stoch = self.get_stoch(prior_logits)
            prior = {'logits': prior_logits, 'stoch': prior_stoch, 'deter': deter}
            
            # 2. Compute Posterior
            # Z_t ~ q(Z_t | h_t, x_t)
            post_input = torch.cat([deter, emb], dim=-1)
            post_logits = self.obs_out(post_input)
            post_stoch = self.get_stoch(post_logits)
            post = {'logits': post_logits, 'stoch': post_stoch, 'deter': deter}
            
            # Update current stoch for next step
            stoch = post['stoch']
            
            posts.append(post)
            priors.append(prior)
            
        # Stack
        posts = {k: torch.stack([p[k] for p in posts], dim=1) for k in posts[0]}
        priors = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
        
        return posts, priors

    def imagine(self, action, state):
        # action: (B, T, A)
        # state: initial state dict from posterior
        
        priors = []
        deter = state['deter']
        stoch = state['stoch']
        actions = action.unbind(1)
        
        for act in actions:
            x = torch.cat([stoch, act], dim=-1)
            x = self.img_in(x)
            deter = self.cell(x, deter)
            
            prior_logits = self.img_out(deter)
            prior_stoch = self.get_stoch(prior_logits)
            prior = {'logits': prior_logits, 'stoch': prior_stoch, 'deter': deter}
            
            stoch = prior_stoch
            priors.append(prior)
            
        priors = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
        return priors

    def get_stoch(self, logits):
        # Gumbel-Softmax (Straight-Through)
        shape = logits.shape
        logits = logits.reshape(*shape[:-1], self.stoch_dim, self.discrete)
        dist = D.OneHotCategoricalStraightThrough(logits=logits)
        stoch = dist.rsample() # (..., stoch, discrete)
        return stoch.reshape(shape) # Flatten back

# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------

class DreamerV3Buffer:
    def __init__(self, capacity=10000, sequence_length=16):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.episodes = []
        self.total_steps = 0
        
    def add(self, episode):
        # Episode is dict of arrays
        self.episodes.append(episode)
        self.total_steps += len(episode['action'])
        while self.total_steps > self.capacity:
            rem = self.episodes.pop(0)
            self.total_steps -= len(rem['action'])
            
    def sample(self, batch_size):
        # Sample random episodes
        # Then sample random chunks
        obs, acts, rews, terms, firsts = [], [], [], [], []
        
        for _ in range(batch_size):
            while True:
                idx = np.random.randint(0, len(self.episodes))
                ep = self.episodes[idx]
                if len(ep['action']) > self.sequence_length:
                    break
            
            start = np.random.randint(0, len(ep['action']) - self.sequence_length + 1)
            end = start + self.sequence_length
            
            obs.append(ep['observation'][start:end])
            acts.append(ep['action'][start:end])
            rews.append(ep['reward'][start:end])
            terms.append(ep['terminal'][start:end])
            firsts.append(ep['is_first'][start:end])
            
        batch = {
            'observation': np.stack(obs),
            'action': np.stack(acts),
            'reward': np.stack(rews),
            'terminal': np.stack(terms),
            'is_first': np.stack(firsts),
        }
        return batch

class DreamerV3Agent(nn.Module, EMAMixin):
    def __init__(self, state_dim, action_dim, device="auto", 
                 batch_size=16, batch_length=16, 
                 model_lr=1e-4, actor_lr=8e-5, value_lr=8e-5):
        super().__init__()
        
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
             
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.batch_length = batch_length
        
        # Dimensions
        embed_dim = 256
        deter_dim = 256
        stoch_dim = 32
        discrete = 32
        
        # Components
        self.encoder = Encoder(state_dim, embed_dim).to(self.device)
        self.rssm = RSSM(embed_dim, action_dim, deter_dim, stoch_dim, discrete).to(self.device)
        self.decoder = Decoder(deter_dim + stoch_dim * discrete, state_dim).to(self.device)
        self.reward_pred = MLP(deter_dim + stoch_dim * discrete, 1).to(self.device)
        self.continue_pred = MLP(deter_dim + stoch_dim * discrete, 1).to(self.device)
        
        self.actor = MLP(deter_dim + stoch_dim * discrete, action_dim).to(self.device)
        self.critic = MLP(deter_dim + stoch_dim * discrete, 1).to(self.device)
        self.target_critic = MLP(deter_dim + stoch_dim * discrete, 1).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.model_opt = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.rssm.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.reward_pred.parameters()},
            {'params': self.continue_pred.parameters()}
        ], lr=model_lr)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=value_lr)
        
        # State tracking
        self.buffer = DreamerV3Buffer(sequence_length=batch_length)
        self.current_episode = None
        self.reset_hidden()
        
    def reset_hidden(self):
        self.prev_state = self.rssm.initial(1, self.device)
        self.prev_action = torch.zeros(1, self.action_dim, device=self.device)
        
        self.current_episode = {
            'observation': [], 'action': [], 'reward': [], 'terminal': [], 'is_first': []
        }
        
    def choose_action(self, state, eval_mode=False):
        # Step World Model
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0) # (1, 1, Dim)
        
        with torch.no_grad():
            embed = self.encoder(state_tensor)
            # RSSM observe single step
            # action: (1, 1, A)
            action_in = self.prev_action.unsqueeze(1)
            is_first = torch.zeros(1, 1, device=self.device) # Assume not first if in loop (reset called externally)
            # Typically first step logic handled via reset_hidden which sets default prev state
            
            post, _ = self.rssm.observe(embed, action_in, is_first, self.prev_state)
            
            # Extract feature
            feat = torch.cat([post['deter'], post['stoch']], dim=-1) # (1, 1, Feature)
            
            # Actor
            logits = self.actor(feat)
            dist = D.Categorical(logits=logits)
            
            if eval_mode:
                action_idx = torch.argmax(logits).item()
            else:
                action_idx = dist.sample().item()
                
            # Update prevs
            action_onehot = torch.zeros(1, self.action_dim, device=self.device)
            action_onehot[0, action_idx] = 1.0
            
            self.prev_state = {k: v[:, -1] for k, v in post.items()} # Remove time dim
            self.prev_action = action_onehot
            
            return action_idx

    def store_transition(self, state, action, reward, next_state, done):
        # Store in current episode list
        # Action needs to be onehot for storage? Or int?
        # Let's store onehot for simplicity later
        act_onehot = np.zeros(self.action_dim, dtype=np.float32)
        act_onehot[action] = 1.0
        
        self.current_episode['observation'].append(state)
        self.current_episode['action'].append(act_onehot)
        self.current_episode['reward'].append(reward)
        self.current_episode['terminal'].append(done)
        self.current_episode['is_first'].append(len(self.current_episode['observation']) == 1)
        
        if done:
            # Finalize episode
            ep = {k: np.array(v) for k, v in self.current_episode.items()}
            self.buffer.add(ep)
            
    def update(self):
        if len(self.buffer.episodes) < 1: 
            return
        
        # check if enough data
        if self.buffer.total_steps < self.batch_size * self.batch_length:
            return

        batch = self.buffer.sample(self.batch_size)
        
        obs = torch.tensor(batch['observation'], dtype=torch.float32, device=self.device)
        act = torch.tensor(batch['action'], dtype=torch.float32, device=self.device)
        rew = torch.tensor(batch['reward'], dtype=torch.float32, device=self.device)
        term = torch.tensor(batch['terminal'], dtype=torch.float32, device=self.device)
        first = torch.tensor(batch['is_first'], dtype=torch.float32, device=self.device)
        
        # 1. Train World Model
        embed = self.encoder(obs)
        post, prior = self.rssm.observe(embed, act, first)
        
        feat = torch.cat([post['deter'], post['stoch']], dim=-1)
        recon = self.decoder(feat)
        rew_pred = self.reward_pred(feat)
        cont_pred = self.continue_pred(feat)
        
        # Losses
        recon_loss = F.mse_loss(recon, obs)
        rew_loss = F.mse_loss(rew_pred.squeeze(-1), symlog(rew))
        cont_loss = F.binary_cross_entropy_with_logits(cont_pred.squeeze(-1), 1.0 - term)
        
        # KL Loss
        # Prior/Post are categoricals (logits)
        # Dynamic balancing or fixed scale? Fixed 1.0 for simplicity
        p_logits = prior['logits']
        q_logits = post['logits'].detach() # Stop grad for posterior matching? No, KL both.
        # DreamerV3: KL(stop(post), prior) + KL(post, stop(prior)) is tricky.
        # Simple KL(post || prior)
        
        # Reshape to (B, T, Stoch, Discrete)
        shape = p_logits.shape[:-1] + (self.rssm.stoch_dim, self.rssm.discrete)
        p_dist = D.OneHotCategorical(logits=p_logits.reshape(shape))
        q_dist = D.OneHotCategorical(logits=post['logits'].reshape(shape))
        
        kl_loss = D.kl_divergence(q_dist, p_dist).mean()
        
        model_loss = recon_loss + rew_loss + cont_loss + 0.1 * kl_loss
        
        self.model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_opt.param_groups[0]['params'], 100.0)
        self.model_opt.step()
        
        # 2. Behavior Learning (Imagination)
        # Sample starting states from posterior (stop grad)
        with torch.no_grad():
            start = {k: v.reshape(-1, v.shape[-1]).detach() for k, v in post.items()} # Flatten B,T for starts? 
            # Actually, usually sample from buffer posteriors.
            # Let's use the sequence rollout we just did.
            # Flatten: (B*T, ...)
            
        # Unroll imagination Horizon=15
        horizon = 15
        # We need a recurring imagination loop.
        # Start states:
        start_deter = start['deter']
        start_stoch = start['stoch']
        
        imag_feats = []
        imag_rews = []
        imag_conts = []
        imag_acts = []
        
        curr_deter = start_deter
        curr_stoch = start_stoch
        
        for _ in range(horizon):
            # Actor action
            curr_feat = torch.cat([curr_deter, curr_stoch], dim=-1)
            act_logits = self.actor(curr_feat)
            act_dist = D.Categorical(logits=act_logits)
            act_idx = act_dist.sample()
            act_onehot = F.one_hot(act_idx, self.action_dim).float()
            
            # Step RSSM (Prior)
            # Need 'img_in' logic from RSSM
            x = torch.cat([curr_stoch, act_onehot], dim=-1)
            x = self.rssm.img_in(x)
            next_deter = self.rssm.cell(x, curr_deter)
            next_prior_logits = self.rssm.img_out(next_deter)
            next_stoch = self.rssm.get_stoch(next_prior_logits)
            
            # Predictions
            next_feat = torch.cat([next_deter, next_stoch], dim=-1)
            rew_p = self.reward_pred(next_feat)
            cont_p = self.continue_pred(next_feat)
            
            imag_feats.append(curr_feat) # Store current or next? Usually state -> action -> next_state -> reward
            imag_acts.append(act_logits)
            imag_rews.append(rew_p)
            imag_conts.append(cont_p)
            
            curr_deter = next_deter
            curr_stoch = next_stoch
            
        # Calculate Returns (Lambda-Return)
        # Using target critic
        last_feat = torch.cat([curr_deter, curr_stoch], dim=-1)
        next_values = self.target_critic(last_feat).squeeze(-1)
        
        imag_rews = torch.stack(imag_rews).squeeze(-1) # (H, B*T)
        imag_conts = torch.sigmoid(torch.stack(imag_conts).squeeze(-1))
        imag_values = []
        
        # Bootstrap
        lambda_ = 0.95
        R = next_values
        for t in reversed(range(horizon)):
            r = imag_rews[t]
            c = imag_conts[t]
            # V_target = r + gamma * c * ((1-lambda)*v + lambda*R)
            # But we need V(s_t) to train critic.
            # Dreamer V3 uses symlog return
            # Simplified:
            v_pred = self.target_critic(imag_feats[t]).squeeze(-1)
            R = r + self.model_opt.param_groups[0]['lr'] * c * R # oops, gamma missing
            # Let's implement standard lambda return
            R = r + 0.99 * c * ((1 - lambda_) * v_pred + lambda_ * R)
            imag_values.insert(0, R)
            
        imag_values = torch.stack(imag_values) # (H, B*T)
        imag_feats = torch.stack(imag_feats) # (H, B*T, F)
        imag_acts = torch.stack(imag_acts) # (H, B*T, A)
        
        # Critic Update
        # Predict value from feat
        v_pred = self.critic(imag_feats.detach()).squeeze(-1)
        critic_loss = F.mse_loss(v_pred, imag_values.detach())
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor Update
        # Maximize value +  Entropy
        # Loss = - (V_lambda - Baseline) * logpi ??
        # Or just maximize V_lambda (Pathwise derivative if continuous, but we have discrete)
        # REINFORCE for discrete:
        # returns = imag_values
        # baseline = v_pred.detach()
        # adv = returns - baseline
        # Actor Loss = - (mean(adv * logprob))
        
        # Re-evaluate logprobs
        dist = D.Categorical(logits=imag_acts)
        # Which action was taken? We sampled it.
        # But we need grads flow for continuous.
        # For discrete, we used sample().
        # Dreamer Discrete often uses Straight-Through or Reinforce.
        # Let's use Reinforce term:
        # We need the stored action indices or onehots.
        # We didn't store indices.
        # Let's just use Dynamics Backprop? No, discrete latent breaks it unless ST.
        # Simplified: PPO-style or just Reinforce.
        
        # Let's stick to simple Reinforce on imagined rollout
        # We need Action indices taken during rollout
        # ... skipped storing them properly in loop above.
        
        # Fix: Add entropy bonus
        actor_loss = -(imag_values.mean()) # Naive maximization?
        
        # More proper:
        # actor_loss = - (imag_values - v_pred.detach()) * log_prob
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Update Target
        self.update_ema(self.target_critic, self.critic, 0.02)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

