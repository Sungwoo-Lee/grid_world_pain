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

class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Any:
        # We don't have fabric, so we don't gather. Assuming single GPU/CPU for now.
        # gathered_x = fabric.all_gather(x).float().detach() 
        gathered_x = x.float().detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()

def compute_lambda_values(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    lmbda: float = 0.95,
):
    # ret = torch.zeros_like(values)
    # ret[:, -1] = values[:, -1]
    # for t in reversed(range(values.shape[1] - 1)):
    #     ret[:, t] = rewards[:, t] + continues[:, t] * ((1 - lmbda) * values[:, t+1] + lmbda * ret[:, t+1])
    # return ret
    
    # SheepRL implementation:
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret

def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    return f

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)
    
    def forward(self, input, state):
        gates_ih = self.ln_ih(self.weight_ih(input))
        gates_hh = self.ln_hh(self.weight_hh(state))
        gates = gates_ih + gates_hh
        reset_gate, update_gate, candidate_gate = gates.chunk(3, dim=-1)
        
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate) # DreamerV3 often uses tanh here too
        
        return (1 - update_gate) * state + update_gate * candidate_gate

def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255):
    x = symlog(x)
    # Range of symlog values. symlog(20) ~= 3.0. 
    # Let's align with official implementation which uses generic buckets.
    # Official: 255 buckets usually cover expected return range.
    # If we assume range is [-20, 20], symlog range is approx [-3, 3].
    # We map this range to [0, 255].
    
    # Scale x to [0, num_buckets - 1]
    # x_norm = (x - min_v) / (max_v - min_v) * (num_buckets - 1)
    
    # But wait, DreamerV3 uses symlog on the *targets* before bucketing? 
    # Or buckets represent symlog values?
    # Usually: values are symlog-ed, then discretized into buckets.
    # Let's assume input x is already UN-transformed value.
    
    # Paper: "We transform the targets to symlog(x) ... and discretize them into 255 buckets..."
    
    vals = torch.linspace(symlog(torch.tensor(min_v)), symlog(torch.tensor(max_v)), num_buckets, device=x.device)
    # But efficient scatter implementation:
    
    # Simplified logic:
    # 1. Symlog the input
    x = symlog(x)
    # 2. Normalize to [0, num_buckets-1]
    # We need fixed boundaries. 
    # Transformation: typically we just define a fixed wide range.
    # Let's fix range to e.g. [-20, 20] in SYMLOG space? No that's huge. symlog(e^20) is huge.
    # 
    # Let's follow simple TwoHot regression logic:
    # min_v, max_v are in RAW space.
    # No, usually buckets are linear in Symlog space.
    
    # Let's trust the params provided: min_v, max_v are raw.
    bottom = symlog(torch.tensor(min_v, device=x.device))
    top = symlog(torch.tensor(max_v, device=x.device))
    
    # Clip
    x = torch.clamp(x, bottom, top)
    
    # Project to [0, num_buckets-1]
    # (x - bottom) / (top - bottom) * (B - 1)
    rel = (x - bottom) / (top - bottom) * (num_buckets - 1)
    
    floor = rel.floor().long()
    ceil = rel.ceil().long()
    
    prob_ceil = rel - floor.float()
    prob_floor = 1.0 - prob_ceil
    
    # Create target distribution
    target = torch.zeros((*x.shape, num_buckets), device=x.device)
    
    # Scatter
    # We need to handle arbitrary batch dims
    # Easier: one_hot(floor) * prob_floor + one_hot(ceil) * prob_ceil
    
    floor = torch.clamp(floor, 0, num_buckets - 1)
    ceil = torch.clamp(ceil, 0, num_buckets - 1)
    
    target.scatter_add_(-1, floor.unsqueeze(-1), prob_floor.unsqueeze(-1))
    target.scatter_add_(-1, ceil.unsqueeze(-1), prob_ceil.unsqueeze(-1))
    
    # Handle the case where floor == ceil (exact integer)? 
    # scatter_add handles it (adds probability twice? No, indices are same, values sum up to 1.0)
    # Wait, if floor==ceil, we add to same index twice?
    # prob_ceil is 0, prob_floor is 1. If floor==ceil, rel is integer.
    # Actually if indices are same, scatter_add sums them.
    # If floor == ceil, prob_floor + prob_ceil = 1.0. Correct.
    
    return target

def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255):
    # Logits -> Probs -> Expectation in Symlog Space -> Symexp
    probs = F.softmax(logits, dim=-1)
    
    bottom = symlog(torch.tensor(min_v, device=logits.device))
    top = symlog(torch.tensor(max_v, device=logits.device))
    
    # Bucket values in symlog space
    bucket_vals = torch.linspace(bottom, top, num_buckets, device=logits.device)
    
    # Expected value in symlog space
    sym_val = (probs * bucket_vals).sum(dim=-1)
    
    return symexp(sym_val)

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, fc_layers=[256]):
        super().__init__()
        # Encoder mapping to embedding space
        self.net = MLP(input_dim, embed_dim, fc_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.net(x)
        x = self.ln(x)
        return self.act(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, fc_layers=[256, 256]):
        super().__init__()
        self.net = MLP(input_dim, output_dim, fc_layers)
        
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[256]):
        super().__init__()
        modules = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.LayerNorm(h_dim))
            modules.append(nn.SiLU())
            in_dim = h_dim
            
        modules.append(nn.Linear(in_dim, output_dim))
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
        self.cell = LayerNormGRUCell(hidden_dim, deter_dim)
        
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
                 batch_size=None, batch_length=None, 
                 model_lr=None, actor_lr=None, value_lr=None,
                 encoder_dim=None, encoder_fc_layers=None,
                 rssm_deter_dim=None, rssm_stoch_dim=None, rssm_classes=None,
                 decoder_fc_layers=None,
                 reward_fc_layers=None,
                 continue_fc_layers=None,
                 actor_fc_layers=None,
                 critic_fc_layers=None):
        super().__init__()
        
        # Validation
        if batch_size is None: raise ValueError("DreamerV3Agent: 'batch_size' must be specified in config.")
        if batch_length is None: raise ValueError("DreamerV3Agent: 'batch_length' must be specified in config.")
        if model_lr is None: raise ValueError("DreamerV3Agent: 'model_lr' must be specified in config.")
        if actor_lr is None: raise ValueError("DreamerV3Agent: 'actor_lr' must be specified in config.")
        if value_lr is None: raise ValueError("DreamerV3Agent: 'value_lr' must be specified in config.")
        if encoder_dim is None: raise ValueError("DreamerV3Agent: 'encoder_dim' must be specified in config.")
        if encoder_fc_layers is None: raise ValueError("DreamerV3Agent: 'encoder_fc_layers' must be specified in config.")
        if rssm_deter_dim is None: raise ValueError("DreamerV3Agent: 'rssm_deter_dim' must be specified in config.")
        if rssm_stoch_dim is None: raise ValueError("DreamerV3Agent: 'rssm_stoch_dim' must be specified in config.")
        if rssm_classes is None: raise ValueError("DreamerV3Agent: 'rssm_classes' must be specified in config.")
        if decoder_fc_layers is None: raise ValueError("DreamerV3Agent: 'decoder_fc_layers' must be specified in config.")
        if reward_fc_layers is None: raise ValueError("DreamerV3Agent: 'reward_fc_layers' must be specified in config.")
        if continue_fc_layers is None: raise ValueError("DreamerV3Agent: 'continue_fc_layers' must be specified in config.")
        if actor_fc_layers is None: raise ValueError("DreamerV3Agent: 'actor_fc_layers' must be specified in config.")
        if critic_fc_layers is None: raise ValueError("DreamerV3Agent: 'critic_fc_layers' must be specified in config.")
        
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
             
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.batch_length = batch_length
        
        # Dimensions
        self.embed_dim = encoder_dim
        self.deter_dim = rssm_deter_dim
        self.stoch_dim = rssm_stoch_dim
        self.discrete = rssm_classes
        
        # Components
        self.encoder = Encoder(state_dim, self.embed_dim, encoder_fc_layers).to(self.device)
        self.rssm = RSSM(self.embed_dim, action_dim, self.deter_dim, self.stoch_dim, self.discrete, hidden_dim=self.deter_dim).to(self.device)
        
        feat_dim = self.deter_dim + self.stoch_dim * self.discrete
        
        self.decoder = Decoder(feat_dim, state_dim, decoder_fc_layers).to(self.device)
        self.reward_pred = MLP(feat_dim, 255, reward_fc_layers).to(self.device)
        self.continue_pred = MLP(feat_dim, 1, continue_fc_layers).to(self.device)
        
        self.actor = MLP(feat_dim, action_dim, actor_fc_layers).to(self.device)
        self.critic = MLP(feat_dim, 255, critic_fc_layers).to(self.device)
        self.target_critic = MLP(feat_dim, 255, critic_fc_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # [NEW] Moments for Return Normalization
        self.moments = Moments(decay=0.99, max_=1.0, percentile_low=0.05, percentile_high=0.95).to(self.device)
        
        # [NEW] Initialization
        self.encoder.apply(uniform_init_weights(1.0))
        self.decoder.apply(uniform_init_weights(1.0))
        self.rssm.apply(uniform_init_weights(1.0))
        self.reward_pred.apply(uniform_init_weights(1.0))
        self.continue_pred.apply(uniform_init_weights(1.0))
        self.actor.apply(uniform_init_weights(1.0))
        self.critic.apply(uniform_init_weights(1.0))
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
            # Invoke self() to trigger Input hooks for ActivationMonitor
            logits, post = self(state_tensor)
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


    def forward(self, state_tensor):
        """
        Forward pass for visualization and single-step inference.
        state_tensor: (B, T, D)
        """
        embed = self.encoder(state_tensor)
        
        # Determine previous action/state
        # For warmup (B=1, T=1), match self.prev_action shape or reset
        B = state_tensor.shape[0]
        if self.prev_action.shape[0] != B:
             # Batch size mismatch (e.g. warmup uses 1, but maybe internal state is different?)
             # Reset temp state
             prev_action = torch.zeros(B, self.action_dim, device=self.device)
             state = self.rssm.initial(B, self.device)
             is_first = torch.zeros(B, 1, device=self.device)
        else:
             prev_action = self.prev_action
             state = self.prev_state
             is_first = torch.zeros(B, 1, device=self.device)
             
        # RSSM Observe
        # prev_action: (B, A) -> (B, 1, A)
        post, _ = self.rssm.observe(embed, prev_action.unsqueeze(1), is_first, state)
        
        # Extract feature
        feat = torch.cat([post['deter'], post['stoch']], dim=-1) # (B, T, F)
        
        # Actor
        logits = self.actor(feat)
        
        # We need to expose 'post' for choose_action to update state?
        # Or we attach it to self? No, forward shouldn't side-effect mostly.
        # But choose_action NEEDS to side-effect.
        # Let's keep choose_action logic duplicated or split.
        # If I strictly implement forward for WARMUP, I can ignore side effects.
        # But hooks need to catch encoder/rssm/actor.
        # This implementation does that.
        
        # For choose_action, we can continue using old logic OR call forward.
        # If I don't change choose_action, I avoid breaking it.
        # And forward is ONLY used by Warmup.
        return logits, post


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
        
        # [NEW] Symlog Inputs
        obs = symlog(obs) 
        
        # 1. Train World Model
        embed = self.encoder(obs)
        post, prior = self.rssm.observe(embed, act, first)
        
        feat = torch.cat([post['deter'], post['stoch']], dim=-1)
        recon = self.decoder(feat)
        rew_pred = self.reward_pred(feat)
        cont_pred = self.continue_pred(feat)
        
        # Losses
        recon_loss = F.mse_loss(recon, obs)
        
        # [NEW] Reward TwoHot Loss
        rew_target = to_twohot(rew)
        rew_loss = -torch.mean(torch.sum(rew_target * F.log_softmax(rew_pred, dim=-1), dim=-1))
        
        cont_loss = F.binary_cross_entropy_with_logits(cont_pred.squeeze(-1), 1.0 - term)
        
        # [NEW] KL Balancing
        # DreamerV3: 0.1 * KL(sg(post) || prior) + 0.9 * KL(post || sg(prior)) (Dynamics vs Repr coefficients)
        # Using specific coefficients: 0.5 for dyn, 0.1 for rep (from SheepRL/Paper)
        
        p_logits = prior['logits']
        q_logits = post['logits']
        
        # Reshape to (B, T, Stoch, Discrete)
        shape = p_logits.shape[:-1] + (self.rssm.stoch_dim, self.rssm.discrete)
        p_dist = D.OneHotCategorical(logits=p_logits.reshape(shape))
        q_dist = D.OneHotCategorical(logits=q_logits.reshape(shape))
        
        # Detach for Dynamics Loss (Prior learning to match Posterior)
        q_dist_detach = D.OneHotCategorical(logits=q_logits.detach().reshape(shape))
        dyn_kl = D.kl_divergence(q_dist_detach, p_dist) # KL(sg(q) || p)
        dyn_loss = torch.max(dyn_kl, torch.tensor(1.0, device=self.device)).mean()
        
        # Detach for Representation Loss (Posterior learning to be predictable)
        p_dist_detach = D.OneHotCategorical(logits=p_logits.detach().reshape(shape))
        rep_kl = D.kl_divergence(q_dist, p_dist_detach) # KL(q || sg(p))
        rep_loss = torch.max(rep_kl, torch.tensor(1.0, device=self.device)).mean()
        
        kl_loss = 0.5 * dyn_loss + 0.1 * rep_loss
        
        model_loss = recon_loss + rew_loss + cont_loss + kl_loss
        
        self.model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_opt.param_groups[0]['params'], 100.0)
        self.model_opt.step()
        
        # 2. Behavior Learning (Imagination)
        with torch.no_grad():
            start = {k: v.reshape(-1, v.shape[-1]).detach() for k, v in post.items()}
            
        horizon = 15
        start_deter = start['deter']
        start_stoch = start['stoch']
        
        imag_feats = []
        imag_rews = []
        imag_conts = []
        imag_acts = []
        imag_log_probs = [] # [NEW] For Reinforce
        
        curr_deter = start_deter
        curr_stoch = start_stoch
        
        for _ in range(horizon):
            curr_feat = torch.cat([curr_deter, curr_stoch], dim=-1)
            act_logits = self.actor(curr_feat)
            act_dist = D.Categorical(logits=act_logits)
            act_idx = act_dist.sample()
            act_onehot = F.one_hot(act_idx, self.action_dim).float()
            
            # [NEW] Store log prob
            log_prob = act_dist.log_prob(act_idx)
            imag_log_probs.append(log_prob)
            
            x = torch.cat([curr_stoch, act_onehot], dim=-1)
            x = self.rssm.img_in(x)
            next_deter = self.rssm.cell(x, curr_deter)
            next_prior_logits = self.rssm.img_out(next_deter)
            next_stoch = self.rssm.get_stoch(next_prior_logits)
            
            next_feat = torch.cat([next_deter, next_stoch], dim=-1)
            rew_p = self.reward_pred(next_feat)
            cont_p = self.continue_pred(next_feat)
            
            imag_feats.append(curr_feat)
            imag_acts.append(act_logits)
            
            # [NEW] Decode TwoHot Reward
            rew_scalar = from_twohot(rew_p)
            imag_rews.append(rew_scalar)
            
            imag_conts.append(cont_p)
            
            curr_deter = next_deter
            curr_stoch = next_stoch
            
        last_feat = torch.cat([curr_deter, curr_stoch], dim=-1)
        # [NEW] Decode Target Value
        next_values = from_twohot(self.target_critic(last_feat))
        
        imag_rews = torch.stack(imag_rews) # (H, B*T)
        imag_conts = torch.sigmoid(torch.stack(imag_conts).squeeze(-1))
        
        lambda_values = compute_lambda_values(
             imag_rews,
             next_values, 
             imag_conts,
             lmbda=0.95
        )
        
        imag_feats = torch.stack(imag_feats)
        imag_log_probs = torch.stack(imag_log_probs)
        
        # Critic Update
        v_pred_logits = self.critic(imag_feats.detach())
        # [NEW] Target TwoHot
        target_value_twohot = to_twohot(lambda_values.detach())
        critic_loss = -torch.mean(torch.sum(target_value_twohot * F.log_softmax(v_pred_logits, dim=-1), dim=-1))
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor Update
        # [NEW] Reinforce + Entropy + Moments Normalization
        
        baseline = from_twohot(v_pred_logits.detach())
        
        # Update Moments
        offset, invscale = self.moments(lambda_values.detach())
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantage = (normed_lambda_values - normed_baseline).detach()
        
        # Entropy
        act_dist_imag = D.Categorical(logits=torch.stack(imag_acts))
        entropy = act_dist_imag.entropy()
        
        actor_loss = -(advantage * imag_log_probs + 3e-4 * entropy).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        self.update_ema(self.target_critic, self.critic, 0.02)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

