import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.hidden_states = [] # (h, c) tuples
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.hidden_states[:]

class ActorCriticRNN(nn.Module):
    def __init__(self, state_dim, action_dim, fc_layers=[64], recurrent_layers=[64], actor_fc_layers=[64], critic_fc_layers=[64]):
        super(ActorCriticRNN, self).__init__()
        
        # Verify recurrent layers
        if len(recurrent_layers) > 0:
            self.hidden_dim = recurrent_layers[0]
            if not all(x == self.hidden_dim for x in recurrent_layers):
                raise ValueError(f"PyTorch LSTM requires uniform hidden size for stacked layers. Got: {recurrent_layers}")
            num_recurrent_layers = len(recurrent_layers)
        else:
             raise ValueError("recurrent_layers cannot be empty for RecurrentPPO")

        # Shared Feature Extractor
        layers = []
        in_dim = state_dim
        for hidden_dim in fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU()) # Shared features usually ReLU?
            in_dim = hidden_dim
        
        self.fc_net = nn.Sequential(*layers)
        
        # Shared LSTM
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_dim, num_layers=num_recurrent_layers, batch_first=True)
        self.num_recurrent_layers = num_recurrent_layers
        
        # Actor Head
        layers = []
        in_dim = self.hidden_dim
        for hidden_dim in actor_fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.actor_head = nn.Sequential(*layers)
        
        # Critic Head
        layers = []
        in_dim = self.hidden_dim
        for hidden_dim in critic_fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.critic_head = nn.Sequential(*layers)
        
    def forward(self, state, hidden=None):
        # state: (batch, seq, dim)
        x = self.fc_net(state)
        x, new_hidden = self.lstm(x, hidden)
        return self.actor_head(x)
        
    def act(self, state, hidden):
        # state: (1, input_dim) -> unsqueeze for seq_len=1: (1, 1, input_dim)
        x = state.unsqueeze(1)
        x = self.fc_net(x)
        
        # LSTM
        # out: (1, 1, hidden), hidden: (num_layers, 1, hidden)
        x, new_hidden = self.lstm(x, hidden)
        x = x[:, -1, :] # Take last step
        
        action_probs = self.actor_head(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic_head(x)
        
        return action.item(), action_logprob.item(), state_val.item(), new_hidden
    
    def evaluate(self, state, action, hidden):
        # state: (batch, seq, dim)
        # hidden: (1, batch, dim) for h and c
        
        x = self.fc_net(state)
        
        # LSTM
        lstm_out, _ = self.lstm(x, hidden)
        
        # Flatten for heads (batch*seq, dim)
        lstm_out_flat = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        action_probs = self.actor_head(lstm_out_flat)
        state_values = self.critic_head(lstm_out_flat)
        
        dist = Categorical(action_probs)
        
        # action is flat (batch*seq)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy

class RecurrentPPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=None, lr_critic=None, gamma=None, K_epochs=None, 
                 eps_clip=None, update_timestep=None, sequence_length=None, entropy_coef=None, 
                 fc_layers=None, recurrent_layers=None, actor_fc_layers=None, critic_fc_layers=None, device="auto"):
        # Validation
        if lr_actor is None: raise ValueError("RecurrentPPOAgent: 'lr_actor' must be specified in config.")
        if lr_critic is None: raise ValueError("RecurrentPPOAgent: 'lr_critic' must be specified in config.")
        if gamma is None: raise ValueError("RecurrentPPOAgent: 'gamma' must be specified in config.")
        if K_epochs is None: raise ValueError("RecurrentPPOAgent: 'K_epochs' must be specified in config.")
        if eps_clip is None: raise ValueError("RecurrentPPOAgent: 'eps_clip' must be specified in config.")
        if update_timestep is None: raise ValueError("RecurrentPPOAgent: 'update_timestep' must be specified in config.")
        if sequence_length is None: raise ValueError("RecurrentPPOAgent: 'sequence_length' must be specified in config.")
        if entropy_coef is None: raise ValueError("RecurrentPPOAgent: 'entropy_coef' must be specified in config.")
        if fc_layers is None: raise ValueError("RecurrentPPOAgent: 'fc_layers' (shared) must be specified in config.")
        if recurrent_layers is None: raise ValueError("RecurrentPPOAgent: 'recurrent_layers' must be specified in config.")
        if actor_fc_layers is None: raise ValueError("RecurrentPPOAgent: 'actor_fc_layers' must be specified in config.")
        if critic_fc_layers is None: raise ValueError("RecurrentPPOAgent: 'critic_fc_layers' must be specified in config.")

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.entropy_coef = entropy_coef
        
        self.buffer = RolloutBuffer()
        
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
             
        # print(f"Recurrent PPO Agent using device: {self.device}")
        
        self.policy = ActorCriticRNN(state_dim, action_dim, fc_layers, recurrent_layers, actor_fc_layers, critic_fc_layers).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic},
            {'params': self.policy.fc_net.parameters(), 'lr': lr_actor}, # Shared params
            {'params': self.policy.lstm.parameters(), 'lr': lr_actor}
        ])
        
        self.policy_old = ActorCriticRNN(state_dim, action_dim, fc_layers, recurrent_layers, actor_fc_layers, critic_fc_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.time_step = 0
        self.hidden_state = None
        self.epsilon = 0.0 # for logging compatibility
        
        self.reset_hidden()
        
    def reset_hidden(self):
        # Hidden state: (h, c)
        # shape: (num_layers, batch_size, hidden_dim)
        num_layers = self.policy.num_recurrent_layers
        hidden_dim = self.policy.hidden_dim
        
        self.hidden_state = (torch.zeros(num_layers, 1, hidden_dim).to(self.device),
                             torch.zeros(num_layers, 1, hidden_dim).to(self.device))
                             
    def choose_action(self, state, eval_mode=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = torch.FloatTensor(np.array(state)).to(self.device)
            
        # Add batch dim: (state_dim) -> (1, state_dim)
        state = state.view(1, -1) 
            
        if eval_mode:
            with torch.no_grad():
                # For eval, just run forward, assuming hidden state is managed externally (reset at start)
                # But notice act return new_hidden
                action, _, _, self.hidden_state = self.policy.act(state, self.hidden_state)
                return action
        
        with torch.no_grad():
            action, log_prob, state_val, next_hidden = self.policy_old.act(state, self.hidden_state)
            
        # Store transition data
        self.buffer.states.append(state) # Storing tensor (1, dim)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(state_val)
        self.buffer.hidden_states.append(self.hidden_state) # Store PRE-UPDATE hidden state
        
        self.hidden_state = next_hidden
        
        return action
        
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.time_step += 1
        
        if done:
            self.reset_hidden()
            
    def update(self):
        if self.time_step < self.update_timestep:
            return
            
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Prepare Data
        # Flatten lists
        # states: List of (1, dim) -> stack to (T, 1, dim) -> squeeze to (T, dim)
        full_states = torch.cat(self.buffer.states, dim=0).detach().to(self.device)
        full_actions = torch.tensor(self.buffer.actions, dtype=torch.float32).detach().to(self.device)
        full_logprobs = torch.tensor(self.buffer.logprobs).detach().to(self.device)
        full_rewards = rewards
        
        # Hidden states: List of tuples (h, c).
        # We need to extract them logic.
        # But wait, we process in sequences.
        # Ideally, we reject data that doesn't fit in perfect sequences or we mask.
        # Simple strategy: Truncate to multiple of sequence_length
        
        T = len(full_states)
        num_sequences = T // self.sequence_length
        trunc_len = num_sequences * self.sequence_length
        
        if num_sequences == 0:
            return # Not enough data for one sequence
        
        # Truncate
        batch_states = full_states[:trunc_len] # (TotalSteps, Dim)
        batch_actions = full_actions[:trunc_len]
        batch_logprobs = full_logprobs[:trunc_len]
        batch_rewards = full_rewards[:trunc_len]
        
        # Reshape to (Batch, Seq, Dim)
        # Batch = NumSequences
        batch_states = batch_states.view(num_sequences, self.sequence_length, -1)
        batch_actions = batch_actions.view(num_sequences, self.sequence_length)
        batch_logprobs = batch_logprobs.view(num_sequences, self.sequence_length)
        # batch_rewards -> needed for advantage calc per step, or return per step?
        # advantages = rewards - values
        # we have full rewards.
        
        # Get hidden states for the START of each sequence
        # indices: 0, 8, 16...
        start_hidden_indices = [i * self.sequence_length for i in range(num_sequences)]
        
        # Extract h and c
        # self.buffer.hidden_states is list of (h, c) where h is (1, 1, 64)
        
        batch_h = []
        batch_c = []
        for i in start_hidden_indices:
            h, c = self.buffer.hidden_states[i]
            batch_h.append(h)
            batch_c.append(c)
            
        # Stack hidden: (NumSequences, 1, 64) -> Permute for LSTM (1, NumSequences, 64)
        batch_h = torch.cat(batch_h, dim=1).detach().to(self.device)
        batch_c = torch.cat(batch_c, dim=1).detach().to(self.device)
        batch_hidden = (batch_h, batch_c)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate using Shared LSTM on batches
            # evaluate returns flat logprobs
            logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions.view(-1), batch_hidden)
            
            # Reshape rewards to match
            # rewards was flat (TotalSteps in trunc)
            # state_values is flat (TotalSteps in trunc)
            
            flat_rewards = batch_rewards
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - batch_logprobs.view(-1).detach())
            
            advantages = flat_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, flat_rewards) - self.entropy_coef * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.time_step = 0

        return {"loss": loss.mean().item()}
        
    def save(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'time_step': self.time_step
        }
        torch.save(checkpoint, checkpoint_path)
   
    def load(self, checkpoint_path, weights_only=False):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])
            if not weights_only:
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'time_step' in checkpoint:
                    self.time_step = checkpoint['time_step']
        else:
            self.policy_old.load_state_dict(checkpoint)
            self.policy.load_state_dict(checkpoint)
