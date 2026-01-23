
import os
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
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_fc_layers=[64, 64], critic_fc_layers=[64, 64]):
        super(ActorCritic, self).__init__()
        
        # Actor
        layers = []
        in_dim = state_dim
        for hidden_dim in actor_fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*layers)
        
        # Critic
        layers = []
        in_dim = state_dim
        for hidden_dim in critic_fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.actor(state)
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.item(), action_logprob.item(), state_val.item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=None, lr_critic=None, gamma=None, K_epochs=None, eps_clip=None, update_timestep=None, entropy_coef=None, actor_fc_layers=None, critic_fc_layers=None, device="auto"):
        # Validation for required config parameters
        if lr_actor is None: raise ValueError("PPOAgent: 'lr_actor' must be specified in config.")
        if lr_critic is None: raise ValueError("PPOAgent: 'lr_critic' must be specified in config.")
        if gamma is None: raise ValueError("PPOAgent: 'gamma' must be specified in config.")
        if K_epochs is None: raise ValueError("PPOAgent: 'K_epochs' must be specified in config.")
        if eps_clip is None: raise ValueError("PPOAgent: 'eps_clip' must be specified in config.")
        if update_timestep is None: raise ValueError("PPOAgent: 'update_timestep' must be specified in config.")
        if entropy_coef is None: raise ValueError("PPOAgent: 'entropy_coef' must be specified in config.")
        if actor_fc_layers is None: raise ValueError("PPOAgent: 'actor_fc_layers' must be specified in config.")
        if critic_fc_layers is None: raise ValueError("PPOAgent: 'critic_fc_layers' must be specified in config.")

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep
        self.entropy_coef = entropy_coef
        
        self.buffer = RolloutBuffer()
        
        # Device Check
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
             
        # print(f"PPO Agent using device: {self.device}")
        
        self.policy = ActorCritic(state_dim, action_dim, actor_fc_layers, critic_fc_layers).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim, actor_fc_layers, critic_fc_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.time_step = 0
        
        # Epsilon placeholder for compatibility with train.py logging
        self.epsilon = 0.0 

    def choose_action(self, state, eval_mode=False):
        # State preprocessing
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = torch.FloatTensor(np.array(state)).to(self.device)
            
        if eval_mode:
            with torch.no_grad():
                action_probs = self.policy.actor(state)
                return torch.argmax(action_probs).item()
        
        with torch.no_grad():
            action,log_prob, state_val = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(state_val)
        
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # PPO stores reward and done flag. State/Action/LogProb stored during choose_action
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        
        self.time_step += 1

    def update(self):
        # Update only if enough timesteps collected
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

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.tensor(self.buffer.actions, dtype=torch.float32)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.tensor(self.buffer.logprobs)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.tensor(self.buffer.state_values, dtype=torch.float32)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        self.time_step = 0

        return {"loss": loss.mean().item()}
        
    def save(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'time_step': self.time_step
        }
        # Note: PPO typically saves 'policy_old' weights for inference, 
        # but for resuming training, we need 'policy' (current weights) 
        # plus optimizer. The separation is subtle.
        # Standard PPO implementation often uses policy_old for sampling.
        # But policy and policy_old are synced after update.
        # Let's save policy state dict as 'model_state_dict'.
        torch.save(checkpoint, checkpoint_path)
   
    def load(self, checkpoint_path, weights_only=False):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
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
