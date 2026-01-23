
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc_layers=[128, 128]):
        """
        Deep Q-Network with dynamic layer configuration.
        
        Args:
            input_dim (int): flattened state dimension.
            output_dim (int): number of actions.
            fc_layers (list): list of hidden layer dimensions.
        """
        super(DQN, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=None, gamma=None, buffer_size=None, batch_size=None, epsilon_start=None, epsilon_end=None, epsilon_decay=None, target_update_freq=None, fc_layers=None, device="auto"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Validation for required config parameters
        if lr is None: raise ValueError("DQNAgent: 'learning_rate' (lr) must be specified in config.")
        if gamma is None: raise ValueError("DQNAgent: 'gamma' must be specified in config.")
        if buffer_size is None: raise ValueError("DQNAgent: 'buffer_size' must be specified in config.")
        if batch_size is None: raise ValueError("DQNAgent: 'batch_size' must be specified in config.")
        if epsilon_start is None: raise ValueError("DQNAgent: 'epsilon_start' must be specified in config.")
        if epsilon_end is None: raise ValueError("DQNAgent: 'epsilon_end' must be specified in config.")
        if epsilon_decay is None: raise ValueError("DQNAgent: 'epsilon_decay' must be specified in config.")
        if target_update_freq is None: raise ValueError("DQNAgent: 'target_update_freq' must be specified in config.")
        if fc_layers is None: raise ValueError("DQNAgent: 'fc_layers' must be specified in config.")

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device Check
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
        
        # print(f"DQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim, fc_layers).to(self.device)
        self.target_net = DQN(state_dim, action_dim, fc_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        
    def choose_action(self, state, eval_mode=False):
        """
        Epsilon-greedy action selection.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Q(s, a)
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Q_target = r + gamma * max(Q_target(s', a'))
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_max = next_q_values.max(1)[0]
            expected_q_value = reward + self.gamma * next_q_max * (1 - done)
            
        loss = F.mse_loss(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        # Update Target Network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return {"loss": loss.item()}
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, path)
        
    def load(self, path, weights_only=False):
        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            
            if not weights_only:
                # Option to load optimizer and other state if needed (e.g. for resume)
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                if 'steps_done' in checkpoint:
                    self.steps_done = checkpoint['steps_done']
        else:
            # Legacy or weights-only load
            self.policy_net.load_state_dict(checkpoint)
