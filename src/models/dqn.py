
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
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Deep Q-Network.
        
        Args:
            input_dim (int): flattened state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer size.
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, device="auto"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device Check
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
        
        print(f"DQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
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
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
