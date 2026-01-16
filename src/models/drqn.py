import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class RecurrentReplayBuffer:
    
    def __init__(self, capacity, burn_in_length=0):
        self.buffer = deque(maxlen=capacity)
        self.burn_in_length = burn_in_length
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, trace_length):
        """
        Samples a batch of sequential episodes.
        
        Args:
            batch_size (int): Number of sequences to sample.
            trace_length (int): Length of each sequence.
            
        Returns:
            sampled_traces: (batch_size, trace_length, state_dim)
            sampled_actions: (batch_size, trace_length)
            sampled_rewards: (batch_size, trace_length)
            sampled_next_states: (batch_size, trace_length, state_dim)
            sampled_dones: (batch_size, trace_length)
        """
        sampled_traces = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_dones = []
        
        buffer_len = len(self.buffer)
        
        total_len = trace_length + self.burn_in_length
        count = 0
        while count < batch_size:
            idx = random.randint(0, buffer_len - total_len - 1)
            
            # Extract trace
            trace = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            is_valid = True
            for i in range(total_len):
                s, a, r, ns, d = self.buffer[idx + i]
                trace.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)
                
                # If we hit a done before the last step, it means the sequence crosses episodes
                if d and i < total_len - 1:
                    is_valid = False
                    break
            
            if is_valid:
                sampled_traces.append(np.array(trace))
                sampled_actions.append(np.array(actions))
                sampled_rewards.append(np.array(rewards))
                sampled_next_states.append(np.array(next_states))
                sampled_dones.append(np.array(dones))
                count += 1
                
        return (np.array(sampled_traces), np.array(sampled_actions), np.array(sampled_rewards), 
                np.array(sampled_next_states), np.array(sampled_dones))
    
    def __len__(self):
        return len(self.buffer)

class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Deep Recurrent Q-Network using LSTM.
        """
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Pre-process with FC layer
        batch_size, seq_len, _ = x.size()
        
        x = F.relu(self.fc1(x))
        
        # LSTM
        # out: (batch, seq, hidden), hidden: (num_layers, batch, hidden)
        out, new_hidden = self.lstm(x, hidden)
        
        # Output head - apply to all steps in sequence
        q_values = self.fc2(out)
        
        return q_values, new_hidden

class DRQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=32, 
                 trace_length=8, burn_in_length=0, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, device="auto"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.burn_in_length = burn_in_length
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device Check
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
        
        print(f"DRQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DRQN(state_dim, action_dim).to(self.device)
        self.target_net = DRQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = RecurrentReplayBuffer(buffer_size, burn_in_length)
        
        self.hidden_state = None
        self.reset_hidden()
        
    def reset_hidden(self):
        """Resets the hidden state for the beginning of a new episode."""
        self.hidden_state = None
        
    def choose_action(self, state, eval_mode=False):
        """
        Epsilon-greedy action selection with recurrent state.
        Args:
            state: Single state observation (flat numpy array)
        """
        if not eval_mode and random.random() < self.epsilon:
            # We still need to update hidden state even if taking random action?
            # Ideally yes, to keep track of history.
            action = random.randrange(self.action_dim)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(self.device)
                _, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)
                
            return action
        
        with torch.no_grad():
            # Add batch and seq dimensions: (1, 1, state_dim)
            state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(self.device)
            q_values, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)
            return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def update(self):
        if len(self.memory) < self.batch_size + self.trace_length + self.burn_in_length:
            return
        
        # Sample sequences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.trace_length)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Forward pass with Burn-in
        # Slice data
        burn_in_states = states[:, :self.burn_in_length, :]
        train_states = states[:, self.burn_in_length:, :]
        
        train_actions = actions[:, self.burn_in_length:]
        train_rewards = rewards[:, self.burn_in_length:]
        train_dones = dones[:, self.burn_in_length:]
        train_next_states = next_states[:, self.burn_in_length:, :]
        
        # Burn-in Phase (No Gradients)
        hidden = None
        if self.burn_in_length > 0:
             with torch.no_grad():
                  _, hidden = self.policy_net(burn_in_states)
        
        # Training Phase (With Gradients, using hidden from burn-in)
        q_values, _ = self.policy_net(train_states, hidden)
        
        # Get Q-value for taken actions
        q_value = q_values.gather(2, train_actions.unsqueeze(2)).squeeze(2)
        
        # Target Network for next states (Need to handle burn-in or just pass through?)
        # For simplicity, we can just run target net on the train sequence, initializing with zero/same hidden?
        # Ideally, target net also burns in.
        
        target_hidden = None
        if self.burn_in_length > 0:
             with torch.no_grad():
                  # Process full next state sequence or just burn-in next states?
                  # We need hidden state for target calculation at step t.
                  # Logic: Target network should process:
                  # burn-in-next -> hidden -> train-next -> q-values
                  
                  burn_in_next_states = next_states[:, :self.burn_in_length, :]
                  _, target_hidden = self.target_net(burn_in_next_states)
        
        with torch.no_grad():
            next_q_values, _ = self.target_net(train_next_states, target_hidden)
            max_next_q_values = next_q_values.max(2)[0]
            expected_q_values = train_rewards + self.gamma * max_next_q_values * (1 - train_dones)
            
        loss = F.mse_loss(q_value, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay (managed externally usually, but if called here)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
