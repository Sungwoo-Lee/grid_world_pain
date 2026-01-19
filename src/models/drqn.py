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
    def __init__(self, input_dim, output_dim, fc_layers=[128], recurrent_layers=[128]):
        """
        Deep Recurrent Q-Network using LSTM.
        
        Args:
            input_dim (int): flattened state dimension.
            output_dim (int): number of actions.
            fc_layers (list): list of dimensions for pre-processing FC layers.
            recurrent_layers (list): list of hidden dimensions for LSTM layers. 
                                     Note: PyTorch LSTM requires uniform hidden size for stacked layers.
                                     We will use recurrent_layers[0] as hidden_size and len() as num_layers,
                                     verifying all elements are equal.
        """
        super(DRQN, self).__init__()
        
        # Verify recurrent layers
        if len(recurrent_layers) > 0:
            hidden_size = recurrent_layers[0]
            if not all(x == hidden_size for x in recurrent_layers):
                raise ValueError(f"PyTorch LSTM requires uniform hidden size for stacked layers. Got: {recurrent_layers}")
            num_recurrent_layers = len(recurrent_layers)
        else:
             # Default or Error? DRQN needs recurrence.
             raise ValueError("recurrent_layers cannot be empty for DRQN")

        # Build FC Pre-processing
        layers = []
        in_dim = input_dim
        
        for hidden_dim in fc_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        self.fc_net = nn.Sequential(*layers)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_recurrent_layers, batch_first=True)
        
        # Output Head
        self.output_head = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Pre-process with FC layer (apply to each step)
        # FC Net expects (N, *, In). Dense layers work on arbitrary last dim usually?
        # Linear works on (..., In).
        
        x = self.fc_net(x)
        
        # LSTM
        # out: (batch, seq, hidden), hidden: (num_layers, batch, hidden)
        out, new_hidden = self.lstm(x, hidden)
        
        # Output head
        q_values = self.output_head(out)
        
        return q_values, new_hidden

class DRQNAgent:
    def __init__(self, state_dim, action_dim, lr=None, gamma=None, buffer_size=None, batch_size=None, 
                 trace_length=None, burn_in_length=None, epsilon_start=None, epsilon_end=None, epsilon_decay=None, target_update_freq=None, 
                 fc_layers=None, recurrent_layers=None, device="auto"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Validation for required config parameters
        if lr is None: raise ValueError("DRQNAgent: 'learning_rate' (lr) must be specified in config.")
        if gamma is None: raise ValueError("DRQNAgent: 'gamma' must be specified in config.")
        if buffer_size is None: raise ValueError("DRQNAgent: 'buffer_size' must be specified in config.")
        if batch_size is None: raise ValueError("DRQNAgent: 'batch_size' must be specified in config.")
        if trace_length is None: raise ValueError("DRQNAgent: 'trace_length' must be specified in config.")
        if burn_in_length is None: raise ValueError("DRQNAgent: 'burn_in_length' must be specified in config.")
        if epsilon_start is None: raise ValueError("DRQNAgent: 'epsilon_start' must be specified in config.")
        if epsilon_end is None: raise ValueError("DRQNAgent: 'epsilon_end' must be specified in config.")
        if epsilon_decay is None: raise ValueError("DRQNAgent: 'epsilon_decay' must be specified in config.")
        if target_update_freq is None: raise ValueError("DRQNAgent: 'target_update_freq' must be specified in config.")
        if fc_layers is None: raise ValueError("DRQNAgent: 'fc_layers' must be specified in config.")
        if recurrent_layers is None: raise ValueError("DRQNAgent: 'recurrent_layers' must be specified in config.")

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.burn_in_length = burn_in_length
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device Check
        if device == "auto" or device is None:
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
             self.device = torch.device(device)
        
        # print(f"DRQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DRQN(state_dim, action_dim, fc_layers, recurrent_layers).to(self.device)
        self.target_net = DRQN(state_dim, action_dim, fc_layers, recurrent_layers).to(self.device)
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
        
        # Epsilon Decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        # Update Target Network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
