
import numpy as np


class QLearningAgent:
    """
    A Q-Learning agent for the GridWorld environment.
    
    Algorithm: Tabular Q-Learning (Off-policy TD control).
    
    State Space (Composite):
    - Row (External)
    - Col (External)
    - Satiation (Internal)
    
    The agent learns a policy pi(state) -> action to maximize future rewards.
    Here, "State" is a combination of where it is and how it feels (hunger).
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, with_satiation=True):
        """
        Initialize the Q-Learning agent.

        Args:
            env: The environment (should provide height, width, max_satiation).
            alpha (float): Learning rate (how much to accept new info).
            gamma (float): Discount factor (importance of future vs immediate rewards).
            epsilon (float): Exploration rate (prob of random action).
            with_satiation (bool): Whether satiation is part of the state space.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.with_satiation = with_satiation
        
        # Check if environment supports health
        self.with_health = getattr(env, 'with_health', False)
        self.max_health = getattr(env, 'max_health', 20)
        
        # Q-Table Initialization:
        # If with_health: 4D Array [Height, Width, Satiation, Health, Actions]
        # If with_satiation: 3D Array [Height, Width, Satiation, Actions] (Legacy compatibility if needed, but we'll assume satiation is dim 3)
        # If not with_satiation: 3D Array [Height, Width, Actions]
        
        if self.with_health and self.with_satiation:
             # Height x Width x Satiation x Health x Actions
             self.q_table = np.zeros((self.env.height, self.env.width, self.env.max_satiation + 2, self.max_health + 2, 5))
        elif self.with_satiation:
            self.q_table = np.zeros((self.env.height, self.env.width, self.env.max_satiation + 2, 5))
        else:
            self.q_table = np.zeros((self.env.height, self.env.width, 5))

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        
        Strategy:
        - With probability epsilon: Explore (Ramdom Action).
        - With probability 1-epsilon: Exploit (Best known action).

        Args:
            state (tuple): Current state. (row, col, satiation, health) or (row, col, satiation) or (row, col).

        Returns:
            int: Chosen action (0-4).
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(0, 5)
        else:
            # Exploit: choose the action with the highest Q-value
            if self.with_health and self.with_satiation:
                row, col, sat, health = state
                sat = int(sat)
                health = int(health)
                return np.argmax(self.q_table[row, col, sat, health])
            elif self.with_satiation:
                row, col, sat = state
                sat = int(sat)
                return np.argmax(self.q_table[row, col, sat])
            else:
                row, col = state
                return np.argmax(self.q_table[row, col])

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule (Bellman Equation).
        
        Q(s,a) <-- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Meaning:
        - New Value = Old Value + Learning Rate * Temporal Difference Error
        - We nudge the current estimate towards the actual received reward plus estimated future value.

        Args:
            state (tuple): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): New state.
        """
        if self.with_health and self.with_satiation:
            row, col, sat, health = state
            next_row, next_col, next_sat, next_health = next_state
            
            sat = int(sat)
            health = int(health)
            next_sat = int(next_sat)
            next_health = int(next_health)
            
            best_next_action_value = np.max(self.q_table[next_row, next_col, next_sat, next_health])
            current_q_value = self.q_table[row, col, sat, health, action]
            
            new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_action_value - current_q_value)
            self.q_table[row, col, sat, health, action] = new_q_value
            
        elif self.with_satiation:
            row, col, sat = state
            next_row, next_col, next_sat = next_state
            
            sat = int(sat)
            next_sat = int(next_sat)
            
            # max_a' Q(s', a') - Best potential future value
            best_next_action_value = np.max(self.q_table[next_row, next_col, next_sat])
            
            # Current estimate
            current_q_value = self.q_table[row, col, sat, action]
            
            # Update rule
            new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_action_value - current_q_value)
            self.q_table[row, col, sat, action] = new_q_value
        else:
            row, col = state
            next_row, next_col = next_state
            
            best_next_action_value = np.max(self.q_table[next_row, next_col])
            current_q_value = self.q_table[row, col, action]
            
            new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_action_value - current_q_value)
            self.q_table[row, col, action] = new_q_value

    def train(self, episodes=1000):
        """
        Train the agent for a given number of episodes.

        Args:
            episodes (int): Number of episodes to train.
        """
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

    def save(self, filepath):
        """
        Save the Q-table to a file.
        
        Args:
            filepath (str): Path to save the file.
        """
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """
        Load the Q-table from a file.
        
        Args:
            filepath (str): Path to load the file from.
        """
        self.q_table = np.load(filepath)

