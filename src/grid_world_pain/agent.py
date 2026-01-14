
import numpy as np
import random

class QLearningAgent:
    """
    A Q-Learning agent for the GridWorld environment.
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-Learning agent.

        Args:
            env: The GridWorld environment.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        # State space size: height * width
        # Action space size: 4 (Up, Right, Down, Left)
        self.q_table = np.zeros((self.env.height, self.env.width, 4))

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (tuple): Current state (row, col).

        Returns:
            int: Chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, 3)
        else:
            # Exploit: choose the action with the highest Q-value
            row, col = state
            return np.argmax(self.q_table[row, col])

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.

        Args:
            state (tuple): Previous state (row, col).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): New state (row, col).
        """
        row, col = state
        next_row, next_col = next_state
        
        # Bellman equation for Q-learning
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
        print(f"Q-table saved to {filepath}")

