
import numpy as np

class InteroceptiveBody:
    """
    Simulates the internal physiological state of the agent (Satiation).
    
    Concept: Homeostatic Regulation
    - The agent has an internal variable `satiation` representing energy.
    - Metabolism: Satiation decreases over time (-1 per step).
    - Intake: Eating food increases satiation (+10).
    - Homeostasis: The goal is to keep satiation within a viable range (0 < satiation < max).
    
    This class acts as the "Internal Environment". Unlike standard RL where the external world
    defines the reward, here the *Body* defines the reward based on its needs.
    """
    def __init__(self, max_satiation=20, start_satiation=10):
        """
        Initialize the body.
        
        Args:
            max_satiation (int): Maximum satiation level (upper bound termination).
            start_satiation (int): Starting satiation level.
        """
        self.max_satiation = max_satiation
        self.start_satiation = start_satiation
        self.satiation = start_satiation
        
    def reset(self):
        """
        Reset internal state.
        
        Returns:
            int: Initial satiation.
        """
        # Randomize start satiation between middle and max to provide varied experiences
        # This helps the agent learn from different levels of need.
        min_start = self.max_satiation // 2
        self.satiation = np.random.randint(min_start, self.max_satiation + 1)
        return self.satiation
    
    def step(self, info):
        """
        Update internal state based on external events.
        
        Args:
            info (dict): Information from the external environment (e.g., 'ate_food').
            
        Returns:
            tuple: (satiation, reward, done)
        """
        # 1. Metabolism: Burn 1 unit of energy per step
        self.satiation -= 1
        
        # 2. Ingestion: React to external possibilities (Eating)
        if info.get('ate_food', False):
            self.satiation += 10
            # Clamp logic (allowing one step over max for termination check consistency)
            # We allow it to go slightly over to trigger the overeating condition below.
            self.satiation = min(self.satiation, self.max_satiation + 1)
            
        # 3. Termination Checks (Death conditions)
        done = False
        if self.satiation <= 0:
            done = True # Death by Starvation
        elif self.satiation >= self.max_satiation:
            done = True # Death by Overeating (Rupture/Obesity limit)
            
        # 4. Generate Reward signal
        # Reward = 1 for every step of SURVIVAL.
        # This implicitly encourages the agent to maintain homeostasis as long as possible.
        reward = 1 if not done else 0
        
        return self.satiation, reward, done
