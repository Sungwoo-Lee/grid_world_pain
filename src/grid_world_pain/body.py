
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
    def __init__(self, max_satiation=20, start_satiation=10, overeating_death=True, random_start_satiation=True, food_satiation_gain=10, use_homeostatic_reward=False, satiation_setpoint=15):
        """
        Initialize the body.
        
        Args:
            max_satiation (int): Maximum satiation level (upper bound termination).
            start_satiation (int): Starting satiation level.
            overeating_death (bool): Whether overeating (>= max_satiation) causes death.
            random_start_satiation (bool): Whether to randomize start satiation on reset.
            food_satiation_gain (int): Satiation increase from eating food.
            use_homeostatic_reward (bool): Whether to use drive reduction reward.
            satiation_setpoint (int): Ideal satiation level for homeostasis.
        """
        self.max_satiation = max_satiation
        self.start_satiation = start_satiation
        self.overeating_death = overeating_death
        self.random_start_satiation = random_start_satiation
        self.food_satiation_gain = food_satiation_gain
        self.use_homeostatic_reward = use_homeostatic_reward
        self.satiation_setpoint = satiation_setpoint
        self.satiation = start_satiation
        
    def reset(self):
        """
        Reset internal state.
        
        Returns:
            int: Initial satiation.
        """
        if self.random_start_satiation:
            # Randomize start satiation between middle and max to provide varied experiences
            # This helps the agent learn from different levels of need.
            min_start = self.max_satiation // 2
            self.satiation = np.random.randint(min_start, self.max_satiation + 1)
        else:
            self.satiation = self.start_satiation
        return self.satiation
    
    def step(self, info):
        """
        Update internal state based on external events.
        
        Args:
            info (dict): Information from the external environment (e.g., 'ate_food').
            
        Returns:
            tuple: (satiation, reward, done)
        """
        # 0. Store previous state for reward calculation
        prev_satiation = self.satiation

        # 1. Metabolism: Burn 1 unit of energy per step
        self.satiation -= 1
        
        # 2. Ingestion: React to external possibilities (Eating)
        if info.get('ate_food', False):
            self.satiation += self.food_satiation_gain
            
            # If overeating death is off, we clamp to max_satiation.
            # If it's on, we allow it to go over to trigger the death condition below.
            if not self.overeating_death:
                self.satiation = min(self.satiation, self.max_satiation)
            else:
                self.satiation = min(self.satiation, self.max_satiation + 1)
            
        # 3. Termination Checks (Death conditions)
        done = False
        if self.satiation <= 0:
            done = True # Death by Starvation
        elif self.overeating_death and self.satiation >= self.max_satiation:
            done = True # Death by Overeating
            
        # 4. Generate Reward signal
        # Homeostatic Reward (Keramati & Gutkin): Drive Reduction
        # Drive D(H) = |H - H*|
        # Reward = D(H_prev) - D(H_current)
        if self.use_homeostatic_reward:
            prev_drive = abs(prev_satiation - self.satiation_setpoint)
            current_drive = abs(self.satiation - self.satiation_setpoint)
            reward = prev_drive - current_drive
            
            # Penalize death heavily in homeostatic mode to ensure survival is prioritized
            if done:
                reward -= 10
        else:
            # Traditional Reward: +1 for every step of SURVIVAL
            reward = 1 if not done else 0
        
        return self.satiation, reward, done
