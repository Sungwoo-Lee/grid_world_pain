
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
    def __init__(self, max_satiation=20, start_satiation=10, overeating_death=True, random_start_satiation=True, food_satiation_gain=10, use_homeostatic_reward=False, satiation_setpoint=15, death_penalty=100,
                 with_health=False, max_health=20, start_health=10, health_recovery=1, start_health_random=True):
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
            death_penalty (float): Penalty subtracted from reward upon death.
            with_health (bool): Whether to simulate health/pain.
            max_health (int): Maximum health level.
            start_health (int): Starting health level.
            health_recovery (int): Health recovery per step when no pain.
            start_health_random (bool): Whether to randomize start health.
        """
        self.max_satiation = max_satiation
        self.start_satiation = start_satiation
        self.overeating_death = overeating_death
        self.random_start_satiation = random_start_satiation
        self.food_satiation_gain = food_satiation_gain
        self.use_homeostatic_reward = use_homeostatic_reward
        self.satiation_setpoint = satiation_setpoint
        self.death_penalty = death_penalty
        self.satiation = start_satiation
        
        # Health / Pain Mechanism
        self.with_health = with_health
        self.max_health = max_health
        self.start_health = start_health
        self.health_recovery = health_recovery
        self.start_health_random = start_health_random
        self.health = start_health
        
    def reset(self):
        """
        Reset internal state.
        
        Returns:
            int or tuple: Initial satiation, or (satiation, health).
        """
        if self.random_start_satiation:
            min_start = self.max_satiation // 2
            self.satiation = np.random.randint(min_start, self.max_satiation + 1)
        else:
            self.satiation = self.start_satiation
            
        if self.with_health:
            if self.start_health_random:
                self.health = np.random.randint(self.max_health // 2, self.max_health + 1)
            else:
                self.health = self.start_health
            return (self.satiation, self.health)
            
        return self.satiation
    
    def step(self, info):
        """
        Update internal state based on external events.
        
        Args:
            info (dict): Information from the external environment (e.g., 'ate_food', 'damage').
            
        Returns:
            tuple: (state, reward, done)
        """
        # 0. Store previous state for reward calculation
        prev_satiation = self.satiation
        prev_health = self.health
        
        # --- Satiation Dynamics ---
        # 1. Metabolism: Burn 1 unit of energy per step
        self.satiation -= 1
        
        # 2. Ingestion: React to external possibilities (Eating)
        if info.get('ate_food', False):
            self.satiation += self.food_satiation_gain
            
            if not self.overeating_death:
                self.satiation = min(self.satiation, self.max_satiation)
            else:
                self.satiation = min(self.satiation, self.max_satiation + 1)
                
        # --- Health Dynamics ---
        if self.with_health:
            damage = info.get('damage', 0)
            if damage > 0:
                self.health -= damage
            else:
                self.health = min(self.health + self.health_recovery, self.max_health)
            
        # 3. Termination Checks (Death conditions)
        done = False
        death_type = None
        
        if self.satiation <= 0:
            done = True 
            death_type = "starvation"
        elif self.overeating_death and self.satiation >= self.max_satiation:
            done = True
            death_type = "overeating"
            
        if self.with_health and self.health <= 0:
            done = True
            death_type = "injury"
            
        # 4. Generate Reward signal
        reward = 0
        if self.use_homeostatic_reward:
            # Drive Reduction for Satiation
            prev_drive = abs(prev_satiation - self.satiation_setpoint)
            current_drive = abs(self.satiation - self.satiation_setpoint)
            reward += (prev_drive - current_drive)
            
            # Penalize death heavily
            if done:
                reward -= self.death_penalty
        else:
            # Traditional Reward: +1 for every step of SURVIVAL
            reward = 1 if not done else -self.death_penalty
        
        # Construct Return State
        if self.with_health:
            state = (self.satiation, self.health)
        else:
            state = self.satiation
            
        return state, reward, done
