
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
    def __init__(self, max_satiation, start_satiation, overeating_death, random_start_satiation, food_satiation_gain, use_homeostatic_reward, satiation_setpoint, death_penalty,
                 with_health, max_health, start_health, health_recovery, start_health_random,
                 injury_smoothing_duration=3, with_satiation=True):
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
            max_health (int): Maximum injury level (critically injured/dead).
            start_health (int): Starting injury level (0 = healthy).
            health_recovery (int): Injury reduction per step when resting.
            start_health_random (bool): Whether to randomize start injury.
            injury_smoothing_duration (int): Duration to spread damage over.
            with_satiation (bool): Whether to simulate satiation/hunger.
        """
        self.max_satiation = max_satiation
        self.start_satiation = start_satiation
        self.overeating_death = overeating_death
        self.random_start_satiation = random_start_satiation
        self.food_satiation_gain = food_satiation_gain
        self.use_homeostatic_reward = use_homeostatic_reward
        self.satiation_setpoint = satiation_setpoint
        self.death_penalty = death_penalty
        self.with_satiation = with_satiation
        self.satiation = start_satiation
        
        # Injury / Interoceptive Nociception Mechanism
        # Note: 'injury' replaces 'health'. Logic is inverted: 0 is good, max is bad.
        self.with_health = with_health
        self.max_injury = max_health      # renamed conceptually
        self.start_injury = 0             # healthy start by default
        self.injury_recovery = health_recovery
        self.start_injury_random = start_health_random
        self.injury_smoothing_duration = injury_smoothing_duration
        
        self.injury_level = self.start_injury
        self.injury_increments_buffer = [] # Buffer for smoothed damage increments
        
    def reset(self):
        """
        Reset internal state.
        """
        if self.random_start_satiation:
            min_start = self.max_satiation // 2
            self.satiation = np.random.randint(min_start, self.max_satiation + 1)
        else:
            self.satiation = self.start_satiation
            
        if self.with_health:
            if self.start_injury_random:
                self.injury_level = np.random.randint(0, self.max_injury // 2)
            else:
                self.injury_level = self.start_injury
            self.injury_increments_buffer = [] 
            return (self.satiation, self.injury_level)
            
        return self.satiation
    
    def step(self, info):
        """
        Update internal state based on external events.
        
        Args:
            info (dict): Information from the external environment.
            
        Returns:
            tuple: (state, reward, done)
        """
        # 0. Store previous state for reward calculation
        prev_satiation = self.satiation
        prev_injury = self.injury_level
        
        # --- Satiation Dynamics ---
        if self.with_satiation:
            self.satiation -= 1
            if info.get('ate_food', False):
                self.satiation += self.food_satiation_gain
                if not self.overeating_death:
                    self.satiation = min(self.satiation, self.max_satiation)
                else:
                    self.satiation = min(self.satiation, self.max_satiation + 1)
        else:
            # Maintain fixed satiation if disabled (so starvation death doesn't trigger)
            self.satiation = self.start_satiation
                
        # --- Injury Dynamics (Interoceptive Nociception) ---
        if self.with_health:
            damage = info.get('damage', 0)
            if damage > 0:
                # Spread damage over future steps
                inc = damage / self.injury_smoothing_duration
                # Add to buffer
                for i in range(self.injury_smoothing_duration):
                    if len(self.injury_increments_buffer) <= i:
                        self.injury_increments_buffer.append(inc)
                    else:
                        self.injury_increments_buffer[i] += inc
            
            # Apply next increment if buffer has content
            if self.injury_increments_buffer:
                current_inc = self.injury_increments_buffer.pop(0)
                self.injury_level += current_inc
            else:
                # Recover ONLY if rested and no pending injury increments
                if info.get('rested', False):
                    self.injury_level = max(self.injury_level - self.injury_recovery, 0)
            
            # Bound injury level
            self.injury_level = max(0, min(self.injury_level, self.max_injury))
            
        # 3. Termination Checks (Death conditions)
        done = False
        death_type = None
        
        if self.with_satiation:
            if self.satiation <= 0:
                done = True 
                death_type = "starvation"
            elif self.overeating_death and self.satiation >= self.max_satiation:
                done = True
                death_type = "overeating"
            
        if self.with_health and self.injury_level >= self.max_injury:
            done = True
            death_type = "injury"
        elif not self.with_health and info.get('damage', 0) > 0:
            # Instant death logic for levels without health system (e.g. L03)
            done = True
            death_type = "instant_damage"
            
        # 4. Generate Reward signal
        reward = 0
        if self.use_homeostatic_reward:
            # Goal state: (satiation_setpoint, 0 injury)
            if self.with_health:
                target_vec = np.array([self.satiation_setpoint, 0.0])
                prev_vec = np.array([prev_satiation, prev_injury])
                curr_vec = np.array([self.satiation, self.injury_level])
            else:
                target_vec = np.array([self.satiation_setpoint])
                prev_vec = np.array([prev_satiation])
                curr_vec = np.array([self.satiation])
            
            # Reduction in drive
            prev_drive = np.linalg.norm(prev_vec - target_vec)
            curr_drive = np.linalg.norm(curr_vec - target_vec)
            reward += (prev_drive - curr_drive)
            
            # Penalize death heavily
            if done:
                reward -= self.death_penalty
        else:
            # Traditional Reward
            reward = 1 if not done else -self.death_penalty
        
        # Construct Return State
        if self.with_health:
            state = (self.satiation, self.injury_level)
        else:
            state = self.satiation
            
        return state, reward, done
