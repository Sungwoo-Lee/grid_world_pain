
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """
    A simple 2D GridWorld environment for Reinforcement Learning.
    
    This class represents the "External Environment" where the agent moves.
    It handles:
    1. Grid Navigation: Movement logic (Up, Right, Down, Left).
    2. Object Placement: Food and Agent positions.
    3. Rendering: visual representation of the world.

    Coordinate System: (row, col)
    - (0, 0) is the top-left corner.
    - Rows increase downward.
    - Columns increase to the right.
    - (height-1, width-1) is the bottom-right corner.
    
    Actions (Discrete):
    - 0: Up (Decreases row index)
    - 1: Right (Increases col index)
    - 2: Down (Increases row index)
    - 3: Left (Decreases col index)
    
    Rewards:
    - This external environment returns a reward of 0.
    - In this "Interoceptive AI" paradigm, strict rewards come from the *Body* (internal state),
      not the external world. The environment only provides signals (like 'ate_food').
    """
    
    def __init__(self, height=5, width=5, start=(0, 0), food_pos=(4, 4), with_satiation=True, max_steps=100,
                 danger_prob=0.1, danger_duration=5, damage_amount=5):
        """
        Initializes the GridWorld foraging environment.

        Args:
            height (int): Number of rows.
            width (int): Number of columns.
            start (tuple): Start position (row, col).
            food_pos (tuple): Food position (row, col).
            with_satiation (bool): Whether to include satiation/homeostasis.
            max_steps (int): Maximum steps allowed per episode.
            danger_prob (float): Probability of food turning into danger.
            danger_duration (int): How long danger persists.
            damage_amount (int): Amount of damage taken in danger state.
        """
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.food_pos = tuple(food_pos)
        self.agent_pos = self.start
        self.with_satiation = with_satiation
        self.max_steps = max_steps
        self.current_step = 0
        
        # Pain / Danger Mechanism
        self.danger_prob = danger_prob
        self.danger_duration = danger_duration
        self.damage_amount = damage_amount
        self.is_danger = False
        self.danger_timer = 0
        
    def reset(self):
        """
        Resets the agent to a random position, avoiding the food location.
        
        Returns:
            tuple: The initial state (row, col).
        """
        self.current_step = 0
        self.is_danger = False
        self.danger_timer = 0
        
        while True:
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            if (row, col) != self.food_pos:
                self.agent_pos = (row, col)
                break
        return self.agent_pos
    
    def step(self, action):
        """
        Moves the agent.
        
        Args:
            action (int): The action to take (0=Up, 1=Right, 2=Down, 3=Left, 4=Stay).
            
        Returns:
            tuple: A tuple containing:
                - next_state (tuple): (row, col).
                - reward (int): 0 (External environment provides no reward).
                - done (bool): True if goal reached or max_steps exceeded.
                - info (dict): {'ate_food': bool, 'damage': int}
        """
        self.current_step += 1
        
        # Update Danger State
        if self.is_danger:
            self.danger_timer -= 1
            if self.danger_timer <= 0:
                self.is_danger = False
        else:
            # Chance to become danger if not already
            if np.random.random() < self.danger_prob:
                self.is_danger = True
                self.danger_timer = self.danger_duration

        row, col = self.agent_pos
        
        # Movement logic
        if action == 0:   # Up
            row = max(0, row - 1)
        elif action == 1: # Right
            col = min(self.width - 1, col + 1)
        elif action == 2: # Down
            row = min(self.height - 1, row + 1)
        elif action == 3: # Left
            col = max(0, col - 1)
        elif action == 4: # Stay
            pass
            
        self.agent_pos = (row, col)
        
        # Check interactions
        ate_food = False
        damage = 0
        
        if self.agent_pos == self.food_pos:
            if self.is_danger:
                damage = self.damage_amount
            else:
                ate_food = True
            
        # Reward/Done: Behavior depends on whether satiation is enabled.
        reward = 0
        done = False
        if not self.with_satiation:
            if ate_food:
                reward = 10 # Conventional goal reward
                done = True # REACHED GOAL
        
        if self.current_step >= self.max_steps:
            done = True
        
        info = {'ate_food': ate_food, 'damage': damage}
        
        return self.agent_pos, reward, done, info

    def render(self):
        """
        Prints grid info.
        """
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    line += "A "
                elif (r, c) == self.food_pos:
                    if self.is_danger:
                        line += "X " # Danger
                    else:
                        line += "F "
                else:
                    line += ". "
            print(line)
        print(f"Step: {self.current_step}, Danger: {self.is_danger}")
        print()

    def render_rgb_array(self, satiation=None, max_satiation=None, health=None, max_health=None, episode=None, step=None):
        """
        Renders the grid as an RGB image using Matplotlib with enhanced visualization.
        
        Args:
            satiation (int, optional): Current satiation level to display.
            max_satiation (int, optional): Max satiation level for display.
            health (int, optional): Current health level.
            max_health (int, optional): Max health level.
            episode (int, optional): Current episode number.
            step (int, optional): Current step number.

        Returns:
            numpy.ndarray: An RGB image array of shape (height, width, 3).
        """
        # Set figsize and dpi to ensure output dimensions are multiples of 16 (640x720) to avoid imageio warning
        fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=100)
        
        # Grid area
        ax.set_ylim(0, self.height)
        ax.set_xlim(0, self.width)
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(True, color='black')
        ax.set_aspect('equal')
        
        # Title Info (Episode / Step)
        title_text = ""
        if episode is not None:
            title_text += f"Episode: {episode}  "
        if step is not None:
            title_text += f"Step: {step}"
        if title_text:
            ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
        
        # Invert y-axis to match array indexing (0,0 at top-left)
        ax.invert_yaxis()
        
        # Draw Agent (Red Circle)
        agent_circle = plt.Circle((self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5), 0.3, color='blue', label='Agent')
        ax.add_patch(agent_circle)
        
        # Draw Food / Danger
        if self.is_danger:
            target_color = 'red'
            target_label = 'Danger'
        else:
            target_color = 'green'
            target_label = 'Food' if self.with_satiation else 'Goal'
            
        food_rect = plt.Rectangle((self.food_pos[1], self.food_pos[0]), 1, 1, color=target_color, alpha=0.5, label=target_label)
        ax.add_patch(food_rect)
        
        # Remove axis ticks/labels for a cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # --- Status Bars ---
        # Satiation Bar
        if self.with_satiation and satiation is not None and max_satiation is not None:
            bar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03]) 
            pct = max(0, min(1, satiation / max_satiation))
            bar_color = (1 - pct, pct, 0) # R->G
            bar_ax.barh(0, pct, color=bar_color, height=0.5)
            bar_ax.set_xlim(0, 1)
            bar_ax.axis('off')
            bar_ax.set_title(f"Satiation: {satiation}", fontsize=10, loc='left')
            rect = plt.Rectangle((0, -0.25), 1, 0.5, fill=False, edgecolor='black')
            bar_ax.add_patch(rect)

        # Health Bar
        if health is not None and max_health is not None:
            health_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
            pct_h = max(0, min(1, health / max_health))
            # Cyan to Blue or distinct color
            health_color = (1-pct_h, 0, pct_h) # Red to Blue? Or just Cyan
            health_color = 'cyan'
            
            health_ax.barh(0, pct_h, color=health_color, height=0.5)
            health_ax.set_xlim(0, 1)
            health_ax.axis('off')
            health_ax.set_title(f"Health: {health}", fontsize=10, loc='left')
            rect_h = plt.Rectangle((0, -0.25), 1, 0.5, fill=False, edgecolor='black')
            health_ax.add_patch(rect_h)

        # Draw the canvas and convert to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        plt.close(fig) # Close the figure to prevent memory leaks
        return data[..., :3]  # Return RGB channels, omitting Alpha
