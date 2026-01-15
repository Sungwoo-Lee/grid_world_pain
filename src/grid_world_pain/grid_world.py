
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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
        Renders the grid as an RGB image using Matplotlib with a professional Light Theme (Scientific/Apple Style).
        """
        # --- Theme Settings ---
        bg_color = '#FFFFFF'       # White Background
        grid_color = '#E9ECEF'     # Very light grey for grid
        agent_color = '#339AF0'    # Soft Blue
        food_color = '#40C057'     # Soft Green
        danger_color = '#FA5252'   # Soft Red
        text_color = '#343A40'     # Dark Grey text
        
        # Setup Figure with GridSpec for Dashboard Layout
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor(bg_color)
        
        # GridSpec: Main Grid (left) and Stats Panel (right)
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_grid = fig.add_subplot(gs[0])
        ax_stats = fig.add_subplot(gs[1])
        
        # --- 1. Draw Grid World ---
        ax_grid.set_facecolor(bg_color)
        ax_grid.set_xlim(-0.5, self.width - 0.5)
        ax_grid.set_ylim(-0.5, self.height - 0.5)
        ax_grid.invert_yaxis() # Top-left is (0,0)
        ax_grid.set_aspect('equal')
        ax_grid.axis('off') # Hide default axes
        
        # Use simple subtle shadow effect
        shadow_effect = [PathEffects.SimpleLineShadow(offset=(2, -2), shadow_color='grey', alpha=0.3), PathEffects.Normal()]
        
        # Draw Custom Grid Lines
        for x in range(self.width + 1):
            ax_grid.vlines(x - 0.5, -0.5, self.height - 0.5, colors=grid_color, linestyles='-', linewidth=1.5)
        for y in range(self.height + 1):
            ax_grid.hlines(y - 0.5, -0.5, self.width - 0.5, colors=grid_color, linestyles='-', linewidth=1.5)
            
        # Draw Food / Danger
        fr, fc = self.food_pos
        if self.is_danger:
            # Danger: Cross / Skull representation
            ax_grid.plot(fc, fr, marker='X', markersize=20, color=danger_color, markeredgecolor='white', markeredgewidth=2, path_effects=shadow_effect)
        else:
            # Food: Diamond representation
            ax_grid.plot(fc, fr, marker='D', markersize=18, color=food_color, markeredgecolor='white', markeredgewidth=2, path_effects=shadow_effect)
            
        # Draw Agent
        ar, ac = self.agent_pos
        # Agent: Circle with thick border
        agent_circle = plt.Circle((ac, ar), 0.35, color=agent_color, path_effects=shadow_effect)
        ax_grid.add_patch(agent_circle)
        # Inner dot for agent
        ax_grid.plot(ac, ar, marker='o', markersize=5, color='white')

        # --- 2. Draw Stats Dashboard ---
        ax_stats.set_facecolor(bg_color)
        ax_stats.axis('off')
        
        # Clean Title
        ax_stats.text(0.5, 0.90, "INTEROCEPTIVE AI", color=text_color, ha='center', fontsize=14, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.plot([0.2, 0.8], [0.88, 0.88], color='#ADB5BD', transform=ax_stats.transAxes, linewidth=1)
        
        # Episode / Step info (Monospace font for numbers)
        ep_str = f"EPISODE: {episode}" if episode is not None else "EPISODE: --"
        step_str = f"STEP:    {step}" if step is not None else "STEP:    --"
        ax_stats.text(0.1, 0.80, ep_str, color='#495057', fontsize=10, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        ax_stats.text(0.1, 0.76, step_str, color='#495057', fontsize=10, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        
        # Bars Helper (Flat Design)
        def draw_bar(y_pos, label, value, max_val, color):
            pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
            # Label
            ax_stats.text(0.1, y_pos + 0.05, f"{label}: {value:.1f}/{max_val}", color=text_color, fontsize=9, fontweight='bold', transform=ax_stats.transAxes)
            # Background Bar (Light Grey)
            rect_bg = plt.Rectangle((0.1, y_pos), 0.8, 0.03, color='#F1F3F5', transform=ax_stats.transAxes, ec='none')
            ax_stats.add_patch(rect_bg)
            # Fill Bar
            rect_fill = plt.Rectangle((0.1, y_pos), 0.8 * pct, 0.03, color=color, transform=ax_stats.transAxes, ec='none')
            ax_stats.add_patch(rect_fill)
            
        # Satiation Bar
        if self.with_satiation and satiation is not None and max_satiation is not None:
            draw_bar(0.60, "SATIATION", satiation, max_satiation, food_color)
            
        # Health Bar
        if health is not None and max_health is not None:
            draw_bar(0.50, "HEALTH", health, max_health, danger_color)
            
        # Status Text (Pill style background)
        status_y = 0.35
        if self.is_danger:
            status_text = "RESOURCE: DANGER"
            status_bg = danger_color
        else:
            status_text = "RESOURCE: FOOD"
            status_bg = food_color
            
        ax_stats.text(0.5, status_y, status_text, color='white', ha='center', va='center', fontsize=10, fontweight='bold', 
                      transform=ax_stats.transAxes,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor=status_bg, edgecolor='none'))

        # Footer
        ax_stats.text(0.5, 0.05, "GridWorld Environment", color='#CED4DA', ha='center', fontsize=8, transform=ax_stats.transAxes)

        # Convert to array
        fig.canvas.draw()
        
        # Modern Matplotlib (buffer_rgba)
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        img = data[..., :3]
        
        plt.close(fig)
        
        return img
