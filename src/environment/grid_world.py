
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


from collections import namedtuple

# Define Resource: Position, Property Vector
Resource = namedtuple('Resource', ['pos', 'property', 'name'])

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
    
    def __init__(self, height, width, start, resource_pos, with_satiation, max_steps,
                 prob_switch_to_danger, min_danger_duration, damage_amount,
                 prob_switch_to_food, min_food_duration, relocate_resource, relocation_steps,
                 vector_size, food_property, danger_property):
        """
        Initializes the GridWorld foraging environment.

        Args:
            height (int): Number of rows.
            width (int): Number of columns.
            start (tuple): Start position (row, col).
            resource_pos (tuple): Resource position (row, col).
            with_satiation (bool): Whether to include satiation/homeostasis.
            max_steps (int): Maximum steps allowed per episode.
            prob_switch_to_danger (float): Probability of switching to danger AFTER min_food_duration.
            min_danger_duration (int): Minimum steps danger persists.
            damage_amount (int): Amount of damage taken in danger state.
            relocate_resource (bool): Whether to randomly relocate resource.
            relocation_steps (int): Relocate every N steps.
        """
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.resource_pos = tuple(resource_pos)
        self.agent_pos = self.start
        self.with_satiation = with_satiation
        self.max_steps = max_steps
        self.current_step = 0
        
        # Resource Property Vectors
        self.vector_size = vector_size
        
        # Properties must be provided explicitly from config
        self.food_property = np.array(food_property, dtype=np.float32)
        self.danger_property = np.array(danger_property, dtype=np.float32)
        
        # Resource State Machine (Food <-> Danger)
        # Requirement: At least one is active.
        self.prob_switch_to_danger = prob_switch_to_danger
        self.min_danger_duration = min_danger_duration
        self.damage_amount = damage_amount
        self.prob_switch_to_food = prob_switch_to_food
        self.min_food_duration = min_food_duration
        
        self.resource_state = 'food' # 'food' or 'danger'
        self.resource_timer = self.min_food_duration
        
        # Relocation
        self.relocate_resource = relocate_resource
        self.relocation_steps = relocation_steps
        self.relocation_timer = relocation_steps
        
    @property
    def is_danger(self):
        return self.resource_state == 'danger'
        

    @property
    def is_food_active(self):
        return self.resource_state == 'food'
        
    def get_active_resources(self):
        """
        Returns a list of active Resource objects in the environment.
        Used by the ResourceSensor.
        """
        resources = []
        if self.is_food_active:
            resources.append(Resource(self.resource_pos, self.food_property, 'Food'))
        elif self.is_danger:
            resources.append(Resource(self.resource_pos, self.danger_property, 'Danger'))
        return resources

    def reset(self):
        """
        Resets the agent to a random position, avoiding the food location.
        
        Returns:
            tuple: The initial state (row, col).
        """
        self.current_step = 0
        
        # Reset Resource State
        self.resource_state = 'food'
        self.resource_state = 'food'
        self.resource_timer = self.min_food_duration
        
        while True:
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            if (row, col) != self.resource_pos and (row, col) != self.start:
                self.agent_pos = (row, col)
                break
        
        if self.relocate_resource:
             # Randomize resource position
             while True:
                rr = np.random.randint(0, self.height)
                rc = np.random.randint(0, self.width)
                # Ensure it doesn't spawn on agent (which was just randomized)
                if (rr, rc) != self.agent_pos:
                    self.resource_pos = (rr, rc)
                    break
             
        self.relocation_timer = self.relocation_steps
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
        
        # --- Resource State Update ---
        if self.resource_timer > 0:
            # Locked State (Minimum Duration)
            self.resource_timer -= 1
        else:
            # Probabilistic Switch
            if self.resource_state == 'danger':
                 if np.random.random() < self.prob_switch_to_food:
                     self.resource_state = 'food'
                     self.resource_timer = self.min_food_duration
            elif self.resource_state == 'food':
                 if np.random.random() < self.prob_switch_to_danger:
                     self.resource_state = 'danger'
                     self.resource_timer = self.min_danger_duration



        # --- Relocation Update ---
        if self.relocate_resource:
            self.relocation_timer -= 1
            if self.relocation_timer <= 0:
                # Relocate Resource
                while True:
                    rr = np.random.randint(0, self.height)
                    rc = np.random.randint(0, self.width)
                    if (rr, rc) != self.agent_pos:
                        self.resource_pos = (rr, rc)
                        break
                self.relocation_timer = self.relocation_steps

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
        
        if self.agent_pos == self.resource_pos:
            if self.is_danger:
                damage = self.damage_amount
            elif self.is_food_active:
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
        
        info = {'ate_food': ate_food, 'damage': damage, 'rested': (action == 4)}
        
        return self.agent_pos, reward, done, info

    def observation_spec(self):
        """
        Returns the observation specification.
        Returns:
            dict: Description of observation space.
        """
        spec = {
            'loc': {'shape': (2,), 'dtype': int},
            'satiation': {'shape': (1,), 'dtype': int},
            'health': {'shape': (1,), 'dtype': float}
        }
        return spec

    def action_spec(self):
        """
        Returns the action space specification.
        Returns:
            dict: {'type': 'discrete', 'n': 5}
        """
        return {'type': 'discrete', 'n': 5}

    def render(self):
        """
        Prints grid info.
        """
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    line += "A "
                elif (r, c) == self.resource_pos:
                    if self.is_danger:
                        line += "X " # Danger
                    elif self.is_food_active:
                        line += "F "
                    else:
                        line += ". "
                else:
                    line += ". "
            print(line)
        print(f"Step: {self.current_step}, Danger: {self.is_danger}, Food: {self.is_food_active}")
        print()

    def render_rgb_array(self, satiation=None, max_satiation=None, health=None, max_health=None, episode=None, step=None, sensory_data=None, action=None):
        """
        Renders the grid as an RGB image using Matplotlib with a professional Light Theme (Scientific/Apple Style).
        Supports visualizing sensory modules if data is provided.
        """
        # --- Theme Settings ---
        bg_color = '#FFFFFF'       # White Background
        grid_color = '#E9ECEF'     # Very light grey for grid
        agent_color = '#339AF0'    # Soft Blue
        food_color = '#40C057'     # Soft Green
        danger_color = '#FA5252'   # Soft Red
        text_color = '#343A40'     # Dark Grey text
        
        # Setup Figure
        # If sensory data exists, we need more space on the right
        fig_width = 10 if sensory_data is not None else 8
        fig = plt.figure(figsize=(fig_width, 6), dpi=100)
        fig.patch.set_facecolor(bg_color)
        
        # GridSpec Layout
        if sensory_data is not None:
             # Layout: [ Main Grid (2) ] [ Stats (Dynamic Height) ]
             #                           [ Sensors (Dynamic Height) ]
             # We allocate 40% height to Stats, 60% to Sensory (roughly)
             gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[0.4, 0.6])
             ax_grid = fig.add_subplot(gs[:, 0]) # Left column full height
             ax_stats = fig.add_subplot(gs[0, 1]) # Top Right
             ax_sensory = fig.add_subplot(gs[1, 1]) # Bottom Right
        else:
             gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
             ax_grid = fig.add_subplot(gs[0])
             ax_stats = fig.add_subplot(gs[1])
             ax_sensory = None
        
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
        fr, fc = self.resource_pos
        if self.is_danger:
            # Danger: Cross / Skull representation
            ax_grid.plot(fc, fr, marker='X', markersize=20, color=danger_color, markeredgecolor='white', markeredgewidth=2, path_effects=shadow_effect)
        elif self.is_food_active:
            # Food: Diamond representation
            ax_grid.plot(fc, fr, marker='D', markersize=18, color=food_color, markeredgecolor='white', markeredgewidth=2, path_effects=shadow_effect)
            
        # Draw Agent
        ar, ac = self.agent_pos
        # Agent: Circle with thick border
        agent_circle = plt.Circle((ac, ar), 0.35, color=agent_color, path_effects=shadow_effect)
        ax_grid.add_patch(agent_circle)
        # Inner dot for agent
        ax_grid.plot(ac, ar, marker='o', markersize=5, color='white')

        # --- 2. Draw Stats Dashboard (Dynamic Stack) ---
        ax_stats.set_facecolor(bg_color)
        ax_stats.axis('off')

        # Cursor for vertical layout (0.0 to 1.0)
        y_cursor = 1.0 
        
        # 2.1 Title
        y_cursor -= 0.15 
        ax_stats.text(0.5, y_cursor, "INTEROCEPTIVE AI", color=text_color, ha='center', fontsize=12, fontweight='bold', transform=ax_stats.transAxes)
        # Underscore line
        ax_stats.plot([0.2, 0.8], [y_cursor - 0.05, y_cursor - 0.05], color='#ADB5BD', transform=ax_stats.transAxes, linewidth=1)
        
        # 2.2 Episode Info
        y_cursor -= 0.15
        ep_str = f"EPISODE: {episode}" if episode is not None else "EP: --"
        step_str = f"STEP:    {step}" if step is not None else "STEP: --"
        ax_stats.text(0.1, y_cursor, ep_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        ax_stats.text(0.1, y_cursor - 0.08, step_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        
        y_cursor -= 0.1 # Gap
        
        # Helper for Bars
        def draw_bar(ax, y_pos, label, value, max_val, color):
            pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
            # Label
            ax.text(0.1, y_pos + 0.1, f"{label}: {value:.1f}/{max_val}", color=text_color, fontsize=8, fontweight='bold', transform=ax.transAxes)
            # Background Bar
            rect_bg = plt.Rectangle((0.1, y_pos), 0.8, 0.08, color='#F1F3F5', transform=ax.transAxes, ec='none')
            ax.add_patch(rect_bg)
            # Fill Bar
            rect_fill = plt.Rectangle((0.1, y_pos), 0.8 * pct, 0.08, color=color, transform=ax.transAxes, ec='none')
            ax.add_patch(rect_fill)
            
        # 2.3 Bars (Dynamic)
        if self.with_satiation and satiation is not None and max_satiation is not None:
             y_cursor -= 0.15
             draw_bar(ax_stats, y_cursor, "SATIATION", satiation, max_satiation, food_color)
        
        if health is not None and max_health is not None:
             y_cursor -= 0.15
             draw_bar(ax_stats, y_cursor, "HEALTH", health, max_health, danger_color)
             
        # 2.4 Status Badge & Action (Fixed at Bottom of Stats Panel usually better, but let's stack)
        y_cursor -= 0.20
        
        if self.is_danger:
            status_text = "DANGER"
            status_bg = danger_color
        else:
            status_text = "SAFE" 
            status_bg = food_color
        if not self.with_satiation:
             status_text = "FOOD" if not self.is_danger else "DANGER"
            
        ax_stats.text(0.8, y_cursor + 0.05, status_text, color='white', ha='center', va='center', fontsize=8, fontweight='bold', 
                      transform=ax_stats.transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=status_bg, edgecolor='none'))

        if action is not None:
             action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", 4: "STAY"}
             act_str = action_names.get(action, "UNKNOWN")
             # Draw action text centered
             ax_stats.text(0.4, y_cursor + 0.05, f"ACTION: {act_str}", color='#495057', ha='center', fontsize=11, weight='bold', transform=ax_stats.transAxes)

        # --- 3. Draw Sensory Modules (Dynamic Slotting) ---
        if ax_sensory and sensory_data:
            ax_sensory.set_facecolor(bg_color)
            ax_sensory.axis('off')
            
            # Header
            ax_sensory.text(0.5, 0.95, "SENSORY MODULES", color=text_color, ha='center', fontsize=10, fontweight='bold', transform=ax_sensory.transAxes)
            
            num_sensors = len(sensory_data)
            if num_sensors > 0:
                # Calculate Spacing
                # Available Y range approx: 0.1 to 0.85 (0.75 height)
                top_y = 0.85
                bottom_y = 0.10
                available_h = top_y - bottom_y
                
                # Each slot height
                slot_h = available_h / num_sensors
                
                for i, sensor in enumerate(sensory_data):
                    # Center of the slot
                    y_center = top_y - (i * slot_h) - (slot_h / 2)
                    # OR simple linspace:
                    # y_centers = np.linspace(0.8, 0.2, num_sensors)
                    
                    # Safer specific logic:
                    # i=0 (Top) -> y = top_y - slot_h/2
                    # i=N-1 (Bottom) -> ...
                    
                    # Sensor Label
                    name = sensor['name']
                    intensity = sensor.get('intensity', None)
                    vector = sensor.get('vector', None)
                    color = sensor['color'] # Hex color
                    
                    # Dynamic Label Position: Top of slot
                    label_y_offset = slot_h * 0.35
                    ax_sensory.text(0.1, y_center + label_y_offset, name.upper(), color=text_color, fontsize=9, fontweight='bold', transform=ax_sensory.transAxes)
                    
                    # --- Draw Content based on Type ---
                    if intensity is not None:
                        # --- INTENSITY (Orb + Bar) ---
                        # Value Text
                        ax_sensory.text(0.85, y_center + label_y_offset, f"{intensity:.2f}", color=color, fontsize=9, fontweight='bold', ha='right', transform=ax_sensory.transAxes)
                        
                        # Bar
                        bar_w = 0.6
                        bar_h = 0.05
                        pct = min(1.0, intensity / 2.0)
                        
                        rect_bg = plt.Rectangle((0.15, y_center), bar_w, bar_h, color='#F1F3F5', transform=ax_sensory.transAxes, ec='none')
                        ax_sensory.add_patch(rect_bg)
                        rect_fill = plt.Rectangle((0.15, y_center), bar_w * pct, bar_h, color=color, transform=ax_sensory.transAxes, ec='none', alpha=0.8)
                        ax_sensory.add_patch(rect_fill)
                        
                        # Orb (Small) - Positioned left of bar
                        orb_r = 0.04
                        cx, cy = 0.10, y_center + 0.025
                        fill_c = plt.Circle((cx, cy), orb_r, facecolor=color, alpha=min(1.0, intensity+0.2), transform=ax_sensory.transAxes)
                        ax_sensory.add_patch(fill_c)

                    elif sensor.get('value_text'):
                         # --- TEXT VALUE (Location, etc) ---
                         v_txt = sensor['value_text']
                         ax_sensory.text(0.5, y_center, v_txt, color=color, fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_sensory.transAxes)

                    elif vector is not None:
                         if sensor.get('type') == 'radial':
                             # --- RADIAL (Collision) ---
                             cx, cy = 0.5, y_center
                             # Scale radius based on slot height to avoid overlap
                             # If slot is small, ray must be small.
                             # Base size 0.15. Cap at slot_h * 0.35
                             max_ray_len = min(0.15, slot_h * 0.35)
                             
                             num_secs = len(vector)
                             import math
                             for si in range(num_secs):
                                  val = vector[si]
                                  # Angle: Sector 0 = Up (pi/2) - goes CW
                                  angle = math.pi/2 - (2 * math.pi * si / num_secs)
                                  
                                  dx = math.cos(angle) * max_ray_len
                                  dy = math.sin(angle) * max_ray_len
                                  
                                  # Base
                                  ax_sensory.plot([cx, cx+dx], [cy, cy+dy], color='#DEE2E6', transform=ax_sensory.transAxes, lw=1)
                                  
                                  # Active
                                  if val > 0:
                                       act_len = max_ray_len * val
                                       adx = math.cos(angle) * act_len
                                       ady = math.sin(angle) * act_len
                                       ax_sensory.plot([cx, cx+adx], [cy, cy+ady], color=color, transform=ax_sensory.transAxes, lw=2, alpha=0.9)
                                       # Tip
                                       tip = plt.Circle((cx+adx, cy+ady), 0.015, color=color, transform=ax_sensory.transAxes)
                                       ax_sensory.add_patch(tip)
                                       
                             # Center Dot
                             agent_dot = plt.Circle((cx, cy), 0.02, color='#868e96', transform=ax_sensory.transAxes)
                             ax_sensory.add_patch(agent_dot)

                         else:
                             # --- SPECTRUM (Olfactory) ---
                             # Horizontal Bars
                             num_ch = len(vector)
                             slot_w = 0.7
                             slot_x = 0.15
                             # Height constrained by slot
                             bar_h = min(0.1, slot_h * 0.6)
                             
                             bar_w = slot_w / num_ch
                             gap = bar_w * 0.2
                             act_w = bar_w - gap
                             
                             cols = ['#40C057', '#FA5252', '#339AF0', '#fab005', '#be4bdb']
                             
                             for vi, val in enumerate(vector):
                                  vx = slot_x + (vi * bar_w)
                                  pct = min(1.0, float(val) / 2.0)
                                  c = cols[vi % len(cols)]
                                  
                                  # BG
                                  ax_sensory.add_patch(plt.Rectangle((vx, y_center - bar_h/2), act_w, bar_h, color='#F1F3F5', transform=ax_sensory.transAxes))
                                  # Fill
                                  fill_h = bar_h * pct
                                  # Align bottom
                                  ax_sensory.add_patch(plt.Rectangle((vx, y_center - bar_h/2), act_w, fill_h, color=c, transform=ax_sensory.transAxes, alpha=0.9))

        # Footer
        ax_stats.text(0.5, 0.02, "GridWorld Env", color='#CED4DA', ha='center', fontsize=7, transform=fig.transFigure)

        # Draw
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        img = data[..., :3]
        plt.close(fig)
        return img
