
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
        # CRITICAL: This must be consistent across all frames in a video. 
        # Since we control main.py, we rely on it passing sensory_data if enabled.
        fig_width = 10 if sensory_data is not None else 8
        fig = plt.figure(figsize=(fig_width, 6), dpi=100)
        fig.patch.set_facecolor(bg_color)
        
        # GridSpec Layout
        if sensory_data is not None:
            # Layout: [ Main Grid (2) ] [ Stats (1) ]
            #                           [ Sensors (1) ]
            gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1.5])
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

        # --- 2. Draw Stats Dashboard ---
        ax_stats.set_facecolor(bg_color)
        ax_stats.axis('off')
        
        # Clean Title
        ax_stats.text(0.5, 0.95, "INTEROCEPTIVE AI", color=text_color, ha='center', fontsize=12, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.plot([0.2, 0.8], [0.92, 0.92], color='#ADB5BD', transform=ax_stats.transAxes, linewidth=1)
        
        # Episode / Step info
        ep_str = f"EPISODE: {episode}" if episode is not None else "EP: --"
        step_str = f"STEP:    {step}" if step is not None else "STEP: --"
        ax_stats.text(0.1, 0.80, ep_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        ax_stats.text(0.1, 0.75, step_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        
        # Bars Helper (Flat Design)
        def draw_bar(y_pos, label, value, max_val, color):
            pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
            # Label
            ax_stats.text(0.1, y_pos + 0.09, f"{label}: {value:.1f}/{max_val}", color=text_color, fontsize=8, fontweight='bold', transform=ax_stats.transAxes)
            # Background Bar
            rect_bg = plt.Rectangle((0.1, y_pos), 0.8, 0.08, color='#F1F3F5', transform=ax_stats.transAxes, ec='none')
            ax_stats.add_patch(rect_bg)
            # Fill Bar
            rect_fill = plt.Rectangle((0.1, y_pos), 0.8 * pct, 0.08, color=color, transform=ax_stats.transAxes, ec='none')
            ax_stats.add_patch(rect_fill)
            
        # Satiation Bar
        if self.with_satiation and satiation is not None and max_satiation is not None:
            draw_bar(0.40, "SATIATION", satiation, max_satiation, food_color)
            
        # Health Bar
        if health is not None and max_health is not None:
            draw_bar(0.20, "HEALTH", health, max_health, danger_color)
            
        # Status Text
        if self.is_danger:
            status_text = "DANGER"
            status_bg = danger_color
        else:
            status_text = "SAFE" # or FOOD?
            status_bg = food_color
        if not self.with_satiation:
             status_text = "FOOD" if not self.is_danger else "DANGER"
            
        ax_stats.text(0.8, 0.85, status_text, color='white', ha='center', va='center', fontsize=8, fontweight='bold', 
                      transform=ax_stats.transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=status_bg, edgecolor='none'))
            
        # Draw Action
        if action is not None:
             action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", 4: "STAY"}
             act_str = action_names.get(action, "UNKNOWN")
             ax_stats.text(0.5, 0.65, f"ACTION: {act_str}", color='#495057', ha='center', fontsize=12, weight='bold', transform=ax_stats.transAxes)

        # --- 3. Draw Sensory Modules ---
        if ax_sensory and sensory_data:
            ax_sensory.set_facecolor(bg_color)
            ax_sensory.axis('off')
            
            # Title
            ax_sensory.text(0.5, 0.95, "SENSORY MODULES", color=text_color, ha='center', fontsize=10, fontweight='bold', transform=ax_sensory.transAxes)
            
            # Divide into sub-areas for each sensor
            num_sensors = len(sensory_data)
            # We'll just place them manually
            

            for i, sensor in enumerate(sensory_data):
                # Calculate center y position for this sensor
                y_center = 0.75 - (i * 0.45) 
                
                # Sensor Label
                name = sensor['name']
                # Check for new 'intensity' key, fallback to 'vector' for safety/legacy
                intensity = sensor.get('intensity', None)
                vector = sensor.get('vector', None)
                radius = sensor['radius']
                color = sensor['color'] # Hex color
                
                ax_sensory.text(0.1, y_center + 0.15, name.upper(), color=text_color, fontsize=9, fontweight='bold', transform=ax_sensory.transAxes)
                
                if intensity is not None:
                    # --- INTENSITY VISUALIZATION (Scalar) ---
                    # 1. Text Value
                    ax_sensory.text(0.8, y_center + 0.15, f"{intensity:.2f}", color=color, fontsize=9, fontweight='bold', ha='right', transform=ax_sensory.transAxes)
                    
                    # 2. Intensity Bar
                    max_val = 2.0
                    bar_width = 0.6
                    bar_height = 0.05
                    pct = min(1.0, intensity / max_val)
                    
                    rect_bg = plt.Rectangle((0.15, y_center), bar_width, bar_height, color='#F1F3F5', transform=ax_sensory.transAxes, ec='none')
                    ax_sensory.add_patch(rect_bg)
                    
                    rect_fill = plt.Rectangle((0.15, y_center), bar_width * pct, bar_height, color=color, transform=ax_sensory.transAxes, ec='none', alpha=0.8)
                    ax_sensory.add_patch(rect_fill)
                    
                    # 3. Glow Orb
                    cx, cy = 0.5, y_center - 0.15
                    orb_radius = 0.12
                    base_circle = plt.Circle((cx, cy), orb_radius, facecolor='none', edgecolor=color, alpha=0.3, transform=ax_sensory.transAxes, lw=1)
                    ax_sensory.add_patch(base_circle)
                    alpha = min(1.0, intensity * 0.8 + 0.1) if intensity > 0 else 0.05
                    fill_circle = plt.Circle((cx, cy), orb_radius * 0.8, facecolor=color, edgecolor='none', alpha=alpha, transform=ax_sensory.transAxes)
                    ax_sensory.add_patch(fill_circle)
                    
                elif vector is not None:
                    # Check if Legacy or Spectrum
                    if sensor.get('type') == 'spectrum' or 'offsets' not in sensor:
                        # --- SPECTRUM VISUALIZATION (Vector Bar Chart) ---
                        # Display a small bar chart of the vector components
                        num_channels = len(vector)
                        ax_sensory.text(0.8, y_center + 0.25, f"Vec[{num_channels}]", color='#868e96', fontsize=8, ha='right', transform=ax_sensory.transAxes)
                        
                        # Use Grid or Bars
                        # Let's do horizontal bars for channels 0..N
                        # Or vertical bars? Horizontal fits better in the slot.
                        
                        slot_width = 0.7
                        slot_height = 0.25
                        slot_x = 0.15
                        slot_y = y_center - 0.1
                        
                        # Bar width depends on N
                        bar_w = slot_width / num_channels
                        bar_gap = bar_w * 0.2
                        actual_bar_w = bar_w - bar_gap
                        
                        max_val = 2.0 # Cap for viz
                        
                        colors = ['#40C057', '#FA5252', '#339AF0', '#fab005', '#be4bdb', '#20c997'] 
                        # Green(Food), Red(Danger), Blue, Yellow, Grape, Teal
                        
                        for idx, val in enumerate(vector):
                            val = float(val) # ensure float
                            bx = slot_x + (idx * bar_w)
                            
                            # Bg
                            bg_h = slot_height
                            rect_bg = plt.Rectangle((bx, slot_y), actual_bar_w, bg_h, color='#F1F3F5', transform=ax_sensory.transAxes, ec='none')
                            ax_sensory.add_patch(rect_bg)
                            
                            # Fill
                            pct = min(1.0, val / max_val)
                            fill_h = bg_h * pct
                            c = colors[idx % len(colors)]
                            
                            rect_fill = plt.Rectangle((bx, slot_y), actual_bar_w, fill_h, color=c, transform=ax_sensory.transAxes, ec='none', alpha=0.9)
                            ax_sensory.add_patch(rect_fill)
                            
                            # Channel Label (tiny)
                            ax_sensory.text(bx + actual_bar_w/2, slot_y - 0.05, str(idx), color='#adb5bd', fontsize=6, ha='center', transform=ax_sensory.transAxes)

                    else:
                        # --- LEGACY VECTOR VISUALIZATION (Grid) ---
                        offsets = sensor['offsets']
                        
                        # Center of this sensor display in Axes Coords
                        cx, cy = 0.5, y_center
                        
                        # Scale factor for dots
                        scale = 0.08
                        
                        # Draw Center (Agent)
                        agent_dot = plt.Circle((cx, cy), scale/1.5, color='grey', transform=ax_sensory.transAxes, alpha=0.5)
                        ax_sensory.add_patch(agent_dot)
                        
                        # Create a set of active offsets for quick lookup
                        active_indices = [idx for idx, val in enumerate(vector) if val == 1]
                        
                        for idx, (dr, dc) in enumerate(offsets):
                            px = cx + dc * scale
                            py = cy - dr * scale # Invert row for plot Y
                            
                            is_active = (idx in active_indices) or (vector[idx] == 1)
                            
                            dot_color = color if is_active else '#DEE2E6' # Active vs Inactive Grey
                            edge_color = 'white'
                            alpha = 1.0 if is_active else 0.5
                            size = scale
                            
                            dot = plt.Circle((px, py), size, facecolor=dot_color, edgecolor=edge_color, transform=ax_sensory.transAxes, alpha=alpha)
                            ax_sensory.add_patch(dot)
                            
                            # Optional: Add small ring if active to make it "glow"
                            if is_active:
                                 glow = plt.Circle((px, py), size*1.3, facecolor='none', edgecolor=color, linewidth=1, transform=ax_sensory.transAxes, alpha=0.5)
                                 ax_sensory.add_patch(glow)



        # Footer
        ax_stats.text(0.5, 0.02, "GridWorld Env", color='#CED4DA', ha='center', fontsize=7, transform=fig.transFigure)

        # Convert to array
        fig.canvas.draw()
        
        # Modern Matplotlib (buffer_rgba)
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        img = data[..., :3]
        
        plt.close(fig)
        
        return img
