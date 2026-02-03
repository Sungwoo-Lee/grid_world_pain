
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os


from src.environment.body import InteroceptiveBody
from collections import namedtuple

# Keep generic Resource namedtuple for compatibility if needed, but we will use ResourceEntity mostly
# Sensor expects: object with .pos, .property, .name
ResourceInfo = namedtuple('ResourceInfo', ['pos', 'property', 'name'])

class ResourceEntity:
    """
    Represents a single resource instance with a lifecycle.
    """
    def __init__(self, uid: int, config: dict):
        """
        Initializes a ResourceEntity with strict configuration.

        Args:
            uid (int): Unique identifier for the resource.
            config (dict): Configuration dictionary containing:
                - name (str): Display name.
                - type (str): 'food' or 'danger'.
                - properties (list): Vector representation.
                - spawn_area (list): [[min_r, min_c], [max_r, max_c]].
                - max_consumption (int): Uses before depletion (-1 for infinite).
                - regeneration_delay (int): Steps to wait before respawning.
                - damage (float): Damage value (if danger).

        Raises:
            ValueError: If any required configuration key is missing.
        """
        self.id = uid
        if 'name' not in config: raise ValueError("Strict Config: Resource 'name' is required.")
        self.name = config['name']
        
        if 'type' not in config: raise ValueError("Strict Config: Resource 'type' is required.")
        self.type = config['type']
        
        if 'properties' not in config: raise ValueError("Strict Config: Resource 'properties' is required.")
        self.property = np.array(config['properties'], dtype=np.float32)
        
        # Spatial config: Defines where the resource can spawn [[min_r, min_c], [max_r, max_c]]
        if 'spawn_area' not in config: raise ValueError("Strict Config: Resource 'spawn_area' is required.")
        self.spawn_area = config['spawn_area'] 
        
        # Lifecycle config
        if 'max_consumption' not in config: raise ValueError("Strict Config: Resource 'max_consumption' is required.")
        self.max_consumption = config['max_consumption'] # -1 for infinite consumption
        
        if 'regeneration_delay' not in config: raise ValueError("Strict Config: Resource 'regeneration_delay' is required.")
        self.regeneration_delay = config['regeneration_delay']
        
        if 'damage' not in config: raise ValueError("Strict Config: Resource 'damage' is required.")
        self.damage = config['damage']
        
        # State
        self.pos = None
        self.active = True
        self.consumption_count = 0
        self.regeneration_timer = 0
        
    def reset(self, grid_height: int, grid_width: int, existing_positions_to_avoid: list = None):
        """
        Resets the resource to a new position within its spawn area.
        
        Args:
            grid_height (int): Height of the grid.
            grid_width (int): Width of the grid.
            existing_positions_to_avoid (list): List of (r, c) tuples to avoid.
        """
        self.active = True
        self.consumption_count = 0
        self.regeneration_timer = 0
        self.respawn(grid_height, grid_width, existing_positions_to_avoid)
        
    def respawn(self, grid_height: int, grid_width: int, existing_positions_to_avoid: list = None):
        """
        Finds a valid position for the resource within its spawn_area.
        
        Args:
            grid_height (int): Height of the grid.
            grid_width (int): Width of the grid.
            existing_positions_to_avoid (list): List of (r, c) tuples to avoid.
        """
        if existing_positions_to_avoid is None:
            existing_positions_to_avoid = []
            
        # Determine bounds
        if self.spawn_area:
            min_r, min_c = self.spawn_area[0]
            max_r, max_c = self.spawn_area[1]
            # Clamp to grid
            min_r = max(0, min_r)
            min_c = max(0, min_c)
            max_r = min(grid_height - 1, max_r)
            max_c = min(grid_width - 1, max_c)
        else:
            min_r, min_c = 0, 0
            max_r, max_c = grid_height - 1, grid_width - 1
            
        # Try to find a free spot
        for _ in range(100):
            r = np.random.randint(min_r, max_r + 1)
            c = np.random.randint(min_c, max_c + 1)
            if (r, c) not in existing_positions_to_avoid:
                self.pos = (r, c)
                return
        # Fallback if crowded
        self.pos = (r, c)

    def consume(self) -> tuple[float, bool]:
        """
        Interacts with the resource.
        
        Returns:
            tuple: (signal, consumed_flag)
                - signal (float): 1.0 if interaction occurred, 0.0 otherwise.
                - consumed_flag (bool): True if successfully consumed.
        """
        if not self.active:
            return 0, False
            
        self.consumption_count += 1
        
        # Check lifecycle
        if self.max_consumption > 0 and self.consumption_count >= self.max_consumption:
            self.active = False
            self.regeneration_timer = self.regeneration_delay
            
        return 1, True # Signal that interaction happened

    def step(self) -> bool:
        """
        Updates internal timer and checks for regeneration.
        
        Returns:
            bool: True if the resource has just respawned (position update needed).
        """
        if not self.active:
            if self.regeneration_timer > 0:
                self.regeneration_timer -= 1
            
            if self.regeneration_timer <= 0:
                # Time to respawn
                self.active = True
                self.consumption_count = 0
                return True # respawn needed
        return False

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
    
    Actions (Discrete, configurable):
    - 0: Up (Decreases row index)
    - 1: Right (Increases col index)
    - 2: Down (Increases row index)
    - 3: Left (Decreases col index)
    - 4: Rest (if rest_action_enabled) - Recovers injury
    - 5: Eat (if eat_action_enabled) - Consumes food
    
    Rewards:
    - This external environment returns a reward of 0.
    - In this "Interoceptive AI" paradigm, strict rewards come from the *Body* (internal state),
      not the external world. The environment only provides signals (like 'ate_food').
    """
    
    def __init__(self, height, width, start, resources, 
                 with_satiation, max_steps,
                 prob_switch_to_danger=None, min_danger_duration=None, damage_amount=None,
                 prob_switch_to_food=None, min_food_duration=None, relocate_resource=None, relocation_steps=None, # Legacy args
                 vector_size=None, food_property=None, danger_property=None, # Legacy args
                 eat_action_enabled=True, rest_action_enabled=True,
                 predator_enabled=False, predator_move_interval=2, predator_damage=3.0,
                 predator_start_pos=(3, 0), predator_random_start_pos=True, predator_property=None,
                 injury_smoothing_duration=3):
        """
        Initializes the GridWorld foraging environment.

        Args:
            height (int): Grid height.
            width (int): Grid width.
            start (tuple): Agent start position (row, col).
            resources (list): List of resource configuration dictionaries (New System).
            with_satiation (bool): Enable internal satiation state.
            max_steps (int): Maximum steps per episode.
            prob_switch_to_danger (float): [Legacy] Probability of danger switch (Ignored if resources config used).
            min_danger_duration (int): [Legacy] Minimum danger duration.
            damage_amount (float): [Legacy] Damage amount.
            prob_switch_to_food (float): [Legacy] Probability of food switch.
            min_food_duration (int): [Legacy] Minimum food duration.
            relocate_resource (bool): [Legacy] Relocate resource randomly.
            relocation_steps (int): [Legacy] Steps between relocations.
            vector_size (int): Size of sensory vector.
            food_property (list): [Legacy] Vector for food (Ignored if resources config used).
            danger_property (list): [Legacy] Vector for danger (Ignored if resources config used).
            eat_action_enabled (bool): If True, Action 5 (Eat) is required to consume.
            rest_action_enabled (bool): If True, Action 4 (Rest) is required to recover injury.
            predator_enabled (bool): Enable moving predator.
            predator_move_interval (int): Steps between predator moves.
            predator_damage (float): Damage on contact with predator.
            predator_start_pos (tuple): Predator start position.
            predator_random_start_pos (bool): Randomize predator start.
            predator_property (list): Vector for predator.
            injury_smoothing_duration (int): Smoothing window for injury rendering.
        """
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.agent_pos = self.start
        self.with_satiation = with_satiation
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action Configuration
        self.eat_action_enabled = eat_action_enabled
        self.rest_action_enabled = rest_action_enabled
        
        # Calculate action indices based on configuration
        # Base: 0=Up, 1=Right, 2=Down, 3=Left
        self._num_actions = 4
        if rest_action_enabled:
            self.REST_ACTION = self._num_actions
            self._num_actions += 1
        else:
            self.REST_ACTION = None
        if eat_action_enabled:
            self.EAT_ACTION = self._num_actions
            self._num_actions += 1
        else:
            self.EAT_ACTION = None
        
        # Resource Property Vectors
        self.vector_size = vector_size
        self.injury_smoothing_duration = injury_smoothing_duration
        
        # --- Resource Initialization ---
        self.resources = []
        uid_counter = 0
        
        # Handle 'resources' list from config
        # If resources arg is present (it should be passed from main/train), iterate
        # If not (legacy call), we might crash or need fallback. Assuming updated config passes it.
        if resources:
            for res_config in resources:
                count = res_config.get('count', 1)
                for _ in range(count):
                    new_res = ResourceEntity(uid_counter, res_config)
                    # Initialize position immediately so they are valid before first reset()
                    # We pass current occupied list (agent + previously added resources)
                    occupied = [self.agent_pos] + [r.pos for r in self.resources if r.pos is not None]
                    new_res.respawn(self.height, self.width, occupied)
                    self.resources.append(new_res)
                    uid_counter += 1
        else:
             # Fallback for legacy init if 'resources' is empty/None but old args exist
             # This prevents breaking if config isn't fully updated or passed correctly
             pass

        # Predator configuration
        self.predator_enabled = predator_enabled
        self.predator_move_interval = predator_move_interval
        self.predator_damage = predator_damage
        self.predator_default_start_pos = tuple(predator_start_pos)
        self.predator_random_start_pos = predator_random_start_pos
        self.predator_property = np.array(predator_property if predator_property is not None else [0.0]*vector_size, dtype=np.float32)
        
        self.predator_pos = self.predator_default_start_pos
        self.predator_timer = self.predator_move_interval
        
        # Load Visual Assets
        self.assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
        self.icons = {}
        icon_files = {
            'agent': 'agent.jpg',
            'food': 'food.jpg',
            'danger': 'danger.jpg',
            'predator': 'predator.jpg',
            'agent_food': 'agent_food.jpg',
            'agent_danger': 'agent_danger.jpg',
            'agent_predator': 'agent_predator.jpg'
        }
        
        for key, filename in icon_files.items():
            path = os.path.join(self.assets_path, filename)
            if os.path.exists(path):
                self.icons[key] = plt.imread(path)
            else:
                self.icons[key] = None
        
    def get_active_resources(self):
        """
        Returns a list of active ResourceInfo objects (compatible with sensor).
        """
        active_list = []
        for res in self.resources:
            if res.active:
                # property is a numpy array
                active_list.append(ResourceInfo(res.pos, res.property, res.name))
                
        if self.predator_enabled:
            active_list.append(ResourceInfo(self.predator_pos, self.predator_property, 'Danger'))
            
        return active_list

    def reset(self):
        """
        Resets the agent and resources.
        """
        self.current_step = 0
        
        # Reset Agent
        while True:
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            if (row, col) != self.start:
                self.agent_pos = (row, col)
                break
        
        # Reset Resources
        for res in self.resources:
            # Pass all occupied positions to avoid overlaps?
            # Ideally avoid agent, predator, and other resources.
            occupied = [self.agent_pos] + [r.pos for r in self.resources if r.pos is not None and r is not res]
            res.reset(self.height, self.width, occupied)
             
        # Reset Predator
        if self.predator_enabled:
            if self.predator_random_start_pos:
                while True:
                    pr = np.random.randint(0, self.height)
                    pc = np.random.randint(0, self.width)
                    # Avoid agent and resources?
                    occupied = [self.agent_pos]
                    if (pr, pc) not in occupied:
                        self.predator_pos = (pr, pc)
                        break
            else:
                self.predator_pos = self.predator_default_start_pos
            self.predator_timer = self.predator_move_interval

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
        
        # 1. Update Resource Lifecycles (Regeneration)
        for res in self.resources:
            respawn_needed = res.step()
            if respawn_needed:
                occupied = [self.agent_pos] + [r.pos for r in self.resources if r.active and r is not res]
                res.respawn(self.height, self.width, occupied)

        # 2. Predator Update
        if self.predator_enabled:
            self.predator_timer -= 1
            if self.predator_timer <= 0:
                pr, pc = self.predator_pos
                ar, ac = self.agent_pos
                dr = ar - pr
                dc = ac - pc
                
                if dr != 0 and dc != 0:
                    if np.random.random() < 0.5: pr += np.sign(dr)
                    else: pc += np.sign(dc)
                elif dr != 0: pr += np.sign(dr)
                elif dc != 0: pc += np.sign(dc)
                
                self.predator_pos = (int(pr), int(pc))
                self.predator_timer = self.predator_move_interval

        # 3. Agent Movement
        row, col = self.agent_pos
        
        # Movement logic (actions 0-3)
        if action == 0:   # Up
            row = max(0, row - 1)
        elif action == 1: # Right
            col = min(self.width - 1, col + 1)
        elif action == 2: # Down
            row = min(self.height - 1, row + 1)
        elif action == 3: # Left
            col = max(0, col - 1)
        # Rest (action 4) and Eat (action 5) don't move
        
        self.agent_pos = (row, col)
        
        # 4. Interactions
        ate_food = False
        damage = 0
        rested = False
        
        # Check interactions with all active resources at current position
        active_here = [r for r in self.resources if r.active and r.pos == self.agent_pos]
        
        for res in active_here:
            # 1. Danger Interaction (Automatic)
            if res.type == 'danger':
                damage += res.damage
                
            # 2. Food Interaction (Action-based or Automatic)
            elif res.type == 'food':
                if self.eat_action_enabled:
                     # Explicit Eat Action Required
                     if action == self.EAT_ACTION:
                         _, consumed = res.consume()
                         if consumed:
                             ate_food = True
                else:
                     # Auto-eat (Collision-based)
                     _, consumed = res.consume()
                     if consumed:
                         ate_food = True
        
        # Damage from predator
        if self.predator_enabled and self.agent_pos == self.predator_pos:
            damage += self.predator_damage
        
        # Resting logic
        if self.rest_action_enabled:
            # Must explicitly perform Rest action
            rested = (action == self.REST_ACTION)
        else:
            # Auto-recovery (legacy behavior) - always resting
            rested = True
            
        # Reward/Done: Behavior depends on whether satiation is enabled.
        reward = 0
        done = False
        if not self.with_satiation:
            if ate_food:
                reward = 10 # Conventional goal reward
                done = True # REACHED GOAL
        
        if self.current_step >= self.max_steps:
            done = True
        
        info = {'ate_food': ate_food, 'damage': damage, 'rested': rested}
        
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
            'injury': {'shape': (1,), 'dtype': float}
        }
        return spec

    def action_spec(self):
        """
        Returns the action space specification.
        Action count is dynamic based on eat/rest configuration.
        Returns:
            dict: {'type': 'discrete', 'n': N}
        """
        return {'type': 'discrete', 'n': self._num_actions}

    def render(self):
        """Prints grid info."""
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                char = ". "
                if (r, c) == self.agent_pos:
                    char = "A "
                elif self.predator_enabled and (r, c) == self.predator_pos:
                    char = "P "
                else:
                    # Check resources
                    for res in self.resources:
                        if res.active and res.pos == (r, c):
                            if res.type == 'danger': char = "X "
                            elif res.type == 'food': char = "F "
                            break
                line += char
            print(line)
        print(f"Step: {self.current_step}")

    def render_rgb_array(self, satiation=None, max_satiation=None, injury=None, max_injury=None, episode=None, step=None, sensory_data=None, action=None, dpi=None, icon_scale=1.0):
        if dpi is None: dpi = 100
        
        bg_color = '#FFFFFF'
        grid_color = '#E9ECEF'
        agent_color = '#339AF0'
        food_color = '#40C057'
        danger_color = '#FA5252'
        text_color = '#343A40'
        
        fig_width = 10 if sensory_data is not None else 8
        fig = plt.figure(figsize=(fig_width, 6), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        
        if sensory_data is not None:
             gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[0.4, 0.6])
             ax_grid = fig.add_subplot(gs[:, 0])
             ax_stats = fig.add_subplot(gs[0, 1])
             ax_sensory = fig.add_subplot(gs[1, 1])
        else:
             gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
             ax_grid = fig.add_subplot(gs[0])
             ax_stats = fig.add_subplot(gs[1])
             ax_sensory = None
        
        # --- 1. Draw Grid World ---
        ax_grid.set_facecolor(bg_color)
        ax_grid.set_xlim(-0.5, self.width - 0.5)
        ax_grid.set_ylim(-0.5, self.height - 0.5)
        ax_grid.invert_yaxis()
        ax_grid.set_aspect('equal')
        ax_grid.axis('off')
        
        # Scaling factor based on baseline 4x4 grid
        scale_factor = (4.0 / max(self.width, self.height)) * icon_scale
        
        for x in range(self.width + 1):
            ax_grid.vlines(x - 0.5, -0.5, self.height - 0.5, colors='#DEE2E6', linestyles='-', linewidth=0.8, alpha=0.5)
        for y in range(self.height + 1):
            ax_grid.hlines(y - 0.5, -0.5, self.width - 0.5, colors='#DEE2E6', linestyles='-', linewidth=0.8, alpha=0.5)
            
        def draw_icon(ax, pos, icon_key, zoom=0.038):
            r, c = pos
            img = self.icons.get(icon_key)
            if img is not None:
                imagebox = OffsetImage(img, zoom=zoom * scale_factor)
                ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0)
                ax.add_artist(ab)
            else:
                markers = {'agent': ('o', agent_color), 'food': ('D', food_color), 
                           'danger': ('X', danger_color), 'predator': ('v', '#212529')}
                m, color = markers.get(icon_key, ('s', 'grey'))
                ax.plot(c, r, marker=m, markersize=14 * scale_factor, color=color, markeredgecolor='white', markeredgewidth=1.5)

        ar, ac = self.agent_pos
        pr, pc = self.predator_pos if self.predator_enabled else (None, None)
        
        # 1. Draw Resources
        any_danger_here = False
        any_food_here = False
        
        for res in self.resources:
            if not res.active: continue
            
            fr, fc = res.pos
            is_here = (fr == ar and fc == ac)
            
            if res.type == 'danger':
                if is_here: any_danger_here = True
                else: draw_icon(ax_grid, (fr, fc), 'danger', zoom=0.035)
            elif res.type == 'food':
                if is_here: any_food_here = True
                else: draw_icon(ax_grid, (fr, fc), 'food', zoom=0.035)
                
        # 2. Draw Predator
        predator_here = False
        if self.predator_enabled:
            if pr == ar and pc == ac:
                predator_here = True
            else:
                draw_icon(ax_grid, (pr, pc), 'predator', zoom=0.045)
                
        # 3. Draw Agent (handling overlaps)
        if predator_here:
             draw_icon(ax_grid, (ar, ac), 'agent_predator', zoom=0.055)
        elif any_danger_here:
             draw_icon(ax_grid, (ar, ac), 'agent_danger', zoom=0.048)
        elif any_food_here:
             draw_icon(ax_grid, (ar, ac), 'agent_food', zoom=0.048)
        else:
             draw_icon(ax_grid, (ar, ac), 'agent', zoom=0.035)

        # --- 2. Draw Stats ---
        ax_stats.set_facecolor(bg_color)
        ax_stats.axis('off')
        y_cursor = 1.0 
        
        y_cursor -= 0.15 
        ax_stats.text(0.5, y_cursor, "INTEROCEPTIVE AI", color=text_color, ha='center', fontsize=12, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.plot([0.2, 0.8], [y_cursor - 0.05, y_cursor - 0.05], color='#ADB5BD', transform=ax_stats.transAxes, linewidth=1)
        
        y_cursor -= 0.15
        ep_str = f"EPISODE: {episode}" if episode is not None else "EP: --"
        step_str = f"STEP:    {step}" if step is not None else "STEP: --"
        ax_stats.text(0.1, y_cursor, ep_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        ax_stats.text(0.1, y_cursor - 0.08, step_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
        
        y_cursor -= 0.12 
        
        if (self.with_satiation and satiation is not None) or (injury is not None):
            y_cursor -= 0.05
            ax_stats.text(0.1, y_cursor, "INTEROCEPTION", color='#868E96', fontsize=8, fontweight='bold', transform=ax_stats.transAxes)
            y_cursor -= 0.05
        
        def draw_bar(ax, y_pos, label, value, max_val, color):
            pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
            ax.text(0.1, y_pos + 0.1, f"{label}: {value:.1f}/{max_val}", color=text_color, fontsize=8, fontweight='bold', transform=ax.transAxes)
            rect_bg = plt.Rectangle((0.1, y_pos), 0.8, 0.08, color='#F1F3F5', transform=ax.transAxes, ec='none')
            ax.add_patch(rect_bg)
            rect_fill = plt.Rectangle((0.1, y_pos), 0.8 * pct, 0.08, color=color, transform=ax.transAxes, ec='none')
            ax.add_patch(rect_fill)
            
        if self.with_satiation and satiation is not None and max_satiation is not None:
             y_cursor -= 0.15
             draw_bar(ax_stats, y_cursor, "SATIATION", satiation, max_satiation, food_color)
        
        if injury is not None and max_injury is not None:
             y_cursor -= 0.15
             draw_bar(ax_stats, y_cursor, "INJURY", injury, max_injury, danger_color)
             
        y_cursor -= 0.20
        
        if any_danger_here:
            status_text = "DANGER"
            status_bg = danger_color
        elif any_food_here:
            status_text = "FOOD"
            status_bg = food_color
        else:
            status_text = "CLEAR"
            status_bg = '#ADB5BD'
            
        ax_stats.text(0.8, y_cursor + 0.05, status_text, color='white', ha='center', va='center', fontsize=8, fontweight='bold', 
                      transform=ax_stats.transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=status_bg, edgecolor='none'))

        if action is not None:
             action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
             if self.REST_ACTION is not None: action_names[self.REST_ACTION] = "REST"
             if self.EAT_ACTION is not None: action_names[self.EAT_ACTION] = "EAT"
             act_str = action_names.get(action, "UNKNOWN")
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
                    
                    # Sensor Label
                    name = sensor['name']
                    intensity = sensor.get('intensity', None)
                    vector = sensor.get('vector', None)
                    color = sensor['color']
                    
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
                             max_ray_len = min(0.15, slot_h * 0.35)
                             
                             num_secs = len(vector)
                             import math
                             for si in range(num_secs):
                                  val = vector[si]
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
