
import numpy as np

class SensoryModule:
    """
    Base class for a sensory module.
    
    Generates a binary vector representing the presence of a target type 
    within a specified radius around the agent.
    """
    def __init__(self, radius=1):
        self.radius = radius
        self.offsets = self._generate_offsets(radius)
        self.vector_size = len(self.offsets)
        
    def _generate_offsets(self, radius):
        """
        Generate list of (dx, dy) relative coordinates within the radius.
        Excludes (0, 0).
        """
        offsets = []
        # Simple bounding box iteration to find points within Euclidean distance
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if np.sqrt(dx**2 + dy**2) <= radius:
                    offsets.append((dx, dy))
        
        # Sort for consistent vector ordering (e.g., top-to-bottom, left-to-right)
        offsets.sort() 
        return offsets

    def sense(self, agent_pos, targets):
        """
        Generate a binary vector observation.
        
        Args:
            agent_pos (tuple): (row, col) of the agent.
            targets (list of tuples or set): List of (row, col) positions of targets.
            
        Returns:
            tuple: Binary tuple of length equal to number of offsets.
        """
        if isinstance(targets, list):
            targets = set(targets)
            
        observation = []
        r, c = agent_pos
        
        for dr, dc in self.offsets:
            target_pos = (r + dr, c + dc)
            if target_pos in targets:
                observation.append(1)
            else:
                observation.append(0)
                
        return tuple(observation)

    @property
    def shape(self):
        """
        Returns the size of the state space for this sensor.
        Since it outputs a binary vector of length N, the state space size is 2^N.
        """
        return 2 ** self.vector_size
        
    def vector_to_index(self, vector):
        """
        Convert the binary vector tuple to a single integer index.
        """
        index = 0
        for i, val in enumerate(vector):
            if val:
                index += (1 << i)
        return index

    def index_to_vector(self, index):
        """
        Convert integer index back to binary vector tuple.
        """
        vector = []
        for i in range(self.vector_size):
            if (index >> i) & 1:
                vector.append(1)
            else:
                vector.append(0)
        return tuple(vector)


class FoodSensor(SensoryModule):
    def __init__(self, radius=1):
        super().__init__(radius)

class DangerSensor(SensoryModule):
    def __init__(self, radius=1):
        super().__init__(radius)

class SensorySystem:
    """
    Composite system holding multiple sensors.
    """
    def __init__(self, food_radius=1, danger_radius=1):
        self.food_sensor = FoodSensor(radius=food_radius)
        self.danger_sensor = DangerSensor(radius=danger_radius)
        
    def sense(self, agent_pos, food_pos, danger_pos_list=[]):
        """
        Get combined sensory state.
        
        Args:
            agent_pos (tuple): Agent (row, col)
            food_pos (tuple): Food (row, col). Note: wrapper expects list/set usually, but single pos is fine if wrapped.
            danger_pos_list (list): List of danger (row, col)
            
        Returns:
            tuple: (food_index, danger_index)
        """
        # Food is usually a single position in this env, but sensor expects collection or checkable
        food_targets = {tuple(food_pos)} if food_pos is not None else set()
        
        food_vec = self.food_sensor.sense(agent_pos, food_targets)
        danger_vec = self.danger_sensor.sense(agent_pos, danger_pos_list)
        
        food_idx = self.food_sensor.vector_to_index(food_vec)
        danger_idx = self.danger_sensor.vector_to_index(danger_vec)
        
        return (food_idx, danger_idx)
    
    def get_visualization_data(self, idx_tuple):
        """
        Retrieve data needed for visualization based on the current state indices.
        
        Args:
            idx_tuple (tuple): (food_idx, danger_idx)
            
        Returns:
            list: List of dicts [{'name': 'Food', 'radius': r, 'offsets': [...], 'vector': [...]}, ...]
        """
        food_idx, danger_idx = idx_tuple
        
        return [
            {
                'name': 'Food',
                'color': '#40C057', # Green
                'radius': self.food_sensor.radius,
                'offsets': self.food_sensor.offsets,
                'vector': self.food_sensor.index_to_vector(food_idx)
            },
            {
                'name': 'Danger',
                'color': '#FA5252', # Red
                'radius': self.danger_sensor.radius,
                'offsets': self.danger_sensor.offsets,
                'vector': self.danger_sensor.index_to_vector(danger_idx)
            }
        ]
    
    @property
    def state_dims(self):
        """
        Returns tuple of state dimensions for Q-table initialization.
        """
        return (self.food_sensor.shape, self.danger_sensor.shape)
