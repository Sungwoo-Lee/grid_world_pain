
import numpy as np

class ResourceSensor:
    """
    A unified gradient-based sensor that detects 'Resource' objects based on their chemical signature.
    
    It accumulates the property vectors of all resources within range, weighted by inverse distance.
    Output is a vector of size N (the resource property size).
    """
    def __init__(self, radius, vector_size, decay_power):
        self.radius = radius
        self.vector_size = vector_size
        self.decay_power = decay_power

    def sense(self, agent_pos, resources):
        """
        Sensory Logic:
        Observation = Sum( Resource.Property * (1 / Distance) ) for all resources in radius.
        
        Args:
            agent_pos (tuple): (row, col)
            resources (list): List of Resource objects (pos, property, name)
            
        Returns:
            np.array: Float vector of size self.vector_size
        """
        observation = np.zeros(self.vector_size, dtype=np.float32)
        r0, c0 = agent_pos
        
        for res in resources:
            r1, c1 = res.pos
            # Calculate Euclidean distance
            dist = np.sqrt((r1 - r0)**2 + (c1 - c0)**2)
            
            if dist > self.radius:
                continue
                
            # Handle singularity (agent ON resource)
            if dist < 0.001:
                decay = 2.0 
            else:
                decay = 1.0 / (dist ** self.decay_power)
                
            # Accumulate
            observation += res.property * decay
            
        return observation

class Nociceptor:
    """
    A contact sensor (range 0) that detects 'Danger' resources at the agent's exact location.
    Behaves as a biological nociceptor: detecting immediate nociception.
    """
    def __init__(self, radius):
        self.radius = radius 
        
    def sense(self, agent_pos, resources):
        """
        Returns 1.0 if the agent is within radius of a Danger resource, else 0.0.
        
        Args:
            agent_pos (tuple): (row, col)
            resources (list): List of Resource objects
            
        Returns:
            np.array: Scalar float [1.0] or [0.0]
        """
        r0, c0 = agent_pos
        for res in resources:
            if res.name == 'Danger':
                r1, c1 = res.pos
                dist = np.sqrt((r1 - r0)**2 + (c1 - c0)**2)
                if dist <= self.radius:
                    return np.array([1.0], dtype=np.float32)
                    
        return np.array([0.0], dtype=np.float32)

class CollisionSensor:
    """
    A contact sensor that detects collisions with walls (grid boundaries) and resources.
    Provides spatial awareness of the agent's immediate surroundings.
    
    Output: [wall_up, wall_right, wall_down, wall_left, on_resource]
    Each value is 0.0 or 1.0.
    """
    def __init__(self, radius=0):
        self.radius = radius
        self.output_size = 5  # 4 directions + resource contact
        
    def sense(self, agent_pos, grid_height, grid_width, resource_pos=None):
        """
        Detects walls at grid boundaries and resource contact.
        
        Args:
            agent_pos (tuple): (row, col)
            grid_height (int): Grid height
            grid_width (int): Grid width
            resource_pos (tuple, optional): Current resource position
            
        Returns:
            np.array: [wall_up, wall_right, wall_down, wall_left, on_resource]
        """
        row, col = agent_pos
        observation = np.zeros(self.output_size, dtype=np.float32)
        
        # Wall detection (at boundary = wall)
        observation[0] = 1.0 if row == 0 else 0.0                    # Up wall
        observation[1] = 1.0 if col == grid_width - 1 else 0.0       # Right wall
        observation[2] = 1.0 if row == grid_height - 1 else 0.0      # Down wall
        observation[3] = 1.0 if col == 0 else 0.0                    # Left wall
        
        # Resource contact
        if resource_pos is not None and agent_pos == resource_pos:
            observation[4] = 1.0
            
        return observation


class SensorySystem:
    """
    Manager for the agent's sensors:
    - ResourceSensor (olfactory): Gradient-based chemical detection
    - Nociceptor: Pain/danger contact detection  
    - CollisionSensor: Wall and resource collision detection
    """
    def __init__(self, sensor_radius, vector_size, decay_power, nociceptor_radius, 
                 collision_sensor_enabled=True):
        # Olfactory sensor
        self.resource_sensor = ResourceSensor(
            radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power
        )
        self.vector_size = vector_size
        
        # Nociceptor
        self.nociceptor = Nociceptor(radius=nociceptor_radius)
        
        # Collision sensor (new)
        self.collision_sensor_enabled = collision_sensor_enabled
        self.collision_sensor = CollisionSensor(radius=0) if collision_sensor_enabled else None
        self.collision_output_size = 5 if collision_sensor_enabled else 0
        
        # Total output dimensions
        # Olfactory (vector_size) + Nociceptor (1) + Collision (5 if enabled)
        total_dim = vector_size + 1 + self.collision_output_size
        self.state_dims = (total_dim,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def sense(self, agent_pos, resources, grid_height=None, grid_width=None, resource_pos=None):
        """
        Returns dictionary with all sensor outputs.
        
        Args:
            agent_pos: Agent (row, col)
            resources: List of Resource objects
            grid_height, grid_width: Grid dimensions (for collision sensor)
            resource_pos: Resource position (for collision sensor)
            
        Returns:
            dict: {'olfactory': np.array, 'nociception': np.array, 'collision': np.array (optional)}
        """
        result = {
            'olfactory': self.resource_sensor.sense(agent_pos, resources),
            'nociception': self.nociceptor.sense(agent_pos, resources)
        }
        
        if self.collision_sensor_enabled and grid_height is not None:
            result['collision'] = self.collision_sensor.sense(
                agent_pos, grid_height, grid_width, resource_pos
            )
        
        return result
        
    def get_visualization_data(self, observation):
        """
        Prepares data for visualization.
        
        Args:
            observation (dict): The sensory observation dictionary.
        """
        olfactory_data = observation.get('olfactory')
        nociception_val = observation.get('nociception')
        
        data = [
            {
                'name': 'Olfactory',
                'color': '#8e44ad',  # Purple for mixed/chemical
                'radius': self.resource_sensor.radius,
                'vector': olfactory_data, 
                'intensity': None,     
                'type': 'spectrum' 
            },
            {
                'name': 'Nociceptor',
                'color': '#c0392b',  # Dark Red
                'radius': 0,
                'vector': None,
                'intensity': nociception_val[0] if isinstance(nociception_val, np.ndarray) else nociception_val,
                'type': 'intensity'
            }
        ]
        
        if self.collision_sensor_enabled and 'collision' in observation:
            data.append({
                'name': 'Collision',
                'color': '#e67e22',  # Orange
                'radius': 0,
                'vector': observation.get('collision'),
                'intensity': None,
                'type': 'directional',
                'labels': ['↑', '→', '↓', '←', '◆']  # Wall directions + resource
            })
        
        return data
