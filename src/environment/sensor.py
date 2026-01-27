
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
    Directional collision sensor with configurable sectors covering 360 degrees.
    Casts rays outward from the agent and detects walls (grid boundaries).
    
    Similar to EVAAA's collision sensor but adapted for discrete grid.
    
    Args:
        num_sectors: Number of directional rays (e.g., 4, 8, 10)
        sensor_range: How far each ray extends (in grid cells)
    
    Output: 1D array of size (num_sectors,)
        - 0.0: No wall within range in that direction
        - >0: Wall detected (value = 1 - (distance-1)/range, closer = higher)
    """
    def __init__(self, sensor_range=2):
        self.sensor_range = sensor_range
        # Auto-calculate sectors: 8 rays per unit of range
        self.num_sectors = sensor_range * 8
        self.output_size = self.num_sectors
        
        # Precompute ray directions for each sector
        # Sector 0 points "up" (-row direction), then clockwise
        self.directions = self._compute_directions()
        
    def _compute_directions(self):
        """
        Returns list of (dr, dc) unit vectors for each sector.
        Sector 0 = Up (0°), proceeds clockwise.
        """
        import math
        directions = []
        for i in range(self.num_sectors):
            # Angle in radians, starting from "up" and going clockwise
            angle = 2 * math.pi * i / self.num_sectors
            # In grid coords: up = -row, right = +col
            # Angle 0 = up = (-1, 0), angle π/2 = right = (0, 1)
            dr = -math.cos(angle)  # Negative because up is -row
            dc = math.sin(angle)   # Positive because right is +col
            directions.append((dr, dc))
        return directions
    
    def sense(self, agent_pos, grid_height, grid_width):
        """
        Cast rays in each sector direction and detect walls.
        
        Args:
            agent_pos (tuple): (row, col) - Agent position
            grid_height (int): Grid height
            grid_width (int): Grid width
            
        Returns:
            np.array: 1D array of shape (num_sectors,)
                      Values: 0.0 (clear) to 1.0 (wall at distance 1)
        """
        observation = np.zeros(self.num_sectors, dtype=np.float32)
        row, col = agent_pos
        
        for i, (dr, dc) in enumerate(self.directions):
            # March along the ray up to range
            for step in range(1, self.sensor_range + 1):
                # Calculate target cell (round to nearest grid cell)
                check_row = int(round(row + dr * step))
                check_col = int(round(col + dc * step))
                
                # Check if out of bounds (wall detected)
                if (check_row < 0 or check_row >= grid_height or
                    check_col < 0 or check_col >= grid_width):
                    # Wall detected - closer = higher value
                    # Distance 1 → 1.0, Distance range → near 0
                    observation[i] = 1.0 - (step - 1) / self.sensor_range
                    break
                # If no wall found within range, observation[i] stays 0.0
        
        return observation


class SensorySystem:
    """
    Manager for the agent's sensors:
    - ResourceSensor (olfactory): Gradient-based chemical detection
    - Nociceptor: Pain/danger contact detection  
    - CollisionSensor: Wall and resource collision detection
    """
    def __init__(self, sensor_radius, vector_size, decay_power, nociceptor_radius, 
                 collision_sensor_enabled=True, collision_sensor_range=2):
        # Olfactory sensor
        self.resource_sensor = ResourceSensor(
            radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power
        )
        self.vector_size = vector_size
        
        # Nociceptor
        self.nociceptor = Nociceptor(radius=nociceptor_radius)
        
        # Collision sensor
        self.collision_sensor_enabled = collision_sensor_enabled
        if collision_sensor_enabled:
            self.collision_sensor = CollisionSensor(
                sensor_range=collision_sensor_range
            )
            self.collision_output_size = self.collision_sensor.output_size
        else:
            self.collision_sensor = None
            self.collision_output_size = 0
        
        # Total output dimensions
        # Olfactory (vector_size) + Nociceptor (1) + Collision (N if enabled)
        total_dim = vector_size + 1 + self.collision_output_size
        self.state_dims = (total_dim,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def observation_spec(self):
        """
        Returns the shape of the sensory output.
        Returns:
             tuple: (total_dimension,)
        """
        return self.state_dims

    def sense(self, agent_pos, resources, grid_height=None, grid_width=None, resource_pos=None):
        """
        Returns dictionary with all sensor outputs.
        
        Args:
            agent_pos: Agent (row, col)
            resources: List of Resource objects
            grid_height, grid_width: Grid dimensions (for collision sensor)
            resource_pos: Resource position (unused by ray sensor but kept for signature)
            
        Returns:
            dict: {'olfactory': np.array, 'nociception': np.array, 'collision': np.array (optional)}
        """
        result = {
            'olfactory': self.resource_sensor.sense(agent_pos, resources),
            'nociception': self.nociceptor.sense(agent_pos, resources)
        }
        
        if self.collision_sensor_enabled and grid_height is not None:
            result['collision'] = self.collision_sensor.sense(
                agent_pos, grid_height, grid_width
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
                'type': 'radial'  # New visualization type for N-sector rays
            })
        
        return data
