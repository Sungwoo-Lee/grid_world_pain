
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
    
    Args:
        radius: Detection radius (0 = contact only)
        output_size: Output dimension (default 1 for binary)
    """
    def __init__(self, radius, output_size=1):
        self.radius = radius
        self.output_size = output_size
        
    def sense(self, agent_pos, resources):
        """
        Returns 1.0 if the agent is within radius of a Danger resource, else 0.0.
        
        Args:
            agent_pos (tuple): (row, col)
            resources (list): List of Resource objects
            
        Returns:
            np.array: Float array of shape (output_size,)
        """
        r0, c0 = agent_pos
        activated = 0.0
        for res in resources:
            if res.name == 'Danger':
                r1, c1 = res.pos
                dist = np.sqrt((r1 - r0)**2 + (c1 - c0)**2)
                if dist <= self.radius:
                    activated = 1.0
                    break
        
        # Return array of output_size, all same value
        return np.full(self.output_size, activated, dtype=np.float32)

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


        return observation


class ProprioceptiveSensor:
    """
    Proprioceptive sensor that encodes the previous action as motor feedback.
    Provides awareness of recent movement, similar to biological proprioception.
    
    Args:
        num_actions: Number of possible actions (determines output size)
    """
    def __init__(self, num_actions=5):
        self.vector_size = num_actions
        self.num_actions = num_actions
        
    def sense(self, previous_action):
        """
        Returns one-hot encoding of previous action.
        
        Args:
            previous_action (int): Action index, or None for initial state
            
        Returns:
            np.array: One-hot vector [num_actions] representing the previous action
                     Default to last action (Rest) if None
        """
        if previous_action is None:
            previous_action = self.num_actions - 2 if self.num_actions > 4 else self.num_actions - 1  # Default to Rest (second to last)
        one_hot = np.zeros(self.vector_size, dtype=np.float32)
        if 0 <= previous_action < self.vector_size:
            one_hot[previous_action] = 1.0
        return one_hot


class LocationSensor:
    """
    Sensor that returns the agent's location, normalized to centered coordinates [-1, 1].
    
    Args:
        output_size: Output dimension (default 2 for row, col)
    """
    def __init__(self, output_size=2):
        self.vector_size = output_size
        self.output_size = output_size
        
    def sense(self, agent_pos, grid_height, grid_width):
        """
        Returns centered coordinates [-1, 1].
        If output_size > 2, additional dimensions are zeros (for future use).
        """
        r, c = agent_pos
        h, w = grid_height, grid_width
        norm_r = 2.0 * (r / max(1, h - 1)) - 1.0
        norm_c = 2.0 * (c / max(1, w - 1)) - 1.0
        
        result = np.zeros(self.output_size, dtype=np.float32)
        result[0] = norm_r
        if self.output_size > 1:
            result[1] = norm_c
        return result


class SensorySystem:
    """
    Manager for the agent's sensors:
    - ResourceSensor (olfactory): Gradient-based chemical detection
    - Nociceptor: Pain/danger contact detection (optional)
    - CollisionSensor: Wall and resource collision detection (optional)
    - LocationSensor: Normalized position sensing (optional)
    - ProprioceptiveSensor: Previous action / motor feedback (optional)
    
    All sensor output sizes are configurable via constructor parameters.
    """
    def __init__(self, sensor_radius, vector_size, decay_power, nociceptor_radius, 
                 location_sensor, nociception_enabled=True, collision_sensor_enabled=True, 
                 collision_sensor_range=2, proprioception_enabled=False,
                 nociception_size=1, location_size=2, num_actions=5):
        """
        Args:
            sensor_radius: Olfactory sensor detection radius
            vector_size: Olfactory sensor output dimension
            decay_power: Olfactory sensor decay power
            nociceptor_radius: Nociceptor detection radius (0 = contact)
            location_sensor: Enable location sensor
            nociception_enabled: Enable nociceptor
            collision_sensor_enabled: Enable collision sensor
            collision_sensor_range: Collision ray length
            proprioception_enabled: Enable proprioceptive feedback
            nociception_size: Nociceptor output dimension (default 1)
            location_size: Location sensor output dimension (default 2)
            num_actions: Number of actions for proprioception (dynamic)
        """
        # Olfactory sensor
        self.resource_sensor = ResourceSensor(
            radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power
        )
        self.vector_size = vector_size
        
        # Nociceptor (Pain Detection)
        self.nociception_enabled = nociception_enabled
        if nociception_enabled:
            self.nociceptor = Nociceptor(radius=nociceptor_radius, output_size=nociception_size)
            self.nociception_size = nociception_size
        else:
            self.nociceptor = None
            self.nociception_size = 0
        
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
            
        # Location
        self.location_sensor_enabled = location_sensor
        self.location_size = location_size if self.location_sensor_enabled else 0
        
        if self.location_sensor_enabled:
            self.location_sensor = LocationSensor(output_size=location_size)
        else:
            self.location_sensor = None
        
        # Proprioception (Motor Feedback)
        self.proprioception_enabled = proprioception_enabled
        self.num_actions = num_actions
        if proprioception_enabled:
            self.proprioceptor = ProprioceptiveSensor(num_actions=num_actions)
            self.proprioception_size = num_actions
        else:
            self.proprioceptor = None
            self.proprioception_size = 0
        
        # Total output dimensions
        total_dim = vector_size + self.nociception_size + self.collision_output_size + self.location_size + self.proprioception_size
        self.state_dims = (total_dim,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def observation_spec(self):
        """
        Returns the shape of the sensory output components.
        All sizes are determined by configuration.
        
        Returns:
             dict: {
                'olfactory': {'shape': (vector_size,), 'dtype': float},
                'nociception': {'shape': (nociception_size,), 'dtype': float} (if enabled),
                'collision': {'shape': (collision_output_size,), 'dtype': float} (if enabled),
                'loc': {'shape': (location_size,), 'dtype': float} (if enabled),
                'proprioception': {'shape': (num_actions,), 'dtype': float} (if enabled)
             }
        """
        spec = {
            'olfactory': {'shape': (self.vector_size,), 'dtype': float}
        }
        
        if self.nociception_enabled:
            spec['nociception'] = {'shape': (self.nociception_size,), 'dtype': float}
        
        if self.collision_sensor_enabled:
            spec['collision'] = {
                'shape': (self.collision_output_size,), 
                'dtype': float
            }
            
        if self.location_sensor_enabled:
            spec['loc'] = {
                'shape': (self.location_size,),
                'dtype': float
            }
        
        if self.proprioception_enabled:
            spec['proprioception'] = {
                'shape': (self.proprioception_size,),
                'dtype': float
            }
            
        return spec

    def sense(self, agent_pos, resources, grid_height=None, grid_width=None, resource_pos=None, previous_action=None):
        """
        Returns dictionary with all sensor outputs.
        
        Args:
            agent_pos: Agent (row, col)
            resources: List of Resource objects
            grid_height, grid_width: Grid dimensions (for collision sensor)
            resource_pos: Resource position (unused by ray sensor but kept for signature)
            previous_action: Previous action index (0-4) for proprioceptive sensing
            
        Returns:
            dict: {'olfactory': np.array, 'nociception': np.array, 'collision': np.array (optional), 
                   'loc': np.array (optional), 'proprioception': np.array (optional)}
        """
        if grid_height is not None and grid_width is not None:
            self.grid_dims = (grid_height, grid_width)
            
        result = {
            'olfactory': self.resource_sensor.sense(agent_pos, resources)
        }
        
        if self.nociception_enabled:
            result['nociception'] = self.nociceptor.sense(agent_pos, resources)
        
        if self.collision_sensor_enabled and grid_height is not None:
            result['collision'] = self.collision_sensor.sense(
                agent_pos, grid_height, grid_width
            )
            
        if self.location_sensor_enabled:
            if grid_height is None or grid_width is None:
                # Fallback or error?
                # If we are training, we usually have dims. 
                # If not provided, we can't center.
                # Assuming provided for now as checked in train.py/main.py
                raise ValueError("Grid dimensions required for LocationSensor")
                
            result['loc'] = self.location_sensor.sense(agent_pos, grid_height, grid_width)
        
        if self.proprioception_enabled:
            result['proprioception'] = self.proprioceptor.sense(previous_action)
        
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
            
        if self.location_sensor_enabled and 'loc' in observation:
            # Value is already centered float array
            val = observation['loc']
            val_text = f"({val[0]:.2f}, {val[1]:.2f})"
                 
            data.append({
                'name': 'LOC',
                'color': '#ADB5BD',
                'value_text': val_text,
                'radius': 0,
                'vector': None,
                'intensity': None,
                'type': 'text'
            })
        
        return data
