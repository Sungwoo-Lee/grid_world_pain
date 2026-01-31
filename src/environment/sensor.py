
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

class ExteroceptiveNociceptor:
    """
    A contact sensor (range 0) that detects 'Danger' resources at the agent's exact location.
    Behaves as a biological exteroceptive nociceptor: detecting immediate external threat.
    
    Args:
        radius: Detection radius (0 = contact only)
        output_size: Output dimension
    """
    def __init__(self, radius, output_size=1):
        self.radius = radius
        self.output_size = output_size
        
    def sense(self, agent_pos, resources):
        """
        Returns 1.0 if the agent is within radius of a Danger resource, else 0.0.
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
        
        return np.full(self.output_size, activated, dtype=np.float32)

class InteroceptiveNociceptor:
    """
    A sensor that monitors the internal injury level of the body.
    Output is the current injury level, usually scaled [0, 1].
    
    Args:
        output_size: Output dimension
    """
    def __init__(self, output_size=1):
        self.output_size = output_size
        
    def sense(self, injury_level, max_injury):
        """
        Returns the normalized injury level [0, 1].
        """
        val = injury_level / max_injury if max_injury > 0 else 0
        return np.full(self.output_size, val, dtype=np.float32)

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
                 location_sensor, nociception_enabled=True, olfactory_enabled=True,
                 collision_sensor_enabled=True, collision_sensor_range=2, 
                 proprioception_enabled=False, nociception_size=1, 
                 location_size=2, num_actions=5):
        """
        Args:
            sensor_radius: Olfactory sensor detection radius
            vector_size: Olfactory sensor output dimension
            decay_power: Olfactory sensor decay power
            nociceptor_radius: Nociceptor detection radius (0 = contact)
            location_sensor: Enable location sensor
            nociception_enabled: Enable nociceptor
            olfactory_enabled: Enable olfactory sensor
            collision_sensor_enabled: Enable collision sensor
            collision_sensor_range: Collision ray length
            proprioception_enabled: Enable proprioceptive feedback
            nociception_size: Nociceptor output dimension (default 1)
            location_size: Location sensor output dimension (default 2)
            num_actions: Number of actions for proprioception (dynamic)
        """
        # Olfactory sensor
        self.olfactory_enabled = olfactory_enabled
        if olfactory_enabled:
            self.resource_sensor = ResourceSensor(
                radius=sensor_radius, 
                vector_size=vector_size, 
                decay_power=decay_power
            )
            self.vector_size = vector_size
        else:
            self.resource_sensor = None
            self.vector_size = 0
        
        # Nociception (Pain Detection)
        self.nociception_enabled = nociception_enabled
        if nociception_enabled:
            # Split into Extero (Phasic) and Intero (Smooth Injury)
            self.extero_nociceptor = ExteroceptiveNociceptor(radius=nociceptor_radius, output_size=nociception_size)
            self.intero_nociceptor = InteroceptiveNociceptor(output_size=nociception_size)
            self.nociception_size = nociception_size * 2 # Combined size
        else:
            self.extero_nociceptor = None
            self.intero_nociceptor = None
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
        self.exteroception_size = (self.vector_size + 
                                   (nociception_size if nociception_enabled else 0) + 
                                   self.collision_output_size + 
                                   self.location_size)
        
        self.interoception_size = (nociception_size if nociception_enabled else 0) + 1 # +1 for Satiation
        self.proprioception_size = self.proprioception_size # Already computed
        
        total_dim = self.exteroception_size + self.interoception_size + self.proprioception_size
        self.state_dims = (total_dim,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def observation_spec(self):
        """
        Returns the shape of the grouped sensory output.
        
        Returns:
             dict: {
                'exteroception': {'shape': (exteroception_size,), 'dtype': float},
                'interoception': {'shape': (interoception_size,), 'dtype': float},
                'proprioception': {'shape': (proprioception_size,), 'dtype': float}
             }
        """
        spec = {
            'exteroception': {'shape': (self.exteroception_size,), 'dtype': float},
            'interoception': {'shape': (self.interoception_size,), 'dtype': float},
            'proprioception': {'shape': (self.proprioception_size,), 'dtype': float}
        }
        return spec
            
    def get_dimension_breakdown(self):
        """Returns a string describing the calculation of each dimension group."""
        extero_parts = []
        if self.olfactory_enabled: extero_parts.append(f"Olf={self.vector_size}")
        if self.nociception_enabled: extero_parts.append(f"ExteroNoc={self.extero_nociceptor.output_size}")
        if self.collision_sensor_enabled: extero_parts.append(f"Col={self.collision_output_size}")
        if self.location_sensor_enabled: extero_parts.append(f"Loc={self.location_size}")
        
        extero_str = f"Extero({self.exteroception_size}) = " + " + ".join(extero_parts)
        
        intero_parts = []
        if self.nociception_enabled: intero_parts.append(f"Injury={self.intero_nociceptor.output_size}")
        intero_parts.append(f"Sat=1")
        
        intero_str = f"Intero({self.interoception_size}) = " + " + ".join(intero_parts)
        
        prop_str = f"Proprio({self.proprioception_size}) = Actions({self.num_actions})"
        
        return f"{extero_str}\n  {intero_str}\n  {prop_str}"

    def sense(self, agent_pos, resources, grid_height=None, grid_width=None, previous_action=None, extra_data=None):
        """
        Returns dictionary with grouped sensor outputs.
        
        Args:
            agent_pos: Agent (row, col)
            resources: List of Resource objects
            grid_height, grid_width: Grid dimensions
            previous_action: Previous action index
            extra_data: Dictionary for additional telemetry 
                       Ex: {'injury_level': x, 'max_injury': y, 'satiation': z, 'max_satiation': s}
        """
        if grid_height is not None and grid_width is not None:
            self.grid_dims = (grid_height, grid_width)
            
        result = {}
        extero_parts = []
        intero_parts = []
        prop_parts = []
            
        # --- Exteroception ---
        if self.olfactory_enabled:
            olf = self.resource_sensor.sense(agent_pos, resources)
            result['olfactory'] = olf
            extero_parts.append(olf)
        
        if self.nociception_enabled:
            extero_noc = self.extero_nociceptor.sense(agent_pos, resources)
            result['extero_nociception'] = extero_noc
            extero_parts.append(extero_noc)
            
            # Injury
            injury_level = extra_data.get('injury_level', 0) if isinstance(extra_data, dict) else 0
            max_injury = extra_data.get('max_injury', 1) if isinstance(extra_data, dict) else 1
            intero_noc = self.intero_nociceptor.sense(injury_level, max_injury)
            result['injury'] = intero_noc
            intero_parts.append(intero_noc)
            
            # For backward compatibility/legacy visualization
            result['nociception'] = np.concatenate([extero_noc, intero_noc])
        
        if self.collision_sensor_enabled and grid_height is not None:
            col = self.collision_sensor.sense(agent_pos, grid_height, grid_width)
            result['collision'] = col
            extero_parts.append(col)
            
        if self.location_sensor_enabled:
            if grid_height is None or grid_width is None:
                raise ValueError("Grid dimensions required for LocationSensor")
            loc = self.location_sensor.sense(agent_pos, grid_height, grid_width)
            result['loc'] = loc
            extero_parts.append(loc)
        
        # --- Proprioception ---
        if self.proprioception_enabled:
            prop = self.proprioceptor.sense(previous_action)
            result['proprioception'] = prop
            prop_parts.append(prop)
            
        # --- Interoception ---
        # Satiation
        satiation = extra_data.get('satiation', 0) if isinstance(extra_data, dict) else 0
        max_satiation = extra_data.get('max_satiation', 1) if isinstance(extra_data, dict) else 1
        sat_val = np.array([satiation / max_satiation], dtype=np.float32)
        result['satiation_sensor'] = sat_val
        intero_parts.append(sat_val)
        
        # Construct grouped results
        result['exteroception'] = np.concatenate(extero_parts) if extero_parts else np.zeros(0, dtype=np.float32)
        result['interoception'] = np.concatenate(intero_parts) if intero_parts else np.zeros(0, dtype=np.float32)
        result['proprioception_out'] = np.concatenate(prop_parts) if prop_parts else np.zeros(0, dtype=np.float32)
        # Note: 'proprioception' key is used for the individual sensor output already
        result['proprioception_vector'] = result['proprioception_out'] 
        
        return result
        
    def get_visualization_data(self, observation):
        """
        Prepares data for visualization.
        
        Args:
            observation (dict): The sensory observation dictionary.
        """
        olfactory_data = observation.get('olfactory')
        nociception_val = observation.get('nociception')
        
        data = []
        
        if self.olfactory_enabled:
            data.append({
                'name': 'Olfactory',
                'color': '#8e44ad',  # Purple for mixed/chemical
                'radius': self.resource_sensor.radius,
                'vector': olfactory_data, 
                'intensity': None,     
                'type': 'spectrum' 
            })

        if self.nociception_enabled:
            # Combined vector is [extero, intero]
            extero_val = nociception_val[0] if len(nociception_val) > 0 else 0
            
            data.append({
                'name': 'Extero Nociception',
                'color': '#c0392b',  # Dark Red
                'radius': 0,
                'vector': None,
                'intensity': extero_val,
                'type': 'intensity'
            })
        
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
