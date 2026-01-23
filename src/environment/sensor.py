
import numpy as np

class ResourceSensor:
    """
    A unified gradient-based sensor that detects 'Resource' objects based on their chemical signature.
    
    It accumulates the property vectors of all resources within range, weighted by inverse distance.
    Output is a vector of size N (the resource property size).
    """
    def __init__(self, radius=2, vector_size=10, decay_power=1.0):
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
    def __init__(self, radius=0):
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

class SensorySystem:
    """
    Manager for the agent's sensors.
    Currently manages the unified ResourceSensor.
    """
    def __init__(self, sensor_radius=2, vector_size=10, decay_power=1.0, nociceptor_radius=0):
        # Unified radius
        radius = sensor_radius
        self.resource_sensor = ResourceSensor(radius=radius, vector_size=vector_size, decay_power=decay_power)
        self.nociceptor = Nociceptor(radius=nociceptor_radius)
        self.vector_size = vector_size
        
        # State Dims: Vector Size (Olfactory) + 1 (Nociceptor)
        self.state_dims = (vector_size + 1,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def sense(self, agent_pos, resources):
        """
        Args:
            agent_pos (tuple): Agent position
            resources (list): List of Resource objects from GridWorld
            
        Returns:
            dict: Dictionary with 'olfactory' and 'nociception' keys.
        """
        olfactory = self.resource_sensor.sense(agent_pos, resources)
        nociception = self.nociceptor.sense(agent_pos, resources)
        return {
            'olfactory': olfactory, 
            'nociception': nociception
        }
        
    def get_visualization_data(self, observation):
        """
        Prepares data for visualization.
        
        Args:
            observation (dict): The sensory observation dictionary.
        """
        # Dictionary Access
        olfactory_data = observation.get('olfactory')
        nociception_val = observation.get('nociception')
        
        return [
            {
                'name': 'Olfactory',
                'color': '#8e44ad', # Purple for mixed/chemical
                'radius': self.resource_sensor.radius,
                'vector': olfactory_data, 
                'intensity': None,     
                'type': 'spectrum' 
            },
            {
                'name': 'Nociceptor',
                'color': '#c0392b', # Dark Red
                'radius': 0,
                'vector': None,
                'intensity': nociception_val[0] if isinstance(nociception_val, np.ndarray) else nociception_val,
                'type': 'intensity'
            }
        ]
