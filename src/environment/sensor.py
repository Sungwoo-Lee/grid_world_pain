
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

class SensorySystem:
    """
    Manager for the agent's sensors.
    Currently manages the unified ResourceSensor.
    """
    def __init__(self, sensor_radius=2, vector_size=10, decay_power=1.0):
        # Unified radius
        radius = sensor_radius
        self.resource_sensor = ResourceSensor(radius=radius, vector_size=vector_size, decay_power=decay_power)
        self.vector_size = vector_size
        
        # State Dims: The output is now a SINGLE vector of size N.
        self.state_dims = (vector_size,)
        
        # For compatibility/access
        self.food_sensor = self.resource_sensor
        self.danger_sensor = self.resource_sensor

    def sense(self, agent_pos, resources):
        """
        Args:
            agent_pos (tuple): Agent position
            resources (list): List of Resource objects from GridWorld
            
        Returns:
            np.array: The sensory observation vector.
        """
        return self.resource_sensor.sense(agent_pos, resources)
        
    def get_visualization_data(self, observation):
        """
        Prepares data for visualization.
        
        Args:
            observation (np.array): The resource sensor output vector.
        """
        return [
            {
                'name': 'Olfactory',
                'color': '#8e44ad', # Purple for mixed/chemical
                'radius': self.resource_sensor.radius,
                'vector': observation, # Pass the vector data for legacy slot (though it's intensity vector now)
                'intensity': None,     # Not a single scalar anymore
                'type': 'spectrum' 
            }
        ]
