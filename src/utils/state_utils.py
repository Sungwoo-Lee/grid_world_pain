
import numpy as np
import collections

class FrameStacker:
    """
    stacks k last frames.
    Returns flattened stacked frames for MLP input.
    """
    def __init__(self, input_dim, stack_size):
        self.input_dim = input_dim
        self.stack_size = stack_size
        self.frames = collections.deque(maxlen=stack_size)
        
    def reset(self, initial_frame):
        """Reset buffer and fill with initial frame"""
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_frame)
        return self._get_stacked_state()

    def step(self, frame):
        """Append frame and return stacked state"""
        self.frames.append(frame)
        return self._get_stacked_state()
        
    def _get_stacked_state(self):
        """Stack frames into (Stack, D) array"""
        return np.stack(list(self.frames), axis=0)

def preprocess_state(state, env_height, env_width, max_satiation=None, max_health=None):
    """
    Converts raw state information (from environment + body + sensors) into a flat, normalized array for agent input.
    Handles both sensory dictionary observations and conventional state tuples.
    
    Args:
        state (dict or tuple): State observation
            - dict: Sensory system output with keys like 'olfactory', 'nociception', 'collision', 'loc', 'proprioception', etc.
            - tuple: Conventional (row, col) coordinates
        env_height, env_width (int): Grid dimensions for normalization
        max_satiation (int, optional): Max satiation for normalization
        max_health (float, optional): Max health for normalization
        
    Returns:
        np.array: Flat 1D float array with all state components concatenated and normalized
        
    Note:
        - Proprioception (if present in state dict) is automatically handled as part of sensory input
        - No special treatment needed - just another sensor like olfactory, collision, etc.
    """
    flat_list = []
    
    # Sensory Inputs (Dict)
    if isinstance(state, dict):
        # Olfactory
        if 'olfactory' in state:
             flat_list.extend(state['olfactory'])
        # Nociception
        if 'nociception' in state:
             flat_list.extend(state['nociception'])
        # Collision
        if 'collision' in state:
             flat_list.extend(state['collision'])
         # Proprioception (Motor Feedback)
        if 'proprioception' in state:
             flat_list.extend(state['proprioception'])
             
        # Coordinates (if not pure sensory, or combined)
        if 'loc' in state:
            loc = state['loc']
            # Check if likely pre-normalized (float array from SensorySystem)
            is_pre_normalized = False
            if isinstance(loc, np.ndarray) and np.issubdtype(loc.dtype, np.floating):
                 is_pre_normalized = True
            
            if is_pre_normalized:
                 flat_list.extend(loc)
            else:
                 r, c = loc
                 # Centered Normalization [-1, 1]
                 norm_r = 2.0 * (r / max(1, env_height - 1)) - 1.0
                 norm_c = 2.0 * (c / max(1, env_width - 1)) - 1.0
                 flat_list.append(norm_r)
                 flat_list.append(norm_c)
            
        # Body States
        if max_satiation and 'satiation' in state:
             flat_list.append(state['satiation'] / max_satiation)
        if max_health and 'health' in state:
             flat_list.append(state['health'] / max_health)

    else:
        # Tuple/List/Array (Legacy or Conventional)
        if isinstance(state, (tuple, list, np.ndarray)):
            # Assuming first two are loc
            flat_list.append(state[0] / env_height)
            flat_list.append(state[1] / env_width)
            # We don't handle body states here as they should be in dict if active
    
    # Previous Action is now handled by proprioceptive sensor in SensorySystem
    # If present in state dict, it's already included via sensory processing above
            
    return np.array(flat_list, dtype=np.float32)
