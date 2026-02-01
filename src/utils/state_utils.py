
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

def preprocess_state(state, env_height, env_width, max_satiation=None, max_injury=None):
    """
    Converts raw state information (from environment + body + sensors) into a flat, normalized array for agent input.
    Handles both sensory dictionary observations and conventional state tuples.
    
    Args:
        state (dict or tuple): State observation
            - dict: Sensory system output with keys like 'olfactory', 'nociception', 'collision', 'loc', 'proprioception', etc.
            - tuple: Conventional (row, col) coordinates
        env_height, env_width (int): Grid dimensions for normalization
        max_satiation (int, optional): Max satiation for normalization
        max_injury (float, optional): Max injury for normalization
        
    Returns:
        np.array: Flat 1D float array with all state components concatenated and normalized
        
    Note:
        - Proprioception (if present in state dict) is automatically handled as part of sensory input
        - No special treatment needed - just another sensor like olfactory, collision, etc.
    """
    flat_list = []
    
    # Sensory Inputs (Dict)
    if isinstance(state, dict):
        # 3-Way Biological Grouping (Preferred)
        if 'exteroception' in state:
             flat_list.extend(state['exteroception'])
        if 'interoception' in state:
             flat_list.extend(state['interoception'])
        if 'proprioception_vector' in state:
             flat_list.extend(state['proprioception_vector'])
             
        # Individual keys (Fallback/Legacy)
        elif any(k in state for k in ['olfactory', 'collision', 'loc', 'proprioception']):
            if 'olfactory' in state:
                 flat_list.extend(state['olfactory'])
            if 'nociception' in state:
                 flat_list.extend(state['nociception'])
            if 'collision' in state:
                 flat_list.extend(state['collision'])
            if 'loc' in state:
                loc = state['loc']
                if isinstance(loc, np.ndarray) and np.issubdtype(loc.dtype, np.floating):
                     flat_list.extend(loc)
                else:
                     r, c = loc
                     norm_r = 2.0 * (r / max(1, env_height - 1)) - 1.0
                     norm_c = 2.0 * (c / max(1, env_width - 1)) - 1.0
                     flat_list.append(norm_r)
                     flat_list.append(norm_c)
            if 'proprioception' in state:
                 flat_list.extend(state['proprioception'])
            
            # Legacy Body States (only if not already in interoception)
            if 'interoception' not in state:
                if max_satiation and 'satiation' in state:
                     val = state['satiation']
                     if isinstance(val, (np.ndarray, list)):
                          flat_list.extend(np.array(val) / max_satiation)
                     else:
                          flat_list.append(val / max_satiation)
                if max_injury and 'injury' in state:
                     val = state['injury']
                     if isinstance(val, (np.ndarray, list)):
                          flat_list.extend(np.array(val) / max_injury)
                     else:
                          flat_list.append(val / max_injury)

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
