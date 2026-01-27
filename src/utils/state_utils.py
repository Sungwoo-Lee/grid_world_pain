
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

def preprocess_state(state, env_height, env_width, max_satiation=None, max_health=None, previous_action=None):
    """
    Flattens state dictionary to float array.
    Handles Dictionary-based inputs (Olfactory, Nociception) and Body states.
    Normalizes coordinates and body states.
    
    Args:
        state: State dictionary or tuple
        env_height: Environment height for normalization
        env_width: Environment width for normalization
        max_satiation: Maximum satiation for normalization (optional)
        max_health: Maximum health for normalization (optional)
        previous_action: Previous action index (0-4) for one-hot encoding (optional)
                        If None, no previous action is appended
    
    Returns:
        np.array: Flattened and normalized state vector
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
        # Collision (new)
        if 'collision' in state:
             flat_list.extend(state['collision'])
             
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
    
    # Previous Action (One-Hot Encoding)
    if previous_action is not None:
        # Create one-hot encoding for 5 actions (0: Up, 1: Right, 2: Down, 3: Left, 4: Stay)
        action_one_hot = np.zeros(5, dtype=np.float32)
        action_one_hot[previous_action] = 1.0
        flat_list.extend(action_one_hot)
            
    return np.array(flat_list, dtype=np.float32)
