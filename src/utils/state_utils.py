
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
    Flattens state dictionary to float array.
    Handles Dictionary-based inputs (Olfactory, Nociception) and Body states.
    Normalizes coordinates and body states.
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
             
        # Coordinates (if not pure sensory, or combined)
        if 'loc' in state:
            r, c = state['loc']
            flat_list.append(r / env_height)
            flat_list.append(c / env_width)
            
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
            
    return np.array(flat_list, dtype=np.float32)
