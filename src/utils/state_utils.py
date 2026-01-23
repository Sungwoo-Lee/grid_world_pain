
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
        """Concatenate frames into single flat vector"""
        return np.concatenate(list(self.frames), axis=0)
