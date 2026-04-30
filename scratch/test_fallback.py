
import numpy as np
import sys
import os

# Mock params and state
class Params:
    def __init__(self):
        self.interoceptive_convolution_enabled = True
        self.interoceptive_kernel = np.array([0.0, 0.1, 0.2, 0.4, 0.2, 0.1])
        self.max_injury = 100.0

class State:
    def __init__(self):
        self.nociception_history_buffer = np.array([0, 10, 20, 30, 20, 10])
        self.injury_level = 10.0

params = Params()
state = State()

def fallback_logic(state, params):
    max_inj = float(params.max_injury)
    if bool(getattr(params, 'interoceptive_convolution_enabled', False)):
        buf = np.asarray(state.nociception_history_buffer)
        ker = np.asarray(params.interoceptive_kernel)
        # Normalize by max_injury
        intero_real = float(np.sum(buf * ker) / max(float(params.max_injury), 1e-6))
    else:
        intero_real = float(state.injury_level) / max_inj
    return intero_real

result = fallback_logic(state, params)
print(f"Fallback result: {result}")
expected = np.sum(state.nociception_history_buffer * params.interoceptive_kernel) / 100.0
print(f"Expected: {expected}")
assert abs(result - expected) < 1e-6
print("Fallback logic verified!")
