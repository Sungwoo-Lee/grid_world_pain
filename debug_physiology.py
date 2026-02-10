import jax
import jax.numpy as jnp
from src.environment.core import jax_step, jax_reset
from src.environment.config_loader import load_env_params
from src.utils.config import Config

# Load live config using the correct class method
config = Config.load_yaml("configs/environment/environment.yaml")
params = load_env_params(config)

key = jax.random.PRNGKey(42)
state = jax_reset(params, key)

print("--- Testing Digestion Dynamics ---")
print(f"Initial: Satiation={state.satiation:.2f}, Nutrition={state.nutrition:.2f}")

# Simulating a few steps without eating
for i in range(1, 11):
    # Action 4 is REST (direction moves are 0-3)
    state, reward, done, info = jax_step(state, 4, params)
    print(f"Step {i:2}: Satiation={state.satiation:6.2f}, Nutrition={state.nutrition:6.2f}")

# Force an 'eat' action if agent is on food (or just check the transition)
print("\n--- Testing Recovery Dynamics ---")
# Manually inject some injury and ENSURE buffer is 0 so no smoothing damage is applied
state = state._replace(
    injury_level=jnp.array(10.0), 
    injury_buffer=jnp.zeros(params.smoothing_duration),
    rest_streak=jnp.array(0)
)
print(f"Initial Injury: {state.injury_level:.2f}")

for i in range(1, 11):
    state, reward, done, info = jax_step(state, 4, params)
    # Action 4 is REST
    print(f"Rest Step {i:2}: Injury={state.injury_level:6.2f}, Streak={state.rest_streak}")
