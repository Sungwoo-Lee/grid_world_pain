import os
import jax
import jax.numpy as jnp
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

def check_params():
    config = get_default_config()
    params = load_env_params(config)
    
    print(f"Environment Height: {params.height}")
    print(f"Environment Width: {params.width}")
    print(f"Max Steps: {params.max_steps}")
    print(f"Start Nutrition: {params.start_nutrition}")
    print(f"Metabolic Cost: {params.metabolic_cost}")
    print(f"Food Nutrition Gain: {params.food_nutrition_gain}")
    print(f"Max Satiation: {params.max_satiation}")
    print(f"Rest Action Enabled: {params.rest_action_enabled}")
    print(f"Eat Action Enabled: {params.eat_action_enabled}")
    print(f"Action Dimension: {params.action_dim}")
    
    # Run a quick simulation of starvation
    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    
    print("\nSimulating Starvation (No Eating)...")
    step = 0
    while not state.terminated and step < 600:
        # Action 4 (Rest)
        state, reward, done, info = jax_step(state, 4, params)
        step += 1
    
    print(f"Terminated at step: {step}")
    print(f"Termination Reason: {info['termination_reason']} (1: truncation, 2: starvation, 4: injury)")
    print(f"Final Nutrition: {state.nutrition}")

if __name__ == "__main__":
    check_params()
